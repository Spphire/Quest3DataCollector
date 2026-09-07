#!/usr/bin/env python3
"""Select stable Collector recordings from audit-performance JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


RATE_CHECK_IDS = (
    "robot_state_rate",
    "aligned_sample_rate",
    "camera_end_rate",
    "camera_third_rate",
)


def _reuse_profile(
    recording_dir: Path,
    *,
    max_consecutive_reuse: int | None = 5,
    max_gaze_age_seconds: float = 0.060,
) -> dict | None:
    """Summarize reuse after trimming stale/terminal Quest rows.

    A terminal reuse run longer than the tolerated internal run is treated as a
    Quest stop/idle tail and is trim-eligible. Only the remaining internal window
    participates in selection.
    """
    samples_path = recording_dir / "robot_realsense" / "samples.jsonl"
    if not samples_path.exists():
        return None
    rows = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        return {"original_count": 0, "trim_start_frames": 0, "trim_end_frames": 0,
                "original_reuse_count": 0, "trimmed_reuse_count": 0,
                "max_internal_consecutive_reuse": 0}

    def stale(row: dict) -> bool:
        aligned = row.get("pc_perf_counter_seconds")
        received = row.get("quest_pc_receive_perf_counter_seconds")
        try:
            return (
                float(aligned) - float(received) > max_gaze_age_seconds
                if aligned is not None and received is not None
                else False
            )
        except (TypeError, ValueError):
            return False

    reused = [bool(row.get("aligned_source_reused")) for row in rows]
    start = 0
    end = len(rows)

    def endpoint_run_from_start() -> int:
        index = start
        while index < end and reused[index]:
            index += 1
        return index - start

    def endpoint_run_from_end() -> int:
        index = end
        while index > start and reused[index - 1]:
            index -= 1
        return end - index

    # Evaluate reuse before stale-age trimming so a partially stale terminal run is
    # removed in full (for example, all 60 rows of a tail with 59 stale rows).
    while True:
        changed = False
        if max_consecutive_reuse is not None and endpoint_run_from_start() > max_consecutive_reuse:
            start += endpoint_run_from_start()
            changed = True
        if max_consecutive_reuse is not None and endpoint_run_from_end() > max_consecutive_reuse:
            end -= endpoint_run_from_end()
            changed = True
        while start < end and stale(rows[start]):
            start += 1
            changed = True
        while end > start and stale(rows[end - 1]):
            end -= 1
            changed = True
        if not changed:
            break

    longest = current = 0
    for flag in reused[start:end]:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return {
        "original_count": len(rows),
        "trim_start_frames": start,
        "trim_end_frames": len(rows) - end,
        "original_reuse_count": sum(reused),
        "trimmed_reuse_count": sum(reused[start:end]),
        "max_internal_consecutive_reuse": longest,
    }


def _check_by_id(report: dict) -> Dict[str, dict]:
    return {
        str(check.get("id")): check
        for check in report.get("checks", [])
        if isinstance(check, dict) and check.get("id")
    }


def _target_ratio(check: dict) -> float:
    performance = check.get("performance", {})
    if isinstance(performance.get("video"), dict):
        performance = performance["video"]
    return float(performance["targetRatio"])


def _quest_reuse_ratio(report: dict) -> float:
    performance = report.get("performance", {})
    reused = float(performance["alignedReusedSourceSamples"])
    count = float(performance["alignedSamples"]["count"])
    if count <= 0:
        raise ValueError("alignedSamples.count must be positive")
    return reused / count


def qualify_report(
    report: dict,
    min_rate_ratio: float = 0.95,
    max_quest_reuse_ratio: float = 0.10,
    max_consecutive_reuse: int | None = 5,
    max_fallback_reuse_ratio: float = 0.20,
    recording_dir: Path | None = None,
    max_gaze_age_seconds: float = 0.060,
) -> Tuple[List[str], Dict[str, float]]:
    checks = _check_by_id(report)
    reasons = []
    metrics = {}
    for check_id in RATE_CHECK_IDS:
        try:
            ratio = _target_ratio(checks[check_id])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing targetRatio for {check_id}") from exc
        metrics[check_id] = ratio
        if ratio < min_rate_ratio:
            reasons.append(check_id)

    try:
        reuse_ratio = _quest_reuse_ratio(report)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing aligned Quest source reuse metrics") from exc
    metrics["aligned_quest_source_reuse"] = reuse_ratio
    metrics["aligned_quest_source_reuse_legacy_threshold"] = float(
        reuse_ratio > max_quest_reuse_ratio
    )
    profile = (
        _reuse_profile(
            recording_dir,
            max_consecutive_reuse=max_consecutive_reuse,
            max_gaze_age_seconds=max_gaze_age_seconds,
        )
        if recording_dir
        else None
    )
    if profile is not None:
        metrics.update({f"aligned_quest_{key}": float(value) for key, value in profile.items()})
        longest_reuse = profile["max_internal_consecutive_reuse"]
        if max_consecutive_reuse is not None and longest_reuse > max_consecutive_reuse:
            reasons.append("aligned_quest_consecutive_reuse")
        retained_count = profile["original_count"] - profile["trim_start_frames"] - profile["trim_end_frames"]
        post_trim_ratio = (
            profile["trimmed_reuse_count"] / retained_count if retained_count else 1.0
        )
        metrics["aligned_quest_post_trim_reuse_ratio"] = float(post_trim_ratio)
        if post_trim_ratio > max_fallback_reuse_ratio:
            reasons.append("aligned_quest_reuse_ratio")
    elif reuse_ratio > max_quest_reuse_ratio:
        # Preserve the old behavior when the raw aligned timeline is unavailable.
        reasons.append("aligned_quest_source_reuse")
    return reasons, metrics


def iter_reports(audit_dir: Path, start_record_id: str) -> Iterable[Tuple[Path, dict]]:
    for path in sorted(audit_dir.glob("record_*.json")):
        record_id = path.stem
        if record_id < start_record_id:
            continue
        yield path, json.loads(path.read_text(encoding="utf-8"))


def select_reports(
    audit_dir: Path,
    start_record_id: str,
    min_rate_ratio: float,
    max_quest_reuse_ratio: float,
    max_consecutive_reuse: int | None = 5,
    max_fallback_reuse_ratio: float = 0.20,
    recordings_root: Path | None = None,
    max_gaze_age_seconds: float = 0.060,
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    selected = []
    excluded = []
    for path, report in iter_reports(audit_dir, start_record_id):
        record_id = str(report.get("recordId") or path.stem)
        if record_id != path.stem:
            raise ValueError(f"recordId mismatch in {path}: {record_id}")
        reasons, _metrics = qualify_report(
            report,
            min_rate_ratio=min_rate_ratio,
            max_quest_reuse_ratio=max_quest_reuse_ratio,
            max_consecutive_reuse=max_consecutive_reuse,
            max_fallback_reuse_ratio=max_fallback_reuse_ratio,
            recording_dir=(recordings_root / record_id) if recordings_root else None,
            max_gaze_age_seconds=max_gaze_age_seconds,
        )
        if reasons:
            excluded.append((record_id, reasons))
        else:
            selected.append(record_id)
    if not selected and not excluded:
        raise FileNotFoundError(f"No audit JSON reports at or after {start_record_id} in {audit_dir}")
    return selected, excluded


def write_outputs(
    selected: List[str],
    excluded: List[Tuple[str, List[str]]],
    selected_output: Path,
    excluded_output: Path,
    start_record_id: str,
    min_rate_ratio: float,
    max_quest_reuse_ratio: float,
    max_consecutive_reuse: int | None,
    max_fallback_reuse_ratio: float,
) -> None:
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(
        "\n".join(
            [
                f"# Selected from {start_record_id} and later.",
                f"# Required effective-rate ratio: >= {min_rate_ratio:.2f} for robot 90 Hz and aligned/end/third 30 Hz.",
                f"# Preferred legacy reuse ratio: <= {max_quest_reuse_ratio:.2f};",
                f"# with raw timelines, max consecutive reuse: <= {max_consecutive_reuse if max_consecutive_reuse is not None else 'disabled'} frames,",
                f"# and fallback cumulative reuse ratio: <= {max_fallback_reuse_ratio:.2f}.",
                *selected,
                "",
            ]
        ),
        encoding="utf-8",
    )
    excluded_output.parent.mkdir(parents=True, exist_ok=True)
    excluded_output.write_text(
        "record_id\treasons\n"
        + "".join(f"{record_id}\t{','.join(reasons)}\n" for record_id, reasons in excluded),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--start-record-id", default="record_20260813_162120")
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--excluded-output", type=Path, required=True)
    parser.add_argument("--min-rate-ratio", type=float, default=0.95)
    parser.add_argument("--max-quest-reuse-ratio", type=float, default=0.10)
    parser.add_argument(
        "--max-consecutive-reuse",
        type=int,
        default=5,
        help="Maximum consecutive aligned rows reusing one Quest source when raw samples are available. Default: 5.",
    )
    parser.add_argument(
        "--max-fallback-reuse-ratio",
        type=float,
        default=0.20,
        help="Cumulative reuse fallback/upper bound when raw samples are available. Default: 0.20.",
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        help="Root containing record_id/robot_realsense/samples.jsonl for consecutive-reuse checks.",
    )
    parser.add_argument("--max-gaze-age-seconds", type=float, default=0.060)
    args = parser.parse_args()

    selected, excluded = select_reports(
        audit_dir=args.audit_dir,
        start_record_id=args.start_record_id,
        min_rate_ratio=args.min_rate_ratio,
        max_quest_reuse_ratio=args.max_quest_reuse_ratio,
        max_consecutive_reuse=args.max_consecutive_reuse,
        max_fallback_reuse_ratio=args.max_fallback_reuse_ratio,
        recordings_root=args.recordings_root,
        max_gaze_age_seconds=args.max_gaze_age_seconds,
    )
    write_outputs(
        selected=selected,
        excluded=excluded,
        selected_output=args.selected_output,
        excluded_output=args.excluded_output,
        start_record_id=args.start_record_id,
        min_rate_ratio=args.min_rate_ratio,
        max_quest_reuse_ratio=args.max_quest_reuse_ratio,
        max_consecutive_reuse=args.max_consecutive_reuse,
        max_fallback_reuse_ratio=args.max_fallback_reuse_ratio,
    )
    print(f"selected={len(selected)} excluded={len(excluded)}")


if __name__ == "__main__":
    main()
