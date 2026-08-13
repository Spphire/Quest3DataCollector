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
    if reuse_ratio > max_quest_reuse_ratio:
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
) -> None:
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(
        "\n".join(
            [
                f"# Selected from {start_record_id} and later.",
                f"# Required effective-rate ratio: >= {min_rate_ratio:.2f} for robot 90 Hz and aligned/end/third 30 Hz.",
                f"# Required aligned Quest source reuse ratio: <= {max_quest_reuse_ratio:.2f}.",
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
    args = parser.parse_args()

    selected, excluded = select_reports(
        audit_dir=args.audit_dir,
        start_record_id=args.start_record_id,
        min_rate_ratio=args.min_rate_ratio,
        max_quest_reuse_ratio=args.max_quest_reuse_ratio,
    )
    write_outputs(
        selected=selected,
        excluded=excluded,
        selected_output=args.selected_output,
        excluded_output=args.excluded_output,
        start_record_id=args.start_record_id,
        min_rate_ratio=args.min_rate_ratio,
        max_quest_reuse_ratio=args.max_quest_reuse_ratio,
    )
    print(f"selected={len(selected)} excluded={len(excluded)}")


if __name__ == "__main__":
    main()
