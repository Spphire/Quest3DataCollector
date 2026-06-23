from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
RECEIVER = SCRIPT_DIR / "quest_pc_receiver.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "pc_recordings"
DEFAULT_EXCLUDE_TOKENS = ("perfprobe", "codex", "smoke")
SUMMARY_NAME = "pc_session_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the newest real Quest A-button PC recording and compact replay health."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="PC recording root.")
    parser.add_argument("--record-id", help="Specific record id or directory name. Default: newest non-probe record.")
    parser.add_argument("--exclude-token", action="append", default=[], help="Case-insensitive token to skip.")
    parser.add_argument("--include-probes", action="store_true", help="Allow perfprobe/codex/smoke records.")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver HTTP host for replay health. Default: 127.0.0.1")
    parser.add_argument("--view-port", type=int, default=8765, help="Receiver viewer HTTP port. Default: 8765")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for audit-performance.")
    parser.add_argument("--output-json", type=Path, help="Optional path for this combined audit JSON.")
    args = parser.parse_args()

    exclude_tokens = [] if args.include_probes else list(DEFAULT_EXCLUDE_TOKENS)
    exclude_tokens.extend(args.exclude_token or [])
    record_dir = resolve_record_dir(args.output_root, args.record_id, exclude_tokens)
    summary = read_json(record_dir / SUMMARY_NAME)
    replay_ids = replay_record_ids(record_dir, summary)
    audit = run_performance_audit(args.python, record_dir)
    replay = replay_health(args.host, args.view_port, replay_ids)
    robot_summary = summary.get("robotRealSense") if isinstance(summary.get("robotRealSense"), dict) else {}
    start_time = record_sort_timestamp(record_dir)
    payload = {
        "ok": bool(audit.get("ok") and replay.get("ok")),
        "recordId": summary.get("recordId") or record_dir.name,
        "recordDirectory": str(record_dir.resolve()),
        "modifiedTime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record_dir.stat().st_mtime)),
        "recordStartTime": None if start_time is None else datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "summary": {
            "samples": summary.get("samples"),
            "closedReason": summary.get("closedReason"),
            "writerDroppedMessages": summary.get("writerDroppedMessages"),
            "writerError": summary.get("writerError"),
            "motionCommands": robot_summary.get("motionCommands"),
            "gripperCommands": robot_summary.get("gripperCommands"),
            "robotStateSamples": robot_summary.get("robotStateSamples"),
            "videoFrames": robot_summary.get("videoFrames"),
            "depthFrames": robot_summary.get("depthFrames"),
        },
        "performanceAudit": {
            "ok": audit.get("ok"),
            "summary": audit.get("summary"),
            "checks": [
                {
                    "id": check.get("id"),
                    "ok": check.get("ok"),
                    "detail": check.get("detail"),
                }
                for check in audit.get("checks", [])
            ],
        },
        "replay": replay,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


def resolve_record_dir(output_root: Path, record_id: str | None, exclude_tokens: list[str]) -> Path:
    root = output_root.resolve()
    if record_id:
        for path in explicit_record_candidates(root, record_id):
            if path.is_dir() and (path / SUMMARY_NAME).exists():
                return path
        raise FileNotFoundError(f"Record not found under {root}: {record_id}")

    if not root.exists():
        raise FileNotFoundError(f"PC recording root not found: {root}")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("record_")
        and (path / SUMMARY_NAME).exists()
        and not is_excluded_record(path, exclude_tokens)
    ]
    if not candidates:
        raise FileNotFoundError(f"No non-probe record_* folders found under {root}")
    return max(candidates, key=record_sort_key)


def explicit_record_candidates(root: Path, record_id: str) -> list[Path]:
    names: list[str] = []
    safe = sanitize_name(record_id)
    for value in (record_id if record_id == safe else "", safe):
        if value and value not in names:
            names.append(value)

    # unique_directory() appends _001, _002, ... when the same recordId is reused.
    # Only consider that exact suffix shape; do not treat user input as a glob.
    if safe and root.exists():
        suffix_pattern = re.compile(rf"^{re.escape(safe)}_\d{{3}}$")
        for item in root.iterdir():
            if item.is_dir() and suffix_pattern.match(item.name) and item.name not in names:
                names.append(item.name)
    return [root / name for name in names]


def is_excluded_record(path: Path, exclude_tokens: list[str]) -> bool:
    text = path.name.lower()
    summary = read_json(path / SUMMARY_NAME)
    record_id = str(summary.get("recordId") or "").lower() if isinstance(summary, dict) else ""
    return any(token.lower() in text or token.lower() in record_id for token in exclude_tokens)


def record_sort_key(path: Path) -> tuple[int, float, float]:
    start = record_sort_timestamp(path)
    if start is not None:
        return (1, start, path.stat().st_mtime)
    return (0, path.stat().st_mtime, path.stat().st_mtime)


def record_sort_timestamp(path: Path) -> float | None:
    summary = read_json(path / SUMMARY_NAME)
    if is_number(summary.get("startPcReceiveUnixSeconds")):
        return float(summary["startPcReceiveUnixSeconds"])
    if is_number(summary.get("recordingStartUnixSeconds")):
        return float(summary["recordingStartUnixSeconds"])
    for key in ("startUtc", "startTimeUtc"):
        parsed = parse_iso_timestamp_seconds(summary.get(key))
        if parsed is not None:
            return parsed
    return None


def run_performance_audit(python: str, record_dir: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="quest_record_audit_") as temp_dir:
        output_json = Path(temp_dir) / "performance_audit.json"
        result = subprocess.run(
            [
                python,
                str(RECEIVER),
                "audit-performance",
                "--pc-session",
                str(record_dir),
                "--output-json",
                str(output_json),
            ],
            cwd=str(SCRIPT_DIR.parents[2]),
            text=True,
            capture_output=True,
        )
        if output_json.exists():
            payload = read_json(output_json)
        else:
            payload = {"ok": False, "summary": "audit JSON not written"}
    payload["returnCode"] = result.returncode
    payload["stdout"] = result.stdout.strip()
    payload["stderr"] = result.stderr.strip()
    return payload


def replay_record_ids(record_dir: Path, summary: dict[str, object]) -> list[str]:
    candidates = []
    for value in (summary.get("recordId"), record_dir.name):
        if isinstance(value, str) and value and value not in candidates:
            candidates.append(value)
    return candidates or [record_dir.name]


def replay_health(host: str, port: int, record_ids: list[str]) -> dict[str, object]:
    errors: list[dict[str, str]] = []
    payload: dict[str, object] | None = None
    resolved_id: str | None = None
    for record_id in record_ids:
        url = f"http://{host}:{port}/recordings/replay?recordId={quote(record_id, safe='')}&compact=1"
        try:
            with urlopen(url, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            resolved_id = record_id
            break
        except Exception as exc:
            errors.append({"recordId": record_id, "error": str(exc)})
    if payload is None:
        return {"ok": False, "errors": errors}
    robot = payload.get("robotRealSense") if isinstance(payload.get("robotRealSense"), dict) else {}
    return {
        "ok": payload.get("ok"),
        "recordIdUsed": resolved_id,
        "auditOk": payload.get("recordingAudit", {}).get("ok") if isinstance(payload.get("recordingAudit"), dict) else None,
        "samples": len(payload.get("samples") or []),
        "robotSamples": len(robot.get("samples") or []),
        "robotStates": len(robot.get("robotStates") or []),
        "videoFrameCount": robot.get("videoFrameCount"),
        "depthFrameCount": robot.get("depthFrameCount"),
    }


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def sanitize_name(value: str) -> str:
    cleaned = []
    for char in str(value).strip():
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._")
    return result or "unnamed"


def is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        try:
            return math.isfinite(float(value))
        except ValueError:
            return False
    return False


def parse_iso_timestamp_seconds(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


if __name__ == "__main__":
    raise SystemExit(main())
