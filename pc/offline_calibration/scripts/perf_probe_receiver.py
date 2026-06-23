from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
RECEIVER = SCRIPT_DIR / "quest_pc_receiver.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "pc_recordings"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Send a synthetic A-button formal recording to a running Quest PC receiver, "
            "then run audit-performance on the saved record."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Receiver HTTP/UDP host. Default: 127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=9100, help="Receiver UDP port. Default: 9100")
    parser.add_argument("--view-port", type=int, default=8765, help="Receiver viewer HTTP port. Default: 8765")
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="Synthetic recording duration. Default: 10")
    parser.add_argument("--sample-hz", type=float, default=90.0, help="Synthetic Quest sample rate. Default: 90")
    parser.add_argument("--record-id", help="Record id. Default: record_perfprobe_<timestamp>")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="PC recording root.")
    parser.add_argument(
        "--connect-robot",
        action="store_true",
        help="Call /robot/connect before recording. The probe never presses teleop/gripper controls.",
    )
    parser.add_argument("--wait-seconds", type=float, default=45.0, help="Max wait for saved summary. Default: 45")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for audit-performance.")
    args = parser.parse_args()

    record_id = args.record_id or f"record_perfprobe_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.connect_robot:
        connect = post_json(args.host, args.view_port, "/robot/connect", {"waitSeconds": 0.2}, timeout=12)
        if not connect.get("ok"):
            raise RuntimeError(f"robot connect failed: {json.dumps(connect, ensure_ascii=False)}")

    status = get_json(args.host, args.view_port, "/robot/status", timeout=8)
    send_recording(args.host, args.udp_port, record_id, args.duration_seconds, args.sample_hz)
    summary = wait_for_summary(args.output_root, record_id, args.host, args.view_port, args.wait_seconds)
    record_dir = resolve_record_directory(args.output_root, record_id, summary)
    audit = run_audit(args.python, record_dir)
    replay_id = replay_record_id(record_dir, summary, record_id)
    replay = replay_health(args.host, args.view_port, replay_id)

    payload = {
        "ok": audit.get("ok") is True and replay.get("ok") is True,
        "recordId": record_id,
        "replayRecordId": replay_id,
        "recordDirectory": str(record_dir.resolve()),
        "receiver": {
            "host": args.host,
            "udpPort": args.udp_port,
            "viewPort": args.view_port,
            "robotConnected": status.get("robot", {}).get("connected") if isinstance(status.get("robot"), dict) else None,
            "robotStateHz": status.get("config", {}).get("robotStateHz") if isinstance(status.get("config"), dict) else None,
            "depthEvery": status.get("config", {}).get("recordDepthEveryNFrames") if isinstance(status.get("config"), dict) else None,
        },
        "summary": {
            "samples": summary.get("samples"),
            "messages": summary.get("messages"),
            "writerDroppedMessages": summary.get("writerDroppedMessages"),
            "closeMetrics": summary.get("closeMetrics"),
            "robotRealSenseDirectory": summary.get("robotRealSenseDirectory"),
        },
        "audit": {
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


def send_recording(host: str, port: int, record_id: str, duration_seconds: float, sample_hz: float) -> None:
    duration_seconds = max(0.1, float(duration_seconds))
    sample_hz = max(1.0, float(sample_hz))
    sample_count = int(round(duration_seconds * sample_hz))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        start_wall = time.time()
        send(sock, host, port, {"type": "recording_start", "recordId": record_id, "sequence": 0})
        next_perf = time.perf_counter()
        for index in range(sample_count):
            t = index / sample_hz
            x = 0.2 + 0.02 * math.sin(t * 2.0)
            send(
                sock,
                host,
                port,
                {
                    "type": "sample",
                    "recordId": record_id,
                    "sequence": index + 1,
                    "sampleIndex": index,
                    "isRecording": True,
                    "telemetryMode": "recording",
                    "recordingTimestampSeconds": t,
                    "unityTimestampSeconds": start_wall + t,
                    "hasGaze": True,
                    "gazeSource": "perf_probe",
                    "gazePoint3DWorld": [x, 1.05, 0.55],
                    "gazePoint3DSource": "perf_probe",
                    "gazeRayOrigin": [0.0, 1.2, 0.0],
                    "gazeRayDirection": [0.0, -0.15, 1.0],
                    "headPose": pose_payload([0.0, 1.5, 0.0]),
                    "leftEyePose": pose_payload([-0.03, 1.5, 0.02]),
                    "rightEyePose": pose_payload([0.03, 1.5, 0.02]),
                    "leftController": controller_pose_payload([-0.2, 1.1, 0.3]),
                    "rightController": controller_pose_payload([0.2, 1.1, 0.3]),
                },
            )
            next_perf += 1.0 / sample_hz
            sleep = next_perf - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
        send(
            sock,
            host,
            port,
            {
                "type": "recording_stop",
                "recordId": record_id,
                "sequence": sample_count + 1,
                "telemetryMode": "live_preview",
                "recordingTimestampSeconds": duration_seconds,
            },
        )
    finally:
        sock.close()


def pose_payload(position: list[float]) -> dict[str, object]:
    return {
        "hasPose": True,
        "position": position,
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "source": "perf_probe",
    }


def controller_pose_payload(position: list[float]) -> dict[str, object]:
    payload = pose_payload(position)
    payload.update(
        {
            "positionTracked": True,
            "rotationTracked": True,
        }
    )
    return payload


def send(sock: socket.socket, host: str, port: int, payload: dict[str, object]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendto(data, (host, port))


def wait_for_summary(output_root: Path, record_id: str, host: str, view_port: int, timeout_seconds: float) -> dict[str, object]:
    deadline = time.time() + max(1.0, float(timeout_seconds))
    last_state: dict[str, object] | None = None
    while time.time() < deadline:
        try:
            preflight = get_json(host, view_port, "/preflight/status", timeout=3)
            last_state = preflight.get("captureState") if isinstance(preflight.get("captureState"), dict) else None
            if isinstance(last_state, dict):
                summary = last_state.get("summary")
                if isinstance(summary, dict) and summary.get("recordId") == record_id and summary.get("closedReason"):
                    return summary
        except Exception:
            pass
        for summary_path in candidate_summary_paths(output_root, record_id):
            if summary_path.exists():
                try:
                    summary = read_json(summary_path)
                except json.JSONDecodeError:
                    time.sleep(0.2)
                    continue
                if summary.get("closedReason") and summary.get("replayVisualizationJson"):
                    return summary
                if last_state and last_state.get("phase") == "live" and summary.get("closedReason"):
                    return summary
        time.sleep(0.2)
    raise TimeoutError(f"summary not ready for {record_id}; last capture state={last_state}")


def resolve_record_directory(output_root: Path, record_id: str, summary: dict[str, object]) -> Path:
    session_directory = summary.get("sessionDirectory")
    if isinstance(session_directory, str) and session_directory:
        return Path(session_directory)
    for summary_path in candidate_summary_paths(output_root, record_id):
        if summary_path.exists():
            return summary_path.parent
    return output_root / sanitize_name(record_id)


def candidate_summary_paths(output_root: Path, record_id: str) -> list[Path]:
    names = []
    safe = sanitize_name(record_id)
    for value in (record_id if record_id == safe else "", safe):
        if value and value not in names:
            names.append(value)
    if output_root.exists():
        suffix_pattern = re.compile(rf"^{re.escape(safe)}_\d{{3}}$")
        for item in sorted(output_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if item.is_dir() and item.name not in names:
                if item.name == safe or suffix_pattern.match(item.name):
                    names.append(item.name)
    return [output_root / name / "pc_session_summary.json" for name in names]


def sanitize_name(value: str) -> str:
    cleaned = []
    for char in str(value).strip():
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._")
    return result or "unnamed"


def run_audit(python: str, record_dir: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            python,
            str(RECEIVER),
            "audit-performance",
            "--pc-session",
            str(record_dir),
            "--output-json",
            str(record_dir / "performance_audit.json"),
        ],
        cwd=str(SCRIPT_DIR.parents[2]),
        text=True,
        capture_output=True,
    )
    audit_path = record_dir / "performance_audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
    else:
        audit = {"ok": False, "summary": "audit JSON not written"}
    audit["stdout"] = result.stdout.strip()
    audit["stderr"] = result.stderr.strip()
    audit["returnCode"] = result.returncode
    return audit


def replay_record_id(record_dir: Path, summary: dict[str, object], requested_id: str) -> str:
    for value in (summary.get("recordId"), record_dir.name, requested_id):
        if isinstance(value, str) and value:
            return value
    return record_dir.name


def replay_health(host: str, view_port: int, record_id: str) -> dict[str, object]:
    payload = get_json(host, view_port, f"/recordings/replay?recordId={quote(record_id, safe='')}&compact=1", timeout=30)
    robot = payload.get("robotRealSense") if isinstance(payload.get("robotRealSense"), dict) else {}
    return {
        "ok": payload.get("ok"),
        "samples": len(payload.get("samples") or []),
        "auditOk": payload.get("recordingAudit", {}).get("ok") if isinstance(payload.get("recordingAudit"), dict) else None,
        "robotSamples": len(robot.get("samples") or []),
        "robotStates": len(robot.get("robotStates") or []),
        "videoFrameCount": robot.get("videoFrameCount"),
        "depthFrameCount": robot.get("depthFrameCount"),
    }


def get_json(host: str, port: int, path: str, *, timeout: float) -> dict[str, object]:
    with urlopen(f"http://{host}:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(host: str, port: int, path: str, payload: dict[str, object], *, timeout: float) -> dict[str, object]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        f"http://{host}:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
