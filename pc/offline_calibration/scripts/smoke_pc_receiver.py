from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
RECEIVER = SCRIPT_DIR / "quest_pc_receiver.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local end-to-end smoke test for quest_pc_receiver.py.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch the receiver.")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver bind host. Default: 127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=0, help="UDP port. Default: choose a free port.")
    parser.add_argument("--view-port", type=int, default=0, help="HTTP viewer port. Default: choose a free port.")
    parser.add_argument("--samples", type=int, default=20, help="Number of sample datagrams to send. Default: 20.")
    parser.add_argument("--keep", action="store_true", help="Keep the temporary output directory.")
    args = parser.parse_args()

    udp_port = args.udp_port or free_udp_port(args.host)
    view_port = args.view_port or free_tcp_port(args.host)
    tmp = Path(tempfile.mkdtemp(prefix="quest_receiver_smoke_"))
    output_root = tmp / "pc_recordings"
    raw_root = tmp / "raw"
    calib_root = tmp / "outputs"
    record_id = f"record_smoke_{int(time.time())}"

    cmd = [
        args.python,
        str(RECEIVER),
        "receive",
        "--host",
        args.host,
        "--port",
        str(udp_port),
        "--output-root",
        str(output_root),
        "--visualize",
        "--visualize-host",
        args.host,
        "--visualize-port",
        str(view_port),
        "--no-open-browser",
        "--no-calibration-http",
        "--no-flexiv-realsense",
        "--calibration-raw-root",
        str(raw_root),
        "--calibration-output-root",
        str(calib_root),
        "--single-session",
        "--quiet",
        "--timeout",
        "10",
        "--flush-every",
        "4",
        "--flush-interval-seconds",
        "0.1",
    ]
    process = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR.parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_http(args.host, view_port, process)
        send_recording(args.host, udp_port, record_id, args.samples)
        stdout, stderr = process.communicate(timeout=20)
        if process.returncode != 0:
            raise RuntimeError(
                f"receiver exited with {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        record_dir = single_record_dir(output_root)
        summary = read_json(record_dir / "pc_session_summary.json")
        samples = read_jsonl(record_dir / "pc_samples.jsonl")
        raw = read_jsonl(record_dir / "pc_telemetry_raw.jsonl")
        cache_path = Path(summary.get("replayVisualizationJson") or "")
        if not cache_path.exists():
            raise AssertionError(f"missing replay visualization cache: {cache_path}")
        cache = read_json(cache_path)
        assert summary["recordId"] == record_id, summary
        assert summary["closedReason"] == "recording_stop", summary
        assert summary["samples"] == args.samples, summary
        assert len(samples) == args.samples, (len(samples), summary)
        assert len(raw) == summary["rawMessagesWritten"], (len(raw), summary)
        assert len(raw) >= 2, (len(raw), summary)
        assert summary["rawSampleSkips"] > 0, summary
        assert "writerDroppedSampleMessages" in summary, summary
        assert "writerDroppedControlMessages" in summary, summary
        assert "writerEnqueueBackpressureEvents" in summary, summary
        assert cache.get("ok") is True, cache
        assert cache.get("cacheVersion") == 5, cache
        assert cache.get("payloadMode") == "visualization_cache", cache
        decimation = cache.get("visualizationDecimation")
        assert isinstance(decimation, dict), cache
        assert decimation.get("originalSamples") == args.samples, decimation
        assert decimation.get("visualizedSamples") == args.samples, decimation
        assert int(decimation.get("maxSamples") or 0) >= args.samples, decimation
        print(
            json.dumps(
                {
                    "ok": True,
                    "recordId": record_id,
                    "outputRoot": str(output_root),
                    "recordDir": str(record_dir),
                    "samples": summary["samples"],
                    "rawRows": len(raw),
                    "rawSampleSkips": summary["rawSampleSkips"],
                    "visualizedSamples": decimation.get("visualizedSamples"),
                    "cachePath": str(cache_path),
                    "kept": bool(args.keep),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if not args.keep:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


def free_udp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_http(host: str, port: int, process: subprocess.Popen[str], timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/robot/status"
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise RuntimeError(f"receiver exited early with {process.returncode}\n{stdout}\n{stderr}")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"viewer did not become ready at {url}: {last_error}")


def send_recording(host: str, port: int, record_id: str, samples: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        start_time = time.time()
        send(sock, host, port, {"type": "recording_start", "recordId": record_id, "sequence": 0})
        for index in range(samples):
            t = index / 90.0
            sample = {
                "type": "sample",
                "recordId": record_id,
                "sequence": index + 1,
                "sampleIndex": index,
                "isRecording": True,
                "telemetryMode": "recording",
                "recordingTimestampSeconds": t,
                "unityTimestampSeconds": start_time + t,
                "hasGaze": True,
                "gazeSource": "smoke",
                "gazePoint3DWorld": [0.25 + 0.001 * index, 1.0, 0.5],
                "gazePoint3DSource": "smoke",
                "gazeRayOrigin": [0.0, 1.0, 0.0],
                "gazeRayDirection": [0.0, 0.0, 1.0],
                "headPose": pose_payload([0.0, 1.5, 0.0]),
                "leftEyePose": pose_payload([-0.03, 1.5, 0.02]),
                "rightEyePose": pose_payload([0.03, 1.5, 0.02]),
                "leftController": controller_payload([-0.2, 1.1, 0.3]),
                "rightController": controller_payload([0.2, 1.1, 0.3]),
            }
            send(sock, host, port, sample)
            time.sleep(0.002)
        send(sock, host, port, {"type": "recording_stop", "recordId": record_id, "sequence": samples + 1})
    finally:
        sock.close()


def send(sock: socket.socket, host: str, port: int, payload: dict[str, object]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendto(data, (host, port))


def pose_payload(position: list[float]) -> dict[str, object]:
    return {
        "hasPose": True,
        "position": position,
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "source": "smoke",
    }


def controller_payload(position: list[float]) -> dict[str, object]:
    payload = pose_payload(position)
    payload.update(
        {
            "positionTracked": True,
            "rotationTracked": True,
            "indexTrigger": 0.0,
            "handTrigger": 0.0,
            "indexTriggerPressed": False,
            "handTriggerPressed": False,
        }
    )
    return payload


def single_record_dir(output_root: Path) -> Path:
    dirs = [item for item in output_root.iterdir() if item.is_dir()]
    if len(dirs) != 1:
        raise AssertionError(f"expected one record directory in {output_root}, got {[d.name for d in dirs]}")
    return dirs[0]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
