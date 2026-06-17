from __future__ import annotations

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import posixpath
import queue
import socket
import subprocess
import sys
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

import cv2
import numpy as np

from flexiv_realsense_bridge import (
    DEFAULT_END_CAMERA_SERIAL,
    DEFAULT_FLEXIV_RDK_ROOT,
    DEFAULT_FLEXIV_ROBOT_SN,
    FlexivRealSenseConfig,
    FlexivRealSenseManager,
    RobotRealsenseSession,
    ee_pose_diversity,
    transform_from_json,
)
from flexiv_realsense_diagnostics import DEFAULT_PORTS as DEFAULT_ROBOT_DIAGNOSTIC_PORTS
from flexiv_realsense_diagnostics import (
    DEFAULT_ROBOT_HOSTS,
    compatibility_info,
    elements_info,
    flexivrdk_info,
    interpret_result,
    network_info,
)
from flexiv_realsense_diagnostics import probe_hosts, realsense_info, robot_connection_info


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "pc_recordings"
DEFAULT_QUEST_LOCAL_ROOT = WORKSPACE_ROOT / "raw"
DEFAULT_CALIBRATION_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "pc_live_calibration"
DEFAULT_RIZON4_URDF = WORKSPACE_ROOT / "assets" / "urdf" / "flexiv_Rizon4_kinematics.urdf"
LATE_RECORDING_SAMPLE_GRACE_SECONDS = 5.0
MAX_CALIBRATION_HTTP_BODY_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
ALLOWED_ARTIFACT_SUFFIXES = {".html", ".json", ".jsonl", ".log", ".jpg", ".jpeg", ".png"}
QUEST_RECORD_COMMAND_PATH = "/sdcard/Android/data/com.Apricity.EyeTrackingTest/files/record_command.txt"
DEFAULT_ADB = Path(
    r"C:\Program Files\Unity\Hub\Editor\6000.0.60f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Receive Quest recording UDP telemetry on PC and compare the PC copy "
            "against the Quest-local trajectory.jsonl."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    receive_parser = subparsers.add_parser(
        "receive",
        help="Listen for Quest UDP recording telemetry and write one PC session folder per recording.",
    )
    receive_parser.add_argument("--host", default="0.0.0.0", help="Local bind address. Default: 0.0.0.0")
    receive_parser.add_argument("--port", type=int, default=9100, help="Local UDP port. Default: 9100")
    receive_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"PC session output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    receive_parser.add_argument(
        "--single-session",
        action="store_true",
        help="Exit after the first recording_stop message closes a session.",
    )
    receive_parser.add_argument(
        "--timeout",
        type=float,
        help="Exit if no datagram is received for this many seconds.",
    )
    receive_parser.add_argument(
        "--max-samples",
        type=int,
        help="Exit after this many sample messages have been written.",
    )
    receive_parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush output every N messages. Default: 1 for safest recording.",
    )
    receive_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print session start/stop and errors.",
    )
    receive_parser.add_argument(
        "--pull-quest-record",
        action="store_true",
        help="After recording_stop, use adb pull on the Quest outputDirectory into --quest-local-root.",
    )
    receive_parser.add_argument(
        "--adb",
        type=Path,
        default=DEFAULT_ADB if DEFAULT_ADB.exists() else Path("adb"),
        help="adb executable used with --pull-quest-record.",
    )
    receive_parser.add_argument(
        "--quest-local-root",
        type=Path,
        default=DEFAULT_QUEST_LOCAL_ROOT,
        help=f"Local folder for pulled Quest recordings. Default: {DEFAULT_QUEST_LOCAL_ROOT}",
    )
    receive_parser.add_argument(
        "--no-analyze-after-pull",
        action="store_true",
        help="Pull the Quest record but skip automatic Quest-vs-PC alignment analysis.",
    )
    receive_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open a live localhost HTML 3D viewer for received head/controller/gaze telemetry.",
    )
    receive_parser.add_argument(
        "--visualize-host",
        default="127.0.0.1",
        help="HTTP bind address for --visualize. Default: 127.0.0.1",
    )
    receive_parser.add_argument(
        "--visualize-port",
        type=int,
        default=8765,
        help="HTTP port for --visualize. Default: 8765",
    )
    receive_parser.add_argument(
        "--visualize-history",
        type=int,
        default=1200,
        help="Number of recent samples retained for newly opened viewers. Default: 1200",
    )
    receive_parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="With --visualize, start the web server but do not open the browser automatically.",
    )
    receive_parser.add_argument(
        "--viewer-adb",
        type=Path,
        default=DEFAULT_ADB if DEFAULT_ADB.exists() else Path("adb"),
        help=(
            "adb executable used by the live viewer debug buttons for Quest file commands. "
            "Only works when the receiver host can see the Quest over adb."
        ),
    )
    receive_parser.add_argument(
        "--record-live-preview",
        action="store_true",
        help="Also write non-recording live_preview samples to pc_recordings. Default: visualize only.",
    )
    receive_parser.add_argument(
        "--no-calibration-http",
        action="store_true",
        help="Disable the HTTP receiver used by Quest B-button PC calibration recording.",
    )
    receive_parser.add_argument(
        "--calibration-http-host",
        default="0.0.0.0",
        help="HTTP bind address for Quest PC calibration image upload. Default: 0.0.0.0",
    )
    receive_parser.add_argument(
        "--calibration-http-port",
        type=int,
        default=9101,
        help="HTTP port for Quest PC calibration image upload. Default: 9101",
    )
    receive_parser.add_argument(
        "--calibration-raw-root",
        type=Path,
        default=DEFAULT_QUEST_LOCAL_ROOT,
        help=f"Raw record folder written by PC calibration recording. Default: {DEFAULT_QUEST_LOCAL_ROOT}",
    )
    receive_parser.add_argument(
        "--calibration-output-root",
        type=Path,
        default=DEFAULT_CALIBRATION_OUTPUT_ROOT,
        help=f"Output root for automatic 25mm calibration. Default: {DEFAULT_CALIBRATION_OUTPUT_ROOT}",
    )
    receive_parser.add_argument(
        "--no-calibrate-after-pc-recording",
        action="store_true",
        help="Write PC calibration raw records but do not run calibrate_records_25mm.py after stop.",
    )
    receive_parser.add_argument(
        "--no-flexiv-realsense",
        action="store_true",
        help="Disable Flexiv/RealSense controls and synchronized robot capture.",
    )
    receive_parser.add_argument(
        "--flexiv-robot-sn",
        default=DEFAULT_FLEXIV_ROBOT_SN,
        help=f"Default Flexiv robot serial shown in the web UI. Default: {DEFAULT_FLEXIV_ROBOT_SN}",
    )
    receive_parser.add_argument(
        "--flexiv-pose-field",
        choices=["flange_pose", "tcp_pose"],
        default="flange_pose",
        help="RobotStates field to record as the end-effector pose. Default: flange_pose",
    )
    receive_parser.add_argument(
        "--flexiv-rdk",
        type=Path,
        default=DEFAULT_FLEXIV_RDK_ROOT,
        help="Optional legacy Flexiv RDK root containing lib_py. Default: use the active Python flexivrdk package.",
    )
    receive_parser.add_argument(
        "--flexiv-network-interface",
        action="append",
        dest="flexiv_network_interfaces",
        help=(
            "Optional local IPv4 address for the Flexiv RDK network interface whitelist. "
            "Repeat to allow multiple interfaces, for example --flexiv-network-interface 192.168.2.108."
        ),
    )
    receive_parser.add_argument(
        "--realsense-serial",
        default=DEFAULT_END_CAMERA_SERIAL,
        help=f"Default end-mounted RealSense serial. Default: {DEFAULT_END_CAMERA_SERIAL}",
    )
    receive_parser.add_argument("--realsense-width", type=int, default=1280, help="RealSense color width. Default: 1280")
    receive_parser.add_argument("--realsense-height", type=int, default=720, help="RealSense color height. Default: 720")
    receive_parser.add_argument("--realsense-fps", type=int, default=30, help="RealSense color FPS. Default: 30")
    receive_parser.add_argument("--realsense-manual-exposure", action="store_true", help="Disable RealSense RGB auto exposure.")
    receive_parser.add_argument("--realsense-exposure", type=float, default=None, help="Manual RealSense RGB exposure value.")
    receive_parser.add_argument("--realsense-gain", type=float, default=None, help="RealSense RGB gain value.")
    receive_parser.add_argument(
        "--robot-capture-interval",
        type=float,
        default=0.35,
        help="Seconds between synchronized robot/RealSense captures during B-button calibration. Default: 0.35",
    )
    receive_parser.add_argument(
        "--no-robot-hand-eye",
        action="store_true",
        help="Record robot/RealSense samples but skip automatic robot hand-eye calibration after B-button stop.",
    )
    receive_parser.add_argument(
        "--controller-motion-scale",
        type=float,
        default=1.0,
        help="Scale from right-controller displacement to robot TCP displacement. Default: 1.0",
    )
    receive_parser.add_argument(
        "--controller-motion-max-offset",
        type=float,
        default=0.18,
        help="Maximum robot TCP offset commanded from the controller anchor, in meters. Default: 0.18",
    )
    receive_parser.add_argument(
        "--controller-motion-max-step",
        type=float,
        default=0.015,
        help="Maximum TCP target position change per received sample, in meters. Default: 0.015",
    )
    receive_parser.set_defaults(func=receive)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compare Quest-local trajectory.jsonl with a PC receiver session.",
    )
    analyze_parser.add_argument(
        "--quest-record",
        type=Path,
        required=True,
        help="Quest record folder or path to trajectory.jsonl.",
    )
    analyze_parser.add_argument(
        "--pc-session",
        type=Path,
        help=(
            "PC session folder, pc_telemetry_raw.jsonl, or pc_samples.jsonl. "
            "Defaults to the newest folder under --output-root."
        ),
    )
    analyze_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Used only when --pc-session is omitted. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    analyze_parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the analysis summary JSON.",
    )
    analyze_parser.set_defaults(func=analyze)

    diagnose_parser = subparsers.add_parser(
        "diagnose-gaze-depth",
        help="Diagnose gaze3D ray-depth spikes in a PC recording, optionally against Quest-local trajectory.jsonl.",
    )
    diagnose_parser.add_argument(
        "--pc-session",
        type=Path,
        required=True,
        help="PC session folder containing pc_samples.jsonl and pc_calibration_snapshot.json.",
    )
    diagnose_parser.add_argument(
        "--quest-record",
        type=Path,
        help="Optional Quest record folder or trajectory.jsonl to compare with the PC samples.",
    )
    diagnose_parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the diagnostic summary JSON.",
    )
    diagnose_parser.set_defaults(func=diagnose_gaze_depth)

    args = parser.parse_args()
    return args.func(args)


class SessionWriter:
    def __init__(
        self,
        output_root: Path,
        record_id: str,
        first_message: dict[str, Any] | None,
        remote: tuple[str, int],
        pc_receive_unix_seconds: float,
        pc_receive_perf_counter_seconds: float,
        flush_every: int,
        calibration_output_root: Path | None = None,
    ) -> None:
        safe_record_id = sanitize_name(record_id or "unknown_record")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = safe_record_id if safe_record_id.startswith("record_") else f"{timestamp}_{safe_record_id}"
        self.directory = unique_directory(output_root / base_name)
        self.directory.mkdir(parents=True, exist_ok=False)

        self.raw_path = self.directory / "pc_telemetry_raw.jsonl"
        self.samples_path = self.directory / "pc_samples.jsonl"
        self.controllers_csv_path = self.directory / "pc_controllers.csv"
        self.summary_path = self.directory / "pc_session_summary.json"
        self.calibration_snapshot_path = self.directory / "pc_calibration_snapshot.json"

        self.raw_file = self.raw_path.open("w", encoding="utf-8", newline="\n")
        self.samples_file = self.samples_path.open("w", encoding="utf-8", newline="\n")
        self.controllers_file = self.controllers_csv_path.open("w", encoding="utf-8", newline="")
        self.controllers_writer = csv.DictWriter(
            self.controllers_file,
            fieldnames=[
                "recordId",
                "sequence",
                "sampleIndex",
                "recordingTimestampSeconds",
                "questUnityTimestampSeconds",
                "pcReceiveUnixSeconds",
                "pcReceivePerfCounterSeconds",
                "remote",
                "hand",
                "hasPose",
                "source",
                "missingReason",
                "positionTracked",
                "rotationTracked",
                "x",
                "y",
                "z",
                "qw",
                "qx",
                "qy",
                "qz",
            ],
        )
        self.controllers_writer.writeheader()

        self.record_id = record_id
        self.quest_output_directory = quest_output_directory(first_message)
        self.start_utc = datetime.now(timezone.utc).isoformat()
        self.start_pc_receive_unix_seconds = pc_receive_unix_seconds
        self.start_pc_receive_perf_counter_seconds = pc_receive_perf_counter_seconds
        self.start_remote = f"{remote[0]}:{remote[1]}"
        self.start_message = first_message
        self.flush_every = max(1, flush_every)
        self.calibration_output_root = calibration_output_root
        self.calibration_snapshot_start = recording_calibration_snapshot(calibration_output_root)
        self.calibration_snapshot_end: dict[str, Any] | None = None

        self.messages = 0
        self.samples = 0
        self.bad_controller_rows = 0
        self.left_pose_samples = 0
        self.right_pose_samples = 0
        self.left_sources: dict[str, int] = {}
        self.right_sources: dict[str, int] = {}
        self.left_missing_reasons: dict[str, int] = {}
        self.right_missing_reasons: dict[str, int] = {}
        self.last_sample_index: int | None = None
        self.closed = False

    def write(self, wrapper: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("session already closed")

        message = wrapper.get("message")
        if not isinstance(message, dict):
            return

        output_directory = quest_output_directory(message)
        if output_directory:
            self.quest_output_directory = output_directory

        self.raw_file.write(json_line(wrapper))
        self.messages += 1

        if message.get("type") == "sample":
            self.samples += 1
            sample_index = message.get("sampleIndex")
            if is_number(sample_index):
                self.last_sample_index = int(sample_index)
            self._write_sample(wrapper, message)

        if self.messages % self.flush_every == 0:
            self.flush()

    def _write_sample(self, wrapper: dict[str, Any], message: dict[str, Any]) -> None:
        compact = {
            "pcReceiveUtc": wrapper.get("pcReceiveUtc"),
            "pcReceiveUnixSeconds": wrapper.get("pcReceiveUnixSeconds"),
            "pcReceivePerfCounterSeconds": wrapper.get("pcReceivePerfCounterSeconds"),
            "remote": wrapper.get("remote"),
            "recordId": message.get("recordId"),
            "sequence": message.get("sequence"),
            "sampleIndex": message.get("sampleIndex"),
            "unityTimestampSeconds": message.get("unityTimestampSeconds"),
            "recordingTimestampSeconds": message.get("recordingTimestampSeconds"),
            "hasGaze": message.get("hasGaze"),
            "gazeSource": message.get("gazeSource"),
            "hasGazeHit": message.get("hasGazeHit"),
            "gazePointWorld": message.get("gazePointWorld"),
            "gazePoint3DWorld": message.get("gazePoint3DWorld"),
            "gazePoint3DSource": message.get("gazePoint3DSource"),
            "gazeRayOrigin": message.get("gazeRayOrigin"),
            "gazeRayDirection": message.get("gazeRayDirection"),
            "leftCameraPose": message.get("leftCameraPose"),
            "rightCameraPose": message.get("rightCameraPose"),
            "hasLeftEyePose": message.get("hasLeftEyePose"),
            "hasRightEyePose": message.get("hasRightEyePose"),
            "leftEyePoseSource": message.get("leftEyePoseSource"),
            "rightEyePoseSource": message.get("rightEyePoseSource"),
            "leftEyePose": message.get("leftEyePose"),
            "rightEyePose": message.get("rightEyePose"),
            "leftEyePosition": message.get("leftEyePosition"),
            "rightEyePosition": message.get("rightEyePosition"),
            "leftController": message.get("leftController"),
            "rightController": message.get("rightController"),
        }
        self.samples_file.write(json_line(compact))
        self._write_controller_csv_row(wrapper, message, "left")
        self._write_controller_csv_row(wrapper, message, "right")

    def _write_controller_csv_row(
        self,
        wrapper: dict[str, Any],
        message: dict[str, Any],
        hand: str,
    ) -> None:
        controller = message.get(f"{hand}Controller")
        if not isinstance(controller, dict):
            self.bad_controller_rows += 1
            controller = {}

        if hand == "left":
            count_source(self.left_sources, controller)
            count_missing_reason(self.left_missing_reasons, controller)
            if has_controller_pose(controller):
                self.left_pose_samples += 1
        else:
            count_source(self.right_sources, controller)
            count_missing_reason(self.right_missing_reasons, controller)
            if has_controller_pose(controller):
                self.right_pose_samples += 1

        position = controller.get("position")
        rotation = controller.get("rotation")
        x, y, z = vec3_or_empty(position)
        qw, qx, qy, qz = quat_or_empty(rotation)
        self.controllers_writer.writerow(
            {
                "recordId": message.get("recordId"),
                "sequence": message.get("sequence"),
                "sampleIndex": message.get("sampleIndex"),
                "recordingTimestampSeconds": message.get("recordingTimestampSeconds"),
                "questUnityTimestampSeconds": message.get("unityTimestampSeconds"),
                "pcReceiveUnixSeconds": wrapper.get("pcReceiveUnixSeconds"),
                "pcReceivePerfCounterSeconds": wrapper.get("pcReceivePerfCounterSeconds"),
                "remote": wrapper.get("remote"),
                "hand": hand,
                "hasPose": bool(controller.get("hasPose")),
                "source": controller.get("source"),
                "missingReason": controller.get("missingReason"),
                "positionTracked": controller.get("positionTracked"),
                "rotationTracked": controller.get("rotationTracked"),
                "x": x,
                "y": y,
                "z": z,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
            }
        )

    def flush(self) -> None:
        self.raw_file.flush()
        self.samples_file.flush()
        self.controllers_file.flush()

    def close(self, reason: str) -> dict[str, Any]:
        if self.closed:
            return self.summary(reason)

        self.calibration_snapshot_end = recording_calibration_snapshot(self.calibration_output_root)
        preferred_snapshot = self.preferred_calibration_snapshot()
        if preferred_snapshot is not None:
            self.calibration_snapshot_path.write_text(
                json.dumps(preferred_snapshot, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.flush()
        self.raw_file.close()
        self.samples_file.close()
        self.controllers_file.close()
        summary = self.summary(reason)
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self.closed = True
        return summary

    def summary(self, reason: str) -> dict[str, Any]:
        summary = {
            "recordId": self.record_id,
            "sessionDirectory": str(self.directory),
            "questOutputDirectory": self.quest_output_directory,
            "closedReason": reason,
            "startUtc": self.start_utc,
            "startRemote": self.start_remote,
            "startPcReceiveUnixSeconds": self.start_pc_receive_unix_seconds,
            "startPcReceivePerfCounterSeconds": self.start_pc_receive_perf_counter_seconds,
            "messages": self.messages,
            "samples": self.samples,
            "lastSampleIndex": self.last_sample_index,
            "leftPoseSamples": self.left_pose_samples,
            "rightPoseSamples": self.right_pose_samples,
            "leftSources": self.left_sources,
            "rightSources": self.right_sources,
            "leftMissingReasons": self.left_missing_reasons,
            "rightMissingReasons": self.right_missing_reasons,
            "rawJsonl": str(self.raw_path),
            "samplesJsonl": str(self.samples_path),
            "controllersCsv": str(self.controllers_csv_path),
        }
        if self.calibration_snapshot_start is not None:
            summary["calibrationSnapshotAtStart"] = self.calibration_snapshot_start
        if self.calibration_snapshot_end is not None:
            summary["calibrationSnapshotAtEnd"] = self.calibration_snapshot_end
        preferred_snapshot = self.preferred_calibration_snapshot()
        if preferred_snapshot is not None:
            summary["calibrationSnapshot"] = preferred_snapshot
            summary["calibrationSnapshotJson"] = str(self.calibration_snapshot_path)
        return summary

    def preferred_calibration_snapshot(self) -> dict[str, Any] | None:
        if is_successful_calibration_snapshot(self.calibration_snapshot_start):
            return self.calibration_snapshot_start
        if is_successful_calibration_snapshot(self.calibration_snapshot_end):
            return self.calibration_snapshot_end
        return self.calibration_snapshot_start or self.calibration_snapshot_end


class LiveTelemetryVisualizer:
    def __init__(
        self,
        host: str,
        port: int,
        history_limit: int = 1200,
        robot_manager: FlexivRealSenseManager | None = None,
        adb_path: Path | None = None,
        calibration_raw_root: Path | None = None,
        calibration_output_root: Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.history_limit = max(1, history_limit)
        self.recording_root = DEFAULT_OUTPUT_ROOT.resolve()
        self.calibration_raw_root = (calibration_raw_root or DEFAULT_QUEST_LOCAL_ROOT).resolve()
        self.calibration_output_root = (calibration_output_root or DEFAULT_CALIBRATION_OUTPUT_ROOT).resolve()
        self.robot_manager = robot_manager
        self.adb_path = adb_path
        self.history: list[dict[str, Any]] = []
        self.clients: list[queue.Queue[str | None]] = []
        self.lock = threading.Lock()
        self.server = self._make_server()
        self.thread = threading.Thread(target=self.server.serve_forever, name="quest-telemetry-visualizer", daemon=True)

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.server.server_address[1]}/"

    def start(self, open_browser: bool) -> None:
        self.thread.start()
        print(f"Live 3D viewer: {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        with self.lock:
            clients = list(self.clients)
            self.clients.clear()
        for client in clients:
            client.put(None)
        self.server.shutdown()
        self.server.server_close()

    def publish(self, message: dict[str, Any], wrapper: dict[str, Any]) -> None:
        event = visualizer_event_from_message(message, wrapper)
        if event is None:
            return
        self.publish_event(event)

    def publish_event(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self.lock:
            self.history.append(event)
            if len(self.history) > self.history_limit:
                del self.history[: len(self.history) - self.history_limit]
            clients = list(self.clients)

        for client in clients:
            try:
                client.put_nowait(payload)
            except queue.Full:
                pass

    def robot_status_payload(self) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        return self.robot_manager.status()

    def camera_list_payload(self) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled", "cameras": []}
        return self.robot_manager.list_cameras()

    def preflight_status_payload(self) -> dict[str, Any]:
        with self.lock:
            recent_samples = [event for event in self.history if event.get("type") == "sample"]
            last_sample = recent_samples[-1] if recent_samples else None
            controller_window = recent_controller_status(recent_samples)
        robot_status = self.robot_status_payload()
        camera_status = self.camera_list_payload()
        board_status = self.latest_robot_board_check_payload()
        model_status = rizon4_model_payload()
        return build_preflight_status(last_sample, controller_window, robot_status, camera_status, board_status, model_status)

    def robot_board_check_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        self.robot_manager.configure(payload)
        try:
            result = self.robot_manager.check_end_camera_board(WORKSPACE_ROOT / "board_checks")
            add_board_check_artifact_urls(result)
            self.publish_event({"type": "robot_status", "stage": "board_check", "boardCheck": result})
            return result
        except Exception as exc:
            result = {
                "ok": False,
                "enabled": True,
                "error": str(exc),
                "message": "end-camera checkerboard check failed",
            }
            self.publish_event({"type": "robot_status", "stage": "board_check_failed", "boardCheck": result})
            return result

    def latest_robot_board_check_payload(self) -> dict[str, Any]:
        path = WORKSPACE_ROOT / "board_checks" / "end_camera" / "latest_board_check.json"
        if not path.exists():
            return {"ok": False, "reason": "no board check yet"}
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("latest board check is not an object")
            add_board_check_artifact_urls(result)
            return result
        except Exception as exc:
            return {"ok": False, "error": str(exc), "reason": "could not read latest board check"}

    def robot_diagnostics_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        self.robot_manager.configure(payload)
        config = self.robot_manager.config
        robot_sn = str(payload.get("robotSn") or config.robot_sn or "").strip()
        interfaces = config.flexiv_network_interfaces
        flexivrdk = flexivrdk_info()
        elements = elements_info(Path("/ssd1/mzc/FlexivElementsStudio"))
        result: dict[str, Any] = {
            "ok": True,
            "enabled": True,
            "timestampUnixSeconds": time.time(),
            "config": self.robot_manager.status().get("config"),
            "flexivrdk": flexivrdk,
            "realsense": realsense_info(),
            "network": network_info(),
            "elements": elements,
            "compatibility": compatibility_info(flexivrdk, elements),
            "probe": probe_hosts(DEFAULT_ROBOT_HOSTS, DEFAULT_ROBOT_DIAGNOSTIC_PORTS),
            "robotConnection": robot_connection_info(robot_sn, interfaces),
        }
        result["interpretation"] = interpret_result(result)
        self.publish_event({"type": "robot_status", "stage": "diagnostics", "diagnostics": result})
        return result

    def configure_robot_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        status = self.robot_manager.configure(payload)
        self.publish_event({"type": "robot_status", "stage": "configured", "status": status})
        return status

    def connect_robot_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        result = self.robot_manager.connect_robot(payload)
        self.publish_event({"type": "robot_status", "stage": "connect", **result})
        return result

    def disconnect_robot_payload(self) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        status = self.robot_manager.disconnect_robot()
        self.publish_event({"type": "robot_status", "stage": "disconnect", "status": status})
        return status

    def arm_robot_motion_payload(self) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        controller_window = self.controller_window_payload()
        right_recent = controller_window.get("right", {}) if isinstance(controller_window, dict) else {}
        if int(right_recent.get("validSamples") or 0) <= 0:
            result = {
                "ok": False,
                "enabled": True,
                "error": "right_controller_pose_missing",
                "message": "Right Touch controller pose is required before arming robot motion.",
                "controllerWindow": controller_window,
            }
            self.publish_event({"type": "robot_status", "stage": "arm_motion_rejected", **result})
            return result
        result = self.robot_manager.arm_motion()
        self.publish_event({"type": "robot_status", "stage": "arm_motion", **result})
        return result

    def disarm_robot_motion_payload(self) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        result = self.robot_manager.disarm_motion()
        self.publish_event({"type": "robot_status", "stage": "disarm_motion", **result})
        return result

    def controller_window_payload(self) -> dict[str, Any]:
        with self.lock:
            recent_samples = [event for event in self.history if event.get("type") == "sample"]
        return recent_controller_status(recent_samples)

    def quest_adb_status_payload(self) -> dict[str, Any]:
        return quest_adb_status(self.adb_path)

    def quest_adb_calibration_command_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = str(payload.get("command") or "").strip().lower()
        aliases = {
            "start": "calib_start",
            "calib_start": "calib_start",
            "calibration_start": "calib_start",
            "stop": "calib_stop",
            "calib_stop": "calib_stop",
            "calibration_stop": "calib_stop",
            "toggle": "calib_toggle",
            "calib_toggle": "calib_toggle",
            "calibration_toggle": "calib_toggle",
        }
        if command not in aliases:
            return {"ok": False, "error": f"unsupported calibration command: {command}"}
        result = send_quest_record_command(self.adb_path, aliases[command])
        self.publish_event({"type": "quest_adb_status", "stage": "calibration_command", **result})
        return result

    def _make_server(self) -> ThreadingHTTPServer:
        visualizer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                if parsed.path in ("/", "/index.html"):
                    self._send_html()
                    return
                if parsed.path == "/events":
                    self._send_events()
                    return
                if parsed.path == "/recordings":
                    self._send_recordings_html()
                    return
                if parsed.path == "/recordings/list":
                    self._send_json(
                        recording_replay_list(
                            visualizer.recording_root,
                            calibration_raw_root=visualizer.calibration_raw_root,
                            calibration_output_root=visualizer.calibration_output_root,
                        )
                    )
                    return
                if parsed.path == "/recordings/replay":
                    self._send_recording_replay_json(parsed)
                    return
                if parsed.path == "/calibration/latest":
                    self._send_json(latest_calibration_snapshot(DEFAULT_CALIBRATION_OUTPUT_ROOT))
                    return
                if parsed.path == "/quest/adb/status":
                    self._send_json(visualizer.quest_adb_status_payload())
                    return
                if parsed.path == "/preflight/status":
                    self._send_json(visualizer.preflight_status_payload())
                    return
                if parsed.path == "/robot/status":
                    self._send_json(visualizer.robot_status_payload())
                    return
                if parsed.path == "/robot/diagnostics":
                    self._send_json(visualizer.robot_diagnostics_payload({}))
                    return
                if parsed.path == "/robot/board-check/latest":
                    self._send_json(visualizer.latest_robot_board_check_payload())
                    return
                if parsed.path == "/cameras/list":
                    self._send_json(visualizer.camera_list_payload())
                    return
                if parsed.path == "/robot/urdf":
                    self._send_urdf()
                    return
                if parsed.path == "/robot/model":
                    self._send_json(rizon4_model_payload())
                    return
                if parsed.path == "/artifact":
                    self._send_artifact(parsed)
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    self.send_error(400, "bad Content-Length")
                    return
                if length < 0 or length > 1024 * 1024:
                    self.send_error(413, "request body too large")
                    return
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8")) if body else {}
                    if not isinstance(payload, dict):
                        raise ValueError("JSON body must be an object")
                    if parsed.path == "/robot/configure":
                        self._send_json(visualizer.configure_robot_payload(payload))
                        return
                    if parsed.path == "/robot/connect":
                        self._send_json(visualizer.connect_robot_payload(payload))
                        return
                    if parsed.path == "/robot/diagnostics":
                        self._send_json(visualizer.robot_diagnostics_payload(payload))
                        return
                    if parsed.path == "/robot/board-check":
                        self._send_json(visualizer.robot_board_check_payload(payload))
                        return
                    if parsed.path == "/robot/disconnect":
                        self._send_json(visualizer.disconnect_robot_payload())
                        return
                    if parsed.path == "/robot/arm-motion":
                        self._send_json(visualizer.arm_robot_motion_payload())
                        return
                    if parsed.path == "/robot/disarm-motion":
                        self._send_json(visualizer.disarm_robot_motion_payload())
                        return
                    if parsed.path == "/quest/adb/calibration-command":
                        self._send_json(visualizer.quest_adb_calibration_command_payload(payload))
                        return
                    self.send_error(404)
                except Exception as exc:
                    self.send_error(500, str(exc))

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_html(self) -> None:
                html = LIVE_VIEWER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)

            def _send_recordings_html(self) -> None:
                html = RECORDINGS_REPLAY_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)

            def _send_json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _send_recording_replay_json(self, parsed: Any) -> None:
                query = parse_qs(parsed.query)
                record_id = first_query(query, "recordId")
                if not record_id:
                    self.send_error(400, "missing recordId")
                    return
                try:
                    source = first_query(query, "source")
                    payload = build_recording_replay_payload(
                        visualizer.recording_root,
                        record_id,
                        source=source,
                        calibration_raw_root=visualizer.calibration_raw_root,
                        calibration_output_root=visualizer.calibration_output_root,
                    )
                except FileNotFoundError as exc:
                    self.send_error(404, str(exc))
                    return
                except ValueError as exc:
                    self.send_error(400, str(exc))
                    return
                self._send_json(payload)

            def _send_artifact(self, parsed: Any) -> None:
                query = parse_qs(parsed.query)
                path_text = first_query(query, "path")
                if not path_text:
                    self.send_error(400, "missing path")
                    return
                try:
                    path = resolve_artifact_path(path_text)
                except ValueError as exc:
                    self.send_error(403, str(exc))
                    return
                if not path.exists() or not path.is_file():
                    self.send_error(404)
                    return
                suffix = path.suffix.lower()
                if suffix not in ALLOWED_ARTIFACT_SUFFIXES:
                    self.send_error(403, "unsupported artifact type")
                    return
                size = path.stat().st_size
                if size > MAX_ARTIFACT_BYTES:
                    self.send_error(413, "artifact too large")
                    return
                if suffix in (".jpg", ".jpeg"):
                    content_type = "image/jpeg"
                elif suffix == ".png":
                    content_type = "image/png"
                elif suffix == ".html":
                    content_type = "text/html; charset=utf-8"
                elif suffix in (".json", ".jsonl"):
                    content_type = "application/json; charset=utf-8"
                elif suffix == ".log":
                    content_type = "text/plain; charset=utf-8"
                else:
                    content_type = "application/octet-stream"
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _send_urdf(self) -> None:
                if not DEFAULT_RIZON4_URDF.exists():
                    self.send_error(404, "Rizon4 URDF asset not found")
                    return
                data = DEFAULT_RIZON4_URDF.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/xml; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _send_events(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                client: queue.Queue[str | None] = queue.Queue(maxsize=256)
                with visualizer.lock:
                    history = list(visualizer.history)
                    visualizer.clients.append(client)

                try:
                    for event in history:
                        self._write_sse(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                    self.wfile.flush()

                    while True:
                        item = client.get()
                        if item is None:
                            return
                        self._write_sse(item)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionError, TimeoutError, OSError):
                    return
                finally:
                    with visualizer.lock:
                        if client in visualizer.clients:
                            visualizer.clients.remove(client)

            def _write_sse(self, payload: str) -> None:
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))

        return ThreadingHTTPServer((self.host, self.port), Handler)


class PcCalibrationSession:
    def __init__(
        self,
        raw_root: Path,
        output_root: Path,
        record_id: str,
        start_message: dict[str, Any],
        visualizer: "LiveTelemetryVisualizer | None",
        run_calibration: bool,
        robot_manager: FlexivRealSenseManager | None = None,
    ) -> None:
        self.record_id = sanitize_name(record_id)
        self.raw_root = raw_root.resolve()
        self.output_root = output_root.resolve()
        self.directory = unique_directory(self.raw_root / self.record_id)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.raw_record_name = self.directory.name
        self.output_directory = unique_directory(self.output_root / self.record_id)
        self.visualizer = visualizer
        self.run_calibration = run_calibration
        self.robot_manager = robot_manager
        self.robot_session: RobotRealsenseSession | None = None
        self.robot_realsense_directory: Path | None = None
        self.robot_start_status: dict[str, Any] | None = None

        metadata = dict(start_message.get("metadata") or {})
        metadata["outputDirectory"] = str(self.directory)
        metadata["startTimeUtc"] = start_message.get("startTimeUtc") or datetime.now(timezone.utc).isoformat()
        metadata.setdefault("schemaVersion", "quest_pc_calibration_receiver_metadata_v1")
        metadata["leftVideoFileName"] = safe_leaf_filename(metadata.get("leftVideoFileName"), "left_recording.mp4")
        metadata["rightVideoFileName"] = safe_leaf_filename(metadata.get("rightVideoFileName"), "right_recording.mp4")
        metadata["leftFrameMetadataFileName"] = safe_leaf_filename(
            metadata.get("leftFrameMetadataFileName"),
            "left_frames.jsonl",
        )
        metadata["rightFrameMetadataFileName"] = safe_leaf_filename(
            metadata.get("rightFrameMetadataFileName"),
            "right_frames.jsonl",
        )
        metadata["trajectoryFileName"] = safe_leaf_filename(metadata.get("trajectoryFileName"), "trajectory.jsonl")
        self.metadata = metadata

        self.left_frames_path = self.directory / str(metadata["leftFrameMetadataFileName"])
        self.right_frames_path = self.directory / str(metadata["rightFrameMetadataFileName"])
        self.trajectory_path = self.directory / str(metadata["trajectoryFileName"])
        self.left_frames_file = self.left_frames_path.open("w", encoding="utf-8", newline="\n")
        self.right_frames_file = self.right_frames_path.open("w", encoding="utf-8", newline="\n")
        self.trajectory_file = self.trajectory_path.open("w", encoding="utf-8", newline="\n")
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.frame_counts = {"left": 0, "right": 0}
        self.sample_count = 0
        self.start_perf = time.perf_counter()
        self.closed = False
        self.lock = threading.Lock()
        self.robot_start_status = self._robot_start_status("not_started")
        if self.robot_manager is not None:
            self.robot_session = self.robot_manager.start_session(
                self.directory,
                self.record_id,
                self.visualizer.publish_event if self.visualizer is not None else None,
            )
            if self.robot_session is not None:
                self.robot_realsense_directory = self.robot_session.directory
            self.robot_start_status = self._robot_start_status("recording" if self.robot_session is not None else "not_recording")
        self.metadata["robotStartStatus"] = self.robot_start_status
        self.publish_status("recording", 0.0, "PC calibration recording", robotStartStatus=self.robot_start_status)

    def write_frame(self, side: str, query: dict[str, list[str]], body: bytes) -> None:
        with self.lock:
            if self.closed:
                return
            frame_index = int_param(query, "frameIndex", self.frame_counts.get(side, 0))
            unity_ts = float_param(query, "unityTimestampSeconds", 0.0)
            camera_ts = float_param(query, "cameraTimestampSeconds", -1.0)
            width = int_param(query, "width", 0)
            height = int_param(query, "height", 0)
            flip_vertical = bool_param(query, "flipVertical", True)
            pose = float_list_param(query, "pose")

            array = np.frombuffer(body, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("could not decode JPEG frame")
            if flip_vertical:
                frame = cv2.flip(frame, 0)
            if width > 0 and height > 0 and (frame.shape[1] != width or frame.shape[0] != height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            writer = self._video_writer(side, frame.shape[1], frame.shape[0])
            writer.write(frame)
            row = {
                "frameIndex": frame_index,
                "cameraTimestampSeconds": camera_ts,
                "unityTimestampSeconds": unity_ts,
                "pose": pose,
            }
            target = self.left_frames_file if side == "left" else self.right_frames_file
            target.write(json_line(row))
            target.flush()
            self.frame_counts[side] = max(self.frame_counts.get(side, 0), frame_index + 1)

    def write_sample(self, message: dict[str, Any]) -> None:
        with self.lock:
            if self.closed:
                return
            sample = message.get("sample")
            if not isinstance(sample, dict):
                return
            sample["telemetryMode"] = sample.get("telemetryMode") or "pc_calibration_recording"
            sample["isRecording"] = True
            sample["recordId"] = self.record_id
            self.trajectory_file.write(json_line(sample))
            self.trajectory_file.flush()
            self.sample_count += 1
            robot_session = self.robot_session
        if robot_session is not None:
            robot_sample = dict(sample)
            robot_sample["pcReceivePerfCounterSeconds"] = message.get("pcReceivePerfCounterSeconds")
            robot_session.update_controller_motion(robot_sample)
            robot_session.record_sample(robot_sample)

    def close(self, stop_message: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return self.summary("already_closed")
            self.closed = True
            for writer in self.video_writers.values():
                writer.release()
            self.video_writers.clear()
            self.left_frames_file.close()
            self.right_frames_file.close()
            self.trajectory_file.close()
            duration = (
                float(stop_message.get("durationSeconds"))
                if isinstance(stop_message, dict) and is_number(stop_message.get("durationSeconds"))
                else time.perf_counter() - self.start_perf
            )
            self.metadata["durationSeconds"] = duration
            self.metadata["leftFrameCount"] = self.frame_counts["left"]
            self.metadata["rightFrameCount"] = self.frame_counts["right"]
            self.metadata["trajectorySampleCount"] = self.sample_count
            self.metadata["recordingQualitySummary"] = self._quality_summary()
            robot_session = self.robot_session
            self.robot_session = None
            if robot_session is not None:
                self.robot_realsense_directory = robot_session.directory
            robot_summary = None
        if self.robot_manager is not None and robot_session is not None:
            robot_summary = self.robot_manager.stop_session(robot_session)
        with self.lock:
            if robot_summary is not None:
                self.metadata["robotRealSenseSummary"] = robot_summary
            self.metadata["robotStartStatus"] = self.robot_start_status
            (self.directory / "quest_camera_metadata.json").write_text(
                json.dumps(self.metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary = self.summary("calibration_stop")
            if robot_summary is not None:
                summary["robotRealSense"] = robot_summary
            (self.directory / "pc_calibration_session_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self.publish_status("recorded", 0.05, "PC calibration recording saved", summary=summary)
        if self.run_calibration:
            thread = threading.Thread(target=self._run_calibration_worker, name=f"calibrate-{self.record_id}", daemon=True)
            thread.start()
        return summary

    def summary(self, reason: str) -> dict[str, Any]:
        return {
            "recordId": self.record_id,
            "rawRecordName": self.raw_record_name,
            "rawRecordDirectory": str(self.directory),
            "outputDirectory": str(self.output_directory),
            "closedReason": reason,
            "leftFrames": self.frame_counts["left"],
            "rightFrames": self.frame_counts["right"],
            "samples": self.sample_count,
            "robotRealSenseDirectory": str(self.robot_realsense_directory) if self.robot_realsense_directory is not None else None,
            "robotStartStatus": self.robot_start_status,
        }

    def publish_status(self, stage: str, progress: float, message: str, **extra: Any) -> None:
        if self.visualizer is None:
            return
        self.visualizer.publish_event(
            {
                "type": "calibration_status",
                "recordId": self.record_id,
                "stage": stage,
                "progress": progress,
                "message": message,
                **extra,
            }
        )

    def _robot_start_status(self, stage: str) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"enabled": False, "stage": stage, "recording": False, "reason": "robot manager disabled"}
        status = self.robot_manager.status()
        robot = status.get("robot") if isinstance(status, dict) else {}
        config = status.get("config") if isinstance(status, dict) else {}
        active = status.get("activeSession") if isinstance(status, dict) else None
        connected = bool(isinstance(robot, dict) and robot.get("connected"))
        motion_armed = bool(isinstance(robot, dict) and robot.get("motionArmed"))
        controller_motion = bool(isinstance(config, dict) and config.get("controllerMotionEnabled"))
        robot_sn = robot.get("robotSn") or config.get("robotSn") if isinstance(robot, dict) and isinstance(config, dict) else None
        pose_field = robot.get("poseField") or config.get("poseField") if isinstance(robot, dict) and isinstance(config, dict) else None
        camera_serial = config.get("cameraSerial") if isinstance(config, dict) else None
        recording = bool(self.robot_session is not None)
        reason = "recording"
        if not connected:
            reason = "Flexiv robot is not connected"
        elif not config.get("cameraSerial"):
            reason = "RealSense camera serial is empty"
        elif not recording:
            reason = self.robot_manager.last_error or "robot RealSense session did not start"
        elif not motion_armed or not controller_motion:
            reason = "robot session records images/poses, but controller motion is not armed"
        return {
            "enabled": True,
            "stage": stage,
            "recording": recording,
            "reason": reason,
            "robotConnected": connected,
            "robotSn": robot_sn,
            "poseField": pose_field,
            "cameraSerial": camera_serial,
            "motionArmed": motion_armed,
            "controllerMotionEnabled": controller_motion,
            "activeSession": active,
        }

    def _video_writer(self, side: str, width: int, height: int) -> cv2.VideoWriter:
        existing = self.video_writers.get(side)
        if existing is not None:
            return existing
        filename = str(self.metadata["leftVideoFileName"] if side == "left" else self.metadata["rightVideoFileName"])
        path = self.directory / filename
        fps = int(self.metadata.get("fps") or 15)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {path}")
        self.video_writers[side] = writer
        return writer

    def _quality_summary(self) -> str:
        if self.frame_counts["left"] <= 0:
            return "missing_left_video_frames"
        if self.metadata.get("recordRightCamera", True) and self.frame_counts["right"] <= 0:
            return "missing_right_video_frames"
        if self.sample_count <= 0:
            return "missing_trajectory_samples"
        return "ok"

    def _run_calibration_worker(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(WORKSPACE_ROOT / "scripts" / "calibrate_records_25mm.py"),
            "--records",
            self.raw_record_name,
            "--raw-root",
            str(self.raw_root),
            "--output-root",
            str(self.output_directory),
        ]
        env = dict(os.environ)
        env["QUEST_CALIB_PROGRESS"] = "1"
        self.publish_status("calibrating", 0.08, "Running 25mm calibration")
        log_path = self.output_directory / "calibration_run.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command,
                cwd=str(WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("PROGRESS_JSON "):
                    try:
                        payload = json.loads(line[len("PROGRESS_JSON ") :])
                    except json.JSONDecodeError:
                        continue
                    self.publish_status(
                        str(payload.get("stage") or "calibrating"),
                        float(payload.get("progress") or 0.1),
                        "Running 25mm calibration",
                        progressPayload=payload,
                    )
            returncode = process.wait()
        if returncode != 0:
            failure_event = calibration_failure_event(
                self.raw_record_name,
                self.output_directory,
                log_path,
                returncode,
            )
            failure_event["recordId"] = self.record_id
            failure_event["rawRecordName"] = self.raw_record_name
            if self.visualizer is not None:
                self.visualizer.publish_event(failure_event)
            self.publish_status(
                "failed",
                1.0,
                failure_event.get("reason") or f"Calibration failed with exit code {returncode}",
                logPath=str(log_path),
                failurePath=failure_event.get("failurePath"),
                diagnostics=failure_event.get("diagnostics"),
                returnCode=returncode,
            )
            if self.robot_manager is not None and self.robot_realsense_directory is not None:
                self._run_robot_hand_eye_worker(failure_event)
            return
        result_path = self.output_directory / "calibration_result_25mm.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        event = calibration_result_event(self.raw_record_name, result, result_path)
        event["recordId"] = self.record_id
        event["rawRecordName"] = self.raw_record_name
        if self.visualizer is not None:
            self.visualizer.publish_event(event)
        self.publish_status("done", 1.0, "Calibration complete", resultPath=str(result_path))
        if self.robot_manager is not None and self.robot_realsense_directory is not None:
            self._run_robot_hand_eye_worker(event)

    def _run_robot_hand_eye_worker(self, quest_calibration_event: dict[str, Any] | None) -> None:
        if self.robot_manager is None or self.robot_realsense_directory is None:
            return
        if not self.robot_manager.config.run_hand_eye:
            return
        has_quest_alignment = bool(is_successful_calibration_snapshot(quest_calibration_event))
        message = (
            "Running Flexiv/RealSense hand-eye calibration"
            if has_quest_alignment
            else "Running Flexiv/RealSense hand-eye calibration without Quest-board alignment"
        )
        self.publish_status("robot_calibrating", 1.0, message)
        result = self.robot_manager.calibrate_session(
            self.robot_realsense_directory,
            quest_calibration_event,
            self.visualizer.publish_event if self.visualizer is not None else None,
        )
        if result.get("ok"):
            self.publish_status("robot_done", 1.0, "Flexiv/RealSense hand-eye calibration complete")
        else:
            self.publish_status("robot_failed", 1.0, result.get("error") or "Flexiv/RealSense hand-eye calibration failed")


class PcCalibrationHttpReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        raw_root: Path,
        output_root: Path,
        visualizer: LiveTelemetryVisualizer | None,
        run_calibration: bool,
        robot_manager: FlexivRealSenseManager | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.raw_root = raw_root
        self.output_root = output_root
        self.visualizer = visualizer
        self.run_calibration = run_calibration
        self.robot_manager = robot_manager
        self.sessions: dict[str, PcCalibrationSession] = {}
        self.lock = threading.Lock()
        self.server = self._make_server()
        self.thread = threading.Thread(target=self.server.serve_forever, name="quest-pc-calibration-http", daemon=True)

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.server.server_address[1]}/"

    def start(self) -> None:
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        print(f"PC calibration HTTP receiver: {self.url}", flush=True)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _session(self, record_id: str) -> PcCalibrationSession | None:
        with self.lock:
            return self.sessions.get(record_id)

    def preflight_status(self) -> dict[str, Any]:
        if self.visualizer is None:
            return {
                "ok": True,
                "ready": True,
                "summary": "preflight unavailable because live visualizer is disabled",
                "checks": [],
            }
        return self.visualizer.preflight_status_payload()

    def publish_calibration_rejected(self, record_id: str, preflight: dict[str, Any]) -> None:
        if self.visualizer is None:
            return
        failed = [
            check
            for check in preflight.get("checks", [])
            if isinstance(check, dict) and check.get("required", True) and not check.get("ok")
        ]
        detail = "; ".join(
            f"{check.get('label') or check.get('id')}: {check.get('detail') or 'not ready'}"
            for check in failed
        )
        self.visualizer.publish_event(
            {
                "type": "calibration_status",
                "recordId": record_id,
                "stage": "rejected",
                "progress": 0.0,
                "message": "B calibration start rejected: preflight is not ready",
                "preflight": preflight,
                "detail": detail,
            }
        )

    def _make_server(self) -> ThreadingHTTPServer:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    self.send_error(400, "bad Content-Length")
                    return
                if length < 0 or length > MAX_CALIBRATION_HTTP_BODY_BYTES:
                    self.send_error(413, "request body too large")
                    return
                body = self.rfile.read(length)
                try:
                    if parsed.path == "/calibration/start":
                        self._handle_start(body)
                    elif parsed.path == "/calibration/frame":
                        self._handle_frame(parsed, body)
                    elif parsed.path == "/calibration/sample":
                        self._handle_sample(body)
                    elif parsed.path == "/calibration/stop":
                        self._handle_stop(body)
                    else:
                        self.send_error(404)
                except Exception as exc:  # pragma: no cover - defensive runtime endpoint
                    print(f"[calibration-http-error] {exc}", file=sys.stderr, flush=True)
                    self.send_error(500, str(exc))

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _handle_start(self, body: bytes) -> None:
                message = json.loads(body.decode("utf-8"))
                record_id = sanitize_name(str(message.get("recordId") or f"record_pc_calib_{datetime.now():%Y%m%d_%H%M%S}"))
                allow_unready = bool(
                    message.get("allowUnready")
                    or message.get("startAnyway")
                    or message.get("ignorePreflight")
                )
                preflight = receiver.preflight_status()
                if not allow_unready and not bool(preflight.get("ready")):
                    payload = {
                        "ok": False,
                        "error": "preflight_not_ready",
                        "recordId": record_id,
                        "preflight": preflight,
                    }
                    receiver.publish_calibration_rejected(record_id, preflight)
                    self._json_response(payload, status=409)
                    return
                with receiver.lock:
                    existing = receiver.sessions.pop(record_id, None)
                    if existing is not None:
                        existing.close({"durationSeconds": 0.0})
                    session = PcCalibrationSession(
                        receiver.raw_root,
                        receiver.output_root,
                        record_id,
                        message,
                        receiver.visualizer,
                        receiver.run_calibration,
                        receiver.robot_manager,
                    )
                    receiver.sessions[record_id] = session
                print(f"Started PC calibration raw record: {session.directory}", flush=True)
                self._json_response({
                    "ok": True,
                    "recordId": record_id,
                    "rawRecordDirectory": str(session.directory),
                    "preflight": preflight,
                    "allowUnready": allow_unready,
                })

            def _handle_frame(self, parsed: Any, body: bytes) -> None:
                query = parse_qs(parsed.query)
                record_id = sanitize_name(str(first_query(query, "recordId") or ""))
                side = str(first_query(query, "side") or "")
                if side not in ("left", "right"):
                    raise ValueError(f"bad side: {side}")
                session = receiver._session(record_id)
                if session is None:
                    raise ValueError(f"unknown calibration recordId: {record_id}")
                session.write_frame(side, query, body)
                self._json_response({"ok": True})

            def _handle_sample(self, body: bytes) -> None:
                message = json.loads(body.decode("utf-8"))
                record_id = sanitize_name(str(message.get("recordId") or ""))
                session = receiver._session(record_id)
                if session is None:
                    raise ValueError(f"unknown calibration recordId: {record_id}")
                message["pcReceiveUnixSeconds"] = time.time()
                message["pcReceivePerfCounterSeconds"] = time.perf_counter()
                session.write_sample(message)
                self._json_response({"ok": True})

            def _handle_stop(self, body: bytes) -> None:
                message = json.loads(body.decode("utf-8"))
                record_id = sanitize_name(str(message.get("recordId") or ""))
                with receiver.lock:
                    session = receiver.sessions.pop(record_id, None)
                if session is None:
                    self._json_response({"ok": False, "error": "unknown_record"})
                    return
                summary = session.close(message)
                print(f"Stopped PC calibration raw record: {summary}", flush=True)
                self._json_response({"ok": True, "summary": summary})

            def _json_response(self, payload: dict[str, Any], status: int = 200) -> None:
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

        return ThreadingHTTPServer((self.host, self.port), Handler)


def receive(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    robot_manager = None
    if not args.no_flexiv_realsense:
        robot_manager = FlexivRealSenseManager(
            FlexivRealSenseConfig(
                robot_sn=args.flexiv_robot_sn,
                robot_pose_field=args.flexiv_pose_field,
                flexiv_rdk=args.flexiv_rdk,
                flexiv_network_interfaces=args.flexiv_network_interfaces,
                camera_serial=args.realsense_serial,
                width=args.realsense_width,
                height=args.realsense_height,
                fps=args.realsense_fps,
                realsense_auto_exposure=not args.realsense_manual_exposure,
                realsense_exposure=args.realsense_exposure,
                realsense_gain=args.realsense_gain,
                capture_interval_seconds=args.robot_capture_interval,
                run_hand_eye=not args.no_robot_hand_eye,
                controller_translation_scale=args.controller_motion_scale,
                controller_max_offset_m=args.controller_motion_max_offset,
                controller_max_step_m=args.controller_motion_max_step,
            )
        )
    visualizer = None
    if args.visualize:
        visualizer = LiveTelemetryVisualizer(
            args.visualize_host,
            args.visualize_port,
            args.visualize_history,
            robot_manager,
            args.viewer_adb,
            args.calibration_raw_root.resolve(),
            args.calibration_output_root.resolve(),
        )
        visualizer.start(open_browser=not args.no_open_browser)
    calibration_receiver = None
    if not args.no_calibration_http:
        calibration_receiver = PcCalibrationHttpReceiver(
            args.calibration_http_host,
            args.calibration_http_port,
            args.calibration_raw_root.resolve(),
            args.calibration_output_root.resolve(),
            visualizer,
            run_calibration=not args.no_calibrate_after_pc_recording,
            robot_manager=robot_manager,
        )
        calibration_receiver.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.25)
    sock.bind((args.host, args.port))

    print(f"Listening on udp://{args.host}:{args.port}", flush=True)
    print(f"PC recording root: {output_root}", flush=True)

    active: SessionWriter | None = None
    recently_closed_records: dict[str, dict[str, Any]] = {}
    total_messages = 0
    total_samples = 0
    last_datagram_perf = time.perf_counter()

    try:
        while True:
            try:
                data, remote = sock.recvfrom(65535)
            except TimeoutError:
                if args.timeout is not None and time.perf_counter() - last_datagram_perf >= args.timeout:
                    if active is not None:
                        summary = active.close("timeout")
                        print_session_summary(summary, stream=sys.stderr)
                        handle_post_recording(summary, args)
                    if total_messages == 0:
                        print(f"Timed out with no UDP telemetry after {args.timeout:.3f}s.", file=sys.stderr)
                        return 2
                    return 0
                continue

            pc_receive_unix_seconds = time.time()
            pc_receive_perf_counter_seconds = time.perf_counter()
            last_datagram_perf = pc_receive_perf_counter_seconds

            text = data.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            try:
                message = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"[bad-json] {remote[0]}:{remote[1]} {exc}: {text[:200]}", file=sys.stderr)
                continue

            if not isinstance(message, dict):
                print(f"[bad-message] {remote[0]}:{remote[1]} expected JSON object", file=sys.stderr)
                continue

            msg_type = str(message.get("type") or "?")
            record_id = str(message.get("recordId") or "orphan")
            is_sample = msg_type == "sample"
            is_recording_sample = bool(message.get("isRecording", True))
            should_write = (not is_sample) or is_recording_sample or args.record_live_preview
            recently_closed_records = {
                key: value
                for key, value in recently_closed_records.items()
                if pc_receive_perf_counter_seconds - float(value.get("closedPerfCounterSeconds", 0.0))
                <= LATE_RECORDING_SAMPLE_GRACE_SECONDS
            }
            recently_closed = recently_closed_records.get(record_id)
            last_closed_sample_index = (
                recently_closed.get("lastSampleIndex")
                if isinstance(recently_closed, dict)
                else None
            )
            sample_index = message.get("sampleIndex")
            is_late_closed_record_sample = (
                is_sample
                and is_recording_sample
                and active is None
                and recently_closed is not None
                and (
                    not is_number(last_closed_sample_index)
                    or not is_number(sample_index)
                    or int(sample_index) <= int(last_closed_sample_index)
                )
            )
            if is_late_closed_record_sample:
                should_write = False

            wrapper = {
                "pcReceiveUtc": datetime.fromtimestamp(pc_receive_unix_seconds, timezone.utc).isoformat(),
                "pcReceiveUnixSeconds": pc_receive_unix_seconds,
                "pcReceivePerfCounterSeconds": pc_receive_perf_counter_seconds,
                "remote": f"{remote[0]}:{remote[1]}",
                "message": message,
            }

            if is_sample:
                total_samples += 1

            if visualizer is not None:
                visualizer.publish(message, wrapper)

            can_open_session = should_write and (msg_type == "recording_start" or is_sample)
            if can_open_session and (active is None or (
                msg_type == "recording_start"
                and active.record_id != record_id
                and active.messages > 0
            )):
                if active is not None:
                    summary = active.close("superseded_by_new_recording_start")
                    print_session_summary(summary, stream=sys.stderr)
                    if args.single_session:
                        return 0

                active = SessionWriter(
                    output_root,
                    record_id,
                    message if msg_type == "recording_start" else None,
                    remote,
                    pc_receive_unix_seconds,
                    pc_receive_perf_counter_seconds,
                    args.flush_every,
                    args.calibration_output_root.resolve() if not args.no_calibration_http else None,
                )
                print(f"Started PC session: {active.directory}", flush=True)

            if should_write and active is not None:
                active.write(wrapper)
            total_messages += 1

            if is_sample:
                if not args.quiet:
                    left = message.get("leftController")
                    right = message.get("rightController")
                    mode = message.get("telemetryMode") or ("recording" if is_recording_sample else "live_preview")
                    print(
                        f"[sample {message.get('sampleIndex')}] "
                        f"mode={mode} "
                        f"record={record_id if is_recording_sample else 'live'} "
                        f"gaze3D={is_vec3(message.get('gazePoint3DWorld'))} "
                        f"L={controller_status(left)} R={controller_status(right)}",
                        flush=True,
                    )
            elif not args.quiet:
                print(f"[{msg_type}] record={record_id} seq={message.get('sequence')}", flush=True)

            if args.max_samples is not None and total_samples >= args.max_samples:
                if active is not None:
                    summary = active.close("max_samples")
                    print_session_summary(summary, stream=sys.stderr)
                return 0

            if msg_type == "recording_stop":
                if active is not None:
                    closed_record_id = active.record_id
                    summary = active.close("recording_stop")
                    print_session_summary(summary, stream=sys.stderr)
                    handle_post_recording(summary, args)
                    active = None
                    recently_closed_records[closed_record_id] = {
                        "closedPerfCounterSeconds": pc_receive_perf_counter_seconds,
                        "lastSampleIndex": summary.get("lastSampleIndex"),
                    }
                if args.single_session:
                    return 0

    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
        if active is not None:
            summary = active.close("keyboard_interrupt")
            print_session_summary(summary, stream=sys.stderr)
        return 130
    finally:
        sock.close()
        if calibration_receiver is not None:
            calibration_receiver.stop()
        if visualizer is not None:
            visualizer.stop()


def handle_post_recording(summary: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.pull_quest_record:
        return

    pulled_path = pull_quest_record(summary, args)
    if pulled_path is None or args.no_analyze_after_pull:
        return

    output_json = Path(summary["sessionDirectory"]) / "quest_pc_alignment_summary.json"
    analysis_args = argparse.Namespace(
        quest_record=pulled_path,
        pc_session=Path(summary["sessionDirectory"]),
        output_root=args.output_root,
        output_json=output_json,
    )
    analyze(analysis_args)


def pull_quest_record(summary: dict[str, Any], args: argparse.Namespace) -> Path | None:
    quest_directory = summary.get("questOutputDirectory")
    if not isinstance(quest_directory, str) or not quest_directory:
        print(
            "Cannot pull Quest record: telemetry did not include outputDirectory.",
            file=sys.stderr,
            flush=True,
        )
        return None

    record_id = str(summary.get("recordId") or Path(quest_directory.rstrip("/")).name or "quest_record")
    local_root = args.quest_local_root.resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    destination = unique_directory(local_root / sanitize_name(record_id))

    command = [str(args.adb), "pull", quest_directory, str(destination)]
    print(f"Pulling Quest record: {quest_directory} -> {destination}", flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr, flush=True)
        print(f"adb pull failed with exit code {result.returncode}", file=sys.stderr, flush=True)
        return None

    if not (destination / "trajectory.jsonl").exists():
        nested = destination / Path(quest_directory.rstrip("/")).name / "trajectory.jsonl"
        if nested.exists():
            return nested.parent
        print(
            f"Pulled Quest record but trajectory.jsonl was not found under {destination}",
            file=sys.stderr,
            flush=True,
        )
        return destination

    return destination


def quest_adb_status(adb_path: Path | None) -> dict[str, Any]:
    examples = manual_quest_command_examples()
    try:
        adb = resolve_adb_executable(adb_path)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "ready": False,
            "adb": str(adb_path or "adb"),
            "error": str(exc),
            "manualPowerShell": examples,
            "commandPath": QUEST_RECORD_COMMAND_PATH,
        }

    result = subprocess.run([str(adb), "devices"], text=True, capture_output=True, timeout=8)
    devices = parse_adb_devices(result.stdout)
    ready_devices = [device for device in devices if device.get("state") == "device"]
    ready = result.returncode == 0 and len(devices) == 1 and len(ready_devices) == 1
    payload = {
        "ok": ready,
        "ready": ready,
        "adb": str(adb),
        "devices": devices,
        "commandPath": QUEST_RECORD_COMMAND_PATH,
        "manualPowerShell": examples,
    }
    if result.returncode != 0:
        payload["error"] = (result.stderr or result.stdout or f"adb devices failed: {result.returncode}").strip()
    elif not devices:
        payload["error"] = "receiver host cannot see a Quest over adb"
    elif devices and not ready_devices:
        payload["error"] = "adb sees Quest but it is not authorized/online"
    elif len(devices) > 1:
        payload["error"] = "multiple adb devices visible; use a dedicated receiver host or manual command"
    return payload


def send_quest_record_command(adb_path: Path | None, command: str) -> dict[str, Any]:
    status = quest_adb_status(adb_path)
    if not status.get("ready"):
        status["sent"] = False
        return status
    adb = Path(str(status["adb"]))
    mkdir_cmd = f"mkdir -p {sh_quote(posixpath.dirname(QUEST_RECORD_COMMAND_PATH))}"
    write_cmd = f"printf %s {sh_quote(command)} > {sh_quote(QUEST_RECORD_COMMAND_PATH)}"
    result = subprocess.run(
        [str(adb), "shell", f"{mkdir_cmd}; {write_cmd}"],
        text=True,
        capture_output=True,
        timeout=8,
    )
    ok = result.returncode == 0
    return {
        **status,
        "ok": ok,
        "sent": ok,
        "command": command,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "error": "" if ok else (result.stderr or result.stdout or f"adb shell failed: {result.returncode}").strip(),
    }


def resolve_adb_executable(adb_path: Path | None) -> Path:
    candidate = adb_path or Path("adb")
    if candidate.name != str(candidate):
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"adb executable not found: {candidate}")
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not folder:
            continue
        path = Path(folder) / candidate
        if path.exists():
            return path
        exe_path = Path(folder) / f"{candidate}.exe"
        if exe_path.exists():
            return exe_path
    raise FileNotFoundError(f"adb executable not found on PATH: {candidate}")


def parse_adb_devices(stdout: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for line in stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        devices.append({"serial": parts[0], "state": parts[1]})
    return devices


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def manual_quest_command_examples() -> dict[str, str]:
    adb = str(DEFAULT_ADB)
    path = QUEST_RECORD_COMMAND_PATH
    prefix = f"& '{adb}' shell \"mkdir -p {posixpath.dirname(path)}; printf "
    suffix = f" > {path}\""
    return {
        "calib_start": f"{prefix}calib_start{suffix}",
        "calib_stop": f"{prefix}calib_stop{suffix}",
        "calib_toggle": f"{prefix}calib_toggle{suffix}",
    }


def analyze(args: argparse.Namespace) -> int:
    quest_trajectory_path = resolve_quest_trajectory(args.quest_record)
    pc_path = resolve_pc_session(args.pc_session, args.output_root)

    quest_rows = read_jsonl(quest_trajectory_path)
    pc_records = read_pc_records(pc_path)
    pc_rows = pc_records["samples"]

    quest_by_index, quest_duplicates = index_by_sample_index(quest_rows)
    pc_by_index, pc_duplicates = index_by_sample_index(pc_rows)
    matched_indices = sorted(set(quest_by_index) & set(pc_by_index))
    quest_only = sorted(set(quest_by_index) - set(pc_by_index))
    pc_only = sorted(set(pc_by_index) - set(quest_by_index))

    hand_summaries = {
        hand: compare_controller_hand(matched_indices, quest_by_index, pc_by_index, hand)
        for hand in ("left", "right")
    }
    timing_summary = compare_timing(
        matched_indices,
        quest_by_index,
        pc_by_index,
        pc_records.get("recording_start_pc_perf"),
    )
    four_point_summary = compare_four_point_alignment(matched_indices, quest_by_index, pc_by_index)

    summary = {
        "questTrajectory": str(quest_trajectory_path),
        "pcSession": str(pc_path),
        "questSamples": len(quest_rows),
        "pcSamples": len(pc_rows),
        "matchedSamples": len(matched_indices),
        "questDuplicateSampleIndices": quest_duplicates,
        "pcDuplicateSampleIndices": pc_duplicates,
        "questOnlySampleCount": len(quest_only),
        "pcOnlySampleCount": len(pc_only),
        "firstQuestOnlySampleIndices": quest_only[:20],
        "firstPcOnlySampleIndices": pc_only[:20],
        "controllerAlignment": hand_summaries,
        "fourPointAlignment": four_point_summary,
        "timing": timing_summary,
        "notes": [
            (
                "Controller pose error compares Quest-local trajectory rows with the UDP payload "
                "received by the PC for the same sampleIndex."
            ),
            (
                "Four-point alignment compares left/right controller positions plus left/right eye "
                "positions between Quest-local trajectory rows and PC-received UDP payloads."
            ),
            (
                "receiveMinusQuestTimelineMs is relative to the PC receive time of recording_start "
                "when available, otherwise relative to the first matched sample. It measures jitter "
                "and phase drift on the PC timeline, not absolute one-way network latency."
            ),
        ],
    }

    print_analysis_summary(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote analysis JSON: {args.output_json}", flush=True)

    return 0 if matched_indices else 4


def diagnose_gaze_depth(args: argparse.Namespace) -> int:
    pc_session = args.pc_session.resolve()
    if not pc_session.exists():
        raise FileNotFoundError(f"PC session not found: {pc_session}")
    pc_samples_path = pc_session / "pc_samples.jsonl" if pc_session.is_dir() else pc_session
    if not pc_samples_path.exists():
        raise FileNotFoundError(f"PC samples not found: {pc_samples_path}")
    snapshot_path = pc_session / "pc_calibration_snapshot.json" if pc_session.is_dir() else pc_session.parent / "pc_calibration_snapshot.json"
    snapshot = read_json_if_exists(snapshot_path)
    if not isinstance(snapshot, dict):
        summary = read_json_if_exists(pc_session / "pc_session_summary.json") if pc_session.is_dir() else None
        snapshot = summary.get("calibrationSnapshot") if isinstance(summary, dict) else None
    if not isinstance(snapshot, dict):
        raise FileNotFoundError(f"Calibration snapshot not found for {pc_session}")

    pc_rows = read_jsonl(pc_samples_path)
    diagnostics = build_gaze_depth_diagnostics(pc_rows, snapshot)
    summary: dict[str, Any] = {
        "pcSession": str(pc_session),
        "pcSamples": len(pc_rows),
        "calibrationRecordId": snapshot.get("recordId"),
        "gazeDepth": diagnostics.get("summary", {}),
        "questComparison": None,
        "notes": [
            (
                "rawDepth is the scalar distance from gazeRayOrigin to gazePoint3DWorld along "
                "gazeRayDirection."
            ),
            (
                "boardDistance is gazePoint3DWorld transformed by T_board_world; board z should "
                "be near 0 when the gaze point lies on the checkerboard plane."
            ),
            (
                "filteredDepth uses a causal 7-sample median on rawDepth only; it is a diagnostic "
                "preview, not a modification of recorded data."
            ),
        ],
    }

    if args.quest_record is not None:
        quest_trajectory_path = resolve_quest_trajectory(args.quest_record)
        quest_rows = read_jsonl(quest_trajectory_path)
        quest_by_index, quest_duplicates = index_by_sample_index(quest_rows)
        pc_by_index, pc_duplicates = index_by_sample_index(pc_rows)
        matched_indices = sorted(set(quest_by_index) & set(pc_by_index))
        fields = [
            "gazePoint3DWorld",
            "gazeRayOrigin",
            "gazeRayDirection",
            "leftEyePose",
            "rightEyePose",
        ]
        field_errors: dict[str, Any] = {}
        for field in fields:
            errors: list[float] = []
            for index in matched_indices:
                left_value = quest_by_index[index].get(field)
                right_value = pc_by_index[index].get(field)
                if not isinstance(left_value, list) or not isinstance(right_value, list):
                    continue
                count = min(len(left_value), len(right_value))
                if count == 0:
                    continue
                try:
                    errors.append(
                        max(abs(float(left_value[i]) - float(right_value[i])) for i in range(count))
                    )
                except (TypeError, ValueError):
                    continue
            field_errors[field] = stats_summary(errors)
        summary["questComparison"] = {
            "questTrajectory": str(quest_trajectory_path),
            "questSamples": len(quest_rows),
            "matchedSamples": len(matched_indices),
            "questOnlySampleCount": len(set(quest_by_index) - set(pc_by_index)),
            "pcOnlySampleCount": len(set(pc_by_index) - set(quest_by_index)),
            "firstQuestOnlySampleIndices": sorted(set(quest_by_index) - set(pc_by_index))[:20],
            "firstPcOnlySampleIndices": sorted(set(pc_by_index) - set(quest_by_index))[:20],
            "questDuplicateSampleIndices": quest_duplicates,
            "pcDuplicateSampleIndices": pc_duplicates,
            "maxAbsFieldErrors": field_errors,
        }

    print_gaze_depth_diagnostics(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote gaze-depth diagnostic JSON: {args.output_json}", flush=True)

    return 0


def resolve_quest_trajectory(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        path = path / "trajectory.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Quest trajectory not found: {path}")
    return path


def resolve_pc_session(path: Path | None, output_root: Path) -> Path:
    if path is None:
        output_root = output_root.resolve()
        candidates = [item for item in output_root.iterdir() if item.is_dir()] if output_root.exists() else []
        if not candidates:
            raise FileNotFoundError(f"No PC sessions found under {output_root}")
        return max(candidates, key=lambda item: item.stat().st_mtime)

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"PC session path not found: {path}")
    return path


def read_pc_records(path: Path) -> dict[str, Any]:
    if path.is_dir():
        raw_path = path / "pc_telemetry_raw.jsonl"
        samples_path = path / "pc_samples.jsonl"
        if raw_path.exists():
            path = raw_path
        elif samples_path.exists():
            path = samples_path
        else:
            raise FileNotFoundError(f"No pc_telemetry_raw.jsonl or pc_samples.jsonl in {path}")

    samples: list[dict[str, Any]] = []
    recording_start_pc_perf: float | None = None
    for row in read_jsonl(path):
        message = row.get("message") if isinstance(row.get("message"), dict) else row
        if not isinstance(message, dict):
            continue

        pc_receive_perf = row.get("pcReceivePerfCounterSeconds")
        pc_receive_unix = row.get("pcReceiveUnixSeconds")
        pc_receive_utc = row.get("pcReceiveUtc")
        remote = row.get("remote")

        if message.get("type") == "recording_start" and is_number(pc_receive_perf):
            recording_start_pc_perf = float(pc_receive_perf)
            continue

        is_sample = message.get("type") == "sample" or "sampleIndex" in message
        if not is_sample:
            continue

        sample = dict(message)
        if pc_receive_perf is not None:
            sample["pcReceivePerfCounterSeconds"] = pc_receive_perf
        if pc_receive_unix is not None:
            sample["pcReceiveUnixSeconds"] = pc_receive_unix
        if pc_receive_utc is not None:
            sample["pcReceiveUtc"] = pc_receive_utc
        if remote is not None:
            sample["remote"] = remote
        samples.append(sample)

    return {
        "samples": samples,
        "recording_start_pc_perf": recording_start_pc_perf,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def index_by_sample_index(rows: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], int]:
    result: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        sample_index = row.get("sampleIndex")
        if not isinstance(sample_index, int):
            continue
        if sample_index in result:
            duplicates += 1
        result[sample_index] = row
    return result, duplicates


def compare_controller_hand(
    matched_indices: list[int],
    quest_by_index: dict[int, dict[str, Any]],
    pc_by_index: dict[int, dict[str, Any]],
    hand: str,
) -> dict[str, Any]:
    quest_pose_count = 0
    pc_pose_count = 0
    both_pose_count = 0
    position_errors_m: list[float] = []
    angle_errors_deg: list[float] = []
    source_mismatches = 0
    missing_reason_mismatches = 0
    first_mismatches: list[dict[str, Any]] = []

    for sample_index in matched_indices:
        quest_controller = controller_dict(quest_by_index[sample_index].get(f"{hand}Controller"))
        pc_controller = controller_dict(pc_by_index[sample_index].get(f"{hand}Controller"))
        quest_has_pose = has_controller_pose(quest_controller)
        pc_has_pose = has_controller_pose(pc_controller)
        if quest_has_pose:
            quest_pose_count += 1
        if pc_has_pose:
            pc_pose_count += 1
        if quest_has_pose and pc_has_pose:
            both_pose_count += 1
            quest_position = vec3(quest_controller.get("position"))
            pc_position = vec3(pc_controller.get("position"))
            if quest_position is not None and pc_position is not None:
                position_errors_m.append(distance(quest_position, pc_position))

            quest_rotation = quat(quest_controller.get("rotation"))
            pc_rotation = quat(pc_controller.get("rotation"))
            if quest_rotation is not None and pc_rotation is not None:
                angle_errors_deg.append(quaternion_angle_degrees(quest_rotation, pc_rotation))

        if quest_controller.get("source") != pc_controller.get("source"):
            source_mismatches += 1
            if len(first_mismatches) < 5:
                first_mismatches.append(
                    {
                        "sampleIndex": sample_index,
                        "kind": "source",
                        "quest": quest_controller.get("source"),
                        "pc": pc_controller.get("source"),
                    }
                )

        if quest_controller.get("missingReason") != pc_controller.get("missingReason"):
            missing_reason_mismatches += 1
            if len(first_mismatches) < 5:
                first_mismatches.append(
                    {
                        "sampleIndex": sample_index,
                        "kind": "missingReason",
                        "quest": quest_controller.get("missingReason"),
                        "pc": pc_controller.get("missingReason"),
                    }
                )

    return {
        "matchedSamples": len(matched_indices),
        "questPoseSamples": quest_pose_count,
        "pcPoseSamples": pc_pose_count,
        "bothPoseSamples": both_pose_count,
        "positionErrorMeters": describe(position_errors_m),
        "positionErrorMillimeters": describe([value * 1000.0 for value in position_errors_m]),
        "rotationErrorDegrees": describe(angle_errors_deg),
        "sourceMismatches": source_mismatches,
        "missingReasonMismatches": missing_reason_mismatches,
        "firstMismatches": first_mismatches,
    }


def compare_four_point_alignment(
    matched_indices: list[int],
    quest_by_index: dict[int, dict[str, Any]],
    pc_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    point_specs = [
        ("leftController", lambda row: controller_pose_position(row.get("leftController"))),
        ("rightController", lambda row: controller_pose_position(row.get("rightController"))),
        ("leftEye", lambda row: pose_position(row.get("leftEyePose")) or vec3(row.get("leftEyePosition"))),
        ("rightEye", lambda row: pose_position(row.get("rightEyePose")) or vec3(row.get("rightEyePosition"))),
    ]
    per_point: dict[str, dict[str, Any]] = {}
    complete_samples = 0
    complete_rms_errors_m: list[float] = []

    for name, _ in point_specs:
        per_point[name] = {
            "questSamples": 0,
            "pcSamples": 0,
            "bothSamples": 0,
            "positionErrorMeters": [],
        }

    for sample_index in matched_indices:
        sample_errors: list[float] = []
        sample_complete = True
        quest_row = quest_by_index[sample_index]
        pc_row = pc_by_index[sample_index]

        for name, getter in point_specs:
            quest_position = getter(quest_row)
            pc_position = getter(pc_row)
            stats = per_point[name]
            if quest_position is not None:
                stats["questSamples"] += 1
            if pc_position is not None:
                stats["pcSamples"] += 1
            if quest_position is not None and pc_position is not None:
                stats["bothSamples"] += 1
                error = distance(quest_position, pc_position)
                stats["positionErrorMeters"].append(error)
                sample_errors.append(error)
            else:
                sample_complete = False

        if sample_complete and len(sample_errors) == len(point_specs):
            complete_samples += 1
            complete_rms_errors_m.append(math.sqrt(sum(value * value for value in sample_errors) / len(sample_errors)))

    summarized_points = {}
    for name, stats in per_point.items():
        errors_m = stats.pop("positionErrorMeters")
        summarized_points[name] = {
            **stats,
            "positionErrorMeters": describe(errors_m),
            "positionErrorMillimeters": describe([value * 1000.0 for value in errors_m]),
        }

    return {
        "matchedSamples": len(matched_indices),
        "completeFourPointSamples": complete_samples,
        "completeFourPointRmsErrorMeters": describe(complete_rms_errors_m),
        "completeFourPointRmsErrorMillimeters": describe([value * 1000.0 for value in complete_rms_errors_m]),
        "points": summarized_points,
    }


def compare_timing(
    matched_indices: list[int],
    quest_by_index: dict[int, dict[str, Any]],
    pc_by_index: dict[int, dict[str, Any]],
    recording_start_pc_perf: float | None,
) -> dict[str, Any]:
    rows: list[tuple[int, float, float]] = []
    for sample_index in matched_indices:
        quest_ts = quest_by_index[sample_index].get("recordingTimestampSeconds")
        pc_perf = pc_by_index[sample_index].get("pcReceivePerfCounterSeconds")
        if is_number(quest_ts) and is_number(pc_perf):
            rows.append((sample_index, float(quest_ts), float(pc_perf)))

    if not rows:
        return {
            "matchedTimedSamples": 0,
            "receiveMinusQuestTimelineMs": None,
            "pcInterarrivalMs": None,
            "questSampleIntervalMs": None,
            "interarrivalMinusQuestIntervalMs": None,
            "reference": "none",
        }

    rows.sort(key=lambda item: item[0])
    if recording_start_pc_perf is not None:
        pc_ref = recording_start_pc_perf
        quest_ref = 0.0
        reference = "recording_start_pc_receive_time"
    else:
        _, quest_ref, pc_ref = rows[0]
        reference = "first_matched_sample"

    receive_minus_quest_ms = [
        ((pc_perf - pc_ref) - (quest_ts - quest_ref)) * 1000.0
        for _, quest_ts, pc_perf in rows
    ]

    pc_interarrival_ms: list[float] = []
    quest_interval_ms: list[float] = []
    interval_diff_ms: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        _, previous_quest_ts, previous_pc_perf = previous
        _, current_quest_ts, current_pc_perf = current
        pc_dt = (current_pc_perf - previous_pc_perf) * 1000.0
        quest_dt = (current_quest_ts - previous_quest_ts) * 1000.0
        pc_interarrival_ms.append(pc_dt)
        quest_interval_ms.append(quest_dt)
        interval_diff_ms.append(pc_dt - quest_dt)

    return {
        "matchedTimedSamples": len(rows),
        "reference": reference,
        "receiveMinusQuestTimelineMs": describe(receive_minus_quest_ms),
        "pcInterarrivalMs": describe(pc_interarrival_ms),
        "questSampleIntervalMs": describe(quest_interval_ms),
        "interarrivalMinusQuestIntervalMs": describe(interval_diff_ms),
        "firstTimedSampleIndex": rows[0][0],
        "lastTimedSampleIndex": rows[-1][0],
    }


def describe(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None

    sorted_values = sorted(values)
    count = len(sorted_values)
    return {
        "count": count,
        "min": sorted_values[0],
        "mean": sum(sorted_values) / count,
        "median": percentile(sorted_values, 50.0),
        "p90": percentile(sorted_values, 90.0),
        "p95": percentile(sorted_values, 95.0),
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[int(position)]

    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    fraction = position - lower_index
    return lower * (1.0 - fraction) + upper * fraction


def print_session_summary(summary: dict[str, Any], *, stream: Any) -> None:
    print(
        "PC session closed: "
        f"dir={summary['sessionDirectory']} "
        f"messages={summary['messages']} samples={summary['samples']} "
        f"leftPose={summary['leftPoseSamples']} rightPose={summary['rightPoseSamples']} "
        f"reason={summary['closedReason']}",
        file=stream,
        flush=True,
    )
    if summary["leftMissingReasons"] or summary["rightMissingReasons"]:
        print(
            f"  missing: left={summary['leftMissingReasons']} right={summary['rightMissingReasons']}",
            file=stream,
            flush=True,
        )


def print_analysis_summary(summary: dict[str, Any]) -> None:
    print("Quest/PC controller alignment")
    print(f"  Quest trajectory: {summary['questTrajectory']}")
    print(f"  PC session:       {summary['pcSession']}")
    print(
        "  samples: "
        f"quest={summary['questSamples']} pc={summary['pcSamples']} "
        f"matched={summary['matchedSamples']} "
        f"questOnly={summary['questOnlySampleCount']} pcOnly={summary['pcOnlySampleCount']}"
    )
    for hand in ("left", "right"):
        hand_summary = summary["controllerAlignment"][hand]
        pos_mm = hand_summary["positionErrorMillimeters"]
        rot_deg = hand_summary["rotationErrorDegrees"]
        print(
            f"  {hand}: questPose={hand_summary['questPoseSamples']} "
            f"pcPose={hand_summary['pcPoseSamples']} bothPose={hand_summary['bothPoseSamples']}"
        )
        print(f"    position error mm: {format_stats(pos_mm)}")
        print(f"    rotation error deg: {format_stats(rot_deg)}")
        if hand_summary["sourceMismatches"] or hand_summary["missingReasonMismatches"]:
            print(
                "    mismatches: "
                f"source={hand_summary['sourceMismatches']} "
                f"missingReason={hand_summary['missingReasonMismatches']} "
                f"examples={hand_summary['firstMismatches']}"
            )

    four_point = summary["fourPointAlignment"]
    print(
        "  four-point: "
        f"completeSamples={four_point['completeFourPointSamples']} "
        f"rmsErrorMm={format_stats(four_point['completeFourPointRmsErrorMillimeters'])}"
    )
    for name, point_summary in four_point["points"].items():
        print(
            f"    {name}: quest={point_summary['questSamples']} "
            f"pc={point_summary['pcSamples']} both={point_summary['bothSamples']} "
            f"error mm={format_stats(point_summary['positionErrorMillimeters'])}"
        )

    timing = summary["timing"]
    print(f"  timing reference: {timing['reference']}")
    print(f"    receive - Quest timeline ms: {format_stats(timing['receiveMinusQuestTimelineMs'])}")
    print(f"    PC interarrival ms:          {format_stats(timing['pcInterarrivalMs'])}")
    print(f"    Quest sample interval ms:    {format_stats(timing['questSampleIntervalMs'])}")
    print(f"    interval diff ms:            {format_stats(timing['interarrivalMinusQuestIntervalMs'])}")


def print_gaze_depth_diagnostics(summary: dict[str, Any]) -> None:
    gaze = summary.get("gazeDepth") if isinstance(summary.get("gazeDepth"), dict) else {}
    print("Gaze depth diagnostic", flush=True)
    print(f"  pc session: {summary.get('pcSession')}", flush=True)
    print(f"  calibration: {summary.get('calibrationRecordId') or 'n/a'}", flush=True)
    print(f"  samples: {summary.get('pcSamples')} valid gaze depths: {gaze.get('validGazeDepthCount')}", flush=True)
    print(f"  sources: {json.dumps(gaze.get('sourceCounts') or {}, ensure_ascii=False)}", flush=True)
    print(f"  raw abs board distance mm:      {format_stats(gaze.get('absBoardDistanceMm'))}", flush=True)
    print(f"  raw abs depth-plane error mm:   {format_stats(gaze.get('absDepthMinusBoardPlaneMm'))}", flush=True)
    print(f"  filtered abs depth-plane mm:    {format_stats(gaze.get('absFilteredDepthMinusBoardPlaneMm'))}", flush=True)
    print(f"  raw frame abs depth delta mm:   {format_stats(gaze.get('absRawDepthDeltaMm'))}", flush=True)
    print(f"  filt frame abs depth delta mm:  {format_stats(gaze.get('absFilteredDepthDeltaMm'))}", flush=True)
    print(f"  spike counts: {json.dumps(gaze.get('rawDepthSpikeCounts') or {}, ensure_ascii=False)}", flush=True)

    spikes = gaze.get("topRawDepthSpikes") if isinstance(gaze.get("topRawDepthSpikes"), list) else []
    if spikes:
        print("  top raw-depth spikes:", flush=True)
        for spike in spikes[:10]:
            raw_delta = spike.get("rawDepthDeltaM")
            board_delta = spike.get("boardDistanceDeltaM")
            timestamp = spike.get("t")
            prefix = f"    {spike.get('from')}->{spike.get('to')} "
            if is_number(timestamp):
                prefix += f"t={float(timestamp):.3f}s "
            print(
                prefix +
                f"raw={float(raw_delta) * 1000.0:+.1f}mm "
                f"boardZ={float(board_delta) * 1000.0:+.1f}mm",
                flush=True,
            )

    quest = summary.get("questComparison")
    if isinstance(quest, dict):
        print("  Quest-vs-PC:", flush=True)
        print(
            f"    quest samples={quest.get('questSamples')} matched={quest.get('matchedSamples')} "
            f"questOnly={quest.get('questOnlySampleCount')} pcOnly={quest.get('pcOnlySampleCount')}",
            flush=True,
        )
        errors = quest.get("maxAbsFieldErrors") if isinstance(quest.get("maxAbsFieldErrors"), dict) else {}
        for field, stats in errors.items():
            print(f"    {field}: {format_stats(stats)}", flush=True)


def visualizer_event_from_message(message: dict[str, Any], wrapper: dict[str, Any]) -> dict[str, Any] | None:
    msg_type = message.get("type")
    if msg_type == "sample":
        return visualizer_event_from_sample(message, wrapper)
    if msg_type in ("recording_start", "recording_stop"):
        is_recording = msg_type == "recording_start"
        return {
            "type": "status",
            "recordId": message.get("recordId"),
            "sequence": message.get("sequence"),
            "isRecording": is_recording,
            "telemetryMode": message.get("telemetryMode") or ("recording" if is_recording else "live_preview"),
            "recordingTimestampSeconds": message.get("recordingTimestampSeconds"),
            "pcReceivePerfCounterSeconds": wrapper.get("pcReceivePerfCounterSeconds"),
            "pcReceiveUtc": wrapper.get("pcReceiveUtc"),
        }
    return None


def recording_replay_list(
    recording_root: Path,
    *,
    calibration_raw_root: Path | None = None,
    calibration_output_root: Path | None = None,
) -> dict[str, Any]:
    root = recording_root.resolve()
    records: list[dict[str, Any]] = []
    roots = [{"source": "pc", "root": root, "label": "PC UDP"}]
    if calibration_raw_root is not None:
        raw_root = calibration_raw_root.resolve()
        if raw_root != root:
            roots.append({"source": "raw", "root": raw_root, "label": "B calibration"})

    sortable_records: list[tuple[tuple[int, int, int, float], dict[str, Any]]] = []
    for root_info in roots:
        current_root = root_info["root"]
        if not isinstance(current_root, Path) or not current_root.exists():
            continue
        directories = [item for item in current_root.iterdir() if item.is_dir()]
        for directory in directories:
            record = recording_replay_list_record(
                directory,
                str(root_info["source"]),
                str(root_info["label"]),
                calibration_output_root,
            )
            if record is None:
                continue
            sample_count = int(record["samples"]) if is_number(record.get("samples")) else 0
            has_calibration = bool(record.get("hasCalibrationSnapshot"))
            sortable_records.append(((1 if has_calibration else 0, sample_count, directory.stat().st_mtime), record))
    sortable_records.sort(key=lambda item: item[0], reverse=True)
    records = [record for _, record in sortable_records]
    return {
        "ok": True,
        "root": str(root),
        "rawRoot": str(calibration_raw_root.resolve()) if calibration_raw_root is not None else None,
        "records": records,
    }


def recording_replay_list_record(
    directory: Path,
    source: str,
    source_label: str,
    calibration_output_root: Path | None,
) -> dict[str, Any] | None:
    samples_path = replay_samples_path(directory, source)
    if samples_path is None or not samples_path.exists() or samples_path.stat().st_size <= 0:
        return None
    summary = replay_summary(directory, source)
    if not isinstance(summary, dict) or not summary:
        return None
    snapshot = replay_snapshot(directory, summary, source, calibration_output_root)
    calibration_failure = replay_calibration_failure(directory, summary, source, calibration_output_root)
    sample_count = None
    if is_number(summary.get("samples")):
        sample_count = int(summary["samples"])
    elif is_number(summary.get("trajectorySampleCount")):
        sample_count = int(summary["trajectorySampleCount"])
    has_calibration = isinstance(snapshot, dict) and bool(snapshot.get("T_world_board"))
    calibration_status = "ok" if has_calibration else ("failed" if calibration_failure else "missing")
    robot_summary = robot_realsense_record_summary(directory)
    return {
        "recordId": directory.name,
        "source": source,
        "sourceLabel": source_label,
        "path": str(directory),
        "mtime": directory.stat().st_mtime,
        "samples": sample_count,
        "hasCalibrationSnapshot": has_calibration,
        "calibrationStatus": calibration_status,
        "calibrationReason": calibration_failure.get("reason") if isinstance(calibration_failure, dict) else None,
        "calibrationReasonCode": calibration_failure.get("reasonCode") if isinstance(calibration_failure, dict) else None,
        "calibrationRecordId": snapshot.get("recordId") if isinstance(snapshot, dict) else None,
        "startUtc": summary.get("startUtc") or summary.get("startTimeUtc"),
        "closedReason": summary.get("closedReason"),
        "robotSummary": robot_summary,
    }


def robot_realsense_record_summary(session_dir: Path) -> dict[str, Any] | None:
    robot_dir = session_dir / "robot_realsense"
    if not robot_dir.exists():
        return None
    session = read_json_if_exists(robot_dir / "session_summary.json")
    config = read_json_if_exists(robot_dir / "capture_config.json")
    result = read_json_if_exists(robot_dir / "robot_hand_eye_result.json")
    failure = read_json_if_exists(robot_dir / "robot_hand_eye_failure.json")
    if not isinstance(session, dict):
        session = {}
    if not isinstance(config, dict):
        config = {}
    status = "pending"
    if isinstance(result, dict) and result.get("ok"):
        status = "ok"
    elif isinstance(failure, dict):
        status = "failed"
    elif session:
        status = "recorded"
    if isinstance(result, dict):
        counts = result.get("counts")
        diversity = result.get("diversity")
    elif isinstance(failure, dict):
        counts = failure.get("counts")
        diversity = failure.get("diversity")
    else:
        counts = {}
        diversity = session.get("poseDiversity")
    last_motion = session.get("lastMotion") if isinstance(session.get("lastMotion"), dict) else {}
    residual = None
    if isinstance(result, dict):
        residual = (
            result.get("end_camera", {})
            .get("residuals", {})
            .get("translation_mm", {})
            if isinstance(result.get("end_camera"), dict)
            else None
        )
    return {
        "status": status,
        "samples": session.get("samples"),
        "images": session.get("images"),
        "motionCommands": session.get("motionCommands"),
        "motionSkips": session.get("motionSkips"),
        "motionErrors": session.get("motionErrors"),
        "lastMotionReason": last_motion.get("reason") or last_motion.get("error"),
        "detections": counts.get("detections") if isinstance(counts, dict) else None,
        "requiredDetections": config.get("minHandEyeDetections"),
        "translationSpanM": diversity.get("eeTranslationSpanM") if isinstance(diversity, dict) else None,
        "rotationSpanDeg": diversity.get("eeRotationSpanDeg") if isinstance(diversity, dict) else None,
        "residualMedianMm": residual.get("median") if isinstance(residual, dict) else None,
        "failureReason": failure.get("error") if isinstance(failure, dict) else None,
    }


def replay_samples_path(session_dir: Path, source: str) -> Path | None:
    if source == "raw":
        return session_dir / "trajectory.jsonl"
    return session_dir / "pc_samples.jsonl"


def replay_summary(session_dir: Path, source: str) -> dict[str, Any]:
    if source == "raw":
        summary = read_json_if_exists(session_dir / "pc_calibration_session_summary.json")
        if isinstance(summary, dict) and summary:
            return summary
        metadata = read_json_if_exists(session_dir / "quest_camera_metadata.json")
        return metadata if isinstance(metadata, dict) else {}
    summary = read_json_if_exists(session_dir / "pc_session_summary.json")
    return summary if isinstance(summary, dict) else {}


def replay_snapshot(
    session_dir: Path,
    summary: dict[str, Any],
    source: str,
    calibration_output_root: Path | None,
) -> dict[str, Any]:
    snapshot = read_json_if_exists(session_dir / "pc_calibration_snapshot.json")
    if isinstance(snapshot, dict):
        return snapshot
    embedded = summary.get("calibrationSnapshot")
    if isinstance(embedded, dict):
        return embedded
    output_dir = summary.get("outputDirectory")
    result_path = Path(output_dir) / "calibration_result_25mm.json" if isinstance(output_dir, str) and output_dir else None
    if result_path is not None and result_path.exists():
        result = read_json_if_exists(result_path)
        if isinstance(result, dict):
            return recording_snapshot_from_calibration_result(session_dir.name, result, result_path)
    if source == "raw" and calibration_output_root is not None:
        candidate = calibration_output_root.resolve() / session_dir.name / "calibration_result_25mm.json"
        if candidate.exists():
            result = read_json_if_exists(candidate)
            if isinstance(result, dict):
                return recording_snapshot_from_calibration_result(session_dir.name, result, candidate)
    return {}


def replay_calibration_failure(
    session_dir: Path,
    summary: dict[str, Any],
    source: str,
    calibration_output_root: Path | None,
) -> dict[str, Any] | None:
    output_dir = summary.get("outputDirectory")
    candidate_dirs: list[Path] = []
    if isinstance(output_dir, str) and output_dir:
        candidate_dirs.append(Path(output_dir))
    if source == "raw" and calibration_output_root is not None:
        candidate_dirs.append(calibration_output_root.resolve() / session_dir.name)
    for output_directory in candidate_dirs:
        failure_path = output_directory / "calibration_failure_25mm.json"
        log_path = output_directory / "calibration_run.log"
        if failure_path.exists() or log_path.exists():
            return compact_calibration_failure(session_dir.name, output_directory, failure_path, log_path)
    return None


def compact_calibration_failure(record_id: str, output_directory: Path, failure_path: Path, log_path: Path) -> dict[str, Any]:
    diagnostics = read_json_if_exists(failure_path)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    detection_summary = read_json_if_exists(output_directory / "checkerboard_detection_summary_25mm.json")
    return {
        "type": "calibration_failure",
        "recordId": record_id,
        "rawRecordName": record_id,
        "outputDirectory": str(output_directory),
        "failurePath": str(failure_path) if failure_path.exists() else None,
        "logPath": str(log_path) if log_path.exists() else None,
        "reason": diagnostics.get("reason") or diagnostics.get("reason_code") or "Calibration failed",
        "reasonCode": diagnostics.get("reason_code"),
        "phase": diagnostics.get("phase"),
        "diagnostics": diagnostics,
        "detectionSummary": detection_summary,
    }


def recording_snapshot_from_calibration_result(record_id: str, result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    event = calibration_result_event(record_id, result, result_path)
    return {
        "capturedUtc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "kind": "result",
        "recordId": event.get("recordId"),
        "resultPath": event.get("resultPath"),
        "pattern": event.get("pattern"),
        "squareSizeM": event.get("squareSizeM"),
        "imageYAxis": event.get("imageYAxis"),
        "T_world_board": event.get("T_world_board"),
        "T_board_world": event.get("T_board_world"),
        "questWorldOriginInBoardM": event.get("questWorldOriginInBoardM"),
        "boardNormalWorld": event.get("boardNormalWorld"),
        "boardNormalAbsAngleToWorldYDeg": event.get("boardNormalAbsAngleToWorldYDeg"),
        "bestLagSeconds": event.get("bestLagSeconds"),
        "keptFrames": event.get("keptFrames"),
        "inputFrames": event.get("inputFrames"),
        "medianReprojectionPx": event.get("medianReprojectionPx"),
        "p90ReprojectionPx": event.get("p90ReprojectionPx"),
    }


def replay_session_dir_for_record(
    recording_root: Path,
    record_id: str,
    source: str | None = None,
    calibration_raw_root: Path | None = None,
) -> tuple[Path, str]:
    safe_id = sanitize_name(record_id)
    if safe_id != record_id:
        raise ValueError("invalid recordId")
    candidates: list[tuple[str, Path]] = []
    if source in (None, "", "pc"):
        candidates.append(("pc", recording_root.resolve()))
    if calibration_raw_root is not None and source in (None, "", "raw"):
        candidates.append(("raw", calibration_raw_root.resolve()))
    if source not in (None, "", "pc", "raw"):
        raise ValueError(f"unsupported recording source: {source}")
    for candidate_source, root in candidates:
        session_dir = (root / safe_id).resolve()
        try:
            session_dir.relative_to(root)
        except ValueError as exc:
            raise ValueError("recordId is outside recording root") from exc
        samples_path = replay_samples_path(session_dir, candidate_source)
        if session_dir.exists() and session_dir.is_dir() and samples_path is not None and samples_path.exists():
            return session_dir, candidate_source
    raise FileNotFoundError(f"recording not found: {safe_id}")


def build_recording_replay_payload(
    recording_root: Path,
    record_id: str,
    *,
    source: str | None = None,
    calibration_raw_root: Path | None = None,
    calibration_output_root: Path | None = None,
) -> dict[str, Any]:
    safe_id = sanitize_name(record_id)
    if safe_id != record_id:
        raise ValueError("invalid recordId")
    session_dir, resolved_source = replay_session_dir_for_record(
        recording_root,
        safe_id,
        source,
        calibration_raw_root,
    )
    samples_path = replay_samples_path(session_dir, resolved_source)
    if samples_path is None or not samples_path.exists():
        raise FileNotFoundError(f"missing samples file for {safe_id}")
    summary = replay_summary(session_dir, resolved_source)
    if not isinstance(summary, dict) or not summary:
        raise FileNotFoundError(f"recording summary not found for {safe_id}")
    snapshot = replay_snapshot(session_dir, summary, resolved_source, calibration_output_root)
    calibration_failure = replay_calibration_failure(session_dir, summary, resolved_source, calibration_output_root)

    board_matrix_world = matrix_from_snapshot(snapshot)
    board_origin_world = matrix_translation(board_matrix_world) if board_matrix_world is not None else [0.0, 0.0, 0.0]
    board_matrix_display = translated_matrix_4x4(board_matrix_world, board_origin_world) if board_matrix_world is not None else None

    raw_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            raw_rows.append(row)
            samples.append(recording_replay_sample(row, board_origin_world))

    gaze_diagnostics = build_gaze_depth_diagnostics(raw_rows, snapshot)
    enrich_replay_samples_with_gaze_diagnostics(samples, gaze_diagnostics, board_origin_world)
    robot_realsense = build_robot_realsense_replay(session_dir, board_origin_world)
    raw_artifacts = replay_raw_artifacts(session_dir, summary, resolved_source, calibration_output_root)

    return {
        "ok": True,
        "recordId": safe_id,
        "source": resolved_source,
        "sessionDir": str(session_dir),
        "summary": summary if isinstance(summary, dict) else {},
        "snapshot": snapshot,
        "calibrationFailure": calibration_failure,
        "coordinateMode": "quest_world_axes_translated_to_board_origin",
        "boardOriginWorld": board_origin_world,
        "boardMatrix": board_matrix_display,
        "gazeDepthDiagnostics": gaze_diagnostics.get("summary", {}),
        "robotRealSense": robot_realsense,
        "rawArtifacts": raw_artifacts,
        "samples": samples,
    }


def replay_raw_artifacts(
    session_dir: Path,
    summary: dict[str, Any],
    source: str,
    calibration_output_root: Path | None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if source == "raw":
        for key, filename, label in (
            ("metadata", "quest_camera_metadata.json", "Quest metadata"),
            ("leftFrames", "left_frames.jsonl", "Left frame metadata"),
            ("rightFrames", "right_frames.jsonl", "Right frame metadata"),
            ("trajectory", "trajectory.jsonl", "Quest trajectory"),
        ):
            path = session_dir / filename
            if path.exists():
                artifacts[key] = artifact_payload(path, label)
    output_dir = summary.get("outputDirectory")
    output_paths: list[Path] = []
    if isinstance(output_dir, str) and output_dir:
        output_paths.append(Path(output_dir))
    if source == "raw" and calibration_output_root is not None:
        output_paths.append(calibration_output_root.resolve() / session_dir.name)
    for root in output_paths:
        if not root.exists():
            continue
        for key, filename in (
            ("calibrationFailure", "calibration_failure_25mm.json"),
            ("calibrationResult", "calibration_result_25mm.json"),
            ("detectionSummary", "checkerboard_detection_summary_25mm.json"),
            ("calibrationLog", "calibration_run.log"),
        ):
            path = root / filename
            if path.exists() and key not in artifacts:
                artifacts[key] = artifact_payload(path, artifact_label(key))
    return artifacts


def artifact_label(key: str) -> str:
    return {
        "calibrationFailure": "Quest calibration failure",
        "calibrationResult": "Quest calibration result",
        "detectionSummary": "Checkerboard detections",
        "calibrationLog": "Calibration log",
    }.get(key, key)


def artifact_payload(path: Path, label: str | None = None) -> dict[str, Any]:
    try:
        resolved = resolve_artifact_path(str(path))
        return {
            "label": label or resolved.name,
            "path": str(resolved),
            "url": "/artifact?path=" + quote_path(str(resolved)),
            "sizeBytes": resolved.stat().st_size,
        }
    except Exception as exc:
        return {
            "label": label or path.name,
            "path": str(path),
            "url": None,
            "sizeBytes": path.stat().st_size if path.exists() else None,
            "error": str(exc),
        }


def build_robot_realsense_replay(session_dir: Path, origin: list[float]) -> dict[str, Any] | None:
    robot_dir = session_dir / "robot_realsense"
    samples_path = robot_dir / "samples.jsonl"
    if not samples_path.exists():
        return None
    session = read_json_if_exists(robot_dir / "session_summary.json")
    result = read_json_if_exists(robot_dir / "robot_hand_eye_result.json")
    failure = read_json_if_exists(robot_dir / "robot_hand_eye_failure.json")
    alignment = result.get("questAlignment") if isinstance(result, dict) else None
    t_world_base = transform_from_json(alignment.get("T_world_base")) if isinstance(alignment, dict) else None
    rows: list[dict[str, Any]] = []
    ee_poses: list[np.ndarray] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            pose = row.get("T_base_ee") if isinstance(row.get("T_base_ee"), dict) else None
            pose_matrix = transform_from_json(pose)
            world_pose = None
            display_pose = None
            display_frame = "unaligned_robot_base"
            if pose_matrix is not None:
                ee_poses.append(pose_matrix)
                if t_world_base is not None:
                    world_matrix = t_world_base @ pose_matrix
                    world_pose = transform_payload_from_matrix(world_matrix)
                    display_pose = translate_transform_payload(world_pose, origin)
                    display_frame = "quest_world_axes_translated_to_board_origin"
            rows.append(
                {
                    "sampleIndex": row.get("sample_index"),
                    "questSampleIndex": row.get("quest_sample_index"),
                    "recordingTimestampSeconds": row.get("quest_recording_timestamp_seconds"),
                    "ok": bool(row.get("ok")),
                    "T_base_ee": pose,
                    "T_world_ee": world_pose,
                    "T_display_ee": display_pose,
                    "displayFrame": display_frame,
                    "jointpose": row.get("jointpose"),
                    "error": row.get("error"),
                }
            )
    return {
        "directory": str(robot_dir),
        "session": session if isinstance(session, dict) else None,
        "samples": rows,
        "displayFrame": "quest_world_axes_translated_to_board_origin" if t_world_base is not None else "unaligned_robot_base",
        "poseDiversity": ee_pose_diversity(ee_poses),
        "result": result if isinstance(result, dict) else None,
        "failure": failure if isinstance(failure, dict) else None,
    }


def transform_payload_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    return {
        "matrix_4x4": [[float(value) for value in row] for row in matrix[:4, :4]],
        "translation_m": [float(value) for value in matrix[:3, 3]],
    }


def translate_transform_payload(payload: dict[str, Any] | None, origin: list[float]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    matrix = payload.get("matrix_4x4")
    if not isinstance(matrix, list) or len(matrix) < 4:
        return None
    try:
        result = [[float(row[col]) for col in range(4)] for row in matrix[:4]]
        for index in range(3):
            result[index][3] -= origin[index]
        translated = dict(payload)
        translated["matrix_4x4"] = result
        translated["translation_m"] = [result[0][3], result[1][3], result[2][3]]
        return translated
    except (TypeError, ValueError, IndexError):
        return None


def matrix_from_snapshot(snapshot: dict[str, Any]) -> list[list[float]] | None:
    t_world_board = snapshot.get("T_world_board")
    matrix = t_world_board.get("matrix_4x4") if isinstance(t_world_board, dict) else None
    if not isinstance(matrix, list) or len(matrix) < 4:
        return None
    result: list[list[float]] = []
    try:
        for row in matrix[:4]:
            if not isinstance(row, list) or len(row) < 4:
                return None
            result.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    except (TypeError, ValueError):
        return None
    return result


def matrix_translation(matrix: list[list[float]]) -> list[float]:
    return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]


def translated_matrix_4x4(matrix: list[list[float]], origin: list[float]) -> list[list[float]]:
    result = [[float(value) for value in row[:4]] for row in matrix[:4]]
    for index in range(3):
        result[index][3] -= origin[index]
    return result


def matrix_from_snapshot_key(snapshot: dict[str, Any], key: str) -> np.ndarray | None:
    transform = snapshot.get(key)
    matrix = transform.get("matrix_4x4") if isinstance(transform, dict) else None
    if not isinstance(matrix, list) or len(matrix) < 4:
        return None
    try:
        return np.array([[float(row[col]) for col in range(4)] for row in matrix[:4]], dtype=float)
    except (TypeError, ValueError, IndexError):
        return None


def transform_point(matrix: np.ndarray, point: list[float] | np.ndarray) -> np.ndarray:
    value = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=float)
    return (matrix @ value)[:3]


def normalize_np(value: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return value / norm


def stats_summary(values: list[float], scale: float = 1.0) -> dict[str, Any]:
    finite = np.array([float(value) * scale for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def causal_median_depth_filter(depths: list[float | None], window: int = 7) -> list[float | None]:
    result: list[float | None] = []
    recent: list[float] = []
    for value in depths:
        if value is not None and math.isfinite(value):
            recent.append(float(value))
            if len(recent) > window:
                recent = recent[-window:]
            result.append(float(np.median(np.array(recent, dtype=float))))
        else:
            result.append(None)
    return result


def build_gaze_depth_diagnostics(rows: list[dict[str, Any]], snapshot: dict[str, Any]) -> dict[str, Any]:
    t_board_world = matrix_from_snapshot_key(snapshot, "T_board_world")
    if t_board_world is None:
        return {"summary": {"ok": False, "reason": "missing_T_board_world"}, "bySampleIndex": {}}

    rotation_board_world = t_board_world[:3, :3]
    per_sample: list[dict[str, Any]] = []
    sources: dict[str, int] = {}
    depths: list[float | None] = []

    for row in rows:
        sample_index = row.get("sampleIndex")
        gaze_point = vec3_list(row.get("gazePoint3DWorld"))
        ray_origin = vec3_list(row.get("gazeRayOrigin"))
        ray_direction_list = vec3_list(row.get("gazeRayDirection"))
        source = str(row.get("gazePoint3DSource") or row.get("gazeSource") or "unknown")
        sources[source] = sources.get(source, 0) + 1
        if gaze_point is None or ray_origin is None or ray_direction_list is None:
            per_sample.append({"sampleIndex": sample_index, "ok": False, "source": source})
            depths.append(None)
            continue

        ray_origin_np = np.array(ray_origin, dtype=float)
        ray_direction = normalize_np(np.array(ray_direction_list, dtype=float))
        gaze_point_np = np.array(gaze_point, dtype=float)
        if ray_direction is None:
            per_sample.append({"sampleIndex": sample_index, "ok": False, "source": source})
            depths.append(None)
            continue

        ray_delta = gaze_point_np - ray_origin_np
        raw_depth = float(np.dot(ray_delta, ray_direction))
        perpendicular_m = float(np.linalg.norm(ray_delta - raw_depth * ray_direction))
        board_xyz = transform_point(t_board_world, gaze_point_np)
        ray_origin_board = transform_point(t_board_world, ray_origin_np)
        ray_direction_board = rotation_board_world @ ray_direction
        board_plane_depth = None
        depth_minus_board_plane_m = None
        if abs(float(ray_direction_board[2])) > 1e-9:
            board_plane_depth = float(-ray_origin_board[2] / ray_direction_board[2])
            depth_minus_board_plane_m = raw_depth - board_plane_depth

        per_sample.append(
            {
                "sampleIndex": sample_index,
                "ok": True,
                "source": source,
                "rawDepthM": raw_depth,
                "boardPlaneDepthM": board_plane_depth,
                "depthMinusBoardPlaneM": depth_minus_board_plane_m,
                "boardXYZM": [float(board_xyz[0]), float(board_xyz[1]), float(board_xyz[2])],
                "boardDistanceM": float(board_xyz[2]),
                "rayPointPerpendicularM": perpendicular_m,
            }
        )
        depths.append(raw_depth)

    filtered_depths = causal_median_depth_filter(depths, window=7)
    by_sample: dict[str, Any] = {}
    for info, filtered_depth in zip(per_sample, filtered_depths):
        if info.get("ok"):
            info["filteredDepthM"] = filtered_depth
            board_plane_depth = info.get("boardPlaneDepthM")
            if filtered_depth is not None and board_plane_depth is not None:
                info["filteredDepthMinusBoardPlaneM"] = float(filtered_depth - board_plane_depth)
        sample_index = info.get("sampleIndex")
        if sample_index is not None:
            by_sample[str(sample_index)] = info

    consecutive_pairs: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row, info in zip(rows, per_sample):
        if not info.get("ok"):
            previous = None
            continue
        if previous is not None:
            prev_info = previous["info"]
            prev_row = previous["row"]
            prev_index = prev_info.get("sampleIndex")
            current_index = info.get("sampleIndex")
            if (
                isinstance(prev_index, int)
                and isinstance(current_index, int)
                and current_index == prev_index + 1
            ):
                raw_delta = float(info["rawDepthM"] - prev_info["rawDepthM"])
                filtered_delta = None
                if info.get("filteredDepthM") is not None and prev_info.get("filteredDepthM") is not None:
                    filtered_delta = float(info["filteredDepthM"] - prev_info["filteredDepthM"])
                board_delta = float(info["boardDistanceM"] - prev_info["boardDistanceM"])
                timestamp = row.get("recordingTimestampSeconds")
                previous_timestamp = prev_row.get("recordingTimestampSeconds")
                dt = None
                if is_number(timestamp) and is_number(previous_timestamp):
                    dt = float(timestamp) - float(previous_timestamp)
                consecutive_pairs.append(
                    {
                        "from": prev_index,
                        "to": current_index,
                        "t": float(timestamp) if is_number(timestamp) else None,
                        "dt": dt,
                        "rawDepthDeltaM": raw_delta,
                        "filteredDepthDeltaM": filtered_delta,
                        "boardDistanceDeltaM": board_delta,
                    }
                )
        previous = {"row": row, "info": info}

    raw_depths = [float(info["rawDepthM"]) for info in per_sample if info.get("ok")]
    board_distances = [float(info["boardDistanceM"]) for info in per_sample if info.get("ok")]
    depth_plane_errors = [
        float(info["depthMinusBoardPlaneM"])
        for info in per_sample
        if info.get("ok") and info.get("depthMinusBoardPlaneM") is not None
    ]
    filtered_depth_plane_errors = [
        float(info["filteredDepthMinusBoardPlaneM"])
        for info in per_sample
        if info.get("ok") and info.get("filteredDepthMinusBoardPlaneM") is not None
    ]
    raw_delta_abs = [abs(float(pair["rawDepthDeltaM"])) for pair in consecutive_pairs]
    filtered_delta_abs = [
        abs(float(pair["filteredDepthDeltaM"]))
        for pair in consecutive_pairs
        if pair.get("filteredDepthDeltaM") is not None
    ]
    board_delta_abs = [abs(float(pair["boardDistanceDeltaM"])) for pair in consecutive_pairs]
    spikes = sorted(
        consecutive_pairs,
        key=lambda pair: abs(float(pair["rawDepthDeltaM"])),
        reverse=True,
    )

    return {
        "summary": {
            "ok": True,
            "sourceCounts": sources,
            "sampleCount": len(rows),
            "validGazeDepthCount": len(raw_depths),
            "filter": {"kind": "causal_median_depth", "window": 7},
            "rawDepthM": stats_summary(raw_depths),
            "boardDistanceMm": stats_summary(board_distances, scale=1000.0),
            "absBoardDistanceMm": stats_summary([abs(value) for value in board_distances], scale=1000.0),
            "depthMinusBoardPlaneMm": stats_summary(depth_plane_errors, scale=1000.0),
            "absDepthMinusBoardPlaneMm": stats_summary([abs(value) for value in depth_plane_errors], scale=1000.0),
            "filteredDepthMinusBoardPlaneMm": stats_summary(filtered_depth_plane_errors, scale=1000.0),
            "absFilteredDepthMinusBoardPlaneMm": stats_summary(
                [abs(value) for value in filtered_depth_plane_errors],
                scale=1000.0,
            ),
            "absRawDepthDeltaMm": stats_summary(raw_delta_abs, scale=1000.0),
            "absFilteredDepthDeltaMm": stats_summary(filtered_delta_abs, scale=1000.0),
            "absBoardDistanceDeltaMm": stats_summary(board_delta_abs, scale=1000.0),
            "rawDepthSpikeCounts": {
                "gt20mm": sum(1 for value in raw_delta_abs if value > 0.02),
                "gt30mm": sum(1 for value in raw_delta_abs if value > 0.03),
                "gt50mm": sum(1 for value in raw_delta_abs if value > 0.05),
                "gt80mm": sum(1 for value in raw_delta_abs if value > 0.08),
                "gt100mm": sum(1 for value in raw_delta_abs if value > 0.10),
            },
            "topRawDepthSpikes": spikes[:20],
        },
        "bySampleIndex": by_sample,
    }


def enrich_replay_samples_with_gaze_diagnostics(
    samples: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    origin: list[float],
) -> None:
    by_sample = diagnostics.get("bySampleIndex")
    if not isinstance(by_sample, dict):
        return
    for sample in samples:
        sample_index = sample.get("sampleIndex")
        info = by_sample.get(str(sample_index))
        if not isinstance(info, dict):
            continue
        sample["gazeDepth"] = {
            "source": info.get("source"),
            "rawDepthM": info.get("rawDepthM"),
            "filteredDepthM": info.get("filteredDepthM"),
            "boardPlaneDepthM": info.get("boardPlaneDepthM"),
            "boardDistanceM": info.get("boardDistanceM"),
            "depthMinusBoardPlaneM": info.get("depthMinusBoardPlaneM"),
            "filteredDepthMinusBoardPlaneM": info.get("filteredDepthMinusBoardPlaneM"),
        }
        ray_origin = sample.get("gazeRayOrigin")
        ray_direction = sample.get("gazeRayDirection")
        if not isinstance(ray_origin, list) or not isinstance(ray_direction, list):
            continue
        if len(ray_origin) < 3 or len(ray_direction) < 3:
            continue
        origin_np = np.array(ray_origin[:3], dtype=float)
        direction_np = normalize_np(np.array(ray_direction[:3], dtype=float))
        if direction_np is None:
            continue
        filtered_depth = info.get("filteredDepthM")
        if is_number(filtered_depth):
            point = origin_np + direction_np * float(filtered_depth)
            sample["gazeFiltered"] = {
                "ok": True,
                "p": [float(point[0]), float(point[1]), float(point[2])],
                "source": "causal_median_depth_7",
            }
        board_plane_depth = info.get("boardPlaneDepthM")
        if is_number(board_plane_depth):
            point = origin_np + direction_np * float(board_plane_depth)
            sample["gazeBoardPlane"] = {
                "ok": True,
                "p": [float(point[0]), float(point[1]), float(point[2])],
                "source": "ray_intersection_board_z0",
            }


def translate_vec3_list(value: Any, origin: list[float]) -> list[float] | None:
    point = vec3_list(value)
    if point is None:
        return None
    return [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]]


def replay_pose_from_array(value: Any, source: str, origin: list[float]) -> dict[str, Any]:
    pose = pose_from_pose_array(value, source)
    if pose["ok"] and pose["p"] is not None:
        pose["p"] = [pose["p"][0] - origin[0], pose["p"][1] - origin[1], pose["p"][2] - origin[2]]
    return pose


def replay_pose_from_controller(value: Any, handedness: str, origin: list[float]) -> dict[str, Any]:
    pose = visualizer_controller_pose(value, handedness)
    if pose["ok"] and pose["p"] is not None:
        pose["p"] = [pose["p"][0] - origin[0], pose["p"][1] - origin[1], pose["p"][2] - origin[2]]
    return pose


def recording_replay_sample(row: dict[str, Any], origin: list[float]) -> dict[str, Any]:
    left_camera = replay_pose_from_array(row.get("leftCameraPose"), "leftCamera", origin)
    right_camera = replay_pose_from_array(row.get("rightCameraPose"), "rightCamera", origin)
    left_eye = replay_pose_from_array(row.get("leftEyePose"), "leftEye", origin)
    right_eye = replay_pose_from_array(row.get("rightEyePose"), "rightEye", origin)
    head = head_pose_from_pair(left_eye, right_eye, "eye_midpoint")
    if not head["ok"]:
        head = head_pose_from_pair(left_camera, right_camera, "camera_midpoint")
    return {
        "sampleIndex": row.get("sampleIndex"),
        "sequence": row.get("sequence"),
        "recordId": row.get("recordId"),
        "isRecording": bool(row.get("isRecording", True)),
        "telemetryMode": row.get("telemetryMode"),
        "recordingTimestampSeconds": row.get("recordingTimestampSeconds"),
        "pcReceiveUtc": row.get("pcReceiveUtc"),
        "pcReceivePerfCounterSeconds": row.get("pcReceivePerfCounterSeconds"),
        "head": head,
        "leftEye": left_eye,
        "rightEye": right_eye,
        "left": replay_pose_from_controller(row.get("leftController"), "left", origin),
        "right": replay_pose_from_controller(row.get("rightController"), "right", origin),
        "gaze": {
            "ok": translate_vec3_list(row.get("gazePoint3DWorld"), origin) is not None,
            "p": translate_vec3_list(row.get("gazePoint3DWorld"), origin),
            "source": row.get("gazePoint3DSource") or row.get("gazeSource"),
        },
        "gazeHit": {
            "ok": translate_vec3_list(row.get("gazePointWorld"), origin) is not None,
            "p": translate_vec3_list(row.get("gazePointWorld"), origin),
            "source": "gazePointWorld",
        },
        "gazeRayOrigin": translate_vec3_list(row.get("gazeRayOrigin"), origin),
        "gazeRayDirection": vec3_list(row.get("gazeRayDirection")),
    }


def visualizer_event_from_sample(message: dict[str, Any], wrapper: dict[str, Any]) -> dict[str, Any]:
    left_camera = pose_from_pose_array(message.get("leftCameraPose"), "leftCamera")
    right_camera = pose_from_pose_array(message.get("rightCameraPose"), "rightCamera")
    left_eye = pose_from_pose_array(message.get("leftEyePose"), "leftEye")
    right_eye = pose_from_pose_array(message.get("rightEyePose"), "rightEye")
    head = head_pose_from_pair(left_eye, right_eye, "eye_midpoint")
    if not head["ok"]:
        head = head_pose_from_pair(left_camera, right_camera, "camera_midpoint")
    left = visualizer_controller_pose(message.get("leftController"), "left")
    right = visualizer_controller_pose(message.get("rightController"), "right")
    gaze = vec3_list(message.get("gazePoint3DWorld"))

    return {
        "type": "sample",
        "recordId": message.get("recordId"),
        "sequence": message.get("sequence"),
        "sampleIndex": message.get("sampleIndex"),
        "liveSampleIndex": message.get("liveSampleIndex"),
        "isRecording": bool(message.get("isRecording", True)),
        "telemetryMode": message.get("telemetryMode"),
        "recordingTimestampSeconds": message.get("recordingTimestampSeconds"),
        "pcReceiveUnixSeconds": wrapper.get("pcReceiveUnixSeconds"),
        "pcReceivePerfCounterSeconds": wrapper.get("pcReceivePerfCounterSeconds"),
        "pcReceiveUtc": wrapper.get("pcReceiveUtc"),
        "head": head,
        "leftEye": left_eye,
        "rightEye": right_eye,
        "left": left,
        "right": right,
        "gaze": {"ok": gaze is not None, "p": gaze, "source": message.get("gazePoint3DSource")},
    }


def head_pose_from_pair(left_pose: dict[str, Any], right_pose: dict[str, Any], source: str) -> dict[str, Any]:
    if left_pose["ok"] and right_pose["ok"]:
        return {
            "ok": True,
            "source": source,
            "p": [
                (left_pose["p"][0] + right_pose["p"][0]) * 0.5,
                (left_pose["p"][1] + right_pose["p"][1]) * 0.5,
                (left_pose["p"][2] + right_pose["p"][2]) * 0.5,
            ],
            "q": average_quaternion(left_pose["q"], right_pose["q"]),
        }

    if left_pose["ok"]:
        return left_pose
    if right_pose["ok"]:
        return right_pose
    return missing_visualizer_pose("missing_" + source)


def visualizer_controller_pose(value: Any, handedness: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return missing_visualizer_pose("missing_controller_payload")
    if not value.get("hasPose"):
        reason = value.get("missingReason")
        return missing_visualizer_pose(reason if isinstance(reason, str) and reason else "hasPose=false")

    position = vec3_list(value.get("position"))
    rotation = quat_list(value.get("rotation"))
    if position is None:
        pose = pose_from_pose_array(value.get("pose"), value.get("source") or handedness)
        if pose["ok"]:
            return pose
        return missing_visualizer_pose("missing_controller_position")

    return {
        "ok": True,
        "source": value.get("source") or handedness,
        "p": position,
        "q": rotation or [1.0, 0.0, 0.0, 0.0],
    }


def pose_from_pose_array(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) < 7:
        return missing_visualizer_pose("missing_" + source)
    if not all(is_number(item) for item in value[:7]):
        return missing_visualizer_pose("bad_" + source)
    return {
        "ok": True,
        "source": source,
        "p": [float(value[0]), float(value[1]), float(value[2])],
        "q": normalize_quaternion([float(value[3]), float(value[4]), float(value[5]), float(value[6])]),
    }


def missing_visualizer_pose(reason: str) -> dict[str, Any]:
    return {"ok": False, "source": reason, "p": None, "q": None}


def vec3_list(value: Any) -> list[float] | None:
    result = vec3(value)
    if result is None:
        return None
    return [result[0], result[1], result[2]]


def quat_list(value: Any) -> list[float] | None:
    result = quat(value)
    if result is None:
        return None
    return normalize_quaternion([result[0], result[1], result[2], result[3]])


def normalize_quaternion(value: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [item / norm for item in value]


def average_quaternion(a: list[float], b: list[float]) -> list[float]:
    dot = sum(left * right for left, right in zip(a, b))
    sign = -1.0 if dot < 0.0 else 1.0
    return normalize_quaternion([a[index] + sign * b[index] for index in range(4)])


def format_stats(stats: dict[str, float | int] | None) -> str:
    if not stats:
        return "n/a"
    if not all(key in stats for key in ("count", "median", "p95", "max")):
        return f"n={stats.get('count', 0)}"
    return (
        f"n={stats['count']} median={stats['median']:.3f} "
        f"p95={stats['p95']:.3f} max={stats['max']:.3f}"
    )


def sanitize_name(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in ("-", "_", "."):
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("._")
    return result or "unnamed"


def safe_leaf_filename(value: Any, default: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    candidate = sanitize_name(Path(value).name)
    if not candidate or candidate in (".", ".."):
        return default
    return candidate


def quest_output_directory(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    value = message.get("outputDirectory")
    if isinstance(value, str) and value:
        return value
    return None


def unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique output directory for {path}")


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def first_query(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def int_param(query: dict[str, list[str]], key: str, default: int) -> int:
    value = first_query(query, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def float_param(query: dict[str, list[str]], key: str, default: float) -> float:
    value = first_query(query, key)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def bool_param(query: dict[str, list[str]], key: str, default: bool) -> bool:
    value = first_query(query, key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def float_list_param(query: dict[str, list[str]], key: str) -> list[float] | None:
    value = first_query(query, key)
    if value is None:
        return None
    try:
        values = [float(part) for part in value.split(",")]
    except ValueError:
        return None
    if not values or not all(math.isfinite(item) for item in values):
        return None
    return values


def calibration_result_event(record_id: str, result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    per_record = result.get("per_record_calibration")
    calibration = per_record.get(record_id) if isinstance(per_record, dict) else None
    if not isinstance(calibration, dict):
        calibration = next(iter(per_record.values())) if isinstance(per_record, dict) and per_record else {}
    t_world_board = calibration.get("T_world_board") if isinstance(calibration, dict) else None
    t_board_world = calibration.get("T_board_world") if isinstance(calibration, dict) else None
    board_normal = None
    board_normal_angle_y = None
    if isinstance(t_world_board, dict):
        rotation_matrix = t_world_board.get("rotation_matrix")
        if isinstance(rotation_matrix, list) and len(rotation_matrix) >= 3:
            try:
                board_normal = [
                    float(rotation_matrix[0][2]),
                    float(rotation_matrix[1][2]),
                    float(rotation_matrix[2][2]),
                ]
                dot_y = max(-1.0, min(1.0, abs(board_normal[1])))
                board_normal_angle_y = math.degrees(math.acos(dot_y))
            except (TypeError, ValueError, IndexError):
                board_normal = None
                board_normal_angle_y = None
    order = result.get("order_summary") if isinstance(result.get("order_summary"), dict) else {}
    stats_obj = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    overall = stats_obj.get("overall") if isinstance(stats_obj.get("overall"), dict) else {}
    return {
        "type": "calibration_result",
        "recordId": record_id,
        "resultPath": str(result_path),
        "pattern": result.get("pattern") or [11, 8],
        "squareSizeM": result.get("square_size_m") or 0.025,
        "imageYAxis": result.get("image_y_axis"),
        "T_world_board": t_world_board,
        "T_board_world": t_board_world,
        "questWorldOriginInBoardM": calibration.get("quest_world_origin_in_board_m") if isinstance(calibration, dict) else None,
        "boardNormalWorld": board_normal,
        "boardNormalAbsAngleToWorldYDeg": board_normal_angle_y,
        "bestLagSeconds": result.get("best_lag_seconds"),
        "keptFrames": order.get("kept_frames"),
        "inputFrames": order.get("input_frames"),
        "medianReprojectionPx": overall.get("median_px"),
        "p90ReprojectionPx": overall.get("p90_px"),
    }


def recording_calibration_snapshot(output_root: Path | None) -> dict[str, Any] | None:
    if output_root is None:
        return None
    snapshot = latest_calibration_snapshot(output_root)
    event = snapshot.get("event") if isinstance(snapshot, dict) else None
    if not isinstance(event, dict):
        return {
            "capturedUtc": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "reason": snapshot.get("reason") if isinstance(snapshot, dict) else "missing_calibration_snapshot",
        }
    compact = {
        "capturedUtc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(snapshot.get("ok")),
        "kind": snapshot.get("kind"),
        "recordId": snapshot.get("recordId"),
        "resultPath": event.get("resultPath"),
        "pattern": event.get("pattern"),
        "squareSizeM": event.get("squareSizeM"),
        "imageYAxis": event.get("imageYAxis"),
        "T_world_board": event.get("T_world_board"),
        "T_board_world": event.get("T_board_world"),
        "questWorldOriginInBoardM": event.get("questWorldOriginInBoardM"),
        "boardNormalWorld": event.get("boardNormalWorld"),
        "boardNormalAbsAngleToWorldYDeg": event.get("boardNormalAbsAngleToWorldYDeg"),
        "bestLagSeconds": event.get("bestLagSeconds"),
        "keptFrames": event.get("keptFrames"),
        "inputFrames": event.get("inputFrames"),
        "medianReprojectionPx": event.get("medianReprojectionPx"),
        "p90ReprojectionPx": event.get("p90ReprojectionPx"),
    }
    if snapshot.get("kind") != "result":
        compact["reason"] = event.get("reason")
        compact["reasonCode"] = event.get("reasonCode")
    return compact


def is_successful_calibration_snapshot(snapshot: dict[str, Any] | None) -> bool:
    return bool(snapshot and snapshot.get("ok") and snapshot.get("kind") == "result" and snapshot.get("T_world_board"))


def calibration_failure_event(record_id: str, output_directory: Path, log_path: Path, return_code: int | None = None) -> dict[str, Any]:
    failure_path = output_directory / "calibration_failure_25mm.json"
    diagnostics: dict[str, Any] = {}
    if failure_path.exists():
        try:
            diagnostics = json.loads(failure_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            diagnostics = {"reason": "Could not parse calibration_failure_25mm.json"}
    detection_summary = read_json_if_exists(output_directory / "checkerboard_detection_summary_25mm.json")
    log_tail = tail_text(log_path, 80)
    artifacts = calibration_artifacts(output_directory, diagnostics, detection_summary)
    return {
        "type": "calibration_failure",
        "recordId": record_id,
        "rawRecordName": record_id,
        "outputDirectory": str(output_directory),
        "failurePath": str(failure_path) if failure_path.exists() else None,
        "logPath": str(log_path),
        "returnCode": return_code,
        "reason": diagnostics.get("reason") or diagnostics.get("reason_code") or "Calibration failed",
        "reasonCode": diagnostics.get("reason_code"),
        "phase": diagnostics.get("phase"),
        "lagSeconds": diagnostics.get("lag_seconds"),
        "diagnostics": diagnostics,
        "detectionSummary": detection_summary,
        "artifacts": artifacts,
        "logTail": log_tail,
    }


def latest_calibration_snapshot(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    if not output_root.exists():
        return {"ok": False, "reason": "missing_output_root", "outputRoot": str(output_root)}
    candidates = [path for path in output_root.iterdir() if path.is_dir() and path.name.startswith("record_pc_calib")]
    if not candidates:
        return {"ok": False, "reason": "no_calibration_records", "outputRoot": str(output_root)}
    def candidate_stamp(path: Path) -> float:
        result_path = path / "calibration_result_25mm.json"
        if result_path.exists():
            return result_path.stat().st_mtime
        failure_path = path / "calibration_failure_25mm.json"
        if failure_path.exists():
            return failure_path.stat().st_mtime
        return path.stat().st_mtime

    latest = max(candidates, key=candidate_stamp)
    result_path = latest / "calibration_result_25mm.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
        return {
            "ok": True,
            "kind": "result",
            "recordId": latest.name,
            "event": calibration_result_event(latest.name, result, result_path),
        }
    log_path = latest / "calibration_run.log"
    return {
        "ok": True,
        "kind": "failure",
        "recordId": latest.name,
        "event": calibration_failure_event(latest.name, latest, log_path, None),
    }


def calibration_artifacts(output_directory: Path, diagnostics: dict[str, Any], detection_summary: Any) -> dict[str, Any]:
    overlays: list[dict[str, Any]] = []
    rows = detection_summary if isinstance(detection_summary, list) else diagnostics.get("detection_summary")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            path_text = row.get("best_overlay")
            if not isinstance(path_text, str) or not path_text:
                continue
            overlays.append(
                {
                    "record": row.get("record"),
                    "side": row.get("side"),
                    "path": str(resolve_artifact_path(path_text, must_exist=False)),
                    "url": "/artifact?path=" + quote_path(str(resolve_artifact_path(path_text, must_exist=False))),
                }
            )
    return {"overlays": overlays}


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def rizon4_model_payload() -> dict[str, Any]:
    if not DEFAULT_RIZON4_URDF.exists():
        return {"ok": False, "reason": "missing_urdf", "path": str(DEFAULT_RIZON4_URDF)}
    try:
        root = ET.fromstring(DEFAULT_RIZON4_URDF.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": "bad_urdf", "error": str(exc), "path": str(DEFAULT_RIZON4_URDF)}

    joints: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        name = str(joint.attrib.get("name") or "")
        joint_type = str(joint.attrib.get("type") or "")
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        if not name or parent is None or child is None:
            continue
        row: dict[str, Any] = {
            "name": name,
            "type": joint_type,
            "parent": parent.attrib.get("link"),
            "child": child.attrib.get("link"),
            "xyz": parse_float_triplet(origin.attrib.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
            "rpy": parse_float_triplet(origin.attrib.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]),
            "axis": parse_float_triplet(axis.attrib.get("xyz") if axis is not None else None, [0.0, 0.0, 1.0]),
        }
        if limit is not None:
            row["limit"] = {
                "lower": parse_optional_float(limit.attrib.get("lower")),
                "upper": parse_optional_float(limit.attrib.get("upper")),
                "velocity": parse_optional_float(limit.attrib.get("velocity")),
            }
        joints.append(row)

    return {
        "ok": True,
        "name": root.attrib.get("name") or "Rizon4",
        "source": str(DEFAULT_RIZON4_URDF),
        "joints": joints,
        "activeJointNames": [joint["name"] for joint in joints if joint.get("type") != "fixed"],
    }


def recent_controller_status(samples: list[dict[str, Any]], window_seconds: float = 5.0) -> dict[str, Any]:
    now = time.time()
    rows: list[dict[str, Any]] = []
    for sample in reversed(samples):
        if not isinstance(sample, dict):
            continue
        timestamp = sample.get("pcReceiveUnixSeconds")
        if not is_number(timestamp):
            continue
        age = max(0.0, now - float(timestamp))
        if age > window_seconds:
            break
        rows.append(sample)

    def summarize(key: str) -> dict[str, Any]:
        valid = 0
        missing: dict[str, int] = {}
        sources: dict[str, int] = {}
        latest: dict[str, Any] = {}
        latest_age = None
        latest_valid_age = None
        latest_valid_source = None
        for row in rows:
            controller = row.get(key)
            if not isinstance(controller, dict):
                continue
            if not latest:
                latest = controller
                row_timestamp = row.get("pcReceiveUnixSeconds")
                if is_number(row_timestamp):
                    latest_age = max(0.0, now - float(row_timestamp))
            source = str(controller.get("source") or "missing")
            sources[source] = sources.get(source, 0) + 1
            if controller.get("ok"):
                valid += 1
                if latest_valid_age is None:
                    row_timestamp = row.get("pcReceiveUnixSeconds")
                    latest_valid_age = max(0.0, now - float(row_timestamp)) if is_number(row_timestamp) else None
                    latest_valid_source = source
            else:
                missing[source] = missing.get(source, 0) + 1

        return {
            "samples": len(rows),
            "validSamples": valid,
            "latestAgeSeconds": latest_age,
            "latestValidAgeSeconds": latest_valid_age,
            "latestValidSource": latest_valid_source,
            "latest": latest,
            "sources": sources,
            "missing": missing,
        }

    return {
        "windowSeconds": window_seconds,
        "sampleCount": len(rows),
        "left": summarize("left"),
        "right": summarize("right"),
    }


def controller_preflight_detail(latest: dict[str, Any], recent: dict[str, Any], prefix: str) -> str:
    latest_ok = bool(latest.get("ok")) if isinstance(latest, dict) else False
    latest_source = str(latest.get("source") or "n/a") if isinstance(latest, dict) else "n/a"
    valid = int(recent.get("validSamples") or 0) if isinstance(recent, dict) else 0
    samples = int(recent.get("samples") or 0) if isinstance(recent, dict) else 0
    window = float(recent.get("windowSeconds", 5.0)) if isinstance(recent, dict) else 5.0
    text = f"{prefix}latest={latest_ok} ({latest_source}); {valid}/{samples} valid in last {window:.0f}s"
    latest_valid_age = recent.get("latestValidAgeSeconds") if isinstance(recent, dict) else None
    if is_number(latest_valid_age):
        text += f", last valid {float(latest_valid_age):.2f}s ago"
    hint = controller_mode_hint(latest_source)
    if hint:
        text += f"; {hint}"
    return text


def controller_mode_hint(source: str) -> str:
    if (
        "ovrConnected=False" in source
        and "xrValid=False" in source
        and "interactionConnected=0" in source
    ):
        return "Quest reports no Touch controller input, likely Hands mode; wake/pair Touch controllers before arming motion"
    if "interactionRefs=0" in source and "anchorActive=False" in source:
        return "Unity scene has no controller refs/anchors resolved"
    return ""


def build_preflight_status(
    last_sample: dict[str, Any] | None,
    controller_window: dict[str, Any],
    robot_status: dict[str, Any],
    camera_status: dict[str, Any],
    board_status: dict[str, Any],
    model_status: dict[str, Any],
) -> dict[str, Any]:
    now = time.time()
    sample_age = None
    if isinstance(last_sample, dict) and is_number(last_sample.get("pcReceiveUnixSeconds")):
        sample_age = max(0.0, now - float(last_sample["pcReceiveUnixSeconds"]))
    quest_live = isinstance(last_sample, dict) and sample_age is not None and sample_age <= 3.0
    head_ok = bool(last_sample.get("head", {}).get("ok")) if isinstance(last_sample, dict) else False
    gaze_ok = bool(last_sample.get("gaze", {}).get("ok")) if isinstance(last_sample, dict) else False
    left_controller = last_sample.get("left", {}) if isinstance(last_sample, dict) else {}
    right_controller = last_sample.get("right", {}) if isinstance(last_sample, dict) else {}
    right_controller_ok = bool(right_controller.get("ok")) if isinstance(right_controller, dict) else False
    right_controller_source = str(right_controller.get("source") or "n/a") if isinstance(right_controller, dict) else "n/a"
    left_recent = controller_window.get("left", {}) if isinstance(controller_window, dict) else {}
    right_recent = controller_window.get("right", {}) if isinstance(controller_window, dict) else {}
    if isinstance(left_recent, dict):
        left_recent["windowSeconds"] = controller_window.get("windowSeconds", 5.0)
    if isinstance(right_recent, dict):
        right_recent["windowSeconds"] = controller_window.get("windowSeconds", 5.0)

    robot = robot_status.get("robot") if isinstance(robot_status, dict) else {}
    robot_config = robot_status.get("config") if isinstance(robot_status, dict) else {}
    robot_connected = bool(isinstance(robot, dict) and robot.get("connected"))
    motion_armed = bool(isinstance(robot, dict) and robot.get("motionArmed"))
    motion_enabled = bool(isinstance(robot_config, dict) and robot_config.get("controllerMotionEnabled"))
    hand_eye_enabled = bool(isinstance(robot_config, dict) and robot_config.get("runHandEye", True))
    motion_required = bool(robot_connected and hand_eye_enabled)
    camera_serial = str(robot_config.get("cameraSerial") or "") if isinstance(robot_config, dict) else ""
    cameras = camera_status.get("cameras") if isinstance(camera_status, dict) else []
    camera_serials = {
        str(camera.get("serial"))
        for camera in cameras
        if isinstance(camera, dict) and camera.get("serial") is not None
    }
    camera_ok = bool(camera_serial and camera_serial in camera_serials)

    board_ok = bool(board_status.get("ok")) if isinstance(board_status, dict) else False
    board_corners = board_status.get("detectedCorners") if isinstance(board_status, dict) else None
    board_rmse = board_status.get("bestReprojectionRmsePx") if isinstance(board_status, dict) else None
    model_ok = bool(model_status.get("ok")) if isinstance(model_status, dict) else False
    active_joints = model_status.get("activeJointNames") if isinstance(model_status, dict) else []
    active_joint_count = len(active_joints) if isinstance(active_joints, list) else 0

    checks = [
        {
            "id": "questLive",
            "label": "Quest live telemetry",
            "ok": bool(quest_live and head_ok and gaze_ok),
            "detail": f"sample age {sample_age:.2f}s, head={head_ok}, gaze={gaze_ok}" if sample_age is not None else "no live sample yet",
        },
        {
            "id": "flexiv",
            "label": "Flexiv robot",
            "ok": robot_connected,
            "detail": f"{robot.get('robotSn') or robot_config.get('robotSn') or 'n/a'} {robot.get('poseField') or robot_config.get('poseField') or ''}".strip()
            if isinstance(robot, dict)
            else "robot disabled",
        },
        {
            "id": "realsense",
            "label": "End RealSense",
            "ok": camera_ok,
            "detail": f"{camera_serial or 'not selected'}; detected {len(camera_serials)} camera(s)",
        },
        {
            "id": "board",
            "label": "End camera checkerboard",
            "ok": board_ok,
            "detail": (
                f"{board_corners or 0} corners"
                + (f", {float(board_rmse):.2f}px rmse" if is_number(board_rmse) else "")
            )
            if board_status
            else "no board check yet",
        },
        {
            "id": "model",
            "label": "URDF robot model",
            "ok": model_ok and active_joint_count >= 7,
            "detail": f"{model_status.get('name') or 'Rizon4'}, {active_joint_count} active joints"
            if model_ok
            else str(model_status.get("reason") or "missing model"),
        },
        {
            "id": "controllerTelemetry",
            "label": "Quest controller telemetry",
            "ok": right_controller_ok,
            "required": motion_required,
            "detail": (
                controller_preflight_detail(left_controller, left_recent, "left: ")
                + " | "
                + controller_preflight_detail(right_controller, right_recent, "right: ")
            ),
        },
        {
            "id": "robotMotion",
            "label": "Right controller robot motion",
            "ok": bool(motion_armed and motion_enabled and right_controller_ok),
            "required": motion_required,
            "detail": (
                f"required={motion_required}, armed={motion_armed}, controllerMotion={motion_enabled}, "
                f"rightPose={right_controller_ok}, right={right_controller_source}"
            ),
        },
    ]
    ok = all(bool(check.get("ok")) for check in checks if check.get("required", True))
    return {
        "ok": ok,
        "ready": ok,
        "timestampUnixSeconds": now,
        "sampleAgeSeconds": sample_age,
        "controllerWindow": controller_window,
        "checks": checks,
        "summary": "ready for B-button calibration" if ok else "check required items before B-button calibration",
    }


def parse_float_triplet(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    parts = text.split()
    if len(parts) < 3:
        return list(default)
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError:
        return list(default)


def parse_optional_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def tail_text(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def resolve_artifact_path(path_text: str, must_exist: bool = True) -> Path:
    raw = Path(path_text)
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (WORKSPACE_ROOT / raw).resolve()
    root = WORKSPACE_ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path is outside offline_calibration") from exc
    if must_exist and not path.exists():
        raise ValueError("artifact path does not exist")
    return path


def add_board_check_artifact_urls(result: dict[str, Any]) -> None:
    for key in ("imagePath", "overlayPath"):
        path_text = result.get(key)
        if isinstance(path_text, str) and path_text:
            result[key[:-4] + "Url"] = "/artifact?path=" + quote_path(str(resolve_artifact_path(path_text)))


def quote_path(path_text: str) -> str:
    from urllib.parse import quote

    return quote(path_text, safe="")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_vec3(value: Any) -> bool:
    return vec3(value) is not None


def vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    if not all(is_number(item) for item in value):
        return None
    return float(value[0]), float(value[1]), float(value[2])


def quat(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(is_number(item) for item in value):
        return None
    return float(value[0]), float(value[1]), float(value[2]), float(value[3])


LIVE_VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quest Live Telemetry 3D</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0d1012;
  --panel: #171b1f;
  --line: #303840;
  --text: #e8edf2;
  --muted: #9aa5b1;
  --head: #f1ecd0;
  --left: #21c7e8;
  --right: #ff62b8;
  --eye: #a8ff9a;
  --gaze: #f2c94c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font: 13px/1.35 system-ui, -apple-system, Segoe UI, sans-serif;
}
#app {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  height: 100vh;
}
#view {
  width: 100%;
  height: 100%;
  display: block;
  background: #080a0c;
  cursor: grab;
  touch-action: none;
}
#view:active { cursor: grabbing; }
aside {
  border-left: 1px solid var(--line);
  background: var(--panel);
  padding: 14px;
  overflow: auto;
}
h1 {
  margin: 0 0 4px;
  font-size: 16px;
}
.sub {
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 12px;
}
.recording-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border: 1px solid #3c4a52;
  border-radius: 8px;
  background: #20262b;
  color: #cbd5df;
  padding: 10px 12px;
  margin-bottom: 12px;
}
.recording-banner.recording {
  border-color: #ff4f5e;
  background: #42181d;
  color: #fff1f2;
  box-shadow: 0 0 0 2px rgba(255,79,94,0.22), 0 0 24px rgba(255,79,94,0.24);
}
.recording-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #5f6d78;
}
.recording-banner.recording .recording-dot {
  background: #ff4f5e;
  box-shadow: 0 0 14px rgba(255,79,94,0.95);
}
.recording-label {
  font-size: 18px;
  font-weight: 760;
  letter-spacing: 0;
}
.recording-detail {
  color: currentColor;
  opacity: 0.72;
  font-size: 12px;
  text-align: right;
}
.toolbar {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
button {
  color: var(--text);
  background: #242a31;
  border: 1px solid #3b4650;
  border-radius: 6px;
  padding: 7px 9px;
}
button:hover { background: #2d343c; }
.checks {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 10px 0 12px;
}
label {
  display: flex;
  gap: 7px;
  align-items: center;
  color: var(--muted);
}
.legend {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 12px 0;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  color: var(--muted);
}
.dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  margin-right: 6px;
}
.metric {
  border-top: 1px solid var(--line);
  padding: 8px 0;
}
.metric strong {
  color: var(--muted);
  display: inline-block;
  min-width: 92px;
  font-weight: 550;
}
.calibration-progress {
  height: 7px;
  border-radius: 4px;
  background: #273039;
  overflow: hidden;
  margin-top: 6px;
}
.calibration-progress-fill {
  width: 0%;
  height: 100%;
  background: #f2c94c;
  transition: width 160ms ease;
}
.preflight-panel {
  border-top: 1px solid var(--line);
  padding: 10px 0;
  display: grid;
  gap: 8px;
}
.preflight-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.preflight-status {
  font-weight: 700;
  color: #ffd66b;
}
.preflight-status.ready {
  color: #b8f58f;
}
.preflight-list {
  display: grid;
  gap: 5px;
}
.preflight-row {
  display: grid;
  grid-template-columns: 58px 1fr;
  gap: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  background: #0e1216;
}
.preflight-badge {
  font-weight: 700;
  color: #ffd66b;
}
.preflight-badge.ok {
  color: #b8f58f;
}
.preflight-badge.note {
  color: #8fc8ff;
}
.preflight-detail {
  color: var(--muted);
  font-size: 12px;
}
.calibration-details {
  display: grid;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}
.calibration-alert {
  color: #ffd5d9;
  background: rgba(176, 48, 64, 0.22);
  border: 1px solid rgba(255, 93, 112, 0.45);
  border-radius: 6px;
  padding: 8px;
}
.calibration-ok {
  color: #e9f5da;
  background: rgba(82, 133, 62, 0.20);
  border: 1px solid rgba(156, 209, 104, 0.38);
  border-radius: 6px;
  padding: 8px;
}
.calibration-kv {
  display: grid;
  grid-template-columns: 112px 1fr;
  gap: 4px 8px;
}
.calibration-kv span:nth-child(odd) {
  color: #8996a2;
}
.calibration-overlays {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.calibration-overlays img {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #11161b;
}
.calibration-log {
  white-space: pre-wrap;
  max-height: 170px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #0e1216;
  color: #aeb8c1;
}
.robot-panel {
  border-top: 1px solid var(--line);
  margin-top: 12px;
  padding-top: 12px;
  display: grid;
  gap: 8px;
}
.robot-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.robot-panel label {
  display: grid;
  gap: 4px;
  color: var(--muted);
  font-size: 12px;
}
.robot-panel input,
.robot-panel select {
  width: 100%;
  color: var(--text);
  background: #10161b;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 8px;
  font: inherit;
}
.robot-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.robot-status {
  white-space: pre-wrap;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #0e1216;
}
.quest-command-panel {
  border-top: 1px solid var(--line);
  margin-top: 12px;
  padding-top: 12px;
  display: grid;
  gap: 8px;
}
.quest-command-status,
.quest-command-manual {
  white-space: pre-wrap;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #0e1216;
}
.quest-command-manual {
  max-height: 170px;
  overflow: auto;
  color: #aeb8c1;
}
.quest-command-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.robot-board-preview img {
  display: block;
  width: 100%;
  max-height: 260px;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #0e1216;
}
.robot-board-card {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #0e1216;
  display: grid;
  gap: 8px;
}
.robot-board-card.ok {
  border-color: rgba(156, 209, 104, 0.46);
}
.robot-board-card.fail {
  border-color: rgba(255, 93, 112, 0.48);
}
.robot-board-title {
  font-weight: 650;
  color: var(--text);
}
.robot-board-hint {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}
.robot-board-link {
  color: #a8d8ff;
  font-size: 12px;
  text-decoration: none;
}
#status {
  white-space: pre-wrap;
  color: var(--muted);
  border-top: 1px solid var(--line);
  padding-top: 10px;
  margin-top: 10px;
}
@media (max-width: 860px) {
  #app { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; }
  aside { max-height: 46vh; border-left: 0; border-top: 1px solid var(--line); }
}
</style>
</head>
<body>
<div id="app">
  <canvas id="view"></canvas>
  <aside>
    <h1>Quest Live Telemetry 3D</h1>
    <div class="sub">Drag the 3D view to rotate. Mouse wheel zooms.</div>
    <div id="recordingBanner" class="recording-banner">
      <div style="display:flex;align-items:center;gap:9px">
        <span class="recording-dot"></span>
        <span id="recordingLabel" class="recording-label">LIVE</span>
      </div>
      <span id="recordingDetail" class="recording-detail">not recording</span>
    </div>
    <div class="toolbar">
      <button id="reset">Reset View</button>
      <button id="clear">Clear Trail</button>
      <button id="records">PC Records</button>
    </div>
    <div class="checks">
      <label><input id="showTrail" type="checkbox" checked> Trails</label>
      <label><input id="showGrid" type="checkbox" checked> Grid</label>
      <label><input id="showGaze" type="checkbox" checked> Gaze</label>
      <label><input id="followHead" type="checkbox"> Follow Head</label>
    </div>
    <div class="legend">
      <div class="pill"><span class="dot" style="background:var(--head)"></span>Head</div>
      <div class="pill"><span class="dot" style="background:var(--eye)"></span>Eyes</div>
      <div class="pill"><span class="dot" style="background:var(--left)"></span>Left</div>
      <div class="pill"><span class="dot" style="background:var(--right)"></span>Right</div>
      <div class="pill"><span class="dot" style="background:var(--gaze)"></span>Gaze</div>
    </div>
    <div class="metric"><strong>Connection</strong><span id="mConn">connecting</span></div>
    <div class="metric"><strong>Samples</strong><span id="mSamples">0</span></div>
    <div class="metric"><strong>Current</strong><span id="mCurrent">n/a</span></div>
    <div class="metric"><strong>Pose Count</strong><span id="mCounts">head 0, left 0, right 0</span></div>
    <div class="preflight-panel">
      <div class="preflight-head">
        <strong>Preflight</strong>
        <button id="preflightRefresh">Refresh</button>
      </div>
      <div id="preflightStatus" class="preflight-status">checking</div>
      <div id="preflightList" class="preflight-list"></div>
    </div>
    <div class="metric">
      <strong>Calibration</strong><span id="mCalibration">idle</span>
      <div class="calibration-progress"><div id="mCalibrationFill" class="calibration-progress-fill"></div></div>
    </div>
    <div id="calibrationDetails" class="calibration-details"></div>
    <div class="robot-panel">
      <h1>Flexiv / RealSense</h1>
      <label>End RealSense
        <select id="robotCamera"></select>
      </label>
      <div class="robot-grid">
        <label>Robot SN
          <input id="robotSn" spellcheck="false">
        </label>
        <label>Pose field
          <select id="robotPoseField">
            <option value="flange_pose">flange_pose</option>
            <option value="tcp_pose">tcp_pose</option>
          </select>
        </label>
      </div>
      <label>RDK local IP
        <input id="robotNetworkInterfaces" spellcheck="false" placeholder="optional, e.g. 192.168.2.108">
      </label>
      <div class="robot-grid">
        <label>Capture interval
          <input id="robotInterval" type="number" min="0.05" step="0.05">
        </label>
        <label>Hand-eye
          <select id="robotHandEye">
            <option value="true">auto</option>
            <option value="false">record only</option>
          </select>
        </label>
      </div>
      <div class="robot-grid">
        <label>RGB exposure
          <select id="robotExposureMode">
            <option value="auto">auto</option>
            <option value="manual">manual</option>
          </select>
        </label>
        <label>Exposure
          <input id="robotExposure" type="number" min="1" max="10000" step="1">
        </label>
      </div>
      <label>Gain
        <input id="robotGain" type="number" min="0" max="128" step="1">
      </label>
      <label>Board warmup
        <input id="robotBoardWarmup" type="number" min="1" max="120" step="1">
      </label>
      <div class="robot-grid">
        <label>Motion scale
          <input id="robotMotionScale" type="number" min="0" step="0.1">
        </label>
        <label>Max offset m
          <input id="robotMaxOffset" type="number" min="0" step="0.01">
        </label>
      </div>
      <label>Max step m
        <input id="robotMaxStep" type="number" min="0" step="0.005">
      </label>
      <div class="robot-actions">
        <button id="robotRefresh">Refresh Cameras</button>
        <button id="robotBoardCheck">Check Board</button>
        <button id="robotDiagnostics">Diagnostics</button>
        <button id="robotConnect">Connect Robot</button>
        <button id="robotArmMotion">Arm Motion</button>
        <button id="robotDisarmMotion">Disarm</button>
        <button id="robotDisconnect">Disconnect</button>
      </div>
      <div id="robotStatus" class="robot-status">disabled or loading</div>
      <div id="robotBoardPreview" class="robot-board-preview"></div>
    </div>
    <div class="quest-command-panel">
      <h1>Quest Trigger</h1>
      <div class="sub">ADB file commands for calibration when Touch buttons are not available.</div>
      <div class="quest-command-actions">
        <button id="questAdbRefresh">Refresh ADB</button>
        <button id="questCalibStart">Calib Start</button>
        <button id="questCalibStop">Calib Stop</button>
      </div>
      <div id="questAdbStatus" class="quest-command-status">loading</div>
      <div id="questAdbManual" class="quest-command-manual"></div>
    </div>
    <div id="status"></div>
  </aside>
</div>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const resetButton = document.getElementById('reset');
const clearButton = document.getElementById('clear');
const showTrail = document.getElementById('showTrail');
const showGrid = document.getElementById('showGrid');
const showGaze = document.getElementById('showGaze');
const followHead = document.getElementById('followHead');
const mConn = document.getElementById('mConn');
const mSamples = document.getElementById('mSamples');
const mCurrent = document.getElementById('mCurrent');
const mCounts = document.getElementById('mCounts');
const mCalibration = document.getElementById('mCalibration');
const mCalibrationFill = document.getElementById('mCalibrationFill');
const preflightRefresh = document.getElementById('preflightRefresh');
const preflightStatus = document.getElementById('preflightStatus');
const preflightList = document.getElementById('preflightList');
const calibrationDetails = document.getElementById('calibrationDetails');
const statusEl = document.getElementById('status');
const recordingBanner = document.getElementById('recordingBanner');
const recordingLabel = document.getElementById('recordingLabel');
const recordingDetail = document.getElementById('recordingDetail');
const robotCamera = document.getElementById('robotCamera');
const robotSn = document.getElementById('robotSn');
const robotPoseField = document.getElementById('robotPoseField');
const robotNetworkInterfaces = document.getElementById('robotNetworkInterfaces');
const robotInterval = document.getElementById('robotInterval');
const robotHandEye = document.getElementById('robotHandEye');
const robotExposureMode = document.getElementById('robotExposureMode');
const robotExposure = document.getElementById('robotExposure');
const robotGain = document.getElementById('robotGain');
const robotBoardWarmup = document.getElementById('robotBoardWarmup');
const robotMotionScale = document.getElementById('robotMotionScale');
const robotMaxOffset = document.getElementById('robotMaxOffset');
const robotMaxStep = document.getElementById('robotMaxStep');
const robotRefresh = document.getElementById('robotRefresh');
const robotBoardCheck = document.getElementById('robotBoardCheck');
const robotDiagnostics = document.getElementById('robotDiagnostics');
const robotConnect = document.getElementById('robotConnect');
const robotArmMotion = document.getElementById('robotArmMotion');
const robotDisarmMotion = document.getElementById('robotDisarmMotion');
const robotDisconnect = document.getElementById('robotDisconnect');
const robotStatus = document.getElementById('robotStatus');
const robotBoardPreview = document.getElementById('robotBoardPreview');
const questAdbRefresh = document.getElementById('questAdbRefresh');
const questCalibStart = document.getElementById('questCalibStart');
const questCalibStop = document.getElementById('questCalibStop');
const questAdbStatus = document.getElementById('questAdbStatus');
const questAdbManual = document.getElementById('questAdbManual');

const state = {
  frames: [],
  maxFrames: 1200,
  origin: null,
  yaw: -0.72,
  pitch: -0.36,
  distance: 1.35,
  target: [0, 0, 0],
  dragging: false,
  lastPointer: [0, 0],
  counts: {head: 0, left: 0, right: 0},
  calibration: null,
  robot: null,
  robotSample: null,
  robotMotion: null,
  robotDiagnostics: null,
  robotBoardCheck: null,
  robotWorldBase: null,
  robotModel: null,
  robotCalibration: null,
  questAdb: null,
  preflight: null,
  lastPreflightRefreshMs: 0
};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resetView() {
  state.yaw = -0.72;
  state.pitch = -0.36;
  const bounds = computeBounds();
  state.target = bounds.center;
  state.distance = Math.max(0.45, bounds.radius * 3.2);
}

function computeBounds() {
  const pts = [];
  for (const frame of state.frames) {
    for (const key of ['head', 'left', 'right']) {
      const p = relPose(frame[key]);
      if (p) pts.push(p);
    }
    for (const key of ['leftEye', 'rightEye']) {
      const p = relPose(frame[key]);
      if (p) pts.push(p);
    }
    const g = relPoint(frame.gaze?.p);
    if (g) pts.push(g);
  }
  if (!pts.length) return {center: [0, 0, 0], radius: 0.35};
  const min = pts[0].slice();
  const max = pts[0].slice();
  for (const p of pts) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  const center = min.map((value, i) => (value + max[i]) * 0.5);
  let radius = 0.1;
  for (const p of pts) radius = Math.max(radius, length(sub(p, center)));
  return {center, radius};
}

function connect() {
  const events = new EventSource('/events');
  events.onopen = () => { mConn.textContent = 'connected'; };
  events.onerror = () => { mConn.textContent = 'reconnecting'; };
  events.onmessage = (event) => {
    const sample = JSON.parse(event.data);
    if (sample.type === 'sample') {
      ingest(sample);
    } else if (sample.type === 'status') {
      updateRecordingStatus(sample);
    } else if (sample.type === 'calibration_status') {
      updateCalibrationStatus(sample);
    } else if (sample.type === 'calibration_result') {
      updateCalibrationResult(sample);
    } else if (sample.type === 'calibration_failure') {
      updateCalibrationFailure(sample);
    } else if (sample.type === 'robot_status') {
      updateRobotEvent(sample);
    } else if (sample.type === 'robot_sample') {
      updateRobotSample(sample);
    } else if (sample.type === 'robot_motion') {
      updateRobotMotion(sample);
    } else if (sample.type === 'robot_calibration_result' || sample.type === 'robot_calibration_failure') {
      updateRobotCalibration(sample);
    } else if (sample.type === 'quest_adb_status') {
      renderQuestAdb(sample);
    }
  };
}

async function loadLatestCalibration() {
  try {
    const response = await fetch('/calibration/latest', {cache: 'no-store'});
    const payload = await response.json();
    if (!payload.ok || !payload.event) return;
    if (payload.kind === 'result') updateCalibrationResult(payload.event);
    if (payload.kind === 'failure') updateCalibrationFailure(payload.event);
  } catch (_) {
    // The live stream still works if the snapshot endpoint is unavailable.
  }
}

async function loadRobotStatus() {
  try {
    const response = await fetch('/robot/status', {cache: 'no-store'});
    const payload = await response.json();
    applyRobotStatus(payload);
  } catch (error) {
    robotStatus.textContent = String(error);
  }
}

async function loadRobotModel() {
  try {
    const response = await fetch('/robot/model', {cache: 'no-store'});
    const payload = await response.json();
    state.robotModel = payload;
  } catch (_) {
    state.robotModel = null;
  }
  if (state.robotBoardCheck) renderRobotBoardCheck(state.robotBoardCheck);
  else if (state.robot) renderRobotStatus(state.robot);
}

async function loadPreflightStatus() {
  try {
    const response = await fetch('/preflight/status', {cache: 'no-store'});
    renderPreflight(await response.json());
  } catch (error) {
    renderPreflight({ok: false, ready: false, summary: String(error), checks: []});
  }
}

function renderPreflight(payload) {
  state.preflight = payload || {};
  const ready = Boolean(payload?.ready);
  preflightStatus.classList.toggle('ready', ready);
  preflightStatus.textContent = ready ? 'READY' : (payload?.summary || 'check required');
  const checks = Array.isArray(payload?.checks) ? payload.checks : [];
  preflightList.innerHTML = checks.map(check => {
    const ok = Boolean(check.ok);
    const advisory = check.required === false;
    const badge = ok ? 'OK' : (advisory ? 'NOTE' : 'CHECK');
    const badgeClass = ok ? 'ok' : (advisory ? 'note' : '');
    return `
      <div class="preflight-row">
        <div class="preflight-badge ${badgeClass}">${badge}</div>
        <div>
          <div>${escapeHtml(check.label || check.id || '')}</div>
          <div class="preflight-detail">${escapeHtml(check.detail || '')}</div>
        </div>
      </div>`;
  }).join('');
}

async function loadQuestAdbStatus() {
  try {
    const response = await fetch('/quest/adb/status', {cache: 'no-store'});
    renderQuestAdb(await response.json());
  } catch (error) {
    renderQuestAdb({ok: false, ready: false, error: String(error)});
  }
}

async function sendQuestCalibrationCommand(command) {
  const button = command === 'calib_start' ? questCalibStart : questCalibStop;
  button.disabled = true;
  questAdbStatus.textContent = `sending ${command}...`;
  try {
    const response = await fetch('/quest/adb/calibration-command', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({command})
    });
    renderQuestAdb(await response.json());
  } catch (error) {
    renderQuestAdb({ok: false, ready: false, error: String(error)});
  } finally {
    button.disabled = !state.questAdb?.ready;
  }
}

function renderQuestAdb(payload) {
  state.questAdb = payload || {};
  const devices = Array.isArray(payload?.devices) ? payload.devices : [];
  const lines = [
    `adb: ${payload?.ready ? 'ready' : 'unavailable'}`,
    `host adb: ${payload?.adb || 'adb'}`,
    `devices: ${devices.length ? devices.map(d => `${d.serial}:${d.state}`).join(', ') : 'none'}`,
    `command file: ${payload?.commandPath || ''}`
  ];
  if (payload?.command) lines.push(`last command: ${payload.command} ${payload.sent ? 'sent' : 'not sent'}`);
  if (payload?.stage) lines.push(`stage: ${payload.stage}`);
  if (payload?.error) lines.push(`error: ${payload.error}`);
  if (payload?.stderr) lines.push(`stderr: ${payload.stderr}`);
  questAdbStatus.textContent = lines.join('\n');
  const ready = Boolean(payload?.ready);
  questCalibStart.disabled = !ready;
  questCalibStop.disabled = !ready;
  const manual = payload?.manualPowerShell || {};
  const manualText = [
    'Manual PowerShell when the Quest is connected to this Windows PC:',
    manual.calib_start ? `START\n${manual.calib_start}` : '',
    manual.calib_stop ? `STOP\n${manual.calib_stop}` : ''
  ].filter(Boolean).join('\n\n');
  questAdbManual.textContent = ready ? 'Remote receiver can see adb; buttons above are active.' : manualText;
}

async function loadLatestBoardCheck() {
  try {
    const response = await fetch('/robot/board-check/latest', {cache: 'no-store'});
    const payload = await response.json();
    if (payload?.createdAtUtc || payload?.imageUrl) {
      state.robotBoardCheck = payload;
      renderRobotBoardCheck(payload);
    }
  } catch (_) {
    // The board check is optional before the first capture.
  }
}

async function refreshCameras() {
  try {
    const response = await fetch('/cameras/list', {cache: 'no-store'});
    const payload = await response.json();
    const current = robotCamera.value;
    robotCamera.innerHTML = '';
    for (const camera of payload.cameras || []) {
      const option = document.createElement('option');
      option.value = camera.serial || '';
      option.textContent = `${camera.serial || 'unknown'} ${camera.name || ''}`;
      robotCamera.appendChild(option);
    }
    if (current) robotCamera.value = current;
    if (!robotCamera.value && state.robot?.config?.cameraSerial) robotCamera.value = state.robot.config.cameraSerial;
    if (!payload.ok) robotStatus.textContent = payload.error || payload.reason || 'camera list unavailable';
  } catch (error) {
    robotStatus.textContent = String(error);
  }
}

function robotPayloadFromControls() {
  return {
    robotSn: robotSn.value.trim(),
    poseField: robotPoseField.value,
    networkInterfaces: robotNetworkInterfaces.value.split(/[,\s;]+/).map(v => v.trim()).filter(Boolean),
    cameraSerial: robotCamera.value,
    captureIntervalSeconds: Number(robotInterval.value || 0.35),
    realsenseAutoExposure: robotExposureMode.value !== 'manual',
    realsenseExposure: robotExposure.value ? Number(robotExposure.value) : null,
    realsenseGain: robotGain.value ? Number(robotGain.value) : null,
    boardCheckWarmupFrames: Number(robotBoardWarmup.value || 60),
    runHandEye: robotHandEye.value === 'true',
    controllerTranslationScale: Number(robotMotionScale.value || 1.0),
    controllerMaxOffsetM: Number(robotMaxOffset.value || 0.18),
    controllerMaxStepM: Number(robotMaxStep.value || 0.015)
  };
}

async function configureRobot() {
  const response = await fetch('/robot/configure', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(robotPayloadFromControls())
  });
  applyRobotStatus(await response.json());
}

async function connectRobot() {
  robotConnect.disabled = true;
  robotStatus.textContent = 'connecting...';
  try {
    const response = await fetch('/robot/connect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(robotPayloadFromControls())
    });
    const payload = await response.json();
    applyRobotStatus(payload.status || payload);
    if (!payload.ok) robotStatus.textContent += `\nerror: ${payload.error || 'connect failed'}`;
    loadPreflightStatus();
  } catch (error) {
    robotStatus.textContent = String(error);
  } finally {
    robotConnect.disabled = false;
  }
}

async function runRobotDiagnostics() {
  await configureRobot();
  robotDiagnostics.disabled = true;
  robotStatus.textContent = 'running diagnostics...';
  try {
    const response = await fetch('/robot/diagnostics', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(robotPayloadFromControls())
    });
    const payload = await response.json();
    state.robotDiagnostics = payload;
    renderRobotDiagnostics(payload);
  } catch (error) {
    robotStatus.textContent = String(error);
  } finally {
    robotDiagnostics.disabled = false;
  }
}

async function checkRobotBoard() {
  await configureRobot();
  robotBoardCheck.disabled = true;
  robotStatus.textContent = 'checking end camera board...';
  try {
    const response = await fetch('/robot/board-check', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(robotPayloadFromControls())
    });
    const payload = await response.json();
    state.robotBoardCheck = payload;
    renderRobotBoardCheck(payload);
    loadPreflightStatus();
  } catch (error) {
    robotStatus.textContent = String(error);
  } finally {
    robotBoardCheck.disabled = false;
  }
}

function renderRobotBoardCheck(payload) {
  const detected = Boolean(payload?.ok);
  const lines = [
    `board check: ${detected ? 'detected' : 'not detected'}`,
    `camera: ${payload?.camera?.serial || robotCamera.value || 'n/a'}`,
    `corners: ${payload?.detectedCorners ?? 0}`,
    `model: ${robotModelStatusText()}`,
  ];
  if (Number.isFinite(payload?.bestReprojectionRmsePx)) {
    lines.push(`reproj: ${payload.bestReprojectionRmsePx.toFixed(2)} px`);
  }
  if (Number.isFinite(payload?.brightness?.mean) && Number.isFinite(payload?.brightness?.p95)) {
    lines.push(`brightness: mean ${payload.brightness.mean.toFixed(1)}, p95 ${payload.brightness.p95.toFixed(1)}`);
  }
  if (payload?.camera?.colorOptions) {
    const opts = payload.camera.colorOptions;
    lines.push(`sensor: ae ${opts.autoExposure ?? 'n/a'}, exp ${opts.exposure ?? 'n/a'}, gain ${opts.gain ?? 'n/a'}`);
  }
  if (payload?.method) lines.push(`method: ${payload.method}`);
  if (payload?.message) lines.push(payload.message);
  if (payload?.overlayUrl) lines.push(`overlay: ${payload.overlayUrl}`);
  else if (payload?.imageUrl) lines.push(`image: ${payload.imageUrl}`);
  if (payload?.error) lines.push(`error: ${payload.error}`);
  robotStatus.textContent = lines.join('\n');
  const url = payload?.overlayUrl || payload?.imageUrl;
  if (!url) {
    robotBoardPreview.innerHTML = '';
    return;
  }
  const title = detected ? 'End camera sees the board' : 'End camera board not detected';
  const brightness = Number.isFinite(payload?.brightness?.p95)
    ? `brightness p95 ${payload.brightness.p95.toFixed(1)}`
    : 'brightness n/a';
  const hint = detected
    ? `Detected ${payload?.detectedCorners ?? 0} corners. This frame is usable for hand-eye calibration.`
    : 'Move the arm/camera or clear occlusion until the full 11x8 inner-corner board is inside the image.';
  robotBoardPreview.innerHTML = `
    <div class="robot-board-card ${detected ? 'ok' : 'fail'}">
      <div class="robot-board-title">${escapeHtml(title)}</div>
      <a href="${escapeHtml(url)}" target="_blank"><img src="${escapeHtml(url)}" alt="end camera board check"></a>
      <div class="robot-board-hint">${escapeHtml(hint)} ${escapeHtml(brightness)}.</div>
      <a class="robot-board-link" href="${escapeHtml(url)}" target="_blank">open full image</a>
    </div>`;
}

async function disconnectRobot() {
  const response = await fetch('/robot/disconnect', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  applyRobotStatus(await response.json());
}

async function armRobotMotion() {
  await configureRobot();
  robotArmMotion.disabled = true;
  robotStatus.textContent = 'arming motion...';
  try {
    const response = await fetch('/robot/arm-motion', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    const payload = await response.json();
    applyRobotStatus(payload.status || payload);
    if (!payload.ok) robotStatus.textContent += `\nerror: ${payload.error || 'arm failed'}`;
  } catch (error) {
    robotStatus.textContent = String(error);
  } finally {
    robotArmMotion.disabled = false;
  }
}

async function disarmRobotMotion() {
  const response = await fetch('/robot/disarm-motion', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  const payload = await response.json();
  applyRobotStatus(payload.status || payload);
}

function updateRobotEvent(event) {
  if (event.diagnostics) {
    state.robotDiagnostics = event.diagnostics;
    renderRobotDiagnostics(event.diagnostics);
    return;
  }
  if (event.boardCheck) {
    state.robotBoardCheck = event.boardCheck;
    renderRobotBoardCheck(event.boardCheck);
    return;
  }
  if (event.status) applyRobotStatus(event.status);
  else renderRobotStatus(event);
}

function updateRobotSample(event) {
  state.robotSample = event;
  renderRobotStatus(state.robot);
}

function updateRobotMotion(event) {
  state.robotMotion = event;
  renderRobotStatus(state.robot);
}

function updateRobotCalibration(event) {
  state.robotCalibration = event;
  if (event.T_world_base?.matrix_4x4) state.robotWorldBase = event.T_world_base.matrix_4x4;
  renderRobotStatus(state.robot);
}

function applyRobotStatus(payload) {
  state.robot = payload;
  const config = payload?.config || {};
  if (config.robotSn && !robotSn.value) robotSn.value = config.robotSn;
  if (config.poseField) robotPoseField.value = config.poseField;
  if (Array.isArray(config.networkInterfaces) && !robotNetworkInterfaces.value) {
    robotNetworkInterfaces.value = config.networkInterfaces.join(', ');
  }
  if (config.cameraSerial && !robotCamera.value) {
    const existing = Array.from(robotCamera.options).some(option => option.value === config.cameraSerial);
    if (!existing) {
      const option = document.createElement('option');
      option.value = config.cameraSerial;
      option.textContent = `${config.cameraSerial} configured`;
      robotCamera.appendChild(option);
    }
    robotCamera.value = config.cameraSerial;
  }
  if (Number.isFinite(config.captureIntervalSeconds)) robotInterval.value = config.captureIntervalSeconds;
  if (typeof config.realsenseAutoExposure === 'boolean') robotExposureMode.value = config.realsenseAutoExposure ? 'auto' : 'manual';
  if (Number.isFinite(config.realsenseExposure)) robotExposure.value = config.realsenseExposure;
  if (Number.isFinite(config.realsenseGain)) robotGain.value = config.realsenseGain;
  if (Number.isFinite(config.boardCheckWarmupFrames)) robotBoardWarmup.value = config.boardCheckWarmupFrames;
  if (typeof config.runHandEye === 'boolean') robotHandEye.value = config.runHandEye ? 'true' : 'false';
  if (Number.isFinite(config.controllerTranslationScale)) robotMotionScale.value = config.controllerTranslationScale;
  if (Number.isFinite(config.controllerMaxOffsetM)) robotMaxOffset.value = config.controllerMaxOffsetM;
  if (Number.isFinite(config.controllerMaxStepM)) robotMaxStep.value = config.controllerMaxStepM;
  renderRobotStatus(payload);
}

function renderRobotStatus(payload) {
  if (!payload) {
    robotStatus.textContent = 'loading';
    return;
  }
  if (payload.enabled === false) {
    robotStatus.textContent = 'disabled';
    return;
  }
  const robot = payload.robot || {};
  const controllerMotion = Boolean(payload.config?.controllerMotionEnabled);
  const active = payload.activeSession || {};
  const lines = [
    `robot: ${robot.connected ? 'connected' : 'not connected'} ${robot.robotSn || ''}`.trim(),
    `motion: ${robot.motionArmed ? 'ARMED' : 'disarmed'} / controller ${controllerMotion ? 'on' : 'off'}`,
    `pose: ${robot.poseField || 'n/a'}`,
    `rdk iface: ${(payload.config?.networkInterfaces || []).join(', ') || 'default'}`,
    `camera: ${payload.config?.cameraSerial || 'n/a'}`,
    `exposure: ${payload.config?.realsenseAutoExposure === false ? 'manual' : 'auto'}${Number.isFinite(payload.config?.realsenseExposure) ? ` ${payload.config.realsenseExposure}` : ''}${Number.isFinite(payload.config?.realsenseGain) ? ` gain ${payload.config.realsenseGain}` : ''}`,
    `session: ${payload.activeSession ? `${active.samples || 0} samples, ${active.images || 0} images` : 'idle'}`
  ];
  lines.push(`model: ${robotModelStatusText()}`);
  if (state.robotSample) {
    lines.push(`last sample: ${state.robotSample.sampleIndex} quest ${state.robotSample.questSampleIndex}`);
    const fkError = robotFkErrorMm(state.robotSample, null);
    if (Number.isFinite(fkError)) lines.push(`URDF FK vs flange: ${fkError.toFixed(1)}mm`);
    lines.push(`hand-eye motion: ${poseDiversityText(state.robotSample.poseDiversity)}`);
  } else if (active.poseDiversity) {
    lines.push(`hand-eye motion: ${poseDiversityText(active.poseDiversity)}`);
  }
  if (payload.activeSession) {
    lines.push(`motion counts: cmd ${active.motionCommands ?? 0}, skip ${active.motionSkips ?? 0}, err ${active.motionErrors ?? 0}`);
    if (active.lastMotion) {
      lines.push(`last motion: ${robotMotionSummaryText(active.lastMotion)}`);
    }
  }
  if (state.robotMotion) {
    const offset = Array.isArray(state.robotMotion.offsetM) ? state.robotMotion.offsetM.map(v => Number(v).toFixed(3)).join(', ') : (state.robotMotion.reason || state.robotMotion.error || 'n/a');
    lines.push(`motion: ${state.robotMotion.ok ? 'sent' : 'skip'} ${offset}`);
    if (state.robotMotion.anchored !== undefined) {
      const step = Array.isArray(state.robotMotion.stepOffsetM) ? state.robotMotion.stepOffsetM.map(v => Number(v).toFixed(3)).join(', ') : 'n/a';
      lines.push(`motion anchor: ${state.robotMotion.anchored ? 'set' : 'waiting'}${state.robotMotion.createdAnchor ? ' (new)' : ''}, step ${step}`);
    }
  }
  if (state.robotCalibration) {
    lines.push(state.robotCalibration.type === 'robot_calibration_result' ? 'hand-eye: done' : `hand-eye: failed ${state.robotCalibration.error || ''}`);
    const resultDiversity = state.robotCalibration.result?.diversity;
    if (resultDiversity) lines.push(`hand-eye result motion: ${poseDiversityText(resultDiversity)}`);
  }
  if (robot.lastError || payload.lastError) lines.push(`error: ${robot.lastError || payload.lastError}`);
  robotStatus.textContent = lines.join('\n');
}

function robotMotionSummaryText(event) {
  if (!event) return 'n/a';
  if (event.ok) {
    const offset = Array.isArray(event.offsetM)
      ? event.offsetM.map(v => Number(v).toFixed(3)).join(', ')
      : 'sent';
    const step = Array.isArray(event.stepOffsetM)
      ? ` step ${event.stepOffsetM.map(v => Number(v).toFixed(3)).join(', ')}`
      : '';
    return `sent offset ${offset}${step}`;
  }
  return event.reason || event.error || 'skipped';
}

function robotModelStatusText() {
  const model = state.robotModel;
  if (!model) return 'loading';
  if (!model.ok) return `unavailable ${model.reason || ''}`.trim();
  const active = Array.isArray(model.activeJointNames) ? model.activeJointNames.length : 0;
  return `${model.name || 'Rizon4'} URDF, ${active} active joints`;
}

function robotFkErrorMm(robotSample, baseMatrix) {
  const model = state.robotModel;
  const jointpose = robotSample?.jointpose;
  const measured = robotSample?.T_base_ee?.matrix_4x4;
  if (!model?.ok || !Array.isArray(jointpose) || jointpose.length < 7 || !measured) return NaN;
  const base = baseMatrix || identityMatrix4();
  const frames = robotFrames(model, jointpose, base);
  const fk = frames.length ? frames[frames.length - 1] : null;
  const measuredInBase = baseMatrix ? multiplyMatrix4(baseMatrix, measured) : measured;
  const fkP = matrixTranslation(fk);
  const measuredP = matrixTranslation(measuredInBase);
  if (!fkP || !measuredP) return NaN;
  return length(sub(fkP, measuredP)) * 1000;
}

function poseDiversityText(diversity) {
  if (!diversity || !Number.isFinite(diversity.eeTranslationSpanM)) return 'n/a';
  const mm = diversity.eeTranslationSpanM * 1000;
  const deg = Number(diversity.eeRotationSpanDeg || 0);
  const minMm = Number(diversity.minTranslationSpanM || 0.02) * 1000;
  const minDeg = Number(diversity.minRotationSpanDeg || 2.0);
  const ok = mm >= minMm || deg >= minDeg;
  return `${ok ? 'ok' : 'need motion'} ${mm.toFixed(1)}mm / ${deg.toFixed(2)}deg`;
}

function renderRobotDiagnostics(payload) {
  const lines = ['diagnostics'];
  const interp = payload?.interpretation || {};
  for (const line of interp.summary || []) lines.push(`- ${line}`);
  if (Array.isArray(interp.nextSteps) && interp.nextSteps.length) {
    lines.push('next:');
    for (const line of interp.nextSteps) lines.push(`- ${line}`);
  }
  const probes = payload?.probe || [];
  for (const row of probes) {
    if (row.host === '192.168.2.100' || row.ping || (row.openPorts || []).length) {
      lines.push(`probe ${row.host}: ping=${row.ping} open=${(row.openPorts || []).join(',') || 'none'}`);
    }
  }
  const conn = payload?.robotConnection;
  if (conn) {
    lines.push(`rdk: ${conn.ok ? 'ok' : 'failed'} sn=${conn.robotSn || ''} iface=${(conn.networkInterfaces || []).join(',') || 'default'}`);
    if (conn.error) lines.push(`rdk error: ${conn.error}`);
  }
  robotStatus.textContent = lines.join('\n');
}

function ingest(sample) {
  if (!state.origin && sample.head?.ok && sample.head.p) {
    state.origin = sample.head.p.slice();
    state.target = [0, 0, 0];
  }
  state.frames.push(sample);
  if (state.frames.length > state.maxFrames) {
    state.frames.splice(0, state.frames.length - state.maxFrames);
  }
  if (sample.head?.ok) state.counts.head++;
  if (sample.left?.ok) state.counts.left++;
  if (sample.right?.ok) state.counts.right++;
  mSamples.textContent = String(Number(mSamples.textContent || '0') + 1);
  mCounts.textContent = `head ${state.counts.head}, left ${state.counts.left}, right ${state.counts.right}`;
  maybeRefreshPreflight();
  if (followHead && followHead.checked) {
    const p = relPose(sample.head);
    if (p) state.target = p;
  }
}

function maybeRefreshPreflight() {
  const now = performance.now();
  if (state.lastPreflightRefreshMs && now - state.lastPreflightRefreshMs < 2500) return;
  state.lastPreflightRefreshMs = now;
  loadPreflightStatus();
}

function updateRecordingStatus(frame) {
  const isRecording = !!frame.isRecording;
  recordingBanner.classList.toggle('recording', isRecording);
  recordingLabel.textContent = isRecording ? 'REC' : 'LIVE';
  recordingDetail.textContent = isRecording ? 'recording' : 'not recording';
  statusEl.textContent =
    `mode: ${frame.telemetryMode ?? (isRecording ? 'recording' : 'live_preview')}\n` +
    `record: ${frame.recordId ?? 'n/a'}\n` +
    `waiting for ${isRecording ? 'recording' : 'live'} samples...`;
}

function updateCalibrationStatus(event) {
  const pct = Number.isFinite(event.progress) ? `${Math.round(event.progress * 100)}%` : '';
  if (Number.isFinite(event.progress)) {
    mCalibrationFill.style.width = `${Math.max(0, Math.min(100, event.progress * 100))}%`;
  }
  if (!(event.stage === 'done' && state.calibration?.result)) {
    mCalibration.textContent = `${event.stage ?? 'calibration'} ${pct}`.trim();
  }
  state.calibration = {
    ...(state.calibration || {}),
    status: event
  };
  if (event.stage === 'rejected') {
    renderCalibrationRejected(event);
    return;
  }
  if (event.robotStartStatus) {
    state.calibration.robotStartStatus = event.robotStartStatus;
    renderCalibrationStatusDetails(event);
  }
  if (event.diagnostics && !state.calibration?.failure) {
    renderCalibrationDiagnostics({
      type: 'calibration_failure',
      recordId: event.recordId,
      reason: event.message,
      diagnostics: event.diagnostics,
      logPath: event.logPath,
      failurePath: event.failurePath
    });
  }
}

function renderCalibrationRejected(event) {
  const checks = Array.isArray(event.preflight?.checks) ? event.preflight.checks : [];
  const failed = checks.filter(check => (check.required !== false) && !check.ok);
  const rows = failed.length ? failed : checks.filter(check => !check.ok);
  const list = rows.map(check =>
    `<div class="preflight-row">
      <div class="preflight-badge">CHECK</div>
      <div>
        <div>${escapeHtml(check.label || check.id || '')}</div>
        <div class="preflight-detail">${escapeHtml(check.detail || '')}</div>
      </div>
    </div>`
  ).join('');
  calibrationDetails.innerHTML = `
    <div class="calibration-alert">${escapeHtml(event.message || 'B calibration rejected')}</div>
    <div class="calibration-kv">
      <span>record</span><span>${escapeHtml(event.recordId || 'n/a')}</span>
      <span>summary</span><span>${escapeHtml(event.preflight?.summary || event.detail || 'preflight not ready')}</span>
    </div>
    <div class="preflight-list">${list}</div>`;
}

function renderCalibrationStatusDetails(event) {
  const robot = event.robotStartStatus || state.calibration?.robotStartStatus;
  if (!robot) return;
  const robotClass = robot.recording && robot.motionArmed && robot.controllerMotionEnabled ? 'calibration-ok' : 'calibration-alert';
  calibrationDetails.innerHTML = `
    <div class="${robotClass}">${escapeHtml(event.message || 'PC calibration recording')}</div>
    <div class="calibration-kv">
      <span>record</span><span>${escapeHtml(event.recordId || 'n/a')}</span>
      <span>robot rec</span><span>${robot.recording ? 'yes' : 'no'}</span>
      <span>robot</span><span>${escapeHtml(robot.robotConnected ? `${robot.robotSn || 'connected'} ${robot.poseField || ''}` : 'not connected')}</span>
      <span>camera</span><span>${escapeHtml(robot.cameraSerial || 'n/a')}</span>
      <span>motion</span><span>armed=${Boolean(robot.motionArmed)}, controller=${Boolean(robot.controllerMotionEnabled)}</span>
      <span>note</span><span>${escapeHtml(robot.reason || 'n/a')}</span>
    </div>`;
}

function updateCalibrationResult(event) {
  state.calibration = {
    ...(state.calibration || {}),
    result: event
  };
  const median = Number.isFinite(event.medianReprojectionPx) ? `${event.medianReprojectionPx.toFixed(2)}px` : 'n/a';
  mCalibration.textContent = `done ${median}`;
  mCalibrationFill.style.width = '100%';
  if (event.T_world_board?.translation_m && !state.origin) {
    state.origin = event.T_world_board.translation_m.slice();
  }
  renderCalibrationResult(event);
}

function updateCalibrationFailure(event) {
  state.calibration = {
    ...(state.calibration || {}),
    failure: event
  };
  mCalibration.textContent = `failed ${event.reasonCode || ''}`.trim();
  mCalibrationFill.style.width = '100%';
  renderCalibrationDiagnostics(event);
}

function renderCalibrationResult(event) {
  const median = Number.isFinite(event.medianReprojectionPx) ? `${event.medianReprojectionPx.toFixed(2)}px` : 'n/a';
  const p90 = Number.isFinite(event.p90ReprojectionPx) ? `${event.p90ReprojectionPx.toFixed(2)}px` : 'n/a';
  const lag = Number.isFinite(event.bestLagSeconds) ? `${(event.bestLagSeconds * 1000).toFixed(1)} ms` : 'n/a';
  const normalAngle = Number.isFinite(event.boardNormalAbsAngleToWorldYDeg)
    ? `${event.boardNormalAbsAngleToWorldYDeg.toFixed(1)} deg`
    : 'n/a';
  calibrationDetails.innerHTML = `
    <div class="calibration-ok">Calibration succeeded</div>
    <div class="calibration-kv">
      <span>record</span><span>${escapeHtml(event.rawRecordName || event.recordId || 'n/a')}</span>
      <span>image y</span><span>${escapeHtml(event.imageYAxis || 'n/a')}</span>
      <span>lag</span><span>${lag}</span>
      <span>kept frames</span><span>${event.keptFrames ?? 'n/a'} / ${event.inputFrames ?? 'n/a'}</span>
      <span>median / p90</span><span>${median} / ${p90}</span>
      <span>board Z vs world Y</span><span>${normalAngle}</span>
    </div>`;
}

function renderCalibrationDiagnostics(event) {
  const diagnostics = event.diagnostics || {};
  const summary = diagnostics.order_summary || {};
  const stats = diagnostics.best_median_px_stats || {};
  const lag = Number.isFinite(event.lagSeconds ?? diagnostics.lag_seconds)
    ? `${((event.lagSeconds ?? diagnostics.lag_seconds) * 1000).toFixed(1)} ms`
    : 'n/a';
  const threshold = Number.isFinite(diagnostics.order_keep_threshold_px)
    ? `${diagnostics.order_keep_threshold_px.toFixed(1)} px`
    : 'n/a';
  const bestStats = Number.isFinite(stats.median)
    ? `min ${fmtPx(stats.min ?? stats.mean)} / med ${fmtPx(stats.median)} / p90 ${fmtPx(stats.p90)} / max ${fmtPx(stats.max)}`
    : 'n/a';
  const detectionRows = Array.isArray(event.detectionSummary) ? event.detectionSummary : diagnostics.detection_summary;
  const detectionsHtml = renderDetectionRows(detectionRows);
  const keepHtml = renderThresholdCounts(diagnostics.threshold_keep_counts);
  const previewHtml = renderFramePreview(diagnostics.frame_preview);
  const overlayHtml = renderOverlays(event.artifacts?.overlays);
  const logHtml = event.logTail ? `<div class="calibration-log">${escapeHtml(event.logTail)}</div>` : '';
  calibrationDetails.innerHTML = `
    <div class="calibration-alert">${escapeHtml(event.reason || diagnostics.reason || 'Calibration failed')}</div>
    <div class="calibration-kv">
      <span>record</span><span>${escapeHtml(event.rawRecordName || event.recordId || 'n/a')}</span>
      <span>phase</span><span>${escapeHtml(event.phase || diagnostics.phase || 'n/a')}</span>
      <span>reason code</span><span>${escapeHtml(event.reasonCode || diagnostics.reason_code || 'n/a')}</span>
      <span>best lag</span><span>${lag}</span>
      <span>gate</span><span>${threshold}</span>
      <span>kept frames</span><span>${summary.kept_frames ?? 'n/a'} / ${summary.input_frames ?? 'n/a'}</span>
      <span>best error</span><span>${bestStats}</span>
    </div>
    ${detectionsHtml}
    ${keepHtml}
    ${previewHtml}
    ${overlayHtml}
    ${logHtml}`;
}

function renderDetectionRows(rows) {
  if (!Array.isArray(rows) || !rows.length) return '';
  const text = rows.map(row => {
    const ratio = Number.isFinite(row.detection_ratio) ? `${Math.round(row.detection_ratio * 100)}%` : 'n/a';
    const err = row.reprojection_error_px?.median;
    return `${row.side}: ${row.detections}/${row.video_frame_count} (${ratio}), detect med ${fmtPx(err)}`;
  }).join('<br>');
  return `<div><strong>Detections</strong><br>${text}</div>`;
}

function renderThresholdCounts(counts) {
  if (!counts || typeof counts !== 'object') return '';
  const text = Object.entries(counts)
    .map(([threshold, count]) => `${threshold}px:${count}`)
    .join('  ');
  return `<div><strong>Gate keep counts</strong><br>${escapeHtml(text)}</div>`;
}

function renderFramePreview(rows) {
  if (!Array.isArray(rows) || !rows.length) return '';
  const text = rows.slice(0, 6).map(row =>
    `${row.side} f${row.frame_index}: ${fmtPx(row.best_median_px)} ${row.best_order}`
  ).join('<br>');
  return `<div><strong>Best frames</strong><br>${text}</div>`;
}

function renderOverlays(overlays) {
  if (!Array.isArray(overlays) || !overlays.length) return '';
  const imgs = overlays.map(item =>
    `<a href="${item.url}" target="_blank" title="${escapeHtml(item.path || '')}"><img src="${item.url}" alt="${escapeHtml(item.side || 'overlay')} overlay"></a>`
  ).join('');
  return `<div><strong>Checkerboard overlays</strong><div class="calibration-overlays">${imgs}</div></div>`;
}

function fmtPx(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}px` : 'n/a';
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#080a0c';
  ctx.fillRect(0, 0, rect.width, rect.height);

  if (showGrid.checked) drawGrid();
  drawCalibrationBoard();
  if (showTrail.checked) drawTrails();

  const frame = state.frames[state.frames.length - 1];
  if (frame) {
    if (showGaze.checked) drawGaze(frame);
    drawRobotLive();
    drawPose(frame.leftEye, '#a8ff9a', 'leftEye', 0.045);
    drawPose(frame.rightEye, '#a8ff9a', 'rightEye', 0.045);
    drawPose(frame.head, '#f1ecd0', 'head', 0.070);
    drawPose(frame.left, '#21c7e8', 'left', 0.080);
    drawPose(frame.right, '#ff62b8', 'right', 0.080);
    drawRecordingOverlay(frame);
    updateStatus(frame);
  } else {
    statusEl.textContent = 'Waiting for UDP telemetry...';
  }
  requestAnimationFrame(draw);
}

function drawRobotLive() {
  if (!state.robotWorldBase) return;
  const pose = state.robotSample?.T_base_ee;
  const matrix = pose?.matrix_4x4;
  if (!matrix) return;
  const worldMatrix = multiplyMatrix4(state.robotWorldBase, matrix);
  drawRobotSkeleton(state.robotSample?.jointpose, state.robotWorldBase);
  const p = relPoint(matrixTranslation(worldMatrix));
  if (!p) return;
  drawPoint(p, '#ffffff', 6, 'EE');
  drawLine(p, relPoint(matrixPoint(worldMatrix, 0.08, 0, 0)), '#ff4545', 2.3);
  drawLine(p, relPoint(matrixPoint(worldMatrix, 0, 0.08, 0)), '#42e875', 2.3);
  drawLine(p, relPoint(matrixPoint(worldMatrix, 0, 0, 0.08)), '#4b7cff', 2.3);
}

function drawRobotSkeleton(jointpose, baseMatrix) {
  const model = state.robotModel;
  if (!model || !Array.isArray(jointpose) || jointpose.length < 7) return;
  const frames = robotFrames(model, jointpose, baseMatrix);
  if (frames.length < 2) return;
  for (let i = 1; i < frames.length; i++) {
    const a = relPoint(matrixTranslation(frames[i - 1]));
    const b = relPoint(matrixTranslation(frames[i]));
    drawLine(a, b, 'rgba(255,255,255,0.56)', 3);
  }
  for (let i = 0; i < frames.length; i++) {
    const p = relPoint(matrixTranslation(frames[i]));
    drawPoint(p, i === 0 ? '#cbd5df' : '#ffffff', i === 0 ? 4 : 3.5, i === frames.length - 1 ? 'flange' : '');
  }
}

function updateStatus(frame) {
  updateRecordingStatus(frame);
  const isRecording = !!frame.isRecording;
  const t = Number.isFinite(frame.recordingTimestampSeconds)
    ? frame.recordingTimestampSeconds.toFixed(3) + 's'
    : 'n/a';
  const index = isRecording ? frame.sampleIndex : frame.liveSampleIndex;
  mCurrent.textContent = `${isRecording ? 'sample' : 'live'} ${index ?? 'n/a'} @ ${t}`;
  statusEl.textContent =
    `mode: ${frame.telemetryMode ?? (isRecording ? 'recording' : 'live_preview')}\n` +
    `record: ${frame.recordId ?? 'n/a'}\n` +
    `head: ${poseStatus(frame.head)}\n` +
    `leftEye: ${poseStatus(frame.leftEye)}\n` +
    `rightEye: ${poseStatus(frame.rightEye)}\n` +
    `left: ${poseStatus(frame.left)}\n` +
    `right: ${poseStatus(frame.right)}\n` +
    `gaze: ${frame.gaze?.ok ? vecText(frame.gaze.p) : 'missing'}`;
}

function drawRecordingOverlay(frame) {
  const isRecording = !!frame.isRecording;
  const label = isRecording ? 'REC' : 'LIVE';
  const detail = isRecording ? (frame.recordId ?? 'recording') : 'not recording';
  ctx.save();
  ctx.font = '700 18px system-ui, sans-serif';
  const labelWidth = ctx.measureText(label).width;
  ctx.font = '12px system-ui, sans-serif';
  const detailWidth = ctx.measureText(detail).width;
  const width = Math.max(104, labelWidth + detailWidth + 54);
  ctx.fillStyle = isRecording ? 'rgba(88,20,28,0.92)' : 'rgba(28,35,41,0.88)';
  ctx.strokeStyle = isRecording ? 'rgba(255,79,94,0.95)' : 'rgba(80,96,108,0.85)';
  roundRect(16, 16, width, 42, 8);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(37, 37, 6, 0, Math.PI * 2);
  ctx.fillStyle = isRecording ? '#ff4f5e' : '#7b8994';
  ctx.fill();
  if (isRecording) {
    ctx.shadowColor = 'rgba(255,79,94,0.95)';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.fillStyle = '#fff';
  ctx.font = '700 18px system-ui, sans-serif';
  ctx.fillText(label, 51, 42);
  ctx.fillStyle = isRecording ? '#ffd9dd' : '#b8c4ce';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText(detail, 51 + labelWidth + 12, 41);
  ctx.restore();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function poseStatus(pose) {
  if (!pose || !pose.ok) return pose?.source || 'missing';
  return `${pose.source} ${vecText(pose.p)}`;
}

function vecText(p) {
  return `(${p[0].toFixed(3)}, ${p[1].toFixed(3)}, ${p[2].toFixed(3)})`;
}

function drawGrid() {
  const extent = 1.0;
  const step = 0.1;
  for (let v = -extent; v <= extent + 1e-6; v += step) {
    const major = Math.abs(Math.round(v * 10) % 5) === 0;
    const color = major ? 'rgba(92,105,118,0.32)' : 'rgba(92,105,118,0.14)';
    drawLine([-extent, 0, v], [extent, 0, v], color, major ? 1.1 : 0.8);
    drawLine([v, 0, -extent], [v, 0, extent], color, major ? 1.1 : 0.8);
  }
  drawLine([0,0,0], [0.35,0,0], 'rgba(255,75,75,0.8)', 2);
  drawLine([0,0,0], [0,0.35,0], 'rgba(75,255,120,0.8)', 2);
  drawLine([0,0,0], [0,0,0.35], 'rgba(75,125,255,0.8)', 2);
}

function drawCalibrationBoard() {
  const result = state.calibration?.result;
  const transform = result?.T_world_board;
  const matrix = transform?.matrix_4x4;
  if (!matrix || !state.origin) return;
  const pattern = result.pattern || [11, 8];
  const square = result.squareSizeM || 0.025;
  const cols = pattern[0];
  const rows = pattern[1];
  const width = (cols - 1) * square;
  const height = (rows - 1) * square;
  const corners = [
    boardPoint(matrix, 0, 0, 0),
    boardPoint(matrix, width, 0, 0),
    boardPoint(matrix, width, height, 0),
    boardPoint(matrix, 0, height, 0)
  ].map(relPoint);
  if (corners.some(p => !p)) return;

  ctx.save();
  ctx.globalAlpha = 0.22;
  const projected = corners.map(project);
  if (projected.every(p => p.visible)) {
    ctx.beginPath();
    ctx.moveTo(projected[0].x, projected[0].y);
    for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i].x, projected[i].y);
    ctx.closePath();
    ctx.fillStyle = '#f2c94c';
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  for (let c = 0; c < cols; c++) {
    const x = c * square;
    drawLine(relPoint(boardPoint(matrix, x, 0, 0)), relPoint(boardPoint(matrix, x, height, 0)), 'rgba(242,201,76,0.52)', 1.2);
  }
  for (let r = 0; r < rows; r++) {
    const y = r * square;
    drawLine(relPoint(boardPoint(matrix, 0, y, 0)), relPoint(boardPoint(matrix, width, y, 0)), 'rgba(242,201,76,0.52)', 1.2);
  }
  const o = relPoint(boardPoint(matrix, 0, 0, 0));
  drawPoint(o, '#f2c94c', 6, 'board');
  drawLine(o, relPoint(boardPoint(matrix, 0.08, 0, 0)), '#ff4545', 3);
  drawLine(o, relPoint(boardPoint(matrix, 0, 0.08, 0)), '#42e875', 3);
  drawLine(o, relPoint(boardPoint(matrix, 0, 0, 0.08)), '#4b7cff', 3);
  ctx.restore();
}

function drawTrails() {
  drawTrail('head', 'rgba(241,236,208,0.42)');
  drawTrail('left', 'rgba(33,199,232,0.55)');
  drawTrail('right', 'rgba(255,98,184,0.55)');
}

function drawTrail(key, color) {
  let prev = null;
  for (const frame of state.frames) {
    const p = relPose(frame[key]);
    if (!p) { prev = null; continue; }
    if (prev) drawLine(prev, p, color, 1.1);
    prev = p;
  }
}

function drawGaze(frame) {
  const head = relPose(frame.head);
  const gaze = relPoint(frame.gaze?.p);
  if (!gaze) return;
  if (head) drawLine(head, gaze, 'rgba(242,201,76,0.72)', 1.6);
  drawPoint(gaze, '#f2c94c', 4, 'gaze');
}

function drawPose(pose, color, label, axisScale) {
  const p = relPose(pose);
  if (!p) return;
  const q = pose.q || [1, 0, 0, 0];
  drawAxis(p, quatRotate(q, [axisScale, 0, 0]), '#ff4545');
  drawAxis(p, quatRotate(q, [0, axisScale, 0]), '#42e875');
  drawAxis(p, quatRotate(q, [0, 0, axisScale]), '#4b7cff');
  drawPoint(p, color, label === 'head' ? 6 : 5, label);
}

function drawAxis(origin, vector, color) {
  drawLine(origin, add(origin, vector), color, 2.3);
}

function drawPoint(p, color, radius, label) {
  if (!p) return;
  const s = project(p);
  if (!s.visible) return;
  ctx.beginPath();
  ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth = 1;
  ctx.stroke();
  if (label) {
    ctx.fillStyle = '#dce4ec';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(label, s.x + radius + 4, s.y - radius - 2);
  }
}

function drawLine(a, b, color, width) {
  if (!a || !b) return;
  const pa = project(a);
  const pb = project(b);
  if (!pa.visible || !pb.visible) return;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function project(p) {
  const rect = canvas.getBoundingClientRect();
  const x = p[0] - state.target[0];
  const y = p[1] - state.target[1];
  const z = p[2] - state.target[2];
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x1 = cy * x - sy * z;
  const z1 = sy * x + cy * z;
  const y1 = y;
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1 + state.distance;
  const f = Math.min(rect.width, rect.height) * 0.92;
  return {
    x: rect.width * 0.5 + x1 * f / Math.max(0.03, z2),
    y: rect.height * 0.5 - y2 * f / Math.max(0.03, z2),
    z: z2,
    visible: z2 > 0.03
  };
}

function relPose(pose) {
  if (!pose || !pose.ok || !pose.p || !state.origin) return null;
  return relPoint(pose.p);
}

function relPoint(p) {
  if (!p || !state.origin) return null;
  return [p[0] - state.origin[0], p[1] - state.origin[1], p[2] - state.origin[2]];
}

function boardPoint(m, x, y, z) {
  return [
    m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
    m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
    m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
  ];
}

function matrixPoint(m, x, y, z) {
  return [
    m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
    m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
    m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
  ];
}

function matrixTranslation(m) {
  if (!Array.isArray(m) || m.length < 3) return null;
  return [Number(m[0]?.[3]), Number(m[1]?.[3]), Number(m[2]?.[3])];
}

function multiplyMatrix4(a, b) {
  const out = Array.from({length: 4}, () => [0, 0, 0, 0]);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[r][c] = 0;
      for (let k = 0; k < 4; k++) out[r][c] += Number(a[r]?.[k] || 0) * Number(b[k]?.[c] || 0);
    }
  }
  return out;
}

function identityMatrix4() {
  return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
}

function robotFrames(model, jointpose, baseMatrix) {
  const frames = [baseMatrix];
  let current = baseMatrix;
  let jointIndex = 0;
  for (const joint of model.joints || []) {
    current = multiplyMatrix4(current, xyzRpyMatrix(joint.xyz, joint.rpy));
    if (joint.type !== 'fixed') {
      const angle = Number(jointpose[jointIndex++] || 0);
      current = multiplyMatrix4(current, axisAngleMatrix(joint.axis, angle));
    }
    frames.push(current);
  }
  return frames;
}

function xyzRpyMatrix(xyz, rpy) {
  const x = Number(xyz?.[0] || 0), y = Number(xyz?.[1] || 0), z = Number(xyz?.[2] || 0);
  const rx = Number(rpy?.[0] || 0), ry = Number(rpy?.[1] || 0), rz = Number(rpy?.[2] || 0);
  return multiplyMatrix4(translationMatrix(x, y, z), eulerXyzMatrix(rx, ry, rz));
}

function translationMatrix(x, y, z) {
  return [[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]];
}

function eulerXyzMatrix(rx, ry, rz) {
  return multiplyMatrix4(multiplyMatrix4(axisAngleMatrix([1,0,0], rx), axisAngleMatrix([0,1,0], ry)), axisAngleMatrix([0,0,1], rz));
}

function axisAngleMatrix(axis, angle) {
  let x = Number(axis?.[0] || 0), y = Number(axis?.[1] || 0), z = Number(axis?.[2] || 0);
  const n = Math.hypot(x, y, z) || 1;
  x /= n; y /= n; z /= n;
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    [t*x*x + c, t*x*y - s*z, t*x*z + s*y, 0],
    [t*x*y + s*z, t*y*y + c, t*y*z - s*x, 0],
    [t*x*z - s*y, t*y*z + s*x, t*z*z + c, 0],
    [0,0,0,1]
  ];
}

function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function length(a) { return Math.hypot(a[0], a[1], a[2]); }
function quatRotate(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx)
  ];
}

canvas.addEventListener('pointerdown', (event) => {
  state.dragging = true;
  state.lastPointer = [event.clientX, event.clientY];
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', (event) => {
  if (!state.dragging) return;
  const dx = event.clientX - state.lastPointer[0];
  const dy = event.clientY - state.lastPointer[1];
  state.lastPointer = [event.clientX, event.clientY];
  state.yaw += dx * 0.006;
  state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.006));
});
canvas.addEventListener('pointerup', () => { state.dragging = false; });
canvas.addEventListener('pointercancel', () => { state.dragging = false; });
canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  state.distance = Math.max(0.06, state.distance * Math.exp(event.deltaY * 0.001));
}, {passive: false});
resetButton.addEventListener('click', resetView);
clearButton.addEventListener('click', () => {
  state.frames = state.frames.slice(-1);
});
robotRefresh.addEventListener('click', refreshCameras);
robotBoardCheck.addEventListener('click', checkRobotBoard);
robotDiagnostics.addEventListener('click', runRobotDiagnostics);
robotConnect.addEventListener('click', connectRobot);
robotArmMotion.addEventListener('click', armRobotMotion);
robotDisarmMotion.addEventListener('click', disarmRobotMotion);
robotDisconnect.addEventListener('click', disconnectRobot);
preflightRefresh.addEventListener('click', loadPreflightStatus);
questAdbRefresh.addEventListener('click', loadQuestAdbStatus);
questCalibStart.addEventListener('click', () => sendQuestCalibrationCommand('calib_start'));
questCalibStop.addEventListener('click', () => sendQuestCalibrationCommand('calib_stop'));
for (const input of [robotCamera, robotSn, robotPoseField, robotNetworkInterfaces, robotInterval, robotHandEye, robotExposureMode, robotExposure, robotGain, robotBoardWarmup, robotMotionScale, robotMaxOffset, robotMaxStep]) {
  input.addEventListener('change', configureRobot);
}
document.getElementById('records').addEventListener('click', () => {
  window.location.href = '/recordings';
});
window.addEventListener('resize', resize);

resize();
connect();
loadLatestCalibration();
loadRobotStatus().then(refreshCameras);
loadRobotModel();
loadLatestBoardCheck();
loadPreflightStatus();
loadQuestAdbStatus();
requestAnimationFrame(draw);
</script>
</body>
</html>
"""


RECORDINGS_REPLAY_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PC Recording Replay</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0c1014;
  --panel: #121820;
  --line: #2c3846;
  --text: #e8edf3;
  --muted: #91a0af;
  --head: #f1ecd0;
  --eye: #a8ff9a;
  --left: #21c7e8;
  --right: #ff62b8;
  --gaze: #ffd65c;
  --hit: #ff9b54;
  --board: #f2c94c;
  --robot: #ffffff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font: 13px/1.35 system-ui, -apple-system, Segoe UI, sans-serif;
}
#app {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr) 340px;
  height: 100vh;
}
nav, aside {
  overflow: auto;
  background: var(--panel);
  padding: 12px;
}
nav { border-right: 1px solid var(--line); }
aside { border-left: 1px solid var(--line); }
canvas {
  width: 100%;
  height: 100%;
  display: block;
  background: #080a0c;
  touch-action: none;
  cursor: grab;
}
canvas:active { cursor: grabbing; }
h1 { margin: 0 0 6px; font-size: 17px; }
.sub, .tiny, .kv { color: var(--muted); }
.topbar, .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.topbar { margin: 10px 0; }
button {
  color: var(--text);
  background: #182333;
  border: 1px solid #2b3a4f;
  border-radius: 6px;
  padding: 6px 9px;
}
button:hover { background: #213047; }
input[type=range], input[type=checkbox] { accent-color: #82adff; }
.search {
  width: 100%;
  color: var(--text);
  background: #0d131a;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
  margin: 8px 0;
}
.record-list { display: grid; gap: 7px; }
.record-item {
  width: 100%;
  text-align: left;
  display: grid;
  gap: 3px;
  background: #111820;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 8px;
}
.record-item.active { border-color: #82adff; background: #172338; }
.record-title { font-weight: 700; overflow-wrap: anywhere; }
.record-meta { color: var(--muted); font-size: 12px; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 8px;
}
.dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
.section { border-top: 1px solid var(--line); margin-top: 12px; padding-top: 12px; }
.scrub { width: 100%; }
.kv { display: grid; grid-template-columns: 120px minmax(0,1fr); gap: 4px 8px; margin-top: 8px; }
.kv span:nth-child(odd) { color: #8492a1; }
.artifact-list { display: grid; gap: 5px; margin-top: 8px; }
.artifact-link {
  color: #a8d8ff;
  text-decoration: none;
  overflow-wrap: anywhere;
}
.artifact-missing {
  color: #8492a1;
  overflow-wrap: anywhere;
}
.ok { color: #9df09d; font-weight: 700; }
.warn { color: #ffd18a; font-weight: 700; }
.rec { color: #ff8c96; font-weight: 700; }
@media (max-width: 1050px) {
  #app { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr) auto; }
  nav, aside { max-height: 34vh; border: 0; border-bottom: 1px solid var(--line); }
  aside { border-top: 1px solid var(--line); }
}
</style>
</head>
<body>
<div id="app">
  <nav>
    <h1>PC Records</h1>
    <div class="sub" id="rootLabel">loading...</div>
    <div class="topbar">
      <button id="liveBtn">Live View</button>
      <button id="refreshBtn">Refresh</button>
    </div>
    <input id="search" class="search" placeholder="filter record id">
    <div id="recordList" class="record-list"></div>
  </nav>
  <canvas id="view"></canvas>
  <aside>
    <h1 id="recordTitle">Select a record</h1>
    <div class="sub" id="recordSub">Drag to orbit, wheel to zoom.</div>
    <div class="topbar">
      <button id="playBtn">Play</button>
      <button id="resetBtn">Reset</button>
      <label class="pill"><input id="centerBoard" type="checkbox" checked> center board</label>
    </div>
    <div class="topbar">
      <label class="pill">Gaze
        <select id="gazeMode">
          <option value="raw">raw</option>
          <option value="filtered">median depth</option>
          <option value="board">board plane</option>
          <option value="all">all</option>
        </select>
      </label>
    </div>
    <input id="scrub" class="scrub" type="range" min="0" max="0" value="0" step="1">
    <div class="row tiny">
      <span id="timeLabel">0.000s</span>
      <span id="sampleLabel">sample 0</span>
      <span id="recordingLabel"></span>
    </div>
    <div class="row" style="margin-top:10px">
      <span class="pill"><span class="dot" style="background:var(--head)"></span>Head</span>
      <span class="pill"><span class="dot" style="background:var(--eye)"></span>Eyes</span>
      <span class="pill"><span class="dot" style="background:var(--left)"></span>Left</span>
      <span class="pill"><span class="dot" style="background:var(--right)"></span>Right</span>
      <span class="pill"><span class="dot" style="background:var(--gaze)"></span>Gaze3D</span>
      <span class="pill"><span class="dot" style="background:var(--hit)"></span>Hit</span>
      <span class="pill"><span class="dot" style="background:var(--robot)"></span>Robot EE</span>
    </div>
    <div class="section">
      <div class="ok">Calibration snapshot</div>
      <div class="kv" id="snapKv"></div>
      <div class="artifact-list" id="artifactList"></div>
    </div>
    <div class="section">
      <div class="ok">Gaze depth</div>
      <div class="kv" id="depthKv"></div>
    </div>
    <div class="section">
      <div class="ok">Robot / RealSense</div>
      <div class="kv" id="robotKv"></div>
    </div>
    <div class="section">
      <div class="ok">Sample</div>
      <div class="kv" id="sampleKv"></div>
    </div>
  </aside>
</div>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const recordList = document.getElementById('recordList');
const rootLabel = document.getElementById('rootLabel');
const search = document.getElementById('search');
const liveBtn = document.getElementById('liveBtn');
const refreshBtn = document.getElementById('refreshBtn');
const playBtn = document.getElementById('playBtn');
const resetBtn = document.getElementById('resetBtn');
const centerBoard = document.getElementById('centerBoard');
const gazeMode = document.getElementById('gazeMode');
const scrub = document.getElementById('scrub');
const recordTitle = document.getElementById('recordTitle');
const recordSub = document.getElementById('recordSub');
const timeLabel = document.getElementById('timeLabel');
const sampleLabel = document.getElementById('sampleLabel');
const recordingLabel = document.getElementById('recordingLabel');
const snapKv = document.getElementById('snapKv');
const artifactList = document.getElementById('artifactList');
const depthKv = document.getElementById('depthKv');
const robotKv = document.getElementById('robotKv');
const sampleKv = document.getElementById('sampleKv');

const state = {
  records: [],
  selected: null,
  selectedSource: null,
  data: null,
  playing: false,
  t: 0,
  idx: 0,
  yaw: -0.82,
  pitch: -0.34,
  distance: 1.45,
  target: [0,0,0],
  dragging: false,
  lastPointer: [0,0],
  robotModel: null,
  trails: {head: [], left: [], right: [], gaze: [], gazeFiltered: [], gazeBoardPlane: [], hit: [], robot: []}
};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

async function loadRecords() {
  const response = await fetch('/recordings/list', {cache: 'no-store'});
  const payload = await response.json();
  state.records = payload.records || [];
  rootLabel.textContent = payload.root || '';
  renderRecordList();
  if (!state.selected && state.records.length) loadRecord(state.records[0].recordId, state.records[0].source);
}

async function loadRobotModel() {
  try {
    const response = await fetch('/robot/model', {cache: 'no-store'});
    const payload = await response.json();
    state.robotModel = payload;
  } catch (_) {
    state.robotModel = null;
  }
  updateRobotInfo();
  render();
}

function renderRecordList() {
  const needle = search.value.trim().toLowerCase();
  recordList.innerHTML = '';
  for (const record of state.records) {
    if (needle && !record.recordId.toLowerCase().includes(needle)) continue;
    const button = document.createElement('button');
    button.className = 'record-item' + (record.recordId === state.selected && record.source === state.selectedSource ? ' active' : '');
    const samples = record.samples ?? 'n/a';
    const calib = calibrationRecordSummaryText(record);
    const robot = robotRecordSummaryText(record.robotSummary);
    const source = record.sourceLabel || record.source || 'record';
    button.innerHTML = `<span class="record-title">${escapeHtml(record.recordId)}</span><span class="record-meta">${escapeHtml(source)} | ${samples} samples | ${escapeHtml(calib)}</span><span class="record-meta">${escapeHtml(robot)}</span><span class="record-meta">${escapeHtml(record.closedReason || '')}</span>`;
    button.onclick = () => loadRecord(record.recordId, record.source);
    recordList.appendChild(button);
  }
}

function robotRecordSummaryText(summary) {
  if (!summary) return 'no robot';
  const parts = [];
  const status = summary.status || 'recorded';
  parts.push(`robot ${status}`);
  if (summary.samples !== undefined || summary.images !== undefined) {
    const samples = summary.samples ?? 'n/a';
    const images = summary.images ?? 'n/a';
    parts.push(`${samples} sample, ${images} img`);
  }
  if (summary.detections !== undefined || summary.requiredDetections !== undefined) {
    const detected = summary.detections ?? 'n/a';
    const required = summary.requiredDetections ?? '?';
    parts.push(`det ${detected}/${required}`);
  }
  if (summary.motionCommands !== undefined) {
    const skips = summary.motionSkips ?? 0;
    const errors = summary.motionErrors ?? 0;
    parts.push(`motion ${summary.motionCommands}/${skips}/${errors}`);
    if (summary.lastMotionReason) parts.push(String(summary.lastMotionReason).slice(0, 48));
  }
  if (summary.residualMedianMm !== null && summary.residualMedianMm !== undefined && Number.isFinite(Number(summary.residualMedianMm))) {
    parts.push(`res ${Number(summary.residualMedianMm).toFixed(1)}mm`);
  }
  if (summary.status === 'failed' && summary.failureReason) {
    parts.push(String(summary.failureReason).slice(0, 80));
  }
  return parts.join(' | ');
}

function calibrationRecordSummaryText(record) {
  if (record.hasCalibrationSnapshot) return `calib ${record.calibrationRecordId || 'ok'}`;
  if (record.calibrationStatus === 'failed') {
    return `calib failed: ${record.calibrationReasonCode || record.calibrationReason || 'failed'}`;
  }
  return 'no calib';
}

async function loadRecord(recordId, source = null) {
  state.selected = recordId;
  state.selectedSource = source;
  renderRecordList();
  recordTitle.textContent = recordId;
  recordSub.textContent = 'loading...';
  state.playing = false;
  playBtn.textContent = 'Play';
  const params = new URLSearchParams({recordId});
  if (source) params.set('source', source);
  const response = await fetch('/recordings/replay?' + params.toString(), {cache: 'no-store'});
  if (!response.ok) {
    recordSub.textContent = 'failed to load';
    return;
  }
  state.data = await response.json();
  state.idx = 0;
  state.t = state.data.samples?.[0]?.recordingTimestampSeconds || 0;
  buildTrails();
  updateScrub();
  updateSnapshotInfo();
  resetView();
  updateLabels();
  render();
}

function buildTrails() {
  state.trails = {head: [], left: [], right: [], gaze: [], gazeFiltered: [], gazeBoardPlane: [], hit: [], robot: []};
  for (const s of state.data?.samples || []) {
    if (s.head?.p) state.trails.head.push(s.head.p);
    if (s.left?.p) state.trails.left.push(s.left.p);
    if (s.right?.p) state.trails.right.push(s.right.p);
    if (s.gaze?.p) state.trails.gaze.push(s.gaze.p);
    if (s.gazeFiltered?.p) state.trails.gazeFiltered.push(s.gazeFiltered.p);
    if (s.gazeBoardPlane?.p) state.trails.gazeBoardPlane.push(s.gazeBoardPlane.p);
    if (s.gazeHit?.p) state.trails.hit.push(s.gazeHit.p);
  }
  for (const row of state.data?.robotRealSense?.samples || []) {
    const p = matrixTranslation(row.T_display_ee?.matrix_4x4);
    if (p) state.trails.robot.push(p);
  }
}

function updateScrub() {
  const n = state.data?.samples?.length || 0;
  scrub.max = Math.max(0, n - 1);
  scrub.value = state.idx;
}

function currentSample() {
  const samples = state.data?.samples || [];
  return samples[Math.max(0, Math.min(samples.length - 1, state.idx))];
}

function updateSnapshotInfo() {
  const data = state.data || {};
  const s = data.snapshot || {};
  const failure = data.calibrationFailure || null;
  const origin = data.boardOriginWorld || [];
  const status = s.T_world_board ? 'ok' : (failure ? 'failed' : 'missing');
  const kv = [
    ['status', status],
    ['record', s.recordId || 'n/a'],
    ['mode', data.coordinateMode || 'n/a'],
    ['origin', origin.length >= 3 ? origin.map(v => Number(v).toFixed(3)).join(', ') + ' m' : 'n/a'],
    ['image y', s.imageYAxis || 'n/a'],
    ['lag', Number.isFinite(s.bestLagSeconds) ? (s.bestLagSeconds * 1000).toFixed(1) + ' ms' : 'n/a'],
    ['median', Number.isFinite(s.medianReprojectionPx) ? s.medianReprojectionPx.toFixed(2) + ' px' : 'n/a'],
    ['p90', Number.isFinite(s.p90ReprojectionPx) ? s.p90ReprojectionPx.toFixed(2) + ' px' : 'n/a']
  ];
  if (failure) {
    kv.push(
      ['reason', failure.reason || 'Calibration failed'],
      ['reason code', failure.reasonCode || failure.diagnostics?.reason_code || 'n/a'],
      ['phase', failure.phase || failure.diagnostics?.phase || 'n/a'],
      ['detections', calibrationDetectionText(failure)]
    );
  }
  snapKv.innerHTML = kv.map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
  updateArtifactLinks();
  updateDepthInfo();
  updateRobotInfo();
  recordSub.textContent = `${state.data.samples.length} samples`;
}

function updateArtifactLinks() {
  const artifacts = state.data?.rawArtifacts || {};
  const order = [
    'metadata',
    'trajectory',
    'leftFrames',
    'rightFrames',
    'calibrationFailure',
    'calibrationResult',
    'detectionSummary',
    'calibrationLog'
  ];
  const keys = order.filter(key => artifacts[key]).concat(
    Object.keys(artifacts).filter(key => !order.includes(key)).sort()
  );
  if (!keys.length) {
    artifactList.innerHTML = '<div class="artifact-missing">No raw artifacts linked.</div>';
    return;
  }
  artifactList.innerHTML = keys.map(key => {
    const item = artifacts[key] || {};
    const label = item.label || key;
    const size = Number.isFinite(item.sizeBytes) ? ` (${formatBytes(item.sizeBytes)})` : '';
    if (item.url) {
      return `<a class="artifact-link" href="${escapeHtml(item.url)}" target="_blank">${escapeHtml(label + size)}</a>`;
    }
    const reason = item.error ? ` - ${item.error}` : '';
    return `<div class="artifact-missing">${escapeHtml(label + size + reason)}</div>`;
  }).join('');
}

function calibrationDetectionText(failure) {
  const rows = Array.isArray(failure?.detectionSummary)
    ? failure.detectionSummary
    : failure?.diagnostics?.detection_summary;
  if (!Array.isArray(rows) || !rows.length) return 'n/a';
  return rows.map(row => {
    const side = row.side || '?';
    const detections = row.detections ?? 'n/a';
    const frames = row.video_frame_count ?? row.frame_metadata_count ?? 'n/a';
    return `${side} ${detections}/${frames}`;
  }).join(', ');
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return 'n/a';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateDepthInfo() {
  const d = state.data?.gazeDepthDiagnostics || {};
  const rawDelta = d.absRawDepthDeltaMm || {};
  const filtDelta = d.absFilteredDepthDeltaMm || {};
  const board = d.absBoardDistanceMm || {};
  const err = d.absDepthMinusBoardPlaneMm || {};
  const spikes = d.rawDepthSpikeCounts || {};
  const sources = d.sourceCounts || {};
  const kv = [
    ['source', Object.entries(sources).map(([k,v]) => `${k}:${v}`).join(', ') || 'n/a'],
    ['raw |z|', statText(board, 'mm')],
    ['raw plane err', statText(err, 'mm')],
    ['raw dDepth', statText(rawDelta, 'mm')],
    ['filtered dDepth', statText(filtDelta, 'mm')],
    ['spikes', `>50mm ${spikes.gt50mm ?? 'n/a'}, >100mm ${spikes.gt100mm ?? 'n/a'}`]
  ];
  depthKv.innerHTML = kv.map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
}

function updateRobotInfo() {
  const rr = state.data?.robotRealSense;
  if (!rr) {
    const start = state.data?.summary?.robotStartStatus;
    if (start) {
      robotKv.innerHTML = [
        ['status', 'no robot recording'],
        ['start', replayRobotStartText(start)],
        ['note', start.reason || 'n/a']
      ].map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
      return;
    }
    robotKv.innerHTML = '<span>status</span><span>no robot recording</span>';
    return;
  }
  const result = rr.result || {};
  const failure = rr.failure || {};
  const session = rr.session || {};
  const counts = result.counts || failure.counts || {};
  const residual = result.end_camera?.residuals?.translation_mm || {};
  const align = result.questAlignment || {};
  const diversity = result.diversity || failure.diversity || rr.poseDiversity;
  const sample = state.data?.samples?.[state.idx] || {};
  const robot = nearestRobotSample(sample.recordingTimestampSeconds);
  const fkError = replayRobotFkErrorMm(robot);
  const detections = counts.requiredDetections !== undefined
    ? `${counts.detections ?? 'n/a'} / ${counts.requiredDetections}`
    : (counts.detections ?? 'n/a');
  const kv = [
    ['samples', String(rr.samples?.length || 0)],
    ['detections', detections],
    ['model', replayRobotModelStatusText()],
    ['URDF FK', Number.isFinite(fkError) ? `${fkError.toFixed(1)}mm vs flange` : 'n/a'],
    ['ee motion', replayPoseDiversityText(diversity)],
    ['motion', replayMotionSummaryText(session)],
    ['hand-eye', result.ok ? 'ok' : (rr.failure ? 'failed' : 'pending')],
    ['failure', rr.failure?.error || 'n/a'],
    ['residual', Number.isFinite(residual.median) ? `med ${residual.median.toFixed(1)}mm p95 ${Number(residual.p95 || 0).toFixed(1)}mm` : 'n/a'],
    ['quest-base', align.ok ? 'T_world_base ready' : (align.reason || 'n/a')]
  ];
  robotKv.innerHTML = kv.map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
}

function replayRobotStartText(start) {
  if (!start) return 'n/a';
  return `recording=${Boolean(start.recording)}, connected=${Boolean(start.robotConnected)}, armed=${Boolean(start.motionArmed)}, controller=${Boolean(start.controllerMotionEnabled)}`;
}

function replayMotionSummaryText(session) {
  if (!session) return 'n/a';
  const commands = session.motionCommands ?? 'n/a';
  const skips = session.motionSkips ?? 0;
  const errors = session.motionErrors ?? 0;
  const last = session.lastMotion || {};
  const reason = last.reason || last.error || '';
  return `cmd ${commands}, skip ${skips}, err ${errors}${reason ? `, last ${reason}` : ''}`;
}

function replayRobotModelStatusText() {
  const model = state.robotModel;
  if (!model) return 'loading';
  if (!model.ok) return `unavailable ${model.reason || ''}`.trim();
  const active = Array.isArray(model.activeJointNames) ? model.activeJointNames.length : 0;
  return `${model.name || 'Rizon4'} URDF, ${active} active joints`;
}

function replayRobotFkErrorMm(robot) {
  const model = state.robotModel;
  const jointpose = robot?.jointpose;
  const measured = robot?.T_base_ee?.matrix_4x4;
  if (!model?.ok || !Array.isArray(jointpose) || jointpose.length < 7 || !measured) return NaN;
  const frames = robotFrames(model, jointpose, identityMatrix4());
  const fk = frames.length ? frames[frames.length - 1] : null;
  const fkP = matrixTranslation(fk);
  const measuredP = matrixTranslation(measured);
  if (!fkP || !measuredP) return NaN;
  return length(sub(fkP, measuredP)) * 1000;
}

function replayPoseDiversityText(diversity) {
  if (!diversity || !Number.isFinite(diversity.eeTranslationSpanM)) return 'n/a';
  const mm = diversity.eeTranslationSpanM * 1000;
  const deg = Number(diversity.eeRotationSpanDeg || 0);
  const minMm = Number(diversity.minTranslationSpanM || 0.02) * 1000;
  const minDeg = Number(diversity.minRotationSpanDeg || 2.0);
  const ok = mm >= minMm || deg >= minDeg;
  return `${ok ? 'ok' : 'need motion'} ${mm.toFixed(1)}mm / ${deg.toFixed(2)}deg`;
}

function statText(stats, unit) {
  if (!stats || !Number.isFinite(stats.p95)) return 'n/a';
  const p95 = Number(stats.p95).toFixed(1);
  const p99 = Number(stats.p99 ?? stats.max).toFixed(1);
  const max = Number(stats.max).toFixed(1);
  return `p95 ${p95}${unit}, p99 ${p99}${unit}, max ${max}${unit}`;
}

function updateLabels() {
  const s = currentSample();
  if (!s) {
    timeLabel.textContent = '0.000s';
    sampleLabel.textContent = 'sample 0';
    sampleKv.innerHTML = '';
    return;
  }
  timeLabel.textContent = `${Number(s.recordingTimestampSeconds || 0).toFixed(3)}s`;
  sampleLabel.textContent = `sample ${s.sampleIndex ?? state.idx}`;
  recordingLabel.textContent = s.isRecording ? 'REC' : 'LIVE';
  recordingLabel.className = s.isRecording ? 'rec' : 'ok';
  const kv = [
    ['gaze3D', s.gaze?.ok ? (s.gaze.source || 'ok') : 'missing'],
    ['depth raw', Number.isFinite(s.gazeDepth?.rawDepthM) ? (s.gazeDepth.rawDepthM * 1000).toFixed(1) + ' mm' : 'n/a'],
    ['board z', Number.isFinite(s.gazeDepth?.boardDistanceM) ? (s.gazeDepth.boardDistanceM * 1000).toFixed(1) + ' mm' : 'n/a'],
    ['plane err', Number.isFinite(s.gazeDepth?.depthMinusBoardPlaneM) ? (s.gazeDepth.depthMinusBoardPlaneM * 1000).toFixed(1) + ' mm' : 'n/a'],
    ['hit', s.gazeHit?.ok ? 'present' : 'missing'],
    ['left', s.left?.ok ? (s.left.source || 'ok') : (s.left?.source || 'missing')],
    ['right', s.right?.ok ? (s.right.source || 'ok') : (s.right?.source || 'missing')],
    ['head', s.head?.ok ? (s.head.source || 'ok') : 'missing']
  ];
  sampleKv.innerHTML = kv.map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
}

function boardSize() {
  const snap = state.data?.snapshot || {};
  const cols = Math.max(1, (snap.pattern?.[0] || 11) - 1);
  const rows = Math.max(1, (snap.pattern?.[1] || 8) - 1);
  const square = snap.squareSizeM || 0.025;
  return {cols, rows, square, width: cols * square, height: rows * square};
}

function boardPoints() {
  const m = state.data?.boardMatrix;
  if (!m) return [];
  const size = boardSize();
  return [
    boardPoint(m, 0, 0, 0),
    boardPoint(m, size.width, 0, 0),
    boardPoint(m, size.width, size.height, 0),
    boardPoint(m, 0, size.height, 0)
  ];
}

function resetView() {
  state.yaw = -0.82;
  state.pitch = -0.34;
  state.target = [0,0,0];
  const pts = boardPoints();
  for (const s of state.data?.samples || []) {
    for (const key of ['head','left','right','leftEye','rightEye','gaze','gazeFiltered','gazeBoardPlane','gazeHit']) {
      if (s[key]?.p) pts.push(s[key].p);
    }
  }
  for (const row of state.data?.robotRealSense?.samples || []) {
    const p = matrixTranslation(row.T_display_ee?.matrix_4x4);
    if (p) pts.push(p);
  }
  if (!pts.length) {
    state.distance = 1.45;
    return;
  }
  const min = pts[0].slice();
  const max = pts[0].slice();
  for (const p of pts) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  const center = [(min[0]+max[0])*0.5, (min[1]+max[1])*0.5, (min[2]+max[2])*0.5];
  if (!centerBoard.checked) state.target = center;
  let radius = 0.12;
  for (const p of pts) radius = Math.max(radius, length(sub(p, state.target)));
  state.distance = Math.max(0.35, radius * 3.1);
}

function render() {
  const rect = canvas.getBoundingClientRect();
  if (centerBoard.checked) state.target = [0,0,0];
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#080a0c';
  ctx.fillRect(0, 0, rect.width, rect.height);
  drawGroundGrid();
  drawWorldAxes();
  drawBoard();
  drawTrails();
  const s = currentSample();
  if (!s) return;
  drawRobotForSample(s);
  const head = s.head?.p;
  const gaze = currentGazePoint(s);
  const hit = s.gazeHit?.p;
  if (head && gaze) drawLine(head, gaze, 'rgba(255,214,92,.72)', 1.6);
  if (gaze && hit) drawLine(gaze, hit, 'rgba(255,155,84,.5)', 1.2);
  if (s.gazeRayOrigin && s.gazeRayDirection) drawLine(s.gazeRayOrigin, add(s.gazeRayOrigin, scale(s.gazeRayDirection, 0.25)), 'rgba(255,255,255,.35)', 1);
  if (hit) drawPoint(hit, '#ff9b54', 4, 'hit');
  drawGazePoints(s);
  drawPose(s.head, '#f1ecd0', 'head', 0.06);
  drawPose(s.leftEye, '#a8ff9a', 'L-eye', 0.04);
  drawPose(s.rightEye, '#a8ff9a', 'R-eye', 0.04);
  drawPose(s.left, '#21c7e8', 'L', 0.06);
  drawPose(s.right, '#ff62b8', 'R', 0.06);
}

function drawGroundGrid() {
  const extent = 1.0, step = 0.1;
  for (let v = -extent; v <= extent + 1e-6; v += step) {
    const alpha = Math.abs(v) < 1e-6 ? 0.24 : 0.10;
    drawLine([-extent,0,v], [extent,0,v], `rgba(255,255,255,${alpha})`, 1);
    drawLine([v,0,-extent], [v,0,extent], `rgba(255,255,255,${alpha})`, 1);
  }
}

function drawWorldAxes() {
  drawLine([0,0,0], [0.12,0,0], '#ff4545', 2.4);
  drawLine([0,0,0], [0,0.12,0], '#42e875', 2.4);
  drawLine([0,0,0], [0,0,0.12], '#4b7cff', 2.4);
}

function drawBoard() {
  const m = state.data?.boardMatrix;
  if (!m) return;
  const size = boardSize();
  const corners = boardPoints();
  const projected = corners.map(project);
  if (projected.every(p => p.visible)) {
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.beginPath();
    ctx.moveTo(projected[0].x, projected[0].y);
    for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i].x, projected[i].y);
    ctx.closePath();
    ctx.fillStyle = '#f2c94c';
    ctx.fill();
    ctx.restore();
  }
  for (let col = 0; col <= size.cols; col++) {
    const x = col * size.square;
    drawLine(boardPoint(m, x, 0, 0), boardPoint(m, x, size.height, 0), 'rgba(242,201,76,.42)', 1);
  }
  for (let row = 0; row <= size.rows; row++) {
    const y = row * size.square;
    drawLine(boardPoint(m, 0, y, 0), boardPoint(m, size.width, y, 0), 'rgba(242,201,76,.42)', 1);
  }
  const o = boardPoint(m, 0, 0, 0);
  drawPoint(o, '#f2c94c', 5, 'board');
  drawLine(o, boardPoint(m, 0.08, 0, 0), '#ff4545', 3);
  drawLine(o, boardPoint(m, 0, 0.08, 0), '#42e875', 3);
  drawLine(o, boardPoint(m, 0, 0, 0.08), '#4b7cff', 3);
}

function drawTrails() {
  drawTrail(state.trails.head, 'rgba(241,236,208,.35)');
  drawTrail(state.trails.left, 'rgba(33,199,232,.45)');
  drawTrail(state.trails.right, 'rgba(255,98,184,.45)');
  if (gazeMode.value === 'raw' || gazeMode.value === 'all') drawTrail(state.trails.gaze, 'rgba(255,214,92,.32)');
  if (gazeMode.value === 'filtered' || gazeMode.value === 'all') drawTrail(state.trails.gazeFiltered, 'rgba(70,220,255,.34)');
  if (gazeMode.value === 'board' || gazeMode.value === 'all') drawTrail(state.trails.gazeBoardPlane, 'rgba(170,255,132,.34)');
  drawTrail(state.trails.hit, 'rgba(255,155,84,.25)');
  drawTrail(state.trails.robot, 'rgba(255,255,255,.38)');
}

function currentGazePoint(s) {
  if (gazeMode.value === 'filtered') return s.gazeFiltered?.p || s.gaze?.p;
  if (gazeMode.value === 'board') return s.gazeBoardPlane?.p || s.gaze?.p;
  return s.gaze?.p;
}

function drawGazePoints(s) {
  if (gazeMode.value === 'all') {
    if (s.gaze?.p) drawPoint(s.gaze.p, '#ffd65c', 4, 'raw');
    if (s.gazeFiltered?.p) drawPoint(s.gazeFiltered.p, '#46dcff', 4, 'med');
    if (s.gazeBoardPlane?.p) drawPoint(s.gazeBoardPlane.p, '#aaff84', 4, 'plane');
    return;
  }
  const gaze = currentGazePoint(s);
  const label = gazeMode.value === 'filtered' ? 'med' : (gazeMode.value === 'board' ? 'plane' : 'gaze');
  const color = gazeMode.value === 'filtered' ? '#46dcff' : (gazeMode.value === 'board' ? '#aaff84' : '#ffd65c');
  if (gaze) drawPoint(gaze, color, 4, label);
}

function drawTrail(points, color) {
  let prev = null;
  for (const p of points) {
    if (prev) drawLine(prev, p, color, 1.1);
    prev = p;
  }
}

function drawPose(pose, color, label, axisScale) {
  if (!pose?.ok || !pose.p) return;
  const q = pose.q || [1,0,0,0];
  drawLine(pose.p, add(pose.p, quatRotate(q, [axisScale,0,0])), '#ff4545', 2.1);
  drawLine(pose.p, add(pose.p, quatRotate(q, [0,axisScale,0])), '#42e875', 2.1);
  drawLine(pose.p, add(pose.p, quatRotate(q, [0,0,axisScale])), '#4b7cff', 2.1);
  drawPoint(pose.p, color, label === 'head' ? 6 : 5, label);
}

function drawRobotForSample(sample) {
  const robot = nearestRobotSample(sample.recordingTimestampSeconds);
  const matrix = robot?.T_display_ee?.matrix_4x4;
  if (!matrix) return;
  drawRobotSkeleton(robot);
  const p = matrixTranslation(matrix);
  if (!p) return;
  drawPoint(p, '#ffffff', 6, 'EE');
  drawLine(p, matrixPoint(matrix, 0.08, 0, 0), '#ff4545', 2.3);
  drawLine(p, matrixPoint(matrix, 0, 0.08, 0), '#42e875', 2.3);
  drawLine(p, matrixPoint(matrix, 0, 0, 0.08), '#4b7cff', 2.3);
}

function drawRobotSkeleton(robot) {
  const model = state.robotModel;
  const jointpose = robot?.jointpose;
  const result = state.data?.robotRealSense?.result;
  const baseMatrix = result?.questAlignment?.T_world_base?.matrix_4x4;
  if (!model || !Array.isArray(jointpose) || jointpose.length < 7 || !baseMatrix) return;
  const displayBase = translateMatrixPayload(baseMatrix, state.data?.boardOriginWorld || [0,0,0]);
  const frames = robotFrames(model, jointpose, displayBase);
  if (frames.length < 2) return;
  for (let i = 1; i < frames.length; i++) {
    drawLine(matrixTranslation(frames[i - 1]), matrixTranslation(frames[i]), 'rgba(255,255,255,.56)', 3);
  }
  for (let i = 0; i < frames.length; i++) {
    drawPoint(matrixTranslation(frames[i]), i === 0 ? '#cbd5df' : '#ffffff', i === 0 ? 4 : 3.5, i === frames.length - 1 ? 'flange' : '');
  }
}

function nearestRobotSample(t) {
  const rows = state.data?.robotRealSense?.samples || [];
  if (!rows.length) return null;
  if (!Number.isFinite(t)) return rows[0];
  let best = rows[0], bestDt = Infinity;
  for (const row of rows) {
    const rt = Number(row.recordingTimestampSeconds);
    if (!Number.isFinite(rt)) continue;
    const dt = Math.abs(rt - t);
    if (dt < bestDt) {
      best = row;
      bestDt = dt;
    }
  }
  return best;
}

function drawPoint(p, color, radius, label) {
  const s = project(p);
  if (!s.visible) return;
  ctx.beginPath();
  ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,.6)';
  ctx.lineWidth = 1;
  ctx.stroke();
  if (label) {
    ctx.fillStyle = '#dce4ec';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(label, s.x + radius + 4, s.y - radius - 2);
  }
}

function drawLine(a, b, color, width) {
  if (!a || !b) return;
  const pa = project(a), pb = project(b);
  if (!pa.visible || !pb.visible) return;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function project(p) {
  const rect = canvas.getBoundingClientRect();
  const x = p[0] - state.target[0], y = p[1] - state.target[1], z = p[2] - state.target[2];
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x1 = cy * x - sy * z;
  const z1 = sy * x + cy * z;
  const y2 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1 + state.distance;
  const focal = Math.min(rect.width, rect.height) * 0.92;
  return {x: rect.width * 0.5 + x1 * focal / Math.max(0.03, z2), y: rect.height * 0.5 - y2 * focal / Math.max(0.03, z2), visible: z2 > 0.03};
}

function boardPoint(m, x, y, z) {
  return matrixPoint(m, x, y, z);
}

function matrixPoint(m, x, y, z) {
  return [
    m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3],
    m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3],
    m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3]
  ];
}

function matrixTranslation(m) {
  if (!Array.isArray(m) || m.length < 3) return null;
  return [Number(m[0]?.[3]), Number(m[1]?.[3]), Number(m[2]?.[3])];
}

function translateMatrixPayload(m, origin) {
  const out = m.map(row => row.slice());
  for (let i = 0; i < 3; i++) out[i][3] -= Number(origin?.[i] || 0);
  return out;
}

function multiplyMatrix4(a, b) {
  const out = Array.from({length: 4}, () => [0, 0, 0, 0]);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[r][c] = 0;
      for (let k = 0; k < 4; k++) out[r][c] += Number(a[r]?.[k] || 0) * Number(b[k]?.[c] || 0);
    }
  }
  return out;
}

function identityMatrix4() {
  return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
}

function robotFrames(model, jointpose, baseMatrix) {
  const frames = [baseMatrix];
  let current = baseMatrix;
  let jointIndex = 0;
  for (const joint of model.joints || []) {
    current = multiplyMatrix4(current, xyzRpyMatrix(joint.xyz, joint.rpy));
    if (joint.type !== 'fixed') {
      const angle = Number(jointpose[jointIndex++] || 0);
      current = multiplyMatrix4(current, axisAngleMatrix(joint.axis, angle));
    }
    frames.push(current);
  }
  return frames;
}

function xyzRpyMatrix(xyz, rpy) {
  const x = Number(xyz?.[0] || 0), y = Number(xyz?.[1] || 0), z = Number(xyz?.[2] || 0);
  const rx = Number(rpy?.[0] || 0), ry = Number(rpy?.[1] || 0), rz = Number(rpy?.[2] || 0);
  return multiplyMatrix4(translationMatrix(x, y, z), eulerXyzMatrix(rx, ry, rz));
}

function translationMatrix(x, y, z) {
  return [[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]];
}

function eulerXyzMatrix(rx, ry, rz) {
  return multiplyMatrix4(multiplyMatrix4(axisAngleMatrix([1,0,0], rx), axisAngleMatrix([0,1,0], ry)), axisAngleMatrix([0,0,1], rz));
}

function axisAngleMatrix(axis, angle) {
  let x = Number(axis?.[0] || 0), y = Number(axis?.[1] || 0), z = Number(axis?.[2] || 0);
  const n = Math.hypot(x, y, z) || 1;
  x /= n; y /= n; z /= n;
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    [t*x*x + c, t*x*y - s*z, t*x*z + s*y, 0],
    [t*x*y + s*z, t*y*y + c, t*y*z - s*x, 0],
    [t*x*z - s*y, t*y*z + s*x, t*z*z + c, 0],
    [0,0,0,1]
  ];
}

function quatRotate(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)];
}

function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
function scale(a,s){return [a[0]*s,a[1]*s,a[2]*s]}
function length(a){return Math.hypot(a[0],a[1],a[2])}
function escapeHtml(text){return String(text).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}

canvas.addEventListener('pointerdown', event => {
  state.dragging = true;
  state.lastPointer = [event.clientX, event.clientY];
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (!state.dragging) return;
  const dx = event.clientX - state.lastPointer[0];
  const dy = event.clientY - state.lastPointer[1];
  state.lastPointer = [event.clientX, event.clientY];
  state.yaw += dx * 0.006;
  state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.006));
});
canvas.addEventListener('pointerup', () => { state.dragging = false; });
canvas.addEventListener('pointercancel', () => { state.dragging = false; });
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  state.distance = Math.max(0.06, state.distance * Math.exp(event.deltaY * 0.001));
}, {passive: false});

playBtn.onclick = () => {
  state.playing = !state.playing;
  playBtn.textContent = state.playing ? 'Pause' : 'Play';
};
resetBtn.onclick = resetView;
centerBoard.onchange = resetView;
gazeMode.onchange = () => { updateLabels(); render(); };
refreshBtn.onclick = loadRecords;
liveBtn.onclick = () => { window.location.href = '/'; };
search.oninput = renderRecordList;
scrub.oninput = () => {
  const samples = state.data?.samples || [];
  state.idx = Math.max(0, Math.min(samples.length - 1, Number(scrub.value) || 0));
  state.t = samples[state.idx]?.recordingTimestampSeconds || 0;
  updateLabels();
  render();
};
window.addEventListener('resize', () => { resize(); resetView(); render(); });

let lastFrame = performance.now();
function tick(now) {
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  const samples = state.data?.samples || [];
  if (state.playing && samples.length) {
    state.t += dt;
    const end = samples[samples.length - 1].recordingTimestampSeconds || 0;
    if (state.t > end) {
      state.t = samples[0].recordingTimestampSeconds || 0;
      state.idx = 0;
    }
    while (state.idx < samples.length - 1 && samples[state.idx + 1].recordingTimestampSeconds <= state.t) state.idx++;
    while (state.idx > 0 && samples[state.idx].recordingTimestampSeconds > state.t) state.idx--;
    scrub.value = state.idx;
    updateLabels();
  }
  render();
  requestAnimationFrame(tick);
}

resize();
loadRobotModel();
loadRecords().catch(error => { rootLabel.textContent = String(error); });
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def vec3_or_empty(value: Any) -> tuple[Any, Any, Any]:
    result = vec3(value)
    if result is None:
        return "", "", ""
    return result


def quat_or_empty(value: Any) -> tuple[Any, Any, Any, Any]:
    result = quat(value)
    if result is None:
        return "", "", "", ""
    return result


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def quaternion_angle_degrees(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    dot = abs(sum(left * right for left, right in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def controller_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def has_controller_pose(value: Any) -> bool:
    controller = controller_dict(value)
    return bool(controller.get("hasPose")) and is_vec3(controller.get("position"))


def controller_pose_position(value: Any) -> tuple[float, float, float] | None:
    controller = controller_dict(value)
    return vec3(controller.get("position")) if has_controller_pose(controller) else None


def pose_position(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    return vec3(value[:3])


def controller_status(value: Any) -> str:
    controller = controller_dict(value)
    source = controller.get("source") or "missing"
    if has_controller_pose(controller):
        position = vec3(controller.get("position"))
        if position is None:
            return f"{source}:pose"
        return f"{source}:({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})"
    reason = controller.get("missingReason")
    if reason:
        return f"{source}:missing/{reason}"
    return f"{source}:missing"


def count_source(counter: dict[str, int], value: Any) -> None:
    controller = controller_dict(value)
    source = controller.get("source")
    if not isinstance(source, str) or not source:
        source = "missing"
    counter[source] = counter.get(source, 0) + 1


def count_missing_reason(counter: dict[str, int], value: Any) -> None:
    controller = controller_dict(value)
    if controller.get("hasPose"):
        return
    reason = controller.get("missingReason")
    if not isinstance(reason, str) or not reason:
        reason = "unspecified"
    counter[reason] = counter.get(reason, 0) + 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
