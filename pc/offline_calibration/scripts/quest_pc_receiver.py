from __future__ import annotations

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import posixpath
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from flexiv_realsense_bridge import (
    DEFAULT_END_CAMERA_SERIAL,
    DEFAULT_FLEXIV_RDK_ROOT,
    DEFAULT_FLEXIV_ROBOT_SN,
    DEFAULT_GRIPPER_CLOSE_WIDTH_M,
    DEFAULT_GRIPPER_DEVICE,
    DEFAULT_GRIPPER_FORCE_N,
    DEFAULT_GRIPPER_INIT_ON_ENABLE,
    DEFAULT_GRIPPER_OPEN_WIDTH_M,
    DEFAULT_GRIPPER_SPEED_MPS,
    DEFAULT_GRIPPER_TRIGGER_CLOSE_THRESHOLD,
    DEFAULT_GRIPPER_TRIGGER_OPEN_THRESHOLD,
    DEFAULT_HAND_EYE_DIVERSE_MIN_SCORE,
    DEFAULT_HAND_EYE_DIVERSE_ROTATION_SCALE_DEG,
    DEFAULT_HAND_EYE_DIVERSE_TRANSLATION_SCALE_M,
    DEFAULT_HAND_EYE_MAX_DIVERSE_SAMPLES,
    DEFAULT_HAND_EYE_MIN_DIVERSE_SAMPLES,
    DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES,
    DEFAULT_RECORD_DEPTH_FORMAT,
    DEFAULT_ROBOT_STATE_HZ,
    DEFAULT_CONTROLLER_JOINT_LIMIT_BUFFER_RAD,
    DEFAULT_CONTROLLER_MAX_STEP_M,
    DEFAULT_CONTROLLER_MAX_ROTATION_STEP_DEG,
    DEFAULT_CONTROLLER_TARGET_UPDATE_HZ,
    DEFAULT_CONTROLLER_TRANSLATION_SCALE,
    ROBOT_SESSION_RECORD_ASYNC,
    ROBOT_SESSION_CONTROL_FREEDRIVE,
    ROBOT_SESSION_CONTROL_TELEOP,
    ColorStreamWriter,
    FlexivRealSenseConfig,
    FlexivRealSenseManager,
    RobotRealsenseSession,
    average_transforms,
    compact_robot_result,
    ee_pose_diversity,
    flexiv_pose_payload_to_transform,
    invert_transform,
    robot_row_tool_transform,
    robot_state_pose_transform,
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
from quest_coordinate_frames import (
    PC_WORLD_FRAME,
    UNITY_WORLD_FRAME,
    WORLD_FRAME_CONVERSION,
    ensure_pc_transform_payload,
    is_pc_world_frame,
    matrix_from_transform_payload,
    transform_payload_from_matrix as coordinate_transform_payload_from_matrix,
    unity_pose_array_to_pc,
    unity_quaternion_wxyz_to_pc,
    unity_vec3_to_pc,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "pc_recordings"
DEFAULT_QUEST_LOCAL_ROOT = WORKSPACE_ROOT / "raw"
DEFAULT_CALIBRATION_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "pc_live_calibration"
DEFAULT_RIZON_URDF = WORKSPACE_ROOT / "assets" / "urdf" / "flexiv_Rizon4_kinematics.urdf"
LATE_RECORDING_SAMPLE_GRACE_SECONDS = 5.0
RECENTLY_CLOSED_RECORD_REOPEN_GUARD_SECONDS = 30.0
MAX_CALIBRATION_HTTP_BODY_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
ARTIFACT_STREAM_CHUNK_BYTES = 1024 * 1024
ALLOWED_ARTIFACT_SUFFIXES = {
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".jpg",
    ".jpeg",
    ".png",
    ".mp4",
    ".mkv",
    ".bin",
}
QUEST_RECORD_COMMAND_PATH = "/sdcard/Android/data/com.Apricity.EyeTrackingTest/files/record_command.txt"
DEFAULT_ADB = Path(
    r"C:\Program Files\Unity\Hub\Editor\6000.0.60f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe"
)
DEFAULT_RECEIVE_FLUSH_EVERY = 32
DEFAULT_RECEIVE_FLUSH_INTERVAL_SECONDS = 0.25
DEFAULT_SAMPLE_LOG_INTERVAL_SECONDS = 1.0
DEFAULT_UDP_RECEIVE_BUFFER_BYTES = 4 * 1024 * 1024
DEFAULT_RECORDING_IDLE_TIMEOUT_SECONDS = 2.0
SESSION_WRITE_QUEUE_MAX = 2048
SESSION_WRITE_QUEUE_CLOSE_DRAIN_SECONDS = 2.0
ROBOT_SESSION_ASYNC_SAMPLE_HZ = 5.0
SESSION_RAW_SAMPLE_HZ = 5.0
SESSION_CONTROLLER_CSV_HZ = 5.0
LIVE_VISUALIZER_SAMPLE_HZ = 30.0
SESSION_CONTROL_THREAD_JOIN_SECONDS = 1.0
SESSION_WRITER_THREAD_JOIN_SECONDS = 10.0
SESSION_CLOSE_THREAD_JOIN_SECONDS = 0.05
REPLAY_VISUALIZATION_CACHE = "replay_visualization.json"
REPLAY_VISUALIZATION_CACHE_VERSION = 4


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
        default=DEFAULT_RECEIVE_FLUSH_EVERY,
        help=(
            "Flush PC telemetry text files every N messages. "
            f"Default: {DEFAULT_RECEIVE_FLUSH_EVERY}."
        ),
    )
    receive_parser.add_argument(
        "--flush-interval-seconds",
        type=float,
        default=DEFAULT_RECEIVE_FLUSH_INTERVAL_SECONDS,
        help=(
            "Flush PC telemetry text files at least this often while messages arrive. "
            f"Default: {DEFAULT_RECEIVE_FLUSH_INTERVAL_SECONDS:g}s."
        ),
    )
    receive_parser.add_argument(
        "--sample-log-interval-seconds",
        type=float,
        default=DEFAULT_SAMPLE_LOG_INTERVAL_SECONDS,
        help=(
            "Print at most one high-rate sample status line per interval; use 0 to print every sample. "
            f"Default: {DEFAULT_SAMPLE_LOG_INTERVAL_SECONDS:g}s."
        ),
    )
    receive_parser.add_argument(
        "--udp-receive-buffer-bytes",
        type=int,
        default=DEFAULT_UDP_RECEIVE_BUFFER_BYTES,
        help=(
            "Requested UDP socket receive buffer size in bytes. "
            f"Default: {DEFAULT_UDP_RECEIVE_BUFFER_BYTES}."
        ),
    )
    receive_parser.add_argument(
        "--recording-idle-timeout-seconds",
        type=float,
        default=DEFAULT_RECORDING_IDLE_TIMEOUT_SECONDS,
        help=(
            "Close an active PC recording if no more recording datagrams arrive for this many seconds. "
            f"Default: {DEFAULT_RECORDING_IDLE_TIMEOUT_SECONDS:g}s."
        ),
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
        "--max-diverse-detection-frames-per-side",
        type=int,
        default=220,
        help="Max pose-diverse Quest video frames checked for checkerboard per eye after B-button stop. Default: 220",
    )
    receive_parser.add_argument(
        "--max-diverse-fit-frames-per-side",
        type=int,
        default=180,
        help="Max pose-diverse detected Quest frames used by the final optimizer per eye. Default: 180",
    )
    receive_parser.add_argument(
        "--min-diverse-frames-per-side",
        type=int,
        default=28,
        help="Minimum Quest frames retained before early stopping the pose-diverse selector. Default: 28",
    )
    receive_parser.add_argument(
        "--diverse-translation-scale",
        type=float,
        default=0.025,
        help="Quest camera translation difference that counts as one diversity unit, in meters. Default: 0.025",
    )
    receive_parser.add_argument(
        "--diverse-rotation-scale",
        type=float,
        default=3.0,
        help="Quest camera rotation difference that counts as one diversity unit, in degrees. Default: 3.0",
    )
    receive_parser.add_argument(
        "--diverse-min-score",
        type=float,
        default=0.75,
        help="Stop selecting extra Quest frames below this diversity score. Default: 0.75",
    )
    receive_parser.add_argument(
        "--disable-diverse-frame-selection",
        action="store_true",
        help="Use every Quest calibration video frame instead of pose-diverse selection.",
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
    receive_parser.add_argument(
        "--third-realsense-serial",
        default="",
        help="Optional third/static RealSense serial recorded alongside the end-mounted camera.",
    )
    receive_parser.add_argument("--realsense-width", type=int, default=1280, help="RealSense color width. Default: 1280")
    receive_parser.add_argument("--realsense-height", type=int, default=720, help="RealSense color height. Default: 720")
    receive_parser.add_argument("--realsense-fps", type=int, default=30, help="RealSense color FPS. Default: 30")
    receive_parser.add_argument(
        "--no-record-realsense-depth",
        action="store_true",
        help="Disable aligned uint16 depth stream recording for RealSense formal recordings.",
    )
    receive_parser.add_argument(
        "--record-realsense-depth-every-n-frames",
        type=int,
        default=DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES,
        help=(
            "Append RealSense uint16 depth to the indexed depth stream every N RGB frames during formal recordings. "
            f"Default: {DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES}"
        ),
    )
    receive_parser.add_argument(
        "--record-realsense-depth-format",
        choices=("ffv1", "raw"),
        default=DEFAULT_RECORD_DEPTH_FORMAT,
        help=(
            "Depth storage format for formal recordings: ffv1 writes lossless 16-bit MKV, "
            "raw writes the legacy .u16le.bin stream. "
            f"Default: {DEFAULT_RECORD_DEPTH_FORMAT}"
        ),
    )
    receive_parser.add_argument("--realsense-manual-exposure", action="store_true", help="Disable RealSense RGB auto exposure.")
    receive_parser.add_argument("--realsense-exposure", type=float, default=None, help="Manual RealSense RGB exposure value.")
    receive_parser.add_argument("--realsense-gain", type=float, default=None, help="RealSense RGB gain value.")
    receive_parser.add_argument(
        "--robot-state-hz",
        type=float,
        default=DEFAULT_ROBOT_STATE_HZ,
        help=f"Robot state polling Hz for A-button formal recordings. Default: {DEFAULT_ROBOT_STATE_HZ:g}",
    )
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
        "--hand-eye-max-diverse-samples",
        type=int,
        default=DEFAULT_HAND_EYE_MAX_DIVERSE_SAMPLES,
        help=f"Max pose-diverse robot samples checked for hand-eye after B-button stop. Default: {DEFAULT_HAND_EYE_MAX_DIVERSE_SAMPLES}",
    )
    receive_parser.add_argument(
        "--hand-eye-min-diverse-samples",
        type=int,
        default=DEFAULT_HAND_EYE_MIN_DIVERSE_SAMPLES,
        help=f"Minimum pose-diverse robot samples before early stopping. Default: {DEFAULT_HAND_EYE_MIN_DIVERSE_SAMPLES}",
    )
    receive_parser.add_argument(
        "--hand-eye-diverse-translation-scale",
        type=float,
        default=DEFAULT_HAND_EYE_DIVERSE_TRANSLATION_SCALE_M,
        help=(
            "Robot TCP translation difference that counts as one hand-eye diversity unit, in meters. "
            f"Default: {DEFAULT_HAND_EYE_DIVERSE_TRANSLATION_SCALE_M:g}"
        ),
    )
    receive_parser.add_argument(
        "--hand-eye-diverse-rotation-scale",
        type=float,
        default=DEFAULT_HAND_EYE_DIVERSE_ROTATION_SCALE_DEG,
        help=(
            "Robot TCP rotation difference that counts as one hand-eye diversity unit, in degrees. "
            f"Default: {DEFAULT_HAND_EYE_DIVERSE_ROTATION_SCALE_DEG:g}"
        ),
    )
    receive_parser.add_argument(
        "--hand-eye-diverse-min-score",
        type=float,
        default=DEFAULT_HAND_EYE_DIVERSE_MIN_SCORE,
        help=f"Stop selecting extra robot samples below this diversity score. Default: {DEFAULT_HAND_EYE_DIVERSE_MIN_SCORE:g}",
    )
    receive_parser.add_argument(
        "--disable-hand-eye-diverse-selection",
        action="store_true",
        help="Use all robot samples for hand-eye instead of pose-diverse selection.",
    )
    receive_parser.add_argument(
        "--controller-motion-scale",
        type=float,
        default=DEFAULT_CONTROLLER_TRANSLATION_SCALE,
        help=f"Scale from right-controller displacement to robot TCP displacement. Default: {DEFAULT_CONTROLLER_TRANSLATION_SCALE:g}",
    )
    receive_parser.add_argument(
        "--controller-motion-max-offset",
        type=float,
        default=None,
        help="Legacy no-op. Teleop no longer clamps cumulative TCP offset; use --controller-motion-max-step instead.",
    )
    receive_parser.add_argument(
        "--controller-motion-max-step",
        type=float,
        default=DEFAULT_CONTROLLER_MAX_STEP_M,
        help=f"Maximum TCP target position change per received sample, in meters. Default: {DEFAULT_CONTROLLER_MAX_STEP_M:g}",
    )
    receive_parser.add_argument(
        "--controller-motion-max-rotation",
        type=float,
        default=None,
        help="Legacy no-op. Teleop no longer clamps cumulative TCP rotation; use --controller-motion-max-rotation-step instead.",
    )
    receive_parser.add_argument(
        "--controller-motion-max-rotation-step",
        type=float,
        default=DEFAULT_CONTROLLER_MAX_ROTATION_STEP_DEG,
        help=(
            "Maximum TCP target rotation change per received sample, in degrees. "
            f"Default: {DEFAULT_CONTROLLER_MAX_ROTATION_STEP_DEG}"
        ),
    )
    receive_parser.add_argument(
        "--controller-target-update-hz",
        type=float,
        default=DEFAULT_CONTROLLER_TARGET_UPDATE_HZ,
        help=(
            "Maximum rate for updating the robot TCP target from the right controller. "
            f"Default: {DEFAULT_CONTROLLER_TARGET_UPDATE_HZ:g}"
        ),
    )
    receive_parser.add_argument(
        "--controller-joint-limit-buffer",
        type=float,
        default=DEFAULT_CONTROLLER_JOINT_LIMIT_BUFFER_RAD,
        help=(
            "Stop controller TCP commands when any robot joint is within this many radians of the URDF soft limit. "
            f"Default: {DEFAULT_CONTROLLER_JOINT_LIMIT_BUFFER_RAD:g}"
        ),
    )
    receive_parser.add_argument(
        "--disable-controller-joint-limit-guard",
        action="store_true",
        help="Disable the controller teleop joint-limit buffer guard.",
    )
    receive_parser.add_argument(
        "--enable-gripper",
        action="store_true",
        help="Enable Flexiv gripper trigger control during synchronized robot sessions.",
    )
    receive_parser.add_argument(
        "--gripper-device",
        default=DEFAULT_GRIPPER_DEVICE,
        help=f"Flexiv gripper device name passed to Gripper.Enable(). Default: {DEFAULT_GRIPPER_DEVICE}",
    )
    receive_parser.add_argument("--gripper-open-width", type=float, default=DEFAULT_GRIPPER_OPEN_WIDTH_M)
    receive_parser.add_argument("--gripper-close-width", type=float, default=DEFAULT_GRIPPER_CLOSE_WIDTH_M)
    receive_parser.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED_MPS)
    receive_parser.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE_N)
    receive_parser.add_argument(
        "--gripper-init-on-enable",
        action="store_true",
        default=DEFAULT_GRIPPER_INIT_ON_ENABLE,
        help=(
            "Call Gripper.Init() after Gripper.Enable(). Use this when the gripper reports states but "
            "does not respond to Move() until initialized."
        ),
    )
    receive_parser.add_argument(
        "--gripper-trigger-close-threshold",
        type=float,
        default=DEFAULT_GRIPPER_TRIGGER_CLOSE_THRESHOLD,
    )
    receive_parser.add_argument(
        "--gripper-trigger-open-threshold",
        type=float,
        default=DEFAULT_GRIPPER_TRIGGER_OPEN_THRESHOLD,
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

    perf_parser = subparsers.add_parser(
        "audit-performance",
        help="Audit robot/RealSense recording performance for a PC record.",
    )
    perf_parser.add_argument(
        "--pc-session",
        type=Path,
        help="PC session folder. Defaults to the newest record_* folder under --output-root.",
    )
    perf_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Used only when --pc-session is omitted. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    perf_parser.add_argument(
        "--min-robot-target-ratio",
        type=float,
        default=0.80,
        help="Minimum robot state effectiveHz / targetHz ratio. Default: 0.80.",
    )
    perf_parser.add_argument(
        "--min-camera-target-ratio",
        type=float,
        default=0.80,
        help="Minimum camera video effectiveHz / targetHz ratio for each recorded role. Default: 0.80.",
    )
    perf_parser.add_argument(
        "--max-camera-drop-ratio",
        type=float,
        default=0.05,
        help="Maximum queue drop ratio for each camera role. Default: 0.05.",
    )
    perf_parser.add_argument(
        "--max-camera-latency-p95-seconds",
        type=float,
        default=1.0,
        help="Maximum p95 capture-to-write latency for each camera role when available. Default: 1.0.",
    )
    perf_parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the audit JSON.",
    )
    perf_parser.set_defaults(func=audit_performance)

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
        flush_interval_seconds: float,
        calibration_output_root: Path | None = None,
        calibration_raw_root: Path | None = None,
        robot_manager: FlexivRealSenseManager | None = None,
        visualizer: "LiveTelemetryVisualizer | None" = None,
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
                "indexTrigger",
                "handTrigger",
                "indexTriggerPressed",
                "handTriggerPressed",
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
        self.flush_interval_seconds = max(0.0, float(flush_interval_seconds))
        self.last_flush_perf = time.perf_counter()
        self.calibration_output_root = calibration_output_root
        self.calibration_raw_root = calibration_raw_root
        self.calibration_snapshot_start = recording_calibration_snapshot(calibration_output_root)
        self.calibration_snapshot_end: dict[str, Any] | None = None
        self.robot_manager = robot_manager
        self.visualizer = visualizer
        self.robot_session: RobotRealsenseSession | None = None
        self.robot_realsense_directory: Path | None = None
        self.robot_start_status: dict[str, Any] | None = None
        self.lock = threading.RLock()
        self.write_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=SESSION_WRITE_QUEUE_MAX)
        self.writer_error: str | None = None
        self.writer_dropped_messages = 0
        self.writer_dropped_on_enqueue = 0
        self.writer_dropped_on_close = 0
        self.robot_record_sample_calls = 0
        self.robot_record_sample_skips = 0
        self.next_robot_record_perf = 0.0
        self.raw_messages_written = 0
        self.raw_sample_skips = 0
        self.next_raw_sample_perf = 0.0
        self.controller_csv_rows_written = 0
        self.controller_csv_sample_skips = 0
        self.next_controller_csv_perf = 0.0
        self.last_close_metrics: dict[str, Any] | None = None
        self.close_errors: list[dict[str, Any]] = []
        self.control_thread: threading.Thread | None = None
        self.control_stop_event = threading.Event()
        self.control_condition = threading.Condition()
        self.latest_control_wrapper: dict[str, Any] | None = None
        self.latest_control_sequence: Any = None
        self.last_control_sequence: Any = None

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
        self.closing = False
        self.robot_start_status = self._robot_start_status("not_started")
        if self.robot_manager is not None:
            robot_alignment = latest_robot_hand_eye_result(calibration_raw_root, calibration_output_root, output_root)
            self.robot_session = self.robot_manager.start_session(
                self.directory,
                self.record_id,
                self.visualizer.publish_event if self.visualizer is not None else None,
                robot_alignment_result=robot_alignment,
                require_controller_alignment=True,
                control_mode=ROBOT_SESSION_CONTROL_TELEOP,
                record_mode=ROBOT_SESSION_RECORD_ASYNC,
            )
            if self.robot_session is not None:
                self.robot_realsense_directory = self.robot_session.directory
                self.control_thread = threading.Thread(
                    target=self._control_loop,
                    name=f"pc-session-control-{sanitize_name(self.record_id)}",
                    daemon=True,
                )
                self.control_thread.start()
            self.robot_start_status = self._robot_start_status(
                "recording" if self.robot_session is not None else "not_recording"
            )
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"pc-session-writer-{sanitize_name(self.record_id)}",
            daemon=True,
        )
        self.writer_thread.start()

    def write(self, wrapper: dict[str, Any]) -> None:
        if self.closing or self.closed:
            raise RuntimeError("session already closed")
        message = wrapper.get("message")
        if isinstance(message, dict) and message.get("type") == "sample":
            self._publish_latest_control_sample(wrapper)
        self._enqueue_write(wrapper)

    def _enqueue_write(self, wrapper: dict[str, Any]) -> None:
        try:
            self.write_queue.put_nowait(wrapper)
            return
        except queue.Full:
            pass

        if self._drop_one_queued_write(reason="enqueue"):
            try:
                self.write_queue.put_nowait(wrapper)
                return
            except queue.Full:
                pass

        self.writer_dropped_messages += 1
        self.writer_dropped_on_enqueue += 1

    def _drop_one_queued_write(self, reason: str) -> bool:
        try:
            dropped = self.write_queue.get_nowait()
        except queue.Empty:
            return False
        try:
            if dropped is not None:
                self.writer_dropped_messages += 1
                if reason == "close":
                    self.writer_dropped_on_close += 1
                else:
                    self.writer_dropped_on_enqueue += 1
        finally:
            self.write_queue.task_done()
        return True

    def _wait_for_writer_queue(self, timeout_seconds: float) -> int:
        deadline = time.perf_counter() + max(0.0, float(timeout_seconds))
        with self.write_queue.all_tasks_done:
            while self.write_queue.unfinished_tasks:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.write_queue.all_tasks_done.wait(timeout=remaining)
            return int(self.write_queue.unfinished_tasks)

    def _drain_writer_queue_for_close(self) -> int:
        dropped = 0
        while self._drop_one_queued_write(reason="close"):
            dropped += 1
        return dropped

    def _enqueue_writer_sentinel(self) -> None:
        while True:
            try:
                self.write_queue.put_nowait(None)
                return
            except queue.Full:
                if not self._drop_one_queued_write(reason="close"):
                    time.sleep(0.001)

    def _writer_loop(self) -> None:
        while True:
            wrapper = self.write_queue.get()
            try:
                if wrapper is None:
                    return
                self._write_now(wrapper)
            except Exception as exc:  # pragma: no cover - background safety net
                self.writer_error = str(exc)
            finally:
                self.write_queue.task_done()

    def _write_now(self, wrapper: dict[str, Any]) -> None:
        if self.closed:
            return

        message = wrapper.get("message")
        if not isinstance(message, dict):
            return

        output_directory = quest_output_directory(message)
        if output_directory:
            self.quest_output_directory = output_directory

        self.messages += 1
        if self._should_write_raw_wrapper(wrapper, message):
            self.raw_file.write(json_line(wrapper))
            self.raw_messages_written += 1

        if message.get("type") == "sample":
            self.samples += 1
            sample_index = message.get("sampleIndex")
            if is_number(sample_index):
                self.last_sample_index = int(sample_index)
            robot_row = self._write_robot_sample_for_record(wrapper, message)
            self._write_sample(wrapper, message, robot_row)

        now_perf = time.perf_counter()
        if (
            self.messages % self.flush_every == 0
            or (
                self.flush_interval_seconds > 0
                and now_perf - self.last_flush_perf >= self.flush_interval_seconds
            )
        ):
            self.flush()

    def _should_write_raw_wrapper(self, wrapper: dict[str, Any], message: dict[str, Any]) -> bool:
        if message.get("type") != "sample":
            return True
        now_perf = wrapper.get("pcReceivePerfCounterSeconds")
        now_perf = float(now_perf) if is_number(now_perf) else time.perf_counter()
        if now_perf < self.next_raw_sample_perf:
            self.raw_sample_skips += 1
            return False
        self.next_raw_sample_perf = now_perf + (1.0 / max(1.0, SESSION_RAW_SAMPLE_HZ))
        return True

    def _publish_latest_control_sample(self, wrapper: dict[str, Any]) -> None:
        if self.robot_session is None or self.control_thread is None:
            return
        message = wrapper.get("message")
        if not isinstance(message, dict):
            return
        with self.control_condition:
            self.latest_control_wrapper = wrapper
            self.latest_control_sequence = message.get("sequence")
            self.control_condition.notify()

    def _control_loop(self) -> None:
        while not self.control_stop_event.is_set():
            with self.control_condition:
                self.control_condition.wait(timeout=0.02)
                wrapper = self.latest_control_wrapper
                sequence = self.latest_control_sequence
            if wrapper is None or sequence == self.last_control_sequence:
                continue
            self.last_control_sequence = sequence
            message = wrapper.get("message")
            if not isinstance(message, dict) or message.get("type") != "sample":
                continue
            robot_session = self.robot_session
            if robot_session is None:
                continue
            robot_sample = dict(message)
            robot_sample["pcReceivePerfCounterSeconds"] = wrapper.get("pcReceivePerfCounterSeconds")
            try:
                robot_session.update_gripper(robot_sample)
                robot_session.update_controller_motion(robot_sample)
            except Exception as exc:  # pragma: no cover - hardware path
                self.writer_error = str(exc)

    def _write_sample(
        self,
        wrapper: dict[str, Any],
        message: dict[str, Any],
        robot_row: dict[str, Any] | None = None,
    ) -> None:
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
        robot_compact = self._compact_robot_row(robot_row)
        if robot_compact is not None:
            compact["robot"] = robot_compact
            compact["robotJointPose"] = robot_compact.get("jointpose")
            compact["robotJointPos"] = robot_compact.get("jointpos")
        pc_world = pc_world_sample_payload(message)
        if pc_world is not None:
            compact["pcWorld"] = pc_world
        self.samples_file.write(json_line(compact))
        write_controller_csv = self._should_write_controller_csv(wrapper)
        if not write_controller_csv:
            self.controller_csv_sample_skips += 1
        self._write_controller_csv_row(wrapper, message, "left", write_csv=write_controller_csv)
        self._write_controller_csv_row(wrapper, message, "right", write_csv=write_controller_csv)

    def _should_write_controller_csv(self, wrapper: dict[str, Any]) -> bool:
        now_perf = wrapper.get("pcReceivePerfCounterSeconds")
        now_perf = float(now_perf) if is_number(now_perf) else time.perf_counter()
        if now_perf < self.next_controller_csv_perf:
            return False
        self.next_controller_csv_perf = now_perf + (1.0 / max(1.0, SESSION_CONTROLLER_CSV_HZ))
        return True

    def _write_robot_sample_for_record(self, wrapper: dict[str, Any], message: dict[str, Any]) -> dict[str, Any] | None:
        robot_session = self.robot_session
        if robot_session is None:
            return None
        if getattr(robot_session, "record_mode", None) == ROBOT_SESSION_RECORD_ASYNC:
            now_perf = wrapper.get("pcReceivePerfCounterSeconds")
            now_perf = float(now_perf) if is_number(now_perf) else time.perf_counter()
            if now_perf < self.next_robot_record_perf:
                self.robot_record_sample_skips += 1
                return None
            self.next_robot_record_perf = now_perf + (1.0 / max(1.0, ROBOT_SESSION_ASYNC_SAMPLE_HZ))
        robot_sample = dict(message)
        robot_sample["pcReceivePerfCounterSeconds"] = wrapper.get("pcReceivePerfCounterSeconds")
        self.robot_record_sample_calls += 1
        return robot_session.record_sample(robot_sample)

    @staticmethod
    def _compact_robot_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
        joint_pose = row.get("jointpose")
        if joint_pose is None:
            joint_pose = row.get("jointpos")
        if joint_pose is None:
            joint_pose = state.get("jointPose") or state.get("jointpos")
        return {
            "sampleIndex": row.get("sample_index"),
            "recordId": row.get("record_id"),
            "capturedAt": row.get("captured_at"),
            "pcPerfCounterSeconds": row.get("pc_perf_counter_seconds"),
            "pcUnixSeconds": row.get("pc_unix_seconds"),
            "ok": bool(row.get("ok")),
            "error": row.get("error"),
            "coordinateFrame": row.get("coordinate_frame") or PC_WORLD_FRAME,
            "poseSource": row.get("pose_source"),
            "questSampleIndex": row.get("quest_sample_index"),
            "questRecordingTimestampSeconds": row.get("quest_recording_timestamp_seconds"),
            "questPcReceivePerfCounterSeconds": row.get("quest_pc_receive_perf_counter_seconds"),
            "robotStateSampleIndex": row.get("robot_state_sample_index"),
            "robotStateCapturedAt": row.get("robot_state_captured_at"),
            "robotStatePcPerfCounterSeconds": row.get("robot_state_pc_perf_counter_seconds"),
            "jointpose": joint_pose,
            "jointpos": joint_pose,
            "rawEndEffectorPoseWxyz": state.get("rawEndEffectorPoseWxyz"),
            "T_base_ee": row.get("T_base_ee"),
            "T_base_tool_tcp": row.get("T_base_tool_tcp"),
            "T_world_tool_tcp": row.get("T_world_tool_tcp"),
            "T_display_tool_tcp": row.get("T_display_tool_tcp"),
            "T_base_end_camera": row.get("T_base_end_camera"),
            "T_world_end_camera": row.get("T_world_end_camera"),
            "T_display_end_camera": row.get("T_display_end_camera"),
            "gripper": row.get("gripper"),
            "videoFrames": row.get("videoFrames"),
        }

    def _write_controller_csv_row(
        self,
        wrapper: dict[str, Any],
        message: dict[str, Any],
        hand: str,
        *,
        write_csv: bool = True,
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

        if not write_csv:
            return
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
                "indexTrigger": controller.get("indexTrigger"),
                "handTrigger": controller.get("handTrigger"),
                "indexTriggerPressed": controller.get("indexTriggerPressed"),
                "handTriggerPressed": controller.get("handTriggerPressed"),
                "x": x,
                "y": y,
                "z": z,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
            }
        )
        self.controller_csv_rows_written += 1

    def flush(self) -> None:
        self.raw_file.flush()
        self.samples_file.flush()
        self.controllers_file.flush()
        self.last_flush_perf = time.perf_counter()

    def close(self, reason: str) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return self.summary(reason)
            self.closing = True

        close_started_perf = time.perf_counter()
        unfinished_before_drop = 0
        dropped_on_close = 0
        robot_summary = None
        try:
            unfinished_before_drop = self._wait_for_writer_queue(SESSION_WRITE_QUEUE_CLOSE_DRAIN_SECONDS)
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error("writer_queue_wait", exc)
        writer_drained_perf = time.perf_counter()
        try:
            dropped_on_close = self._drain_writer_queue_for_close() if unfinished_before_drop else 0
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error("writer_queue_drain", exc)
        writer_queue_dropped_perf = time.perf_counter()
        try:
            self._enqueue_writer_sentinel()
            self.writer_thread.join(timeout=SESSION_WRITER_THREAD_JOIN_SECONDS)
            if self.writer_thread.is_alive():
                self._note_close_error(
                    "writer_thread_alive",
                    RuntimeError("PC session writer thread did not stop before files were closed"),
                )
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error("writer_thread_join", exc)
        writer_joined_perf = time.perf_counter()
        try:
            self.control_stop_event.set()
            with self.control_condition:
                self.control_condition.notify_all()
            if self.control_thread is not None and self.control_thread.is_alive():
                self.control_thread.join(timeout=SESSION_CONTROL_THREAD_JOIN_SECONDS)
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error("control_thread_join", exc)
        control_joined_perf = time.perf_counter()

        try:
            self.calibration_snapshot_end = recording_calibration_snapshot(self.calibration_output_root)
            preferred_snapshot = self.preferred_calibration_snapshot()
            if preferred_snapshot is not None:
                self.calibration_snapshot_path.write_text(
                    json.dumps(preferred_snapshot, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error("calibration_snapshot", exc)
        self._safe_flush_close_file("pc_telemetry_raw", self.raw_file)
        self._safe_flush_close_file("pc_samples", self.samples_file)
        self._safe_flush_close_file("pc_controllers", self.controllers_file)
        robot_session = self.robot_session
        self.robot_session = None
        if self.robot_manager is not None and robot_session is not None:
            if robot_session is not None:
                self.robot_realsense_directory = robot_session.directory
            try:
                robot_summary = self.robot_manager.stop_session(robot_session)
            except Exception as exc:  # pragma: no cover - hardware/close path
                self._note_close_error("robot_session_stop", exc)
        robot_stopped_perf = time.perf_counter()
        self.last_close_metrics = {
            "writerDrainSeconds": writer_drained_perf - close_started_perf,
            "writerQueueDropSeconds": writer_queue_dropped_perf - writer_drained_perf,
            "writerJoinSeconds": writer_joined_perf - writer_queue_dropped_perf,
            "controlJoinSeconds": control_joined_perf - writer_joined_perf,
            "robotStopSeconds": robot_stopped_perf - control_joined_perf,
            "totalSeconds": robot_stopped_perf - close_started_perf,
            "writerUnfinishedTasksBeforeDrop": unfinished_before_drop,
            "writerDroppedOnClose": dropped_on_close,
        }
        summary = self.summary(reason)
        if robot_summary is not None:
            summary["robotRealSense"] = robot_summary
        self._write_summary_safely(summary, "summary")
        try:
            cache_path = write_replay_visualization_cache(
                self.directory,
                self.record_id,
                source="pc",
                calibration_raw_root=self.calibration_raw_root,
                calibration_output_root=self.calibration_output_root,
            )
            summary["replayVisualizationJson"] = str(cache_path)
        except Exception as exc:
            summary["replayVisualizationError"] = str(exc)
            self._note_close_error("replay_visualization_cache", exc)
        self._write_summary_safely(summary, "summary_final")
        self.closed = True
        return summary

    def _note_close_error(self, stage: str, exc: Exception) -> None:
        payload = {
            "stage": stage,
            "type": type(exc).__name__,
            "error": str(exc),
        }
        self.close_errors.append(payload)
        print(
            f"[session-close-error] record={self.record_id} stage={stage} {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    def _safe_flush_close_file(self, label: str, handle: Any) -> None:
        try:
            handle.flush()
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error(f"{label}_flush", exc)
        try:
            handle.close()
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error(f"{label}_close", exc)

    def _write_summary_safely(self, summary: dict[str, Any], stage: str) -> None:
        if self.close_errors:
            summary["closeErrors"] = list(self.close_errors)
        try:
            atomic_write_json(self.summary_path, summary)
        except Exception as exc:  # pragma: no cover - defensive close path
            self._note_close_error(stage, exc)

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
            "rawMessagesWritten": self.raw_messages_written,
            "rawSampleSkips": self.raw_sample_skips,
            "controllerCsvRowsWritten": self.controller_csv_rows_written,
            "controllerCsvSampleSkips": self.controller_csv_sample_skips,
            "leftPoseSamples": self.left_pose_samples,
            "rightPoseSamples": self.right_pose_samples,
            "leftSources": self.left_sources,
            "rightSources": self.right_sources,
            "leftMissingReasons": self.left_missing_reasons,
            "rightMissingReasons": self.right_missing_reasons,
            "rawJsonl": str(self.raw_path),
            "samplesJsonl": str(self.samples_path),
            "controllersCsv": str(self.controllers_csv_path),
            "robotRealSenseDirectory": str(self.robot_realsense_directory) if self.robot_realsense_directory is not None else None,
            "robotStartStatus": self.robot_start_status,
            "writerQueueMax": SESSION_WRITE_QUEUE_MAX,
            "writerQueueBacklog": self.write_queue.qsize(),
            "writerDroppedMessages": self.writer_dropped_messages,
            "writerDroppedOnEnqueue": self.writer_dropped_on_enqueue,
            "writerDroppedOnClose": self.writer_dropped_on_close,
            "robotRecordSampleCalls": self.robot_record_sample_calls,
            "robotRecordSampleSkips": self.robot_record_sample_skips,
            "writerError": self.writer_error,
        }
        if self.close_errors:
            summary["closeErrors"] = list(self.close_errors)
        if self.last_close_metrics is not None:
            summary["closeMetrics"] = dict(self.last_close_metrics)
        if self.calibration_snapshot_start is not None:
            summary["calibrationSnapshotAtStart"] = self.calibration_snapshot_start
        if self.calibration_snapshot_end is not None:
            summary["calibrationSnapshotAtEnd"] = self.calibration_snapshot_end
        preferred_snapshot = self.preferred_calibration_snapshot()
        if preferred_snapshot is not None:
            summary["calibrationSnapshot"] = preferred_snapshot
            summary["calibrationSnapshotJson"] = str(self.calibration_snapshot_path)
        return summary

    def _robot_start_status(self, stage: str) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"enabled": False, "stage": stage, "recording": False, "reason": "robot manager disabled"}
        status = self.robot_manager.status()
        robot = status.get("robot") if isinstance(status, dict) else {}
        config = status.get("config") if isinstance(status, dict) else {}
        active = status.get("activeSession") if isinstance(status, dict) else None
        connected = bool(isinstance(robot, dict) and robot.get("connected"))
        motion_armed = bool(isinstance(robot, dict) and robot.get("motionArmed"))
        freedrive_enabled = bool(
            isinstance(robot, dict) and (robot.get("freedriveEnabled") or robot.get("freeDragEnabled"))
        )
        freedrive_method = robot.get("freedriveMethod") or robot.get("freeDragMethod") if isinstance(robot, dict) else None
        freedrive_plan = robot.get("freedrivePlan") or robot.get("freeDragPlan") if isinstance(robot, dict) else None
        freedrive_loop_alive = bool(
            isinstance(robot, dict)
            and (
                robot.get("freedriveLoopAlive")
                or robot.get("freeDragLoopAlive")
                or robot.get("cartesianControlLoopAlive")
            )
        )
        freedrive_last_error = (
            robot.get("freedriveLastError") or robot.get("freeDragLastError") if isinstance(robot, dict) else None
        )
        freedrive_send_signature = robot.get("cartesianSendSignature") if isinstance(robot, dict) else None
        controller_motion = bool(isinstance(config, dict) and config.get("controllerMotionEnabled"))
        control_mode = active.get("controlMode") if isinstance(active, dict) else ROBOT_SESSION_CONTROL_TELEOP
        if control_mode not in (ROBOT_SESSION_CONTROL_TELEOP, ROBOT_SESSION_CONTROL_FREEDRIVE):
            control_mode = ROBOT_SESSION_CONTROL_TELEOP
        recording = bool(self.robot_session is not None)
        reason = "recording"
        if not connected:
            reason = "Flexiv robot is not connected"
        elif not isinstance(config, dict) or not config.get("cameraSerial"):
            reason = "RealSense camera serial is empty"
        elif not recording:
            reason = self.robot_manager.last_error or "robot RealSense session did not start"
        elif control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE and freedrive_enabled:
            reason = "recording; robot free-drag mode is enabled"
        elif control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE:
            reason = self.robot_manager.last_error or "robot free-drag mode is not enabled"
        elif active and active.get("controllerAlignmentRequired") and not active.get("controllerAlignmentAvailable"):
            reason = "recording; Quest-robot alignment is required before right-controller teleop"
        elif control_mode == ROBOT_SESSION_CONTROL_TELEOP and not controller_motion:
            reason = self.robot_manager.last_error or "controller teleop is disabled"
        elif not motion_armed:
            reason = "recording; hold right middle-finger trigger to teleoperate"
        else:
            reason = "recording; hold right middle-finger trigger to teleoperate"
        return {
            "enabled": True,
            "stage": stage,
            "recording": recording,
            "reason": reason,
            "robotConnected": connected,
            "robotSn": robot.get("robotSn") if isinstance(robot, dict) else None,
            "poseField": robot.get("poseField") if isinstance(robot, dict) else None,
            "cameraSerial": config.get("cameraSerial") if isinstance(config, dict) else None,
            "thirdCameraSerial": config.get("thirdCameraSerial") if isinstance(config, dict) else None,
            "motionArmed": motion_armed,
            "freedriveEnabled": freedrive_enabled,
            "freeDragEnabled": freedrive_enabled,
            "freedriveMethod": freedrive_method,
            "freeDragMethod": freedrive_method,
            "freedrivePlan": freedrive_plan,
            "freeDragPlan": freedrive_plan,
            "freedriveLoopAlive": freedrive_loop_alive,
            "freeDragLoopAlive": freedrive_loop_alive,
            "cartesianControlLoopAlive": freedrive_loop_alive,
            "freedriveLastError": freedrive_last_error,
            "freeDragLastError": freedrive_last_error,
            "cartesianSendSignature": freedrive_send_signature,
            "controlMode": control_mode,
            "controllerMotionEnabled": controller_motion,
            "teleopRequiresRightSideButton": control_mode == ROBOT_SESSION_CONTROL_TELEOP,
            "teleopRequiresRightHandTrigger": control_mode == ROBOT_SESSION_CONTROL_TELEOP,
            "gripperEnabled": bool(isinstance(config, dict) and config.get("gripperEnabled")),
            "activeSession": active,
        }

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
        recording_root: Path | None = None,
        calibration_raw_root: Path | None = None,
        calibration_output_root: Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.history_limit = max(1, history_limit)
        self.recording_root = (recording_root or DEFAULT_OUTPUT_ROOT).resolve()
        self.calibration_raw_root = (calibration_raw_root or DEFAULT_QUEST_LOCAL_ROOT).resolve()
        self.calibration_output_root = (calibration_output_root or DEFAULT_CALIBRATION_OUTPUT_ROOT).resolve()
        self.robot_manager = robot_manager
        self.adb_path = adb_path
        self.history: list[dict[str, Any]] = []
        self.clients: list[queue.Queue[str | None]] = []
        self.lock = threading.Lock()
        self.calibration_state_cleared = False
        self.capture_state: dict[str, Any] = {
            "phase": "live",
            "recordId": None,
            "detail": "not recording",
            "sinceUnixSeconds": time.time(),
            "updatedUtc": datetime.now(timezone.utc).isoformat(),
            "source": "pc_receiver",
        }
        self.udp_status: dict[str, Any] = {
            "listening": False,
            "bind": None,
            "totalDatagrams": 0,
            "sampleDatagrams": 0,
            "recordingDatagrams": 0,
            "lastReceiveUnixSeconds": None,
            "lastReceiveUtc": None,
            "lastRemote": None,
            "lastType": None,
            "lastRecordId": None,
            "lastSampleIndex": None,
            "lastIsRecording": None,
            "lastBytes": None,
            "receiveBufferBytes": None,
        }
        self.next_sample_publish_perf = 0.0
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
        if message.get("type") == "sample" and not self._claim_sample_publish_slot(wrapper):
            return
        event = visualizer_event_from_message(message, wrapper)
        if event is None:
            return
        self.publish_event(event)

    def _claim_sample_publish_slot(self, wrapper: dict[str, Any]) -> bool:
        now_perf = wrapper.get("pcReceivePerfCounterSeconds")
        now_perf = float(now_perf) if is_number(now_perf) else time.perf_counter()
        period = 1.0 / max(1.0, LIVE_VISUALIZER_SAMPLE_HZ)
        with self.lock:
            if now_perf < self.next_sample_publish_perf:
                return False
            self.next_sample_publish_perf = now_perf + period
            return True

    def set_udp_listener(self, host: str, port: int, receive_buffer_bytes: int | None = None) -> None:
        with self.lock:
            self.udp_status["listening"] = True
            self.udp_status["bind"] = f"{host}:{port}"
            self.udp_status["receiveBufferBytes"] = receive_buffer_bytes

    def note_udp_datagram(self, message: dict[str, Any], wrapper: dict[str, Any], byte_count: int) -> None:
        msg_type = str(message.get("type") or "?")
        is_sample = msg_type == "sample"
        is_recording = bool(message.get("isRecording", True)) if is_sample else msg_type in ("recording_start", "recording_stop")
        with self.lock:
            self.udp_status["totalDatagrams"] = int(self.udp_status.get("totalDatagrams") or 0) + 1
            if is_sample:
                self.udp_status["sampleDatagrams"] = int(self.udp_status.get("sampleDatagrams") or 0) + 1
            if is_recording:
                self.udp_status["recordingDatagrams"] = int(self.udp_status.get("recordingDatagrams") or 0) + 1
            self.udp_status["lastReceiveUnixSeconds"] = wrapper.get("pcReceiveUnixSeconds")
            self.udp_status["lastReceiveUtc"] = wrapper.get("pcReceiveUtc")
            self.udp_status["lastRemote"] = wrapper.get("remote")
            self.udp_status["lastType"] = msg_type
            self.udp_status["lastRecordId"] = message.get("recordId")
            self.udp_status["lastSampleIndex"] = message.get("sampleIndex")
            self.udp_status["lastIsRecording"] = is_recording
            self.udp_status["lastBytes"] = int(byte_count)

    def set_capture_state(self, phase: str, record_id: str | None, detail: str, **extra: Any) -> dict[str, Any]:
        event = {
            "type": "capture_state",
            "phase": str(phase or "live"),
            "recordId": str(record_id) if record_id else None,
            "detail": str(detail or ""),
            "sinceUnixSeconds": time.time(),
            "updatedUtc": datetime.now(timezone.utc).isoformat(),
            "source": "pc_receiver",
            **extra,
        }
        with self.lock:
            self.capture_state = dict(event)
        self.publish_event(event)
        return event

    def capture_state_payload(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.capture_state)

    def publish_event(self, event: dict[str, Any]) -> None:
        if event.get("type") in ("calibration_result", "robot_calibration_result"):
            self.calibration_state_cleared = False
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

    def robot_status_payload(self, *, lightweight: bool = False) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        payload = self.robot_manager.status(lightweight=lightweight)
        if not lightweight and not self.calibration_state_cleared and not isinstance(payload.get("lastCalibration"), dict):
            latest = latest_robot_calibration_event(self.calibration_raw_root, self.calibration_output_root, self.recording_root)
            if latest is not None:
                payload["lastCalibration"] = latest
        return payload

    def clear_calibration_payload(self) -> dict[str, Any]:
        self.calibration_state_cleared = True
        result = clear_latest_calibration_state(
            self.calibration_raw_root,
            self.calibration_output_root,
            self.robot_manager,
        )
        self.publish_event({"type": "calibration_cleared", **result})
        self.publish_event({"type": "robot_calibration_cleared", **result})
        return result

    def camera_list_payload(
        self,
        *,
        lightweight: bool = False,
        robot_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled", "cameras": []}
        if lightweight:
            status = robot_status if isinstance(robot_status, dict) else self.robot_manager.status(lightweight=True)
            stream = status.get("realsenseStream") if isinstance(status, dict) else {}
            metadata = stream.get("metadata") if isinstance(stream, dict) and isinstance(stream.get("metadata"), dict) else {}
            cameras = []
            for role, row in metadata.items():
                if not isinstance(row, dict):
                    continue
                cameras.append(
                    {
                        "serial": row.get("serial"),
                        "name": row.get("name") or row.get("productLine") or role,
                        "role": role,
                        "cached": True,
                    }
                )
            return {"ok": True, "enabled": True, "cameras": cameras, "cached": True}
        return self.robot_manager.list_cameras()

    def start_realsense_stream_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        self.robot_manager.configure(payload)
        result = self.robot_manager.start_realsense_stream()
        self.publish_event({"type": "robot_status", "stage": "realsense_stream_start", **result})
        return result

    def stop_realsense_stream_payload(self) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        result = self.robot_manager.stop_realsense_stream()
        self.publish_event({"type": "robot_status", "stage": "realsense_stream_stop", **result})
        return result

    def robot_end_camera_frame_payload(self, overlay: bool = False) -> tuple[bytes, dict[str, Any]]:
        return self.robot_camera_frame_payload("end", overlay=overlay)

    def robot_camera_frame_payload(self, role: str = "end", overlay: bool = False) -> tuple[bytes, dict[str, Any]]:
        if self.robot_manager is None:
            raise RuntimeError("Flexiv/RealSense support is disabled")
        role = str(role or "end").strip().lower()
        if overlay and role == "end":
            return self.robot_manager.capture_end_camera_overlay_jpeg()
        return self.robot_manager.capture_camera_preview_jpeg(role)

    def preflight_status_payload(self) -> dict[str, Any]:
        with self.lock:
            recent_samples = [event for event in self.history if event.get("type") == "sample"]
            last_sample = recent_samples[-1] if recent_samples else None
            controller_window = recent_controller_status(recent_samples)
            udp_status = dict(self.udp_status)
            capture_state = dict(self.capture_state)
        capture_phase = str(capture_state.get("phase") or "live") if isinstance(capture_state, dict) else "live"
        lightweight = capture_phase in ("recording", "saving")
        robot_status = self.robot_status_payload(lightweight=lightweight)
        camera_status = self.camera_list_payload(lightweight=lightweight, robot_status=robot_status)
        board_status = self.latest_robot_board_check_payload()
        model_status = rizon4_model_payload()
        return build_preflight_status(
            last_sample,
            controller_window,
            robot_status,
            camera_status,
            board_status,
            model_status,
            udp_status,
            capture_state,
        )

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

    def robot_gripper_status_payload(self) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        return self.robot_manager.gripper_status()

    def robot_gripper_move_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.robot_manager is None:
            return {"ok": False, "enabled": False, "reason": "disabled"}
        self.robot_manager.configure(payload)
        try:
            result = self.robot_manager.move_gripper(payload)
        except Exception as exc:
            result = {
                "ok": False,
                "enabled": True,
                "commandSent": False,
                "action": str(payload.get("action") or "move"),
                "targetWidthM": payload.get("widthM"),
                "error": str(exc),
            }
        self.publish_event({"type": "robot_gripper", **result})
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
                "status": self.robot_manager.status(),
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
                    self._send_json(latest_calibration_snapshot(visualizer.calibration_output_root))
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
                if parsed.path == "/robot/gripper/status":
                    self._send_json(visualizer.robot_gripper_status_payload())
                    return
                if parsed.path == "/robot/diagnostics":
                    self._send_json(visualizer.robot_diagnostics_payload({}))
                    return
                if parsed.path == "/robot/board-check/latest":
                    self._send_json(visualizer.latest_robot_board_check_payload())
                    return
                if parsed.path == "/robot/end-camera/frame":
                    self._send_robot_camera_frame(parsed, default_role="end")
                    return
                if parsed.path == "/robot/camera-frame":
                    self._send_robot_camera_frame(parsed, default_role="end")
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
                if parsed.path == "/robot/mesh":
                    self._send_robot_mesh(parsed)
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
                    if parsed.path == "/robot/gripper/move":
                        self._send_json(visualizer.robot_gripper_move_payload(payload))
                        return
                    if parsed.path == "/robot/board-check":
                        self._send_json(visualizer.robot_board_check_payload(payload))
                        return
                    if parsed.path == "/robot/realsense-stream/start":
                        self._send_json(visualizer.start_realsense_stream_payload(payload))
                        return
                    if parsed.path == "/robot/realsense-stream/stop":
                        self._send_json(visualizer.stop_realsense_stream_payload())
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
                    if parsed.path == "/calibration/clear":
                        self._send_json(visualizer.clear_calibration_payload())
                        return
                    self.send_error(404)
                except Exception as exc:
                    self.send_error(500, str(exc))

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _write_response_body(self, data: bytes) -> bool:
                try:
                    self.wfile.write(data)
                    return True
                except (BrokenPipeError, ConnectionError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
                    return False

            def _send_html(self) -> None:
                html = LIVE_VIEWER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self._write_response_body(html)

            def _send_recordings_html(self) -> None:
                html = RECORDINGS_REPLAY_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self._write_response_body(html)

            def _send_json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self._write_response_body(data)

            def _send_robot_camera_frame(self, parsed: Any, default_role: str = "end") -> None:
                # Keep live preview on the shared stream hub, but do not run per-frame
                # checkerboard overlay here; OpenCV detection is too heavy for live UI.
                query = parse_qs(parsed.query)
                role = (first_query(query, "role") or default_role or "end").strip().lower()
                overlay = False
                try:
                    data, meta = visualizer.robot_camera_frame_payload(role=role, overlay=overlay)
                except Exception as exc:
                    self.send_error(503, str(exc))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                if meta.get("createdAtUtc"):
                    self.send_header("X-Frame-Created-At-Utc", str(meta["createdAtUtc"]))
                if meta.get("sequence") is not None:
                    self.send_header("X-Frame-Sequence", str(meta["sequence"]))
                if meta.get("role"):
                    self.send_header("X-Camera-Role", str(meta["role"]))
                camera = meta.get("camera") if isinstance(meta.get("camera"), dict) else {}
                if camera.get("serial") is not None:
                    self.send_header("X-Camera-Serial", str(camera["serial"]))
                board = meta.get("board") if isinstance(meta.get("board"), dict) else {}
                if board:
                    self.send_header("X-Checkerboard-Ok", "1" if board.get("ok") else "0")
                    self.send_header("X-Checkerboard-Corners", str(board.get("detectedCorners") or 0))
                    if board.get("method"):
                        self.send_header("X-Checkerboard-Method", str(board["method"]))
                self.end_headers()
                self._write_response_body(data)

            def _send_recording_replay_json(self, parsed: Any) -> None:
                query = parse_qs(parsed.query)
                record_id = first_query(query, "recordId")
                if not record_id:
                    self.send_error(400, "missing recordId")
                    return
                try:
                    source = first_query(query, "source")
                    compact = first_query(query, "compact")
                    full = first_query(query, "full")
                    use_compact = not (
                        str(compact or "").lower() in ("0", "false", "no")
                        or str(full or "").lower() in ("1", "true", "yes")
                    )
                    payload = build_recording_replay_payload(
                        visualizer.recording_root,
                        record_id,
                        source=source,
                        calibration_raw_root=visualizer.calibration_raw_root,
                        calibration_output_root=visualizer.calibration_output_root,
                        compact=use_compact,
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
                content_type = artifact_content_type(suffix)
                try:
                    send_http_path(self, path, content_type)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{path.stat().st_size}")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()

            def _send_urdf(self) -> None:
                if not DEFAULT_RIZON_URDF.exists():
                    self.send_error(404, "Rizon URDF asset not found")
                    return
                data = DEFAULT_RIZON_URDF.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/xml; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self._write_response_body(data)

            def _send_robot_mesh(self, parsed: Any) -> None:
                query = parse_qs(parsed.query)
                path_text = first_query(query, "path")
                if not path_text:
                    self.send_error(400, "missing path")
                    return
                try:
                    path = resolve_robot_mesh_path(path_text)
                except ValueError as exc:
                    self.send_error(403, str(exc))
                    return
                if not path.exists() or not path.is_file():
                    self.send_error(404, "robot mesh not found")
                    return
                suffix = path.suffix.lower()
                if suffix not in (".obj", ".mtl"):
                    self.send_error(403, "unsupported robot mesh type")
                    return
                size = path.stat().st_size
                if size > MAX_ARTIFACT_BYTES:
                    self.send_error(413, "robot mesh too large")
                    return
                send_http_path(self, path, "text/plain; charset=utf-8")

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
                self._write_response_body(f"data: {payload}\n\n".encode("utf-8"))

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
        calibration_options: dict[str, Any] | None = None,
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
        self.calibration_options = dict(calibration_options or {})
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
        self.video_writers: dict[str, ColorStreamWriter] = {}
        self.frame_counts = {"left": 0, "right": 0}
        self.legacy_flip_vertical_params: set[bool] = set()
        self.applied_vertical_flips: set[bool] = set()
        self.pc_receiver_flip_policies: set[str] = set()
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
                require_controller_alignment=False,
                control_mode=ROBOT_SESSION_CONTROL_FREEDRIVE,
            )
            if self.robot_session is not None:
                self.robot_realsense_directory = self.robot_session.directory
            self.robot_start_status = self._robot_start_status("recording" if self.robot_session is not None else "not_recording")
        self.metadata["robotStartStatus"] = self.robot_start_status
        self.metadata["videoFrameConvention"] = "opencv_top_left_y_down"
        self.metadata["pcReceiverVideoOrientation"] = "canonical_top_left_y_down"
        self.metadata["pcReceiverFlipPolicy"] = (
            "pcReceiverFlipVertical query param when present; otherwise invert legacy Unity flipVertical "
            "for PC calibration JPEG frames"
        )
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
            legacy_flip_vertical = bool_param(query, "flipVertical", True)
            explicit_receiver_flip = first_query(query, "pcReceiverFlipVertical")
            if explicit_receiver_flip is None:
                applied_vertical_flip = not legacy_flip_vertical
                flip_policy = "legacy_flipVertical_inverted"
            else:
                applied_vertical_flip = bool_param(query, "pcReceiverFlipVertical", False)
                flip_policy = "explicit_pcReceiverFlipVertical"
            pose = float_list_param(query, "pose")

            array = np.frombuffer(body, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("could not decode JPEG frame")
            if applied_vertical_flip:
                frame = cv2.flip(frame, 0)
            if width > 0 and height > 0 and (frame.shape[1] != width or frame.shape[0] != height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            self.legacy_flip_vertical_params.add(bool(legacy_flip_vertical))
            self.applied_vertical_flips.add(bool(applied_vertical_flip))
            self.pc_receiver_flip_policies.add(flip_policy)
            writer = self._video_writer(side, frame.shape[1], frame.shape[0])
            writer.write(frame)
            row = {
                "frameIndex": frame_index,
                "cameraTimestampSeconds": camera_ts,
                "unityTimestampSeconds": unity_ts,
                "pose": pose,
                "legacyFlipVerticalParam": legacy_flip_vertical,
                "appliedVerticalFlip": applied_vertical_flip,
                "pcReceiverFlipPolicy": flip_policy,
                "videoFrameConvention": "opencv_top_left_y_down",
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
            robot_session.record_sample(robot_sample)

    def close(self, stop_message: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return self.summary("already_closed")
            self.closed = True
            video_close_errors: list[dict[str, str]] = []
            for side, writer in list(self.video_writers.items()):
                try:
                    writer.close()
                except Exception as exc:
                    video_close_errors.append({"side": str(side), "error": str(exc)})
                    print(f"[calibration] video writer close failed: {exc}", file=sys.stderr)
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
            self.metadata["legacyFlipVerticalParams"] = sorted(self.legacy_flip_vertical_params)
            self.metadata["appliedVerticalFlips"] = sorted(self.applied_vertical_flips)
            self.metadata["pcReceiverFlipPolicies"] = sorted(self.pc_receiver_flip_policies)
            self.metadata["videoCloseErrors"] = video_close_errors
            self.metadata["recordingQualitySummary"] = (
                "video_close_failed" if video_close_errors else self._quality_summary()
            )
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
            summary["videoCloseErrors"] = video_close_errors
            if video_close_errors:
                summary["ok"] = False
                summary["recordingQualitySummary"] = "video_close_failed"
            (self.directory / "pc_calibration_session_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if video_close_errors:
            self.publish_status(
                "failed",
                1.0,
                "PC calibration video close failed",
                summary=summary,
                videoCloseErrors=video_close_errors,
            )
        else:
            self.publish_status("recorded", 0.05, "PC calibration recording saved", summary=summary)
        if self.run_calibration and not video_close_errors:
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
        freedrive_enabled = bool(
            isinstance(robot, dict) and (robot.get("freedriveEnabled") or robot.get("freeDragEnabled"))
        )
        freedrive_method = robot.get("freedriveMethod") or robot.get("freeDragMethod") if isinstance(robot, dict) else None
        freedrive_plan = robot.get("freedrivePlan") or robot.get("freeDragPlan") if isinstance(robot, dict) else None
        freedrive_loop_alive = bool(
            isinstance(robot, dict)
            and (
                robot.get("freedriveLoopAlive")
                or robot.get("freeDragLoopAlive")
                or robot.get("cartesianControlLoopAlive")
            )
        )
        freedrive_last_error = (
            robot.get("freedriveLastError") or robot.get("freeDragLastError") if isinstance(robot, dict) else None
        )
        freedrive_send_signature = robot.get("cartesianSendSignature") if isinstance(robot, dict) else None
        controller_motion = bool(isinstance(config, dict) and config.get("controllerMotionEnabled"))
        control_mode = active.get("controlMode") if isinstance(active, dict) else ROBOT_SESSION_CONTROL_FREEDRIVE
        if control_mode not in (ROBOT_SESSION_CONTROL_TELEOP, ROBOT_SESSION_CONTROL_FREEDRIVE):
            control_mode = ROBOT_SESSION_CONTROL_FREEDRIVE
        robot_sn = robot.get("robotSn") or config.get("robotSn") if isinstance(robot, dict) and isinstance(config, dict) else None
        pose_field = robot.get("poseField") or config.get("poseField") if isinstance(robot, dict) and isinstance(config, dict) else None
        camera_serial = config.get("cameraSerial") if isinstance(config, dict) else None
        third_camera_serial = config.get("thirdCameraSerial") if isinstance(config, dict) else None
        recording = bool(self.robot_session is not None)
        reason = "recording"
        if not connected:
            reason = "Flexiv robot is not connected"
        elif not config.get("cameraSerial"):
            reason = "RealSense camera serial is empty"
        elif not recording:
            reason = self.robot_manager.last_error or "robot RealSense session did not start"
        elif control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE and freedrive_enabled:
            reason = "recording; robot free-drag mode is enabled"
        elif control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE:
            reason = self.robot_manager.last_error or "robot free-drag mode is not enabled"
        elif active and active.get("controllerAlignmentRequired") and not active.get("controllerAlignmentAvailable"):
            reason = "recording; Quest-robot alignment is required before right-controller teleop"
        elif control_mode == ROBOT_SESSION_CONTROL_TELEOP and not controller_motion:
            reason = self.robot_manager.last_error or "controller teleop is disabled"
        elif not motion_armed:
            reason = "recording; hold right middle-finger trigger to teleoperate"
        else:
            reason = "recording; hold right middle-finger trigger to teleoperate"
        return {
            "enabled": True,
            "stage": stage,
            "recording": recording,
            "reason": reason,
            "robotConnected": connected,
            "robotSn": robot_sn,
            "poseField": pose_field,
            "cameraSerial": camera_serial,
            "thirdCameraSerial": third_camera_serial,
            "motionArmed": motion_armed,
            "freedriveEnabled": freedrive_enabled,
            "freeDragEnabled": freedrive_enabled,
            "freedriveMethod": freedrive_method,
            "freeDragMethod": freedrive_method,
            "freedrivePlan": freedrive_plan,
            "freeDragPlan": freedrive_plan,
            "freedriveLoopAlive": freedrive_loop_alive,
            "freeDragLoopAlive": freedrive_loop_alive,
            "cartesianControlLoopAlive": freedrive_loop_alive,
            "freedriveLastError": freedrive_last_error,
            "freeDragLastError": freedrive_last_error,
            "cartesianSendSignature": freedrive_send_signature,
            "controlMode": control_mode,
            "controllerMotionEnabled": controller_motion,
            "teleopRequiresRightSideButton": control_mode == ROBOT_SESSION_CONTROL_TELEOP,
            "teleopRequiresRightHandTrigger": control_mode == ROBOT_SESSION_CONTROL_TELEOP,
            "gripperEnabled": bool(isinstance(config, dict) and config.get("gripperEnabled")),
            "activeSession": active,
        }

    def _video_writer(self, side: str, width: int, height: int) -> ColorStreamWriter:
        existing = self.video_writers.get(side)
        if existing is not None:
            return existing
        filename = str(self.metadata["leftVideoFileName"] if side == "left" else self.metadata["rightVideoFileName"])
        path = self.directory / filename
        fps = max(1, int(self.metadata.get("fps") or 15))
        # H.264 via system ffmpeg (libx264) for ~2-4x smaller files than mp4v.
        writer = ColorStreamWriter(path, width=int(width), height=int(height), fps=fps)
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
            "--max-diverse-detection-frames-per-side",
            str(self.calibration_options.get("max_diverse_detection_frames_per_side", 220)),
            "--max-diverse-fit-frames-per-side",
            str(self.calibration_options.get("max_diverse_fit_frames_per_side", 180)),
            "--min-diverse-frames-per-side",
            str(self.calibration_options.get("min_diverse_frames_per_side", 28)),
            "--diverse-translation-scale-m",
            str(self.calibration_options.get("diverse_translation_scale", 0.025)),
            "--diverse-rotation-scale-deg",
            str(self.calibration_options.get("diverse_rotation_scale", 3.0)),
            "--diverse-min-score",
            str(self.calibration_options.get("diverse_min_score", 0.75)),
            "--image-y-axis",
            "down",
        ]
        if bool(self.calibration_options.get("disable_diverse_frame_selection", False)):
            command.append("--disable-diverse-frame-selection")
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
        cleanup = keep_only_latest_calibration_state(
            self.raw_root,
            self.output_root,
            self.raw_record_name,
            self.output_directory.name,
        )
        if self.visualizer is not None and cleanup.get("removed"):
            self.visualizer.publish_event({"type": "calibration_retained_latest", **cleanup})

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
        self._copy_robot_calibration_artifacts()
        if result.get("ok"):
            self.publish_status("robot_done", 1.0, "Flexiv/RealSense hand-eye calibration complete")
        else:
            self.publish_status("robot_failed", 1.0, result.get("error") or "Flexiv/RealSense hand-eye calibration failed")

    def _copy_robot_calibration_artifacts(self) -> None:
        if self.robot_realsense_directory is None:
            return
        destination = self.output_directory / "robot_realsense"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("robot_hand_eye_result.json", "robot_hand_eye_failure.json", "session_summary.json", "capture_config.json"):
            source = self.robot_realsense_directory / filename
            if source.exists():
                (destination / filename).write_bytes(source.read_bytes())


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
        calibration_options: dict[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.raw_root = raw_root
        self.output_root = output_root
        self.visualizer = visualizer
        self.run_calibration = run_calibration
        self.robot_manager = robot_manager
        self.calibration_options = dict(calibration_options or {})
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
        if preflight.get("error") == "capture_busy":
            detail = str(preflight.get("captureState", {}).get("detail") or "capture is busy")
            message = "B calibration start rejected: PC is recording or saving"
        else:
            failed = [
                check
                for check in preflight.get("checks", [])
                if isinstance(check, dict) and check.get("required", True) and not check.get("ok")
            ]
            detail = "; ".join(
                f"{check.get('label') or check.get('id')}: {check.get('detail') or 'not ready'}"
                for check in failed
            )
            message = "B calibration start rejected: preflight is not ready"
        self.visualizer.publish_event(
            {
                "type": "calibration_status",
                "recordId": record_id,
                "stage": "rejected",
                "progress": 0.0,
                "message": message,
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
                if receiver.visualizer is not None:
                    capture_state = receiver.visualizer.capture_state_payload()
                    if capture_state.get("phase") in ("recording", "saving"):
                        payload = {
                            "ok": False,
                            "error": "capture_busy",
                            "recordId": record_id,
                            "captureState": capture_state,
                        }
                        receiver.visualizer.publish_calibration_rejected(record_id, payload)
                        self._json_response(payload, status=409)
                        return
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
                        receiver.calibration_options,
                    )
                    receiver.sessions[record_id] = session
                print(f"Started PC calibration raw record: {session.directory}", flush=True)
                if receiver.visualizer is not None:
                    receiver.visualizer.set_capture_state("recording", record_id, "B calibration recording")
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
                if receiver.visualizer is not None:
                    receiver.visualizer.set_capture_state("saving", record_id, "Saving B calibration recording")
                summary = session.close(message)
                print(f"Stopped PC calibration raw record: {summary}", flush=True)
                if receiver.visualizer is not None:
                    receiver.visualizer.set_capture_state(
                        "live",
                        record_id,
                        "B calibration saved",
                        summary=summary,
                    )
                self._json_response({"ok": True, "summary": summary})

            def _json_response(self, payload: dict[str, Any], status: int = 200) -> None:
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                write_http_body_safely(self, data)

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
                third_camera_serial=args.third_realsense_serial,
                width=args.realsense_width,
                height=args.realsense_height,
                fps=args.realsense_fps,
                robot_state_hz=args.robot_state_hz,
                record_depth=not args.no_record_realsense_depth,
                record_depth_every_n_frames=max(1, int(args.record_realsense_depth_every_n_frames)),
                record_depth_format=args.record_realsense_depth_format,
                realsense_auto_exposure=not args.realsense_manual_exposure,
                realsense_exposure=args.realsense_exposure,
                realsense_gain=args.realsense_gain,
                capture_interval_seconds=args.robot_capture_interval,
                run_hand_eye=not args.no_robot_hand_eye,
                hand_eye_max_diverse_samples=args.hand_eye_max_diverse_samples,
                hand_eye_min_diverse_samples=args.hand_eye_min_diverse_samples,
                hand_eye_diverse_translation_scale_m=args.hand_eye_diverse_translation_scale,
                hand_eye_diverse_rotation_scale_deg=args.hand_eye_diverse_rotation_scale,
                hand_eye_diverse_min_score=args.hand_eye_diverse_min_score,
                hand_eye_disable_diverse_selection=args.disable_hand_eye_diverse_selection,
                controller_translation_scale=args.controller_motion_scale,
                controller_max_step_m=args.controller_motion_max_step,
                controller_max_rotation_step_deg=args.controller_motion_max_rotation_step,
                controller_target_update_hz=max(1.0, float(args.controller_target_update_hz)),
                controller_joint_limit_buffer_rad=args.controller_joint_limit_buffer,
                controller_joint_limit_guard_enabled=not args.disable_controller_joint_limit_guard,
                gripper_enabled=args.enable_gripper,
                gripper_device=args.gripper_device,
                gripper_open_width_m=args.gripper_open_width,
                gripper_close_width_m=args.gripper_close_width,
                gripper_speed_mps=args.gripper_speed,
                gripper_force_n=args.gripper_force,
                gripper_trigger_close_threshold=args.gripper_trigger_close_threshold,
                gripper_trigger_open_threshold=args.gripper_trigger_open_threshold,
                gripper_init_on_enable=args.gripper_init_on_enable,
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
            output_root,
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
            calibration_options=calibration_options_from_args(args),
        )
        calibration_receiver.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    requested_rcvbuf = max(0, int(args.udp_receive_buffer_bytes or 0))
    if requested_rcvbuf > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested_rcvbuf)
    actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    sock.settimeout(0.25)
    sock.bind((args.host, args.port))
    if visualizer is not None:
        visualizer.set_udp_listener(args.host, int(args.port), actual_rcvbuf)

    print(f"Listening on udp://{args.host}:{args.port} rcvbuf={actual_rcvbuf}", flush=True)
    print(f"PC recording root: {output_root}", flush=True)

    active: SessionWriter | None = None
    recently_closed_records: dict[str, dict[str, Any]] = {}
    recently_closed_lock = threading.Lock()
    saving_record_id: str | None = None
    close_threads: list[threading.Thread] = []
    total_messages = 0
    total_samples = 0
    last_datagram_perf = time.perf_counter()
    last_active_recording_perf: float | None = None
    last_sample_log_perf = float("-inf")

    def remember_recently_closed(record_id: str, payload: dict[str, Any]) -> None:
        with recently_closed_lock:
            recently_closed_records[record_id] = dict(payload)

    def prune_and_get_recently_closed(record_id: str, now_perf: float) -> dict[str, Any] | None:
        with recently_closed_lock:
            expired = [
                key
                for key, value in recently_closed_records.items()
                if now_perf - float(value.get("closedPerfCounterSeconds", 0.0))
                > RECENTLY_CLOSED_RECORD_REOPEN_GUARD_SECONDS
            ]
            for key in expired:
                recently_closed_records.pop(key, None)
            value = recently_closed_records.get(record_id)
            return dict(value) if isinstance(value, dict) else None

    def finalize_session(
        session: SessionWriter,
        reason: str,
        *,
        saving_detail: str,
        closed_perf_counter: float | None = None,
        async_close: bool | None = None,
    ) -> dict[str, Any]:
        nonlocal saving_record_id
        closed_record_id = session.record_id
        saving_record_id = closed_record_id
        if visualizer is not None:
            visualizer.set_capture_state("saving", closed_record_id, saving_detail)

        def close_work() -> dict[str, Any]:
            nonlocal saving_record_id
            try:
                summary = session.close(reason)
                print_session_summary(summary, stream=sys.stderr)
                handle_post_recording(summary, args)
                remember_recently_closed(
                    closed_record_id,
                    {
                        "closedPerfCounterSeconds": closed_perf_counter or time.perf_counter(),
                        "lastSampleIndex": summary.get("lastSampleIndex"),
                        "reason": summary.get("closedReason"),
                    },
                )
                if visualizer is not None:
                    visualizer.set_capture_state("live", closed_record_id, "PC recording saved", summary=summary)
                return summary
            except Exception as exc:  # pragma: no cover - defensive close path
                error_summary = {
                    "recordId": closed_record_id,
                    "sessionDirectory": str(session.directory),
                    "closedReason": reason,
                    "closeError": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"[session-close-fatal] record={closed_record_id} {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                remember_recently_closed(
                    closed_record_id,
                    {
                        "closedPerfCounterSeconds": closed_perf_counter or time.perf_counter(),
                        "lastSampleIndex": getattr(session, "last_sample_index", None),
                        "reason": reason,
                        "error": error_summary["closeError"],
                    },
                )
                if visualizer is not None:
                    visualizer.set_capture_state(
                        "live",
                        closed_record_id,
                        "PC recording save failed",
                        summary=error_summary,
                    )
                return error_summary
            finally:
                if saving_record_id == closed_record_id:
                    saving_record_id = None

        if async_close is None:
            async_close = not bool(args.single_session)
        if not async_close:
            return close_work()

        thread = threading.Thread(
            target=close_work,
            name=f"pc-session-close-{sanitize_name(closed_record_id)}",
            daemon=True,
        )
        thread.start()
        close_threads.append(thread)
        close_threads[:] = [item for item in close_threads if item.is_alive()]
        return {
            "recordId": closed_record_id,
            "sessionDirectory": str(session.directory),
            "closedReason": reason,
            "savingAsync": True,
        }

    try:
        while True:
            try:
                data, remote = sock.recvfrom(65535)
            except TimeoutError:
                now_perf = time.perf_counter()
                idle_timeout = max(0.0, float(args.recording_idle_timeout_seconds or 0.0))
                if (
                    active is not None
                    and idle_timeout > 0
                    and last_active_recording_perf is not None
                    and now_perf - last_active_recording_perf >= idle_timeout
                ):
                    closed_record_id = active.record_id
                    summary = finalize_session(
                        active,
                        "recording_idle_timeout",
                        saving_detail="Saving PC recording after idle timeout",
                        closed_perf_counter=now_perf,
                    )
                    active = None
                    last_active_recording_perf = None
                    if args.single_session:
                        return 0
                    continue
                if args.timeout is not None and time.perf_counter() - last_datagram_perf >= args.timeout:
                    if active is not None:
                        closed_record_id = active.record_id
                        finalize_session(
                            active,
                            "timeout",
                            saving_detail="Saving PC recording after timeout",
                            closed_perf_counter=time.perf_counter(),
                        )
                        active = None
                        last_active_recording_perf = None
                    if total_messages == 0:
                        print(f"Timed out with no UDP telemetry after {args.timeout:.3f}s.", file=sys.stderr)
                        return 2
                    return 0
                continue

            pc_receive_unix_seconds = time.time()
            pc_receive_perf_counter_seconds = time.perf_counter()
            last_datagram_perf = pc_receive_perf_counter_seconds

            if not data or data.isspace():
                continue

            try:
                message = json.loads(data)
            except json.JSONDecodeError as exc:
                print(f"[bad-json] {remote[0]}:{remote[1]} {exc}: {data[:200]!r}", file=sys.stderr)
                continue

            if not isinstance(message, dict):
                print(f"[bad-message] {remote[0]}:{remote[1]} expected JSON object", file=sys.stderr)
                continue

            msg_type = str(message.get("type") or "?")
            record_id = str(message.get("recordId") or "orphan")
            is_sample = msg_type == "sample"
            is_recording_sample = bool(message.get("isRecording", True))
            should_write = (not is_sample) or is_recording_sample or args.record_live_preview
            is_stop_like_sample = is_sample and active is not None and active.record_id == record_id and not is_recording_sample
            recently_closed = prune_and_get_recently_closed(record_id, pc_receive_perf_counter_seconds)
            sample_index = message.get("sampleIndex")
            is_late_closed_record_sample = (
                is_sample
                and is_recording_sample
                and active is None
                and recently_closed is not None
            )
            if is_late_closed_record_sample:
                should_write = False
            if msg_type == "recording_start" and active is None and recently_closed is not None:
                should_write = False
            if active is None and saving_record_id is not None and (msg_type == "recording_start" or is_sample):
                should_write = False
            suppress_visualizer_event = bool(
                active is None
                and (recently_closed is not None or saving_record_id is not None)
                and (
                    msg_type == "recording_start"
                    or (is_sample and is_recording_sample)
                )
            )

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
                visualizer.note_udp_datagram(message, wrapper, len(data))

            if msg_type == "recording_stop" or is_stop_like_sample:
                if active is not None and record_id == active.record_id:
                    closed_record_id = active.record_id
                    close_reason = "recording_stop" if msg_type == "recording_stop" else "recording_false_sample"
                    finalize_session(
                        active,
                        close_reason,
                        saving_detail="Saving PC recording",
                        closed_perf_counter=time.perf_counter(),
                    )
                    active = None
                    last_active_recording_perf = None
                elif visualizer is not None:
                    visualizer.publish_event(
                        {
                            "type": "capture_rejected",
                            "recordId": record_id,
                            "reason": "stop_without_active_session",
                            "messageType": msg_type,
                            "sampleIndex": sample_index,
                        }
                    )
                if args.single_session:
                    return 0
                continue

            can_open_session = should_write and (msg_type == "recording_start" or is_sample)
            if can_open_session and saving_record_id is not None:
                can_open_session = False
                should_write = False
                if visualizer is not None:
                    visualizer.publish_event(
                        {
                            "type": "capture_rejected",
                            "recordId": record_id,
                            "reason": "capture_saving",
                            "savingRecordId": saving_record_id,
                            "messageType": msg_type,
                            "sampleIndex": sample_index,
                        }
                    )
            if can_open_session and visualizer is not None:
                capture_state = visualizer.capture_state_payload()
                if capture_state.get("phase") == "saving":
                    can_open_session = False
                    should_write = False
                    visualizer.publish_event(
                        {
                            "type": "capture_rejected",
                            "recordId": record_id,
                            "reason": "capture_saving",
                            "captureState": capture_state,
                            "messageType": msg_type,
                            "sampleIndex": sample_index,
                        }
                    )
            if can_open_session and (active is None or (
                msg_type == "recording_start"
                and active.record_id != record_id
                and active.messages > 0
            )):
                if active is not None:
                    closed_record_id = active.record_id
                    finalize_session(
                        active,
                        "superseded_by_new_recording_start",
                        saving_detail=f"Saving PC recording before starting {record_id}",
                        closed_perf_counter=time.perf_counter(),
                        async_close=False,
                    )
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
                    args.flush_interval_seconds,
                    args.calibration_output_root.resolve() if not args.no_calibration_http else None,
                    args.calibration_raw_root.resolve() if not args.no_calibration_http else None,
                    robot_manager,
                    visualizer,
                )
                last_active_recording_perf = pc_receive_perf_counter_seconds
                print(f"Started PC session: {active.directory}", flush=True)
                if visualizer is not None:
                    visualizer.set_capture_state("recording", record_id, "PC formal recording")

            if active is not None and record_id == active.record_id and (
                msg_type == "recording_start" or (is_sample and is_recording_sample)
            ):
                last_active_recording_perf = pc_receive_perf_counter_seconds

            if should_write and active is not None:
                try:
                    active.write(wrapper)
                except Exception as exc:
                    print(
                        f"[session-write-error] record={active.record_id} type={msg_type} error={exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            if visualizer is not None and not suppress_visualizer_event:
                visualizer.publish(message, wrapper)
            total_messages += 1

            if is_sample:
                if not args.quiet:
                    log_interval = max(0.0, float(args.sample_log_interval_seconds))
                    should_log_sample = (
                        log_interval <= 0.0
                        or pc_receive_perf_counter_seconds - last_sample_log_perf >= log_interval
                    )
                else:
                    should_log_sample = False
                if should_log_sample:
                    last_sample_log_perf = pc_receive_perf_counter_seconds
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

    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
        if active is not None:
            closed_record_id = active.record_id
            finalize_session(
                active,
                "keyboard_interrupt",
                saving_detail="Saving PC recording after interrupt",
                async_close=False,
            )
        return 130
    finally:
        for thread in list(close_threads):
            thread.join(timeout=SESSION_CLOSE_THREAD_JOIN_SECONDS)
        sock.close()
        if calibration_receiver is not None:
            calibration_receiver.stop()
        if visualizer is not None:
            visualizer.stop()


def calibration_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_diverse_detection_frames_per_side": int(getattr(args, "max_diverse_detection_frames_per_side", 220)),
        "max_diverse_fit_frames_per_side": int(getattr(args, "max_diverse_fit_frames_per_side", 180)),
        "min_diverse_frames_per_side": int(getattr(args, "min_diverse_frames_per_side", 28)),
        "diverse_translation_scale": float(getattr(args, "diverse_translation_scale", 0.025)),
        "diverse_rotation_scale": float(getattr(args, "diverse_rotation_scale", 3.0)),
        "diverse_min_score": float(getattr(args, "diverse_min_score", 0.75)),
        "disable_diverse_frame_selection": bool(getattr(args, "disable_diverse_frame_selection", False)),
    }


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


def audit_performance(args: argparse.Namespace) -> int:
    session_dir = resolve_performance_audit_session(args.pc_session, args.output_root)
    pc_summary = read_json_if_exists(session_dir / "pc_session_summary.json")
    if not isinstance(pc_summary, dict):
        pc_summary = {}
    robot_summary = robot_realsense_record_summary(session_dir)
    thresholds = {
        "minRobotTargetRatio": float(args.min_robot_target_ratio),
        "minCameraTargetRatio": float(args.min_camera_target_ratio),
        "maxCameraDropRatio": float(args.max_camera_drop_ratio),
        "maxCameraLatencyP95Seconds": float(args.max_camera_latency_p95_seconds),
    }
    audit = build_performance_audit(session_dir, robot_summary, thresholds, pc_summary=pc_summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote performance audit JSON: {args.output_json}", flush=True)
    print_performance_audit(audit)
    return 0 if audit.get("ok") else 1


def resolve_performance_audit_session(pc_session: Path | None, output_root: Path) -> Path:
    if pc_session is not None:
        path = pc_session.resolve()
        if not path.exists():
            raise FileNotFoundError(f"PC session path not found: {path}")
        return path.parent if path.is_file() else path
    root = output_root.resolve()
    candidates = [item for item in root.iterdir() if item.is_dir() and item.name.startswith("record_")] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No record_* folders under {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def pc_recording_integrity_checks(pc_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not pc_summary:
        return [
            audit_item(
                "pc_summary_present",
                "PC session summary present",
                False,
                "pc_session_summary.json missing or invalid",
            )
        ]
    close_errors = pc_summary.get("closeErrors") if isinstance(pc_summary.get("closeErrors"), list) else []
    writer_error = pc_summary.get("writerError")
    dropped = finite_int(pc_summary.get("writerDroppedMessages"), 0)
    replay_cache = pc_summary.get("replayVisualizationJson")
    replay_cache_ok = isinstance(replay_cache, str) and Path(replay_cache).exists()
    checks = [
        audit_item(
            "pc_summary_present",
            "PC session summary present",
            True,
            f"closedReason={pc_summary.get('closedReason') or 'n/a'}",
        ),
        audit_item(
            "pc_writer_no_drops",
            "PC writer queue drops",
            dropped == 0,
            f"{dropped} dropped message(s)",
        ),
        audit_item(
            "pc_writer_no_error",
            "PC writer background errors",
            not writer_error,
            "none" if not writer_error else str(writer_error),
        ),
        audit_item(
            "pc_close_errors",
            "PC close/save errors",
            len(close_errors) == 0,
            f"{len(close_errors)} close error(s)",
            closeErrors=close_errors[:8],
        ),
        audit_item(
            "pc_replay_cache",
            "Replay visualization cache",
            replay_cache_ok,
            str(replay_cache) if replay_cache else "missing replayVisualizationJson",
        ),
    ]
    return checks


def build_performance_audit(
    session_dir: Path,
    robot_summary: dict[str, Any] | None,
    thresholds: dict[str, float],
    *,
    pc_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "ok": False,
        "recordId": session_dir.name,
        "sessionDirectory": str(session_dir),
        "thresholds": thresholds,
        "checks": checks,
        "robotSummary": robot_summary,
        "pcSummary": pc_summary,
    }
    pc_summary = pc_summary if isinstance(pc_summary, dict) else {}
    checks.extend(pc_recording_integrity_checks(pc_summary))
    if not isinstance(robot_summary, dict):
        checks.append(
            audit_item(
                "robot_realsense_present",
                "Robot/RealSense recording present",
                False,
                "missing robot_realsense folder or session summary",
            )
        )
        payload["summary"] = audit_summary_text(checks)
        return payload

    checks.append(
        audit_item(
            "robot_realsense_present",
            "Robot/RealSense recording present",
            True,
            f"status {robot_summary.get('status') or 'n/a'}",
        )
    )
    performance = effective_performance_payload(
        robot_summary.get("performance") if isinstance(robot_summary.get("performance"), dict) else {}
    )
    robot_perf = performance.get("robotState") if isinstance(performance.get("robotState"), dict) else {}
    robot_ratio = robot_perf.get("targetRatio")
    checks.append(
        audit_item(
            "robot_state_rate",
            "Robot state effective rate",
            bool(is_number(robot_ratio) and float(robot_ratio) >= thresholds["minRobotTargetRatio"]),
            robot_rate_detail(robot_perf),
            performance=robot_perf,
        )
    )
    robot_gap = robot_perf.get("gapSeconds") if isinstance(robot_perf.get("gapSeconds"), dict) else {}
    if isinstance(robot_gap, dict) and is_number(robot_gap.get("p95")):
        target_hz = max(1e-6, finite_float(robot_perf.get("targetHz"), 90.0))
        checks.append(
            audit_item(
                "robot_state_gap_p95",
                "Robot state p95 gap",
                float(robot_gap["p95"]) <= (2.5 / target_hz),
                f"gap p95 {format_seconds(robot_gap.get('p95'))}",
                performance=robot_gap,
                required=False,
            )
        )

    camera_perf = performance.get("cameras") if isinstance(performance.get("cameras"), dict) else {}
    close_errors = robot_summary.get("closeErrors") if isinstance(robot_summary.get("closeErrors"), list) else []
    checks.append(
        audit_item(
            "robot_close_errors",
            "Robot/RealSense close errors",
            len(close_errors) == 0,
            f"{len(close_errors)} close error(s)",
            closeErrors=close_errors[:8],
        )
    )
    alive_threads = robot_summary.get("cameraWriterThreadsAliveOnClose")
    alive_roles = [
        str(role)
        for role, alive in (alive_threads.items() if isinstance(alive_threads, dict) else [])
        if alive
    ]
    checks.append(
        audit_item(
            "camera_writer_threads_closed",
            "Camera writer threads closed",
            not alive_roles,
            "all camera writer threads closed" if not alive_roles else "alive roles: " + ", ".join(alive_roles),
        )
    )
    if not camera_perf:
        checks.append(audit_item("camera_streams", "Camera streams", False, "no camera performance rows"))
    for role, role_perf_value in sorted(camera_perf.items()):
        role_perf = role_perf_value if isinstance(role_perf_value, dict) else {}
        video_perf = role_perf.get("video") if isinstance(role_perf.get("video"), dict) else {}
        video_ratio = video_perf.get("targetRatio")
        drop_ratio = role_perf.get("queueDropRatio")
        latency = (
            role_perf.get("captureToWriteLatencySeconds")
            if isinstance(role_perf.get("captureToWriteLatencySeconds"), dict)
            else {}
        )
        latency_p95 = latency.get("p95") if isinstance(latency, dict) else None
        checks.append(
            audit_item(
                f"camera_{role}_rate",
                f"{role} camera effective rate",
                bool(is_number(video_ratio) and float(video_ratio) >= thresholds["minCameraTargetRatio"]),
                camera_rate_detail(role, role_perf),
                performance=role_perf,
            )
        )
        checks.append(
            audit_item(
                f"camera_{role}_drop",
                f"{role} camera queue drops",
                bool((not is_number(drop_ratio)) or float(drop_ratio) <= thresholds["maxCameraDropRatio"]),
                camera_queue_detail(role, role_perf),
                performance=role_perf,
            )
        )
        if is_number(latency_p95):
            checks.append(
                audit_item(
                    f"camera_{role}_latency_p95",
                    f"{role} camera capture-to-write latency p95",
                    float(latency_p95) <= thresholds["maxCameraLatencyP95Seconds"],
                    f"{role}: latency p95 {format_seconds(latency_p95)}",
                    performance=latency,
                )
            )

    payload["ok"] = all(check.get("ok") or check.get("required") is False for check in checks)
    payload["summary"] = audit_summary_text(checks)
    payload["performance"] = performance
    return payload


def print_performance_audit(audit: dict[str, Any]) -> None:
    print(
        f"Performance audit {audit.get('recordId')}: "
        f"{'PASS' if audit.get('ok') else 'FAIL'} ({audit.get('summary')})",
        flush=True,
    )
    for check in audit.get("checks") or []:
        status = "ok" if check.get("ok") else "FAIL"
        print(f"  [{status}] {check.get('label')}: {check.get('detail')}", flush=True)


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
            if is_late_tail_record(record):
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


def is_late_tail_record(record: dict[str, Any]) -> bool:
    record_id = str(record.get("recordId") or "")
    if not record_id.rsplit("_", 1)[-1].isdigit():
        return False
    samples = int(record.get("samples") or 0) if is_number(record.get("samples")) else 0
    if samples > 3:
        return False
    if record.get("closedReason") != "recording_idle_timeout":
        return False
    return True


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
    performance = robot_realsense_performance_summary(robot_dir, session, config)
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
    performance_payload = session.get("performance") if isinstance(session.get("performance"), dict) else {}
    record_depth_format = config.get("recordDepthFormat") or performance_payload.get("recordDepthFormat")
    return {
        "status": status,
        "samples": session.get("samples"),
        "questAlignedSamples": session.get("questAlignedSamples"),
        "robotStateSamples": session.get("robotStateSamples"),
        "images": session.get("images"),
        "videoFrames": session.get("videoFrames"),
        "depthFrames": session.get("depthFrames"),
        "recordDepth": config.get("recordDepth"),
        "recordDepthEveryNFrames": config.get("recordDepthEveryNFrames"),
        "recordDepthFormat": record_depth_format,
        "depthStreamEncodings": session.get("depthStreamEncodings")
        if isinstance(session.get("depthStreamEncodings"), dict)
        else {},
        "depthStreamWarnings": session.get("depthStreamWarnings")
        if isinstance(session.get("depthStreamWarnings"), list)
        else [],
        "depthStreamDisabledRoles": session.get("depthStreamDisabledRoles")
        if isinstance(session.get("depthStreamDisabledRoles"), dict)
        else {},
        "performance": performance,
        "motionCommands": session.get("motionCommands"),
        "motionSkips": session.get("motionSkips"),
        "motionErrors": session.get("motionErrors"),
        "lastMotionReason": last_motion.get("reason") or last_motion.get("error"),
        "closeErrors": session.get("closeErrors") if isinstance(session.get("closeErrors"), list) else [],
        "cameraWriterThreadsAliveOnClose": session.get("cameraWriterThreadsAliveOnClose")
        if isinstance(session.get("cameraWriterThreadsAliveOnClose"), dict)
        else {},
        "detections": counts.get("detections") if isinstance(counts, dict) else None,
        "requiredDetections": config.get("minHandEyeDetections"),
        "translationSpanM": diversity.get("eeTranslationSpanM") if isinstance(diversity, dict) else None,
        "rotationSpanDeg": diversity.get("eeRotationSpanDeg") if isinstance(diversity, dict) else None,
        "residualMedianMm": residual.get("median") if isinstance(residual, dict) else None,
        "failureReason": failure.get("error") if isinstance(failure, dict) else None,
    }


def robot_realsense_performance_summary(
    robot_dir: Path,
    session: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    existing = session.get("performance") if isinstance(session.get("performance"), dict) else None
    robot_states = read_jsonl_relaxed(robot_dir / "robot_states.jsonl")
    video_frames = read_jsonl_relaxed(robot_dir / "video_frames.jsonl")
    target_robot_hz = max(1e-6, finite_float(config.get("robotStateHz") or session.get("robotStateHz"), 90.0))
    target_camera_hz = max(1e-6, finite_float(config.get("fps"), 30.0))
    depth_every = max(1, finite_int(config.get("recordDepthEveryNFrames"), 1))
    record_depth = config.get("recordDepth") is not False
    target_depth_hz = target_camera_hz / depth_every if record_depth else 0.0

    robot_times = [float(row["pc_perf_counter_seconds"]) for row in robot_states if is_number(row.get("pc_perf_counter_seconds"))]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in video_frames:
        role = str(row.get("role") or "camera")
        by_role.setdefault(role, []).append(row)
    cameras: dict[str, Any] = {}
    queue_drops = session.get("cameraQueueDrops") if isinstance(session.get("cameraQueueDrops"), dict) else {}
    subscriber_drops = (
        session.get("cameraSubscriberDrops")
        if isinstance(session.get("cameraSubscriberDrops"), dict)
        else {}
    )
    for role, rows in sorted(by_role.items()):
        video_times = [
            timestamp
            for row in rows
            for timestamp in [row_time_seconds(row, "pc_perf_counter_seconds", "captured_at")]
            if timestamp is not None
        ]
        write_durations = [
            float(row["write_duration_seconds"])
            for row in rows
            if is_number(row.get("write_duration_seconds"))
        ]
        latencies = []
        for row in rows:
            latency = row_capture_to_write_latency_seconds(row)
            if latency is not None:
                latencies.append(latency)
        depth_rows = [row for row in rows if isinstance(row.get("depth"), dict) and row["depth"].get("path")]
        depth_times = [
            timestamp
            for row in depth_rows
            for timestamp in [row_time_seconds(row, "pc_perf_counter_seconds", "captured_at")]
            if timestamp is not None
        ]
        session_drops = finite_int(queue_drops.get(role), 0)
        hub_drops = finite_int(subscriber_drops.get(role), 0)
        drops = session_drops + hub_drops
        cameras[role] = {
            "video": time_series_summary(video_times, target_camera_hz),
            "depth": time_series_summary(depth_times, target_depth_hz) if record_depth else {"count": 0},
            "writeDurationSeconds": stats_summary(write_durations),
            "captureToWriteLatencySeconds": stats_summary(latencies),
            "queueDrops": drops,
            "sessionQueueDrops": session_drops,
            "hubSubscriberDrops": hub_drops,
            "queueDropRatio": float(drops / max(1, len(rows) + drops)),
            "videoFramesWritten": len(rows),
            "depthFramesWritten": len(depth_rows),
        }
    computed = {
        "robotState": time_series_summary(robot_times, target_robot_hz),
        "cameras": cameras,
        "targetRobotStateHz": target_robot_hz,
        "targetCameraHz": target_camera_hz,
        "recordDepth": record_depth,
        "recordDepthEveryNFrames": depth_every,
        "targetDepthHz": target_depth_hz,
    }
    if existing:
        merged = dict(existing)
        merged["computedFromJsonl"] = computed
        return merged
    return computed


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
        "coordinateFrame": event.get("coordinateFrame"),
        "rawTrajectoryFrame": event.get("rawTrajectoryFrame"),
        "worldFrameConversion": event.get("worldFrameConversion"),
        "T_world_board": event.get("T_world_board"),
        "T_board_world": event.get("T_board_world"),
        "T_unity_world_board": event.get("T_unity_world_board"),
        "T_board_unity_world": event.get("T_board_unity_world"),
        "questWorldOriginInBoardM": event.get("questWorldOriginInBoardM"),
        "unityQuestWorldOriginInBoardM": event.get("unityQuestWorldOriginInBoardM"),
        "boardNormalWorld": event.get("boardNormalWorld"),
        "boardNormalAbsAngleToWorldZDeg": event.get("boardNormalAbsAngleToWorldZDeg"),
        "bestLagSeconds": event.get("bestLagSeconds"),
        "keptFrames": event.get("keptFrames"),
        "inputFrames": event.get("inputFrames"),
        "medianReprojectionPx": event.get("medianReprojectionPx"),
        "p90ReprojectionPx": event.get("p90ReprojectionPx"),
    }


def canonicalize_calibration_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    result = dict(snapshot)
    frame = result.get("coordinateFrame") or result.get("coordinate_frame")
    legacy_unity = not is_pc_world_frame(frame)
    if legacy_unity:
        result.setdefault("T_unity_world_board", result.get("T_world_board"))
        result.setdefault("T_board_unity_world", result.get("T_board_world"))
    t_world_board = ensure_pc_transform_payload(result.get("T_world_board"), frame)
    if t_world_board is not None:
        result["T_world_board"] = t_world_board
        matrix = matrix_from_transform_payload(t_world_board)
        if matrix is not None:
            result["T_board_world"] = coordinate_transform_payload_from_matrix(np.linalg.inv(matrix), PC_WORLD_FRAME)
    else:
        t_board_world = ensure_pc_transform_payload(result.get("T_board_world"), frame)
        if t_board_world is not None:
            result["T_board_world"] = t_board_world
            matrix = matrix_from_transform_payload(t_board_world)
            if matrix is not None:
                result["T_world_board"] = coordinate_transform_payload_from_matrix(np.linalg.inv(matrix), PC_WORLD_FRAME)
    result["coordinateFrame"] = PC_WORLD_FRAME
    result.setdefault("rawTrajectoryFrame", UNITY_WORLD_FRAME)
    result.setdefault("worldFrameConversion", WORLD_FRAME_CONVERSION)
    return result


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
    compact: bool = True,
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
    if compact:
        return replay_visualization_payload(
            session_dir,
            safe_id,
            resolved_source,
            recording_root=recording_root,
            calibration_raw_root=calibration_raw_root,
            calibration_output_root=calibration_output_root,
        )
    return build_recording_replay_payload_uncached(
        session_dir,
        safe_id,
        resolved_source,
        calibration_output_root=calibration_output_root,
    )


def build_recording_replay_payload_uncached(
    session_dir: Path,
    safe_id: str,
    resolved_source: str,
    *,
    calibration_output_root: Path | None = None,
) -> dict[str, Any]:
    samples_path = replay_samples_path(session_dir, resolved_source)
    if samples_path is None or not samples_path.exists():
        raise FileNotFoundError(f"missing samples file for {safe_id}")
    summary = replay_summary(session_dir, resolved_source)
    if not isinstance(summary, dict) or not summary:
        raise FileNotFoundError(f"recording summary not found for {safe_id}")
    snapshot = canonicalize_calibration_snapshot(replay_snapshot(session_dir, summary, resolved_source, calibration_output_root))
    calibration_failure = replay_calibration_failure(session_dir, summary, resolved_source, calibration_output_root)

    board_matrix_world = matrix_from_snapshot(snapshot)
    if board_matrix_world is None:
        board_origin_world = [0.0, 0.0, 0.0]
        board_matrix_display = default_board_matrix_4x4()
    else:
        board_origin_world = matrix_translation(board_matrix_world)
        board_matrix_display = translated_matrix_4x4(board_matrix_world, board_origin_world)

    raw_rows = read_jsonl_relaxed(samples_path)
    samples = [recording_replay_sample(row, board_origin_world) for row in raw_rows]

    gaze_diagnostics = build_gaze_depth_diagnostics(raw_rows, snapshot)
    enrich_replay_samples_with_gaze_diagnostics(samples, gaze_diagnostics, board_origin_world)
    robot_realsense = build_robot_realsense_replay(session_dir, board_origin_world)
    raw_artifacts = replay_raw_artifacts(session_dir, summary, resolved_source, calibration_output_root)
    recording_audit = build_recording_audit(session_dir, raw_rows, samples, snapshot, robot_realsense)

    return {
        "ok": True,
        "recordId": safe_id,
        "source": resolved_source,
        "sessionDir": str(session_dir),
        "summary": summary if isinstance(summary, dict) else {},
        "snapshot": snapshot,
        "calibrationFailure": calibration_failure,
        "coordinateMode": "pc_right_handed_world_axes_translated_to_board_origin",
        "coordinateFrame": PC_WORLD_FRAME,
        "rawTrajectoryFrame": snapshot.get("rawTrajectoryFrame") or UNITY_WORLD_FRAME,
        "worldFrameConversion": snapshot.get("worldFrameConversion") or WORLD_FRAME_CONVERSION,
        "boardOriginWorld": board_origin_world,
        "boardMatrix": board_matrix_display,
        "gazeDepthDiagnostics": gaze_diagnostics.get("summary", {}),
        "robotRealSense": robot_realsense,
        "recordingAudit": recording_audit,
        "rawArtifacts": raw_artifacts,
        "samples": samples,
    }


def replay_visualization_path(session_dir: Path) -> Path:
    return session_dir / REPLAY_VISUALIZATION_CACHE


def replay_visualization_payload(
    session_dir: Path,
    safe_id: str,
    resolved_source: str,
    *,
    recording_root: Path | None = None,
    calibration_raw_root: Path | None = None,
    calibration_output_root: Path | None = None,
) -> dict[str, Any]:
    cache_path = replay_visualization_path(session_dir)
    cached = read_json_if_exists(cache_path)
    samples_path = replay_samples_path(session_dir, resolved_source)
    source_mtime = max(
        [
            path.stat().st_mtime
            for path in (
                samples_path,
                session_dir / "pc_session_summary.json",
                session_dir / "pc_calibration_snapshot.json",
                session_dir / "robot_realsense" / "samples.jsonl",
                session_dir / "robot_realsense" / "robot_states.jsonl",
                session_dir / "robot_realsense" / "session_summary.json",
                session_dir / "robot_realsense" / "robot_hand_eye_result.json",
                session_dir / "robot_realsense" / "robot_hand_eye_failure.json",
            )
            if path is not None and path.exists()
        ]
        or [0.0]
    )
    if (
        isinstance(cached, dict)
        and cached.get("ok")
        and cached.get("cacheVersion") == REPLAY_VISUALIZATION_CACHE_VERSION
        and cached.get("recordId") == safe_id
        and cached.get("source") == resolved_source
        and cache_path.stat().st_mtime >= source_mtime
    ):
        cached["cacheHit"] = True
        cached["cachePath"] = str(cache_path)
        return cached
    if recording_root is None:
        recording_root = DEFAULT_OUTPUT_ROOT
    return build_and_write_replay_visualization_cache(
        session_dir,
        safe_id,
        resolved_source,
        recording_root=recording_root,
        calibration_raw_root=calibration_raw_root,
        calibration_output_root=calibration_output_root,
        source_mtime=source_mtime,
    )


def write_replay_visualization_cache(
    session_dir: Path,
    record_id: str,
    *,
    source: str = "pc",
    calibration_raw_root: Path | None = None,
    calibration_output_root: Path | None = None,
) -> Path:
    safe_id = session_dir.name
    payload = build_and_write_replay_visualization_cache(
        session_dir,
        safe_id,
        source,
        recording_root=session_dir.parent,
        calibration_raw_root=calibration_raw_root,
        calibration_output_root=calibration_output_root,
    )
    return Path(payload["cachePath"])


def build_and_write_replay_visualization_cache(
    session_dir: Path,
    safe_id: str,
    resolved_source: str,
    *,
    recording_root: Path,
    calibration_raw_root: Path | None,
    calibration_output_root: Path | None,
    source_mtime: float | None = None,
) -> dict[str, Any]:
    payload = build_recording_replay_payload_uncached(
        session_dir,
        safe_id,
        resolved_source,
        calibration_output_root=calibration_output_root,
    )
    payload = compact_recording_replay_payload(payload)
    payload["cacheVersion"] = REPLAY_VISUALIZATION_CACHE_VERSION
    payload["cacheHit"] = False
    payload["cacheCreatedUtc"] = datetime.now(timezone.utc).isoformat()
    payload["sourceMtime"] = float(source_mtime if source_mtime is not None else time.time())
    payload["cachePath"] = str(replay_visualization_path(session_dir))
    cache_path = replay_visualization_path(session_dir)
    atomic_write_json(cache_path, payload, compact=True)
    return payload


def compact_recording_replay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["summary"] = compact_replay_summary(result.get("summary"))
    samples = result.get("samples")
    if isinstance(samples, list):
        result["samples"] = [compact_replay_sample(row) for row in samples if isinstance(row, dict)]
    robot_realsense = result.get("robotRealSense")
    if isinstance(robot_realsense, dict):
        result["robotRealSense"] = compact_robot_replay(robot_realsense)
    result["payloadMode"] = "visualization_cache"
    return result


def compact_replay_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    keep = (
        "recordId",
        "sessionDirectory",
        "questOutputDirectory",
        "closedReason",
        "startUtc",
        "messages",
        "samples",
        "lastSampleIndex",
        "leftPoseSamples",
        "rightPoseSamples",
        "robotStartStatus",
        "robotRealSenseDirectory",
        "closeMetrics",
        "replayVisualizationJson",
        "replayVisualizationError",
    )
    return {key: summary.get(key) for key in keep if key in summary}


def compact_robot_replay(robot_realsense: dict[str, Any]) -> dict[str, Any]:
    result = dict(robot_realsense)
    sample_rows = [row for row in robot_realsense.get("samples", []) if isinstance(row, dict)]
    state_rows = [row for row in robot_realsense.get("robotStates", []) if isinstance(row, dict)]
    result["samples"] = [compact_robot_media_row(row) for row in sample_rows]
    result["robotStates"] = [compact_robot_pose_row(row) for row in state_rows]
    if not result["robotStates"]:
        result["robotStates"] = [compact_robot_pose_row(row) for row in sample_rows]
    return result


def compact_robot_pose_row(row: dict[str, Any]) -> dict[str, Any]:
    result = compact_row_scalars(
        row,
        (
            "sampleIndex",
            "questSampleIndex",
            "recordingTimestampSeconds",
            "pcPerfCounterSeconds",
            "ok",
            "sourceStream",
            "error",
        ),
    )
    for key in (
        "T_base_ee",
        "T_base_tool_tcp",
        "T_display_tool_tcp",
        "T_display_ee",
        "T_base_end_camera",
        "T_display_end_camera",
    ):
        compact = compact_matrix_payload(row.get(key))
        if compact is not None:
            result[key] = compact
    jointpose = compact_number_list(row.get("jointpose") or row.get("jointpos"), digits=6)
    if jointpose:
        result["jointpose"] = jointpose
    return result


def compact_robot_media_row(row: dict[str, Any]) -> dict[str, Any]:
    result = compact_row_scalars(
        row,
        (
            "sampleIndex",
            "questSampleIndex",
            "recordingTimestampSeconds",
            "pcPerfCounterSeconds",
            "ok",
            "sourceStream",
            "questGaze3DSource",
            "error",
        ),
    )
    images = compact_media_artifacts(row.get("images"))
    if images:
        result["images"] = images
    videos = compact_media_artifacts(row.get("videos"))
    if videos:
        result["videos"] = videos
    gaze = compact_number_list(row.get("questGaze3DWorld"), digits=6)
    if gaze:
        result["questGaze3DWorld"] = gaze
    return result


def compact_replay_sample(row: dict[str, Any]) -> dict[str, Any]:
    result = compact_row_scalars(
        row,
        (
            "sampleIndex",
            "recordingTimestampSeconds",
            "isRecording",
        ),
    )
    for key in ("head", "leftEye", "rightEye", "left", "right"):
        pose = compact_visual_pose(row.get(key))
        if pose is not None:
            result[key] = pose
    for key in ("gaze", "gazeHit", "gazeFiltered", "gazeBoardPlane"):
        point = compact_visual_point(row.get(key))
        if point is not None:
            result[key] = point
    ray_origin = compact_number_list(row.get("gazeRayOrigin"), digits=6)
    if ray_origin:
        result["gazeRayOrigin"] = ray_origin
    ray_direction = compact_number_list(row.get("gazeRayDirection"), digits=6)
    if ray_direction:
        result["gazeRayDirection"] = ray_direction
    gaze_depth = compact_gaze_depth(row.get("gazeDepth"))
    if gaze_depth:
        result["gazeDepth"] = gaze_depth
    return result


def compact_visual_pose(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {"ok": bool(value.get("ok"))}
    source = value.get("source")
    if source:
        result["source"] = source
    point = compact_number_list(value.get("p"), digits=6)
    if point:
        result["p"] = point
    quat_value = compact_number_list(value.get("q"), digits=7)
    if quat_value:
        result["q"] = quat_value
    controller_input = compact_controller_input(value.get("input"))
    if controller_input:
        result["input"] = controller_input
    return result


def compact_visual_point(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {"ok": bool(value.get("ok"))}
    point = compact_number_list(value.get("p"), digits=6)
    if point:
        result["p"] = point
    source = value.get("source")
    if source:
        result["source"] = source
    return result


def compact_gaze_depth(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return compact_row_scalars(
        value,
        (
            "source",
            "rawDepthM",
            "filteredDepthM",
            "boardPlaneDepthM",
            "boardDistanceM",
            "depthMinusBoardPlaneM",
            "filteredDepthMinusBoardPlaneM",
        ),
    )


def compact_controller_input(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value.get("hasAny"):
        return {}
    return compact_row_scalars(
        value,
        (
            "hasAny",
            "handTrigger",
            "indexTrigger",
            "handTriggerPressed",
            "indexTriggerPressed",
            "aButton",
            "bButton",
            "teleopHeld",
        ),
    )


def compact_matrix_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    matrix = value.get("matrix_4x4")
    if not isinstance(matrix, list) or len(matrix) < 4:
        return None
    rows: list[list[float | int]] = []
    try:
        for row in matrix[:4]:
            if not isinstance(row, list) or len(row) < 4:
                return None
            rows.append([compact_float(row[col], digits=7) for col in range(4)])
    except (TypeError, ValueError):
        return None
    return {"matrix_4x4": rows}


def compact_media_artifacts(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for role, artifact in value.items():
        if not isinstance(role, str) or not isinstance(artifact, dict):
            continue
        item: dict[str, Any] = {}
        for key in ("url", "frameIndex", "serial", "error"):
            if artifact.get(key) is not None:
                item[key] = artifact.get(key)
        if item:
            result[role] = item
    return result


def compact_row_scalars(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        if key not in row or row.get(key) is None:
            continue
        value = row.get(key)
        if isinstance(value, float):
            result[key] = compact_float(value)
        elif isinstance(value, (int, bool, str)):
            result[key] = value
        else:
            result[key] = value
    return result


def compact_number_list(value: Any, *, digits: int = 6) -> list[float | int] | None:
    if not isinstance(value, list):
        return None
    result: list[float | int] = []
    for item in value:
        if not is_number(item):
            return None
        result.append(compact_float(item, digits=digits))
    return result


def compact_float(value: Any, *, digits: int = 6) -> float | int:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    rounded = round(number, digits)
    if rounded == 0:
        return 0
    return rounded


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
    robot_dir = session_dir / "robot_realsense"
    if robot_dir.exists():
        for key, filename, label in (
            ("robotSamples", "samples.jsonl", "Robot quest-aligned samples"),
            ("robotStates", "robot_states.jsonl", "Robot 90Hz states"),
            ("robotVideoFrames", "video_frames.jsonl", "Robot camera frame index"),
            ("robotGripper", "gripper_commands.jsonl", "Robot gripper commands"),
            ("robotMotion", "controller_motion.jsonl", "Robot controller motion commands"),
            ("robotSummary", "session_summary.json", "Robot recording summary"),
            ("robotConfig", "capture_config.json", "Robot capture config"),
            ("robotCameras", "cameras.json", "Robot camera metadata"),
            ("robotHandEyeResult", "robot_hand_eye_result.json", "Robot hand-eye result"),
            ("robotHandEyeFailure", "robot_hand_eye_failure.json", "Robot hand-eye failure"),
        ):
            path = robot_dir / filename
            if path.exists():
                artifacts[key] = artifact_payload(path, label)
        session_payload = read_json_if_exists(robot_dir / "session_summary.json")
        depth_streams = session_payload.get("depthStreams") if isinstance(session_payload, dict) else None
        if isinstance(depth_streams, dict):
            for role, rel_path in depth_streams.items():
                if not isinstance(rel_path, str) or not rel_path:
                    continue
                path = resolve_child_path(robot_dir, rel_path)
                if path is not None and path.exists():
                    artifacts[f"robotDepth_{role}"] = artifact_payload(path, f"Robot depth stream {role}")
        depth_dir = robot_dir / "depth"
        if depth_dir.exists():
            for pattern in ("*.mkv", "*.bin"):
                for path in sorted(depth_dir.glob(pattern)):
                    key = f"robotDepth_{path.stem}"
                    if key not in artifacts:
                        artifacts[key] = artifact_payload(path, f"Robot depth stream {path.name}")
            for path in sorted(depth_dir.glob("*.ffmpeg.log")):
                key = f"robotDepthLog_{path.stem}"
                if key not in artifacts:
                    artifacts[key] = artifact_payload(path, f"Robot depth encoder log {path.name}")
        video_dir = robot_dir / "videos"
        if video_dir.exists():
            for path in sorted(video_dir.glob("*.mp4")):
                artifacts[f"robotVideo_{path.stem}"] = artifact_payload(path, f"Robot video {path.name}")
    return artifacts


def audit_item(id_value: str, label: str, ok: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"id": id_value, "label": label, "ok": bool(ok), "detail": detail, **extra}


def build_recording_audit(
    session_dir: Path,
    raw_rows: list[dict[str, Any]],
    replay_samples: list[dict[str, Any]],
    snapshot: dict[str, Any] | None,
    robot_realsense: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(
        audit_item(
            "quest_gaze3d",
            "Quest gaze3D",
            any(sample.get("gaze", {}).get("ok") for sample in replay_samples),
            f"{sum(1 for sample in replay_samples if sample.get('gaze', {}).get('ok'))}/{len(replay_samples)} replay samples with gaze3D",
        )
    )
    has_snapshot = bool(snapshot and snapshot.get("T_world_board"))
    checks.append(
        audit_item(
            "quest_board_calibration",
            "Quest board calibration",
            has_snapshot,
            "T_world_board available" if has_snapshot else "missing T_world_board calibration snapshot",
        )
    )
    robot_dir = session_dir / "robot_realsense"
    cameras = read_json_if_exists(robot_dir / "cameras.json")
    camera_roles = sorted(cameras.keys()) if isinstance(cameras, dict) else []
    expected_camera_roles = ["end"]
    session_config = read_json_if_exists(robot_dir / "capture_config.json")
    if isinstance(session_config, dict) and session_config.get("thirdCameraSerial"):
        expected_camera_roles.append("third")
    intr_roles = []
    if isinstance(cameras, dict):
        for role, camera in cameras.items():
            if not isinstance(camera, dict):
                continue
            if camera.get("serial") and all(is_number(camera.get(key)) for key in ("fx", "fy", "cx", "cy")):
                intr_roles.append(role)
    checks.append(
        audit_item(
            "camera_metadata",
            "Camera serial/intrinsics metadata",
            all(role in intr_roles for role in expected_camera_roles),
            f"expected {', '.join(expected_camera_roles)}; roles with serial+fx/fy/cx/cy: {', '.join(sorted(intr_roles)) or 'none'}",
            roles=camera_roles,
            expectedRoles=expected_camera_roles,
        )
    )
    videos = sorted((robot_dir / "videos").glob("*.mp4")) if (robot_dir / "videos").exists() else []
    video_role_names = {path.name.split("_", 1)[0] for path in videos}
    checks.append(
        audit_item(
            "robot_camera_mp4",
            "Robot camera MP4",
            all(role in video_role_names for role in expected_camera_roles),
            f"expected {', '.join(expected_camera_roles)}; files: {', '.join(path.name for path in videos) or 'no mp4 videos'}",
            files=[str(path) for path in videos],
            expectedRoles=expected_camera_roles,
        )
    )
    depth_rows = read_jsonl_relaxed(robot_dir / "video_frames.jsonl")
    expected_depth = not (isinstance(session_config, dict) and session_config.get("recordDepth") is False)
    depth_payloads = [
        row["depth"]
        for row in depth_rows
        if isinstance(row.get("depth"), dict) and row["depth"].get("path")
    ]
    depth_count = len(depth_payloads)
    depth_paths = {
        str(payload.get("path"))
        for payload in depth_payloads
        if isinstance(payload.get("path"), str) and payload.get("path")
    }
    missing_depth_paths = [
        rel_path
        for rel_path in sorted(depth_paths)
        for resolved in [resolve_child_path(robot_dir, rel_path)]
        if resolved is None or not resolved.exists()
    ]
    depth_encodings = sorted(
        {
            str(payload.get("encoding") or payload.get("codec") or "unknown")
            for payload in depth_payloads
        }
    )
    checks.append(
        audit_item(
            "robot_realsense_depth",
            "RealSense depth stream",
            (not expected_depth) or (depth_count > 0 and not missing_depth_paths),
            (
                f"{depth_count}/{len(depth_rows)} video frame rows with indexed depth payload; "
                f"files {len(depth_paths) - len(missing_depth_paths)}/{len(depth_paths)}; "
                f"encodings {', '.join(depth_encodings) or 'none'}"
                if expected_depth
                else "depth recording disabled in capture_config"
            ),
            required=expected_depth,
            missingPaths=missing_depth_paths[:8],
            encodings=depth_encodings,
        )
    )
    if robot_realsense is None:
        for id_value, label in (
            ("robot_tool_tcp", "Robot tool TCP trajectory"),
            ("robot_end_camera", "Robot end-camera trajectory"),
            ("robot_unified_coords", "Robot unified coordinates"),
            ("robot_replay_media", "Robot replay media"),
        ):
            checks.append(audit_item(id_value, label, False, "no robot_realsense recording"))
        ok = all(check["ok"] for check in checks)
        return {"ok": ok, "checks": checks, "summary": audit_summary_text(checks)}

    robot_samples = robot_realsense.get("samples") if isinstance(robot_realsense.get("samples"), list) else []
    robot_states = robot_realsense.get("robotStates") if isinstance(robot_realsense.get("robotStates"), list) else []
    performance = effective_performance_payload(
        robot_realsense.get("performance") if isinstance(robot_realsense.get("performance"), dict) else {}
    )
    robot_pose_rows = robot_states or robot_samples
    tcp_count = sum(1 for row in robot_pose_rows if isinstance(row.get("T_base_tool_tcp"), dict))
    end_cam_count = sum(1 for row in robot_pose_rows if isinstance(row.get("T_base_end_camera"), dict))
    world_tcp_count = sum(1 for row in robot_pose_rows if isinstance(row.get("T_world_tool_tcp"), dict) and isinstance(row.get("T_display_tool_tcp"), dict))
    world_end_count = sum(1 for row in robot_pose_rows if isinstance(row.get("T_world_end_camera"), dict) and isinstance(row.get("T_display_end_camera"), dict))
    replay_video_count = sum(1 for row in robot_samples if any((item or {}).get("url") for item in (row.get("videos") or {}).values()))
    gaze_in_robot_count = sum(1 for row in robot_samples if isinstance(row.get("questGaze3DWorld"), list) and len(row.get("questGaze3DWorld")) >= 3)
    checks.extend(
        [
            audit_item(
                "robot_state_stream",
                "Robot high-rate state stream",
                len(robot_states) > 0,
                f"{len(robot_states)} robot_states rows; {len(robot_samples)} quest-aligned rows",
            ),
            audit_item("robot_tool_tcp", "Robot tool TCP trajectory", tcp_count > 0, f"{tcp_count}/{len(robot_pose_rows)} robot pose rows with T_base_tool_tcp"),
            audit_item("robot_end_camera", "Robot end-camera trajectory", end_cam_count > 0, f"{end_cam_count}/{len(robot_pose_rows)} robot pose rows with T_base_end_camera"),
            audit_item(
                "robot_unified_coords",
                "Robot unified coordinates",
                world_tcp_count > 0 and world_end_count > 0,
                f"TCP world/display {world_tcp_count}/{len(robot_pose_rows)}, end camera world/display {world_end_count}/{len(robot_pose_rows)}",
            ),
            audit_item("robot_replay_media", "Robot replay media", replay_video_count > 0, f"{replay_video_count}/{len(robot_samples)} samples with replayable video artifact"),
            audit_item("robot_gaze3d_copy", "gaze3D copied into robot samples", gaze_in_robot_count > 0, f"{gaze_in_robot_count}/{len(robot_samples)} robot samples with questGaze3DWorld"),
        ]
    )
    robot_perf = performance.get("robotState") if isinstance(performance.get("robotState"), dict) else {}
    robot_ratio = robot_perf.get("targetRatio")
    checks.append(
        audit_item(
            "robot_state_rate",
            "Robot state effective rate",
            bool(is_number(robot_ratio) and float(robot_ratio) >= 0.70),
            robot_rate_detail(robot_perf),
            performance=robot_perf,
        )
    )
    camera_perf = performance.get("cameras") if isinstance(performance.get("cameras"), dict) else {}
    for role in expected_camera_roles:
        role_perf = camera_perf.get(role) if isinstance(camera_perf.get(role), dict) else {}
        video_perf = role_perf.get("video") if isinstance(role_perf.get("video"), dict) else {}
        video_ratio = video_perf.get("targetRatio")
        drop_ratio = role_perf.get("queueDropRatio")
        checks.append(
            audit_item(
                f"robot_camera_{role}_rate",
                f"{role} camera effective rate",
                bool(is_number(video_ratio) and float(video_ratio) >= 0.80),
                camera_rate_detail(role, role_perf),
                performance=role_perf,
            )
        )
        checks.append(
            audit_item(
                f"robot_camera_{role}_queue",
                f"{role} camera queue drops",
                bool((not is_number(drop_ratio)) or float(drop_ratio) <= 0.05),
                camera_queue_detail(role, role_perf),
                performance=role_perf,
            )
        )
    ok = all(check["ok"] for check in checks)
    return {"ok": ok, "checks": checks, "summary": audit_summary_text(checks)}


def robot_rate_detail(perf: dict[str, Any]) -> str:
    hz = perf.get("effectiveHz")
    target = perf.get("targetHz")
    count = perf.get("count")
    p95 = None
    gaps = perf.get("gapSeconds") if isinstance(perf.get("gapSeconds"), dict) else None
    if isinstance(gaps, dict):
        p95 = gaps.get("p95")
    return (
        f"{format_hz(hz)} / target {format_hz(target)}, rows {count if count is not None else 'n/a'}, "
        f"gap p95 {format_seconds(p95)}"
    )


def effective_performance_payload(performance: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(performance, dict):
        return {}
    robot = performance.get("robotState") if isinstance(performance.get("robotState"), dict) else {}
    cameras = performance.get("cameras") if isinstance(performance.get("cameras"), dict) else {}
    has_robot = is_number(robot.get("effectiveHz")) or finite_int(robot.get("count"), 0) > 0
    has_camera = any(
        isinstance(value, dict)
        and isinstance(value.get("video"), dict)
        and (
            is_number(value["video"].get("effectiveHz"))
            or finite_int(value["video"].get("count"), 0) > 0
        )
        for value in cameras.values()
    )
    if has_robot or has_camera:
        return performance
    computed = performance.get("computedFromJsonl")
    return computed if isinstance(computed, dict) else performance


def row_time_seconds(row: dict[str, Any], numeric_key: str, iso_key: str) -> float | None:
    value = row.get(numeric_key)
    if is_number(value):
        return float(value)
    timestamp = parse_iso_timestamp_seconds(row.get(iso_key))
    return timestamp


def row_capture_to_write_latency_seconds(row: dict[str, Any]) -> float | None:
    write_t = row_time_seconds(row, "pc_perf_counter_seconds", "captured_at")
    capture_t = row_time_seconds(row, "frame_captured_perf_counter_seconds", "frame_captured_at_utc")
    if write_t is None or capture_t is None:
        return None
    return float(write_t - capture_t)


def parse_iso_timestamp_seconds(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def camera_rate_detail(role: str, perf: dict[str, Any]) -> str:
    video = perf.get("video") if isinstance(perf.get("video"), dict) else {}
    latency = perf.get("captureToWriteLatencySeconds") if isinstance(perf.get("captureToWriteLatencySeconds"), dict) else {}
    return (
        f"{role}: {format_hz(video.get('effectiveHz'))} / target {format_hz(video.get('targetHz'))}, "
        f"frames {perf.get('videoFramesWritten', video.get('count', 'n/a'))}, "
        f"latency p95 {format_seconds(latency.get('p95'))}"
    )


def camera_queue_detail(role: str, perf: dict[str, Any]) -> str:
    drops = perf.get("queueDrops")
    ratio = perf.get("queueDropRatio")
    return f"{role}: drops {drops if drops is not None else 'n/a'}, ratio {format_percent(ratio)}"


def format_hz(value: Any) -> str:
    return f"{float(value):.1f}Hz" if is_number(value) else "n/a"


def format_seconds(value: Any) -> str:
    return f"{float(value) * 1000.0:.1f}ms" if is_number(value) else "n/a"


def format_percent(value: Any) -> str:
    return f"{float(value) * 100.0:.1f}%" if is_number(value) else "n/a"


def audit_summary_text(checks: list[dict[str, Any]]) -> str:
    required = [check for check in checks if check.get("required") is not False]
    passed = sum(1 for check in required if check.get("ok"))
    optional_failed = sum(1 for check in checks if check.get("required") is False and not check.get("ok"))
    suffix = f", {optional_failed} optional warning(s)" if optional_failed else ""
    return f"{passed}/{len(required)} required checks passing{suffix}"


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


def read_jsonl_relaxed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def estimate_recording_seconds_offset(rows: list[dict[str, Any]], *perf_keys: str) -> float | None:
    deltas: list[float] = []
    for row in rows:
        quest_seconds = row.get("quest_recording_timestamp_seconds")
        if not is_number(quest_seconds):
            continue
        for key in perf_keys:
            perf_seconds = row.get(key)
            if is_number(perf_seconds):
                deltas.append(float(quest_seconds) - float(perf_seconds))
                break
    if not deltas:
        return None
    deltas.sort()
    return float(deltas[len(deltas) // 2])


def row_base_ee_transform(row: dict[str, Any]) -> np.ndarray | None:
    transform = transform_from_json(row.get("T_base_ee"))
    if transform is not None:
        return transform
    state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
    return robot_state_pose_transform(state) if isinstance(state, dict) else None


def row_explicit_tool_tcp_transform(row: dict[str, Any]) -> np.ndarray | None:
    transform = transform_from_json(row.get("T_base_tool_tcp"))
    if transform is not None:
        return transform
    state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
    tcp = state.get("tcp_pose") if isinstance(state, dict) and isinstance(state.get("tcp_pose"), dict) else None
    if not isinstance(tcp, dict):
        return None
    transform = transform_from_json(tcp.get("T_base_pose"))
    if transform is not None:
        return transform
    return flexiv_pose_payload_to_transform(tcp.get("pose"))


def estimate_pose_to_tool_tcp_transform(rows: list[dict[str, Any]]) -> np.ndarray | None:
    transforms: list[np.ndarray] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        base_ee = row_base_ee_transform(row)
        base_tool = row_explicit_tool_tcp_transform(row)
        if base_ee is None or base_tool is None:
            continue
        delta = invert_transform(base_ee) @ base_tool
        if np.all(np.isfinite(delta)) and np.linalg.norm(delta[:3, 3]) < 1.0:
            transforms.append(delta)
    if not transforms:
        return None
    return average_transforms(transforms)


def calibration_tool_tcp_rows(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    candidates: list[Path] = []
    run_dir = result.get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        candidates.append(Path(run_dir) / "samples.jsonl")
    source_path = result.get("sourcePath")
    if isinstance(source_path, str) and source_path:
        source = Path(source_path)
        candidates.append(source.parent / "samples.jsonl")
    for path in candidates:
        try:
            rows = read_jsonl_relaxed(path)
        except Exception:
            rows = []
        if rows:
            return rows
    return []


def robot_replay_row(
    row: dict[str, Any],
    robot_dir: Path,
    origin: list[float],
    t_world_base: np.ndarray | None,
    t_ee_end_camera: np.ndarray | None,
    t_pose_tool_tcp: np.ndarray | None = None,
    *,
    recording_timestamp_seconds: float | None,
    source_stream: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    ee_matrix = transform_from_json(row.get("T_base_ee"))
    state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
    if ee_matrix is None and isinstance(state, dict):
        ee_matrix = robot_state_pose_transform(state)
    pose_matrix = robot_row_tool_transform(row, t_pose_tool_tcp)
    if pose_matrix is None:
        pose_matrix = ee_matrix
    if ee_matrix is None:
        ee_matrix = pose_matrix
    ee_pose = transform_payload_from_matrix(ee_matrix) if ee_matrix is not None else None
    pose = transform_payload_from_matrix(pose_matrix) if pose_matrix is not None else None
    end_camera_pose = row.get("T_base_end_camera") if isinstance(row.get("T_base_end_camera"), dict) else None
    end_camera_matrix = transform_from_json(end_camera_pose)
    if end_camera_matrix is not None:
        end_camera_pose = transform_payload_from_matrix(end_camera_matrix)
    world_matrix = transform_from_json(row.get("T_world_tool_tcp"))
    world_pose = transform_payload_from_matrix(world_matrix) if world_matrix is not None else None
    display_pose = None
    world_end_camera_matrix = transform_from_json(row.get("T_world_end_camera"))
    world_end_camera_pose = (
        transform_payload_from_matrix(world_end_camera_matrix) if world_end_camera_matrix is not None else None
    )
    display_end_camera_pose = None
    display_frame = "unaligned_robot_base"
    if pose_matrix is not None:
        if end_camera_matrix is None and t_ee_end_camera is not None:
            end_camera_matrix = pose_matrix @ t_ee_end_camera
            end_camera_pose = transform_payload_from_matrix(end_camera_matrix)
        if t_world_base is not None and world_pose is None:
            world_matrix = t_world_base @ pose_matrix
            world_pose = transform_payload_from_matrix(world_matrix)
        if world_pose is not None:
            display_pose = translate_transform_payload(world_pose, origin)
        if t_world_base is not None and end_camera_matrix is not None and world_end_camera_pose is None:
            world_end_camera_matrix = t_world_base @ end_camera_matrix
            world_end_camera_pose = transform_payload_from_matrix(world_end_camera_matrix)
        if world_end_camera_pose is not None:
            display_end_camera_pose = translate_transform_payload(world_end_camera_pose, origin)
        if world_pose is not None or world_end_camera_pose is not None:
            display_frame = "quest_world_axes_translated_to_board_origin"
    videos = row.get("videos") if isinstance(row.get("videos"), dict) else {}
    images = row.get("images") if isinstance(row.get("images"), dict) else {}
    result = {
        "sampleIndex": row.get("sample_index"),
        "questSampleIndex": row.get("quest_sample_index"),
        "recordingTimestampSeconds": recording_timestamp_seconds,
        "pcPerfCounterSeconds": row.get("pc_perf_counter_seconds"),
        "pcUnixSeconds": row.get("pc_unix_seconds"),
        "capturedAt": row.get("captured_at"),
        "ok": bool(row.get("ok")),
        "sourceStream": source_stream,
        "T_base_ee": ee_pose,
        "T_base_tool_tcp": pose,
        "T_world_tool_tcp": world_pose,
        "T_display_tool_tcp": display_pose,
        "T_world_ee": world_pose,
        "T_display_ee": display_pose,
        "T_base_end_camera": end_camera_pose,
        "T_world_end_camera": world_end_camera_pose,
        "T_display_end_camera": display_end_camera_pose,
        "displayFrame": display_frame,
        "jointpose": row.get("jointpose") or row.get("jointpos"),
        "jointpos": row.get("jointpos") or row.get("jointpose"),
        "images": robot_image_artifacts(robot_dir, images),
        "videos": robot_video_artifacts(robot_dir, videos),
        "videoFrames": row.get("videoFrames"),
        "questGaze3DWorld": row.get("quest_gaze3d_pc_world") or pc_world_vec3(row.get("quest_gaze3d_world")),
        "questGaze3DUnityWorld": row.get("quest_gaze3d_world"),
        "questGaze3DSource": row.get("quest_gaze3d_source"),
        "gripper": row.get("gripper"),
        "poseDiversity": row.get("poseDiversity"),
        "error": row.get("error"),
    }
    return result, pose_matrix


def build_robot_realsense_replay(session_dir: Path, origin: list[float]) -> dict[str, Any] | None:
    robot_dir = session_dir / "robot_realsense"
    samples_path = robot_dir / "samples.jsonl"
    if not samples_path.exists():
        return None
    session = read_json_if_exists(robot_dir / "session_summary.json")
    config = read_json_if_exists(robot_dir / "capture_config.json")
    result = read_json_if_exists(robot_dir / "robot_hand_eye_result.json")
    failure = read_json_if_exists(robot_dir / "robot_hand_eye_failure.json")
    alignment = result.get("questAlignment") if isinstance(result, dict) else None
    t_world_base = transform_from_json(alignment.get("T_world_base")) if isinstance(alignment, dict) else None
    t_ee_end_camera = None
    if isinstance(result, dict):
        end_camera_result = result.get("end_camera")
        if isinstance(end_camera_result, dict):
            t_ee_end_camera = transform_from_json(end_camera_result.get("T_ee_realsense"))
    sample_rows_raw = read_jsonl_relaxed(samples_path)
    robot_states_path = robot_dir / "robot_states.jsonl"
    robot_state_rows_raw = read_jsonl_relaxed(robot_states_path)
    t_pose_tool_tcp = estimate_pose_to_tool_tcp_transform(sample_rows_raw + robot_state_rows_raw)
    tool_tcp_source = "record"
    if t_pose_tool_tcp is None:
        calibration_rows = calibration_tool_tcp_rows(result if isinstance(result, dict) else None)
        t_pose_tool_tcp = estimate_pose_to_tool_tcp_transform(calibration_rows)
        tool_tcp_source = "calibration_run" if t_pose_tool_tcp is not None else None
    recording_offset = estimate_recording_seconds_offset(
        sample_rows_raw,
        "robot_state_pc_perf_counter_seconds",
        "pc_perf_counter_seconds",
    )
    rows: list[dict[str, Any]] = []
    robot_state_rows: list[dict[str, Any]] = []
    ee_poses: list[np.ndarray] = []
    for row in sample_rows_raw:
        replay_row, pose_matrix = robot_replay_row(
            row,
            robot_dir,
            origin,
            t_world_base,
            t_ee_end_camera,
            t_pose_tool_tcp,
            recording_timestamp_seconds=float(row["quest_recording_timestamp_seconds"])
            if is_number(row.get("quest_recording_timestamp_seconds"))
            else None,
            source_stream="quest_aligned_samples",
        )
        rows.append(replay_row)
        if pose_matrix is not None:
            ee_poses.append(pose_matrix)
    for row in robot_state_rows_raw:
        derived_recording_seconds = None
        if is_number(row.get("quest_recording_timestamp_seconds")):
            derived_recording_seconds = float(row["quest_recording_timestamp_seconds"])
        elif recording_offset is not None and is_number(row.get("pc_perf_counter_seconds")):
            derived_recording_seconds = float(row["pc_perf_counter_seconds"]) + recording_offset
        replay_row, pose_matrix = robot_replay_row(
            row,
            robot_dir,
            origin,
            t_world_base,
            t_ee_end_camera,
            t_pose_tool_tcp,
            recording_timestamp_seconds=derived_recording_seconds,
            source_stream="robot_states",
        )
        robot_state_rows.append(replay_row)
        if pose_matrix is not None:
            ee_poses.append(pose_matrix)
    if not robot_state_rows:
        robot_state_rows = list(rows)
    video_frame_rows = read_jsonl_relaxed(robot_dir / "video_frames.jsonl")
    depth_frame_count = sum(
        1
        for row in video_frame_rows
        if isinstance(row.get("depth"), dict) and row["depth"].get("path")
    )
    gripper_rows = read_robot_gripper_rows(robot_dir)
    performance = robot_realsense_performance_summary(
        robot_dir,
        session if isinstance(session, dict) else {},
        config if isinstance(config, dict) else {},
    )
    return {
        "directory": str(robot_dir),
        "session": session if isinstance(session, dict) else None,
        "config": config if isinstance(config, dict) else None,
        "samples": rows,
        "robotStates": robot_state_rows,
        "videoFrameCount": len(video_frame_rows),
        "depthFrameCount": depth_frame_count,
        "performance": performance,
        "gripper": gripper_rows,
        "displayFrame": "quest_world_axes_translated_to_board_origin" if t_world_base is not None else "unaligned_robot_base",
        "poseDiversity": ee_pose_diversity(ee_poses),
        "recordingSecondsOffsetFromPerfCounter": recording_offset,
        "toolTcpFallback": {
            "source": tool_tcp_source,
            "T_pose_tool_tcp": transform_payload_from_matrix(t_pose_tool_tcp) if t_pose_tool_tcp is not None else None,
        },
        "result": result if isinstance(result, dict) else None,
        "failure": failure if isinstance(failure, dict) else None,
    }


def robot_image_artifacts(robot_dir: Path, images: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for role, rel_path in images.items():
        if not isinstance(rel_path, str) or not rel_path:
            continue
        path = resolve_child_path(robot_dir, rel_path)
        if path is None:
            payload[role] = {"label": f"{role} camera", "path": rel_path, "url": None, "error": "path outside robot directory"}
            continue
        if path.exists():
            payload[role] = artifact_payload(path, f"{role} camera")
        else:
            payload[role] = {"label": f"{role} camera", "path": str(path), "url": None, "error": "missing image"}
    return payload


def robot_video_artifacts(robot_dir: Path, videos: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for role, item in videos.items():
        if isinstance(item, dict):
            rel_path = item.get("path")
            frame_index = item.get("frameIndex")
            serial = item.get("serial")
        else:
            rel_path = item
            frame_index = None
            serial = None
        if not isinstance(rel_path, str) or not rel_path:
            continue
        path = resolve_child_path(robot_dir, rel_path)
        if path is None:
            payload[role] = {
                "label": f"{role} camera video",
                "path": rel_path,
                "url": None,
                "frameIndex": frame_index,
                "serial": serial,
                "error": "path outside robot directory",
            }
            continue
        if path.exists():
            artifact = artifact_payload(path, f"{role} camera video")
            artifact["frameIndex"] = frame_index
            artifact["serial"] = serial
            payload[role] = artifact
        else:
            payload[role] = {
                "label": f"{role} camera video",
                "path": str(path),
                "url": None,
                "frameIndex": frame_index,
                "serial": serial,
                "error": "missing video",
            }
    return payload


def read_robot_gripper_rows(robot_dir: Path) -> list[dict[str, Any]]:
    path = robot_dir / "gripper_commands.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(
                    {
                        "ok": bool(row.get("ok")),
                        "commandSent": bool(row.get("commandSent")),
                        "action": row.get("action"),
                        "questSampleIndex": row.get("quest_sample_index"),
                        "recordingTimestampSeconds": row.get("quest_recording_timestamp_seconds"),
                        "trigger": row.get("trigger"),
                        "targetWidthM": row.get("target_width_m"),
                        "reason": row.get("reason"),
                        "error": row.get("error"),
                        "status": row.get("status"),
                    }
                )
    return rows


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


def default_board_matrix_4x4() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


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


def time_series_summary(times: list[float], target_hz: float | None = None) -> dict[str, Any]:
    finite = np.array([float(value) for value in times if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {"count": 0}
    duration = float(max(0.0, finite[-1] - finite[0])) if finite.size >= 2 else 0.0
    hz = float((finite.size - 1) / duration) if duration > 1e-9 and finite.size >= 2 else None
    payload: dict[str, Any] = {
        "count": int(finite.size),
        "durationSeconds": duration,
        "effectiveHz": hz,
        "firstPerfCounterSeconds": float(finite[0]),
        "lastPerfCounterSeconds": float(finite[-1]),
    }
    if target_hz is not None and target_hz > 0:
        payload["targetHz"] = float(target_hz)
        if hz is not None:
            payload["targetRatio"] = float(hz / float(target_hz))
    if finite.size >= 2:
        payload["gapSeconds"] = stats_summary(list(np.diff(finite)))
    return payload


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
        gaze_point = pc_world_vec3(row.get("gazePoint3DWorld"))
        ray_origin = pc_world_vec3(row.get("gazeRayOrigin"))
        ray_direction_list = pc_world_direction(row.get("gazeRayDirection"))
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


def translate_pc_vec3_list(value: Any, origin: list[float]) -> list[float] | None:
    point = pc_world_vec3(value)
    if point is None:
        return None
    return [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]]


def pc_world_vec3(value: Any) -> list[float] | None:
    return unity_vec3_to_pc(value)


def pc_world_direction(value: Any) -> list[float] | None:
    direction = unity_vec3_to_pc(value)
    if direction is None:
        return None
    norm = math.sqrt(sum(item * item for item in direction))
    if norm <= 1e-12:
        return None
    return [item / norm for item in direction]


def pc_world_pose_array(value: Any) -> list[float] | None:
    return unity_pose_array_to_pc(value)


def pc_world_controller_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result = dict(value)
    position = unity_vec3_to_pc(value.get("position"))
    rotation = unity_quaternion_wxyz_to_pc(value.get("rotation"))
    pose = unity_pose_array_to_pc(value.get("pose"))
    if position is not None:
        result["position"] = position
    if rotation is not None:
        result["rotation"] = rotation
    if pose is not None:
        result["pose"] = pose
    result["coordinateFrame"] = PC_WORLD_FRAME
    result["sourceCoordinateFrame"] = value.get("coordinateFrame") or UNITY_WORLD_FRAME
    return result


def pc_world_sample_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    payload: dict[str, Any] = {
        "coordinateFrame": PC_WORLD_FRAME,
        "sourceCoordinateFrame": UNITY_WORLD_FRAME,
        "worldFrameConversion": WORLD_FRAME_CONVERSION,
    }
    mappings = {
        "gazePointWorld": pc_world_vec3,
        "gazePoint3DWorld": pc_world_vec3,
        "gazeRayOrigin": pc_world_vec3,
        "gazeRayDirection": pc_world_direction,
        "gazeFallbackPointWorld": pc_world_vec3,
        "leftCameraPose": pc_world_pose_array,
        "rightCameraPose": pc_world_pose_array,
        "leftPassthroughCameraPose": pc_world_pose_array,
        "rightPassthroughCameraPose": pc_world_pose_array,
        "leftEyePose": pc_world_pose_array,
        "rightEyePose": pc_world_pose_array,
        "leftEyePosition": pc_world_vec3,
        "rightEyePosition": pc_world_vec3,
    }
    for key, converter in mappings.items():
        converted = converter(message.get(key))
        if converted is not None:
            payload[key] = converted
    for key in ("leftController", "rightController"):
        converted_controller = pc_world_controller_payload(message.get(key))
        if converted_controller is not None:
            payload[key] = converted_controller
    return payload if len(payload) > 3 else None


def replay_pose_from_array(value: Any, source: str, origin: list[float]) -> dict[str, Any]:
    pose = pose_from_pose_array(pc_world_pose_array(value), source)
    if pose["ok"] and pose["p"] is not None:
        pose["p"] = [pose["p"][0] - origin[0], pose["p"][1] - origin[1], pose["p"][2] - origin[2]]
    return pose


def replay_pose_from_controller(value: Any, handedness: str, origin: list[float]) -> dict[str, Any]:
    pose = visualizer_controller_pose(pc_world_controller_payload(value), handedness)
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
            "ok": translate_pc_vec3_list(row.get("gazePoint3DWorld"), origin) is not None,
            "p": translate_pc_vec3_list(row.get("gazePoint3DWorld"), origin),
            "source": row.get("gazePoint3DSource") or row.get("gazeSource"),
        },
        "gazeHit": {
            "ok": translate_pc_vec3_list(row.get("gazePointWorld"), origin) is not None,
            "p": translate_pc_vec3_list(row.get("gazePointWorld"), origin),
            "source": "gazePointWorld",
        },
        "gazeRayOrigin": translate_pc_vec3_list(row.get("gazeRayOrigin"), origin),
        "gazeRayDirection": pc_world_direction(row.get("gazeRayDirection")),
    }


def visualizer_event_from_sample(message: dict[str, Any], wrapper: dict[str, Any]) -> dict[str, Any]:
    left_camera = pose_from_pose_array(pc_world_pose_array(message.get("leftCameraPose")), "leftCamera")
    right_camera = pose_from_pose_array(pc_world_pose_array(message.get("rightCameraPose")), "rightCamera")
    left_eye = pose_from_pose_array(pc_world_pose_array(message.get("leftEyePose")), "leftEye")
    right_eye = pose_from_pose_array(pc_world_pose_array(message.get("rightEyePose")), "rightEye")
    head = head_pose_from_pair(left_eye, right_eye, "eye_midpoint")
    if not head["ok"]:
        head = head_pose_from_pair(left_camera, right_camera, "camera_midpoint")
    left = visualizer_controller_pose(pc_world_controller_payload(message.get("leftController")), "left")
    right = visualizer_controller_pose(pc_world_controller_payload(message.get("rightController")), "right")
    gaze = pc_world_vec3(message.get("gazePoint3DWorld"))

    return {
        "type": "sample",
        "coordinateFrame": PC_WORLD_FRAME,
        "sourceCoordinateFrame": UNITY_WORLD_FRAME,
        "worldFrameConversion": WORLD_FRAME_CONVERSION,
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
    input_summary = controller_input_summary(value)
    if not value.get("hasPose"):
        reason = value.get("missingReason")
        pose = missing_visualizer_pose(reason if isinstance(reason, str) and reason else "hasPose=false")
        pose["input"] = input_summary
        return pose

    position = vec3_list(value.get("position"))
    rotation = quat_list(value.get("rotation"))
    if position is None:
        pose = pose_from_pose_array(value.get("pose"), value.get("source") or handedness)
        if pose["ok"]:
            pose["input"] = input_summary
            return pose
        pose = missing_visualizer_pose("missing_controller_position")
        pose["input"] = input_summary
        return pose

    return {
        "ok": True,
        "source": value.get("source") or handedness,
        "p": position,
        "q": rotation or [1.0, 0.0, 0.0, 0.0],
        "input": input_summary,
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
    coordinate_frame = None
    if isinstance(calibration, dict):
        coordinate_frame = (
            calibration.get("coordinate_frame")
            or calibration.get("coordinateFrame")
            or result.get("coordinate_frame")
            or result.get("coordinateFrame")
        )
    raw_trajectory_frame = result.get("raw_trajectory_frame") or result.get("rawTrajectoryFrame") or UNITY_WORLD_FRAME
    conversion = result.get("world_frame_conversion") or result.get("worldFrameConversion") or WORLD_FRAME_CONVERSION
    t_world_board_raw = calibration.get("T_world_board") if isinstance(calibration, dict) else None
    t_board_world_raw = calibration.get("T_board_world") if isinstance(calibration, dict) else None
    t_world_board = ensure_pc_transform_payload(t_world_board_raw, coordinate_frame)
    t_board_world = None
    if t_world_board is not None:
        matrix = matrix_from_transform_payload(t_world_board)
        if matrix is not None:
            t_board_world = coordinate_transform_payload_from_matrix(np.linalg.inv(matrix), PC_WORLD_FRAME)
    if t_board_world is None:
        t_board_world = ensure_pc_transform_payload(t_board_world_raw, coordinate_frame)
    t_unity_world_board = calibration.get("T_unity_world_board") if isinstance(calibration, dict) else None
    t_board_unity_world = calibration.get("T_board_unity_world") if isinstance(calibration, dict) else None
    if not is_pc_world_frame(coordinate_frame):
        t_unity_world_board = t_unity_world_board or t_world_board_raw
        t_board_unity_world = t_board_unity_world or t_board_world_raw
    board_normal = None
    board_normal_angle_z = None
    if isinstance(t_world_board, dict):
        rotation_matrix = t_world_board.get("rotation_matrix")
        if isinstance(rotation_matrix, list) and len(rotation_matrix) >= 3:
            try:
                board_normal = [
                    float(rotation_matrix[0][2]),
                    float(rotation_matrix[1][2]),
                    float(rotation_matrix[2][2]),
                ]
                dot_z = max(-1.0, min(1.0, abs(board_normal[2])))
                board_normal_angle_z = math.degrees(math.acos(dot_z))
            except (TypeError, ValueError, IndexError):
                board_normal = None
                board_normal_angle_z = None
    quest_world_origin_in_board = (
        t_board_world.get("translation_m")
        if isinstance(t_board_world, dict) and isinstance(t_board_world.get("translation_m"), list)
        else None
    )
    order = result.get("order_summary") if isinstance(result.get("order_summary"), dict) else {}
    stats_obj = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    overall = stats_obj.get("overall") if isinstance(stats_obj.get("overall"), dict) else {}
    unity_quest_world_origin_in_board = None
    if isinstance(calibration, dict):
        unity_quest_world_origin_in_board = (
            calibration.get("unity_quest_world_origin_in_board_m")
            or calibration.get("quest_world_origin_in_board_m")
        )
    return {
        "type": "calibration_result",
        "recordId": record_id,
        "resultPath": str(result_path),
        "pattern": result.get("pattern") or [11, 8],
        "squareSizeM": result.get("square_size_m") or 0.025,
        "imageYAxis": result.get("image_y_axis"),
        "coordinateFrame": PC_WORLD_FRAME,
        "rawTrajectoryFrame": raw_trajectory_frame,
        "worldFrameConversion": conversion,
        "T_world_board": t_world_board,
        "T_board_world": t_board_world,
        "T_unity_world_board": t_unity_world_board,
        "T_board_unity_world": t_board_unity_world,
        "questWorldOriginInBoardM": quest_world_origin_in_board,
        "unityQuestWorldOriginInBoardM": unity_quest_world_origin_in_board,
        "boardNormalWorld": board_normal,
        "boardNormalAbsAngleToWorldZDeg": board_normal_angle_z,
        "bestLagSeconds": result.get("best_lag_seconds"),
        "keptFrames": order.get("kept_frames"),
        "inputFrames": order.get("input_frames"),
        "appearanceAnchorFrames": order.get("appearance_anchor_frames"),
        "reprojectionOrderFrames": order.get("reprojection_frames"),
        "rot180Frames": order.get("rot180_frames"),
        "appearanceAnchorPolicy": result.get("appearance_anchor_policy") if isinstance(result.get("appearance_anchor_policy"), dict) else {},
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
    compact = canonicalize_calibration_snapshot({
        "capturedUtc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(snapshot.get("ok")),
        "kind": snapshot.get("kind"),
        "recordId": snapshot.get("recordId"),
        "resultPath": event.get("resultPath"),
        "pattern": event.get("pattern"),
        "squareSizeM": event.get("squareSizeM"),
        "imageYAxis": event.get("imageYAxis"),
        "coordinateFrame": event.get("coordinateFrame"),
        "rawTrajectoryFrame": event.get("rawTrajectoryFrame"),
        "worldFrameConversion": event.get("worldFrameConversion"),
        "T_world_board": event.get("T_world_board"),
        "T_board_world": event.get("T_board_world"),
        "T_unity_world_board": event.get("T_unity_world_board"),
        "T_board_unity_world": event.get("T_board_unity_world"),
        "questWorldOriginInBoardM": event.get("questWorldOriginInBoardM"),
        "unityQuestWorldOriginInBoardM": event.get("unityQuestWorldOriginInBoardM"),
        "boardNormalWorld": event.get("boardNormalWorld"),
        "boardNormalAbsAngleToWorldZDeg": event.get("boardNormalAbsAngleToWorldZDeg"),
        "bestLagSeconds": event.get("bestLagSeconds"),
        "keptFrames": event.get("keptFrames"),
        "inputFrames": event.get("inputFrames"),
        "appearanceAnchorFrames": event.get("appearanceAnchorFrames"),
        "reprojectionOrderFrames": event.get("reprojectionOrderFrames"),
        "rot180Frames": event.get("rot180Frames"),
        "medianReprojectionPx": event.get("medianReprojectionPx"),
        "p90ReprojectionPx": event.get("p90ReprojectionPx"),
    })
    if snapshot.get("kind") != "result":
        compact["reason"] = event.get("reason")
        compact["reasonCode"] = event.get("reasonCode")
    return compact


def latest_robot_hand_eye_result(
    calibration_raw_root: Path | None,
    calibration_output_root: Path | None,
    recording_root: Path | None = None,
) -> dict[str, Any] | None:
    candidates: list[Path] = []
    for root in (calibration_raw_root, calibration_output_root, recording_root):
        if root is None:
            continue
        resolved = root.resolve()
        if not resolved.exists():
            continue
        candidates.extend(resolved.glob("record_pc_calib*/robot_realsense/robot_hand_eye_result.json"))
        candidates.extend(resolved.glob("record*/robot_realsense/robot_hand_eye_result.json"))
    valid: list[Path] = []
    for path in candidates:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        payload = read_json_if_exists(path)
        if not isinstance(payload, dict) or not payload.get("ok"):
            continue
        alignment = payload.get("questAlignment")
        if not isinstance(alignment, dict) or not alignment.get("ok"):
            continue
        valid.append(path)
    if not valid:
        return None
    latest = max(valid, key=lambda path: path.stat().st_mtime)
    payload = read_json_if_exists(latest)
    if isinstance(payload, dict):
        payload.setdefault("sourcePath", str(latest))
        return payload
    return None


def latest_robot_calibration_event(
    calibration_raw_root: Path | None,
    calibration_output_root: Path | None,
    recording_root: Path | None = None,
) -> dict[str, Any] | None:
    payload = latest_robot_hand_eye_result(calibration_raw_root, calibration_output_root, recording_root)
    if not isinstance(payload, dict):
        return None
    alignment = payload.get("questAlignment") if isinstance(payload.get("questAlignment"), dict) else {}
    event = {
        "type": "robot_calibration_result",
        "recordId": payload.get("record_id"),
        "runDir": str(Path(str(payload.get("sourcePath"))).parent) if payload.get("sourcePath") else None,
        "result": compact_robot_result(payload),
        "T_world_base": alignment.get("T_world_base"),
        "sourcePath": payload.get("sourcePath"),
        "restored": True,
    }
    return event


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

    def failure_payload(path: Path) -> dict[str, Any]:
        log_path = path / "calibration_run.log"
        return {
            "kind": "failure",
            "recordId": path.name,
            "event": calibration_failure_event(path.name, path, log_path, None),
        }

    latest_any = max(candidates, key=candidate_stamp)
    latest_success = max(
        (path for path in candidates if (path / "calibration_result_25mm.json").exists()),
        key=lambda path: (path / "calibration_result_25mm.json").stat().st_mtime,
        default=None,
    )
    if latest_success is None:
        latest_failure = failure_payload(latest_any)
        return {
            "ok": True,
            **latest_failure,
            "latestFailure": latest_failure,
        }

    result_path = latest_success / "calibration_result_25mm.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
        latest_failure = None
        if latest_any != latest_success and (latest_any / "calibration_failure_25mm.json").exists():
            latest_failure = failure_payload(latest_any)
        payload = {
            "ok": True,
            "kind": "result",
            "recordId": latest_success.name,
            "event": calibration_result_event(latest_success.name, result, result_path),
        }
        if latest_failure is not None:
            payload["latestFailure"] = latest_failure
        return payload
    latest_failure = failure_payload(latest_any)
    return {"ok": True, **latest_failure, "latestFailure": latest_failure}


def calibration_session_dirs(root: Path | None) -> list[Path]:
    if root is None:
        return []
    resolved = root.resolve()
    if not resolved.exists():
        return []
    return [
        path
        for path in resolved.iterdir()
        if path.is_dir() and path.name.startswith("record_pc_calib")
    ]


def remove_directory_inside_root(path: Path, root: Path) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"refusing to remove path outside calibration root: {resolved_path}")
    shutil.rmtree(resolved_path)
    return True


def remove_calibration_dirs(paths: list[Path], roots: list[Path]) -> list[str]:
    removed: list[str] = []
    resolved_roots = [root.resolve() for root in roots if root is not None]
    for path in paths:
        if not path.exists():
            continue
        matching_root = next((root for root in resolved_roots if root in path.resolve().parents), None)
        if matching_root is None:
            continue
        remove_directory_inside_root(path, matching_root)
        removed.append(str(path.resolve()))
    return removed


def keep_only_latest_calibration_state(
    raw_root: Path | None,
    output_root: Path | None,
    keep_raw_name: str,
    keep_output_name: str,
) -> dict[str, Any]:
    roots = [root.resolve() for root in (raw_root, output_root) if root is not None]
    remove: list[Path] = []
    for root in roots:
        for path in calibration_session_dirs(root):
            if path.name in (keep_raw_name, keep_output_name):
                continue
            remove.append(path)
    removed = remove_calibration_dirs(remove, roots)
    return {
        "ok": True,
        "removed": removed,
        "removedCount": len(removed),
        "keptRawName": keep_raw_name,
        "keptOutputName": keep_output_name,
    }


def clear_latest_calibration_state(
    raw_root: Path | None,
    output_root: Path | None,
    robot_manager: FlexivRealSenseManager | None = None,
) -> dict[str, Any]:
    roots = [root.resolve() for root in (raw_root, output_root) if root is not None]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(calibration_session_dirs(root))
    removed = remove_calibration_dirs(candidates, roots)
    if robot_manager is not None:
        with robot_manager.lock:
            robot_manager.last_calibration = None
    return {
        "ok": True,
        "cleared": True,
        "removed": removed,
        "removedCount": len(removed),
        "message": "Calibration state cleared",
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


def atomic_write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def rizon4_model_payload() -> dict[str, Any]:
    if not DEFAULT_RIZON_URDF.exists():
        return {"ok": False, "reason": "missing_urdf", "path": str(DEFAULT_RIZON_URDF)}
    try:
        root = ET.fromstring(DEFAULT_RIZON_URDF.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": "bad_urdf", "error": str(exc), "path": str(DEFAULT_RIZON_URDF)}

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

    links: dict[str, Any] = {}
    for link in root.findall("link"):
        link_name = str(link.attrib.get("name") or "")
        if not link_name:
            continue
        visuals: list[dict[str, Any]] = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = str(mesh.attrib.get("filename") or "")
            asset_path = package_mesh_asset_path(filename)
            visual_row: dict[str, Any] = {
                "name": visual.attrib.get("name") or "",
                "xyz": parse_float_triplet(origin.attrib.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0]),
                "rpy": parse_float_triplet(origin.attrib.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0]),
                "mesh": {
                    "filename": filename,
                    "assetPath": asset_path,
                    "url": "/robot/mesh?path=" + quote_path(asset_path) if asset_path else None,
                    "exists": resolve_robot_mesh_path(asset_path).exists() if asset_path else False,
                    "scale": parse_float_triplet(mesh.attrib.get("scale"), [1.0, 1.0, 1.0]),
                },
            }
            visuals.append(visual_row)
        links[link_name] = {"visuals": visuals}

    return {
        "ok": True,
        "name": root.attrib.get("name") or "Rizon4",
        "source": str(DEFAULT_RIZON_URDF),
        "sourceLabel": "Flexiv RDK 1.8 Rizon4 resources",
        "joints": joints,
        "links": links,
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
        input_samples = 0
        missing: dict[str, int] = {}
        sources: dict[str, int] = {}
        latest: dict[str, Any] = {}
        latest_input: dict[str, Any] = {}
        latest_age = None
        latest_valid_age = None
        latest_valid_source = None
        latest_input_age = None
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
            input_summary = controller_input_summary(controller)
            if input_summary["hasAny"]:
                input_samples += 1
                if not latest_input:
                    latest_input = input_summary
                    row_timestamp = row.get("pcReceiveUnixSeconds")
                    latest_input_age = max(0.0, now - float(row_timestamp)) if is_number(row_timestamp) else None

        return {
            "samples": len(rows),
            "validSamples": valid,
            "inputSamples": input_samples,
            "latestAgeSeconds": latest_age,
            "latestValidAgeSeconds": latest_valid_age,
            "latestValidSource": latest_valid_source,
            "latestInputAgeSeconds": latest_input_age,
            "latest": latest,
            "latestInput": latest_input,
            "sources": sources,
            "missing": missing,
        }

    return {
        "windowSeconds": window_seconds,
        "sampleCount": len(rows),
        "left": summarize("left"),
        "right": summarize("right"),
    }


def controller_input_summary(controller: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(controller, dict):
        controller = {}
    nested_input = controller.get("input")
    if isinstance(nested_input, dict) and nested_input.get("hasAny"):
        return {
            "hasAny": True,
            "handTrigger": nested_input.get("handTrigger"),
            "indexTrigger": nested_input.get("indexTrigger"),
            "handTriggerPressed": nested_input.get("handTriggerPressed"),
            "indexTriggerPressed": nested_input.get("indexTriggerPressed"),
            "aButton": nested_input.get("aButton"),
            "bButton": nested_input.get("bButton"),
            "teleopHeld": bool(nested_input.get("teleopHeld")),
            "teleopThreshold": nested_input.get("teleopThreshold") if is_number(nested_input.get("teleopThreshold")) else 0.65,
        }
    buttons = controller.get("buttons")
    if not isinstance(buttons, dict):
        buttons = {}

    def first_float(*keys: str) -> float | None:
        for key in keys:
            value = controller.get(key)
            if is_number(value):
                return max(0.0, min(1.0, float(value)))
            value = buttons.get(key)
            if is_number(value):
                return max(0.0, min(1.0, float(value)))
        return None

    def first_bool(*keys: str) -> bool | None:
        for key in keys:
            value = controller.get(key)
            if isinstance(value, bool):
                return value
            value = buttons.get(key)
            if isinstance(value, bool):
                return value
        return None

    hand_trigger = first_float("handTrigger", "rightHandTrigger", "grip", "primaryHandTrigger")
    index_trigger = first_float("indexTrigger", "rightIndexTrigger", "trigger", "primaryIndexTrigger")
    hand_pressed = first_bool("handTriggerPressed", "rightHandTriggerPressed", "gripPressed", "gripButton")
    index_pressed = first_bool("indexTriggerPressed", "rightIndexTriggerPressed", "triggerPressed", "triggerButton")
    a_button = first_bool("aButton", "buttonA", "primaryButton")
    b_button = first_bool("bButton", "buttonB", "secondaryButton")
    has_any = any(
        value is not None
        for value in (hand_trigger, index_trigger, hand_pressed, index_pressed, a_button, b_button)
    )
    teleop_held = bool(hand_pressed) or (hand_trigger is not None and hand_trigger >= 0.65)
    return {
        "hasAny": has_any,
        "handTrigger": hand_trigger,
        "indexTrigger": index_trigger,
        "handTriggerPressed": hand_pressed,
        "indexTriggerPressed": index_pressed,
        "aButton": a_button,
        "bButton": b_button,
        "teleopHeld": teleop_held,
        "teleopThreshold": 0.65,
    }


def controller_input_text(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary.get("hasAny"):
        return "input missing"
    hand = summary.get("handTrigger")
    index = summary.get("indexTrigger")
    parts = []
    if is_number(hand):
        parts.append(f"hand={float(hand):.2f}")
    elif summary.get("handTriggerPressed") is not None:
        parts.append("hand=n/a")
    if summary.get("handTriggerPressed") is not None:
        parts.append(f"handPressed={bool(summary.get('handTriggerPressed'))}")
    if is_number(index):
        parts.append(f"index={float(index):.2f}")
    elif summary.get("indexTriggerPressed") is not None:
        parts.append("index=n/a")
    if summary.get("indexTriggerPressed") is not None:
        parts.append(f"indexPressed={bool(summary.get('indexTriggerPressed'))}")
    if summary.get("aButton") is not None:
        parts.append(f"A={bool(summary.get('aButton'))}")
    if summary.get("bButton") is not None:
        parts.append(f"B={bool(summary.get('bButton'))}")
    parts.append(f"teleopHeld={bool(summary.get('teleopHeld'))}")
    return ", ".join(parts)


def controller_preflight_detail(latest: dict[str, Any], recent: dict[str, Any], prefix: str) -> str:
    latest_ok = bool(latest.get("ok")) if isinstance(latest, dict) else False
    latest_source = str(latest.get("source") or "n/a") if isinstance(latest, dict) else "n/a"
    valid = int(recent.get("validSamples") or 0) if isinstance(recent, dict) else 0
    samples = int(recent.get("samples") or 0) if isinstance(recent, dict) else 0
    input_samples = int(recent.get("inputSamples") or 0) if isinstance(recent, dict) else 0
    window = float(recent.get("windowSeconds", 5.0)) if isinstance(recent, dict) else 5.0
    input_summary = (
        recent.get("latestInput")
        if isinstance(recent.get("latestInput"), dict) and recent.get("latestInput", {}).get("hasAny")
        else controller_input_summary(latest if isinstance(latest, dict) else {})
    )
    text = (
        f"{prefix}latest={latest_ok} ({latest_source}); "
        f"{valid}/{samples} pose valid, {input_samples}/{samples} input in last {window:.0f}s; "
        f"{controller_input_text(input_summary)}"
    )
    latest_valid_age = recent.get("latestValidAgeSeconds") if isinstance(recent, dict) else None
    if is_number(latest_valid_age):
        text += f", last valid {float(latest_valid_age):.2f}s ago"
    latest_input_age = recent.get("latestInputAgeSeconds") if isinstance(recent, dict) else None
    if is_number(latest_input_age):
        text += f", last input {float(latest_input_age):.2f}s ago"
    hint = controller_mode_hint(latest_source)
    if hint:
        text += f"; {hint}"
    elif samples <= 0:
        text += "; no Quest controller samples received in this window"
    elif not latest_ok:
        text += "; Quest controller pose is missing"
    elif not input_summary.get("hasAny"):
        text += "; Quest sample has controller pose but no button/trigger fields"
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
    udp_status: dict[str, Any] | None = None,
    capture_state: dict[str, Any] | None = None,
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
    stream = robot_status.get("realsenseStream") if isinstance(robot_status, dict) else {}
    robot_connected = bool(isinstance(robot, dict) and robot.get("connected"))
    hand_eye_enabled = bool(isinstance(robot_config, dict) and robot_config.get("runHandEye", True))
    camera_serial = str(robot_config.get("cameraSerial") or "") if isinstance(robot_config, dict) else ""
    third_camera_serial = str(robot_config.get("thirdCameraSerial") or "") if isinstance(robot_config, dict) else ""
    cameras = camera_status.get("cameras") if isinstance(camera_status, dict) else []
    camera_serials = {
        str(camera.get("serial"))
        for camera in cameras
        if isinstance(camera, dict) and camera.get("serial") is not None
    }
    camera_ok = bool(camera_serial and camera_serial in camera_serials)
    third_camera_ok = bool(not third_camera_serial or third_camera_serial in camera_serials)

    model_ok = bool(model_status.get("ok")) if isinstance(model_status, dict) else False
    active_joints = model_status.get("activeJointNames") if isinstance(model_status, dict) else []
    active_joint_count = len(active_joints) if isinstance(active_joints, list) else 0
    stream_roles = set(stream.get("roles") or []) if isinstance(stream, dict) else set()
    stream_latest = stream.get("latest") if isinstance(stream, dict) else {}
    stream_running = bool(isinstance(stream, dict) and stream.get("running"))

    def stream_detail(role: str, serial: str) -> tuple[bool, str]:
        latest = stream_latest.get(role) if isinstance(stream_latest, dict) else {}
        sequence = latest.get("sequence") if isinstance(latest, dict) else None
        captured = latest.get("capturedAtUtc") if isinstance(latest, dict) else None
        ok = bool(stream_running and role in stream_roles and sequence is not None)
        detail = f"{serial or 'not configured'}; running={stream_running}, role={role in stream_roles}"
        if sequence is not None:
            detail += f", frame #{sequence}"
        if captured:
            detail += f", {captured}"
        if isinstance(stream, dict) and stream.get("lastError"):
            detail += f", error={stream.get('lastError')}"
        return ok, detail

    end_stream_ok, end_stream_detail = stream_detail("end", camera_serial)
    third_stream_ok, third_stream_detail = stream_detail("third", third_camera_serial)
    udp_status = udp_status if isinstance(udp_status, dict) else {}
    capture_state = capture_state if isinstance(capture_state, dict) else {}
    capture_phase = str(capture_state.get("phase") or "live")
    capture_busy = capture_phase in ("recording", "saving")
    capture_detail = str(capture_state.get("detail") or capture_phase)
    if capture_state.get("recordId"):
        capture_detail += f" record={capture_state.get('recordId')}"
    last_udp_time = udp_status.get("lastReceiveUnixSeconds")
    last_udp_age = max(0.0, now - float(last_udp_time)) if is_number(last_udp_time) else None
    total_datagrams = int(udp_status.get("totalDatagrams") or 0)
    sample_datagrams = int(udp_status.get("sampleDatagrams") or 0)
    udp_detail = (
        f"listening {udp_status.get('bind') or 'n/a'}, datagrams={total_datagrams}, samples={sample_datagrams}"
    )
    if last_udp_age is not None:
        udp_detail += (
            f", last {float(last_udp_age):.2f}s ago"
            f" type={udp_status.get('lastType') or 'n/a'}"
            f" record={udp_status.get('lastRecordId') or 'n/a'}"
            f" sample={udp_status.get('lastSampleIndex') if udp_status.get('lastSampleIndex') is not None else 'n/a'}"
            f" from={udp_status.get('lastRemote') or 'n/a'}"
        )

    checks = [
        {
            "id": "captureIdle",
            "label": "PC capture state",
            "ok": not capture_busy,
            "detail": capture_detail,
        },
        {
            "id": "questUdp",
            "label": "Quest UDP receiver",
            "ok": total_datagrams > 0,
            "required": False,
            "detail": udp_detail,
        },
        {
            "id": "questLive",
            "label": "Quest live telemetry",
            "ok": bool(quest_live and head_ok and gaze_ok),
            "detail": (
                f"sample age {sample_age:.2f}s, head={head_ok}, gaze={gaze_ok}"
                if sample_age is not None
                else f"no live sample yet; {udp_detail}"
            ),
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
            "label": "RealSense cameras",
            "ok": camera_ok and third_camera_ok,
            "detail": (
                f"end={camera_serial or 'not selected'} {'seen' if camera_ok else 'missing'}; "
                f"third={third_camera_serial or 'off'} {'seen' if third_camera_ok else 'missing'}; "
                f"detected {len(camera_serials)} camera(s)"
            ),
        },
        {
            "id": "endCameraStream",
            "label": "End camera stream",
            "ok": end_stream_ok,
            "detail": end_stream_detail,
        },
        {
            "id": "thirdCameraStream",
            "label": "Third camera stream",
            "ok": third_stream_ok if third_camera_serial else True,
            "required": bool(third_camera_serial),
            "detail": third_stream_detail if third_camera_serial else "third camera not configured",
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
            "required": bool(robot_connected and hand_eye_enabled),
            "detail": (
                controller_preflight_detail(left_controller, left_recent, "left: ")
                + " | "
                + controller_preflight_detail(right_controller, right_recent, "right: ")
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
        "captureState": capture_state,
        "checks": checks,
        "summary": "ready for Quest recording" if ok else "check required items before Quest recording",
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


def resolve_child_path(root: Path, path_text: str) -> Path | None:
    if not path_text:
        return None
    candidate = (root / path_text).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def package_mesh_asset_path(filename: str) -> str | None:
    prefix = "package://"
    if not filename.startswith(prefix):
        return None
    asset_path = filename[len(prefix) :].replace("\\", "/").lstrip("/")
    return asset_path if asset_path.startswith("meshes/") else None


def resolve_robot_mesh_path(path_text: str) -> Path:
    if not path_text:
        raise ValueError("robot mesh path is empty")
    normalized = path_text.replace("\\", "/").lstrip("/")
    if not normalized.startswith("meshes/"):
        raise ValueError("robot mesh path must be under meshes/")
    path = (DEFAULT_RIZON_URDF.parent / normalized).resolve()
    root = (DEFAULT_RIZON_URDF.parent / "meshes").resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("robot mesh path is outside mesh root") from exc
    return path


def artifact_content_type(suffix: str) -> str:
    suffix = str(suffix or "").lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".mkv":
        return "video/x-matroska"
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix in (".json", ".jsonl"):
        return "application/json; charset=utf-8"
    if suffix == ".log":
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def parse_http_byte_range(header_value: str | None, size: int) -> tuple[int, int] | None:
    if not header_value:
        return None
    text = str(header_value).strip()
    if not text.startswith("bytes="):
        raise ValueError("unsupported range unit")
    spec = text[6:].strip()
    if "," in spec:
        raise ValueError("multiple byte ranges are not supported")
    start_text, sep, end_text = spec.partition("-")
    if not sep:
        raise ValueError("invalid byte range")
    if start_text == "":
        if end_text == "":
            raise ValueError("invalid byte range")
        length = int(end_text)
        if length <= 0:
            raise ValueError("invalid byte range")
        if size <= 0:
            raise ValueError("invalid byte range")
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_text)
        end = size - 1 if end_text == "" else int(end_text)
        if start < 0 or end < start:
            raise ValueError("invalid byte range")
    if start >= size:
        raise ValueError("range start exceeds file size")
    end = min(end, size - 1)
    return start, end


def send_http_path(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    size = int(path.stat().st_size)
    range_header = None
    try:
        range_header = handler.headers.get("Range")
    except Exception:
        range_header = None
    byte_range = parse_http_byte_range(range_header, size)
    if byte_range is None and size > MAX_ARTIFACT_BYTES and path.suffix.lower() not in (".mp4", ".mkv", ".bin"):
        handler.send_error(413, "artifact too large")
        return
    start = 0
    end = max(0, size - 1)
    status = 200
    if byte_range is not None:
        start, end = byte_range
        status = 206
    length = 0 if size <= 0 else end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(length))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Accept-Ranges", "bytes")
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if length <= 0:
        return
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(ARTIFACT_STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                break
            if not write_http_body_safely(handler, chunk):
                return
            remaining -= len(chunk)


def write_http_body_safely(handler: BaseHTTPRequestHandler, data: bytes) -> bool:
    try:
        handler.wfile.write(data)
        return True
    except (BrokenPipeError, ConnectionError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError):
        return False


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


def finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return default
    return result


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
  --sidebar-width: 330px;
  grid-template-columns: minmax(0, 1fr) 8px var(--sidebar-width);
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
.sidebar-resizer {
  width: 8px;
  border-left: 1px solid var(--line);
  border-right: 1px solid rgba(255,255,255,0.03);
  background: #101417;
  cursor: col-resize;
  touch-action: none;
}
.sidebar-resizer:hover,
.sidebar-resizer.dragging {
  background: #24303a;
}
aside {
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
.recording-banner.saving {
  border-color: #f2c94c;
  background: #3b3214;
  color: #fff7d6;
  box-shadow: 0 0 0 2px rgba(242,201,76,0.20), 0 0 22px rgba(242,201,76,0.20);
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
.recording-banner.saving .recording-dot {
  background: #f2c94c;
  box-shadow: 0 0 14px rgba(242,201,76,0.9);
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
button.danger {
  background: #4a171d;
  border-color: #8f2b38;
  color: #ffe5e8;
}
button.danger:hover,
button.danger.confirm {
  background: #842330;
  border-color: #ff5d70;
}
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
.metric-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
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
.robot-live-camera {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #0e1216;
  display: grid;
  gap: 8px;
}
.robot-live-camera.running {
  border-color: rgba(130, 173, 255, 0.5);
}
.robot-live-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}
.robot-live-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: var(--text);
  font-weight: 650;
}
.robot-live-status {
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
}
.robot-live-camera img {
  display: block;
  width: 100%;
  max-height: 260px;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #080a0c;
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
#status {
  white-space: pre-wrap;
  color: var(--muted);
  border-top: 1px solid var(--line);
  padding-top: 10px;
  margin-top: 10px;
}
@media (max-width: 860px) {
  #app { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; }
  .sidebar-resizer { display: none; }
  aside { max-height: 46vh; border-left: 0; border-top: 1px solid var(--line); }
}
</style>
</head>
<body>
<div id="app">
  <canvas id="view"></canvas>
  <div id="sidebarResizer" class="sidebar-resizer" title="Drag to resize controls"></div>
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
      <div class="metric-head">
        <div><strong>Calibration</strong><span id="mCalibration">idle</span></div>
        <button id="clearCalibration" class="danger" type="button">Clear</button>
      </div>
      <div class="calibration-progress"><div id="mCalibrationFill" class="calibration-progress-fill"></div></div>
    </div>
    <div id="calibrationDetails" class="calibration-details"></div>
    <div class="robot-panel">
      <h1>Flexiv / RealSense</h1>
      <label>End RealSense
        <select id="robotCamera"></select>
      </label>
      <label>Third RealSense
        <select id="robotThirdCamera"></select>
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
        <label>Robot state Hz
          <input id="robotStateHz" type="number" min="1" step="1">
        </label>
        <label>Calib interval s
          <input id="robotInterval" type="number" min="0.05" step="0.05">
        </label>
      </div>
      <div class="robot-grid">
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
      <div class="robot-grid">
        <label>Board warmup
          <input id="robotBoardWarmup" type="number" min="1" max="120" step="1">
        </label>
        <label>Depth record
          <select id="robotRecordDepth">
            <option value="true">on</option>
            <option value="false">off</option>
          </select>
        </label>
        <label>Depth align
          <select id="robotRecordDepthAlign">
            <option value="false">raw</option>
            <option value="true">color</option>
          </select>
        </label>
        <label>Depth every
          <input id="robotRecordDepthEveryNFrames" type="number" min="1" step="1">
        </label>
      </div>
      <div class="robot-grid">
        <label>Motion scale
          <input id="robotMotionScale" type="number" min="0" step="0.1">
        </label>
        <label>Max step m
          <input id="robotMaxStep" type="number" min="0" step="0.005">
        </label>
      </div>
      <div class="robot-grid">
        <label>Rot step deg
          <input id="robotMaxRotationStep" type="number" min="0" step="0.5">
        </label>
      </div>
      <div class="robot-grid">
        <label>Joint buffer rad
          <input id="robotJointLimitBuffer" type="number" min="0" step="0.01">
        </label>
        <label>Joint guard
          <select id="robotJointLimitGuard">
            <option value="true">on</option>
            <option value="false">off</option>
          </select>
        </label>
      </div>
      <div class="robot-grid">
        <label>Gripper
          <select id="robotGripperEnabled">
            <option value="false">off</option>
            <option value="true">trigger</option>
          </select>
        </label>
        <label>Device
          <input id="robotGripperDevice" spellcheck="false">
        </label>
      </div>
      <div class="robot-grid">
        <label>Open width m
          <input id="robotGripperOpen" type="number" min="0" step="0.005">
        </label>
        <label>Close width m
          <input id="robotGripperClose" type="number" min="0" step="0.005">
        </label>
      </div>
      <div class="robot-grid">
        <label>Grip speed
          <input id="robotGripperSpeed" type="number" min="0" step="0.005">
        </label>
        <label>Grip force
          <input id="robotGripperForce" type="number" min="0" step="1">
        </label>
      </div>
      <div class="robot-actions">
        <button id="robotRefresh">Refresh Cameras</button>
        <button id="robotStreamStart">Start Stream</button>
        <button id="robotStreamStop">Stop Stream</button>
        <button id="robotBoardCheck">Check Board</button>
        <button id="robotDiagnostics">Diagnostics</button>
        <button id="robotGripOpen">Grip Open</button>
        <button id="robotGripClose">Grip Close</button>
        <button id="robotConnect">Connect Robot</button>
        <button id="robotDisconnect">Disconnect</button>
      </div>
      <div id="robotStatus" class="robot-status">disabled or loading</div>
      <div class="robot-live-grid">
        <div id="robotLiveCamera" class="robot-live-camera">
          <div class="robot-live-head">
            <span>End camera</span>
            <span id="robotLiveStatus" class="robot-live-status">stream stopped</span>
          </div>
          <img id="robotLiveImage" alt="end camera live frame">
        </div>
        <div id="robotThirdLiveCamera" class="robot-live-camera">
          <div class="robot-live-head">
            <span>Third camera</span>
            <span id="robotThirdLiveStatus" class="robot-live-status">stream stopped</span>
          </div>
          <img id="robotThirdLiveImage" alt="third camera live frame">
        </div>
      </div>
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
const app = document.getElementById('app');
const sidebarResizer = document.getElementById('sidebarResizer');
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
const clearCalibration = document.getElementById('clearCalibration');
const preflightRefresh = document.getElementById('preflightRefresh');
const preflightStatus = document.getElementById('preflightStatus');
const preflightList = document.getElementById('preflightList');
const calibrationDetails = document.getElementById('calibrationDetails');
const statusEl = document.getElementById('status');
const recordingBanner = document.getElementById('recordingBanner');
const recordingLabel = document.getElementById('recordingLabel');
const recordingDetail = document.getElementById('recordingDetail');
const robotCamera = document.getElementById('robotCamera');
const robotThirdCamera = document.getElementById('robotThirdCamera');
const robotSn = document.getElementById('robotSn');
const robotPoseField = document.getElementById('robotPoseField');
const robotNetworkInterfaces = document.getElementById('robotNetworkInterfaces');
const robotStateHz = document.getElementById('robotStateHz');
const robotInterval = document.getElementById('robotInterval');
const robotHandEye = document.getElementById('robotHandEye');
const robotExposureMode = document.getElementById('robotExposureMode');
const robotExposure = document.getElementById('robotExposure');
const robotGain = document.getElementById('robotGain');
const robotBoardWarmup = document.getElementById('robotBoardWarmup');
const robotRecordDepth = document.getElementById('robotRecordDepth');
const robotRecordDepthAlign = document.getElementById('robotRecordDepthAlign');
const robotRecordDepthEveryNFrames = document.getElementById('robotRecordDepthEveryNFrames');
const robotMotionScale = document.getElementById('robotMotionScale');
const robotMaxStep = document.getElementById('robotMaxStep');
const robotMaxRotationStep = document.getElementById('robotMaxRotationStep');
const robotJointLimitBuffer = document.getElementById('robotJointLimitBuffer');
const robotJointLimitGuard = document.getElementById('robotJointLimitGuard');
const robotGripperEnabled = document.getElementById('robotGripperEnabled');
const robotGripperDevice = document.getElementById('robotGripperDevice');
const robotGripperOpen = document.getElementById('robotGripperOpen');
const robotGripperClose = document.getElementById('robotGripperClose');
const robotGripperSpeed = document.getElementById('robotGripperSpeed');
const robotGripperForce = document.getElementById('robotGripperForce');
const robotRefresh = document.getElementById('robotRefresh');
const robotStreamStart = document.getElementById('robotStreamStart');
const robotStreamStop = document.getElementById('robotStreamStop');
const robotBoardCheck = document.getElementById('robotBoardCheck');
const robotDiagnostics = document.getElementById('robotDiagnostics');
const robotGripOpen = document.getElementById('robotGripOpen');
const robotGripClose = document.getElementById('robotGripClose');
const robotConnect = document.getElementById('robotConnect');
const robotDisconnect = document.getElementById('robotDisconnect');
const robotStatus = document.getElementById('robotStatus');
const robotLiveCamera = document.getElementById('robotLiveCamera');
const robotLiveImage = document.getElementById('robotLiveImage');
const robotLiveStatus = document.getElementById('robotLiveStatus');
const robotThirdLiveCamera = document.getElementById('robotThirdLiveCamera');
const robotThirdLiveImage = document.getElementById('robotThirdLiveImage');
const robotThirdLiveStatus = document.getElementById('robotThirdLiveStatus');
const questAdbRefresh = document.getElementById('questAdbRefresh');
const questCalibStart = document.getElementById('questCalibStart');
const questCalibStop = document.getElementById('questCalibStop');
const questAdbStatus = document.getElementById('questAdbStatus');
const questAdbManual = document.getElementById('questAdbManual');

const state = {
  frames: [],
  maxFrames: 1200,
  origin: [0, 0, 0],
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
  robotGripper: null,
  robotDiagnostics: null,
  robotBoardCheck: null,
  robotWorldBase: null,
  robotModel: null,
  robotMeshes: {},
  robotMeshPending: {},
  robotCalibration: null,
  robotLiveTimer: null,
  robotLiveLastSequence: {},
  robotLiveLastStatusRefreshMs: 0,
  robotStatusTimer: null,
  robotStatusIntervalMs: 0,
  robotStatusInFlight: false,
  robotCalibrationKey: null,
  viewFitTimer: null,
  latestCalibrationRecordId: null,
  clearCalibrationConfirmUntilMs: 0,
  questAdb: null,
  preflight: null,
  captureState: {phase: 'live', detail: 'not recording'},
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
  state.pitch = -0.62;
  const bounds = computeBounds();
  state.target = [0, 0, 0];
  state.distance = Math.max(0.45, bounds.radius * 3.2);
}

function requestViewFit() {
  if (state.viewFitTimer) window.clearTimeout(state.viewFitTimer);
  state.viewFitTimer = window.setTimeout(() => {
    state.viewFitTimer = null;
    resetView();
  }, 80);
}

function hasDrawableRobot() {
  return Boolean(state.robotModel?.ok && liveRobotJointpose() && robotLiveBaseMatrix());
}

function computeBounds() {
  const pts = [];
  pts.push(...liveBoardPoints());
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
  const liveJointpose = liveRobotJointpose();
  if (liveJointpose && state.robotModel?.ok) {
    for (const m of robotFrames(state.robotModel, liveJointpose, robotLiveBaseMatrix())) {
      const p = relPoint(matrixTranslation(m));
      if (p) pts.push(p);
    }
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
  for (const p of pts) radius = Math.max(radius, length(p));
  return {center, radius};
}

function liveBoardSpec() {
  const result = state.calibration?.result || {};
  const matrix = result?.T_world_board?.matrix_4x4 || defaultBoardMatrix4();
  const pattern = Array.isArray(result.pattern) ? result.pattern : [11, 8];
  const square = Number.isFinite(Number(result.squareSizeM)) ? Number(result.squareSizeM) : 0.025;
  return {matrix, pattern, square};
}

function liveBoardPoints() {
  const {matrix, pattern, square} = liveBoardSpec();
  if (!matrix || !state.origin) return [];
  const width = (Number(pattern[0] || 11) - 1) * square;
  const height = (Number(pattern[1] || 8) - 1) * square;
  return [
    boardPoint(matrix, 0, 0, 0),
    boardPoint(matrix, width, 0, 0),
    boardPoint(matrix, width, height, 0),
    boardPoint(matrix, 0, height, 0),
  ].map(relPoint).filter(Boolean);
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
    } else if (sample.type === 'capture_state') {
      updateCaptureState(sample);
    } else if (sample.type === 'capture_rejected') {
      state.captureRejected = sample;
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
    } else if (sample.type === 'robot_gripper') {
      updateRobotGripper(sample);
    } else if (sample.type === 'robot_calibration_result' || sample.type === 'robot_calibration_failure') {
      updateRobotCalibration(sample);
    } else if (sample.type === 'calibration_cleared') {
      applyCalibrationCleared(sample);
    } else if (sample.type === 'robot_calibration_cleared') {
      applyRobotCalibrationCleared(sample);
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
    const nextId = payload.event.recordId || payload.event.rawRecordName || null;
    if (nextId && nextId === state.latestCalibrationRecordId) return;
    if (payload.kind === 'result') updateCalibrationResult(payload.event);
    if (payload.kind === 'failure') updateCalibrationFailure(payload.event);
  } catch (_) {
    // The live stream still works if the snapshot endpoint is unavailable.
  }
}

async function loadRobotStatus() {
  if (state.robotStatusInFlight) return;
  state.robotStatusInFlight = true;
  try {
    const response = await fetch('/robot/status', {cache: 'no-store'});
    const payload = await response.json();
    applyRobotStatus(payload);
  } catch (error) {
    robotStatus.textContent = String(error);
  } finally {
    state.robotStatusInFlight = false;
    syncRobotStatusLoop();
  }
}

function syncRobotStatusLoop() {
  const connected = Boolean(state.robot?.robot?.connected);
  const intervalMs = connected ? 250 : 2000;
  if (state.robotStatusTimer && state.robotStatusIntervalMs === intervalMs) return;
  if (state.robotStatusTimer) window.clearInterval(state.robotStatusTimer);
  state.robotStatusIntervalMs = intervalMs;
  state.robotStatusTimer = window.setInterval(loadRobotStatus, intervalMs);
}

async function startRobotStream() {
  await configureRobot();
  robotStreamStart.disabled = true;
  robotLiveStatus.textContent = 'starting stream...';
  try {
    const response = await fetch('/robot/realsense-stream/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(robotPayloadFromControls())
    });
    const payload = await response.json();
    applyRobotStatus(payload.status || payload);
    if (!payload.ok) robotStatus.textContent += `\nerror: ${payload.error || 'stream start failed'}`;
    else refreshRobotLiveFrame();
  } catch (error) {
    robotLiveStatus.textContent = String(error);
  } finally {
    syncRobotLiveLoop();
  }
}

async function stopRobotStream() {
  robotStreamStop.disabled = true;
  robotLiveStatus.textContent = 'stopping stream...';
  try {
    const response = await fetch('/robot/realsense-stream/stop', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: '{}'
    });
    const payload = await response.json();
    applyRobotStatus(payload.status || payload);
  } catch (error) {
    robotLiveStatus.textContent = String(error);
  } finally {
    syncRobotLiveLoop();
  }
}

function syncRobotLiveLoop() {
  const running = Boolean(state.robot?.realsenseStream?.running);
  robotLiveCamera.classList.toggle('running', running);
  robotThirdLiveCamera.classList.toggle('running', running && robotThirdCamera.value);
  robotStreamStart.disabled = running;
  robotStreamStop.disabled = !running;
  if (running && !state.robotLiveTimer) {
    state.robotLiveTimer = window.setInterval(refreshRobotLiveFrame, 700);
    refreshRobotLiveFrame();
  } else if (!running && state.robotLiveTimer) {
    window.clearInterval(state.robotLiveTimer);
    state.robotLiveTimer = null;
  }
  if (!running) {
    state.robotLiveLastSequence = {};
    robotLiveStatus.textContent = 'stream stopped';
    robotThirdLiveStatus.textContent = robotThirdCamera.value ? 'stream stopped' : 'third off';
  }
}

function refreshRobotLiveFrame() {
  if (!state.robot?.realsenseStream?.running) return;
  refreshRobotLiveFrameRole('end', robotLiveImage, robotLiveStatus);
  if (robotThirdCamera.value) refreshRobotLiveFrameRole('third', robotThirdLiveImage, robotThirdLiveStatus);
  else {
    robotThirdLiveImage.removeAttribute('src');
    robotThirdLiveStatus.textContent = 'third off';
  }
}

function refreshRobotLiveFrameRole(role, image, statusEl) {
  const url = `/robot/camera-frame?role=${encodeURIComponent(role)}&ts=${Date.now()}`;
  image.onload = () => {
    const stream = state.robot?.realsenseStream || {};
    const latest = stream.latest?.[role] || {};
    const sequence = latest.sequence ?? stream.frameCount ?? '';
    state.robotLiveLastSequence[role] = sequence;
    statusEl.textContent = `live ${sequence ? `#${sequence}` : ''}`.trim();
    const now = performance.now();
    if (!state.robotLiveLastStatusRefreshMs || now - state.robotLiveLastStatusRefreshMs > 2000) {
      state.robotLiveLastStatusRefreshMs = now;
      loadRobotStatus();
    }
  };
  image.onerror = () => {
    statusEl.textContent = 'frame unavailable';
  };
  image.src = url;
}

async function loadRobotModel() {
  const hadDrawableRobot = hasDrawableRobot();
  try {
    const response = await fetch('/robot/model', {cache: 'no-store'});
    const payload = await response.json();
    state.robotModel = payload;
    state.robotMeshes = {};
    state.robotMeshPending = {};
    preloadRobotMeshes(payload);
  } catch (_) {
    state.robotModel = null;
  }
  if (!hadDrawableRobot && hasDrawableRobot()) requestViewFit();
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
  if (payload?.captureState) updateCaptureState(payload.captureState);
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
  if (command === 'calib_start' && captureBusy()) {
    renderQuestAdb({...(state.questAdb || {}), ok: false, ready: Boolean(state.questAdb?.ready), error: 'PC is recording or saving'});
    syncQuestCommandButtons();
    return;
  }
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
    syncQuestCommandButtons();
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
  syncQuestCommandButtons();
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
    const currentThird = robotThirdCamera.value;
    const configured = state.robot?.config || {};
    const configuredEnd = configured.cameraSerial || '';
    const configuredThird = configured.thirdCameraSerial || '';
    robotCamera.innerHTML = '';
    robotThirdCamera.innerHTML = '<option value="">off</option>';
    for (const camera of payload.cameras || []) {
      const option = document.createElement('option');
      option.value = camera.serial || '';
      option.textContent = `${camera.serial || 'unknown'} ${camera.name || ''}`;
      robotCamera.appendChild(option);
      const thirdOption = option.cloneNode(true);
      robotThirdCamera.appendChild(thirdOption);
    }
    setCameraSelectValue(robotCamera, configuredEnd || current);
    setCameraSelectValue(robotThirdCamera, configuredThird || currentThird);
    if (!payload.ok) robotStatus.textContent = payload.error || payload.reason || 'camera list unavailable';
  } catch (error) {
    robotStatus.textContent = String(error);
  }
}

function setCameraSelectValue(select, serial) {
  const value = String(serial || '').trim();
  if (!value) {
    select.value = '';
    return;
  }
  const existing = Array.from(select.options).some(option => option.value === value);
  if (!existing) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = `${value} configured`;
    select.appendChild(option);
  }
  select.value = value;
}

function robotPayloadFromControls() {
  const endSerial = robotCamera.value;
  const thirdSerial = robotThirdCamera.value === endSerial ? '' : robotThirdCamera.value;
  return {
    robotSn: robotSn.value.trim(),
    poseField: robotPoseField.value,
    networkInterfaces: robotNetworkInterfaces.value.split(/[,\s;]+/).map(v => v.trim()).filter(Boolean),
    cameraSerial: endSerial,
    thirdCameraSerial: thirdSerial,
    robotStateHz: Number(robotStateHz.value || 90),
    captureIntervalSeconds: Number(robotInterval.value || 0.35),
    recordDepth: robotRecordDepth.value !== 'false',
    recordDepthAlignToColor: robotRecordDepthAlign.value === 'true',
    recordDepthEveryNFrames: Math.max(1, Math.floor(Number(robotRecordDepthEveryNFrames.value || 3))),
    realsenseAutoExposure: robotExposureMode.value !== 'manual',
    realsenseExposure: robotExposure.value ? Number(robotExposure.value) : null,
    realsenseGain: robotGain.value ? Number(robotGain.value) : null,
    boardCheckWarmupFrames: Number(robotBoardWarmup.value || 60),
    runHandEye: robotHandEye.value === 'true',
    controllerTranslationScale: Number(robotMotionScale.value || 1.0),
    controllerMaxStepM: Number(robotMaxStep.value || 0.04),
    controllerMaxRotationStepDeg: Number(robotMaxRotationStep.value || 4.0),
    controllerJointLimitBufferRad: Number(robotJointLimitBuffer.value || 0.04),
    controllerJointLimitGuardEnabled: robotJointLimitGuard.value !== 'false',
    gripperEnabled: robotGripperEnabled.value === 'true',
    gripperDevice: robotGripperDevice.value.trim(),
    gripperOpenWidthM: Number(robotGripperOpen.value || 0.08),
    gripperCloseWidthM: Number(robotGripperClose.value || 0.0),
    gripperSpeedMps: Number(robotGripperSpeed.value || 0.04),
    gripperForceN: Number(robotGripperForce.value || 20.0)
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
    else if (payload.realsenseWarmup?.ok || payload.status?.realsenseStream?.running) {
      refreshRobotLiveFrame();
      syncRobotLiveLoop();
    }
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

async function moveRobotGripper(action) {
  const button = action === 'open' ? robotGripOpen : robotGripClose;
  const payload = robotPayloadFromControls();
  payload.action = action;
  payload.widthM = action === 'open'
    ? Number(robotGripperOpen.value || 0.08)
    : Number(robotGripperClose.value || 0.0);
  payload.speedMps = Number(robotGripperSpeed.value || 0.04);
  payload.forceN = Number(robotGripperForce.value || 20.0);
  robotGripOpen.disabled = true;
  robotGripClose.disabled = true;
  button.textContent = action === 'open' ? 'Opening...' : 'Closing...';
  robotStatus.textContent = `moving gripper ${action}...`;
  try {
    const response = await fetch('/robot/gripper/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    updateRobotGripper(result);
    await loadRobotStatus();
    if (!result.ok) robotStatus.textContent += `\ngripper error: ${result.error || 'move failed'}`;
  } catch (error) {
    robotStatus.textContent = String(error);
  } finally {
    robotGripOpen.disabled = false;
    robotGripClose.disabled = false;
    robotGripOpen.textContent = 'Grip Open';
    robotGripClose.textContent = 'Grip Close';
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
    syncRobotLiveLoop();
    loadRobotStatus();
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
  if (url) lines.push(`saved check: ${url}`);
  robotLiveStatus.textContent = detected
    ? `checkerboard ${payload?.detectedCorners ?? 0} corners`
    : 'checkerboard not detected';
}

async function disconnectRobot() {
  const response = await fetch('/robot/disconnect', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  applyRobotStatus(await response.json());
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
  const hadDrawableRobot = hasDrawableRobot();
  state.robotSample = event;
  if (!hadDrawableRobot && hasDrawableRobot()) requestViewFit();
  renderRobotStatus(state.robot);
}

function updateRobotMotion(event) {
  state.robotMotion = event;
  renderRobotStatus(state.robot);
}

function updateRobotGripper(event) {
  state.robotGripper = event;
  renderRobotStatus(state.robot);
}

function updateRobotCalibration(event) {
  const key = robotCalibrationKey(event);
  const changed = key && key !== state.robotCalibrationKey;
  state.robotCalibration = event;
  if (key) state.robotCalibrationKey = key;
  if (event.T_world_base?.matrix_4x4) state.robotWorldBase = event.T_world_base.matrix_4x4;
  if (changed && hasDrawableRobot()) requestViewFit();
  renderRobotStatus(state.robot);
}

function applyRobotCalibrationCleared(event) {
  state.robotCalibration = null;
  state.robotCalibrationKey = null;
  state.robotWorldBase = null;
  renderRobotStatus(state.robot);
}

function robotCalibrationKey(event) {
  if (!event) return '';
  return String(event.sourcePath || event.runDir || event.recordId || JSON.stringify(event.T_world_base?.matrix_4x4 || null));
}

function applyRobotStatus(payload) {
  const hadDrawableRobot = hasDrawableRobot();
  state.robot = payload;
  if (payload?.lastCalibration) updateRobotCalibration(payload.lastCalibration);
  const statusSample = robotSampleFromStatus(payload);
  if (statusSample) state.robotSample = {...(state.robotSample || {}), ...statusSample};
  const config = payload?.config || {};
  if (config.robotSn && !robotSn.value) robotSn.value = config.robotSn;
  if (config.poseField) robotPoseField.value = config.poseField;
  if (Array.isArray(config.networkInterfaces) && !robotNetworkInterfaces.value) {
    robotNetworkInterfaces.value = config.networkInterfaces.join(', ');
  }
  setCameraSelectValue(robotCamera, config.cameraSerial);
  setCameraSelectValue(robotThirdCamera, config.thirdCameraSerial);
  if (Number.isFinite(config.robotStateHz)) robotStateHz.value = config.robotStateHz;
  if (Number.isFinite(config.captureIntervalSeconds)) robotInterval.value = config.captureIntervalSeconds;
  if (typeof config.realsenseAutoExposure === 'boolean') robotExposureMode.value = config.realsenseAutoExposure ? 'auto' : 'manual';
  if (Number.isFinite(config.realsenseExposure)) robotExposure.value = config.realsenseExposure;
  if (Number.isFinite(config.realsenseGain)) robotGain.value = config.realsenseGain;
  if (Number.isFinite(config.boardCheckWarmupFrames)) robotBoardWarmup.value = config.boardCheckWarmupFrames;
  if (typeof config.recordDepth === 'boolean') robotRecordDepth.value = config.recordDepth ? 'true' : 'false';
  if (typeof config.recordDepthAlignToColor === 'boolean') robotRecordDepthAlign.value = config.recordDepthAlignToColor ? 'true' : 'false';
  if (Number.isFinite(config.recordDepthEveryNFrames)) robotRecordDepthEveryNFrames.value = config.recordDepthEveryNFrames;
  if (typeof config.runHandEye === 'boolean') robotHandEye.value = config.runHandEye ? 'true' : 'false';
  if (Number.isFinite(config.controllerTranslationScale)) robotMotionScale.value = config.controllerTranslationScale;
  if (Number.isFinite(config.controllerMaxStepM)) robotMaxStep.value = config.controllerMaxStepM;
  if (Number.isFinite(config.controllerMaxRotationStepDeg)) robotMaxRotationStep.value = config.controllerMaxRotationStepDeg;
  if (Number.isFinite(config.controllerJointLimitBufferRad)) robotJointLimitBuffer.value = config.controllerJointLimitBufferRad;
  if (typeof config.controllerJointLimitGuardEnabled === 'boolean') robotJointLimitGuard.value = config.controllerJointLimitGuardEnabled ? 'true' : 'false';
  if (typeof config.gripperEnabled === 'boolean') robotGripperEnabled.value = config.gripperEnabled ? 'true' : 'false';
  if (config.gripperDevice && document.activeElement !== robotGripperDevice) {
    robotGripperDevice.value = config.gripperDevice;
  }
  if (Number.isFinite(config.gripperOpenWidthM)) robotGripperOpen.value = config.gripperOpenWidthM;
  if (Number.isFinite(config.gripperCloseWidthM)) robotGripperClose.value = config.gripperCloseWidthM;
  if (Number.isFinite(config.gripperSpeedMps)) robotGripperSpeed.value = config.gripperSpeedMps;
  if (Number.isFinite(config.gripperForceN)) robotGripperForce.value = config.gripperForceN;
  syncRobotLiveLoop();
  renderRobotStatus(payload);
  if (!hadDrawableRobot && hasDrawableRobot()) requestViewFit();
  syncRobotStatusLoop();
}

function robotSampleFromStatus(payload) {
  const robot = payload?.robot || {};
  const statePayload = robot.state || {};
  const jointpose = numericArray(statePayload.jointPose || statePayload.jointpose || statePayload.jointpos);
  if (!jointpose || jointpose.length < 7) return null;
  const tcpPose = statePayload.tcp_pose || statePayload.tcpPose || {};
  return {
    type: 'robot_sample',
    sampleIndex: 'live',
    questSampleIndex: null,
    robotSn: robot.robotSn,
    poseField: robot.poseField,
    readUnixSeconds: statePayload.readUnixSeconds,
    jointpose,
    jointpos: jointpose,
    T_base_ee: statePayload.endEffectorPose,
    T_base_tool_tcp: tcpPose.T_base_pose || statePayload.T_base_tool_tcp || statePayload.endEffectorPose,
    jointLimitGuard: statePayload.jointLimitGuard,
    mode: statePayload.mode,
    operationalStatus: statePayload.operationalStatus,
    enablingButtonPressed: statePayload.enablingButtonPressed,
    busy: statePayload.busy,
    stopped: statePayload.stopped,
    reduced: statePayload.reduced
  };
}

function numericArray(value) {
  if (!Array.isArray(value)) return null;
  const out = value.map(v => Number(v));
  return out.every(Number.isFinite) ? out : null;
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
  const active = payload.activeSession || {};
  const controlMode = active.controlMode || (robot.freeDragEnabled || robot.freedriveEnabled ? 'freedrive' : 'idle');
  const freeDragEnabled = Boolean(robot.freeDragEnabled || robot.freedriveEnabled);
  const freeDragMethod = robot.freeDragMethod || robot.freedriveMethod || '';
  const freeDragPlan = robot.freeDragPlan || robot.freedrivePlan || '';
  const freeDragDetail = freeDragPlan || freeDragMethod;
  const cartesianLoopAlive = Boolean(robot.cartesianControlLoopAlive || robot.freeDragLoopAlive || robot.freedriveLoopAlive);
  const freeDragLastError = robot.freeDragLastError || robot.freedriveLastError || '';
  const cartesianSendSignature = robot.cartesianSendSignature || '';
  const gripper = robot.gripper || {};
  const depthRecordingEnabled = payload.config?.recordDepth !== false;
  const sessionText = payload.activeSession
    ? `${active.samples || 0} samples, ${active.images || 0} images${depthRecordingEnabled ? `, ${active.depthFrames ?? 0} depth` : ', depth off'}`
    : 'idle';
  const teleopText = controlMode === 'controller_teleop'
    ? `hold right middle-finger trigger${robot.motionArmed ? ' (motion mode active)' : ' (motion not armed)'}`
    : 'off';
  const lines = [
    `robot: ${robot.connected ? 'connected' : 'not connected'} ${robot.robotSn || ''}`.trim(),
    `control: ${controlMode}`,
    `teleop: ${teleopText}`,
    `free-drag: ${freeDragEnabled ? `enabled${freeDragDetail ? ` via ${freeDragDetail}` : ''}` : 'disabled'}`,
    `cartesian loop: ${cartesianLoopAlive ? 'running' : 'off'}${cartesianSendSignature ? ` (${cartesianSendSignature})` : ''}`,
    `controller motion: ${payload.config?.controllerMotionEnabled ? 'enabled' : 'disabled'}`,
    `joint guard: ${robotJointGuardText(robot.state?.jointLimitGuard, payload.config)}`,
    `gripper: ${robotGripperStatusText(gripper, payload.config)}`,
    `devices: ${robotDevicesText(robot.devices)}`,
    `pose: ${robot.poseField || 'n/a'}`,
    `rdk iface: ${(payload.config?.networkInterfaces || []).join(', ') || 'default'}`,
    `camera: ${payload.config?.cameraSerial || 'n/a'}`,
    `third camera: ${payload.config?.thirdCameraSerial || 'off'}`,
    `depth record: ${depthRecordingEnabled ? `${payload.config?.recordDepthAlignToColor ? 'aligned' : 'raw'} every ${payload.config?.recordDepthEveryNFrames || 1} frame(s)` : 'off'}`,
    `stream: ${realsenseStreamText(payload.realsenseStream)}`,
    `exposure: ${payload.config?.realsenseAutoExposure === false ? 'manual' : 'auto'}${Number.isFinite(payload.config?.realsenseExposure) ? ` ${payload.config.realsenseExposure}` : ''}${Number.isFinite(payload.config?.realsenseGain) ? ` gain ${payload.config.realsenseGain}` : ''}`,
    `robot frame: ${state.robotWorldBase ? 'aligned to Quest/world' : 'unaligned at viewer origin, Z up'}`,
    `session: ${sessionText}`
  ];
  lines.push(`model: ${robotModelStatusText()}`);
  if (state.robotSample) {
    lines.push(`last sample: ${robotSampleLabel(state.robotSample)}`);
    if (Number.isFinite(state.robotSample.readUnixSeconds)) {
      const age = Math.max(0, Date.now() / 1000 - Number(state.robotSample.readUnixSeconds));
      lines.push(`joint age: ${age.toFixed(2)}s`);
    }
    if (Array.isArray(state.robotSample.jointpose)) {
      const preview = state.robotSample.jointpose.slice(0, 4).map(v => Number(v).toFixed(3)).join(', ');
      lines.push(`joints: ${preview}${state.robotSample.jointpose.length > 4 ? ', ...' : ''}`);
    }
    if (state.robotSample.mode || state.robotSample.operationalStatus) {
      lines.push(`rdk state: ${state.robotSample.mode || 'n/a'} / ${state.robotSample.operationalStatus || 'n/a'} busy=${Boolean(state.robotSample.busy)} enableBtn=${Boolean(state.robotSample.enablingButtonPressed)}`);
    }
    const fkError = robotFkErrorMm(state.robotSample, null);
    if (Number.isFinite(fkError)) lines.push(`URDF FK vs flange: ${fkError.toFixed(1)}mm`);
    lines.push(`hand-eye motion: ${poseDiversityText(state.robotSample.poseDiversity)}`);
    const rightInput = robotControllerInputText(state.robotSample.rightController);
    if (rightInput) lines.push(`right input: ${rightInput}`);
    if (state.robotSample.jointLimitGuard) lines.push(`sample joint guard: ${robotJointGuardText(state.robotSample.jointLimitGuard, payload.config)}`);
  } else if (active.poseDiversity) {
    lines.push(`hand-eye motion: ${poseDiversityText(active.poseDiversity)}`);
  }
  if (freeDragLastError) lines.push(`free-drag error: ${freeDragLastError}`);
  if (payload.activeSession) {
    lines.push(`motion counts: cmd ${active.motionCommands ?? 0}, skip ${active.motionSkips ?? 0}, err ${active.motionErrors ?? 0}`);
    lines.push(`gripper counts: cmd ${active.gripperCommands ?? 0}, skip ${active.gripperSkips ?? 0}, err ${active.gripperErrors ?? 0}`);
    if (active.lastMotion) {
      lines.push(`last motion: ${robotMotionSummaryText(active.lastMotion)}`);
      if (active.lastMotion.jointLimitGuard) lines.push(`motion joint guard: ${robotJointGuardText(active.lastMotion.jointLimitGuard, payload.config)}`);
    }
    if (active.lastGripper) {
      lines.push(`last gripper: ${robotGripperSummaryText(active.lastGripper)}`);
    }
  }
  if (state.robotMotion) {
    const offset = Array.isArray(state.robotMotion.offsetM) ? state.robotMotion.offsetM.map(v => Number(v).toFixed(3)).join(', ') : (state.robotMotion.reason || state.robotMotion.error || 'n/a');
    const rot = Number.isFinite(state.robotMotion.rotationDeg) ? ` rot ${Number(state.robotMotion.rotationDeg).toFixed(1)}deg` : '';
    const hold = robotTeleopHoldText(state.robotMotion);
    lines.push(`motion: ${state.robotMotion.ok ? 'sent' : 'skip'} ${offset}${rot}${hold}`);
    if (state.robotMotion.anchored !== undefined) {
      const step = Array.isArray(state.robotMotion.stepOffsetM) ? state.robotMotion.stepOffsetM.map(v => Number(v).toFixed(3)).join(', ') : 'n/a';
      const rotStep = Number.isFinite(state.robotMotion.stepRotationDeg) ? `, rot step ${Number(state.robotMotion.stepRotationDeg).toFixed(1)}deg` : '';
      lines.push(`motion anchor: ${state.robotMotion.anchored ? 'set' : 'waiting'}${state.robotMotion.createdAnchor ? ' (new)' : ''}, step ${step}${rotStep}`);
    }
    if (state.robotMotion.jointLimitGuard) {
      lines.push(`motion joint guard: ${robotJointGuardText(state.robotMotion.jointLimitGuard, payload.config)}`);
    }
  }
  if (state.robotGripper) {
    lines.push(`gripper: ${robotGripperSummaryText(state.robotGripper)}`);
  }
  if (state.robotCalibration) {
    lines.push(state.robotCalibration.type === 'robot_calibration_result' ? 'hand-eye: done' : `hand-eye: failed ${state.robotCalibration.error || ''}`);
    const resultDiversity = state.robotCalibration.result?.diversity;
    if (resultDiversity) lines.push(`hand-eye result motion: ${poseDiversityText(resultDiversity)}`);
  }
  if (robot.lastError || payload.lastError) lines.push(`error: ${robot.lastError || payload.lastError}`);
  robotStatus.textContent = lines.join('\n');
}

function robotSampleLabel(sample) {
  if (sample.sampleIndex === 'live') return 'live robot status';
  const quest = sample.questSampleIndex ?? 'n/a';
  return `${sample.sampleIndex ?? 'n/a'} quest ${quest}`;
}

function realsenseStreamText(stream) {
  if (!stream) return 'n/a';
  if (!stream.running) return stream.lastError ? `stopped (${stream.lastError})` : 'stopped';
  const roleList = Array.isArray(stream.roles) ? stream.roles : [];
  const roles = roleList.length ? roleList.join(',') : 'n/a';
  const metadata = stream.metadata || {};
  const depthRoles = roleList.filter(role => metadata?.[role]?.depthEnabled);
  const depthText = depthRoles.length ? ` depth=${depthRoles.join(',')}` : ' depth=off';
  return `running ${roles} frames=${stream.frameCount ?? 0}${depthText}`;
}

function robotMotionSummaryText(event) {
  if (!event) return 'n/a';
  const mapping = event.motionMapping || event.motion_mapping;
  const mappingText = mapping ? ` ${mapping}` : '';
  const hold = robotTeleopHoldText(event);
  if (event.ok) {
    const offset = Array.isArray(event.offsetM)
      ? event.offsetM.map(v => Number(v).toFixed(3)).join(', ')
      : 'sent';
    const step = Array.isArray(event.stepOffsetM)
      ? ` step ${event.stepOffsetM.map(v => Number(v).toFixed(3)).join(', ')}`
      : '';
    const rotation = Number.isFinite(event.rotationDeg) ? ` rot ${Number(event.rotationDeg).toFixed(1)}deg` : '';
    const rotationStep = Number.isFinite(event.stepRotationDeg) ? ` rotStep ${Number(event.stepRotationDeg).toFixed(1)}deg` : '';
    return `sent offset ${offset}${step}${rotation}${rotationStep}${mappingText}${hold}`;
  }
  return `${event.reason || event.error || 'skipped'}${mappingText}${hold}`;
}

function robotJointGuardText(guard, config) {
  const enabled = guard?.enabled ?? config?.controllerJointLimitGuardEnabled;
  if (enabled === false) return 'off';
  if (!guard) {
    const buffer = Number(config?.controllerJointLimitBufferRad);
    return Number.isFinite(buffer) ? `on buffer ${buffer.toFixed(2)}rad` : 'on';
  }
  const buffer = Number(guard.bufferRad);
  const margin = Number(guard.minMarginRad);
  const suffix = [
    Number.isFinite(buffer) ? `buffer ${buffer.toFixed(2)}rad` : '',
    Number.isFinite(margin) ? `margin ${margin.toFixed(2)}rad` : ''
  ].filter(Boolean).join(', ');
  if (guard.ok) return `ok${suffix ? ` (${suffix})` : ''}`;
  const violations = Array.isArray(guard.violations)
    ? guard.violations.map(row => `${row.name || `j${Number(row.index) + 1}`}:${row.violation || 'limit'}`).join(', ')
    : '';
  return `${guard.reason || 'limit'}${violations ? ` ${violations}` : ''}${suffix ? ` (${suffix})` : ''}`;
}

function robotTeleopHoldText(event) {
  if (!event) return '';
  const value = event.teleopHoldValue ?? event.teleop_hold_value;
  const held = event.teleopHeld ?? event.teleop_held;
  const parts = [];
  if (Number.isFinite(Number(value))) parts.push(`hold ${Number(value).toFixed(2)}`);
  if (typeof held === 'boolean') parts.push(`held=${held}`);
  return parts.length ? ` (${parts.join(', ')})` : '';
}

function robotControllerInputText(controller) {
  if (!controller) return '';
  const input = controller.input || {};
  if (!input.hasAny) return '';
  const parts = [];
  if (Number.isFinite(input.handTrigger)) parts.push(`hand=${Number(input.handTrigger).toFixed(2)}`);
  if (input.handTriggerPressed !== undefined && input.handTriggerPressed !== null) parts.push(`handPressed=${Boolean(input.handTriggerPressed)}`);
  if (Number.isFinite(input.indexTrigger)) parts.push(`index=${Number(input.indexTrigger).toFixed(2)}`);
  if (input.indexTriggerPressed !== undefined && input.indexTriggerPressed !== null) parts.push(`indexPressed=${Boolean(input.indexTriggerPressed)}`);
  if (input.aButton !== undefined && input.aButton !== null) parts.push(`A=${Boolean(input.aButton)}`);
  if (input.bButton !== undefined && input.bButton !== null) parts.push(`B=${Boolean(input.bButton)}`);
  if (input.teleopHeld !== undefined && input.teleopHeld !== null) parts.push(`hold=${Boolean(input.teleopHeld)}`);
  return parts.join(', ');
}

function robotGripperSummaryText(event) {
  if (!event) return 'n/a';
  const device = event.device ? ` ${event.device}` : '';
  const trigger = Number.isFinite(event.trigger) ? ` trig ${Number(event.trigger).toFixed(2)}` : '';
  const width = Number.isFinite(event.targetWidthM) ? ` width ${(Number(event.targetWidthM) * 1000).toFixed(1)}mm` : '';
  const attempts = Array.isArray(event.status?.enableAttempts) ? ` attempts ${event.status.enableAttempts.length}` : '';
  if (event.ok && event.commandSent) return `${event.action || 'move'}${device}${trigger}${width}`;
  return `${event.reason || event.error || event.action || 'skip'}${device}${trigger}${attempts}`;
}

function robotGripperStatusText(gripper, config) {
  if (config?.gripperEnabled !== true) return 'off';
  if (!gripper) return 'enabled, status missing';
  const parts = [gripper.enabled ? 'ready' : 'not ready'];
  if (gripper.device) parts.push(`device=${gripper.device}`);
  else if (config?.gripperDevice) parts.push(`target=${config.gripperDevice}`);
  const paramsName = gripper.params?.name;
  if (paramsName && paramsName !== gripper.device) parts.push(`params=${paramsName}`);
  if (Number.isFinite(gripper.states?.width)) parts.push(`width=${(Number(gripper.states.width) * 1000).toFixed(1)}mm`);
  if (gripper.lastError) parts.push(`error=${gripper.lastError}`);
  if (Array.isArray(gripper.enableAttempts) && gripper.enableAttempts.length) {
    const last = gripper.enableAttempts[gripper.enableAttempts.length - 1];
    parts.push(`lastAttempt=${last.device || 'n/a'}:${last.ok ? 'ok' : 'fail'}`);
  }
  return parts.join(', ');
}

function robotDevicesText(devices) {
  const list = devices?.list;
  if (!list || typeof list !== 'object') return devices?.lastError ? `unavailable (${devices.lastError})` : 'n/a';
  const names = Object.keys(list);
  if (!names.length) return 'none';
  return names.map(name => `${name}${list[name] ? '*' : ''}`).join(', ');
}

function robotModelStatusText() {
  const model = state.robotModel;
  if (!model) return 'loading';
  if (!model.ok) return `unavailable ${model.reason || ''}`.trim();
  const active = Array.isArray(model.activeJointNames) ? model.activeJointNames.length : 0;
  const source = model.sourceLabel ? ` (${model.sourceLabel})` : '';
  return `${model.name || 'Rizon'} URDF${source}, ${active} active joints`;
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
  const minDeg = Number(diversity.minRotationSpanDeg || 2.0);
  const ok = deg >= minDeg;
  return `${ok ? 'ok' : 'need rotation'} ${mm.toFixed(1)}mm / ${deg.toFixed(2)}deg`;
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
  const currentPhase = state.captureState?.phase || 'live';
  if (currentPhase === 'saving') {
    renderCaptureBanner();
    return;
  }
  const isRecording = !!frame.isRecording;
  if (frame.type === 'status') {
    state.captureState = {
      phase: isRecording ? 'recording' : 'live',
      recordId: frame.recordId,
      detail: isRecording ? 'PC formal recording' : 'not recording'
    };
  }
  renderCaptureBanner();
  statusEl.textContent =
    `mode: ${frame.telemetryMode ?? (isRecording ? 'recording' : 'live_preview')}\n` +
    `record: ${frame.recordId ?? 'n/a'}\n` +
    `waiting for ${isRecording ? 'recording' : 'live'} samples...`;
}

function updateCaptureState(event) {
  state.captureState = {
    ...(state.captureState || {}),
    ...(event || {})
  };
  renderCaptureBanner();
  syncQuestCommandButtons();
}

function renderCaptureBanner() {
  const phase = state.captureState?.phase || 'live';
  const isRecording = phase === 'recording';
  const isSaving = phase === 'saving';
  recordingBanner.classList.toggle('recording', isRecording);
  recordingBanner.classList.toggle('saving', isSaving);
  recordingLabel.textContent = isSaving ? 'SAVING' : (isRecording ? 'REC' : 'LIVE');
  recordingDetail.textContent = state.captureState?.detail || (isSaving ? 'saving recording' : (isRecording ? 'recording' : 'not recording'));
}

function captureBusy() {
  const phase = state.captureState?.phase || 'live';
  return phase === 'recording' || phase === 'saving';
}

function syncQuestCommandButtons() {
  const ready = Boolean(state.questAdb?.ready);
  const busy = captureBusy();
  questCalibStart.disabled = !ready || busy;
  questCalibStop.disabled = !ready;
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
  const robotClass = robot.recording ? 'calibration-ok' : 'calibration-alert';
  const mode = robot.controlMode || 'n/a';
  const cartesianLoop = robot.cartesianControlLoopAlive || robot.freeDragLoopAlive || robot.freedriveLoopAlive;
  const controlText = mode === 'freedrive'
    ? `free-drag ${robot.freeDragEnabled || robot.freedriveEnabled ? 'enabled' : 'not enabled'}${robot.freeDragPlan || robot.freedrivePlan || robot.freeDragMethod || robot.freedriveMethod ? ` via ${robot.freeDragPlan || robot.freedrivePlan || robot.freeDragMethod || robot.freedriveMethod}` : ''}${cartesianLoop ? ', cartesian loop running' : ''}${robot.cartesianSendSignature ? `, ${robot.cartesianSendSignature}` : ''}`
    : (mode === 'controller_teleop' ? 'hold right middle-finger trigger' : mode);
  calibrationDetails.innerHTML = `
    <div class="${robotClass}">${escapeHtml(event.message || 'PC calibration recording')}</div>
    <div class="calibration-kv">
      <span>record</span><span>${escapeHtml(event.recordId || 'n/a')}</span>
      <span>robot rec</span><span>${robot.recording ? 'yes' : 'no'}</span>
      <span>robot</span><span>${escapeHtml(robot.robotConnected ? `${robot.robotSn || 'connected'} ${robot.poseField || ''}` : 'not connected')}</span>
      <span>camera</span><span>${escapeHtml(robot.cameraSerial || 'n/a')}</span>
      <span>control</span><span>${escapeHtml(controlText)}</span>
      <span>note</span><span>${escapeHtml(robot.reason || 'n/a')}</span>
    </div>`;
}

function updateCalibrationResult(event) {
  state.latestCalibrationRecordId = event.recordId || event.rawRecordName || state.latestCalibrationRecordId;
  state.calibration = {
    ...(state.calibration || {}),
    result: event
  };
  const median = Number.isFinite(event.medianReprojectionPx) ? `${event.medianReprojectionPx.toFixed(2)}px` : 'n/a';
  mCalibration.textContent = `done ${median}`;
  mCalibrationFill.style.width = '100%';
  if (event.T_world_board?.translation_m) {
    state.origin = event.T_world_board.translation_m.slice();
    state.target = [0, 0, 0];
    const bounds = computeBounds();
    state.distance = Math.max(0.55, bounds.radius * 2.4);
  }
  renderCalibrationResult(event);
}

function updateCalibrationFailure(event) {
  state.latestCalibrationRecordId = event.recordId || event.rawRecordName || state.latestCalibrationRecordId;
  state.calibration = {
    ...(state.calibration || {}),
    failure: event
  };
  mCalibration.textContent = `failed ${event.reasonCode || ''}`.trim();
  mCalibrationFill.style.width = '100%';
  renderCalibrationDiagnostics(event);
}

function applyCalibrationCleared(event) {
  state.calibration = null;
  state.latestCalibrationRecordId = null;
  state.origin = [0, 0, 0];
  state.target = [0, 0, 0];
  mCalibration.textContent = event?.removedCount ? `cleared ${event.removedCount}` : 'idle';
  mCalibrationFill.style.width = '0%';
  calibrationDetails.innerHTML = '<div class="calibration-alert">Calibration cleared. Using default board at origin.</div>';
  requestViewFit();
  loadPreflightStatus();
}

async function clearCalibrationClick() {
  const now = performance.now();
  if (now > state.clearCalibrationConfirmUntilMs) {
    state.clearCalibrationConfirmUntilMs = now + 3500;
    clearCalibration.textContent = 'Confirm Clear';
    clearCalibration.classList.add('confirm');
    window.setTimeout(() => {
      if (performance.now() > state.clearCalibrationConfirmUntilMs) {
        clearCalibration.textContent = 'Clear';
        clearCalibration.classList.remove('confirm');
      }
    }, 3600);
    return;
  }
  state.clearCalibrationConfirmUntilMs = 0;
  clearCalibration.disabled = true;
  clearCalibration.textContent = 'Clearing...';
  clearCalibration.classList.remove('confirm');
  try {
    const response = await fetch('/calibration/clear', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: '{}'
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || payload.reason || `HTTP ${response.status}`);
    }
    applyCalibrationCleared(payload);
    applyRobotCalibrationCleared(payload);
    await loadRobotStatus();
  } catch (error) {
    calibrationDetails.innerHTML = `<div class="calibration-alert">${escapeHtml(String(error))}</div>`;
  } finally {
    clearCalibration.disabled = false;
    clearCalibration.textContent = 'Clear';
  }
}

function renderCalibrationResult(event) {
  const median = Number.isFinite(event.medianReprojectionPx) ? `${event.medianReprojectionPx.toFixed(2)}px` : 'n/a';
  const p90 = Number.isFinite(event.p90ReprojectionPx) ? `${event.p90ReprojectionPx.toFixed(2)}px` : 'n/a';
  const lag = Number.isFinite(event.bestLagSeconds) ? `${(event.bestLagSeconds * 1000).toFixed(1)} ms` : 'n/a';
  const normalAngle = Number.isFinite(event.boardNormalAbsAngleToWorldZDeg)
    ? `${event.boardNormalAbsAngleToWorldZDeg.toFixed(1)} deg`
    : 'n/a';
  calibrationDetails.innerHTML = `
    <div class="calibration-ok">Calibration succeeded</div>
    <div class="calibration-kv">
      <span>record</span><span>${escapeHtml(event.rawRecordName || event.recordId || 'n/a')}</span>
      <span>image y</span><span>${escapeHtml(event.imageYAxis || 'n/a')}</span>
      <span>lag</span><span>${lag}</span>
      <span>kept frames</span><span>${event.keptFrames ?? 'n/a'} / ${event.inputFrames ?? 'n/a'}</span>
      <span>appearance anchor</span><span>${event.appearanceAnchorFrames ?? 'n/a'} / ${event.keptFrames ?? 'n/a'}</span>
      <span>rot180</span><span>${event.rot180Frames ?? 'n/a'}</span>
      <span>median / p90</span><span>${median} / ${p90}</span>
      <span>board Z vs world Z</span><span>${normalAngle}</span>
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
      <span>appearance anchor</span><span>${summary.appearance_anchor_frames ?? 'n/a'} / ${summary.kept_frames ?? 'n/a'}</span>
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
    `${row.side} f${row.frame_index}: ${fmtPx(row.best_median_px)} ${row.best_order} ${row.order_source || ''}` +
    (Number.isFinite(row.appearance_anchor_observed_index) && row.appearance_anchor_observed_index >= 0
      ? ` app ${row.appearance_anchor_observed_index}->${row.appearance_anchor_target_index}`
      : '')
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
  drawRobotLive();

  const frame = state.frames[state.frames.length - 1];
  if (frame) {
    if (showGaze.checked) drawGaze(frame);
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
  const jointpose = liveRobotJointpose();
  if (!jointpose) return;
  const sample = state.robotSample || {};
  const baseMatrix = robotLiveBaseMatrix();
  drawRobotMeshes(jointpose, baseMatrix);
  drawRobotSkeleton(jointpose, baseMatrix);
  let tcpMatrix = sample.T_world_tool_tcp?.matrix_4x4;
  const baseTcp = sample.T_base_tool_tcp?.matrix_4x4 || sample.T_base_ee?.matrix_4x4;
  if (!tcpMatrix && baseTcp) tcpMatrix = multiplyMatrix4(baseMatrix, baseTcp);
  let endCameraMatrix = sample.T_world_end_camera?.matrix_4x4;
  if (!endCameraMatrix && sample.T_base_end_camera?.matrix_4x4) {
    endCameraMatrix = multiplyMatrix4(baseMatrix, sample.T_base_end_camera.matrix_4x4);
  }
  const p = relPoint(matrixTranslation(tcpMatrix));
  if (p) {
    drawPoint(p, '#ffffff', 6, 'TCP');
    drawLine(p, relPoint(matrixPoint(tcpMatrix, 0.08, 0, 0)), '#ff4545', 2.3);
    drawLine(p, relPoint(matrixPoint(tcpMatrix, 0, 0.08, 0)), '#42e875', 2.3);
    drawLine(p, relPoint(matrixPoint(tcpMatrix, 0, 0, 0.08)), '#4b7cff', 2.3);
  }
  const cp = relPoint(matrixTranslation(endCameraMatrix));
  if (cp) {
    drawPoint(cp, '#82adff', 5, 'end cam');
    drawLine(cp, relPoint(matrixPoint(endCameraMatrix, 0.06, 0, 0)), '#ff4545', 1.9);
    drawLine(cp, relPoint(matrixPoint(endCameraMatrix, 0, 0.06, 0)), '#42e875', 1.9);
    drawLine(cp, relPoint(matrixPoint(endCameraMatrix, 0, 0, 0.06)), '#4b7cff', 1.9);
    if (p) drawLine(p, cp, 'rgba(130,173,255,.42)', 1.2);
  }
}

function robotLiveBaseMatrix() {
  return state.robotWorldBase || unalignedRobotBaseMatrix();
}

function unalignedRobotBaseMatrix() {
  return identityMatrix4();
}

function liveRobotJointpose() {
  if (Array.isArray(state.robotSample?.jointpose) && state.robotSample.jointpose.length >= 7) {
    return state.robotSample.jointpose;
  }
  if (!state.robotModel?.ok) return null;
  const count = Math.max(7, Number(state.robotModel.activeJointNames?.length || 0));
  return Array.from({length: count}, () => 0);
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

function drawRobotMeshes(jointpose, baseMatrix) {
  const model = state.robotModel;
  if (!model || !Array.isArray(jointpose) || jointpose.length < 7 || !baseMatrix) return;
  const linkFrames = robotLinkFrames(model, jointpose, baseMatrix);
  const polygons = [];
  for (const [linkName, link] of Object.entries(model.links || {})) {
    const linkFrame = linkFrames[linkName];
    if (!linkFrame) continue;
    for (const visual of link.visuals || []) {
      const assetPath = visual.mesh?.assetPath;
      const mesh = assetPath ? state.robotMeshes[assetPath] : null;
      if (!mesh) continue;
      const visualFrame = multiplyMatrix4(linkFrame, xyzRpyMatrix(visual.xyz, visual.rpy));
      collectMeshPolygons(polygons, mesh, visualFrame, visual.mesh?.scale, visual.name || linkName, relPoint);
    }
  }
  renderMeshPolygons(polygons);
}

function preloadRobotMeshes(model) {
  if (!model?.ok) return;
  for (const link of Object.values(model.links || {})) {
    for (const visual of link.visuals || []) {
      const mesh = visual.mesh || {};
      if (!mesh.exists || !mesh.assetPath || !mesh.url) continue;
      if (state.robotMeshes[mesh.assetPath] || state.robotMeshPending[mesh.assetPath]) continue;
      state.robotMeshPending[mesh.assetPath] = fetch(mesh.url, {cache: 'no-store'})
        .then(response => response.ok ? response.text() : '')
        .then(text => {
          if (text) state.robotMeshes[mesh.assetPath] = parseObjMesh(text);
        })
        .catch(() => {})
        .finally(() => { delete state.robotMeshPending[mesh.assetPath]; });
    }
  }
}

function parseObjMesh(text) {
  const vertices = [];
  const faces = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line[0] === '#') continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) {
      vertices.push([Number(parts[1]), Number(parts[2]), Number(parts[3])]);
    } else if (parts[0] === 'f' && parts.length >= 4) {
      const indices = parts.slice(1).map(token => objIndex(token, vertices.length)).filter(index => index !== null);
      for (let i = 1; i + 1 < indices.length; i++) faces.push([indices[0], indices[i], indices[i + 1]]);
    }
  }
  return {vertices, faces};
}

function objIndex(token, count) {
  const raw = Number.parseInt(String(token).split('/')[0], 10);
  if (!Number.isFinite(raw) || raw === 0) return null;
  const index = raw > 0 ? raw - 1 : count + raw;
  return index >= 0 && index < count ? index : null;
}

function collectMeshPolygons(polygons, mesh, matrix, scaleValue, visualName, pointMapper) {
  const vertices = mesh.vertices || [];
  const faces = mesh.faces || [];
  if (!vertices.length || !faces.length) return;
  const scale = Array.isArray(scaleValue) ? scaleValue : [1, 1, 1];
  const stride = Math.max(1, Math.ceil(faces.length / 900));
  const isRing = String(visualName || '').toLowerCase().includes('ring');
  for (let i = 0; i < faces.length; i += stride) {
    const face = faces[i];
    const pts = [];
    for (const index of face) {
      const v = vertices[index];
      const p = pointMapper(matrixPointScaled(matrix, v, scale));
      if (!p) {
        pts.length = 0;
        break;
      }
      pts.push(p);
    }
    if (pts.length === 3) polygons.push({pts, ring: isRing});
  }
}

function renderMeshPolygons(polygons) {
  if (!polygons.length) return;
  const projected = [];
  for (const poly of polygons) {
    const pts = poly.pts.map(project);
    if (pts.some(p => !p.visible)) continue;
    const area = Math.abs(
      (pts[1].x - pts[0].x) * (pts[2].y - pts[0].y) -
      (pts[2].x - pts[0].x) * (pts[1].y - pts[0].y)
    );
    if (area < 0.18) continue;
    projected.push({
      pts,
      depth: (pts[0].z + pts[1].z + pts[2].z) / 3,
      ring: poly.ring,
      area
    });
  }
  projected.sort((a, b) => b.depth - a.depth);
  ctx.save();
  ctx.lineWidth = 0.45;
  for (const poly of projected) {
    ctx.beginPath();
    ctx.moveTo(poly.pts[0].x, poly.pts[0].y);
    ctx.lineTo(poly.pts[1].x, poly.pts[1].y);
    ctx.lineTo(poly.pts[2].x, poly.pts[2].y);
    ctx.closePath();
    ctx.fillStyle = poly.ring ? 'rgba(238,240,244,0.32)' : 'rgba(150,164,184,0.24)';
    ctx.strokeStyle = poly.ring ? 'rgba(255,255,255,0.15)' : 'rgba(210,220,235,0.08)';
    ctx.fill();
    if (poly.area > 8) ctx.stroke();
  }
  ctx.restore();
}

function updateStatus(frame) {
  updateRecordingStatus(frame);
  const phase = state.captureState?.phase || 'live';
  const isRecording = phase === 'recording' || !!frame.isRecording;
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
  const phase = state.captureState?.phase || (!!frame.isRecording ? 'recording' : 'live');
  const isRecording = phase === 'recording';
  const isSaving = phase === 'saving';
  const label = isSaving ? 'SAVING' : (isRecording ? 'REC' : 'LIVE');
  const detail = isSaving
    ? (state.captureState?.detail || 'saving recording')
    : (isRecording ? (frame.recordId ?? state.captureState?.recordId ?? 'recording') : 'not recording');
  ctx.save();
  ctx.font = '700 18px system-ui, sans-serif';
  const labelWidth = ctx.measureText(label).width;
  ctx.font = '12px system-ui, sans-serif';
  const detailWidth = ctx.measureText(detail).width;
  const width = Math.max(104, labelWidth + detailWidth + 54);
  ctx.fillStyle = isSaving ? 'rgba(59,50,20,0.94)' : (isRecording ? 'rgba(88,20,28,0.92)' : 'rgba(28,35,41,0.88)');
  ctx.strokeStyle = isSaving ? 'rgba(242,201,76,0.95)' : (isRecording ? 'rgba(255,79,94,0.95)' : 'rgba(80,96,108,0.85)');
  roundRect(16, 16, width, 42, 8);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(37, 37, 6, 0, Math.PI * 2);
  ctx.fillStyle = isSaving ? '#f2c94c' : (isRecording ? '#ff4f5e' : '#7b8994');
  ctx.fill();
  if (isRecording || isSaving) {
    ctx.shadowColor = isSaving ? 'rgba(242,201,76,0.9)' : 'rgba(255,79,94,0.95)';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.fillStyle = '#fff';
  ctx.font = '700 18px system-ui, sans-serif';
  ctx.fillText(label, 51, 42);
  ctx.fillStyle = isSaving ? '#fff2bd' : (isRecording ? '#ffd9dd' : '#b8c4ce');
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
    drawLine([-extent, v, 0], [extent, v, 0], color, major ? 1.1 : 0.8);
    drawLine([v, -extent, 0], [v, extent, 0], color, major ? 1.1 : 0.8);
  }
  drawLine([0,0,0], [0.35,0,0], 'rgba(255,75,75,0.8)', 2);
  drawLine([0,0,0], [0,0.35,0], 'rgba(75,255,120,0.8)', 2);
  drawLine([0,0,0], [0,0,0.35], 'rgba(75,125,255,0.8)', 2);
}

function drawCalibrationBoard() {
  const {matrix, pattern, square} = liveBoardSpec();
  if (!matrix || !state.origin) return;
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
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const z1 = z;
  const screenY = cp * z1 - sp * y1;
  const depth = sp * z1 + cp * y1 + state.distance;
  const f = Math.min(rect.width, rect.height) * 0.92;
  return {
    x: rect.width * 0.5 + x1 * f / Math.max(0.03, depth),
    y: rect.height * 0.5 - screenY * f / Math.max(0.03, depth),
    z: depth,
    visible: depth > 0.03
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

function matrixPointScaled(m, v, scaleValue) {
  const scale = Array.isArray(scaleValue) ? scaleValue : [1, 1, 1];
  return matrixPoint(
    m,
    Number(v?.[0] || 0) * Number(scale[0] || 1),
    Number(v?.[1] || 0) * Number(scale[1] || 1),
    Number(v?.[2] || 0) * Number(scale[2] || 1)
  );
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

function defaultBoardMatrix4() {
  return identityMatrix4();
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

function robotLinkFrames(model, jointpose, baseMatrix) {
  const frames = {base_link: baseMatrix};
  let jointIndex = 0;
  for (const joint of model.joints || []) {
    const parentFrame = frames[joint.parent] || baseMatrix;
    let childFrame = multiplyMatrix4(parentFrame, xyzRpyMatrix(joint.xyz, joint.rpy));
    if (joint.type !== 'fixed') {
      const angle = Number(jointpose[jointIndex++] || 0);
      childFrame = multiplyMatrix4(childFrame, axisAngleMatrix(joint.axis, angle));
    }
    if (joint.child) frames[joint.child] = childFrame;
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
robotStreamStart.addEventListener('click', startRobotStream);
robotStreamStop.addEventListener('click', stopRobotStream);
robotBoardCheck.addEventListener('click', checkRobotBoard);
robotDiagnostics.addEventListener('click', runRobotDiagnostics);
robotGripOpen.addEventListener('click', () => moveRobotGripper('open'));
robotGripClose.addEventListener('click', () => moveRobotGripper('close'));
robotConnect.addEventListener('click', connectRobot);
robotDisconnect.addEventListener('click', disconnectRobot);
preflightRefresh.addEventListener('click', loadPreflightStatus);
clearCalibration.addEventListener('click', clearCalibrationClick);
questAdbRefresh.addEventListener('click', loadQuestAdbStatus);
questCalibStart.addEventListener('click', () => sendQuestCalibrationCommand('calib_start'));
questCalibStop.addEventListener('click', () => sendQuestCalibrationCommand('calib_stop'));
for (const input of [robotCamera, robotThirdCamera, robotSn, robotPoseField, robotNetworkInterfaces, robotStateHz, robotInterval, robotHandEye, robotExposureMode, robotExposure, robotGain, robotBoardWarmup, robotRecordDepth, robotRecordDepthAlign, robotRecordDepthEveryNFrames, robotMotionScale, robotMaxStep, robotMaxRotationStep, robotJointLimitBuffer, robotJointLimitGuard, robotGripperEnabled, robotGripperDevice, robotGripperOpen, robotGripperClose, robotGripperSpeed, robotGripperForce]) {
  input.addEventListener('change', configureRobot);
}
document.getElementById('records').addEventListener('click', () => {
  window.location.href = '/recordings';
});
window.addEventListener('resize', resize);

function initSidebarResize(storageKey, minWidth, maxWidth, defaultWidth) {
  const saved = Number(localStorage.getItem(storageKey));
  setSidebarWidth(Number.isFinite(saved) ? saved : defaultWidth, minWidth, maxWidth, storageKey);
  let active = false;
  sidebarResizer.addEventListener('pointerdown', event => {
    active = true;
    sidebarResizer.classList.add('dragging');
    sidebarResizer.setPointerCapture(event.pointerId);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    event.preventDefault();
  });
  sidebarResizer.addEventListener('pointermove', event => {
    if (!active) return;
    const width = window.innerWidth - event.clientX - 4;
    setSidebarWidth(width, minWidth, maxWidth, storageKey);
    resize();
  });
  const stop = event => {
    if (!active) return;
    active = false;
    sidebarResizer.classList.remove('dragging');
    try { sidebarResizer.releasePointerCapture(event.pointerId); } catch (_) {}
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    resize();
  };
  sidebarResizer.addEventListener('pointerup', stop);
  sidebarResizer.addEventListener('pointercancel', stop);
}

function setSidebarWidth(width, minWidth, maxWidth, storageKey) {
  const viewportLimit = Math.max(minWidth, window.innerWidth - 360);
  const clamped = Math.max(minWidth, Math.min(maxWidth, viewportLimit, Math.round(width)));
  app.style.setProperty('--sidebar-width', `${clamped}px`);
  localStorage.setItem(storageKey, String(clamped));
}

initSidebarResize('questLiveSidebarWidth', 300, 720, 330);
resize();
connect();
loadLatestCalibration();
setInterval(loadLatestCalibration, 2000);
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
  --detail-width: 340px;
  grid-template-columns: 320px minmax(0, 1fr) 8px var(--detail-width);
  height: 100vh;
}
nav, aside {
  overflow: auto;
  background: var(--panel);
  padding: 12px;
}
nav { border-right: 1px solid var(--line); }
.detail-resizer {
  width: 8px;
  border-left: 1px solid var(--line);
  background: #101820;
  cursor: col-resize;
  touch-action: none;
}
.detail-resizer:hover,
.detail-resizer.dragging {
  background: #24344a;
}
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
.camera-strip {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}
.camera-strip a {
  display: grid;
  gap: 4px;
  color: #a8d8ff;
  text-decoration: none;
  font-size: 12px;
}
.camera-strip img {
  width: 100%;
  max-height: 120px;
  object-fit: contain;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #080a0c;
}
.ok { color: #9df09d; font-weight: 700; }
.warn { color: #ffd18a; font-weight: 700; }
.rec { color: #ff8c96; font-weight: 700; }
@media (max-width: 1050px) {
  #app { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr) auto; }
  .detail-resizer { display: none; }
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
  <div id="detailResizer" class="detail-resizer" title="Drag to resize details"></div>
  <aside>
    <h1 id="recordTitle">Select a record</h1>
    <div class="sub" id="recordSub">Drag to orbit, wheel to zoom.</div>
    <div class="topbar">
      <button id="playBtn">Play</button>
      <button id="resetBtn">Reset</button>
      <span class="pill">pivot origin</span>
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
      <div class="camera-strip" id="cameraStrip"></div>
    </div>
  </aside>
</div>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const app = document.getElementById('app');
const detailResizer = document.getElementById('detailResizer');
const recordList = document.getElementById('recordList');
const rootLabel = document.getElementById('rootLabel');
const search = document.getElementById('search');
const liveBtn = document.getElementById('liveBtn');
const refreshBtn = document.getElementById('refreshBtn');
const playBtn = document.getElementById('playBtn');
const resetBtn = document.getElementById('resetBtn');
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
const cameraStrip = document.getElementById('cameraStrip');

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
  robotMeshes: {},
  robotMeshPending: {},
  cameraStripKey: '',
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
    state.robotMeshes = {};
    state.robotMeshPending = {};
    preloadRobotMeshes(payload);
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
  const perf = effectiveRobotPerformance(summary.performance);
  const robotHz = perf?.robotState?.effectiveHz;
  if (Number.isFinite(Number(robotHz))) parts.push(`state ${Number(robotHz).toFixed(1)}Hz`);
  const cameraText = robotPerformanceCameraBrief(perf);
  if (cameraText) parts.push(cameraText);
  if (summary.residualMedianMm !== null && summary.residualMedianMm !== undefined && Number.isFinite(Number(summary.residualMedianMm))) {
    parts.push(`res ${Number(summary.residualMedianMm).toFixed(1)}mm`);
  }
  if (summary.status === 'failed' && summary.failureReason) {
    parts.push(String(summary.failureReason).slice(0, 80));
  }
  return parts.join(' | ');
}

function effectiveRobotPerformance(performancePayload) {
  const payload = performancePayload || {};
  const robot = payload.robotState || {};
  const cameras = payload.cameras || {};
  const hasRobot = Number.isFinite(Number(robot.effectiveHz)) || Number(robot.count || 0) > 0;
  const hasCamera = Object.values(cameras).some(role => {
    const video = role?.video || {};
    return Number.isFinite(Number(video.effectiveHz)) || Number(video.count || 0) > 0;
  });
  if (hasRobot || hasCamera) return payload;
  return payload.computedFromJsonl || payload;
}

function robotPerformanceCameraBrief(performancePayload) {
  const perf = effectiveRobotPerformance(performancePayload);
  const cameras = perf?.cameras || {};
  const parts = [];
  for (const [role, row] of Object.entries(cameras)) {
    const hz = Number(row?.video?.effectiveHz);
    const drop = Number(row?.queueDropRatio);
    if (Number.isFinite(hz)) {
      parts.push(`${role} ${hz.toFixed(1)}Hz${Number.isFinite(drop) && drop > 0 ? ` drop ${(drop * 100).toFixed(1)}%` : ''}`);
    }
  }
  return parts.join(', ');
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
  state.cameraStripKey = '';
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
  for (const row of robotPoseRows()) {
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
  const audit = state.data?.recordingAudit;
  if (!rr) {
    const start = state.data?.summary?.robotStartStatus;
    const auditRows = renderAuditKvRows(audit);
    if (start) {
      robotKv.innerHTML = [
        ['status', 'no robot recording'],
        ['start', replayRobotStartText(start)],
        ['note', start.reason || 'n/a']
      ].map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('') + auditRows;
      return;
    }
    robotKv.innerHTML = '<span>status</span><span>no robot recording</span>' + auditRows;
    return;
  }
  const result = rr.result || {};
  const failure = rr.failure || {};
  const session = rr.session || {};
  const config = rr.config || {};
  const counts = result.counts || failure.counts || {};
  const residual = result.end_camera?.residuals?.translation_mm || {};
  const align = result.questAlignment || {};
  const diversity = result.diversity || failure.diversity || rr.poseDiversity;
  const robotStates = Array.isArray(rr.robotStates) ? rr.robotStates.length : 0;
  const sample = state.data?.samples?.[state.idx] || {};
  const robot = nearestRobotSample(sample.recordingTimestampSeconds);
  const fkError = replayRobotFkErrorMm(robot);
  const cameraRoles = Array.isArray(session.cameraRoles) ? session.cameraRoles.join(', ') : 'n/a';
  const gripperEvents = Array.isArray(rr.gripper) ? rr.gripper.filter(row => row.commandSent).length : 0;
  const perf = effectiveRobotPerformance(rr.performance || {});
  const robotHz = Number(perf?.robotState?.effectiveHz);
  const robotGap = Number(perf?.robotState?.gapSeconds?.p95);
  const cameraPerfText = robotPerformanceCameraBrief(perf) || 'n/a';
  const depthEnabled = config.recordDepth !== false;
  const depthFrames = session.depthFrames ?? rr.depthFrameCount ?? 0;
  const detections = counts.requiredDetections !== undefined
    ? `${counts.detections ?? 'n/a'} / ${counts.requiredDetections}`
    : (counts.detections ?? 'n/a');
  const kv = [
    ['samples', `${rr.samples?.length || 0} aligned / ${robotStates} states`],
    ['detections', detections],
    ['model', replayRobotModelStatusText()],
    ['URDF FK', Number.isFinite(fkError) ? `${fkError.toFixed(1)}mm vs flange` : 'n/a'],
    ['ee motion', replayPoseDiversityText(diversity)],
    ['cameras', cameraRoles],
    ['perf', `${Number.isFinite(robotHz) ? `state ${robotHz.toFixed(1)}Hz` : 'state n/a'}${Number.isFinite(robotGap) ? ` gap95 ${(robotGap * 1000).toFixed(1)}ms` : ''}; ${cameraPerfText}`],
    ['depth', depthEnabled ? `${depthFrames} depth frames / ${rr.videoFrameCount ?? 'n/a'} video rows` : 'disabled'],
    ['motion', replayMotionSummaryText(session)],
    ['gripper', `cmd ${session.gripperCommands ?? gripperEvents}, err ${session.gripperErrors ?? 0}`],
    ['hand-eye', result.ok ? 'ok' : (rr.failure ? 'failed' : 'pending')],
    ['failure', rr.failure?.error || 'n/a'],
    ['residual', Number.isFinite(residual.median) ? `med ${residual.median.toFixed(1)}mm p95 ${Number(residual.p95 || 0).toFixed(1)}mm` : 'n/a'],
    ['quest-base', align.ok ? 'T_world_base ready' : (align.reason || 'n/a')]
  ];
  robotKv.innerHTML = kv.map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
  robotKv.innerHTML += renderAuditKvRows(audit);
}

function renderAuditKvRows(audit) {
  if (!audit) return '';
  const rows = [];
  if (audit.summary) rows.push(['audit', audit.summary]);
  if (Array.isArray(audit.checks)) {
    for (const check of audit.checks) {
      const mark = check.ok ? 'ok' : 'missing';
      rows.push([check.label || check.id, `${mark}: ${check.detail || ''}`]);
    }
  }
  return rows.map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
}

function replayRobotStartText(start) {
  if (!start) return 'n/a';
  const mode = start.controlMode || (start.teleopRequiresRightHandTrigger ? 'controller_teleop' : 'legacy');
  const detail = mode === 'freedrive'
    ? `freeDrag=${Boolean(start.freeDragEnabled || start.freedriveEnabled)}${start.freeDragPlan || start.freedrivePlan ? ` plan=${start.freeDragPlan || start.freedrivePlan}` : ''}`
    : `teleop=${start.teleopRequiresRightHandTrigger ? 'right middle trigger' : (start.teleopRequiresRightSideButton ? 'right side button' : 'legacy')}`;
  return `recording=${Boolean(start.recording)}, connected=${Boolean(start.robotConnected)}, control=${mode}, ${detail}`;
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
  const source = model.sourceLabel ? ` (${model.sourceLabel})` : '';
  return `${model.name || 'Rizon'} URDF${source}, ${active} active joints`;
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
  const minDeg = Number(diversity.minRotationSpanDeg || 2.0);
  const ok = deg >= minDeg;
  return `${ok ? 'ok' : 'need rotation'} ${mm.toFixed(1)}mm / ${deg.toFixed(2)}deg`;
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
    cameraStrip.innerHTML = '';
    return;
  }
  timeLabel.textContent = `${Number(s.recordingTimestampSeconds || 0).toFixed(3)}s`;
  sampleLabel.textContent = `sample ${s.sampleIndex ?? state.idx}`;
  recordingLabel.textContent = s.isRecording ? 'REC' : 'LIVE';
  recordingLabel.className = s.isRecording ? 'rec' : 'ok';
  const robot = nearestRobotSample(s.recordingTimestampSeconds);
  const robotMedia = nearestRobotMediaSample(s.recordingTimestampSeconds);
  const imageRoles = robotMedia?.images ? Object.keys(robotMedia.images).filter(role => robotMedia.images[role]?.url) : [];
  const videoRoles = robotMedia?.videos ? Object.keys(robotMedia.videos).filter(role => robotMedia.videos[role]?.url) : [];
  const gripper = nearestGripperEvent(s.recordingTimestampSeconds);
  const kv = [
    ['gaze3D', s.gaze?.ok ? (s.gaze.source || 'ok') : 'missing'],
    ['depth raw', Number.isFinite(s.gazeDepth?.rawDepthM) ? (s.gazeDepth.rawDepthM * 1000).toFixed(1) + ' mm' : 'n/a'],
    ['board z', Number.isFinite(s.gazeDepth?.boardDistanceM) ? (s.gazeDepth.boardDistanceM * 1000).toFixed(1) + ' mm' : 'n/a'],
    ['plane err', Number.isFinite(s.gazeDepth?.depthMinusBoardPlaneM) ? (s.gazeDepth.depthMinusBoardPlaneM * 1000).toFixed(1) + ' mm' : 'n/a'],
    ['hit', s.gazeHit?.ok ? 'present' : 'missing'],
    ['left', s.left?.ok ? (s.left.source || 'ok') : (s.left?.source || 'missing')],
    ['right', s.right?.ok ? (s.right.source || 'ok') : (s.right?.source || 'missing')],
    ['left input', robotControllerInputText(s.left) || 'n/a'],
    ['right input', robotControllerInputText(s.right) || 'n/a'],
    ['head', s.head?.ok ? (s.head.source || 'ok') : 'missing'],
    ['robot sample', robot ? `${robot.sampleIndex ?? 'n/a'} / ${robot.sourceStream || 'robot'}` : 'n/a'],
    ['media', [...new Set([...imageRoles, ...videoRoles])].join(', ') || 'n/a'],
    ['gripper', gripper ? replayGripperText(gripper) : 'n/a']
  ];
  sampleKv.innerHTML = kv.map(([k,v]) => `<span>${escapeHtml(k)}</span><span>${escapeHtml(String(v))}</span>`).join('');
  renderCameraStrip(robotMedia);
}

function renderCameraStrip(robot) {
  const images = robot?.images || {};
  const videos = robot?.videos || {};
  const videoEntries = Object.entries(videos).filter(([, item]) => item?.url);
  const imageEntries = Object.entries(images).filter(([role, item]) => item?.url && !videos[role]?.url);
  const entries = [...videoEntries, ...imageEntries];
  if (!entries.length) {
    if (state.cameraStripKey !== '') {
      cameraStrip.innerHTML = '';
      state.cameraStripKey = '';
    }
    return;
  }
  const stripKey = entries.map(([role, item]) => {
    const url = String(item.url || '');
    const frame = Number.isFinite(item.frameIndex) ? String(item.frameIndex) : '';
    return `${role}:${url}:${frame}`;
  }).join('|');
  if (stripKey === state.cameraStripKey) return;
  state.cameraStripKey = stripKey;
  cameraStrip.innerHTML = entries.map(([role, item]) => {
    const label = `${role}${Number.isFinite(item.frameIndex) ? ` #${item.frameIndex}` : ''}`;
    if (String(item.url || '').toLowerCase().includes('.mp4')) {
      return `
        <a href="${escapeHtml(item.url)}" target="_blank">
          <video src="${escapeHtml(item.url)}" controls preload="metadata"></video>
          <span>${escapeHtml(label)}</span>
        </a>`;
    }
    return `
      <a href="${escapeHtml(item.url)}" target="_blank">
        <img src="${escapeHtml(item.url)}" alt="${escapeHtml(role)} camera">
        <span>${escapeHtml(label)}</span>
      </a>`;
  }).join('');
}

function robotControllerInputText(controller) {
  if (!controller) return '';
  const input = controller.input || {};
  if (!input.hasAny) return '';
  const parts = [];
  if (Number.isFinite(input.handTrigger)) parts.push(`hand=${Number(input.handTrigger).toFixed(2)}`);
  if (input.handTriggerPressed !== undefined && input.handTriggerPressed !== null) parts.push(`handPressed=${Boolean(input.handTriggerPressed)}`);
  if (Number.isFinite(input.indexTrigger)) parts.push(`index=${Number(input.indexTrigger).toFixed(2)}`);
  if (input.indexTriggerPressed !== undefined && input.indexTriggerPressed !== null) parts.push(`indexPressed=${Boolean(input.indexTriggerPressed)}`);
  if (input.aButton !== undefined && input.aButton !== null) parts.push(`A=${Boolean(input.aButton)}`);
  if (input.bButton !== undefined && input.bButton !== null) parts.push(`B=${Boolean(input.bButton)}`);
  if (input.teleopHeld !== undefined && input.teleopHeld !== null) parts.push(`hold=${Boolean(input.teleopHeld)}`);
  return parts.join(', ');
}

function nearestGripperEvent(t) {
  const rows = state.data?.robotRealSense?.gripper || [];
  if (!rows.length || !Number.isFinite(t)) return rows[rows.length - 1] || null;
  let best = null, bestDt = Infinity;
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

function replayGripperText(row) {
  const trig = Number.isFinite(row.trigger) ? ` trig ${Number(row.trigger).toFixed(2)}` : '';
  if (row.commandSent) return `${row.action || 'move'}${trig}`;
  return `${row.reason || row.error || row.action || 'skip'}${trig}`;
}

function boardSize() {
  const snap = state.data?.snapshot || {};
  const cols = Math.max(1, (snap.pattern?.[0] || 11) - 1);
  const rows = Math.max(1, (snap.pattern?.[1] || 8) - 1);
  const square = snap.squareSizeM || 0.025;
  return {cols, rows, square, width: cols * square, height: rows * square};
}

function boardPoints() {
  const m = replayBoardMatrix();
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
  state.pitch = -0.62;
  state.target = [0,0,0];
  const pts = boardPoints();
  for (const s of state.data?.samples || []) {
    for (const key of ['head','left','right','leftEye','rightEye','gaze','gazeFiltered','gazeBoardPlane','gazeHit']) {
      if (s[key]?.p) pts.push(s[key].p);
    }
  }
  for (const row of robotPoseRows()) {
    for (const key of ['T_display_tool_tcp', 'T_display_ee', 'T_display_end_camera']) {
      const p = matrixTranslation(row[key]?.matrix_4x4);
      if (p) pts.push(p);
    }
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
  state.target = [0,0,0];
  let radius = 0.12;
  for (const p of pts) radius = Math.max(radius, length(sub(p, state.target)));
  state.distance = Math.max(0.35, radius * 3.1);
}

function render() {
  const rect = canvas.getBoundingClientRect();
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
    drawLine([-extent,v,0], [extent,v,0], `rgba(255,255,255,${alpha})`, 1);
    drawLine([v,-extent,0], [v,extent,0], `rgba(255,255,255,${alpha})`, 1);
  }
}

function drawWorldAxes() {
  drawLine([0,0,0], [0.12,0,0], '#ff4545', 2.4);
  drawLine([0,0,0], [0,0.12,0], '#42e875', 2.4);
  drawLine([0,0,0], [0,0,0.12], '#4b7cff', 2.4);
}

function drawBoard() {
  const m = replayBoardMatrix();
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
  const baseMatrix = robotReplayBaseMatrix();
  const baseTcp = robot?.T_base_tool_tcp?.matrix_4x4 || robot?.T_base_ee?.matrix_4x4;
  const matrix = robot?.T_display_tool_tcp?.matrix_4x4 || robot?.T_display_ee?.matrix_4x4 || (baseTcp ? multiplyMatrix4(baseMatrix, baseTcp) : null);
  if (!matrix) return;
  drawRobotMeshes(robot);
  drawRobotSkeleton(robot);
  const p = matrixTranslation(matrix);
  if (!p) return;
  drawPoint(p, '#ffffff', 6, 'TCP');
  drawLine(p, matrixPoint(matrix, 0.08, 0, 0), '#ff4545', 2.3);
  drawLine(p, matrixPoint(matrix, 0, 0.08, 0), '#42e875', 2.3);
  drawLine(p, matrixPoint(matrix, 0, 0, 0.08), '#4b7cff', 2.3);
  const camMatrix = robot?.T_display_end_camera?.matrix_4x4 || (robot?.T_base_end_camera?.matrix_4x4 ? multiplyMatrix4(baseMatrix, robot.T_base_end_camera.matrix_4x4) : null);
  const cp = matrixTranslation(camMatrix);
  if (cp) {
    drawPoint(cp, '#82adff', 5, 'end cam');
    drawLine(cp, matrixPoint(camMatrix, 0.06, 0, 0), '#ff4545', 1.9);
    drawLine(cp, matrixPoint(camMatrix, 0, 0.06, 0), '#42e875', 1.9);
    drawLine(cp, matrixPoint(camMatrix, 0, 0, 0.06), '#4b7cff', 1.9);
    drawLine(p, cp, 'rgba(130,173,255,.42)', 1.2);
  }
}

function drawRobotMeshes(robot) {
  const model = state.robotModel;
  const jointpose = robot?.jointpose;
  if (!model || !Array.isArray(jointpose) || jointpose.length < 7) return;
  const displayBase = robotReplayBaseMatrix();
  const linkFrames = robotLinkFrames(model, jointpose, displayBase);
  const polygons = [];
  for (const [linkName, link] of Object.entries(model.links || {})) {
    const linkFrame = linkFrames[linkName];
    if (!linkFrame) continue;
    for (const visual of link.visuals || []) {
      const assetPath = visual.mesh?.assetPath;
      const mesh = assetPath ? state.robotMeshes[assetPath] : null;
      if (!mesh) continue;
      const visualFrame = multiplyMatrix4(linkFrame, xyzRpyMatrix(visual.xyz, visual.rpy));
      collectMeshPolygons(polygons, mesh, visualFrame, visual.mesh?.scale, visual.name || linkName, p => p);
    }
  }
  renderMeshPolygons(polygons);
}

function drawRobotSkeleton(robot) {
  const model = state.robotModel;
  const jointpose = robot?.jointpose;
  if (!model || !Array.isArray(jointpose) || jointpose.length < 7) return;
  const displayBase = robotReplayBaseMatrix();
  const frames = robotFrames(model, jointpose, displayBase);
  if (frames.length < 2) return;
  for (let i = 1; i < frames.length; i++) {
    drawLine(matrixTranslation(frames[i - 1]), matrixTranslation(frames[i]), 'rgba(255,255,255,.56)', 3);
  }
  for (let i = 0; i < frames.length; i++) {
    drawPoint(matrixTranslation(frames[i]), i === 0 ? '#cbd5df' : '#ffffff', i === 0 ? 4 : 3.5, i === frames.length - 1 ? 'flange' : '');
  }
}

function robotReplayBaseMatrix() {
  const result = state.data?.robotRealSense?.result;
  const baseMatrix = result?.questAlignment?.T_world_base?.matrix_4x4;
  return baseMatrix ? translateMatrixPayload(baseMatrix, state.data?.boardOriginWorld || [0,0,0]) : unalignedRobotBaseMatrix();
}

function unalignedRobotBaseMatrix() {
  return identityMatrix4();
}

function preloadRobotMeshes(model) {
  if (!model?.ok) return;
  for (const link of Object.values(model.links || {})) {
    for (const visual of link.visuals || []) {
      const mesh = visual.mesh || {};
      if (!mesh.exists || !mesh.assetPath || !mesh.url) continue;
      if (state.robotMeshes[mesh.assetPath] || state.robotMeshPending[mesh.assetPath]) continue;
      state.robotMeshPending[mesh.assetPath] = fetch(mesh.url, {cache: 'no-store'})
        .then(response => response.ok ? response.text() : '')
        .then(text => {
          if (text) state.robotMeshes[mesh.assetPath] = parseObjMesh(text);
        })
        .catch(() => {})
        .finally(() => { delete state.robotMeshPending[mesh.assetPath]; });
    }
  }
}

function parseObjMesh(text) {
  const vertices = [];
  const faces = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line[0] === '#') continue;
    const parts = line.split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) {
      vertices.push([Number(parts[1]), Number(parts[2]), Number(parts[3])]);
    } else if (parts[0] === 'f' && parts.length >= 4) {
      const indices = parts.slice(1).map(token => objIndex(token, vertices.length)).filter(index => index !== null);
      for (let i = 1; i + 1 < indices.length; i++) faces.push([indices[0], indices[i], indices[i + 1]]);
    }
  }
  return {vertices, faces};
}

function objIndex(token, count) {
  const raw = Number.parseInt(String(token).split('/')[0], 10);
  if (!Number.isFinite(raw) || raw === 0) return null;
  const index = raw > 0 ? raw - 1 : count + raw;
  return index >= 0 && index < count ? index : null;
}

function collectMeshPolygons(polygons, mesh, matrix, scaleValue, visualName, pointMapper) {
  const vertices = mesh.vertices || [];
  const faces = mesh.faces || [];
  if (!vertices.length || !faces.length) return;
  const scale = Array.isArray(scaleValue) ? scaleValue : [1, 1, 1];
  const stride = Math.max(1, Math.ceil(faces.length / 900));
  const isRing = String(visualName || '').toLowerCase().includes('ring');
  for (let i = 0; i < faces.length; i += stride) {
    const face = faces[i];
    const pts = [];
    for (const index of face) {
      const v = vertices[index];
      const p = pointMapper(matrixPointScaled(matrix, v, scale));
      if (!p) {
        pts.length = 0;
        break;
      }
      pts.push(p);
    }
    if (pts.length === 3) polygons.push({pts, ring: isRing});
  }
}

function renderMeshPolygons(polygons) {
  if (!polygons.length) return;
  const projected = [];
  for (const poly of polygons) {
    const pts = poly.pts.map(project);
    if (pts.some(p => !p.visible)) continue;
    const area = Math.abs(
      (pts[1].x - pts[0].x) * (pts[2].y - pts[0].y) -
      (pts[2].x - pts[0].x) * (pts[1].y - pts[0].y)
    );
    if (area < 0.18) continue;
    projected.push({
      pts,
      depth: (pts[0].z + pts[1].z + pts[2].z) / 3,
      ring: poly.ring,
      area
    });
  }
  projected.sort((a, b) => b.depth - a.depth);
  ctx.save();
  ctx.lineWidth = 0.45;
  for (const poly of projected) {
    ctx.beginPath();
    ctx.moveTo(poly.pts[0].x, poly.pts[0].y);
    ctx.lineTo(poly.pts[1].x, poly.pts[1].y);
    ctx.lineTo(poly.pts[2].x, poly.pts[2].y);
    ctx.closePath();
    ctx.fillStyle = poly.ring ? 'rgba(238,240,244,0.32)' : 'rgba(150,164,184,0.24)';
    ctx.strokeStyle = poly.ring ? 'rgba(255,255,255,0.15)' : 'rgba(210,220,235,0.08)';
    ctx.fill();
    if (poly.area > 8) ctx.stroke();
  }
  ctx.restore();
}

function nearestRobotSample(t) {
  const rows = robotPoseRows();
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

function nearestRobotMediaSample(t) {
  const rows = robotMediaRows();
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
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const screenY = cp * z - sp * y1;
  const depth = sp * z + cp * y1 + state.distance;
  const focal = Math.min(rect.width, rect.height) * 0.92;
  return {x: rect.width * 0.5 + x1 * focal / Math.max(0.03, depth), y: rect.height * 0.5 - screenY * focal / Math.max(0.03, depth), z: depth, visible: depth > 0.03};
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

function matrixPointScaled(m, v, scaleValue) {
  const scale = Array.isArray(scaleValue) ? scaleValue : [1, 1, 1];
  return matrixPoint(
    m,
    Number(v?.[0] || 0) * Number(scale[0] || 1),
    Number(v?.[1] || 0) * Number(scale[1] || 1),
    Number(v?.[2] || 0) * Number(scale[2] || 1)
  );
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

function replayBoardMatrix() {
  return state.data?.boardMatrix || identityMatrix4();
}

function robotPoseRows() {
  const rr = state.data?.robotRealSense;
  if (Array.isArray(rr?.robotStates) && rr.robotStates.length) return rr.robotStates;
  return Array.isArray(rr?.samples) ? rr.samples : [];
}

function robotMediaRows() {
  const rows = state.data?.robotRealSense?.samples;
  return Array.isArray(rows) ? rows : [];
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

function robotLinkFrames(model, jointpose, baseMatrix) {
  const frames = {base_link: baseMatrix};
  let jointIndex = 0;
  for (const joint of model.joints || []) {
    const parentFrame = frames[joint.parent] || baseMatrix;
    let childFrame = multiplyMatrix4(parentFrame, xyzRpyMatrix(joint.xyz, joint.rpy));
    if (joint.type !== 'fixed') {
      const angle = Number(jointpose[jointIndex++] || 0);
      childFrame = multiplyMatrix4(childFrame, axisAngleMatrix(joint.axis, angle));
    }
    if (joint.child) frames[joint.child] = childFrame;
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
window.addEventListener('resize', () => { resize(); render(); });

function initDetailResize(storageKey, minWidth, maxWidth, defaultWidth) {
  const saved = Number(localStorage.getItem(storageKey));
  setDetailWidth(Number.isFinite(saved) ? saved : defaultWidth, minWidth, maxWidth, storageKey);
  let active = false;
  detailResizer.addEventListener('pointerdown', event => {
    active = true;
    detailResizer.classList.add('dragging');
    detailResizer.setPointerCapture(event.pointerId);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    event.preventDefault();
  });
  detailResizer.addEventListener('pointermove', event => {
    if (!active) return;
    const width = window.innerWidth - event.clientX - 4;
    setDetailWidth(width, minWidth, maxWidth, storageKey);
    resize();
    render();
  });
  const stop = event => {
    if (!active) return;
    active = false;
    detailResizer.classList.remove('dragging');
    try { detailResizer.releasePointerCapture(event.pointerId); } catch (_) {}
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    resize();
    render();
  };
  detailResizer.addEventListener('pointerup', stop);
  detailResizer.addEventListener('pointercancel', stop);
}

function setDetailWidth(width, minWidth, maxWidth, storageKey) {
  const viewportLimit = Math.max(minWidth, window.innerWidth - 520);
  const clamped = Math.max(minWidth, Math.min(maxWidth, viewportLimit, Math.round(width)));
  app.style.setProperty('--detail-width', `${clamped}px`);
  localStorage.setItem(storageKey, String(clamped));
}

let lastFrame = performance.now();
function tick(now) {
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  const samples = state.data?.samples || [];
  if (state.playing && samples.length) {
    state.t += dt;
    const end = samples[samples.length - 1].recordingTimestampSeconds || 0;
    if (state.t > end) {
      state.t = end;
      state.idx = samples.length - 1;
      state.playing = false;
      playBtn.textContent = 'Play';
    }
    while (state.idx < samples.length - 1 && samples[state.idx + 1].recordingTimestampSeconds <= state.t) state.idx++;
    while (state.idx > 0 && samples[state.idx].recordingTimestampSeconds > state.t) state.idx--;
    scrub.value = state.idx;
    updateLabels();
  }
  render();
  requestAnimationFrame(tick);
}

initDetailResize('questReplayDetailWidth', 300, 760, 340);
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
