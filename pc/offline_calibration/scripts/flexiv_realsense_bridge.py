from __future__ import annotations

import copy
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np

try:
    sys.setswitchinterval(0.001)
except (AttributeError, ValueError):
    pass

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from quest_coordinate_frames import (
    PC_WORLD_FRAME,
    UNITY_WORLD_FRAME,
    WORLD_FRAME_CONVERSION,
    ensure_pc_transform_payload,
    matrix_from_transform_payload,
    unity_quaternion_wxyz_to_pc,
    unity_vec3_to_pc,
)
from checkerboard_orientation import (
    detect_checkerboard_appearance_anchor,
    draw_checkerboard_appearance_anchor_overlay,
)


DEFAULT_PATTERN_COLS = 11
DEFAULT_PATTERN_ROWS = 8
DEFAULT_SQUARE_SIZE_M = 0.025
DEFAULT_REALSENSE_WIDTH = 1280
DEFAULT_REALSENSE_HEIGHT = 720
DEFAULT_REALSENSE_FPS = 30
DEFAULT_ROBOT_STATE_HZ = 90.0
DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES = 3
DEFAULT_RECORD_DEPTH_FORMAT = "ffv1"
RECORD_DEPTH_FORMATS = {"ffv1", "raw"}
DEFAULT_FFMPEG_ENCODER_THREADS = 2
DEFAULT_FFMPEG_NICE_LEVEL = 10
ASYNC_JSONL_FLUSH_ROWS = 16
ASYNC_JSONL_FLUSH_INTERVAL_SECONDS = 0.25
ASYNC_QUEST_ALIGNED_SAMPLE_HZ = 30.0
ASYNC_CAMERA_QUEUE_SECONDS = 0.25
ASYNC_CAMERA_QUEUE_MIN_FRAMES = 4
ASYNC_CAMERA_QUEUE_MAX_FRAMES = 8
REALSENSE_CONNECT_WARMUP_SECONDS = 3.0
REALSENSE_RECORD_START_WAIT_SECONDS = 2.0
ROBOT_SESSION_THREAD_JOIN_SECONDS = 5.0
ROBOT_SESSION_CAMERA_WRITER_FINAL_JOIN_SECONDS = 1.0
ROBOT_SESSION_VIDEO_RELEASE_LOCK_SECONDS = 1.0
DEFAULT_CAPTURE_INTERVAL_SECONDS = 0.35
DEFAULT_FLEXIV_RDK_ROOT: Path | None = None
DEFAULT_FLEXIV_ROBOT_SN = "Rizon4-062713"
DEFAULT_END_CAMERA_SERIAL = "750612070265"
MIN_HAND_EYE_EE_TRANSLATION_SPAN_M = 0.02
MIN_HAND_EYE_EE_ROTATION_SPAN_DEG = 2.0
DEFAULT_CONTROLLER_TRANSLATION_SCALE = 1.0
CONTROLLER_TRANSLATION_SCALE_MIN = 0.1
CONTROLLER_TRANSLATION_SCALE_MAX = 3.0
DEFAULT_CONTROLLER_MAX_STEP_M = 0.04
DEFAULT_CONTROLLER_MAX_ROTATION_STEP_DEG = 4.0
DEFAULT_CONTROLLER_TARGET_UPDATE_HZ = 30.0
DEFAULT_CONTROLLER_JOINT_LIMIT_BUFFER_RAD = 0.04
DEFAULT_HAND_EYE_MAX_DIVERSE_SAMPLES = 120
DEFAULT_HAND_EYE_MIN_DIVERSE_SAMPLES = 20
DEFAULT_HAND_EYE_DIVERSE_TRANSLATION_SCALE_M = 0.02
DEFAULT_HAND_EYE_DIVERSE_ROTATION_SCALE_DEG = 3.0
DEFAULT_HAND_EYE_DIVERSE_MIN_SCORE = 0.75
DEFAULT_GRIPPER_DEVICE = "auto"
DEFAULT_GRIPPER_DEVICE_CANDIDATES = [
    "Robotiq",
    "Robotiq-2F-85",
    "Robotiq 2F-85",
    "Robotiq_2F_85",
    "robotiq",
    "robotiq_2f_85",
    "gripper",
]
DEFAULT_GRIPPER_OPEN_WIDTH_M = 0.08
DEFAULT_GRIPPER_CLOSE_WIDTH_M = 0.0
DEFAULT_GRIPPER_SPEED_MPS = 0.04
DEFAULT_GRIPPER_FORCE_N = 20.0
DEFAULT_GRIPPER_TRIGGER_CLOSE_THRESHOLD = 0.65
DEFAULT_GRIPPER_TRIGGER_OPEN_THRESHOLD = 0.25
DEFAULT_GRIPPER_INIT_ON_ENABLE = False
MAX_HAND_EYE_CAMERA_OFFSET_M = 0.50
MAX_HAND_EYE_TRANSLATION_MEDIAN_RESIDUAL_MM = 15.0
MAX_HAND_EYE_TRANSLATION_P95_RESIDUAL_MM = 35.0
APPEARANCE_ANCHOR_MIN_CONTRAST = 18.0
QUEST_TO_ROBOT_UNALIGNED_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
ROBOT_SESSION_CONTROL_TELEOP = "controller_teleop"
ROBOT_SESSION_CONTROL_FREEDRIVE = "freedrive"
ROBOT_SESSION_CONTROL_RECORD_ONLY = "record_only"
ROBOT_SESSION_CONTROL_MODES = {
    ROBOT_SESSION_CONTROL_TELEOP,
    ROBOT_SESSION_CONTROL_FREEDRIVE,
    ROBOT_SESSION_CONTROL_RECORD_ONLY,
}
ROBOT_SESSION_RECORD_SYNC = "sync"
ROBOT_SESSION_RECORD_ASYNC = "async"
ROBOT_SESSION_RECORD_MODES = {
    ROBOT_SESSION_RECORD_SYNC,
    ROBOT_SESSION_RECORD_ASYNC,
}
DEFAULT_FREEDRIVE_HOLD_STIFFNESS = [10000.0, 10000.0, 10000.0, 1500.0, 1500.0, 1500.0]
DEFAULT_FREEDRIVE_COMPLIANT_STIFFNESS = [1000.0, 1000.0, 1000.0, 8.0, 8.0, 8.0]
DEFAULT_FREEDRIVE_DAMPING = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
DEFAULT_FREEDRIVE_CONTROL_HZ = 60.0
DEFAULT_FREEDRIVE_MAX_LINEAR_VEL = 0.2
DEFAULT_FREEDRIVE_MAX_ANGULAR_VEL = 0.6
DEFAULT_FREEDRIVE_MAX_LINEAR_ACC = 0.8
DEFAULT_FREEDRIVE_MAX_ANGULAR_ACC = 2.0
ROBOT_CARTESIAN_SEND_SLOW_SECONDS = 0.05
ROBOT_CARTESIAN_SEND_FATAL_MARKERS = (
    "not in an applicable control mode",
    "safety error",
    "cat2",
    "fault",
)
ROBOT_CARTESIAN_TARGET_POSITION_EPS_M = 0.001
ROBOT_CARTESIAN_TARGET_ROTATION_EPS_DEG = 0.25
POSE_DIVERSITY_MAX_PAIRWISE_SAMPLES = 512
FREEDRIVE_FLOATING_CARTESIAN_PRIMITIVE = "FloatingCartesian()"


def positive_cartesian_limit(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


class RobotHandEyeCalibrationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        counts: dict[str, Any] | None = None,
        diversity: dict[str, Any] | None = None,
        observations: list[dict[str, Any]] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.counts = counts or {}
        self.diversity = diversity
        self.observations = observations
        self.diagnostics = diagnostics


class OnlineNumericStats:
    def __init__(self, *, max_recent: int = 4096) -> None:
        self.max_recent = max(32, int(max_recent))
        self.count = 0
        self.total = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None
        self.recent: list[float] = []

    def add(self, value: Any) -> None:
        if not is_number(value):
            return
        number = float(value)
        if not math.isfinite(number):
            return
        self.count += 1
        self.total += number
        self.min_value = number if self.min_value is None else min(self.min_value, number)
        self.max_value = number if self.max_value is None else max(self.max_value, number)
        self.recent.append(number)
        if len(self.recent) > self.max_recent:
            del self.recent[: len(self.recent) - self.max_recent]

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "count": int(self.count),
        }
        if self.count <= 0:
            return payload
        payload["mean"] = float(self.total / self.count)
        payload["min"] = self.min_value
        payload["max"] = self.max_value
        if self.recent:
            recent = np.asarray(self.recent, dtype=float)
            payload["recentWindowCount"] = int(recent.size)
            payload["median"] = float(np.median(recent))
            payload["p90"] = float(np.percentile(recent, 90))
            payload["p95"] = float(np.percentile(recent, 95))
            payload["p99"] = float(np.percentile(recent, 99))
        return payload


class OnlineTimeSeriesStats:
    def __init__(self, *, max_recent_gaps: int = 4096) -> None:
        self.count = 0
        self.first: float | None = None
        self.last: float | None = None
        self.gaps = OnlineNumericStats(max_recent=max_recent_gaps)

    def add(self, value: Any) -> None:
        if not is_number(value):
            return
        number = float(value)
        if not math.isfinite(number):
            return
        if self.first is None:
            self.first = number
        if self.last is not None:
            self.gaps.add(max(0.0, number - self.last))
        self.last = number
        self.count += 1

    def summary(self, target_hz: float | None = None) -> dict[str, Any]:
        if self.count <= 0 or self.first is None or self.last is None:
            return {"count": 0}
        duration = float(max(0.0, self.last - self.first))
        hz = float((self.count - 1) / duration) if duration > 1e-9 and self.count >= 2 else None
        payload: dict[str, Any] = {
            "count": int(self.count),
            "durationSeconds": duration,
            "effectiveHz": hz,
            "firstPerfCounterSeconds": self.first,
            "lastPerfCounterSeconds": self.last,
        }
        if target_hz is not None and target_hz > 0:
            payload["targetHz"] = float(target_hz)
            if hz is not None:
                payload["targetRatio"] = float(hz / float(target_hz))
        gap_summary = self.gaps.summary()
        if gap_summary.get("count"):
            payload["gapSeconds"] = gap_summary
        return payload


def time_series_rate_quality(
    summary: dict[str, Any],
    target_hz: float,
    *,
    min_target_ratio: float = 0.95,
    max_p95_gap_factor: float = 1.5,
) -> dict[str, Any]:
    effective_hz = summary.get("effectiveHz")
    gap_summary = summary.get("gapSeconds") if isinstance(summary.get("gapSeconds"), dict) else {}
    p95_gap = gap_summary.get("p95")
    expected_gap = 1.0 / max(1e-9, float(target_hz))
    min_hz = float(target_hz) * float(min_target_ratio)
    max_p95_gap = expected_gap * float(max_p95_gap_factor)
    enough_samples = int(summary.get("count") or 0) >= 2
    rate_ok = is_number(effective_hz) and float(effective_hz) >= min_hz
    gap_ok = is_number(p95_gap) and float(p95_gap) <= max_p95_gap
    return {
        "ok": bool(enough_samples and rate_ok and gap_ok),
        "targetHz": float(target_hz),
        "minimumEffectiveHz": min_hz,
        "maximumP95GapSeconds": max_p95_gap,
        "effectiveHz": effective_hz,
        "p95GapSeconds": p95_gap,
        "enoughSamples": enough_samples,
        "rateOk": bool(rate_ok),
        "gapOk": bool(gap_ok),
    }


@dataclass
class FlexivRealSenseConfig:
    robot_sn: str = DEFAULT_FLEXIV_ROBOT_SN
    robot_pose_field: str = "flange_pose"
    flexiv_rdk: Path | None = DEFAULT_FLEXIV_RDK_ROOT
    flexiv_network_interfaces: list[str] | None = None
    camera_serial: str = DEFAULT_END_CAMERA_SERIAL
    third_camera_serial: str = ""
    width: int = DEFAULT_REALSENSE_WIDTH
    height: int = DEFAULT_REALSENSE_HEIGHT
    fps: int = DEFAULT_REALSENSE_FPS
    robot_state_hz: float = DEFAULT_ROBOT_STATE_HZ
    record_depth: bool = True
    record_depth_align_to_color: bool = False
    record_depth_every_n_frames: int = DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES
    record_depth_format: str = DEFAULT_RECORD_DEPTH_FORMAT
    warmup_frames: int = 2
    realsense_auto_exposure: bool = True
    realsense_exposure: float | None = None
    realsense_gain: float | None = None
    board_check_warmup_frames: int = 60
    capture_interval_seconds: float = DEFAULT_CAPTURE_INTERVAL_SECONDS
    pattern_cols: int = DEFAULT_PATTERN_COLS
    pattern_rows: int = DEFAULT_PATTERN_ROWS
    square_size_m: float = DEFAULT_SQUARE_SIZE_M
    run_hand_eye: bool = True
    min_hand_eye_detections: int = 6
    hand_eye_max_diverse_samples: int = DEFAULT_HAND_EYE_MAX_DIVERSE_SAMPLES
    hand_eye_min_diverse_samples: int = DEFAULT_HAND_EYE_MIN_DIVERSE_SAMPLES
    hand_eye_diverse_translation_scale_m: float = DEFAULT_HAND_EYE_DIVERSE_TRANSLATION_SCALE_M
    hand_eye_diverse_rotation_scale_deg: float = DEFAULT_HAND_EYE_DIVERSE_ROTATION_SCALE_DEG
    hand_eye_diverse_min_score: float = DEFAULT_HAND_EYE_DIVERSE_MIN_SCORE
    hand_eye_disable_diverse_selection: bool = False
    controller_motion_enabled: bool = False
    controller_translation_scale: float = DEFAULT_CONTROLLER_TRANSLATION_SCALE
    controller_max_step_m: float = DEFAULT_CONTROLLER_MAX_STEP_M
    controller_max_rotation_step_deg: float = DEFAULT_CONTROLLER_MAX_ROTATION_STEP_DEG
    controller_target_update_hz: float = DEFAULT_CONTROLLER_TARGET_UPDATE_HZ
    cartesian_max_linear_velocity_mps: float = DEFAULT_FREEDRIVE_MAX_LINEAR_VEL
    cartesian_max_angular_velocity_radps: float = DEFAULT_FREEDRIVE_MAX_ANGULAR_VEL
    cartesian_max_linear_acceleration_mps2: float = DEFAULT_FREEDRIVE_MAX_LINEAR_ACC
    cartesian_max_angular_acceleration_radps2: float = DEFAULT_FREEDRIVE_MAX_ANGULAR_ACC
    controller_joint_limit_buffer_rad: float = DEFAULT_CONTROLLER_JOINT_LIMIT_BUFFER_RAD
    controller_joint_limit_guard_enabled: bool = True
    gripper_enabled: bool = False
    gripper_device: str = DEFAULT_GRIPPER_DEVICE
    gripper_open_width_m: float = DEFAULT_GRIPPER_OPEN_WIDTH_M
    gripper_close_width_m: float = DEFAULT_GRIPPER_CLOSE_WIDTH_M
    gripper_speed_mps: float = DEFAULT_GRIPPER_SPEED_MPS
    gripper_force_n: float = DEFAULT_GRIPPER_FORCE_N
    gripper_trigger_close_threshold: float = DEFAULT_GRIPPER_TRIGGER_CLOSE_THRESHOLD
    gripper_trigger_open_threshold: float = DEFAULT_GRIPPER_TRIGGER_OPEN_THRESHOLD
    gripper_init_on_enable: bool = DEFAULT_GRIPPER_INIT_ON_ENABLE

    def __post_init__(self) -> None:
        self.cartesian_max_linear_velocity_mps = positive_cartesian_limit(
            self.cartesian_max_linear_velocity_mps,
            "cartesian max linear velocity",
        )
        self.cartesian_max_angular_velocity_radps = positive_cartesian_limit(
            self.cartesian_max_angular_velocity_radps,
            "cartesian max angular velocity",
        )
        self.cartesian_max_linear_acceleration_mps2 = positive_cartesian_limit(
            self.cartesian_max_linear_acceleration_mps2,
            "cartesian max linear acceleration",
        )
        self.cartesian_max_angular_acceleration_radps2 = positive_cartesian_limit(
            self.cartesian_max_angular_acceleration_radps2,
            "cartesian max angular acceleration",
        )


class FlexivRobotClient:
    def __init__(
        self,
        *,
        cartesian_max_linear_velocity_mps: float = DEFAULT_FREEDRIVE_MAX_LINEAR_VEL,
        cartesian_max_angular_velocity_radps: float = DEFAULT_FREEDRIVE_MAX_ANGULAR_VEL,
        cartesian_max_linear_acceleration_mps2: float = DEFAULT_FREEDRIVE_MAX_LINEAR_ACC,
        cartesian_max_angular_acceleration_radps2: float = DEFAULT_FREEDRIVE_MAX_ANGULAR_ACC,
    ) -> None:
        self.robot: Any | None = None
        self.robot_sn: str | None = None
        self.pose_field = "flange_pose"
        self.lock = threading.RLock()
        self.last_error: str | None = None
        self.motion_armed = False
        self.motion_last_target_pose: list[float] | None = None
        self.freedrive_enabled = False
        self.freedrive_method: str | None = None
        self.freedrive_plan: str | None = None
        self.freedrive_stop_event: threading.Event | None = None
        self.freedrive_thread: threading.Thread | None = None
        self.freedrive_hold_pose: list[float] | None = None
        self.freedrive_last_tick_unix: float | None = None
        self.freedrive_last_error: str | None = None
        self.freedrive_send_signature: str | None = None
        self.cartesian_send_count = 0
        self.cartesian_send_error_count = 0
        self.cartesian_send_slow_count = 0
        self.cartesian_last_send: dict[str, Any] | None = None
        self.gripper: Any | None = None
        self.gripper_enabled = False
        self.gripper_device: str | None = None
        self.gripper_last_error: str | None = None
        self.gripper_enable_attempts: list[dict[str, Any]] = []
        self.device_list_cache: dict[str, bool] | None = None
        self.device_list_last_error: str | None = None
        self.joint_limits = default_rizon4_joint_limits()
        self.cartesian_max_linear_velocity_mps = positive_cartesian_limit(
            cartesian_max_linear_velocity_mps,
            "cartesian max linear velocity",
        )
        self.cartesian_max_angular_velocity_radps = positive_cartesian_limit(
            cartesian_max_angular_velocity_radps,
            "cartesian max angular velocity",
        )
        self.cartesian_max_linear_acceleration_mps2 = positive_cartesian_limit(
            cartesian_max_linear_acceleration_mps2,
            "cartesian max linear acceleration",
        )
        self.cartesian_max_angular_acceleration_radps2 = positive_cartesian_limit(
            cartesian_max_angular_acceleration_radps2,
            "cartesian max angular acceleration",
        )

    def connect(
        self,
        robot_sn: str,
        pose_field: str = "flange_pose",
        flexiv_rdk: Path | None = None,
        network_interfaces: list[str] | None = None,
        wait_seconds: float = 0.2,
    ) -> dict[str, Any]:
        robot_sn = str(robot_sn or "").strip()
        if not robot_sn:
            raise ValueError("robot serial number is empty")
        if pose_field not in ("flange_pose", "tcp_pose"):
            raise ValueError(f"unsupported pose field: {pose_field}")

        with self.lock:
            self.disconnect()
            flexivrdk = import_flexivrdk(flexiv_rdk)
            interface_whitelist = normalize_network_interfaces(network_interfaces)
            if interface_whitelist:
                self.robot = flexivrdk.Robot(robot_sn, interface_whitelist)
            else:
                self.robot = flexivrdk.Robot(robot_sn)
            self.robot_sn = robot_sn
            self.pose_field = pose_field
            self.last_error = None
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self.ensure_operational_locked("connecting Flexiv robot")
            current_tcp = [float(v) for v in self.robot.states().tcp_pose]
            self.freedrive_hold_pose = current_tcp
            self.motion_armed = False
            self.motion_last_target_pose = None
            self.freedrive_enabled = False
            self.freedrive_method = None
            self.freedrive_plan = None
            self.freedrive_last_tick_unix = None
            self.freedrive_last_error = None
            self.freedrive_send_signature = None
            self.cartesian_send_count = 0
            self.cartesian_send_error_count = 0
            self.cartesian_send_slow_count = 0
            self.cartesian_last_send = None
            self.device_list_cache = None
            self.device_list_last_error = None
            return self.read_state_locked()

    def disconnect(self) -> None:
        self.disarm_motion_locked()
        self.disable_freedrive_locked()
        self.stop_cartesian_control_loop_locked()
        self.disable_gripper_locked()
        if self.robot is not None:
            try:
                self.robot.Stop()
            except Exception:
                pass
        self.robot = None
        self.robot_sn = None
        self.device_list_cache = None

    def status(self, *, include_state: bool = True, include_devices: bool = True) -> dict[str, Any]:
        with self.lock:
            connected = self.robot is not None
            payload = {
                "connected": connected,
                "robotSn": self.robot_sn,
                "poseField": self.pose_field,
                "lastError": self.last_error,
                "motionArmed": self.motion_armed,
                "freedriveEnabled": self.freedrive_enabled,
                "freeDragEnabled": self.freedrive_enabled,
                "freedriveMethod": self.freedrive_method,
                "freeDragMethod": self.freedrive_method,
                "freedrivePlan": self.freedrive_plan,
                "freeDragPlan": self.freedrive_plan,
                "freedriveLoopAlive": self.freedrive_thread is not None and self.freedrive_thread.is_alive(),
                "freeDragLoopAlive": self.freedrive_thread is not None and self.freedrive_thread.is_alive(),
                "cartesianControlLoopAlive": self.freedrive_thread is not None and self.freedrive_thread.is_alive(),
                "freedriveLastTickUnix": self.freedrive_last_tick_unix,
                "freeDragLastTickUnix": self.freedrive_last_tick_unix,
                "freedriveLastError": self.freedrive_last_error,
                "freeDragLastError": self.freedrive_last_error,
                "freedriveControlHz": DEFAULT_FREEDRIVE_CONTROL_HZ,
                "freeDragControlHz": DEFAULT_FREEDRIVE_CONTROL_HZ,
                "cartesianMotionLimits": self.cartesian_motion_limits_locked(),
                "cartesianSendSignature": self.freedrive_send_signature,
                "cartesianSend": self.cartesian_send_status_locked(),
                "gripper": self.gripper_status_locked(
                    include_params=include_state,
                    include_states=include_state,
                ),
                "devices": self.device_status_locked()
                if include_devices
                else {
                    "ok": self.device_list_cache is not None,
                    "list": self.device_list_cache,
                    "lastError": self.device_list_last_error,
                    "skipped": True,
                },
            }
            if connected and include_state:
                try:
                    payload["state"] = self.read_state_locked()
                except Exception as exc:  # pragma: no cover - hardware path
                    self.last_error = str(exc)
                    payload["lastError"] = self.last_error
            return payload

    def read_state(self, *, include_diagnostics: bool = True) -> dict[str, Any]:
        with self.lock:
            return self.read_state_locked(include_diagnostics=include_diagnostics)

    def read_state_locked(self, *, include_diagnostics: bool = True) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        states = self.robot.states()
        pose = np.asarray(getattr(states, self.pose_field), dtype=float).reshape(7)
        t_base_ee = flexiv_pose_to_transform(pose)
        payload: dict[str, Any] = {
            "ok": True,
            "robotSn": self.robot_sn,
            "poseField": self.pose_field,
            "readUnixSeconds": time.time(),
            "endEffectorPose": transform_to_json(t_base_ee),
            "rawEndEffectorPoseWxyz": [float(v) for v in pose],
            "jointPose": first_numeric_state_list(
                states,
                [
                    "q",
                    "theta",
                    "joint_pos",
                    "joint_position",
                    "joint_positions",
                    "jointPosition",
                    "actual_q",
                ],
            ),
        }
        if include_diagnostics:
            for name, getter in (
                ("mode", lambda: self.robot.mode()),
                ("operationalStatus", lambda: self.robot.operational_status()),
                ("enablingButtonPressed", lambda: self.robot.enabling_button_pressed()),
                ("busy", lambda: self.robot.busy()),
                ("stopped", lambda: self.robot.stopped()),
                ("reduced", lambda: self.robot.reduced()),
            ):
                try:
                    value = getter()
                    payload[name] = getattr(value, "name", str(value)) if name in ("mode", "operationalStatus") else bool(value)
                except Exception:
                    pass
        payload["jointLimitGuard"] = joint_limit_guard_state(
            payload.get("jointPose"),
            self.joint_limits,
            0.0,
            True,
        )
        for field in ("flange_pose", "tcp_pose"):
            if field == self.pose_field or not hasattr(states, field):
                continue
            try:
                other_pose = np.asarray(getattr(states, field), dtype=float).reshape(7)
                payload[field] = {
                    "pose": [float(v) for v in other_pose],
                    "T_base_pose": transform_to_json(flexiv_pose_to_transform(other_pose)),
                }
            except Exception:
                pass
        return payload

    def read_record_state(
        self,
        joint_limit_buffer_rad: float,
        joint_limit_guard_enabled: bool,
        *,
        lightweight: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            if lightweight:
                if self.robot is None:
                    raise RuntimeError("Flexiv robot is not connected")
                states = self.robot.states()
                pose = np.asarray(getattr(states, self.pose_field), dtype=float).reshape(7)
                payload: dict[str, Any] = {
                    "ok": True,
                    "robotSn": self.robot_sn,
                    "poseField": self.pose_field,
                    "readUnixSeconds": time.time(),
                    "rawEndEffectorPoseWxyz": [float(v) for v in pose],
                    "jointPose": first_numeric_state_list(
                        states,
                        [
                            "q",
                            "theta",
                            "joint_pos",
                            "joint_position",
                            "joint_positions",
                            "jointPosition",
                            "actual_q",
                        ],
                    ),
                }
                for field in ("flange_pose", "tcp_pose"):
                    if field == self.pose_field or not hasattr(states, field):
                        continue
                    try:
                        other_pose = np.asarray(getattr(states, field), dtype=float).reshape(7)
                        payload[field] = {"pose": [float(v) for v in other_pose]}
                    except Exception:
                        pass
            else:
                payload = self.read_state_locked(include_diagnostics=False)
            payload["jointLimitGuard"] = joint_limit_guard_state(
                payload.get("jointPose"),
                self.joint_limits,
                joint_limit_buffer_rad,
                joint_limit_guard_enabled,
            )
            if lightweight:
                payload["gripper"] = {
                    "enabled": self.gripper_enabled,
                    "device": self.gripper_device,
                    "lastError": self.gripper_last_error,
                }
            else:
                payload["gripper"] = self.gripper_status_locked(include_params=False)
            return payload

    def read_tcp_pose(self) -> list[float]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            return [float(v) for v in self.robot.states().tcp_pose]

    def arm_motion(self) -> dict[str, Any]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            if (
                self.motion_armed
                and not self.freedrive_enabled
                and self.mode_name_locked() == "NRT_CARTESIAN_MOTION_FORCE"
            ):
                return self.read_state_locked()
            self.disable_freedrive_locked()
            self.stop_cartesian_control_loop_locked()
            robot = self.robot
            self.ensure_operational_locked("arming controller motion")
            self.switch_mode_locked("NRT_CARTESIAN_MOTION_FORCE")
            robot.SetForceControlAxis([False, False, False, False, False, False])
            self.set_cartesian_impedance_locked(
                DEFAULT_FREEDRIVE_HOLD_STIFFNESS,
                DEFAULT_FREEDRIVE_DAMPING,
            )
            self.motion_armed = True
            self.motion_last_target_pose = [float(v) for v in robot.states().tcp_pose]
            self.freedrive_hold_pose = list(self.motion_last_target_pose)
            return self.read_state_locked()

    def enable_freedrive(self) -> dict[str, Any]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            self.disarm_motion_locked()
            self.disable_freedrive_locked()
            try:
                return self.enable_floating_cartesian_freedrive_locked()
            except Exception as exc:
                self.freedrive_last_error = f"FloatingCartesian fallback: {exc}"
                self.last_error = self.freedrive_last_error
            robot = self.robot
            self.ensure_operational_locked("enabling free-drag mode")
            self.switch_mode_locked("NRT_CARTESIAN_MOTION_FORCE")
            current_tcp = [float(v) for v in robot.states().tcp_pose]
            robot.SetForceControlAxis([False, False, False, False, False, False])
            self.set_cartesian_impedance_locked(
                DEFAULT_FREEDRIVE_COMPLIANT_STIFFNESS,
                DEFAULT_FREEDRIVE_DAMPING,
            )
            self.freedrive_enabled = True
            self.freedrive_method = "cartesian_compliance"
            self.freedrive_plan = None
            self.freedrive_hold_pose = current_tcp
            self.freedrive_last_tick_unix = None
            self.motion_armed = False
            self.motion_last_target_pose = None
            self.start_cartesian_control_loop_locked()
            return self.read_state_locked()

    def enable_floating_cartesian_freedrive_locked(self) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        execute_primitive = getattr(self.robot, "ExecutePrimitive", None)
        if not callable(execute_primitive):
            raise RuntimeError("Flexiv RDK Robot does not expose ExecutePrimitive")
        self.stop_cartesian_control_loop_locked()
        self.switch_mode_locked("NRT_PRIMITIVE_EXECUTION")
        execute_primitive(FREEDRIVE_FLOATING_CARTESIAN_PRIMITIVE)
        self.freedrive_enabled = True
        self.freedrive_method = "floating_cartesian_primitive"
        self.freedrive_plan = FREEDRIVE_FLOATING_CARTESIAN_PRIMITIVE
        self.freedrive_hold_pose = None
        self.freedrive_last_tick_unix = time.time()
        self.freedrive_last_error = None
        self.motion_armed = False
        self.motion_last_target_pose = None
        return self.read_state_locked()

    def disable_freedrive(self) -> dict[str, Any]:
        with self.lock:
            self.disable_freedrive_locked()
            return self.status_unlocked()

    def disable_freedrive_locked(self) -> None:
        if self.robot is not None and (self.freedrive_enabled or self.freedrive_method is not None):
            try:
                if self.freedrive_method == "floating_cartesian_primitive":
                    try:
                        self.robot.Stop()
                        time.sleep(0.1)
                    except Exception:
                        pass
                current_tcp = [float(v) for v in self.robot.states().tcp_pose]
                self.freedrive_hold_pose = current_tcp
                self.switch_mode_locked("NRT_CARTESIAN_MOTION_FORCE")
                self.robot.SetForceControlAxis([False, False, False, False, False, False])
                self.set_cartesian_impedance_locked(
                    DEFAULT_FREEDRIVE_HOLD_STIFFNESS,
                    DEFAULT_FREEDRIVE_DAMPING,
                )
                self.send_cartesian_motion_force_compat(self.robot, current_tcp)
                self.start_cartesian_control_loop_locked()
            except Exception as exc:
                self.freedrive_last_error = str(exc)
                self.last_error = str(exc)
        self.freedrive_enabled = False
        self.freedrive_method = None
        self.freedrive_plan = None

    def disarm_motion(self) -> dict[str, Any]:
        with self.lock:
            self.disarm_motion_locked()
            return self.status_unlocked()

    def disarm_motion_locked(self) -> None:
        if self.robot is not None and self.motion_armed:
            try:
                current_tcp = [float(v) for v in self.robot.states().tcp_pose]
                self.freedrive_hold_pose = current_tcp
                self.switch_mode_locked("NRT_CARTESIAN_MOTION_FORCE")
                self.robot.SetForceControlAxis([False, False, False, False, False, False])
                self.set_cartesian_impedance_locked(
                    DEFAULT_FREEDRIVE_HOLD_STIFFNESS,
                    DEFAULT_FREEDRIVE_DAMPING,
                )
                self.send_cartesian_motion_force_compat(self.robot, current_tcp)
            except Exception as exc:
                self.last_error = str(exc)
        self.motion_armed = False
        self.motion_last_target_pose = None

    def status_unlocked(self) -> dict[str, Any]:
        connected = self.robot is not None
        payload: dict[str, Any] = {
            "connected": connected,
            "robotSn": self.robot_sn,
            "poseField": self.pose_field,
            "lastError": self.last_error,
            "motionArmed": self.motion_armed,
            "freedriveEnabled": self.freedrive_enabled,
            "freeDragEnabled": self.freedrive_enabled,
            "freedriveMethod": self.freedrive_method,
            "freeDragMethod": self.freedrive_method,
            "freedrivePlan": self.freedrive_plan,
            "freeDragPlan": self.freedrive_plan,
            "freedriveLoopAlive": self.freedrive_thread is not None and self.freedrive_thread.is_alive(),
            "freeDragLoopAlive": self.freedrive_thread is not None and self.freedrive_thread.is_alive(),
            "cartesianControlLoopAlive": self.freedrive_thread is not None and self.freedrive_thread.is_alive(),
            "freedriveLastTickUnix": self.freedrive_last_tick_unix,
            "freeDragLastTickUnix": self.freedrive_last_tick_unix,
            "freedriveLastError": self.freedrive_last_error,
            "freeDragLastError": self.freedrive_last_error,
            "freedriveControlHz": DEFAULT_FREEDRIVE_CONTROL_HZ,
            "freeDragControlHz": DEFAULT_FREEDRIVE_CONTROL_HZ,
            "cartesianMotionLimits": self.cartesian_motion_limits_locked(),
            "cartesianSendSignature": self.freedrive_send_signature,
            "cartesianSend": self.cartesian_send_status_locked(),
            "gripper": self.gripper_status_locked(),
            "devices": self.device_status_locked(),
        }
        if connected:
            try:
                payload["state"] = self.read_state_locked()
            except Exception as exc:
                self.last_error = str(exc)
                payload["lastError"] = self.last_error
        return payload

    def send_cartesian_target(
        self,
        target_pose_wxyz: list[float],
        joint_limit_buffer_rad: float,
        joint_limit_guard_enabled: bool,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            if not self.motion_armed:
                return {"enabled": bool(joint_limit_guard_enabled), "ok": False, "reason": "motion_not_armed"}
            if len(target_pose_wxyz) < 7:
                raise ValueError("target pose must be [x,y,z,qw,qx,qy,qz]")
            target = [float(v) for v in target_pose_wxyz[:7]]
            if not all(math.isfinite(v) for v in target):
                self.motion_armed = False
                self.motion_last_target_pose = None
                return {
                    "enabled": bool(joint_limit_guard_enabled),
                    "ok": False,
                    "reason": "target_not_finite",
                    "target": target,
                    "commandSent": False,
                }
            mode_name = self.mode_name_locked()
            if mode_name != "NRT_CARTESIAN_MOTION_FORCE":
                self.motion_armed = False
                self.motion_last_target_pose = None
                return {
                    "enabled": bool(joint_limit_guard_enabled),
                    "ok": False,
                    "reason": "cartesian_motion_mode_not_active",
                    "modeName": mode_name,
                    "commandSent": False,
                }
            joint_pose = self.read_joint_pose_locked()
            guard = joint_limit_guard_state(
                joint_pose,
                self.joint_limits,
                joint_limit_buffer_rad,
                joint_limit_guard_enabled,
            )
            if not bool(guard.get("ok")):
                return guard
            last_target = self.motion_last_target_pose
            if last_target is not None and cartesian_target_near(
                target,
                last_target,
                ROBOT_CARTESIAN_TARGET_POSITION_EPS_M,
                ROBOT_CARTESIAN_TARGET_ROTATION_EPS_DEG,
            ):
                guard["targetQueued"] = False
                guard["commandSent"] = False
                guard["reason"] = "target_unchanged"
                guard["positionEpsilonM"] = ROBOT_CARTESIAN_TARGET_POSITION_EPS_M
                guard["rotationEpsilonDeg"] = ROBOT_CARTESIAN_TARGET_ROTATION_EPS_DEG
                guard["targetUpdateDurationSeconds"] = time.perf_counter() - started
                return guard
            self.send_cartesian_motion_force_compat(self.robot, target)
            self.motion_last_target_pose = target
            self.freedrive_hold_pose = target
            last_send = self.cartesian_last_send if isinstance(self.cartesian_last_send, dict) else {}
            guard["targetQueued"] = False
            guard["commandSent"] = True
            guard["sendDurationSeconds"] = last_send.get("durationSeconds")
            guard["sendSlow"] = last_send.get("slow")
            guard["sendSignature"] = last_send.get("signature")
            guard["targetUpdateDurationSeconds"] = time.perf_counter() - started
            return guard

    def start_cartesian_control_loop_locked(self) -> None:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        if self.freedrive_thread is not None and self.freedrive_thread.is_alive():
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self.freedrive_control_loop,
            args=(self.robot, stop_event),
            name="flexiv-cartesian-control-loop",
            daemon=True,
        )
        self.freedrive_stop_event = stop_event
        self.freedrive_thread = thread
        thread.start()

    def stop_cartesian_control_loop_locked(self) -> None:
        stop_event = self.freedrive_stop_event
        thread = self.freedrive_thread
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self.freedrive_stop_event = None
        self.freedrive_thread = None

    def freedrive_control_loop(self, robot: Any, stop_event: threading.Event) -> None:
        period = 1.0 / max(1.0, DEFAULT_FREEDRIVE_CONTROL_HZ)
        while not stop_event.is_set():
            start = time.perf_counter()
            try:
                with self.lock:
                    if self.motion_armed:
                        target = self.motion_last_target_pose or self.freedrive_hold_pose
                        if target is None:
                            target = [float(v) for v in robot.states().tcp_pose]
                            self.freedrive_hold_pose = target
                    elif self.freedrive_enabled:
                        current_tcp = [float(v) for v in robot.states().tcp_pose]
                        target = current_tcp
                        self.freedrive_hold_pose = current_tcp
                    else:
                        if self.freedrive_hold_pose is None:
                            current_tcp = [float(v) for v in robot.states().tcp_pose]
                            self.freedrive_hold_pose = current_tcp
                        target = self.freedrive_hold_pose
                    self.send_cartesian_motion_force_compat(robot, target)
                    self.freedrive_last_tick_unix = time.time()
                    self.freedrive_last_error = None
            except Exception as exc:  # pragma: no cover - hardware path
                with self.lock:
                    self.freedrive_last_error = str(exc)
                    self.last_error = str(exc)
                    if self.cartesian_error_is_fatal(exc):
                        self.motion_armed = False
                        self.motion_last_target_pose = None
                        self.freedrive_hold_pose = None
                        stop_event.set()
            elapsed = time.perf_counter() - start
            stop_event.wait(max(0.0, period - elapsed))

    def ensure_operational_locked(self, action: str, timeout_s: float = 5.0) -> None:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        try:
            if self.robot.fault() or self.robot.stopped():
                clear_fault = getattr(self.robot, "ClearFault", None)
                if callable(clear_fault):
                    clear_fault()
                    time.sleep(0.2)
        except Exception:
            pass
        if self.robot.fault():
            raise RuntimeError(f"Flexiv robot has fault; clear it before {action}")
        self.robot.Enable()
        start = time.time()
        while not self.robot.operational():
            if time.time() - start > timeout_s:
                details: list[str] = []
                for name, getter in (
                    ("mode", self.robot.mode),
                    ("operationalStatus", self.robot.operational_status),
                    ("stopped", self.robot.stopped),
                    ("busy", self.robot.busy),
                    ("enablingButtonPressed", self.robot.enabling_button_pressed),
                ):
                    try:
                        value = getter()
                        details.append(f"{name}={getattr(value, 'name', value)}")
                    except Exception:
                        pass
                suffix = f" ({', '.join(details)})" if details else ""
                raise RuntimeError(f"Timed out waiting for Flexiv robot to become operational while {action}{suffix}")
            time.sleep(0.05)

    def mode_name_locked(self) -> str:
        if self.robot is None:
            return ""
        value = self.robot.mode()
        return str(getattr(value, "name", value)).split(".")[-1]

    def switch_mode_locked(self, mode_name: str, timeout_s: float = 2.0) -> None:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        if self.mode_name_locked() == mode_name:
            return
        flexivrdk = sys.modules.get("flexivrdk") or import_flexivrdk(None)
        mode = getattr(flexivrdk.Mode, mode_name, None)
        if mode is None:
            raise RuntimeError(f"Flexiv RDK does not expose mode {mode_name}")
        self.robot.SwitchMode(mode)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.mode_name_locked() == mode_name:
                return
            time.sleep(0.05)
        raise RuntimeError(f"Timed out switching Flexiv mode to {mode_name}; current mode is {self.mode_name_locked()}")

    def set_cartesian_impedance_locked(self, stiffness: list[float], damping: list[float]) -> None:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        try:
            self.robot.SetCartesianImpedance(list(stiffness), list(damping))
        except TypeError:
            self.robot.SetCartesianImpedance(list(stiffness))

    def configure_cartesian_motion_limits(
        self,
        *,
        max_linear_velocity_mps: float,
        max_angular_velocity_radps: float,
        max_linear_acceleration_mps2: float,
        max_angular_acceleration_radps2: float,
    ) -> dict[str, float]:
        values = {
            "maxLinearVelocityMps": positive_cartesian_limit(
                max_linear_velocity_mps,
                "cartesian max linear velocity",
            ),
            "maxAngularVelocityRadps": positive_cartesian_limit(
                max_angular_velocity_radps,
                "cartesian max angular velocity",
            ),
            "maxLinearAccelerationMps2": positive_cartesian_limit(
                max_linear_acceleration_mps2,
                "cartesian max linear acceleration",
            ),
            "maxAngularAccelerationRadps2": positive_cartesian_limit(
                max_angular_acceleration_radps2,
                "cartesian max angular acceleration",
            ),
        }
        with self.lock:
            self.cartesian_max_linear_velocity_mps = values["maxLinearVelocityMps"]
            self.cartesian_max_angular_velocity_radps = values["maxAngularVelocityRadps"]
            self.cartesian_max_linear_acceleration_mps2 = values["maxLinearAccelerationMps2"]
            self.cartesian_max_angular_acceleration_radps2 = values["maxAngularAccelerationRadps2"]
            return self.cartesian_motion_limits_locked()

    def cartesian_motion_limits_locked(self) -> dict[str, float]:
        return {
            "maxLinearVelocityMps": float(self.cartesian_max_linear_velocity_mps),
            "maxAngularVelocityRadps": float(self.cartesian_max_angular_velocity_radps),
            "maxLinearAccelerationMps2": float(self.cartesian_max_linear_acceleration_mps2),
            "maxAngularAccelerationRadps2": float(self.cartesian_max_angular_acceleration_radps2),
        }

    def cartesian_send_status_locked(self) -> dict[str, Any]:
        return {
            "count": self.cartesian_send_count,
            "errors": self.cartesian_send_error_count,
            "slowCalls": self.cartesian_send_slow_count,
            "last": self.cartesian_last_send,
            "slowThresholdSeconds": ROBOT_CARTESIAN_SEND_SLOW_SECONDS,
        }

    def cartesian_send_status(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.cartesian_send_status_locked())

    @staticmethod
    def cartesian_error_is_fatal(error: Exception | str) -> bool:
        message = str(error).lower()
        return any(marker in message for marker in ROBOT_CARTESIAN_SEND_FATAL_MARKERS)

    def send_cartesian_motion_force_compat(self, robot: Any, pose: list[float]) -> None:
        target = [float(v) for v in pose[:7]]
        zero6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with self.lock:
            limits = self.cartesian_motion_limits_locked()
        started = time.perf_counter()
        signature: str | None = None
        error: Exception | None = None
        fallback_errors: list[str] = []
        try:
            robot.SendCartesianMotionForce(
                target,
                zero6,
                limits["maxLinearVelocityMps"],
                limits["maxAngularVelocityRadps"],
                limits["maxLinearAccelerationMps2"],
                limits["maxAngularAccelerationRadps2"],
            )
            signature = "pose+wrench+limits"
        except TypeError as exc:
            fallback_errors.append(f"limits: {exc}")
            try:
                robot.SendCartesianMotionForce(target, zero6)
                signature = "pose+wrench"
            except TypeError as exc:
                fallback_errors.append(f"wrench: {exc}")
                try:
                    robot.SendCartesianMotionForce(target)
                    signature = "pose"
                except Exception as exc:
                    error = exc
            except Exception as exc:
                error = exc
        except Exception as exc:
            error = exc
        duration = time.perf_counter() - started
        with self.lock:
            self.cartesian_send_count += 1
            if duration >= ROBOT_CARTESIAN_SEND_SLOW_SECONDS:
                self.cartesian_send_slow_count += 1
            if error is not None:
                self.cartesian_send_error_count += 1
            if signature is not None:
                self.freedrive_send_signature = signature
            self.cartesian_last_send = {
                "ok": error is None,
                "durationSeconds": duration,
                "slow": duration >= ROBOT_CARTESIAN_SEND_SLOW_SECONDS,
                "signature": signature or self.freedrive_send_signature,
                "error": str(error) if error is not None else None,
                "fallbackErrors": fallback_errors[-3:],
                "limits": limits,
                "targetPoseWxyz": target,
                "unixSeconds": time.time(),
            }
        if error is not None:
            raise error

    def joint_limit_guard(self, buffer_rad: float, enabled: bool) -> dict[str, Any]:
        with self.lock:
            if self.robot is None:
                return {"enabled": bool(enabled), "ok": False, "reason": "robot_not_connected"}
            joint_pose = self.read_joint_pose_locked()
            return joint_limit_guard_state(joint_pose, self.joint_limits, buffer_rad, enabled)

    def read_joint_pose_locked(self) -> list[float] | None:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        states = self.robot.states()
        return first_numeric_state_list(
            states,
            [
                "q",
                "theta",
                "joint_pos",
                "joint_position",
                "joint_positions",
                "jointPosition",
                "actual_q",
            ],
        )

    def enable_gripper(self, device_name: str, *, init_on_enable: bool = DEFAULT_GRIPPER_INIT_ON_ENABLE) -> dict[str, Any]:
        with self.lock:
            return self.enable_gripper_locked(device_name, init_on_enable=init_on_enable)

    def device_status_locked(self, *, refresh: bool = False) -> dict[str, Any]:
        if self.robot is None:
            return {
                "ok": False,
                "reason": "robot_not_connected",
                "list": self.device_list_cache,
                "lastError": self.device_list_last_error,
            }
        if self.device_list_cache is not None and not refresh:
            return {"ok": True, "list": self.device_list_cache, "lastError": self.device_list_last_error}
        try:
            flexivrdk = sys.modules.get("flexivrdk") or import_flexivrdk(None)
            device = flexivrdk.Device(self.robot)
            items = device.list()
            self.device_list_cache = {str(name): bool(enabled) for name, enabled in dict(items).items()}
            self.device_list_last_error = None
            return {"ok": True, "list": self.device_list_cache, "lastError": None}
        except Exception as exc:
            self.device_list_last_error = str(exc)
            return {
                "ok": False,
                "list": self.device_list_cache,
                "lastError": self.device_list_last_error,
            }

    def gripper_device_candidates_locked(self, requested: str | None) -> list[str]:
        requested_text = str(requested or "").strip()
        candidates: list[str] = []
        if requested_text and requested_text.lower() not in ("auto", "default"):
            candidates.append(requested_text)
        device_status = self.device_status_locked(refresh=True)
        device_list = device_status.get("list") if isinstance(device_status, dict) else None
        if isinstance(device_list, dict):
            names = [str(name) for name in device_list.keys()]
            online_names = [name for name in names if bool(device_list.get(name))]
            offline_names = [name for name in names if not bool(device_list.get(name))]

            def is_robotiq(name: str) -> bool:
                lower = name.lower()
                return any(token in lower for token in ("robotiq", "2f", "hand-e", "hande"))

            def is_gripperish(name: str) -> bool:
                lower = name.lower()
                return is_robotiq(name) or "gripper" in lower

            prioritized = [
                *[name for name in online_names if is_robotiq(name)],
                *[name for name in offline_names if is_robotiq(name)],
                *[name for name in online_names if is_gripperish(name) and not is_robotiq(name)],
                *[name for name in offline_names if is_gripperish(name) and not is_robotiq(name)],
            ]
            for name in prioritized:
                if name not in candidates:
                    candidates.append(name)
        for name in DEFAULT_GRIPPER_DEVICE_CANDIDATES:
            if name not in candidates:
                candidates.append(name)
        return candidates

    def enable_gripper_locked(
        self,
        device_name: str,
        *,
        init_on_enable: bool = DEFAULT_GRIPPER_INIT_ON_ENABLE,
    ) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        flexivrdk = sys.modules.get("flexivrdk") or import_flexivrdk(None)
        requested = str(device_name or DEFAULT_GRIPPER_DEVICE).strip() or DEFAULT_GRIPPER_DEVICE
        if self.gripper is not None and self.gripper_enabled:
            current = str(self.gripper_device or "").strip()
            if requested.lower() in ("auto", "default") or not current or current == requested:
                status = self.gripper_status_locked()
                status["reusedExisting"] = True
                return status
        attempts: list[dict[str, Any]] = []
        last_error: str | None = None
        for device in self.gripper_device_candidates_locked(requested):
            gripper = flexivrdk.Gripper(self.robot)
            try:
                gripper.Enable(device)
                init_error: str | None = None
                if init_on_enable:
                    try:
                        gripper.Init()
                    except Exception as exc:
                        # Some Robotiq/Flexiv gripper configurations initialize automatically on power-on.
                        init_error = str(exc)
                self.gripper = gripper
                self.gripper_enabled = True
                self.gripper_device = device
                self.gripper_last_error = None
                attempts.append({"device": device, "ok": True, "initError": init_error})
                self.gripper_enable_attempts = attempts
                return self.gripper_status_locked(include_params=True, include_states=True)
            except Exception as exc:
                last_error = str(exc)
                attempts.append({"device": device, "ok": False, "error": last_error})
                if "already enabled" in last_error.lower():
                    self.gripper = gripper
                    self.gripper_enabled = True
                    self.gripper_device = device
                    self.gripper_last_error = None
                    attempts[-1]["ok"] = True
                    attempts[-1]["reusedAlreadyEnabled"] = True
                    self.gripper_enable_attempts = attempts
                    return self.gripper_status_locked(include_params=True, include_states=True)
        self.gripper = None
        self.gripper_enabled = False
        self.gripper_device = None
        self.gripper_last_error = last_error or "no gripper device candidates were available"
        self.gripper_enable_attempts = attempts
        tried = ", ".join(row.get("device", "") for row in attempts) or requested
        raise RuntimeError(f"Could not enable Flexiv gripper; tried [{tried}]. Last error: {self.gripper_last_error}")

    def disable_gripper_locked(self) -> None:
        if self.gripper is not None:
            try:
                self.gripper.Stop()
            except Exception:
                pass
            try:
                self.gripper.Disable()
            except Exception:
                pass
        self.gripper = None
        self.gripper_enabled = False
        self.gripper_device = None

    def gripper_status_locked(
        self,
        *,
        include_params: bool = False,
        include_states: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.gripper_enabled,
            "device": self.gripper_device,
            "lastError": self.gripper_last_error,
            "enableAttempts": self.gripper_enable_attempts[-12:],
        }
        if self.gripper is not None and include_states:
            try:
                states = self.gripper.states()
                payload["states"] = {
                    "width": safe_float(getattr(states, "width", None)),
                    "force": safe_float(getattr(states, "force", None)),
                    "isMoving": bool(getattr(states, "is_moving", False)),
                }
            except Exception as exc:
                self.gripper_last_error = str(exc)
                payload["lastError"] = self.gripper_last_error
            if include_params:
                try:
                    params = self.gripper.params()
                    payload["params"] = {
                        "name": str(getattr(params, "name", "")),
                        "minWidth": safe_float(getattr(params, "min_width", None)),
                        "maxWidth": safe_float(getattr(params, "max_width", None)),
                        "minVel": safe_float(getattr(params, "min_vel", None)),
                        "maxVel": safe_float(getattr(params, "max_vel", None)),
                        "minForce": safe_float(getattr(params, "min_force", None)),
                        "maxForce": safe_float(getattr(params, "max_force", None)),
                    }
                except Exception:
                    pass
        return payload

    def gripper_status(
        self,
        *,
        include_params: bool = False,
        include_states: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            return self.gripper_status_locked(include_params=include_params, include_states=include_states)

    def move_gripper(self, width_m: float, speed_mps: float, force_n: float) -> dict[str, Any]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            if self.gripper is None:
                self.enable_gripper_locked(
                    str(self.gripper_device or DEFAULT_GRIPPER_DEVICE),
                    init_on_enable=DEFAULT_GRIPPER_INIT_ON_ENABLE,
                )
            assert self.gripper is not None
            params_payload: dict[str, Any] = {}
            try:
                params = self.gripper.params()
                params_payload = {
                    "minWidth": safe_float(getattr(params, "min_width", None)),
                    "maxWidth": safe_float(getattr(params, "max_width", None)),
                    "minVel": safe_float(getattr(params, "min_vel", None)),
                    "maxVel": safe_float(getattr(params, "max_vel", None)),
                    "minForce": safe_float(getattr(params, "min_force", None)),
                    "maxForce": safe_float(getattr(params, "max_force", None)),
                }
            except Exception:
                params_payload = {}
            requested = {
                "width": float(width_m),
                "speed": float(speed_mps),
                "force": float(force_n),
            }
            width = clamp_optional_range(requested["width"], params_payload.get("minWidth"), params_payload.get("maxWidth"))
            speed = clamp_optional_range(requested["speed"], params_payload.get("minVel"), params_payload.get("maxVel"))
            force = clamp_optional_range(requested["force"], params_payload.get("minForce"), params_payload.get("maxForce"))
            self.gripper.Move(width, speed, force)
            self.gripper_last_error = None
            status = self.gripper_status_locked(include_states=True)
            status["command"] = {
                "requestedWidth": requested["width"],
                "requestedSpeed": requested["speed"],
                "requestedForce": requested["force"],
                "width": width,
                "speed": speed,
                "force": force,
                "clamped": width != requested["width"] or speed != requested["speed"] or force != requested["force"],
                "params": params_payload,
            }
            return status


class RealSenseColorCamera:
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.profile: Any | None = None
        self.serial: str | None = None
        self.width = DEFAULT_REALSENSE_WIDTH
        self.height = DEFAULT_REALSENSE_HEIGHT
        self.fps = DEFAULT_REALSENSE_FPS
        self.metadata: dict[str, Any] | None = None
        self.align_to_color: Any | None = None
        self.record_depth = False
        self.record_depth_align_to_color = False
        self.record_depth_every_n_frames = DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES

    def start(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        record_depth: bool = True,
        record_depth_align_to_color: bool = False,
        record_depth_every_n_frames: int = DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES,
        auto_exposure: bool = True,
        exposure: float | None = None,
        gain: float | None = None,
    ) -> dict[str, Any]:
        requested_depth = bool(record_depth)
        requested_depth_align = bool(record_depth and record_depth_align_to_color)
        if (
            self.pipeline is not None
            and self.serial == serial
            and self.width == width
            and self.height == height
            and self.fps == fps
            and self.record_depth == requested_depth
            and self.record_depth_align_to_color == requested_depth_align
            and self.record_depth_every_n_frames == max(1, int(record_depth_every_n_frames))
        ):
            return self.metadata or {}
        self.stop()
        import pyrealsense2 as rs  # type: ignore[import-not-found]

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        # OpenCV VideoWriter/imencode/checkerboard detection all operate naturally on
        # BGR frames, so request bgr8 from RealSense and avoid a per-frame RGB->BGR copy.
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        if record_depth:
            config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
        profile = pipeline.start(config)
        color_options = configure_color_sensor(profile, auto_exposure, exposure, gain)
        self.pipeline = pipeline
        self.profile = profile
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.record_depth = requested_depth
        self.record_depth_align_to_color = requested_depth_align
        self.record_depth_every_n_frames = max(1, int(record_depth_every_n_frames))
        self.align_to_color = rs.align(rs.stream.color) if self.record_depth_align_to_color else None
        self.metadata = camera_metadata_from_profile(profile, serial)
        self.metadata["colorOptions"] = color_options
        self.metadata["colorFormat"] = "bgr8"
        self.metadata["imageColorConvention"] = "opencv_bgr"
        self.metadata["depthEnabled"] = bool(record_depth)
        self.metadata["depthAlignedToColor"] = bool(self.record_depth_align_to_color)
        self.metadata["depthEveryNFrames"] = int(self.record_depth_every_n_frames)
        if record_depth:
            self.metadata["depthScaleM"] = depth_scale_from_profile(profile)
        return self.metadata

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.profile = None
        self.metadata = None
        self.align_to_color = None
        self.record_depth = False
        self.record_depth_align_to_color = False
        self.record_depth_every_n_frames = DEFAULT_RECORD_DEPTH_EVERY_N_FRAMES

    def capture_frame(self, warmup_frames: int, *, include_depth: bool = True) -> dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("RealSense camera is not started")
        color_frame = None
        depth_frame = None
        for _ in range(max(1, int(warmup_frames))):
            frames = self.pipeline.wait_for_frames(5000)
            if include_depth and self.align_to_color is not None:
                frames = self.align_to_color.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.record_depth and include_depth else None
        if not color_frame:
            raise RuntimeError("No RealSense color frame received")
        result: dict[str, Any] = {"bgr": np.asanyarray(color_frame.get_data()).copy()}
        if depth_frame:
            result["depth"] = np.asanyarray(depth_frame.get_data()).copy()
        return result

    def capture_rgb(self, warmup_frames: int) -> np.ndarray:
        return cv2.cvtColor(self.capture_frame(warmup_frames, include_depth=False)["bgr"], cv2.COLOR_BGR2RGB)

    def capture_bgr(self, warmup_frames: int) -> np.ndarray:
        return self.capture_frame(warmup_frames, include_depth=False)["bgr"]


class RealSenseStreamHub:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.cameras: dict[str, RealSenseColorCamera] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, dict[str, Any]] = {}
        self.subscribers: dict[str, list[queue.Queue[Any]]] = {}
        self.subscriber_drop_counts: dict[str, int] = {}
        self.signature: tuple[Any, ...] | None = None
        self.stop_event: threading.Event | None = None
        self.threads: dict[str, threading.Thread] = {}
        self.frame_count = 0
        self.frame_sequence = 0
        self.role_frame_counts: dict[str, int] = {}
        self.role_errors: dict[str, str | None] = {}
        self.last_error: str | None = None

    def start(self, config: FlexivRealSenseConfig) -> dict[str, Any]:
        signature = self._signature(config)
        with self.lock:
            if self._running_locked() and self.signature == signature:
                return self.status_locked()
        self.stop()

        cameras: dict[str, RealSenseColorCamera] = {}
        metadata: dict[str, dict[str, Any]] = {}
        try:
            for role, serial in self._camera_serials(config).items():
                camera = RealSenseColorCamera()
                current_metadata = camera.start(
                    serial,
                    int(config.width),
                    int(config.height),
                    int(config.fps),
                    bool(config.record_depth),
                    bool(config.record_depth_align_to_color),
                    max(1, int(config.record_depth_every_n_frames)),
                    bool(config.realsense_auto_exposure),
                    config.realsense_exposure,
                    config.realsense_gain,
                )
                current_metadata["role"] = role
                cameras[role] = camera
                metadata[role] = current_metadata
        except Exception:
            for camera in cameras.values():
                camera.stop()
            raise
        if not cameras:
            raise RuntimeError("No RealSense camera serials are configured")

        stop_event = threading.Event()
        threads = {
            role: threading.Thread(
                target=self._capture_role_loop,
                args=(role, camera, stop_event),
                name=f"realsense-stream-{role}",
                daemon=True,
            )
            for role, camera in cameras.items()
        }
        with self.lock:
            self.cameras = cameras
            self.metadata = metadata
            self.latest = {}
            self.subscribers = {role: [] for role in cameras}
            self.subscriber_drop_counts = {role: 0 for role in cameras}
            self.signature = signature
            self.stop_event = stop_event
            self.threads = threads
            self.frame_count = 0
            self.frame_sequence = 0
            self.role_frame_counts = {role: 0 for role in cameras}
            self.role_errors = {role: None for role in cameras}
            self.last_error = None
        for thread in threads.values():
            thread.start()
        return self.status()

    def stop(self) -> None:
        with self.lock:
            stop_event = self.stop_event
            threads = list(self.threads.values())
        if stop_event is not None:
            stop_event.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        with self.lock:
            cameras = self.cameras
            subscribers = {
                role: list(queues)
                for role, queues in self.subscribers.items()
            }
            self.cameras = {}
            self.metadata = {}
            self.latest = {}
            self.subscribers = {}
            self.subscriber_drop_counts = {}
            self.signature = None
            self.stop_event = None
            self.threads = {}
            self.role_frame_counts = {}
            self.role_errors = {}
        for queues in subscribers.values():
            for frame_queue in queues:
                try:
                    frame_queue.put_nowait(None)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait(None)
                    except queue.Full:
                        pass
        for camera in cameras.values():
            camera.stop()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.status_locked()

    def status_locked(self) -> dict[str, Any]:
        role_threads = {role: thread.is_alive() for role, thread in self.threads.items()}
        running = any(role_threads.values())
        return {
            "running": running,
            "roles": sorted(self.cameras.keys()),
            "metadata": self.metadata,
            "frameCount": self.frame_count,
            "frameCountByRole": dict(self.role_frame_counts),
            "roleThreads": role_threads,
            "roleErrors": {role: error for role, error in self.role_errors.items() if error},
            "subscriberDropCounts": dict(self.subscriber_drop_counts),
            "latest": {
                role: {
                    "capturedAtUtc": frame.get("capturedAtUtc"),
                    "serial": frame.get("serial"),
                    "sequence": frame.get("sequence"),
                    "roleSequence": frame.get("roleSequence"),
                    "hasDepth": isinstance(frame.get("depth"), np.ndarray),
                }
                for role, frame in self.latest.items()
            },
            "lastError": self.last_error,
        }

    def get_latest(
        self,
        role: str = "end",
        wait_timeout: float = 2.0,
        *,
        copy_arrays: bool = True,
        include_depth: bool = True,
    ) -> dict[str, Any]:
        deadline = time.perf_counter() + max(0.0, float(wait_timeout))
        while True:
            with self.lock:
                frame = self.latest.get(role)
                role_thread = self.threads.get(role)
                running = role_thread is not None and role_thread.is_alive()
                last_error = self.role_errors.get(role) or self.last_error
            if frame is not None:
                bgr = frame.get("bgr")
                depth = frame.get("depth")
                return {
                    **frame,
                    "bgr": bgr.copy() if copy_arrays and isinstance(bgr, np.ndarray) else bgr,
                    "depth": (
                        depth.copy()
                        if include_depth and copy_arrays and isinstance(depth, np.ndarray)
                        else (depth if include_depth else None)
                    ),
                    "metadata": dict(frame["metadata"]),
                }
            if not running:
                raise RuntimeError(last_error or "RealSense stream is not running")
            if time.perf_counter() >= deadline:
                raise RuntimeError(f"No {role} RealSense frame is available yet")
            time.sleep(0.02)

    def snapshot_roles(self, roles: list[str], wait_timeout: float = 2.0) -> dict[str, dict[str, Any]]:
        return {role: self.get_latest(role, wait_timeout=wait_timeout) for role in roles}

    def wait_for_roles(self, roles: list[str], timeout_seconds: float) -> dict[str, Any]:
        role_names = [str(role or "").strip().lower() for role in roles if str(role or "").strip()]
        deadline = time.perf_counter() + max(0.0, float(timeout_seconds))
        while True:
            with self.lock:
                latest = {
                    role: self.latest.get(role)
                    for role in role_names
                }
                missing = [role for role, frame in latest.items() if not isinstance(frame, dict)]
                role_threads = {
                    role: bool(self.threads.get(role) and self.threads[role].is_alive())
                    for role in role_names
                }
                errors = {
                    role: self.role_errors.get(role)
                    for role in role_names
                    if self.role_errors.get(role)
                }
                status = self.status_locked()
            if not missing:
                return {"ok": True, "roles": role_names, "missingRoles": [], "stream": status}
            if time.perf_counter() >= deadline or any(not role_threads.get(role) for role in missing):
                return {
                    "ok": False,
                    "roles": role_names,
                    "missingRoles": missing,
                    "roleThreads": role_threads,
                    "roleErrors": errors,
                    "lastError": self.last_error,
                    "stream": status,
                }
            time.sleep(0.02)

    def subscribe(self, role: str, *, max_queue_size: int = 128) -> queue.Queue[Any]:
        role_name = str(role or "").strip().lower()
        if not role_name:
            raise ValueError("RealSense subscriber role is empty")
        frame_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(max_queue_size)))
        with self.lock:
            if role_name not in self.cameras:
                raise RuntimeError(f"RealSense role is not active: {role_name}")
            self.subscribers.setdefault(role_name, []).append(frame_queue)
        return frame_queue

    def unsubscribe(self, role: str, frame_queue: queue.Queue[Any]) -> None:
        role_name = str(role or "").strip().lower()
        with self.lock:
            queues = self.subscribers.get(role_name) or []
            if frame_queue in queues:
                queues.remove(frame_queue)

    def _running_locked(self) -> bool:
        return any(thread.is_alive() for thread in self.threads.values())

    def _capture_role_loop(
        self,
        role: str,
        camera: RealSenseColorCamera,
        stop_event: threading.Event,
    ) -> None:
        local_capture_index = 0
        while not stop_event.is_set():
            try:
                every_n = max(1, int(camera.record_depth_every_n_frames))
                include_depth = bool(camera.record_depth and local_capture_index % every_n == 0)
                captured = camera.capture_frame(1, include_depth=include_depth)
                bgr = captured["bgr"]
                captured_at_perf = time.perf_counter()
                captured_at = datetime.now(timezone.utc).isoformat()
                subscribers: list[queue.Queue[Any]] = []
                with self.lock:
                    self.frame_sequence += 1
                    self.frame_count += 1
                    role_sequence = int(self.role_frame_counts.get(role, 0)) + 1
                    self.role_frame_counts[role] = role_sequence
                    frame_payload = {
                        "role": role,
                        "serial": camera.serial,
                        "metadata": self.metadata.get(role, {}),
                        "bgr": bgr,
                        "depth": captured.get("depth"),
                        "depthCaptureIndex": local_capture_index if include_depth else None,
                        "capturedAtUtc": captured_at,
                        "capturedAtPerfCounterSeconds": captured_at_perf,
                        "sequence": self.frame_sequence,
                        "roleSequence": role_sequence,
                    }
                    self.latest[role] = frame_payload
                    subscribers = list(self.subscribers.get(role) or [])
                    self.role_errors[role] = None
                    if not any(self.role_errors.values()):
                        self.last_error = None
                for frame_queue in subscribers:
                    try:
                        frame_queue.put_nowait(frame_payload)
                    except queue.Full:
                        try:
                            frame_queue.get_nowait()
                            try:
                                frame_queue.task_done()
                            except ValueError:
                                pass
                        except queue.Empty:
                            pass
                        with self.lock:
                            self.subscriber_drop_counts[role] = int(self.subscriber_drop_counts.get(role, 0)) + 1
                        try:
                            frame_queue.put_nowait(frame_payload)
                        except queue.Full:
                            pass
                local_capture_index += 1
            except Exception as exc:  # pragma: no cover - hardware path
                message = str(exc)
                with self.lock:
                    self.role_errors[role] = message
                    self.last_error = f"{role}: {message}"
                time.sleep(0.1)

    def _signature(self, config: FlexivRealSenseConfig) -> tuple[Any, ...]:
        serials = tuple(sorted(self._camera_serials(config).items()))
        return (
            serials,
            int(config.width),
            int(config.height),
            int(config.fps),
            bool(config.record_depth),
            bool(config.record_depth_align_to_color),
            max(1, int(config.record_depth_every_n_frames)),
            bool(config.realsense_auto_exposure),
            config.realsense_exposure,
            config.realsense_gain,
        )

    def _camera_serials(self, config: FlexivRealSenseConfig) -> dict[str, str]:
        end = str(config.camera_serial or "").strip()
        third = str(config.third_camera_serial or "").strip()
        serials: dict[str, str] = {}
        if end:
            serials["end"] = end
        if third and third != end:
            serials["third"] = third
        return serials


DEFAULT_COLOR_VIDEO_CRF = 23
DEFAULT_COLOR_VIDEO_PRESET = "veryfast"


def ffmpeg_encoder_command_prefix(ffmpeg: str) -> list[str]:
    if os.name == "posix" and DEFAULT_FFMPEG_NICE_LEVEL > 0:
        nice = shutil.which("nice")
        if nice:
            return [nice, "-n", str(DEFAULT_FFMPEG_NICE_LEVEL), ffmpeg]
    return [ffmpeg]


class ColorStreamWriter:
    """Encode BGR frames to H.264 via a system ffmpeg pipe (libx264).

    OpenCV's bundled FFmpeg on the lab machine only exposes the v4l2m2m H.264
    backend (no valid device), so writing H.264 through cv2.VideoWriter silently
    fails. The system ffmpeg has libx264, and the depth stream already uses this
    same pipe pattern, so color reuses it to get ~2-4x smaller files than mp4v.
    """

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = DEFAULT_COLOR_VIDEO_CRF,
        preset: str = DEFAULT_COLOR_VIDEO_PRESET,
    ) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self.crf = int(crf)
        self.preset = str(preset)
        self.codec = "libx264"
        self.frame_count = 0
        self.process: subprocess.Popen[bytes] | None = None
        self.stderr_handle: Any | None = None
        self.stderr_path: Path | None = None
        self.error: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg executable not found")
        command = [
            *ffmpeg_encoder_command_prefix(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            str(DEFAULT_FFMPEG_ENCODER_THREADS),
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self.stderr_path = self.path.with_name(self.path.name + ".ffmpeg.log")
        self.stderr_handle = self.stderr_path.open("ab")
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.stderr_handle,
            )
        except Exception:
            if self.stderr_handle is not None:
                self.stderr_handle.close()
                self.stderr_handle = None
            raise
        if self.process.stdin is None:
            try:
                self.process.kill()
            except Exception:
                pass
            if self.stderr_handle is not None:
                self.stderr_handle.close()
                self.stderr_handle = None
            raise RuntimeError("ffmpeg stdin pipe is unavailable")

    def write(self, bgr: np.ndarray) -> None:
        if bgr.shape[0] != self.height or bgr.shape[1] != self.width:
            raise ValueError(
                f"color frame shape changed from {self.width}x{self.height} "
                f"to {int(bgr.shape[1])}x{int(bgr.shape[0])}"
            )
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("ffmpeg color writer is not open")
        if not bgr.flags["C_CONTIGUOUS"] or bgr.dtype != np.uint8:
            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        self.process.stdin.write(memoryview(bgr).cast("B"))
        self.frame_count += 1

    def close(self) -> None:
        if self.process is None:
            return
        process = self.process
        try:
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait(timeout=15.0)
        except Exception as exc:
            self.error = str(exc)
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=2.0)
            except Exception:
                pass
            raise
        finally:
            self.process = None
            if self.stderr_handle is not None:
                self.stderr_handle.close()
                self.stderr_handle = None
        if return_code != 0:
            message = ""
            if self.stderr_path is not None and self.stderr_path.exists():
                try:
                    message = self.stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                except Exception:
                    message = ""
            self.error = message or f"ffmpeg exited with code {return_code}"
            raise RuntimeError(self.error)
        if self.stderr_path is not None and self.stderr_path.exists():
            try:
                if self.stderr_path.stat().st_size == 0:
                    self.stderr_path.unlink()
            except Exception:
                pass


class DepthStreamWriter:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        encoding: str,
    ) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1.0, float(fps))
        self.encoding = str(encoding or "raw")
        self.frame_count = 0
        self.byte_count = 0
        self.process: subprocess.Popen[bytes] | None = None
        self.handle: Any | None = None
        self.stderr_handle: Any | None = None
        self.stderr_path: Path | None = None
        self.error: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.encoding == "ffv1":
            self._open_ffv1()
        else:
            self.encoding = "raw"
            self.handle = self.path.open("ab")

    def _open_ffv1(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg executable not found")
        command = [
            *ffmpeg_encoder_command_prefix(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-threads",
            str(DEFAULT_FFMPEG_ENCODER_THREADS),
            "-level",
            "3",
            "-g",
            "1",
            "-slices",
            "24",
            "-slicecrc",
            "1",
            str(self.path),
        ]
        self.stderr_path = self.path.with_name(self.path.name + ".ffmpeg.log")
        self.stderr_handle = self.stderr_path.open("ab")
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.stderr_handle,
            )
        except Exception:
            if self.stderr_handle is not None:
                self.stderr_handle.close()
                self.stderr_handle = None
            raise
        if self.process.stdin is None:
            try:
                self.process.kill()
            except Exception:
                pass
            if self.stderr_handle is not None:
                self.stderr_handle.close()
                self.stderr_handle = None
            raise RuntimeError("ffmpeg stdin pipe is unavailable")

    def write(self, depth: np.ndarray) -> dict[str, Any]:
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            raise ValueError(
                f"depth frame shape changed from {self.width}x{self.height} "
                f"to {int(depth.shape[1])}x{int(depth.shape[0])}"
            )
        if depth.dtype != np.uint16 or not depth.flags["C_CONTIGUOUS"]:
            depth = np.ascontiguousarray(depth, dtype=np.uint16)
        data = memoryview(depth).cast("B")
        byte_length = data.nbytes
        frame_index = self.frame_count
        byte_offset = self.byte_count
        if self.encoding == "ffv1":
            if self.process is None or self.process.stdin is None:
                raise RuntimeError("ffmpeg depth writer is not open")
            self.process.stdin.write(data)
        else:
            if self.handle is None:
                raise RuntimeError("raw depth writer is not open")
            self.handle.write(data)
        self.frame_count += 1
        self.byte_count += byte_length
        return {
            "streamFrameIndex": frame_index,
            "byteOffset": byte_offset if self.encoding == "raw" else None,
            "byteLength": byte_length if self.encoding == "raw" else None,
            "sourceByteOffset": byte_offset,
            "sourceByteLength": byte_length,
        }

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.process is not None:
            process = self.process
            try:
                if process.stdin is not None:
                    process.stdin.close()
                return_code = process.wait(timeout=10.0)
            except Exception as exc:
                self.error = str(exc)
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=2.0)
                except Exception:
                    pass
                raise
            finally:
                self.process = None
                if self.stderr_handle is not None:
                    self.stderr_handle.close()
                    self.stderr_handle = None
            if return_code != 0:
                message = ""
                if self.stderr_path is not None and self.stderr_path.exists():
                    try:
                        message = self.stderr_path.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:
                        message = ""
                self.error = message or f"ffmpeg exited with code {return_code}"
                raise RuntimeError(self.error)
            if self.stderr_path is not None and self.stderr_path.exists():
                try:
                    if self.stderr_path.stat().st_size == 0:
                        self.stderr_path.unlink()
                except Exception:
                    pass


class RobotRealsenseSession:
    def __init__(
        self,
        root: Path,
        record_id: str,
        config: FlexivRealSenseConfig,
        robot: FlexivRobotClient,
        stream_hub: RealSenseStreamHub,
        publish_event: Callable[[dict[str, Any]], None] | None = None,
        robot_alignment_result: dict[str, Any] | None = None,
        require_controller_alignment: bool = False,
        control_mode: str = ROBOT_SESSION_CONTROL_TELEOP,
        record_mode: str = ROBOT_SESSION_RECORD_SYNC,
    ) -> None:
        self.root = root.resolve()
        self.record_id = record_id
        self.config = config
        self.robot = robot
        self.stream_hub = stream_hub
        self.publish_event = publish_event
        self.control_mode = (
            control_mode if control_mode in ROBOT_SESSION_CONTROL_MODES else ROBOT_SESSION_CONTROL_RECORD_ONLY
        )
        self.record_mode = record_mode if record_mode in ROBOT_SESSION_RECORD_MODES else ROBOT_SESSION_RECORD_SYNC
        self.directory = self.root / "robot_realsense"
        self.image_dir = self.directory / "images"
        self.video_dir = self.directory / "videos"
        self.depth_dir = self.directory / "depth"
        self.samples_path = self.directory / "samples.jsonl"
        self.robot_states_path = self.directory / "robot_states.jsonl"
        self.motion_path = self.directory / "controller_motion.jsonl"
        self.gripper_path = self.directory / "gripper_commands.jsonl"
        self.video_frames_path = self.directory / "video_frames.jsonl"
        self.summary_path = self.directory / "session_summary.json"
        self.samples_handle: Any | None = None
        self.robot_states_handle: Any | None = None
        self.motion_handle: Any | None = None
        self.gripper_handle: Any | None = None
        self.video_frames_handle: Any | None = None
        self.video_writers: dict[str, ColorStreamWriter] = {}
        self.video_codecs: dict[str, str] = {}
        self.video_paths: dict[str, str] = {}
        self.video_frame_counts: dict[str, int] = {}
        self.depth_stream_handles: dict[str, DepthStreamWriter] = {}
        self.depth_stream_paths: dict[str, str] = {}
        self.depth_stream_encodings: dict[str, str] = {}
        self.depth_stream_frame_counts: dict[str, int] = {}
        self.depth_stream_warnings: list[dict[str, Any]] = []
        self.depth_stream_disabled_roles: dict[str, str] = {}
        self.camera_frame_queues: dict[str, queue.Queue[Any]] = {}
        self.camera_writer_threads: dict[str, threading.Thread] = {}
        self.video_role_locks: dict[str, threading.RLock] = {}
        self.camera_queue_drop_counts: dict[str, int] = {}
        self.camera_subscriber_drop_start_counts: dict[str, int] = {}
        self.camera_queue_max_sizes: dict[str, int] = {}
        self.camera_writer_thread_alive_on_close: dict[str, bool] = {}
        self.close_errors: list[dict[str, Any]] = []
        self.depth_count = 0
        self.latest_video_frame_rows: dict[str, dict[str, Any]] = {}
        self.latest_video_sequences: dict[str, Any] = {}
        self.recording_start_perf_counter: float | None = None
        self.recording_start_unix_seconds: float | None = None
        self.robot_state_perf_stats = OnlineTimeSeriesStats()
        self.robot_state_target_perf_stats = OnlineTimeSeriesStats()
        self.robot_state_lateness_stats = OnlineNumericStats()
        self.robot_state_missed_tick_count = 0
        self.aligned_sample_perf_stats = OnlineTimeSeriesStats()
        self.aligned_sample_target_perf_stats = OnlineTimeSeriesStats()
        self.aligned_sample_lateness_stats = OnlineNumericStats()
        self.aligned_quest_source_age_stats = OnlineNumericStats()
        self.aligned_reused_source_count = 0
        self.video_frame_perf_stats_by_role: dict[str, OnlineTimeSeriesStats] = {}
        self.depth_frame_perf_stats_by_role: dict[str, OnlineTimeSeriesStats] = {}
        self.camera_write_duration_stats_by_role: dict[str, OnlineNumericStats] = {}
        self.camera_capture_to_write_latency_stats_by_role: dict[str, OnlineNumericStats] = {}
        self.lock = threading.RLock()
        self.video_io_lock = threading.RLock()
        self.stop_event: threading.Event | None = None
        self.robot_state_thread: threading.Thread | None = None
        self.camera_record_thread: threading.Thread | None = None
        self.next_capture_perf = 0.0
        self.sample_count = 0
        self.robot_state_count = 0
        self.quest_aligned_sample_count = 0
        self.image_count = 0
        self.video_frame_count = 0
        self.error_count = 0
        self.robot_state_error_count = 0
        self.camera_record_error_count = 0
        self.closed = False
        self.last_error: str | None = None
        self.last_robot_state_error: str | None = None
        self.last_camera_record_error: str | None = None
        self.camera_metadata: dict[str, dict[str, Any]] = {}
        self.latest_robot_state_row: dict[str, Any] | None = None
        self.controller_anchor_world: np.ndarray | None = None
        self.controller_anchor_rotation_world: np.ndarray | None = None
        self.robot_anchor_tcp_pose: list[float] | None = None
        self.motion_command_count = 0
        self.motion_skip_count = 0
        self.motion_rate_limit_count = 0
        self.motion_error_count = 0
        self.last_motion_event: dict[str, Any] | None = None
        self.gripper_command_count = 0
        self.gripper_skip_count = 0
        self.gripper_error_count = 0
        self.gripper_available = False
        self.last_gripper_event: dict[str, Any] | None = None
        self.last_gripper_closed: bool | None = None
        self.last_gripper_unavailable_closed: bool | None = None
        self.next_controller_motion_target_perf = 0.0
        self.ee_pose_history: list[np.ndarray] = []
        self.text_pending_rows = {
            "samples": 0,
            "robot_states": 0,
            "motion": 0,
            "gripper": 0,
        }
        self.video_pending_rows = 0
        now_perf = time.perf_counter()
        self.text_last_flush_perf = now_perf
        self.video_last_flush_perf = now_perf
        self.robot_alignment_result = robot_alignment_result if isinstance(robot_alignment_result, dict) else None
        self.require_controller_alignment = require_controller_alignment
        self.t_ee_end_camera = self._alignment_transform("end_camera", "T_ee_realsense")
        self.t_base_world = self._alignment_transform("questAlignment", "T_base_world")
        self.cartesian_send_baseline = self.robot.cartesian_send_status()

    def start(self) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.samples_handle = self.samples_path.open("a", encoding="utf-8", newline="\n")
        self.robot_states_handle = self.robot_states_path.open("a", encoding="utf-8", newline="\n")
        self.motion_handle = self.motion_path.open("a", encoding="utf-8", newline="\n")
        self.gripper_handle = self.gripper_path.open("a", encoding="utf-8", newline="\n")
        self.video_frames_handle = self.video_frames_path.open("a", encoding="utf-8", newline="\n")
        config_payload = config_to_json(self.config)
        config_payload["recordId"] = self.record_id
        config_payload["controlMode"] = self.control_mode
        config_payload["recordMode"] = self.record_mode
        config_payload["startedAtUtc"] = datetime.now(timezone.utc).isoformat()
        write_json(config_payload, self.directory / "capture_config.json")
        stream_status = self.stream_hub.start(self.config)
        metadata = stream_status.get("metadata") if isinstance(stream_status, dict) else None
        self.camera_metadata = metadata if isinstance(metadata, dict) else {}
        warmup_status = self.stream_hub.wait_for_roles(
            sorted(self.camera_metadata.keys()),
            REALSENSE_RECORD_START_WAIT_SECONDS,
        )
        config_payload["realsenseWarmup"] = {
            "ok": warmup_status.get("ok"),
            "missingRoles": warmup_status.get("missingRoles"),
            "timeoutSeconds": REALSENSE_RECORD_START_WAIT_SECONDS,
        }
        write_json(config_payload, self.directory / "capture_config.json")
        self.camera_subscriber_drop_start_counts = dict(
            warmup_status.get("stream", {}).get("subscriberDropCounts")
            if isinstance(warmup_status.get("stream"), dict)
            and isinstance(warmup_status.get("stream", {}).get("subscriberDropCounts"), dict)
            else stream_status.get("subscriberDropCounts")
            if isinstance(stream_status, dict) and isinstance(stream_status.get("subscriberDropCounts"), dict)
            else {}
        )
        write_json(self.camera_metadata, self.directory / "cameras.json")
        if self.robot_alignment_result is not None:
            write_json(self.robot_alignment_result, self.directory / "robot_hand_eye_result.json")
        if self.config.gripper_enabled:
            self._try_initialize_gripper()
        self.stop_event = threading.Event()
        with self.lock:
            self.recording_start_perf_counter = time.perf_counter()
            self.recording_start_unix_seconds = time.time()
        if self.record_mode == ROBOT_SESSION_RECORD_ASYNC:
            self.robot_state_thread = threading.Thread(
                target=self._robot_state_loop,
                name=f"robot-state-record-{safe_filename(self.record_id)}",
                daemon=True,
            )
            self.robot_state_thread.start()
            for role in sorted(self.camera_metadata.keys()):
                queue_size = async_camera_queue_size(self.config.fps)
                frame_queue = self.stream_hub.subscribe(
                    role,
                    max_queue_size=queue_size,
                )
                self.camera_frame_queues[role] = frame_queue
                self.camera_queue_max_sizes[role] = queue_size
                thread = threading.Thread(
                    target=self._camera_role_record_loop,
                    args=(role, frame_queue),
                    name=f"robot-camera-record-{safe_filename(self.record_id)}-{role}",
                    daemon=True,
                )
                self.camera_writer_threads[role] = thread
                thread.start()
        summary = self.summary("recording")
        self._publish({"type": "robot_status", "stage": "robot_realsense_recording", **summary})
        return summary

    def update_controller_motion(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
        if self.control_mode != ROBOT_SESSION_CONTROL_TELEOP:
            return None
        with self.lock:
            if self.closed:
                return None
        update_started_perf = time.perf_counter()
        try:
            event = self._update_controller_motion_unlocked(quest_sample)
        except Exception as exc:  # pragma: no cover - hardware path
            self.motion_error_count += 1
            event = {
                "ok": False,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "error": str(exc),
            }
        update_finished_perf = time.perf_counter()
        event.setdefault("pc_perf_counter_seconds", update_finished_perf)
        event.setdefault("pc_unix_seconds", time.time())
        event["update_duration_seconds"] = update_finished_perf - update_started_perf
        should_record_event = self._should_record_motion_event(event)
        with self.lock:
            if event.get("reason") != "rate_limited":
                self.last_motion_event = event
            if should_record_event and self.motion_handle is not None:
                self.motion_handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._note_text_write_locked("motion")
        if event.get("reason") == "rate_limited":
            self.motion_rate_limit_count += 1
        elif event.get("ok") and event.get("commandSent") is not False:
            self.motion_command_count += 1
        else:
            self.motion_skip_count += 1
        if should_record_event:
            self._publish(robot_motion_event(event))
        return event

    def update_gripper(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
        if not self.config.gripper_enabled:
            return None
        with self.lock:
            if self.closed:
                return None
        update_started_perf = time.perf_counter()
        try:
            if not self.gripper_available:
                event = self._gripper_unavailable_event(quest_sample)
            else:
                event = self._update_gripper_unlocked(quest_sample)
        except Exception as exc:  # pragma: no cover - hardware path
            self.gripper_available = False
            self.gripper_error_count += 1
            event = {
                "ok": False,
                "commandSent": False,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "error": str(exc),
                "disabledAfterError": True,
            }
        update_finished_perf = time.perf_counter()
        event.setdefault("pc_perf_counter_seconds", update_finished_perf)
        event.setdefault("pc_unix_seconds", time.time())
        event["update_duration_seconds"] = update_finished_perf - update_started_perf
        should_record_event = self._should_record_gripper_event(event)
        with self.lock:
            self.last_gripper_event = event
            if should_record_event and self.gripper_handle is not None:
                self.gripper_handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._note_text_write_locked("gripper")
        if event.get("ok") and event.get("commandSent"):
            self.gripper_command_count += 1
        else:
            self.gripper_skip_count += 1
        if should_record_event:
            self._publish(robot_gripper_event(event))
        return event

    @staticmethod
    def _should_record_motion_event(event: dict[str, Any]) -> bool:
        if event.get("reason") == "rate_limited":
            return False
        if event.get("ok") or event.get("error"):
            return True
        return event.get("teleopHeld") is True

    @staticmethod
    def _should_record_gripper_event(event: dict[str, Any]) -> bool:
        if event.get("commandSent") or event.get("error") or event.get("disabledAfterError"):
            return True
        if event.get("reason") == "gripper_not_available" and event.get("triggerStateChanged"):
            return True
        return event.get("action") == "initialize"

    def _gripper_unavailable_event(self, quest_sample: dict[str, Any]) -> dict[str, Any]:
        controller = quest_sample.get("rightController")
        trigger = controller_trigger_value(controller) if isinstance(controller, dict) else None
        desired_closed: bool | None = None
        if trigger is not None:
            close_threshold = float(self.config.gripper_trigger_close_threshold)
            open_threshold = float(self.config.gripper_trigger_open_threshold)
            if trigger >= close_threshold:
                desired_closed = True
            elif trigger <= open_threshold:
                desired_closed = False
        changed = desired_closed is not None and self.last_gripper_unavailable_closed is not desired_closed
        if desired_closed is not None:
            self.last_gripper_unavailable_closed = desired_closed
        event = self._gripper_skip_event(quest_sample, "gripper_not_available", trigger)
        event["triggerState"] = (
            "close" if desired_closed is True else "open" if desired_closed is False else "hysteresis"
        )
        event["triggerStateChanged"] = bool(changed)
        event["status"] = self.robot.gripper_status()
        return event

    def record_sample(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
        if self.record_mode == ROBOT_SESSION_RECORD_ASYNC:
            return self._record_async_sample(quest_sample)
        return self._record_sync_sample(quest_sample)

    def _record_sync_sample(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
        now_perf = time.perf_counter()
        if now_perf < self.next_capture_perf:
            return None
        with self.lock:
            if self.closed:
                return None
            now_perf = time.perf_counter()
            if now_perf < self.next_capture_perf:
                return None
            interval = max(0.0, float(self.config.capture_interval_seconds))
            self.next_capture_perf = now_perf + interval
            try:
                row = self._capture_row(quest_sample)
            except Exception as exc:  # pragma: no cover - hardware path
                self.error_count += 1
                self.last_error = str(exc)
                row = {
                    "sample_index": self.sample_count,
                    "record_id": self.record_id,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "ok": False,
                    "error": self.last_error,
                    "quest_sample_index": quest_sample.get("sampleIndex"),
                    "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
                }
            if row.get("ok"):
                images = row.get("images") if isinstance(row.get("images"), dict) else {}
                self.image_count += len(images)
                t_base_ee = self._row_tool_or_ee_transform(row)
                if t_base_ee is not None:
                    self._remember_ee_pose(t_base_ee)
            assert self.samples_handle is not None
            self.samples_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._note_text_write_locked("samples")
            if self.robot_states_handle is not None:
                self.robot_states_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._note_text_write_locked("robot_states")
            self.sample_count += 1
            self.robot_state_count += 1
            self.latest_robot_state_row = row
            self._publish(robot_sample_event(row))
            return row

    def _record_async_sample(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            if self.closed:
                return None
            aligned_index = self.quest_aligned_sample_count
            self.quest_aligned_sample_count += 1
        try:
            row = self._compose_async_sample_row(quest_sample, sample_index=aligned_index)
        except Exception as exc:
            with self.lock:
                self.error_count += 1
                self.last_error = str(exc)
            row = {
                "sample_index": aligned_index,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "pc_perf_counter_seconds": time.perf_counter(),
                "pc_unix_seconds": time.time(),
                "ok": False,
                "error": str(exc),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
            }

        with self.lock:
            if self.closed:
                return row
            if self.samples_handle is not None:
                row["written_sample_index"] = self.sample_count
                self.samples_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._note_text_write_locked("samples")
                self.sample_count += 1
                actual_perf = row.get("pc_perf_counter_seconds")
                target_perf = row.get("aligned_target_perf_counter_seconds")
                quest_perf = row.get("quest_pc_receive_perf_counter_seconds")
                self.aligned_sample_perf_stats.add(actual_perf)
                self.aligned_sample_target_perf_stats.add(target_perf)
                self.aligned_sample_lateness_stats.add(row.get("aligned_lateness_seconds"))
                if is_number(actual_perf) and is_number(quest_perf):
                    self.aligned_quest_source_age_stats.add(max(0.0, float(actual_perf) - float(quest_perf)))
                if row.get("aligned_source_reused"):
                    self.aligned_reused_source_count += 1
        self._publish(robot_sample_event(row))
        return row

    def close(self) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return self.summary("already_closed")
            self.closed = True
            stop_event = self.stop_event
            robot_state_thread = self.robot_state_thread
            camera_record_thread = self.camera_record_thread
            camera_writer_threads = dict(self.camera_writer_threads)
            camera_frame_queues = dict(self.camera_frame_queues)
            self.stop_event = None
            self.robot_state_thread = None
            self.camera_record_thread = None
            self.camera_writer_threads = {}
            self.camera_frame_queues = {}
        if stop_event is not None:
            stop_event.set()
        for role, frame_queue in camera_frame_queues.items():
            self.stream_hub.unsubscribe(role, frame_queue)
            sentinel_dropped = self._enqueue_camera_queue_sentinel(frame_queue)
            if sentinel_dropped:
                with self.lock:
                    self.camera_queue_drop_counts[role] = (
                        int(self.camera_queue_drop_counts.get(role, 0)) + sentinel_dropped
                    )
        for thread in [robot_state_thread, camera_record_thread, *camera_writer_threads.values()]:
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=ROBOT_SESSION_THREAD_JOIN_SECONDS)
        for role, thread in camera_writer_threads.items():
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=ROBOT_SESSION_CAMERA_WRITER_FINAL_JOIN_SECONDS)
                if thread.is_alive():
                    self._note_close_error(
                        "camera_writer_thread_alive",
                        f"{role} camera writer did not stop before video finalization",
                    )
        for role, frame_queue in camera_frame_queues.items():
            dropped = self._drain_camera_frame_queue(frame_queue)
            if dropped:
                with self.lock:
                    self.camera_queue_drop_counts[role] = int(self.camera_queue_drop_counts.get(role, 0)) + dropped
        with self.lock:
            self.camera_writer_thread_alive_on_close = {
                role: bool(thread is not None and thread.is_alive())
                for role, thread in camera_writer_threads.items()
            }
            alive_camera_roles = {
                role for role, alive in self.camera_writer_thread_alive_on_close.items() if alive
            }
        with self.lock:
            self._flush_text_handles_locked()
            if self.samples_handle is not None:
                self.samples_handle.close()
                self.samples_handle = None
            if self.robot_states_handle is not None:
                self.robot_states_handle.close()
                self.robot_states_handle = None
            if self.motion_handle is not None:
                self.motion_handle.close()
                self.motion_handle = None
            if self.gripper_handle is not None:
                self.gripper_handle.close()
                self.gripper_handle = None
        with self.video_io_lock:
            self._flush_video_handles_locked()
            if self.video_frames_handle is not None:
                self.video_frames_handle.close()
                self.video_frames_handle = None
        for role in list(self.video_role_locks.keys()):
            if self.camera_writer_thread_alive_on_close.get(role):
                continue
            role_lock = self._video_role_lock(role)
            acquired = role_lock.acquire(timeout=ROBOT_SESSION_VIDEO_RELEASE_LOCK_SECONDS)
            if not acquired:
                self._note_close_error("video_release_lock_timeout", f"{role} video writer lock timed out")
                continue
            try:
                writer = self.video_writers.pop(role, None)
                if writer is not None:
                    self._close_color_video_writer(role, writer)
                self.video_codecs.pop(role, None)
                self._close_depth_stream(role)
            finally:
                role_lock.release()
        with self.video_io_lock:
            for role in list(self.depth_stream_handles.keys()):
                if role in alive_camera_roles:
                    continue
                self._close_depth_stream(role)
            for role, writer in list(self.video_writers.items()):
                if role in alive_camera_roles:
                    continue
                self._close_color_video_writer(role, writer)
                self.video_writers.pop(role, None)
                self.video_codecs.pop(role, None)
            for role in list(self.video_role_locks.keys()):
                if role not in alive_camera_roles:
                    self.video_role_locks.pop(role, None)
        with self.lock:
            stream_status = self.stream_hub.status()
            subscriber_drops = self._camera_subscriber_drops_since_start(stream_status)
            summary = self.summary("closed")
            summary["cameraSubscriberDrops"] = dict(subscriber_drops)
            if self.close_errors:
                summary["closeErrors"] = list(self.close_errors)
            write_json(summary, self.summary_path)
        self._publish({"type": "robot_status", "stage": "robot_realsense_closed", **summary})
        return summary

    def summary(self, stage: str) -> dict[str, Any]:
        elapsed = None
        if self.recording_start_perf_counter is not None:
            elapsed = max(0.0, float(time.perf_counter() - self.recording_start_perf_counter))
        subscriber_drops = self._camera_subscriber_drops_since_start()
        return {
            "ok": (
                self.error_count == 0
                and self.robot_state_error_count == 0
                and self.camera_record_error_count == 0
                and not self.close_errors
            ),
            "recordId": self.record_id,
            "stage": stage,
            "directory": str(self.directory),
            "recordMode": self.record_mode,
            "recordingDurationSeconds": elapsed,
            "recordingStartUnixSeconds": self.recording_start_unix_seconds,
            "robotStateHz": float(self.config.robot_state_hz),
            "cameraSerial": self.config.camera_serial,
            "thirdCameraSerial": self.config.third_camera_serial,
            "cameraRoles": sorted(self.camera_metadata.keys()),
            "samples": self.sample_count,
            "questAlignedSamples": self.quest_aligned_sample_count,
            "robotStateSamples": self.robot_state_count,
            "images": self.image_count,
            "videos": self.video_paths,
            "depthStreams": self.depth_stream_paths,
            "depthStreamEncodings": dict(self.depth_stream_encodings),
            "depthStreamFrameCounts": dict(self.depth_stream_frame_counts),
            "depthStreamWarnings": list(self.depth_stream_warnings),
            "depthStreamDisabledRoles": dict(self.depth_stream_disabled_roles),
            "videoFrames": self.video_frame_count,
            "depthFrames": self.depth_count,
            "cameraQueueDrops": dict(self.camera_queue_drop_counts),
            "cameraSubscriberDrops": dict(subscriber_drops),
            "cameraQueueMaxSize": dict(self.camera_queue_max_sizes),
            "cameraWriterThreadsAliveOnClose": dict(self.camera_writer_thread_alive_on_close),
            "errors": self.error_count,
            "robotStateErrors": self.robot_state_error_count,
            "cameraRecordErrors": self.camera_record_error_count,
            "motionCommands": self.motion_command_count,
            "motionSkips": self.motion_skip_count,
            "motionRateLimited": self.motion_rate_limit_count,
            "motionErrors": self.motion_error_count,
            "lastMotion": self.last_motion_event,
            "controlMode": self.control_mode,
            "controllerAlignmentRequired": self.require_controller_alignment,
            "controllerAlignmentAvailable": self.t_base_world is not None,
            "gripperEnabled": self.config.gripper_enabled,
            "gripperAvailable": self.gripper_available,
            "gripperCommands": self.gripper_command_count,
            "gripperSkips": self.gripper_skip_count,
            "gripperErrors": self.gripper_error_count,
            "lastGripper": self.last_gripper_event,
            "cartesianControl": self._cartesian_control_summary(),
            "poseDiversity": ee_pose_diversity(self.ee_pose_history),
            "performance": self._performance_summary(),
            "closeErrors": list(self.close_errors),
            "lastError": self.last_error,
            "lastRobotStateError": self.last_robot_state_error,
            "lastCameraRecordError": self.last_camera_record_error,
        }

    def _cartesian_control_summary(self) -> dict[str, Any]:
        current = self.robot.cartesian_send_status()
        baseline = self.cartesian_send_baseline if isinstance(self.cartesian_send_baseline, dict) else {}
        count = max(0, int(current.get("count") or 0) - int(baseline.get("count") or 0))
        errors = max(0, int(current.get("errors") or 0) - int(baseline.get("errors") or 0))
        slow = max(0, int(current.get("slowCalls") or 0) - int(baseline.get("slowCalls") or 0))
        return {
            "count": count,
            "errors": errors,
            "slowCalls": slow,
            "last": current.get("last"),
            "slowThresholdSeconds": current.get("slowThresholdSeconds"),
            "globalCount": current.get("count"),
            "globalErrors": current.get("errors"),
            "globalSlowCalls": current.get("slowCalls"),
        }

    def _performance_summary(self) -> dict[str, Any]:
        elapsed = None
        if self.recording_start_perf_counter is not None:
            elapsed = max(0.0, float(time.perf_counter() - self.recording_start_perf_counter))
        target_camera_hz = float(self.config.fps)
        depth_every = max(1, int(self.config.record_depth_every_n_frames))
        target_depth_hz = target_camera_hz / depth_every if self.config.record_depth else 0.0
        roles = sorted(
            set(self.camera_metadata.keys())
            | set(self.video_frame_perf_stats_by_role.keys())
            | set(self.depth_frame_perf_stats_by_role.keys())
        )
        stream_status = self.stream_hub.status()
        subscriber_drops = self._camera_subscriber_drops_since_start(stream_status)
        cameras: dict[str, Any] = {}
        for role in roles:
            video_stats = self.video_frame_perf_stats_by_role.get(role) or OnlineTimeSeriesStats()
            depth_stats = self.depth_frame_perf_stats_by_role.get(role) or OnlineTimeSeriesStats()
            write_stats = self.camera_write_duration_stats_by_role.get(role) or OnlineNumericStats()
            latency_stats = self.camera_capture_to_write_latency_stats_by_role.get(role) or OnlineNumericStats()
            written = int(video_stats.count)
            session_drops = int(self.camera_queue_drop_counts.get(role, 0))
            hub_drops = int(subscriber_drops.get(role, 0))
            drops = session_drops + hub_drops
            cameras[role] = {
                "video": video_stats.summary(target_camera_hz),
                "depth": depth_stats.summary(target_depth_hz) if self.config.record_depth else {"count": 0},
                "writeDurationSeconds": write_stats.summary(),
                "captureToWriteLatencySeconds": latency_stats.summary(),
                "queueDrops": drops,
                "sessionQueueDrops": session_drops,
                "hubSubscriberDrops": hub_drops,
                "queueDropRatio": float(drops / max(1, written + drops)),
                "videoFramesWritten": written,
                "depthFramesWritten": int(depth_stats.count),
            }
        robot_state_summary = self.robot_state_perf_stats.summary(float(self.config.robot_state_hz))
        robot_state_target_summary = self.robot_state_target_perf_stats.summary(float(self.config.robot_state_hz))
        aligned_sample_summary = self.aligned_sample_perf_stats.summary(ASYNC_QUEST_ALIGNED_SAMPLE_HZ)
        aligned_target_summary = self.aligned_sample_target_perf_stats.summary(ASYNC_QUEST_ALIGNED_SAMPLE_HZ)
        robot_quality = time_series_rate_quality(robot_state_summary, float(self.config.robot_state_hz))
        aligned_quality = time_series_rate_quality(aligned_sample_summary, ASYNC_QUEST_ALIGNED_SAMPLE_HZ)
        robot_lateness_summary = self.robot_state_lateness_stats.summary()
        aligned_lateness_summary = self.aligned_sample_lateness_stats.summary()
        robot_lateness_p95 = robot_lateness_summary.get("p95")
        aligned_lateness_p95 = aligned_lateness_summary.get("p95")
        robot_quality["p95LatenessSeconds"] = robot_lateness_p95
        robot_quality["missedTicks"] = int(self.robot_state_missed_tick_count)
        robot_quality["ok"] = bool(
            robot_quality.get("ok")
            and is_number(robot_lateness_p95)
            and float(robot_lateness_p95) <= 0.006
            and self.robot_state_missed_tick_count == 0
        )
        aligned_quality["p95LatenessSeconds"] = aligned_lateness_p95
        aligned_quality["ok"] = bool(
            aligned_quality.get("ok")
            and is_number(aligned_lateness_p95)
            and float(aligned_lateness_p95) <= 0.016
        )
        camera_quality = {
            role: {
                **time_series_rate_quality(payload.get("video") or {}, target_camera_hz),
                "queueDrops": int(payload.get("queueDrops") or 0),
                "queueDropRatio": float(payload.get("queueDropRatio") or 0.0),
            }
            for role, payload in cameras.items()
        }
        for payload in camera_quality.values():
            payload["ok"] = bool(payload.get("ok") and int(payload.get("queueDrops") or 0) == 0)
        quality_checks = [robot_quality, aligned_quality, *camera_quality.values()]
        return {
            "recordingDurationSeconds": elapsed,
            "robotState": robot_state_summary,
            "robotStateTargets": robot_state_target_summary,
            "robotStateLatenessSeconds": robot_lateness_summary,
            "robotStateMissedTicks": int(self.robot_state_missed_tick_count),
            "alignedSamples": aligned_sample_summary,
            "alignedSampleTargets": aligned_target_summary,
            "alignedSampleLatenessSeconds": aligned_lateness_summary,
            "alignedQuestSourceAgeSeconds": self.aligned_quest_source_age_stats.summary(),
            "alignedReusedSourceSamples": int(self.aligned_reused_source_count),
            "targetAlignedSampleHz": ASYNC_QUEST_ALIGNED_SAMPLE_HZ,
            "cameras": cameras,
            "targetCameraHz": target_camera_hz,
            "recordDepth": bool(self.config.record_depth),
            "recordDepthFormat": normalize_depth_format(self.config.record_depth_format),
            "depthStreamEncodings": dict(self.depth_stream_encodings),
            "recordDepthEveryNFrames": depth_every,
            "targetDepthHz": target_depth_hz,
            "quality": {
                "ok": bool(quality_checks and all(bool(item.get("ok")) for item in quality_checks)),
                "robotState": robot_quality,
                "alignedSamples": aligned_quality,
                "cameras": camera_quality,
                "criteria": "effective rate >= 95% of target, p95 gap <= 1.5x target period, robot p95 lateness <= 6ms with zero missed ticks, aligned p95 lateness <= 16ms, and zero camera queue drops",
            },
        }

    def _camera_subscriber_drops_since_start(
        self,
        stream_status: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        if stream_status is None:
            stream_status = self.stream_hub.status()
        raw_counts = (
            stream_status.get("subscriberDropCounts")
            if isinstance(stream_status, dict) and isinstance(stream_status.get("subscriberDropCounts"), dict)
            else {}
        )
        roles = set(raw_counts.keys()) | set(self.camera_subscriber_drop_start_counts.keys()) | set(self.camera_metadata.keys())
        result: dict[str, int] = {}
        for role in roles:
            current = int(raw_counts.get(role, 0)) if is_number(raw_counts.get(role, 0)) else 0
            start = (
                int(self.camera_subscriber_drop_start_counts.get(role, 0))
                if is_number(self.camera_subscriber_drop_start_counts.get(role, 0))
                else 0
            )
            result[str(role)] = max(0, current - start)
        return result

    def _capture_row(self, quest_sample: dict[str, Any]) -> dict[str, Any]:
        sample_index = self.sample_count
        row = self._build_robot_state_row(sample_index=sample_index)
        videos: dict[str, dict[str, Any]] = {}
        video_frames: dict[str, dict[str, Any]] = {}
        frames = self.stream_hub.snapshot_roles(sorted(self.camera_metadata.keys()), wait_timeout=2.0)
        for role, frame in frames.items():
            serial = frame.get("serial") or self.camera_serials().get(role) or role
            frame_row = self._write_video_frame(role, serial, frame, sample_index=sample_index)
            videos[role] = {
                "path": frame_row.get("video"),
                "frameIndex": frame_row.get("frame_index"),
                "serial": serial,
            }
            video_frames[role] = {
                "role": role,
                "serial": serial,
                "frameIndex": frame_row.get("frame_index"),
                "capturedAtUtc": frame_row.get("frame_captured_at_utc"),
                "streamSequence": frame_row.get("stream_sequence"),
            }
        quest_gaze3d_pc_world = unity_vec3_to_pc(quest_sample.get("gazePoint3DWorld"))
        right_controller_pc = controller_payload_to_pc_world(quest_sample.get("rightController"))
        row.update(
            {
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
                "quest_pc_receive_perf_counter_seconds": quest_sample.get("pcReceivePerfCounterSeconds"),
                "quest_gaze3d_world": quest_sample.get("gazePoint3DWorld"),
                "quest_gaze3d_pc_world": quest_gaze3d_pc_world,
                "quest_gaze3d_source": quest_sample.get("gazePoint3DSource") or quest_sample.get("gazeSource"),
                "right_controller": visualizer_controller_payload(right_controller_pc),
                "right_controller_unity": visualizer_controller_payload(quest_sample.get("rightController")),
                "images": {},
                "videos": videos,
                "videoFrames": video_frames,
            }
        )
        return row

    def _build_robot_state_row(
        self,
        *,
        sample_index: int | None = None,
        captured_at: str | None = None,
        pc_perf_counter_seconds: float | None = None,
        pc_unix_seconds: float | None = None,
        lightweight: bool = False,
        derived_transforms: bool = True,
    ) -> dict[str, Any]:
        captured_at = captured_at or datetime.now(timezone.utc).isoformat()
        pc_perf_counter_seconds = (
            time.perf_counter() if pc_perf_counter_seconds is None else float(pc_perf_counter_seconds)
        )
        pc_unix_seconds = time.time() if pc_unix_seconds is None else float(pc_unix_seconds)
        index = self.robot_state_count if sample_index is None else int(sample_index)
        robot_state = self.robot.read_record_state(
            self.config.controller_joint_limit_buffer_rad,
            self.config.controller_joint_limit_guard_enabled,
            lightweight=lightweight,
        )
        transform_encoder = transform_to_compact_json if lightweight else transform_to_json
        t_base_ee = None
        t_base_tool_tcp = None
        t_base_end_camera_payload = None
        t_world_tool_tcp_payload = None
        t_world_end_camera_payload = None
        t_display_tool_tcp_payload = None
        t_display_end_camera_payload = None
        if derived_transforms:
            t_base_ee = transform_from_json(robot_state.get("endEffectorPose"))
            if t_base_ee is None:
                t_base_ee = robot_state_pose_transform(robot_state)
            t_base_tool_tcp = self._tool_tcp_transform(robot_state, t_base_ee)
            if t_base_tool_tcp is not None:
                if self.t_ee_end_camera is not None:
                    t_base_end_camera = t_base_tool_tcp @ self.t_ee_end_camera
                    t_base_end_camera_payload = transform_encoder(t_base_end_camera)
                if self.t_base_world is not None:
                    t_world_tool_tcp = invert_transform(self.t_base_world) @ t_base_tool_tcp
                    t_world_tool_tcp_payload = transform_encoder(t_world_tool_tcp, PC_WORLD_FRAME)
                    t_display_tool_tcp_payload = transform_encoder(t_world_tool_tcp, PC_WORLD_FRAME)
                    if t_base_end_camera_payload is not None:
                        t_base_end_camera_matrix = transform_from_json(t_base_end_camera_payload)
                        if t_base_end_camera_matrix is not None:
                            t_world_end_camera = invert_transform(self.t_base_world) @ t_base_end_camera_matrix
                            t_world_end_camera_payload = transform_encoder(t_world_end_camera, PC_WORLD_FRAME)
                            t_display_end_camera_payload = transform_encoder(t_world_end_camera, PC_WORLD_FRAME)
        joint_pose = robot_state.get("jointPose")
        t_base_ee_payload = transform_encoder(t_base_ee) if t_base_ee is not None else None
        t_base_tool_tcp_payload = transform_encoder(t_base_tool_tcp) if t_base_tool_tcp is not None else None
        return {
            "schema": "robot_state_light_v1" if lightweight else "robot_state_full_v1",
            "sample_index": index,
            "record_id": self.record_id,
            "captured_at": captured_at,
            "pc_perf_counter_seconds": pc_perf_counter_seconds,
            "pc_unix_seconds": pc_unix_seconds,
            "ok": True,
            "coordinate_frame": PC_WORLD_FRAME,
            "pose_source": f"flexiv:{self.config.robot_pose_field}:{self.config.robot_sn}",
            "robot_state": compact_robot_state_for_record(robot_state),
            "T_base_ee": t_base_ee_payload,
            "T_base_tool_tcp": t_base_tool_tcp_payload,
            "T_world_tool_tcp": t_world_tool_tcp_payload,
            "T_display_tool_tcp": t_display_tool_tcp_payload,
            "T_base_end_camera": t_base_end_camera_payload,
            "T_world_end_camera": t_world_end_camera_payload,
            "T_display_end_camera": t_display_end_camera_payload,
            "jointpose": joint_pose,
            "jointpos": joint_pose,
            "gripper": robot_state.get("gripper"),
        }

    def _compose_async_sample_row(self, quest_sample: dict[str, Any], *, sample_index: int) -> dict[str, Any]:
        captured_at = datetime.now(timezone.utc).isoformat()
        pc_perf_counter_seconds = time.perf_counter()
        pc_unix_seconds = time.time()
        with self.lock:
            state_row = self.latest_robot_state_row
            latest_video_frame_rows = {
                role: dict(frame_row)
                for role, frame_row in self.latest_video_frame_rows.items()
                if isinstance(frame_row, dict)
            }
        if not isinstance(state_row, dict):
            state_row = self._build_robot_state_row(
                sample_index=self.robot_state_count,
                captured_at=captured_at,
                pc_perf_counter_seconds=pc_perf_counter_seconds,
                pc_unix_seconds=pc_unix_seconds,
                lightweight=True,
            )
        videos: dict[str, dict[str, Any]] = {}
        video_frames: dict[str, dict[str, Any]] = {}
        for role, frame_row in latest_video_frame_rows.items():
            frame_perf = frame_row.get("capturedPerfCounterSeconds")
            videos[role] = {
                "path": frame_row.get("video"),
                "frameIndex": frame_row.get("frameIndex"),
                "serial": frame_row.get("serial"),
            }
            video_frames[role] = {
                "role": role,
                "serial": frame_row.get("serial"),
                "frameIndex": frame_row.get("frameIndex"),
                "capturedAtUtc": frame_row.get("capturedAtUtc"),
                "capturedPerfCounterSeconds": frame_perf,
                "writtenPerfCounterSeconds": frame_row.get("pcPerfCounterSeconds"),
                "ageSeconds": (
                    max(0.0, pc_perf_counter_seconds - float(frame_perf))
                    if is_number(frame_perf)
                    else None
                ),
                "streamSequence": frame_row.get("streamSequence"),
            }
        quest_gaze3d_pc_world = unity_vec3_to_pc(quest_sample.get("gazePoint3DWorld"))
        right_controller_pc = controller_payload_to_pc_world(quest_sample.get("rightController"))
        return {
            "sample_index": sample_index,
            "record_id": self.record_id,
            "captured_at": captured_at,
            "pc_perf_counter_seconds": pc_perf_counter_seconds,
            "pc_unix_seconds": pc_unix_seconds,
            "ok": True,
            "coordinate_frame": PC_WORLD_FRAME,
            "pose_source": state_row.get("pose_source") or f"flexiv:{self.config.robot_pose_field}:{self.config.robot_sn}",
            "quest_sample_index": quest_sample.get("sampleIndex"),
            "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
            "quest_pc_receive_perf_counter_seconds": quest_sample.get("pcReceivePerfCounterSeconds"),
            "aligned_target_perf_counter_seconds": quest_sample.get("alignedTargetPerfCounterSeconds"),
            "aligned_actual_perf_counter_seconds": quest_sample.get("alignedActualPerfCounterSeconds"),
            "aligned_lateness_seconds": quest_sample.get("alignedLatenessSeconds"),
            "aligned_source_version": quest_sample.get("alignedSourceVersion"),
            "aligned_source_reused": bool(quest_sample.get("alignedSourceReused")),
            "quest_gaze3d_world": quest_sample.get("gazePoint3DWorld"),
            "quest_gaze3d_pc_world": quest_gaze3d_pc_world,
            "quest_gaze3d_source": quest_sample.get("gazePoint3DSource") or quest_sample.get("gazeSource"),
            "right_controller": visualizer_controller_payload(right_controller_pc),
            "right_controller_unity": visualizer_controller_payload(quest_sample.get("rightController")),
            "robot_state_sample_index": state_row.get("sample_index"),
            "robot_state_captured_at": state_row.get("captured_at"),
            "robot_state_pc_perf_counter_seconds": state_row.get("pc_perf_counter_seconds"),
            "robot_state_age_seconds": (
                max(0.0, pc_perf_counter_seconds - float(state_row.get("pc_perf_counter_seconds")))
                if is_number(state_row.get("pc_perf_counter_seconds"))
                else None
            ),
            "images": {},
            "videos": videos,
            "videoFrames": video_frames,
        }

    def _robot_state_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None:
            return
        period = 1.0 / max(1.0, float(self.config.robot_state_hz))
        next_target_perf = time.perf_counter()
        while not stop_event.is_set():
            now_perf = time.perf_counter()
            if now_perf < next_target_perf and stop_event.wait(next_target_perf - now_perf):
                return
            started = time.perf_counter()
            lateness_seconds = max(0.0, started - next_target_perf)
            try:
                row = self._build_robot_state_row(
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    pc_perf_counter_seconds=started,
                    pc_unix_seconds=time.time(),
                    lightweight=True,
                    derived_transforms=True,
                )
                row["target_perf_counter_seconds"] = next_target_perf
                row["lateness_seconds"] = lateness_seconds
                with self.lock:
                    if self.closed or self.robot_states_handle is None:
                        return
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                with self.lock:
                    if self.closed or self.robot_states_handle is None:
                        return
                    self.robot_states_handle.write(line)
                    self._note_text_write_locked("robot_states")
                    self.robot_state_count += 1
                    self.latest_robot_state_row = row
                    self.robot_state_perf_stats.add(row.get("pc_perf_counter_seconds") or time.perf_counter())
                    self.robot_state_target_perf_stats.add(next_target_perf)
                    self.robot_state_lateness_stats.add(lateness_seconds)
            except Exception as exc:  # pragma: no cover - hardware path
                with self.lock:
                    self.robot_state_error_count += 1
                    self.last_robot_state_error = str(exc)
                    self.last_error = str(exc)
            next_target_perf += period
            overdue_seconds = time.perf_counter() - next_target_perf
            if overdue_seconds >= period:
                missed_ticks = int(overdue_seconds // period)
                next_target_perf += missed_ticks * period
                with self.lock:
                    self.robot_state_missed_tick_count += missed_ticks

    def _camera_record_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None:
            return
        roles = sorted(self.camera_metadata.keys())
        while not stop_event.is_set():
            wrote_any = False
            for role in roles:
                if stop_event.is_set():
                    return
                try:
                    frame = self.stream_hub.get_latest(
                        role,
                        wait_timeout=0.2,
                        copy_arrays=False,
                        include_depth=bool(self.config.record_depth),
                    )
                    sequence = frame.get("sequence")
                    with self.lock:
                        if self.closed:
                            return
                        if self.latest_video_sequences.get(role) == sequence:
                            continue
                        serial = frame.get("serial") or self.camera_serials().get(role) or role
                    frame_row = self._write_video_frame(role, serial, frame, sample_index=None)
                    with self.lock:
                        self.latest_video_sequences[role] = frame.get("sequence")
                        self.latest_video_frame_rows[role] = {
                            "role": role,
                            "serial": serial,
                            "frameIndex": frame_row.get("frame_index"),
                            "video": frame_row.get("video"),
                            "capturedAtUtc": frame.get("capturedAtUtc"),
                            "capturedPerfCounterSeconds": frame.get("capturedAtPerfCounterSeconds"),
                            "pcPerfCounterSeconds": frame_row.get("pc_perf_counter_seconds"),
                            "streamSequence": frame.get("sequence"),
                            "depth": frame_row.get("depth"),
                        }
                    wrote_any = True
                except Exception as exc:  # pragma: no cover - hardware path
                    if "No " in str(exc) and " frame is available yet" in str(exc):
                        stop_event.wait(0.01)
                        continue
                    with self.lock:
                        self.camera_record_error_count += 1
                        self.last_camera_record_error = str(exc)
                        self.last_error = str(exc)
                    stop_event.wait(0.02)
            if not wrote_any:
                stop_event.wait(0.005)

    def _camera_role_record_loop(self, role: str, frame_queue: queue.Queue[Any]) -> None:
        while True:
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                if self.closed:
                    return
                continue
            try:
                if frame is None:
                    return
                serial = frame.get("serial") or self.camera_serials().get(role) or role
                frame_row = self._write_video_frame(role, serial, frame, sample_index=None)
                with self.lock:
                    self.latest_video_sequences[role] = frame.get("sequence")
                    self.latest_video_frame_rows[role] = {
                        "role": role,
                        "serial": serial,
                        "frameIndex": frame_row.get("frame_index"),
                        "video": frame_row.get("video"),
                        "capturedAtUtc": frame.get("capturedAtUtc"),
                        "capturedPerfCounterSeconds": frame.get("capturedAtPerfCounterSeconds"),
                        "pcPerfCounterSeconds": frame_row.get("pc_perf_counter_seconds"),
                        "streamSequence": frame.get("sequence"),
                        "depth": frame_row.get("depth"),
                    }
            except Exception as exc:  # pragma: no cover - hardware path
                with self.lock:
                    self.camera_record_error_count += 1
                    self.last_camera_record_error = str(exc)
                    self.last_error = str(exc)
            finally:
                frame_queue.task_done()

    @staticmethod
    def _drain_camera_frame_queue(frame_queue: queue.Queue[Any]) -> int:
        dropped = 0
        while True:
            try:
                item = frame_queue.get_nowait()
            except queue.Empty:
                return dropped
            if item is not None:
                dropped += 1
            try:
                frame_queue.task_done()
            except ValueError:
                pass

    @staticmethod
    def _enqueue_camera_queue_sentinel(frame_queue: queue.Queue[Any]) -> int:
        dropped = 0
        while True:
            try:
                frame_queue.put_nowait(None)
                return dropped
            except queue.Full:
                try:
                    item = frame_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.001)
                    continue
                try:
                    if item is None:
                        return dropped
                    dropped += 1
                finally:
                    try:
                        frame_queue.task_done()
                    except ValueError:
                        pass

    def _remember_ee_pose(self, pose: np.ndarray) -> None:
        self.ee_pose_history.append(np.asarray(pose, dtype=float))
        max_history = 512 if self.record_mode == ROBOT_SESSION_RECORD_ASYNC else 2048
        if len(self.ee_pose_history) > max_history:
            del self.ee_pose_history[: len(self.ee_pose_history) - max_history]

    def _row_tool_or_ee_transform(self, row: dict[str, Any]) -> np.ndarray | None:
        return robot_row_tool_transform(row)

    def _write_video_frame(
        self,
        role: str,
        serial: str,
        frame: dict[str, Any],
        *,
        sample_index: int | None,
    ) -> dict[str, Any]:
        write_started_perf = time.perf_counter()
        bgr = frame.get("bgr")
        if not isinstance(bgr, np.ndarray):
            rgb = frame.get("rgb")
            if isinstance(rgb, np.ndarray):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not isinstance(bgr, np.ndarray):
            raise RuntimeError(f"{role} frame has no BGR image")
        with self._video_role_lock(role):
            writer = self.video_writers.get(role)
            if writer is None:
                height, width = int(bgr.shape[0]), int(bgr.shape[1])
                rel_path = Path("videos") / f"{role}_{safe_filename(serial)}.mp4"
                path = self.directory / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                if self.record_mode == ROBOT_SESSION_RECORD_ASYNC:
                    fps = max(1.0, float(self.config.fps))
                else:
                    fps = max(1.0, 1.0 / max(1e-6, float(self.config.capture_interval_seconds)))
                writer = ColorStreamWriter(path, width=width, height=height, fps=fps)
                self.video_writers[role] = writer
                self.video_codecs[role] = writer.codec
                self.video_paths[role] = str(rel_path).replace("\\", "/")
                self.video_frame_counts[role] = 0
            frame_index = int(self.video_frame_counts.get(role, 0))
            writer.write(bgr)
            self.video_frame_counts[role] = frame_index + 1
            depth_info = self._write_depth_frame(role, serial, frame, frame_index)
            write_finished_perf = time.perf_counter()
            row = {
                "sample_index": sample_index,
                "record_id": self.record_id,
                "role": role,
                "serial": serial,
                "frame_index": frame_index,
                "video": self.video_paths.get(role),
                "video_codec": self.video_codecs.get(role),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "pc_perf_counter_seconds": write_finished_perf,
                "write_duration_seconds": float(write_finished_perf - write_started_perf),
                "frame_captured_at_utc": frame.get("capturedAtUtc"),
                "frame_captured_perf_counter_seconds": frame.get("capturedAtPerfCounterSeconds"),
                "stream_sequence": frame.get("sequence"),
                "depth": depth_info,
            }
        with self.video_io_lock:
            self.video_frame_count += 1
            captured_perf = frame.get("capturedAtPerfCounterSeconds")
            frame_perf = float(captured_perf) if is_number(captured_perf) else write_finished_perf
            self.video_frame_perf_stats_by_role.setdefault(role, OnlineTimeSeriesStats()).add(frame_perf)
            self.camera_write_duration_stats_by_role.setdefault(role, OnlineNumericStats()).add(
                float(write_finished_perf - write_started_perf)
            )
            if is_number(captured_perf):
                self.camera_capture_to_write_latency_stats_by_role.setdefault(role, OnlineNumericStats()).add(
                    float(write_finished_perf - float(captured_perf))
                )
            if depth_info is not None:
                self.depth_frame_perf_stats_by_role.setdefault(role, OnlineTimeSeriesStats()).add(frame_perf)
            if self.video_frames_handle is not None:
                self.video_frames_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._note_video_write_locked()
        return row

    def _video_role_lock(self, role: str) -> threading.RLock:
        role_name = str(role or "").strip().lower() or "camera"
        with self.lock:
            lock = self.video_role_locks.get(role_name)
            if lock is None:
                lock = threading.RLock()
                self.video_role_locks[role_name] = lock
            return lock

    def _close_color_video_writer(self, role: str, writer: ColorStreamWriter | None) -> None:
        if writer is None:
            return
        try:
            writer.close()
        except Exception as exc:
            self._note_close_error("video_stream_close", f"{role}: {exc}")
        self.video_frame_counts[role] = int(
            getattr(writer, "frame_count", self.video_frame_counts.get(role, 0)) or 0
        )

    def _close_depth_stream(self, role: str) -> None:
        writer = self.depth_stream_handles.pop(role, None)
        if writer is None:
            return
        try:
            writer.close()
        except Exception as exc:
            self._note_close_error("depth_stream_close", f"{role}: {exc}")
        self.depth_stream_frame_counts[role] = int(writer.frame_count)

    def _depth_stream_handle(
        self,
        role: str,
        serial: str,
        depth: np.ndarray,
        *,
        force_raw: bool = False,
    ) -> tuple[DepthStreamWriter, str]:
        handle = self.depth_stream_handles.get(role)
        rel_path = self.depth_stream_paths.get(role)
        if handle is not None and isinstance(rel_path, str) and rel_path:
            if handle.width != int(depth.shape[1]) or handle.height != int(depth.shape[0]):
                raise ValueError(
                    f"{role} depth dimensions changed from {handle.width}x{handle.height} "
                    f"to {int(depth.shape[1])}x{int(depth.shape[0])}"
                )
            return handle, rel_path
        requested = "raw" if force_raw else normalize_depth_format(self.config.record_depth_format)
        depth_every = max(1, int(self.config.record_depth_every_n_frames))
        depth_fps = float(self.config.fps) / float(depth_every)
        if requested == "ffv1":
            rel = Path("depth") / f"{role}_{safe_filename(serial)}.ffv1.mkv"
            path = self.directory / rel
            try:
                handle = DepthStreamWriter(
                    path,
                    width=int(depth.shape[1]),
                    height=int(depth.shape[0]),
                    fps=depth_fps,
                    encoding="ffv1",
                )
            except Exception as exc:
                self._note_depth_warning("depth_ffv1_open_fallback", role, exc)
                requested = "raw"
        if requested != "ffv1":
            rel = Path("depth") / f"{role}_{safe_filename(serial)}.u16le.bin"
            path = self.directory / rel
            handle = DepthStreamWriter(
                path,
                width=int(depth.shape[1]),
                height=int(depth.shape[0]),
                fps=depth_fps,
                encoding="raw",
            )
        rel_path = str(rel).replace("\\", "/")
        self.depth_stream_handles[role] = handle
        self.depth_stream_paths[role] = rel_path
        self.depth_stream_encodings[role] = handle.encoding
        return handle, rel_path

    def _write_depth_frame(
        self,
        role: str,
        serial: str,
        frame: dict[str, Any],
        frame_index: int,
    ) -> dict[str, Any] | None:
        depth = frame.get("depth")
        if not isinstance(depth, np.ndarray):
            return None
        if role in self.depth_stream_disabled_roles:
            return None
        every_n = max(1, int(self.config.record_depth_every_n_frames))
        if frame.get("depthCaptureIndex") is None and every_n > 1 and int(frame_index) % every_n != 0:
            return None
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16, copy=False)
        if not depth.flags.c_contiguous:
            depth = np.ascontiguousarray(depth)
        metadata = frame.get("metadata") if isinstance(frame.get("metadata"), dict) else {}
        handle, rel_path = self._depth_stream_handle(role, serial, depth)
        try:
            stream_info = handle.write(depth)
        except Exception as exc:
            if handle.encoding == "ffv1" and handle.frame_count == 0:
                self._note_depth_warning("depth_ffv1_first_write_fallback", role, exc)
                self.depth_stream_handles.pop(role, None)
                self.depth_stream_paths.pop(role, None)
                self.depth_stream_encodings.pop(role, None)
                try:
                    handle.close()
                except Exception:
                    pass
                for candidate in (handle.path, handle.stderr_path):
                    if candidate is not None:
                        try:
                            if candidate.exists():
                                candidate.unlink()
                        except Exception:
                            pass
                handle, rel_path = self._depth_stream_handle(role, serial, depth, force_raw=True)
                stream_info = handle.write(depth)
            else:
                self.depth_stream_disabled_roles[role] = str(exc)
                self._note_close_error("depth_stream_write_failed", f"{role}: {exc}")
                self._close_depth_stream(role)
                return None
        self.depth_stream_frame_counts[role] = int(handle.frame_count)
        self.depth_count += 1
        depth_aligned = bool(metadata.get("depthAlignedToColor"))
        encoding = (
            "uint16_ffv1_mkv_aligned_to_color"
            if handle.encoding == "ffv1" and depth_aligned
            else "uint16_ffv1_mkv_camera_native"
            if handle.encoding == "ffv1"
            else "uint16_raw_aligned_to_color"
            if depth_aligned
            else "uint16_raw_camera_native"
        )
        return {
            "path": rel_path,
            "encoding": encoding,
            "container": "matroska" if handle.encoding == "ffv1" else "raw",
            "codec": "ffv1" if handle.encoding == "ffv1" else None,
            "dtype": "uint16",
            "endianness": "little",
            "layout": "row_major",
            "alignedToColor": depth_aligned,
            "streamFrameIndex": stream_info.get("streamFrameIndex"),
            "byteOffset": stream_info.get("byteOffset"),
            "byteLength": stream_info.get("byteLength"),
            "sourceByteOffset": stream_info.get("sourceByteOffset"),
            "sourceByteLength": stream_info.get("sourceByteLength"),
            "captureIndex": frame.get("depthCaptureIndex"),
            "depthScaleM": metadata.get("depthScaleM"),
            "width": int(depth.shape[1]),
            "height": int(depth.shape[0]),
            "strideBytes": int(depth.shape[1]) * int(depth.dtype.itemsize),
        }

    def _tool_tcp_transform(self, robot_state: dict[str, Any], fallback: np.ndarray | None) -> np.ndarray | None:
        tcp = robot_state.get("tcp_pose")
        if isinstance(tcp, dict):
            matrix = transform_from_json(tcp.get("T_base_pose"))
            if matrix is not None:
                return matrix
            matrix = flexiv_pose_payload_to_transform(tcp.get("pose"))
            if matrix is not None:
                return matrix
        if self.config.robot_pose_field == "tcp_pose" and fallback is not None:
            return fallback
        return fallback

    def _publish(self, event: dict[str, Any]) -> None:
        if self.publish_event is not None:
            self.publish_event(event)

    def _note_close_error(self, stage: str, error: Any) -> None:
        payload = {
            "stage": stage,
            "error": str(error),
        }
        self.close_errors.append(payload)
        self.last_error = payload["error"]

    def _note_depth_warning(self, stage: str, role: str, message: Any) -> None:
        self.depth_stream_warnings.append(
            {
                "stage": stage,
                "role": str(role),
                "message": str(message),
                "capturedAt": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _flush_text_handles_locked(self) -> None:
        for handle in (self.samples_handle, self.robot_states_handle, self.motion_handle, self.gripper_handle):
            if handle is not None:
                handle.flush()
        for key in self.text_pending_rows:
            self.text_pending_rows[key] = 0
        self.text_last_flush_perf = time.perf_counter()

    def _note_text_write_locked(self, key: str) -> None:
        if self.record_mode != ROBOT_SESSION_RECORD_ASYNC:
            self._flush_text_handles_locked()
            return
        self.text_pending_rows[key] = int(self.text_pending_rows.get(key, 0)) + 1
        now_perf = time.perf_counter()
        if (
            self.text_pending_rows[key] >= ASYNC_JSONL_FLUSH_ROWS
            or (now_perf - self.text_last_flush_perf) >= ASYNC_JSONL_FLUSH_INTERVAL_SECONDS
        ):
            self._flush_text_handles_locked()

    def _flush_video_handles_locked(self) -> None:
        if self.video_frames_handle is not None:
            self.video_frames_handle.flush()
        self.video_pending_rows = 0
        self.video_last_flush_perf = time.perf_counter()

    def _note_video_write_locked(self) -> None:
        if self.record_mode != ROBOT_SESSION_RECORD_ASYNC:
            self._flush_video_handles_locked()
            return
        self.video_pending_rows += 1
        now_perf = time.perf_counter()
        if (
            self.video_pending_rows >= ASYNC_JSONL_FLUSH_ROWS
            or (now_perf - self.video_last_flush_perf) >= ASYNC_JSONL_FLUSH_INTERVAL_SECONDS
        ):
            self._flush_video_handles_locked()

    def _try_initialize_gripper(self) -> None:
        try:
            if self.record_mode == ROBOT_SESSION_RECORD_ASYNC and not self.robot.gripper_enabled:
                status = self.robot.gripper_status()
                self.gripper_available = False
                event = {
                    "ok": False,
                    "commandSent": False,
                    "record_id": self.record_id,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "action": "initialize",
                    "reason": "gripper_not_preinitialized",
                    "status": status,
                }
                self.last_gripper_event = event
                if self.gripper_handle is not None:
                    self.gripper_handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                    self.gripper_handle.flush()
                self._publish(robot_gripper_event(event))
                return
            status = self.robot.enable_gripper(
                self.config.gripper_device,
                init_on_enable=self.config.gripper_init_on_enable,
            )
            self.gripper_available = bool(status.get("enabled", True)) if isinstance(status, dict) else True
            event = {
                "ok": self.gripper_available,
                "commandSent": False,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "action": "initialize",
                "device": status.get("device") if isinstance(status, dict) else None,
                "status": status,
            }
        except Exception as exc:  # pragma: no cover - hardware path
            gripper_status = self.robot.gripper_status()
            device_status = self.robot.status().get("devices") if self.robot is not None else None
            self.gripper_available = False
            self.gripper_error_count += 1
            event = {
                "ok": False,
                "commandSent": False,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "action": "initialize",
                "error": str(exc),
                "status": gripper_status,
                "devices": device_status,
            }
        self.last_gripper_event = event
        if self.gripper_handle is not None:
            self.gripper_handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.gripper_handle.flush()
        self._publish(robot_gripper_event(event))

    def _update_controller_motion_unlocked(self, quest_sample: dict[str, Any]) -> dict[str, Any]:
        raw_controller = quest_sample.get("rightController")
        input_summary = controller_input_summary(raw_controller)
        controller = controller_payload_to_pc_world(raw_controller)
        if not isinstance(raw_controller, dict) or not raw_controller.get("hasPose"):
            self.reset_controller_motion_anchor()
            return {
                "ok": False,
                "reason": "missing_right_controller",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "anchored": self.controller_motion_anchor_ready(),
                "right_controller_input": input_summary,
                "teleopHeld": bool(input_summary.get("teleopHeld")),
                "teleopHoldValue": input_summary.get("handTrigger"),
            }
        side_pressed, side_value = right_controller_side_button_pressed(raw_controller)
        if not side_pressed:
            self.reset_controller_motion_anchor()
            return {
                "ok": False,
                "reason": "right_hand_trigger_not_held",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
                "anchored": self.controller_motion_anchor_ready(),
                "teleopHeld": False,
                "teleopHoldValue": side_value,
                "right_controller_input": input_summary,
            }
        now_perf = time.perf_counter()
        min_period = 1.0 / max(1.0, float(self.config.controller_target_update_hz))
        if now_perf < self.next_controller_motion_target_perf:
            return {
                "ok": False,
                "reason": "rate_limited",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
                "anchored": self.controller_motion_anchor_ready(),
                "right_controller_input": input_summary,
                "teleopHeld": True,
                "teleopHoldValue": side_value,
                "targetUpdateHz": float(self.config.controller_target_update_hz),
            }
        self.next_controller_motion_target_perf = now_perf + min_period
        if not self.robot.motion_armed:
            self.reset_controller_motion_anchor()
            return {
                "ok": False,
                "reason": "motion_not_armed",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
                "anchored": self.controller_motion_anchor_ready(),
                "right_controller_input": input_summary,
                "teleopHeld": True,
                "teleopHoldValue": side_value,
            }
        position = vec3_array(controller.get("position") if isinstance(controller, dict) else None)
        if position is None:
            self.reset_controller_motion_anchor()
            return {
                "ok": False,
                "reason": "missing_controller_position",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "anchored": self.controller_motion_anchor_ready(),
                "right_controller_input": input_summary,
                "teleopHeld": True,
                "teleopHoldValue": side_value,
            }
        rotation_world = quat_rotation_matrix(controller.get("rotation") if isinstance(controller, dict) else None)
        if rotation_world is None:
            self.reset_controller_motion_anchor()
            return {
                "ok": False,
                "reason": "missing_controller_rotation",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "anchored": self.controller_motion_anchor_ready(),
                "right_controller_input": input_summary,
                "teleopHeld": True,
                "teleopHoldValue": side_value,
            }
        if self.require_controller_alignment and self.t_base_world is None:
            return {
                "ok": False,
                "reason": "missing_quest_robot_alignment",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "anchored": self.controller_motion_anchor_ready(),
                "right_controller_input": input_summary,
                "teleopHeld": True,
                "teleopHoldValue": side_value,
            }
        created_anchor = not self.controller_motion_anchor_ready()
        if created_anchor:
            self.controller_anchor_world = position
            self.controller_anchor_rotation_world = rotation_world
            self.robot_anchor_tcp_pose = self.robot.read_tcp_pose()
        raw_offset_world = (position - self.controller_anchor_world) * float(self.config.controller_translation_scale)
        mapping_mode = "quest_alignment"
        raw_rotation_world = rotation_world @ self.controller_anchor_rotation_world.T
        if self.t_base_world is not None:
            mapping_rotation = self.t_base_world[:3, :3]
            raw_offset = mapping_rotation @ raw_offset_world
        else:
            mapping_rotation = QUEST_TO_ROBOT_UNALIGNED_ROTATION
            raw_offset = mapping_rotation @ raw_offset_world
            mapping_mode = "unaligned_pc_z_up_identity"
        raw_rotation = mapping_rotation @ raw_rotation_world @ mapping_rotation.T
        offset = raw_offset
        rotation_delta = raw_rotation
        target = list(self.robot_anchor_tcp_pose)
        target[:3] = [float(target[i] + offset[i]) for i in range(3)]
        anchor_rotation = quaternion_wxyz_to_matrix(target[3:7])
        target_rotation = rotation_delta @ anchor_rotation
        last = self.robot.motion_last_target_pose
        step_offset = np.zeros(3, dtype=float)
        step_rotation = np.eye(3, dtype=float)
        if last is not None:
            step = np.asarray(target[:3], dtype=float) - np.asarray(last[:3], dtype=float)
            step = clamp_vector_norm(step, float(self.config.controller_max_step_m))
            step_offset = step
            target[:3] = [float(last[i] + step[i]) for i in range(3)]
            last_rotation = quaternion_wxyz_to_matrix(last[3:7])
            step_rotation = target_rotation @ last_rotation.T
            step_rotation = clamp_rotation_angle(step_rotation, float(self.config.controller_max_rotation_step_deg))
            target_rotation = step_rotation @ last_rotation
        target[3:7] = matrix_to_quaternion_wxyz(target_rotation)
        joint_guard = self.robot.send_cartesian_target(
            target,
            self.config.controller_joint_limit_buffer_rad,
            self.config.controller_joint_limit_guard_enabled,
        )
        if not bool(joint_guard.get("ok")):
            self.reset_controller_motion_anchor()
            return {
                "ok": False,
                "reason": joint_guard.get("reason") or "joint_limit_guard",
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
                "teleopHeld": True,
                "teleopHoldValue": side_value,
                "teleop_held": True,
                "teleop_hold_value": side_value,
                "right_controller_input": input_summary,
                "controller_position_world": [float(v) for v in position],
                "controller_world_frame": PC_WORLD_FRAME,
                "controller_rotation_world_wxyz": matrix_to_quaternion_wxyz(rotation_world),
                "anchored": False,
                "created_anchor": created_anchor,
                "quest_alignment_used": self.t_base_world is not None,
                "motion_mapping": mapping_mode,
                "raw_offset_world_m": [float(v) for v in raw_offset_world],
                "raw_offset_m": [float(v) for v in raw_offset],
                "offset_m": [float(v) for v in offset],
                "step_offset_m": [float(v) for v in step_offset],
                "raw_rotation_angle_world_deg": rotation_angle_deg(raw_rotation_world),
                "raw_rotation_angle_deg": rotation_angle_deg(raw_rotation),
                "rotation_angle_deg": rotation_angle_deg(rotation_delta),
                "step_rotation_deg": rotation_angle_deg(step_rotation),
                "target_tcp_pose_wxyz": target,
                "joint_limit_guard": joint_guard,
                "commandSent": bool(joint_guard.get("commandSent")),
                "limits": {
                    "scale": self.config.controller_translation_scale,
                    "workspaceLimitEnabled": False,
                    "maxOffsetM": None,
                    "maxStepM": self.config.controller_max_step_m,
                    "maxRotationDeg": None,
                    "maxRotationStepDeg": self.config.controller_max_rotation_step_deg,
                    "targetUpdateHz": self.config.controller_target_update_hz,
                    "jointLimitBufferRad": self.config.controller_joint_limit_buffer_rad,
                    "jointLimitGuardEnabled": self.config.controller_joint_limit_guard_enabled,
                },
            }
        return {
            "ok": True,
            "record_id": self.record_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "quest_sample_index": quest_sample.get("sampleIndex"),
            "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
            "teleopHeld": True,
            "teleopHoldValue": side_value,
            "teleop_held": True,
            "teleop_hold_value": side_value,
            "right_controller_input": input_summary,
            "controller_position_world": [float(v) for v in position],
            "controller_world_frame": PC_WORLD_FRAME,
            "controller_rotation_world_wxyz": matrix_to_quaternion_wxyz(rotation_world),
            "controller_anchor_world": [float(v) for v in self.controller_anchor_world],
            "controller_anchor_rotation_world_wxyz": matrix_to_quaternion_wxyz(self.controller_anchor_rotation_world),
            "robot_anchor_tcp_pose_wxyz": [float(v) for v in self.robot_anchor_tcp_pose],
            "anchored": True,
            "created_anchor": created_anchor,
            "quest_alignment_used": self.t_base_world is not None,
            "motion_mapping": mapping_mode,
            "unaligned_rotation_robot_from_quest": QUEST_TO_ROBOT_UNALIGNED_ROTATION.tolist()
            if self.t_base_world is None
            else None,
            "raw_offset_world_m": [float(v) for v in raw_offset_world],
            "raw_offset_m": [float(v) for v in raw_offset],
            "offset_m": [float(v) for v in offset],
            "step_offset_m": [float(v) for v in step_offset],
            "raw_rotation_angle_world_deg": rotation_angle_deg(raw_rotation_world),
            "raw_rotation_angle_deg": rotation_angle_deg(raw_rotation),
            "rotation_angle_deg": rotation_angle_deg(rotation_delta),
            "step_rotation_deg": rotation_angle_deg(step_rotation),
            "target_tcp_pose_wxyz": target,
            "joint_limit_guard": joint_guard,
            "commandSent": bool(joint_guard.get("commandSent")),
            "reason": joint_guard.get("reason"),
            "limits": {
                "scale": self.config.controller_translation_scale,
                "workspaceLimitEnabled": False,
                "maxOffsetM": None,
                "maxStepM": self.config.controller_max_step_m,
                "maxRotationDeg": None,
                "maxRotationStepDeg": self.config.controller_max_rotation_step_deg,
                "targetUpdateHz": self.config.controller_target_update_hz,
                "jointLimitBufferRad": self.config.controller_joint_limit_buffer_rad,
                "jointLimitGuardEnabled": self.config.controller_joint_limit_guard_enabled,
            },
        }

    def controller_motion_anchor_ready(self) -> bool:
        return (
            self.controller_anchor_world is not None
            and self.controller_anchor_rotation_world is not None
            and self.robot_anchor_tcp_pose is not None
        )

    def reset_controller_motion_anchor(self) -> None:
        self.controller_anchor_world = None
        self.controller_anchor_rotation_world = None
        self.robot_anchor_tcp_pose = None
        self.next_controller_motion_target_perf = 0.0

    def _update_gripper_unlocked(self, quest_sample: dict[str, Any]) -> dict[str, Any]:
        controller = quest_sample.get("rightController")
        if not isinstance(controller, dict):
            return self._gripper_skip_event(quest_sample, "missing_right_controller")
        trigger = controller_trigger_value(controller)
        if trigger is None:
            return self._gripper_skip_event(quest_sample, "missing_trigger")
        close_threshold = float(self.config.gripper_trigger_close_threshold)
        open_threshold = float(self.config.gripper_trigger_open_threshold)
        desired_closed: bool | None = None
        if trigger >= close_threshold:
            desired_closed = True
        elif trigger <= open_threshold:
            desired_closed = False
        if desired_closed is None:
            return self._gripper_skip_event(quest_sample, "trigger_hysteresis", trigger)
        if self.last_gripper_closed is desired_closed:
            return self._gripper_skip_event(quest_sample, "unchanged", trigger)
        target_width = (
            float(self.config.gripper_close_width_m)
            if desired_closed
            else float(self.config.gripper_open_width_m)
        )
        try:
            status = self.robot.move_gripper(
                target_width,
                float(self.config.gripper_speed_mps),
                float(self.config.gripper_force_n),
            )
            self.last_gripper_closed = desired_closed
            return {
                "ok": True,
                "commandSent": True,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
                "trigger": trigger,
                "action": "close" if desired_closed else "open",
                "target_width_m": target_width,
                "speed_mps": float(self.config.gripper_speed_mps),
                "force_n": float(self.config.gripper_force_n),
                "status": status,
            }
        except Exception as exc:
            self.robot.gripper_last_error = str(exc)
            raise

    def _gripper_skip_event(
        self,
        quest_sample: dict[str, Any],
        reason: str,
        trigger: float | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "commandSent": False,
            "record_id": self.record_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "quest_sample_index": quest_sample.get("sampleIndex"),
            "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
            "trigger": trigger,
            "reason": reason,
        }

    def camera_serials(self) -> dict[str, str]:
        return {
            "end": str(self.config.camera_serial or "").strip(),
            "third": str(self.config.third_camera_serial or "").strip(),
        }

    def _alignment_transform(self, group: str, key: str) -> np.ndarray | None:
        if not isinstance(self.robot_alignment_result, dict):
            return None
        payload = self.robot_alignment_result.get(group)
        if not isinstance(payload, dict):
            return None
        return transform_from_json(payload.get(key))


class FlexivRealSenseManager:
    def __init__(
        self,
        config: FlexivRealSenseConfig | None = None,
        *,
        formal_control_mode: str = ROBOT_SESSION_CONTROL_TELEOP,
    ) -> None:
        self.config = config or FlexivRealSenseConfig()
        self.formal_control_mode = (
            formal_control_mode
            if formal_control_mode in (ROBOT_SESSION_CONTROL_TELEOP, ROBOT_SESSION_CONTROL_RECORD_ONLY)
            else ROBOT_SESSION_CONTROL_RECORD_ONLY
        )
        self.motion_commands_allowed = self.formal_control_mode != ROBOT_SESSION_CONTROL_RECORD_ONLY
        if not self.motion_commands_allowed:
            self.config.controller_motion_enabled = False
        self.robot = FlexivRobotClient(
            cartesian_max_linear_velocity_mps=self.config.cartesian_max_linear_velocity_mps,
            cartesian_max_angular_velocity_radps=self.config.cartesian_max_angular_velocity_radps,
            cartesian_max_linear_acceleration_mps2=self.config.cartesian_max_linear_acceleration_mps2,
            cartesian_max_angular_acceleration_radps2=self.config.cartesian_max_angular_acceleration_radps2,
        )
        self.lock = threading.RLock()
        self.stream_hub = RealSenseStreamHub()
        self.active_session: RobotRealsenseSession | None = None
        self.last_calibration: dict[str, Any] | None = None
        self.last_error: str | None = None

    def status(self, *, lightweight: bool = False) -> dict[str, Any]:
        with self.lock:
            active = self.active_session.summary("recording") if self.active_session is not None else None
            robot_status = self.robot.status(include_state=not lightweight, include_devices=not lightweight)
            state = robot_status.get("state") if isinstance(robot_status, dict) else None
            if not lightweight and isinstance(state, dict) and robot_status.get("connected"):
                state["jointLimitGuard"] = self.robot.joint_limit_guard(
                    self.config.controller_joint_limit_buffer_rad,
                    self.config.controller_joint_limit_guard_enabled,
                )
            return {
                "ok": True,
                "enabled": True,
                "formalControlMode": self.formal_control_mode,
                "motionCommandsAllowed": self.motion_commands_allowed,
                "config": config_to_json(self.config),
                "robot": robot_status,
                "activeSession": active,
                "realsenseStream": self.stream_hub.status(),
                "lastCalibration": self.last_calibration,
                "lastError": self.last_error,
            }

    def safety_status(self) -> dict[str, Any]:
        with self.lock:
            robot_status = self.robot.status(include_state=True, include_devices=False)
            state = robot_status.get("state") if isinstance(robot_status, dict) else None
            if isinstance(state, dict) and robot_status.get("connected"):
                state["jointLimitGuard"] = self.robot.joint_limit_guard(
                    self.config.controller_joint_limit_buffer_rad,
                    self.config.controller_joint_limit_guard_enabled,
                )
            active = None
            if self.active_session is not None:
                active = {
                    "recordId": self.active_session.record_id,
                    "controlMode": self.active_session.control_mode,
                    "closed": bool(self.active_session.closed),
                }
            return {
                "ok": True,
                "enabled": True,
                "formalControlMode": self.formal_control_mode,
                "motionCommandsAllowed": self.motion_commands_allowed,
                "robot": robot_status,
                "activeSession": active,
                "lastError": self.last_error,
            }

    def gripper_status(self) -> dict[str, Any]:
        with self.lock:
            robot_status = self.robot.status(include_state=False, include_devices=True)
            gripper = self.robot.gripper_status(include_params=True, include_states=True)
            if isinstance(robot_status, dict):
                robot_status["gripper"] = gripper
            return {
                "ok": True,
                "enabled": True,
                "config": config_to_json(self.config),
                "robotConnected": bool(isinstance(robot_status, dict) and robot_status.get("connected")),
                "robot": robot_status,
                "gripper": gripper,
                "devices": robot_status.get("devices") if isinstance(robot_status, dict) else None,
            }

    def move_gripper(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            action = str(payload.get("action") or "").strip().lower()
            if action == "open":
                width_m = float(payload.get("widthM", self.config.gripper_open_width_m))
            elif action == "close":
                width_m = float(payload.get("widthM", self.config.gripper_close_width_m))
            elif is_number(payload.get("widthM")):
                width_m = float(payload["widthM"])
                action = "move"
            else:
                raise ValueError("gripper move requires action=open|close or numeric widthM")
            speed_mps = float(payload.get("speedMps", self.config.gripper_speed_mps))
            force_n = float(payload.get("forceN", self.config.gripper_force_n))
            status = self.robot.move_gripper(width_m, speed_mps, force_n)
            return {
                "ok": True,
                "enabled": True,
                "commandSent": True,
                "action": action,
                "device": status.get("device"),
                "targetWidthM": width_m,
                "target_width_m": width_m,
                "speedMps": speed_mps,
                "forceN": force_n,
                "status": status,
                "config": config_to_json(self.config),
            }

    def _stream_config_snapshot(self) -> FlexivRealSenseConfig:
        with self.lock:
            source = self.active_session.config if self.active_session is not None else self.config
            return copy.deepcopy(source)

    def _ensure_stream_for_current_config(self) -> FlexivRealSenseConfig:
        config = self._stream_config_snapshot()
        self.stream_hub.start(config)
        return config

    def warm_realsense_stream(self, timeout_seconds: float = REALSENSE_CONNECT_WARMUP_SECONDS) -> dict[str, Any]:
        try:
            config = self._ensure_stream_for_current_config()
            status = self.stream_hub.status()
            roles = sorted(status.get("roles") or [])
            warmup = self.stream_hub.wait_for_roles(roles, timeout_seconds)
            return {
                "ok": bool(warmup.get("ok")),
                "timeoutSeconds": float(timeout_seconds),
                "roles": roles,
                "missingRoles": warmup.get("missingRoles") or [],
                "stream": warmup.get("stream") or status,
                "config": config_to_json(config),
            }
        except Exception as exc:  # pragma: no cover - hardware path
            return {
                "ok": False,
                "timeoutSeconds": float(timeout_seconds),
                "error": str(exc),
                "stream": self.stream_hub.status(),
            }

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if "robotSn" in payload:
                self.config.robot_sn = str(payload.get("robotSn") or "").strip()
            if payload.get("poseField") in ("flange_pose", "tcp_pose"):
                self.config.robot_pose_field = str(payload["poseField"])
            if "cameraSerial" in payload:
                self.config.camera_serial = str(payload.get("cameraSerial") or "").strip()
            if "thirdCameraSerial" in payload:
                self.config.third_camera_serial = str(payload.get("thirdCameraSerial") or "").strip()
            if (
                self.config.third_camera_serial
                and self.config.third_camera_serial == self.config.camera_serial
            ):
                self.config.third_camera_serial = ""
            if "networkInterfaces" in payload:
                self.config.flexiv_network_interfaces = normalize_network_interfaces(payload.get("networkInterfaces"))
            for key, attr in (
                ("width", "width"),
                ("height", "height"),
                ("fps", "fps"),
                ("warmupFrames", "warmup_frames"),
                ("minHandEyeDetections", "min_hand_eye_detections"),
                ("handEyeMaxDiverseSamples", "hand_eye_max_diverse_samples"),
                ("handEyeMinDiverseSamples", "hand_eye_min_diverse_samples"),
            ):
                if key in payload and is_number(payload[key]):
                    setattr(self.config, attr, int(payload[key]))
            if "robotStateHz" in payload and is_number(payload["robotStateHz"]):
                self.config.robot_state_hz = max(1.0, float(payload["robotStateHz"]))
            if "recordDepth" in payload:
                self.config.record_depth = bool(payload["recordDepth"])
            if "recordDepthEveryNFrames" in payload and is_number(payload["recordDepthEveryNFrames"]):
                self.config.record_depth_every_n_frames = max(1, int(payload["recordDepthEveryNFrames"]))
            if "recordDepthFormat" in payload:
                self.config.record_depth_format = normalize_depth_format(payload.get("recordDepthFormat"))
            if "captureIntervalSeconds" in payload and is_number(payload["captureIntervalSeconds"]):
                self.config.capture_interval_seconds = float(payload["captureIntervalSeconds"])
            if "realsenseAutoExposure" in payload:
                self.config.realsense_auto_exposure = bool(payload["realsenseAutoExposure"])
            if "realsenseExposure" in payload:
                self.config.realsense_exposure = (
                    float(payload["realsenseExposure"]) if is_number(payload["realsenseExposure"]) else None
                )
            if "realsenseGain" in payload:
                self.config.realsense_gain = (
                    float(payload["realsenseGain"]) if is_number(payload["realsenseGain"]) else None
                )
            if "boardCheckWarmupFrames" in payload and is_number(payload["boardCheckWarmupFrames"]):
                self.config.board_check_warmup_frames = max(1, int(payload["boardCheckWarmupFrames"]))
            if "runHandEye" in payload:
                self.config.run_hand_eye = bool(payload["runHandEye"])
            if "handEyeDisableDiverseSelection" in payload:
                self.config.hand_eye_disable_diverse_selection = bool(payload["handEyeDisableDiverseSelection"])
            for key, attr in (
                ("handEyeDiverseTranslationScaleM", "hand_eye_diverse_translation_scale_m"),
                ("handEyeDiverseRotationScaleDeg", "hand_eye_diverse_rotation_scale_deg"),
                ("handEyeDiverseMinScore", "hand_eye_diverse_min_score"),
            ):
                if key in payload and is_number(payload[key]):
                    setattr(self.config, attr, float(payload[key]))
            if "controllerMotionEnabled" in payload:
                self.config.controller_motion_enabled = bool(payload["controllerMotionEnabled"])
            if "controllerTranslationScale" in payload and is_number(payload["controllerTranslationScale"]):
                self.config.controller_translation_scale = max(
                    CONTROLLER_TRANSLATION_SCALE_MIN,
                    min(CONTROLLER_TRANSLATION_SCALE_MAX, float(payload["controllerTranslationScale"])),
                )
            if "controllerMaxStepM" in payload and is_number(payload["controllerMaxStepM"]):
                self.config.controller_max_step_m = float(payload["controllerMaxStepM"])
            if "controllerMaxRotationStepDeg" in payload and is_number(payload["controllerMaxRotationStepDeg"]):
                self.config.controller_max_rotation_step_deg = float(payload["controllerMaxRotationStepDeg"])
            if "controllerTargetUpdateHz" in payload and is_number(payload["controllerTargetUpdateHz"]):
                self.config.controller_target_update_hz = max(1.0, float(payload["controllerTargetUpdateHz"]))
            if "controllerJointLimitBufferRad" in payload and is_number(payload["controllerJointLimitBufferRad"]):
                self.config.controller_joint_limit_buffer_rad = max(0.0, float(payload["controllerJointLimitBufferRad"]))
            if "controllerJointLimitGuardEnabled" in payload:
                self.config.controller_joint_limit_guard_enabled = bool(payload["controllerJointLimitGuardEnabled"])
            if "gripperEnabled" in payload:
                self.config.gripper_enabled = bool(payload["gripperEnabled"])
            if "gripperDevice" in payload:
                self.config.gripper_device = str(payload.get("gripperDevice") or DEFAULT_GRIPPER_DEVICE).strip() or DEFAULT_GRIPPER_DEVICE
            if "gripperInitOnEnable" in payload:
                self.config.gripper_init_on_enable = bool(payload["gripperInitOnEnable"])
            for key, attr in (
                ("gripperOpenWidthM", "gripper_open_width_m"),
                ("gripperCloseWidthM", "gripper_close_width_m"),
                ("gripperSpeedMps", "gripper_speed_mps"),
                ("gripperForceN", "gripper_force_n"),
                ("gripperTriggerCloseThreshold", "gripper_trigger_close_threshold"),
                ("gripperTriggerOpenThreshold", "gripper_trigger_open_threshold"),
            ):
                if key in payload and is_number(payload[key]):
                    setattr(self.config, attr, float(payload[key]))
        return self.status()

    def connect_robot(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.configure(payload)
        try:
            state = self.robot.connect(
                self.config.robot_sn,
                self.config.robot_pose_field,
                self.config.flexiv_rdk,
                self.config.flexiv_network_interfaces,
                wait_seconds=float(payload.get("waitSeconds") or 0.2),
            )
            gripper_warmup = None
            if self.config.gripper_enabled:
                gripper_warmup = self.robot.enable_gripper(
                    self.config.gripper_device,
                    init_on_enable=self.config.gripper_init_on_enable,
                )
            motion_warmup = None
            if self.motion_commands_allowed:
                motion_warmup = self.robot.arm_motion()
                self.config.controller_motion_enabled = True
            else:
                self.config.controller_motion_enabled = False
            warmup = self.warm_realsense_stream(REALSENSE_CONNECT_WARMUP_SECONDS)
            self.last_error = None
            return {
                "ok": True,
                "state": state,
                "gripperWarmup": gripper_warmup,
                "motionWarmup": motion_warmup,
                "formalControlMode": self.formal_control_mode,
                "motionCommandsAllowed": self.motion_commands_allowed,
                "realsenseWarmup": warmup,
                "status": self.status(),
            }
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def disconnect_robot(self) -> dict[str, Any]:
        self.robot.disconnect()
        self.config.controller_motion_enabled = False
        return self.status()

    def arm_motion(self) -> dict[str, Any]:
        if not self.motion_commands_allowed:
            return self._motion_command_blocked("arm")
        try:
            state = self.robot.arm_motion()
            with self.lock:
                if self.active_session is not None:
                    self.active_session.reset_controller_motion_anchor()
            self.config.controller_motion_enabled = True
            self.last_error = None
            return {"ok": True, "state": state, "status": self.status()}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def disarm_motion(self) -> dict[str, Any]:
        if not self.motion_commands_allowed:
            return self._motion_command_blocked("disarm")
        try:
            self.robot.disarm_motion()
            with self.lock:
                if self.active_session is not None:
                    self.active_session.reset_controller_motion_anchor()
            self.config.controller_motion_enabled = False
            self.last_error = None
            return {"ok": True, "status": self.status()}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def enable_freedrive(self) -> dict[str, Any]:
        if not self.motion_commands_allowed:
            return self._motion_command_blocked("enable_freedrive")
        try:
            state = self.robot.enable_freedrive()
            with self.lock:
                if self.active_session is not None:
                    self.active_session.reset_controller_motion_anchor()
            self.config.controller_motion_enabled = False
            self.last_error = None
            return {"ok": True, "state": state, "status": self.status()}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def disable_freedrive(self) -> dict[str, Any]:
        if not self.motion_commands_allowed:
            return self._motion_command_blocked("disable_freedrive")
        try:
            self.robot.disable_freedrive()
            self.config.controller_motion_enabled = False
            self.last_error = None
            return {"ok": True, "status": self.status()}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def _motion_command_blocked(self, action: str) -> dict[str, Any]:
        self.config.controller_motion_enabled = False
        error = f"{action} blocked by {ROBOT_SESSION_CONTROL_RECORD_ONLY} mode"
        return {
            "ok": False,
            "blocked": True,
            "action": action,
            "error": error,
            "formalControlMode": self.formal_control_mode,
            "status": self.status(),
        }

    def list_cameras(self) -> dict[str, Any]:
        try:
            return {"ok": True, "cameras": list_realsense_cameras()}
        except Exception as exc:  # pragma: no cover - hardware path
            return {"ok": False, "error": str(exc), "cameras": []}

    def start_realsense_stream(self) -> dict[str, Any]:
        try:
            self._ensure_stream_for_current_config()
            self.last_error = None
            status = self.status()
            return {"ok": True, "stream": status.get("realsenseStream"), "status": status}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def stop_realsense_stream(self) -> dict[str, Any]:
        with self.lock:
            if self.active_session is not None:
                self.last_error = "Cannot stop RealSense stream while robot recording is active"
                return {"ok": False, "error": self.last_error, "status": self.status()}
        self.stream_hub.stop()
        return {"ok": True, "status": self.status()}

    def capture_end_camera_preview_jpeg(self) -> tuple[bytes, dict[str, Any]]:
        return self.capture_camera_preview_jpeg("end")

    def capture_camera_preview_jpeg(self, role: str = "end") -> tuple[bytes, dict[str, Any]]:
        role = str(role or "end").strip().lower()
        if role not in ("end", "third"):
            raise RuntimeError(f"Unsupported RealSense camera role: {role}")
        config = self._ensure_stream_for_current_config()
        configured_serial = config.third_camera_serial if role == "third" else config.camera_serial
        serial = str(configured_serial or "").strip()
        if not serial:
            raise RuntimeError(f"RealSense {role} camera serial is empty")
        frame = self.stream_hub.get_latest(role, wait_timeout=5.0, copy_arrays=False, include_depth=False)
        bgr = frame.get("bgr")
        if not isinstance(bgr, np.ndarray):
            raise RuntimeError(f"RealSense {role} preview frame has no BGR image")
        return encode_bgr_jpeg(bgr, quality=86), {
            "createdAtUtc": frame.get("capturedAtUtc") or datetime.now(timezone.utc).isoformat(),
            "camera": frame.get("metadata"),
            "sequence": frame.get("sequence"),
            "role": role,
            "stream": self.stream_hub.status(),
        }

    def capture_end_camera_overlay_jpeg(self) -> tuple[bytes, dict[str, Any]]:
        config = self._ensure_stream_for_current_config()
        serial = str(config.camera_serial or "").strip()
        if not serial:
            raise RuntimeError("RealSense end camera serial is empty")
        frame = self.stream_hub.get_latest("end", wait_timeout=5.0)
        bgr = frame["bgr"]
        overlay_bgr, board = checkerboard_overlay_bgr(
            bgr,
            config.pattern_cols,
            config.pattern_rows,
        )
        meta = {
            "createdAtUtc": frame.get("capturedAtUtc") or datetime.now(timezone.utc).isoformat(),
            "camera": frame.get("metadata"),
            "sequence": frame.get("sequence"),
            "stream": self.stream_hub.status(),
            "overlay": True,
            "board": board,
        }
        return encode_bgr_jpeg(overlay_bgr, quality=86), meta

    def check_end_camera_board(self, output_root: Path) -> dict[str, Any]:
        output_dir = output_root.resolve() / "end_camera"
        output_dir.mkdir(parents=True, exist_ok=True)
        config = self._ensure_stream_for_current_config()
        frame = self.stream_hub.get_latest("end", wait_timeout=2.0)
        metadata = frame.get("metadata") or {}
        bgr = frame["bgr"]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightness = {
            "mean": float(np.mean(gray)),
            "median": float(np.median(gray)),
            "p95": float(np.percentile(gray, 95)),
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        serial_name = safe_filename(config.camera_serial or "camera")
        image_path = output_dir / f"end_camera_board_check_{serial_name}_{stamp}.jpg"
        save_bgr_jpeg(bgr, image_path)
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        sample = {
            "sample_index": 0,
            "quest_sample_index": None,
            "quest_recording_timestamp_seconds": None,
        }
        observation = detect_end_observation(
            image_path=image_path,
            sample=sample,
            t_base_ee=np.eye(4, dtype=float),
            camera=metadata,
            cols=config.pattern_cols,
            rows=config.pattern_rows,
            square_size_m=config.square_size_m,
            overlay_dir=overlay_dir,
        )
        result = {
            "ok": observation is not None,
            "createdAtUtc": datetime.now(timezone.utc).isoformat(),
            "camera": metadata,
            "pattern": {
                "cols": config.pattern_cols,
                "rows": config.pattern_rows,
                "squareSizeM": config.square_size_m,
            },
            "brightness": brightness,
            "imagePath": str(image_path),
            "streamFrame": {
                "capturedAtUtc": frame.get("capturedAtUtc"),
                "sequence": frame.get("sequence"),
                "serial": frame.get("serial"),
            },
            "detectedCorners": config.pattern_cols * config.pattern_rows if observation is not None else 0,
            "method": observation.get("method") if observation is not None else None,
            "bestReprojectionRmsePx": None,
            "bestReprojectionMedianPx": None,
            "appearanceAnchor": observation.get("appearance_anchor") if observation is not None else None,
            "overlayPath": None,
            "message": board_check_message(observation is not None, brightness),
        }
        if observation is not None:
            candidates = observation.get("candidates") or []
            best = min(candidates, key=lambda row: float(row.get("reprojection_rmse_px") or float("inf"))) if candidates else {}
            result["bestReprojectionRmsePx"] = best.get("reprojection_rmse_px")
            result["bestReprojectionMedianPx"] = best.get("reprojection_median_px")
            result["overlayPath"] = observation.get("overlay")
        write_json(result, output_dir / "latest_board_check.json")
        return result

    def start_session(
        self,
        parent_directory: Path,
        record_id: str,
        publish_event: Callable[[dict[str, Any]], None] | None,
        robot_alignment_result: dict[str, Any] | None = None,
        require_controller_alignment: bool = False,
        control_mode: str = ROBOT_SESSION_CONTROL_TELEOP,
        record_mode: str = ROBOT_SESSION_RECORD_SYNC,
    ) -> RobotRealsenseSession | None:
        if control_mode not in ROBOT_SESSION_CONTROL_MODES:
            control_mode = ROBOT_SESSION_CONTROL_TELEOP
        if not self.motion_commands_allowed:
            control_mode = ROBOT_SESSION_CONTROL_RECORD_ONLY
        if record_mode not in ROBOT_SESSION_RECORD_MODES:
            record_mode = ROBOT_SESSION_RECORD_SYNC
        with self.lock:
            if self.active_session is not None:
                previous = self.active_session
                previous.close()
                previous_mode = getattr(previous, "control_mode", ROBOT_SESSION_CONTROL_TELEOP)
                if self.motion_commands_allowed and previous_mode == ROBOT_SESSION_CONTROL_FREEDRIVE:
                    self.robot.disable_freedrive()
                elif self.motion_commands_allowed and previous_mode == ROBOT_SESSION_CONTROL_TELEOP:
                    self.robot.disarm_motion()
                self.config.controller_motion_enabled = False
                self.active_session = None
            if self.robot.robot is None:
                self.last_error = "Flexiv robot is not connected"
                if publish_event is not None:
                    publish_event({"type": "robot_status", "ok": False, "stage": "not_connected", "error": self.last_error})
                return None
            if not self.config.camera_serial:
                self.last_error = "RealSense camera serial is empty"
                if publish_event is not None:
                    publish_event({"type": "robot_status", "ok": False, "stage": "no_camera", "error": self.last_error})
                return None
            session_config = copy.deepcopy(self.config)
            session_config.controller_motion_enabled = control_mode == ROBOT_SESSION_CONTROL_TELEOP
            try:
                if control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE:
                    self.robot.enable_freedrive()
                    self.config.controller_motion_enabled = False
                    stage = "freedrive_enabled"
                elif control_mode == ROBOT_SESSION_CONTROL_TELEOP:
                    self.robot.disable_freedrive()
                    self.robot.arm_motion()
                    self.config.controller_motion_enabled = True
                    stage = "teleop_ready"
                else:
                    self.config.controller_motion_enabled = False
                    stage = "record_only"
                self.last_error = None
            except Exception as exc:  # pragma: no cover - hardware path
                self.config.controller_motion_enabled = False
                self.last_error = str(exc)
                if publish_event is not None:
                    publish_event(
                        {
                            "type": "robot_status",
                            "ok": False,
                            "stage": f"{control_mode}_start_failed",
                            "controlMode": control_mode,
                            "error": self.last_error,
                            "status": self.status(),
                        }
                    )
                return None
            session = RobotRealsenseSession(
                parent_directory,
                record_id,
                session_config,
                self.robot,
                self.stream_hub,
                publish_event,
                robot_alignment_result=robot_alignment_result,
                require_controller_alignment=require_controller_alignment,
                control_mode=control_mode,
                record_mode=record_mode,
            )
            try:
                session.start()
            except Exception as exc:  # pragma: no cover - hardware path
                session.close()
                self.last_error = str(exc)
                if publish_event is not None:
                    publish_event({"type": "robot_status", "ok": False, "stage": "start_failed", "error": self.last_error})
                return None
            self.active_session = session
            if publish_event is not None:
                publish_event(
                    {
                        "type": "robot_status",
                        "ok": True,
                        "stage": stage,
                        "controlMode": control_mode,
                        "activeSession": session.summary("recording"),
                        "realsenseStream": self.stream_hub.status(),
                    }
                )
            return session

    def fail_safe_stop_session_control(
        self,
        session: RobotRealsenseSession | None,
        reason: str,
    ) -> dict[str, Any]:
        if session is None:
            return {"ok": True, "stopped": False, "reason": "no_session"}
        with self.lock:
            if self.active_session is not session:
                return {"ok": True, "stopped": False, "reason": "session_not_active"}
            session.reset_controller_motion_anchor()
            session.config.controller_motion_enabled = False
            self.config.controller_motion_enabled = False
            control_mode = getattr(session, "control_mode", ROBOT_SESSION_CONTROL_TELEOP)
            if not self.motion_commands_allowed:
                return {
                    "ok": True,
                    "stopped": True,
                    "reason": reason,
                    "controlMode": control_mode,
                    "motionCommandsAllowed": False,
                }
            if control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE:
                self.robot.disable_freedrive()
            else:
                self.robot.disarm_motion()
            return {
                "ok": True,
                "stopped": True,
                "reason": reason,
                "controlMode": control_mode,
                "motionCommandsAllowed": True,
            }

    def stop_session(
        self,
        session: RobotRealsenseSession | None,
        *,
        restore_motion: bool = True,
    ) -> dict[str, Any] | None:
        if session is None:
            return None
        close_errors: list[dict[str, str]] = []

        def note_close_error(stage: str, exc: BaseException) -> None:
            message = str(exc)
            close_errors.append({"stage": stage, "error": message})
            self.last_error = message

        summary: dict[str, Any] | None = None
        try:
            summary = session.close()
        except Exception as exc:  # pragma: no cover - hardware/video close path
            note_close_error("session_close", exc)
            try:
                summary = session.summary("close_failed")
            except Exception as summary_exc:  # pragma: no cover - defensive fallback
                note_close_error("session_summary_after_close_failure", summary_exc)
                summary = {
                    "ok": False,
                    "closedReason": "close_failed",
                    "recordDirectory": str(getattr(session, "directory", "")),
                }
        finally:
            session_control_mode = getattr(session, "control_mode", ROBOT_SESSION_CONTROL_TELEOP)
            if self.motion_commands_allowed and session_control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE:
                try:
                    self.robot.disable_freedrive()
                except Exception as exc:  # pragma: no cover - hardware path
                    note_close_error("disable_freedrive", exc)
            if self.motion_commands_allowed and restore_motion:
                try:
                    self.robot.arm_motion()
                    self.config.controller_motion_enabled = True
                except Exception as exc:  # pragma: no cover - hardware path
                    note_close_error("robot_mode_restore", exc)
                    self.config.controller_motion_enabled = False
            elif self.motion_commands_allowed:
                try:
                    self.robot.disarm_motion()
                except Exception as exc:  # pragma: no cover - hardware path
                    note_close_error("robot_fail_safe_disarm", exc)
                self.config.controller_motion_enabled = False
            else:
                self.config.controller_motion_enabled = False
            with self.lock:
                if self.active_session is session:
                    self.active_session = None
        if summary is None:
            summary = {"ok": False, "closedReason": "close_failed"}
        if close_errors:
            summary["ok"] = False
            existing_errors = summary.get("closeErrors")
            if not isinstance(existing_errors, list):
                existing_errors = []
            existing_errors.extend(close_errors)
            summary["closeErrors"] = existing_errors
            try:
                write_json(summary, session.summary_path)
            except Exception as exc:  # pragma: no cover - filesystem path
                self.last_error = str(exc)
        return summary

    def calibrate_session(
        self,
        session_dir: Path,
        quest_calibration_event: dict[str, Any] | None,
        publish_event: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        try:
            calibration_config = self._session_calibration_config(session_dir)
            result = calibrate_robot_realsense_run(
                session_dir,
                calibration_config.pattern_cols,
                calibration_config.pattern_rows,
                calibration_config.square_size_m,
                calibration_config.min_hand_eye_detections,
                calibration_config,
            )
            result["questAlignment"] = build_quest_robot_alignment(result, quest_calibration_event)
            write_json(result, session_dir / "robot_hand_eye_result.json")
            event = {
                "type": "robot_calibration_result",
                "recordId": result.get("record_id"),
                "runDir": str(session_dir),
                "result": compact_robot_result(result),
                "T_world_base": result.get("questAlignment", {}).get("T_world_base"),
            }
            self.last_calibration = event
            if publish_event is not None:
                publish_event(event)
            return result
        except Exception as exc:
            counts = exc.counts if isinstance(exc, RobotHandEyeCalibrationError) else None
            diversity = exc.diversity if isinstance(exc, RobotHandEyeCalibrationError) else None
            observations = exc.observations if isinstance(exc, RobotHandEyeCalibrationError) else None
            diagnostics = exc.diagnostics if isinstance(exc, RobotHandEyeCalibrationError) else None
            failure = {
                "type": "robot_calibration_failure",
                "ok": False,
                "runDir": str(session_dir),
                "error": str(exc),
                "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                "counts": counts,
                "diversity": diversity,
                "diagnostics": diagnostics,
                "observations": observations_to_json(observations) if observations is not None else None,
            }
            write_json(failure, session_dir / "robot_hand_eye_failure.json")
            self.last_calibration = failure
            self.last_error = str(exc)
            if publish_event is not None:
                publish_event(failure)
            return failure

    def _session_calibration_config(self, session_dir: Path) -> FlexivRealSenseConfig:
        with self.lock:
            fallback = copy.deepcopy(self.config)
        payload = read_json_if_exists(session_dir / "capture_config.json")
        return config_from_json(payload, fallback)


def import_flexivrdk(flexiv_rdk: Path | None) -> Any:
    if flexiv_rdk is not None:
        lib_py = flexiv_rdk.expanduser() / "lib_py"
        if lib_py.is_dir() and str(lib_py) not in sys.path:
            sys.path.insert(0, str(lib_py))
    import flexivrdk  # type: ignore[import-not-found]

    return flexivrdk


def list_realsense_cameras() -> list[dict[str, Any]]:
    import pyrealsense2 as rs  # type: ignore[import-not-found]

    context = rs.context()
    cameras: list[dict[str, Any]] = []
    for device in context.query_devices():
        cameras.append(
            {
                "name": device_info(device, rs.camera_info.name),
                "serial": device_info(device, rs.camera_info.serial_number),
                "firmware": device_info(device, rs.camera_info.firmware_version),
                "usb": device_info(device, rs.camera_info.usb_type_descriptor),
                "productLine": device_info(device, rs.camera_info.product_line),
                "productId": device_info(device, rs.camera_info.product_id),
            }
        )
    return cameras


def configure_color_sensor(
    profile: Any,
    auto_exposure: bool = True,
    exposure: float | None = None,
    gain: float | None = None,
) -> dict[str, Any]:
    import pyrealsense2 as rs  # type: ignore[import-not-found]

    payload: dict[str, Any] = {
        "autoExposureRequested": bool(auto_exposure),
        "exposureRequested": exposure,
        "gainRequested": gain,
        "sensorName": None,
        "autoExposure": None,
        "exposure": None,
        "gain": None,
        "errors": [],
    }
    try:
        device = profile.get_device()
    except Exception:
        return payload
    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name) if sensor.supports(rs.camera_info.name) else ""
        except Exception:
            name = ""
        if "RGB" not in name and "Color" not in name:
            continue
        payload["sensorName"] = name
        if sensor.supports(rs.option.enable_auto_exposure):
            try:
                sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exposure else 0)
            except Exception as exc:
                payload["errors"].append(f"set auto exposure failed: {exc}")
        if not auto_exposure and exposure is not None and sensor.supports(rs.option.exposure):
            try:
                sensor.set_option(rs.option.exposure, float(exposure))
            except Exception as exc:
                payload["errors"].append(f"set exposure failed: {exc}")
        if gain is not None and sensor.supports(rs.option.gain):
            try:
                sensor.set_option(rs.option.gain, float(gain))
            except Exception as exc:
                payload["errors"].append(f"set gain failed: {exc}")
        for key, option in (("autoExposure", rs.option.enable_auto_exposure), ("exposure", rs.option.exposure), ("gain", rs.option.gain)):
            if sensor.supports(option):
                try:
                    payload[key] = float(sensor.get_option(option))
                except Exception as exc:
                    payload["errors"].append(f"read {key} failed: {exc}")
        break
    return payload


def safe_filename(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    return name or "camera"


def normalize_depth_format(value: Any) -> str:
    text = str(value or DEFAULT_RECORD_DEPTH_FORMAT).strip().lower()
    aliases = {
        "ffv1_mkv": "ffv1",
        "mkv": "ffv1",
        "lossless": "ffv1",
        "bin": "raw",
        "u16le": "raw",
        "raw_u16le": "raw",
    }
    text = aliases.get(text, text)
    return text if text in RECORD_DEPTH_FORMATS else DEFAULT_RECORD_DEPTH_FORMAT


def async_camera_queue_size(fps: Any) -> int:
    try:
        fps_value = max(1.0, float(fps))
    except (TypeError, ValueError):
        fps_value = float(DEFAULT_REALSENSE_FPS)
    desired = int(math.ceil(fps_value * ASYNC_CAMERA_QUEUE_SECONDS))
    return max(ASYNC_CAMERA_QUEUE_MIN_FRAMES, min(ASYNC_CAMERA_QUEUE_MAX_FRAMES, desired))


def board_check_message(detected: bool, brightness: dict[str, float]) -> str:
    if detected:
        return "checkerboard detected"
    if float(brightness.get("p95") or 0.0) < 20.0:
        return "checkerboard not detected; image is very dark"
    return "checkerboard not detected; make sure the full 11x8 inner-corner board is visible and not cropped or occluded"


def camera_metadata_from_profile(profile: Any, serial: str) -> dict[str, Any]:
    import pyrealsense2 as rs  # type: ignore[import-not-found]

    device = profile.get_device()
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    return {
        "serial": serial,
        "name": device_info(device, rs.camera_info.name),
        "firmware": device_info(device, rs.camera_info.firmware_version),
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "distortion_model": str(intr.model),
        "distortion_coeffs": [float(v) for v in intr.coeffs],
    }


def depth_scale_from_profile(profile: Any) -> float | None:
    try:
        device = profile.get_device()
        try:
            depth_sensor = device.first_depth_sensor()
            return float(depth_sensor.get_depth_scale())
        except Exception:
            pass
        for sensor in device.query_sensors():
            if hasattr(sensor, "get_depth_scale"):
                return float(sensor.get_depth_scale())
    except Exception:
        return None
    return None


def device_info(device: Any, key: Any) -> str:
    try:
        if device.supports(key):
            return str(device.get_info(key))
    except Exception:
        pass
    return "unknown"


def calibrate_robot_realsense_run(
    run_dir: Path,
    cols: int,
    rows: int,
    square_size_m: float,
    min_detections: int,
    config: FlexivRealSenseConfig | None = None,
) -> dict[str, Any]:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    del least_squares, Rotation
    run_dir = run_dir.resolve()
    samples = read_jsonl(run_dir / "samples.jsonl")
    cameras = json.loads((run_dir / "cameras.json").read_text(encoding="utf-8"))
    camera = cameras.get("end") if isinstance(cameras, dict) else None
    if not isinstance(camera, dict):
        raise ValueError("missing end camera metadata")

    config = config or FlexivRealSenseConfig()
    selected_samples, sample_selection = select_diverse_hand_eye_samples(samples, config)
    observations = []
    overlay_dir = run_dir / "detections"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    video_cache: dict[str, cv2.VideoCapture] = {}
    try:
        for sample in selected_samples:
            if not sample.get("ok"):
                continue
            t_base_ee = robot_row_tool_transform(sample)
            if t_base_ee is None:
                continue
            frame_payload = load_end_observation_frame(run_dir, sample, video_cache)
            if frame_payload is None:
                continue
            obs = detect_end_observation(
                image=frame_payload["image"],
                image_label=frame_payload["label"],
                sample=sample,
                t_base_ee=t_base_ee,
                camera=camera,
                cols=cols,
                rows=rows,
                square_size_m=square_size_m,
                overlay_dir=overlay_dir,
            )
            if obs is not None:
                observations.append(obs)
    finally:
        for cap in video_cache.values():
            cap.release()

    counts = {
        "samples": len(samples),
        "selectedSamples": len(selected_samples),
        "detections": len(observations),
        "requiredDetections": int(min_detections),
        "sampleSelection": sample_selection,
    }
    if len(observations) < int(min_detections):
        raise RobotHandEyeCalibrationError(
            f"Need at least {min_detections} valid end-camera detections, got {len(observations)}",
            counts=counts,
            observations=observations,
        )
    diversity = observation_pose_diversity(observations)
    try:
        require_hand_eye_pose_diversity(diversity)
    except ValueError as exc:
        raise RobotHandEyeCalibrationError(
            str(exc),
            counts=counts,
            diversity=diversity,
            observations=observations,
        ) from exc
    solution = solve_end_hand_eye(observations)
    validate_hand_eye_solution(solution, counts, diversity, observations)
    result = {
        "ok": True,
        "record_id": samples[0].get("record_id") if samples else None,
        "run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Flexiv end-mounted RealSense hand-eye calibration. T_A_B maps coordinates from B into A.",
        "pattern": {"cols": cols, "rows": rows, "square_size_m": square_size_m},
        "camera": camera,
        "counts": {
            **counts,
            "rot180_selected": int(sum(obs["selected"] == 1 for obs in observations)),
            "appearance_anchor_ok": int(sum(bool((obs.get("appearance_anchor") or {}).get("ok")) for obs in observations)),
            "appearance_anchor_rot180": int(sum((obs.get("appearance_anchor") or {}).get("order") == "rot180" for obs in observations)),
        },
        "diversity": diversity,
        "end_camera": {
            "T_ee_realsense": transform_to_json(solution["T_ee_camera"]),
            "T_realsense_ee": transform_to_json(invert_transform(solution["T_ee_camera"])),
            "residuals": solution["residual_summary"],
        },
        "board": {
            "T_base_board": transform_to_json(solution["T_base_board"]),
            "T_board_base": transform_to_json(invert_transform(solution["T_base_board"])),
        },
        "observations": observations_to_json(observations),
    }
    write_json(result, run_dir / "robot_hand_eye_result.json")
    return result


def load_end_observation_frame(
    run_dir: Path,
    sample: dict[str, Any],
    video_cache: dict[str, cv2.VideoCapture],
) -> dict[str, Any] | None:
    image_rel = (sample.get("images") or {}).get("end")
    if isinstance(image_rel, str) and image_rel:
        image_path = (run_dir / image_rel).resolve()
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                return {"image": image, "label": str(image_path)}

    videos = sample.get("videos") if isinstance(sample.get("videos"), dict) else {}
    end_video = videos.get("end")
    if not isinstance(end_video, dict):
        return None
    rel_video = end_video.get("path")
    frame_index = end_video.get("frameIndex")
    if not isinstance(rel_video, str) or not is_number(frame_index):
        return None
    video_path = (run_dir / rel_video).resolve()
    if not video_path.exists():
        return None
    key = str(video_path)
    cap = video_cache.get(key)
    if cap is None:
        cap = cv2.VideoCapture(key)
        if not cap.isOpened():
            cap.release()
            return None
        video_cache[key] = cap
    index = int(frame_index)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    if not ok or image is None:
        return None
    return {"image": image, "label": f"{video_path}#frame={index}"}


def select_diverse_hand_eye_samples(
    samples: list[dict[str, Any]],
    config: FlexivRealSenseConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any], np.ndarray]] = []
    for index, sample in enumerate(samples):
        if not sample.get("ok"):
            continue
        transform = robot_row_tool_transform(sample)
        if transform is None:
            continue
        candidates.append((index, sample, transform))
    if config.hand_eye_disable_diverse_selection:
        return [sample for _, sample, _ in candidates], {
            "enabled": False,
            "inputSamples": len(samples),
            "candidateSamples": len(candidates),
            "selectedSamples": len(candidates),
            "reason": "disabled",
        }
    if not candidates:
        return [], {
            "enabled": True,
            "inputSamples": len(samples),
            "candidateSamples": 0,
            "selectedSamples": 0,
            "reason": "no_pose_candidates",
        }
    max_count = max(1, int(config.hand_eye_max_diverse_samples))
    min_count = max(1, int(config.hand_eye_min_diverse_samples))
    if len(candidates) <= max_count:
        return [sample for _, sample, _ in candidates], {
            "enabled": True,
            "inputSamples": len(samples),
            "candidateSamples": len(candidates),
            "selectedSamples": len(candidates),
            "reason": "candidate_below_limit",
            "poseSpan": transform_pose_span([matrix for _, _, matrix in candidates]),
        }
    selected_indices, summary = select_diverse_transforms(
        [matrix for _, _, matrix in candidates],
        max_count=max_count,
        min_count=min_count,
        translation_scale_m=float(config.hand_eye_diverse_translation_scale_m),
        rotation_scale_deg=float(config.hand_eye_diverse_rotation_scale_deg),
        min_score=float(config.hand_eye_diverse_min_score),
    )
    selected_set = {int(i) for i in selected_indices}
    selected = [sample for local_index, (_, sample, _) in enumerate(candidates) if local_index in selected_set]
    summary.update(
        {
            "inputSamples": len(samples),
            "candidateSamples": len(candidates),
            "selectedSamples": len(selected),
            "selectedOriginalSampleIndices": [int(candidates[i][0]) for i in selected_indices],
        }
    )
    return selected, summary


def select_diverse_transforms(
    transforms: list[np.ndarray],
    *,
    max_count: int,
    min_count: int,
    translation_scale_m: float,
    rotation_scale_deg: float,
    min_score: float,
) -> tuple[list[int], dict[str, Any]]:
    count = len(transforms)
    if count <= 0:
        return [], {"enabled": True, "reason": "empty"}
    translations = np.asarray([matrix[:3, 3] for matrix in transforms], dtype=float)
    rotations = [np.asarray(matrix[:3, :3], dtype=float) for matrix in transforms]
    center = np.median(translations, axis=0)
    selected = sorted({0, int(np.argmax(np.linalg.norm(translations - center[None, :], axis=1)))})
    trans_scale = max(1e-6, float(translation_scale_m))
    rot_scale = max(1e-6, math.radians(float(rotation_scale_deg)))
    min_dist = transform_distance_to_set(translations, rotations, selected, trans_scale, rot_scale)
    while len(selected) < max_count:
        candidate_scores = min_dist.copy()
        candidate_scores[selected] = -np.inf
        candidate = int(np.argmax(candidate_scores))
        score = float(candidate_scores[candidate])
        if len(selected) >= min_count and score < float(min_score):
            break
        if not math.isfinite(score):
            break
        selected.append(candidate)
        new_dist = transform_distance_to_set(translations, rotations, [candidate], trans_scale, rot_scale)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[selected] = 0.0
    selected = sorted(set(selected))
    return selected, {
        "enabled": True,
        "reason": "pose_diverse_selection",
        "maxCount": int(max_count),
        "minCount": int(min_count),
        "translationScaleM": trans_scale,
        "rotationScaleDeg": float(rotation_scale_deg),
        "minScore": float(min_score),
        "selectedFraction": float(len(selected) / count),
        "lastBestScore": float(np.max(min_dist)) if len(min_dist) else 0.0,
        "poseSpan": transform_pose_span([transforms[i] for i in selected]),
    }


def transform_distance_to_set(
    translations: np.ndarray,
    rotations: list[np.ndarray],
    selected: list[int],
    translation_scale_m: float,
    rotation_scale_rad: float,
) -> np.ndarray:
    distances = np.full(len(translations), np.inf, dtype=float)
    for index in selected:
        translation = np.linalg.norm(translations - translations[index][None, :], axis=1) / translation_scale_m
        rotation = np.asarray(
            [rotation_angle_deg(rotations[index].T @ matrix) for matrix in rotations],
            dtype=float,
        )
        rotation = np.deg2rad(rotation) / rotation_scale_rad
        distances = np.minimum(distances, np.sqrt(translation * translation + rotation * rotation))
    return distances


def validate_hand_eye_solution(
    solution: dict[str, Any],
    counts: dict[str, Any],
    diversity: dict[str, Any],
    observations: list[dict[str, Any]],
) -> None:
    t_ee_camera = np.asarray(solution["T_ee_camera"], dtype=float)
    offset_m = float(np.linalg.norm(t_ee_camera[:3, 3]))
    residual = solution.get("residual_summary") if isinstance(solution, dict) else {}
    translation = residual.get("translation_mm") if isinstance(residual, dict) else {}
    median_mm = float(translation.get("median") or 0.0) if isinstance(translation, dict) else 0.0
    p95_mm = float(translation.get("p95") or 0.0) if isinstance(translation, dict) else 0.0
    diagnostics = {
        "cameraOffsetM": offset_m,
        "maxCameraOffsetM": MAX_HAND_EYE_CAMERA_OFFSET_M,
        "translationResidualMedianMm": median_mm,
        "maxTranslationResidualMedianMm": MAX_HAND_EYE_TRANSLATION_MEDIAN_RESIDUAL_MM,
        "translationResidualP95Mm": p95_mm,
        "maxTranslationResidualP95Mm": MAX_HAND_EYE_TRANSLATION_P95_RESIDUAL_MM,
        "hint": (
            "Check that the end camera serial is the camera mounted on the robot end-effector, "
            "and that the end-camera checkerboard view is not coming from the third/static camera."
        ),
    }
    reasons = []
    if offset_m > MAX_HAND_EYE_CAMERA_OFFSET_M:
        reasons.append(f"end-camera extrinsic offset {offset_m:.3f}m > {MAX_HAND_EYE_CAMERA_OFFSET_M:.3f}m")
    if median_mm > MAX_HAND_EYE_TRANSLATION_MEDIAN_RESIDUAL_MM:
        reasons.append(
            f"translation residual median {median_mm:.1f}mm > {MAX_HAND_EYE_TRANSLATION_MEDIAN_RESIDUAL_MM:.1f}mm"
        )
    if p95_mm > MAX_HAND_EYE_TRANSLATION_P95_RESIDUAL_MM:
        reasons.append(
            f"translation residual p95 {p95_mm:.1f}mm > {MAX_HAND_EYE_TRANSLATION_P95_RESIDUAL_MM:.1f}mm"
        )
    if reasons:
        raise RobotHandEyeCalibrationError(
            "Implausible Flexiv/RealSense hand-eye result: " + "; ".join(reasons),
            counts=counts,
            diversity=diversity,
            observations=observations,
            diagnostics=diagnostics,
        )


def detect_end_observation(
    *,
    image: np.ndarray | None = None,
    image_label: str | None = None,
    image_path: Path | None = None,
    sample: dict[str, Any],
    t_base_ee: np.ndarray,
    camera: dict[str, Any],
    cols: int,
    rows: int,
    square_size_m: float,
    overlay_dir: Path,
) -> dict[str, Any] | None:
    if image is None and image_path is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_label = str(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    ok, corners = cv2.findChessboardCornersSB(gray, pattern)
    method = "findChessboardCornersSB"
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        method = "findChessboardCorners"
        if ok and corners is not None:
            cv2.cornerSubPix(
                gray,
                corners,
                (7, 7),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
            )
    if not ok or corners is None:
        return None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    appearance_anchor = detect_checkerboard_appearance_anchor(
        image,
        corners.reshape(-1, 2),
        cols,
        rows,
        min_contrast=APPEARANCE_ANCHOR_MIN_CONTRAST,
    )
    appearance_order = appearance_anchor.get("order") if appearance_anchor.get("ok") else None
    camera_matrix, distortion = camera_intrinsics(camera)
    candidates = []
    for order, obj in (
        ("identity", object_points(cols, rows, square_size_m)),
        ("rot180", object_points(cols, rows, square_size_m)[::-1].copy()),
    ):
        success, rvec, tvec = cv2.solvePnP(obj, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            continue
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        error = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        candidates.append(
            {
                "order": order,
                "T_camera_board": transform_to_json(transform_from_rvec_tvec(rvec, tvec)),
                "reprojection_rmse_px": float(np.sqrt(np.mean(error * error))),
                "reprojection_median_px": float(np.median(error)),
                "matches_appearance_anchor": bool(order == appearance_order) if appearance_order in ("identity", "rot180") else None,
            }
        )
    if not candidates:
        return None
    overlay_path = overlay_dir / f"sample_{int(sample.get('sample_index', 0)):06d}_end_overlay.jpg"
    overlay = image.copy()
    cv2.drawChessboardCorners(overlay, pattern, corners, True)
    draw_checkerboard_appearance_anchor_overlay(overlay, corners.reshape(-1, 2), appearance_anchor, cols, rows)
    cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return {
        "sample_index": int(sample.get("sample_index", 0)),
        "quest_sample_index": sample.get("quest_sample_index"),
        "quest_recording_timestamp_seconds": sample.get("quest_recording_timestamp_seconds"),
        "image": image_label or (str(image_path) if image_path is not None else None),
        "overlay": str(overlay_path),
        "method": method,
        "T_base_ee": t_base_ee,
        "appearance_anchor": appearance_anchor,
        "candidates": candidates,
        "selected": 0,
    }


def checkerboard_overlay_bgr(bgr: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    overlay = bgr.copy()
    pattern = (int(cols), int(rows))
    ok = False
    corners = None
    method = "findChessboardCorners"
    try:
        scale = min(1.0, 640.0 / max(1, max(bgr.shape[0], bgr.shape[1])))
        if scale < 1.0:
            small = cv2.resize(bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        method = "findChessboardCorners"
        if ok and corners is not None:
            cv2.cornerSubPix(
                gray,
                corners,
                (7, 7),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
            )
            if scale < 1.0:
                corners = corners / scale
    except Exception as exc:
        ok = False
        corners = None
        method = f"overlay_error:{type(exc).__name__}"
    if ok and corners is not None:
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        appearance_anchor = detect_checkerboard_appearance_anchor(
            bgr,
            corners.reshape(-1, 2),
            int(cols),
            int(rows),
            min_contrast=APPEARANCE_ANCHOR_MIN_CONTRAST,
        )
        cv2.drawChessboardCorners(overlay, pattern, corners, True)
        draw_checkerboard_appearance_anchor_overlay(overlay, corners.reshape(-1, 2), appearance_anchor, int(cols), int(rows))
        status = f"checkerboard detected: {len(corners)}/{int(cols) * int(rows)}"
        color = (80, 220, 120)
    else:
        appearance_anchor = None
        status = "checkerboard not detected"
        color = (80, 120, 255)
    cv2.rectangle(overlay, (10, 10), (min(520, overlay.shape[1] - 10), 46), (20, 24, 28), -1)
    cv2.putText(overlay, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return overlay, {
        "ok": bool(ok and corners is not None),
        "detectedCorners": int(len(corners)) if ok and corners is not None else 0,
        "expectedCorners": int(cols) * int(rows),
        "method": method,
        "appearanceAnchor": appearance_anchor,
    }


def solve_end_hand_eye(observations: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.optimize import least_squares

    selected = [initial_candidate_index(obs) for obs in observations]
    x0 = np.eye(4, dtype=float)
    x0[:3, 3] = np.array([0.0, 0.0, 0.08], dtype=float)
    b0 = average_transforms([obs["T_base_ee"] @ x0 @ candidate_matrix(obs, 0) for obs in observations])
    params = pack_two_transforms(x0, b0)
    previous: list[int] | None = None
    for _ in range(8):
        result = least_squares(
            lambda p: hand_eye_residual_vector(p, observations, selected),
            params,
            max_nfev=500,
            loss="soft_l1",
        )
        params = result.x
        t_ee_camera, t_base_board = unpack_two_transforms(params)
        selected = [
            choose_candidate(obs, t_ee_camera, t_base_board)
            for obs in observations
        ]
        if selected == previous:
            break
        previous = selected[:]
    result = least_squares(
        lambda p: hand_eye_residual_vector(p, observations, selected),
        params,
        max_nfev=800,
        loss="soft_l1",
    )
    t_ee_camera, t_base_board = unpack_two_transforms(result.x)
    for obs, choice in zip(observations, selected):
        obs["selected"] = int(choice)
    residuals = [
        pose_residual(t_base_board, obs["T_base_ee"] @ t_ee_camera @ candidate_matrix(obs, obs["selected"]))
        for obs in observations
    ]
    return {
        "T_ee_camera": t_ee_camera,
        "T_base_board": t_base_board,
        "residual_summary": residual_summary(residuals),
    }


def initial_candidate_index(obs: dict[str, Any]) -> int:
    appearance_anchor = obs.get("appearance_anchor") if isinstance(obs.get("appearance_anchor"), dict) else {}
    appearance_order = appearance_anchor.get("order") if appearance_anchor.get("ok") else None
    if appearance_order in ("identity", "rot180"):
        for index, candidate in enumerate(obs.get("candidates") or []):
            if candidate.get("order") == appearance_order:
                return int(index)
    return 0


def observation_pose_diversity(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return ee_pose_diversity([np.asarray(obs["T_base_ee"], dtype=float) for obs in observations])


def transform_pose_span(poses: list[np.ndarray]) -> dict[str, Any]:
    if not poses:
        return {
            "samples": 0,
            "translationSpanM": 0.0,
            "rotationSpanDeg": 0.0,
            "axisSpanM": [0.0, 0.0, 0.0],
        }
    translations = np.asarray([pose[:3, 3] for pose in poses], dtype=float)
    max_translation = 0.0
    max_rotation = 0.0
    for i, a in enumerate(poses):
        for b in poses[i + 1 :]:
            delta = invert_transform(a) @ b
            max_translation = max(max_translation, float(np.linalg.norm(delta[:3, 3])))
            max_rotation = max(max_rotation, rotation_angle_deg(delta[:3, :3]))
    axis_span = np.ptp(translations, axis=0) if translations.size else np.zeros(3, dtype=float)
    return {
        "samples": len(poses),
        "translationSpanM": float(max_translation),
        "rotationSpanDeg": float(max_rotation),
        "axisSpanM": [float(v) for v in axis_span],
    }


def ee_pose_diversity(poses: list[np.ndarray]) -> dict[str, Any]:
    translations = np.asarray([pose[:3, 3] for pose in poses], dtype=float)
    pairwise_poses = poses
    if len(poses) > POSE_DIVERSITY_MAX_PAIRWISE_SAMPLES:
        indices = np.linspace(
            0,
            len(poses) - 1,
            num=POSE_DIVERSITY_MAX_PAIRWISE_SAMPLES,
            dtype=int,
        )
        pairwise_poses = [poses[int(index)] for index in indices]
    max_translation = 0.0
    max_rotation = 0.0
    pair_count = 0
    for i, a in enumerate(pairwise_poses):
        for b in pairwise_poses[i + 1 :]:
            pair_count += 1
            delta = invert_transform(a) @ b
            max_translation = max(max_translation, float(np.linalg.norm(delta[:3, 3])))
            max_rotation = max(max_rotation, rotation_angle_deg(delta[:3, :3]))
    axis_span = np.ptp(translations, axis=0) if translations.size else np.zeros(3, dtype=float)
    return {
        "samples": len(poses),
        "analyzedSamples": len(pairwise_poses),
        "pairCount": pair_count,
        "totalPairCount": len(poses) * max(0, len(poses) - 1) // 2,
        "pairwiseDownsampled": len(pairwise_poses) < len(poses),
        "eeTranslationSpanM": float(max_translation),
        "eeRotationSpanDeg": float(max_rotation),
        "eeAxisSpanM": [float(v) for v in axis_span],
        "minTranslationSpanM": MIN_HAND_EYE_EE_TRANSLATION_SPAN_M,
        "minRotationSpanDeg": MIN_HAND_EYE_EE_ROTATION_SPAN_DEG,
    }


def require_hand_eye_pose_diversity(diversity: dict[str, Any]) -> None:
    translation_span = float(diversity.get("eeTranslationSpanM") or 0.0)
    rotation_span = float(diversity.get("eeRotationSpanDeg") or 0.0)
    if rotation_span >= MIN_HAND_EYE_EE_ROTATION_SPAN_DEG:
        return
    raise ValueError(
        "Need robot/end-camera pose diversity for hand-eye calibration: "
        f"rotation span {rotation_span:.2f} deg "
        f"(need >= {MIN_HAND_EYE_EE_ROTATION_SPAN_DEG:.2f} deg). "
        f"Translation span was {translation_span * 1000.0:.1f} mm, but translation alone "
        "does not constrain the end-camera hand-eye transform well enough."
    )


def hand_eye_residual_vector(params: np.ndarray, observations: list[dict[str, Any]], selected: list[int]) -> np.ndarray:
    t_ee_camera, t_base_board = unpack_two_transforms(params)
    residuals = []
    for obs, choice in zip(observations, selected):
        predicted = obs["T_base_ee"] @ t_ee_camera @ candidate_matrix(obs, choice)
        residual = pose_residual(t_base_board, predicted)
        residuals.extend((residual["translation"] / 0.01).tolist())
        residuals.extend((residual["rotation"] / math.radians(1.0)).tolist())
    return np.asarray(residuals, dtype=float)


def choose_candidate(obs: dict[str, Any], t_ee_camera: np.ndarray, t_base_board: np.ndarray) -> int:
    appearance_anchor = obs.get("appearance_anchor") if isinstance(obs.get("appearance_anchor"), dict) else {}
    appearance_order = appearance_anchor.get("order") if appearance_anchor.get("ok") else None
    if appearance_order in ("identity", "rot180"):
        for index, candidate in enumerate(obs.get("candidates") or []):
            if candidate.get("order") == appearance_order:
                return int(index)
    scores = []
    for index in range(len(obs["candidates"])):
        predicted = obs["T_base_ee"] @ t_ee_camera @ candidate_matrix(obs, index)
        residual = pose_residual(t_base_board, predicted)
        scores.append(float(np.linalg.norm(residual["translation"]) + 0.01 * np.linalg.norm(residual["rotation"])))
    return int(np.argmin(np.asarray(scores, dtype=float)))


def candidate_matrix(obs: dict[str, Any], index: int) -> np.ndarray:
    candidate = obs["candidates"][index]
    matrix = transform_from_json(candidate.get("T_camera_board"))
    if matrix is None:
        raise ValueError("bad candidate transform")
    return matrix


def build_quest_robot_alignment(
    robot_result: dict[str, Any],
    quest_calibration_event: dict[str, Any] | None,
) -> dict[str, Any]:
    if not quest_calibration_event:
        return {"ok": False, "reason": "missing_quest_calibration"}
    frame = quest_calibration_event.get("coordinateFrame") or quest_calibration_event.get("coordinate_frame")
    t_world_board_payload = ensure_pc_transform_payload(quest_calibration_event.get("T_world_board"), frame)
    t_world_board = matrix_from_transform_payload(t_world_board_payload)
    board_payload = robot_result.get("board") if isinstance(robot_result.get("board"), dict) else {}
    t_base_board = transform_from_json(board_payload.get("T_base_board"))
    if t_world_board is None or t_base_board is None:
        return {"ok": False, "reason": "missing_transform"}
    t_world_base = t_world_board @ invert_transform(t_base_board)
    return {
        "ok": True,
        "coordinateFrame": PC_WORLD_FRAME,
        "sourceQuestFrame": frame or UNITY_WORLD_FRAME,
        "worldFrameConversion": WORLD_FRAME_CONVERSION,
        "T_world_base": transform_to_json(t_world_base, PC_WORLD_FRAME),
        "T_base_world": transform_to_json(invert_transform(t_world_base), PC_WORLD_FRAME),
    }


def compact_robot_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "recordId": result.get("record_id"),
        "counts": result.get("counts"),
        "diversity": result.get("diversity"),
        "endCamera": result.get("end_camera"),
        "board": result.get("board"),
        "questAlignment": result.get("questAlignment"),
    }


def robot_sample_event(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
    return {
        "type": "robot_sample",
        "ok": bool(row.get("ok")),
        "coordinateFrame": PC_WORLD_FRAME,
        "recordId": row.get("record_id"),
        "sampleIndex": row.get("sample_index"),
        "questSampleIndex": row.get("quest_sample_index"),
        "capturedAt": row.get("captured_at"),
        "T_base_ee": row.get("T_base_ee"),
        "T_base_tool_tcp": row.get("T_base_tool_tcp"),
        "T_world_tool_tcp": row.get("T_world_tool_tcp"),
        "T_base_end_camera": row.get("T_base_end_camera"),
        "T_world_end_camera": row.get("T_world_end_camera"),
        "jointpose": row.get("jointpose") or row.get("jointpos"),
        "jointpos": row.get("jointpos") or row.get("jointpose"),
        "images": row.get("images"),
        "videos": row.get("videos"),
        "videoFrames": row.get("videoFrames"),
        "questGaze3DWorld": row.get("quest_gaze3d_pc_world") or row.get("quest_gaze3d_world"),
        "questGaze3DUnityWorld": row.get("quest_gaze3d_world"),
        "questGaze3DSource": row.get("quest_gaze3d_source"),
        "rightController": row.get("right_controller"),
        "gripper": row.get("gripper"),
        "jointLimitGuard": (state.get("jointLimitGuard") if isinstance(state, dict) else None),
        "poseDiversity": row.get("poseDiversity"),
        "poseField": state.get("poseField"),
        "robotSn": state.get("robotSn"),
        "error": row.get("error"),
    }


def robot_motion_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "robot_motion",
        "ok": bool(row.get("ok")),
        "recordId": row.get("record_id"),
        "questSampleIndex": row.get("quest_sample_index"),
        "targetTcpPose": row.get("target_tcp_pose_wxyz"),
        "offsetM": row.get("offset_m"),
        "stepOffsetM": row.get("step_offset_m"),
        "rotationDeg": row.get("rotation_angle_deg"),
        "stepRotationDeg": row.get("step_rotation_deg"),
        "rawRotationWorldDeg": row.get("raw_rotation_angle_world_deg"),
        "anchored": row.get("anchored"),
        "createdAnchor": row.get("created_anchor"),
        "questAlignmentUsed": row.get("quest_alignment_used"),
        "motionMapping": row.get("motion_mapping"),
        "teleopHeld": row.get("teleopHeld") if row.get("teleopHeld") is not None else row.get("teleop_held"),
        "teleopHoldValue": row.get("teleopHoldValue")
        if row.get("teleopHoldValue") is not None
        else row.get("teleop_hold_value"),
        "rightControllerInput": row.get("right_controller_input"),
        "jointLimitGuard": row.get("joint_limit_guard"),
        "reason": row.get("reason"),
        "error": row.get("error"),
    }


def robot_gripper_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "robot_gripper",
        "ok": bool(row.get("ok")),
        "recordId": row.get("record_id"),
        "questSampleIndex": row.get("quest_sample_index"),
        "capturedAt": row.get("captured_at"),
        "commandSent": bool(row.get("commandSent")),
        "action": row.get("action"),
        "device": row.get("device"),
        "trigger": row.get("trigger"),
        "targetWidthM": row.get("target_width_m"),
        "status": row.get("status"),
        "devices": row.get("devices"),
        "reason": row.get("reason"),
        "error": row.get("error"),
    }


def controller_trigger_value(controller: dict[str, Any]) -> float | None:
    summary = controller_input_summary(controller)
    value = summary.get("indexTrigger") if isinstance(summary, dict) else None
    if is_number(value):
        return max(0.0, min(1.0, float(value)))
    pressed = summary.get("indexTriggerPressed") if isinstance(summary, dict) else None
    if isinstance(pressed, bool):
        return 1.0 if pressed else 0.0
    if not isinstance(controller, dict):
        return None
    for key in (
        "indexTrigger",
        "rightIndexTrigger",
        "trigger",
        "primaryIndexTrigger",
        "primaryTrigger",
    ):
        value = controller.get(key)
        if is_number(value):
            return max(0.0, min(1.0, float(value)))
    buttons = controller.get("buttons")
    if isinstance(buttons, dict):
        for key in ("indexTrigger", "trigger", "primaryIndexTrigger"):
            value = buttons.get(key)
            if is_number(value):
                return max(0.0, min(1.0, float(value)))
        for key in ("indexTriggerPressed", "triggerPressed", "primaryIndexTriggerPressed"):
            value = buttons.get(key)
            if isinstance(value, bool):
                return 1.0 if value else 0.0
    for key in ("indexTriggerPressed", "triggerPressed", "primaryIndexTriggerPressed"):
        value = controller.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
    return None


def right_controller_side_button_pressed(controller: dict[str, Any], threshold: float = 0.65) -> tuple[bool, float | None]:
    summary = controller_input_summary(controller, threshold=threshold)
    value = summary.get("handTrigger") if isinstance(summary, dict) else None
    pressed = summary.get("handTriggerPressed") if isinstance(summary, dict) else None
    if isinstance(pressed, bool) and pressed:
        return True, 1.0 if not is_number(value) else max(0.0, min(1.0, float(value)))
    if is_number(value):
        clamped = max(0.0, min(1.0, float(value)))
        return clamped >= float(threshold), clamped
    return bool(summary.get("teleopHeld")) if isinstance(summary, dict) else False, None


def controller_input_summary(controller: dict[str, Any] | None, threshold: float = 0.65) -> dict[str, Any]:
    if not isinstance(controller, dict):
        return {
            "hasAny": False,
            "handTrigger": None,
            "indexTrigger": None,
            "handTriggerPressed": None,
            "indexTriggerPressed": None,
            "aButton": None,
            "bButton": None,
            "teleopHeld": False,
            "teleopThreshold": float(threshold),
        }
    nested_input = controller.get("input")
    buttons = controller.get("buttons")
    if not isinstance(nested_input, dict):
        nested_input = {}
    if not isinstance(buttons, dict):
        buttons = {}

    def first_float(*keys: str) -> float | None:
        for source in (controller, nested_input, buttons):
            for key in keys:
                value = source.get(key)
                if is_number(value):
                    return max(0.0, min(1.0, float(value)))
        return None

    def first_bool(*keys: str) -> bool | None:
        for source in (controller, nested_input, buttons):
            for key in keys:
                value = source.get(key)
                if isinstance(value, bool):
                    return value
        return None

    hand_trigger = first_float("handTrigger", "rightHandTrigger", "grip", "primaryHandTrigger")
    index_trigger = first_float(
        "indexTrigger",
        "rightIndexTrigger",
        "trigger",
        "primaryIndexTrigger",
        "primaryTrigger",
    )
    hand_pressed = first_bool("handTriggerPressed", "rightHandTriggerPressed", "gripPressed", "gripButton")
    index_pressed = first_bool(
        "indexTriggerPressed",
        "rightIndexTriggerPressed",
        "triggerPressed",
        "triggerButton",
        "primaryIndexTriggerPressed",
    )
    a_button = first_bool("aButton", "buttonA", "primaryButton")
    b_button = first_bool("bButton", "buttonB", "secondaryButton")
    has_any = bool(nested_input.get("hasAny")) or any(
        value is not None
        for value in (hand_trigger, index_trigger, hand_pressed, index_pressed, a_button, b_button)
    )
    teleop_held = bool(hand_pressed) or (hand_trigger is not None and hand_trigger >= float(threshold))
    if nested_input.get("teleopHeld") is not None:
        teleop_held = bool(nested_input.get("teleopHeld")) or teleop_held
    return {
        "hasAny": has_any,
        "handTrigger": hand_trigger,
        "indexTrigger": index_trigger,
        "handTriggerPressed": hand_pressed,
        "indexTriggerPressed": index_pressed,
        "aButton": a_button,
        "bButton": b_button,
        "teleopHeld": teleop_held,
        "teleopThreshold": float(threshold),
    }


def visualizer_controller_payload(controller: Any) -> dict[str, Any]:
    input_summary = controller_input_summary(controller)
    if not isinstance(controller, dict):
        return {"ok": False, "source": "missing_controller_payload", "input": input_summary}
    position = vec3_array(controller.get("position"))
    rotation = controller.get("rotation")
    q = None
    if isinstance(rotation, list) and len(rotation) >= 4:
        try:
            q = [float(rotation[index]) for index in range(4)]
        except (TypeError, ValueError):
            q = None
    if not q or not all(math.isfinite(item) for item in q):
        q = [1.0, 0.0, 0.0, 0.0]
    return {
        "ok": bool(controller.get("hasPose") and position is not None),
        "source": controller.get("source") or controller.get("missingReason") or "right",
        "p": [float(v) for v in position] if position is not None else None,
        "q": q,
        "input": input_summary,
    }


def controller_payload_to_pc_world(controller: Any) -> dict[str, Any] | None:
    if not isinstance(controller, dict):
        return None
    result = dict(controller)
    position = unity_vec3_to_pc(controller.get("position"))
    rotation = unity_quaternion_wxyz_to_pc(controller.get("rotation"))
    pose = controller.get("pose")
    if isinstance(pose, list) and len(pose) >= 7:
        pose_position = unity_vec3_to_pc(pose[:3])
        pose_rotation = unity_quaternion_wxyz_to_pc(pose[3:7])
        if pose_position is not None and pose_rotation is not None:
            result["pose"] = [*pose_position, *pose_rotation]
    if position is not None:
        result["position"] = position
    if rotation is not None:
        result["rotation"] = rotation
    result["coordinateFrame"] = PC_WORLD_FRAME
    result["sourceCoordinateFrame"] = controller.get("coordinateFrame") or UNITY_WORLD_FRAME
    return result


def flexiv_pose_to_transform(pose: np.ndarray) -> np.ndarray:
    x, y, z, qw, qx, qy, qz = [float(v) for v in pose]
    rotation = quaternion_wxyz_to_matrix([qw, qx, qy, qz])
    return make_transform(rotation, np.asarray([x, y, z], dtype=float))


def quaternion_wxyz_to_matrix(q: list[float]) -> np.ndarray:
    qw, qx, qy, qz = normalize_quaternion(q)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    qx, qy, qz, qw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return normalize_quaternion([float(qw), float(qx), float(qy), float(qz)])


def quat_rotation_matrix(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        quat = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in quat):
        return None
    return quaternion_wxyz_to_matrix(quat)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = translation.reshape(3)
    return result


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_transform(rotation, np.asarray(tvec, dtype=float).reshape(3))


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def transform_to_json(matrix: np.ndarray, coordinate_frame: str | None = None) -> dict[str, Any]:
    quat = matrix_to_quaternion_wxyz(matrix)
    payload = {
        "matrix_4x4": [[float(value) for value in row] for row in matrix[:4, :4]],
        "translation_m": [float(value) for value in matrix[:3, 3]],
        "quaternion_wxyz": quat,
        "quaternion_xyzw": [quat[1], quat[2], quat[3], quat[0]],
        "rotation_matrix": [[float(value) for value in row] for row in matrix[:3, :3]],
    }
    if coordinate_frame:
        payload["coordinateFrame"] = coordinate_frame
    return payload


def transform_to_compact_json(matrix: np.ndarray, coordinate_frame: str | None = None) -> dict[str, Any]:
    quat = matrix_to_quaternion_wxyz(matrix)
    payload = {
        "translation_m": [float(value) for value in matrix[:3, 3]],
        "quaternion_wxyz": quat,
    }
    if coordinate_frame:
        payload["coordinateFrame"] = coordinate_frame
    return payload


def transform_from_json(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    matrix = payload.get("matrix_4x4")
    if isinstance(matrix, list) and len(matrix) >= 4:
        try:
            return np.asarray([[float(row[col]) for col in range(4)] for row in matrix[:4]], dtype=float)
        except Exception:
            return None
    translation = payload.get("translation_m")
    quat = payload.get("quaternion_wxyz")
    if isinstance(translation, list) and isinstance(quat, list) and len(translation) >= 3 and len(quat) >= 4:
        return make_transform(
            quaternion_wxyz_to_matrix([float(v) for v in quat[:4]]),
            np.asarray([float(v) for v in translation[:3]], dtype=float),
        )
    return None


def flexiv_pose_payload_to_transform(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, list) or len(payload) < 7:
        return None
    try:
        pose = np.asarray([float(payload[index]) for index in range(7)], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(pose)):
        return None
    return flexiv_pose_to_transform(pose)


def robot_state_pose_transform(state: dict[str, Any], key: str = "rawEndEffectorPoseWxyz") -> np.ndarray | None:
    if not isinstance(state, dict):
        return None
    direct = transform_from_json(state.get("endEffectorPose"))
    if direct is not None and key == "rawEndEffectorPoseWxyz":
        return direct
    return flexiv_pose_payload_to_transform(state.get(key))


def robot_row_tool_transform(row: dict[str, Any], pose_to_tool_tcp: np.ndarray | None = None) -> np.ndarray | None:
    transform = transform_from_json(row.get("T_base_tool_tcp"))
    if transform is not None:
        return transform
    state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
    tcp = state.get("tcp_pose") if isinstance(state, dict) and isinstance(state.get("tcp_pose"), dict) else None
    if isinstance(tcp, dict):
        transform = transform_from_json(tcp.get("T_base_pose"))
        if transform is not None:
            return transform
        transform = flexiv_pose_payload_to_transform(tcp.get("pose"))
        if transform is not None:
            return transform
    transform = transform_from_json(row.get("T_base_ee"))
    if transform is not None:
        return transform @ pose_to_tool_tcp if pose_to_tool_tcp is not None else transform
    if isinstance(state, dict):
        transform = robot_state_pose_transform(state)
        if transform is not None and pose_to_tool_tcp is not None:
            return transform @ pose_to_tool_tcp
        return transform
    return None


def pack_two_transforms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([pack_transform(a), pack_transform(b)])


def unpack_two_transforms(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return unpack_transform(params[:6]), unpack_transform(params[6:12])


def pack_transform(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return np.concatenate([Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]])


def unpack_transform(params: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec(np.asarray(params[:3], dtype=float)).as_matrix()
    translation = np.asarray(params[3:6], dtype=float)
    return make_transform(rotation, translation)


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    translations = np.asarray([t[:3, 3] for t in transforms], dtype=float)
    rotations = Rotation.from_matrix(np.asarray([t[:3, :3] for t in transforms], dtype=float))
    return make_transform(rotations.mean().as_matrix(), np.mean(translations, axis=0))


def pose_residual(target: np.ndarray, value: np.ndarray) -> dict[str, np.ndarray]:
    from scipy.spatial.transform import Rotation

    delta = invert_transform(target) @ value
    return {
        "translation": delta[:3, 3],
        "rotation": Rotation.from_matrix(delta[:3, :3]).as_rotvec(),
    }


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return float(math.degrees(math.acos(cosine)))


def cartesian_target_near(
    lhs: list[float],
    rhs: list[float],
    position_eps_m: float,
    rotation_eps_deg: float,
) -> bool:
    if len(lhs) < 7 or len(rhs) < 7:
        return False
    try:
        delta_position = np.asarray(lhs[:3], dtype=float) - np.asarray(rhs[:3], dtype=float)
        if float(np.linalg.norm(delta_position)) > float(position_eps_m):
            return False
        lhs_rotation = quaternion_wxyz_to_matrix(lhs[3:7])
        rhs_rotation = quaternion_wxyz_to_matrix(rhs[3:7])
        return rotation_angle_deg(lhs_rotation @ rhs_rotation.T) <= float(rotation_eps_deg)
    except Exception:
        return False


def clamp_rotation_angle(rotation: np.ndarray, max_angle_deg: float) -> np.ndarray:
    if max_angle_deg <= 0:
        return np.eye(3, dtype=float)
    from scipy.spatial.transform import Rotation

    rotvec = Rotation.from_matrix(rotation[:3, :3]).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if not math.isfinite(angle) or angle <= 1e-12:
        return np.eye(3, dtype=float)
    max_angle_rad = math.radians(float(max_angle_deg))
    if angle <= max_angle_rad:
        return np.asarray(rotation[:3, :3], dtype=float)
    return Rotation.from_rotvec(rotvec * (max_angle_rad / angle)).as_matrix()


def residual_summary(residuals: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    translation_mm = np.asarray([np.linalg.norm(r["translation"]) * 1000.0 for r in residuals], dtype=float)
    rotation_deg = np.asarray([np.linalg.norm(r["rotation"]) * 180.0 / math.pi for r in residuals], dtype=float)
    return {
        "translation_mm": numeric_stats(translation_mm),
        "rotation_deg": numeric_stats(rotation_deg),
    }


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def monotonic_time_series_summary(times: list[float], target_hz: float | None = None) -> dict[str, Any]:
    values = np.asarray([float(value) for value in times if math.isfinite(float(value))], dtype=float)
    if values.size == 0:
        return {"count": 0}
    duration = float(max(0.0, values[-1] - values[0])) if values.size >= 2 else 0.0
    hz = float((values.size - 1) / duration) if duration > 1e-9 and values.size >= 2 else None
    gaps = np.diff(values) if values.size >= 2 else np.asarray([], dtype=float)
    payload: dict[str, Any] = {
        "count": int(values.size),
        "durationSeconds": duration,
        "effectiveHz": hz,
        "firstPerfCounterSeconds": float(values[0]),
        "lastPerfCounterSeconds": float(values[-1]),
    }
    if target_hz is not None and target_hz > 0:
        payload["targetHz"] = float(target_hz)
        if hz is not None:
            payload["targetRatio"] = float(hz / float(target_hz))
    if gaps.size:
        payload["gapSeconds"] = numeric_stats(gaps)
    return payload


def duration_series_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    return numeric_stats(finite)


def camera_intrinsics(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    coeffs = np.asarray(camera.get("distortion_coeffs") or [0, 0, 0, 0, 0], dtype=float).reshape(-1, 1)
    return matrix, coeffs


def object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid.astype(np.float32) * float(square_size_m)
    return points


def vec3_array(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        result = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(result)):
        return None
    return result


def clamp_vector_norm(value: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return np.zeros(3, dtype=float)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= max_norm:
        return np.asarray(value, dtype=float)
    return np.asarray(value, dtype=float) * (max_norm / norm)


def observations_to_json(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for obs in observations:
        rows.append(
            {
                "sample_index": obs.get("sample_index"),
                "quest_sample_index": obs.get("quest_sample_index"),
                "quest_recording_timestamp_seconds": obs.get("quest_recording_timestamp_seconds"),
                "image": obs.get("image"),
                "overlay": obs.get("overlay"),
                "method": obs.get("method"),
                "selected": obs.get("selected"),
                "appearance_anchor": obs.get("appearance_anchor"),
                "candidates": obs.get("candidates"),
            }
        )
    return rows


def first_numeric_state_list(states: Any, names: list[str]) -> list[float] | None:
    for name in names:
        if not hasattr(states, name):
            continue
        try:
            values = np.asarray(getattr(states, name), dtype=float).reshape(-1)
        except Exception:
            continue
        if values.size > 0:
            return [float(v) for v in values]
    return None


def default_rizon4_joint_limits() -> list[dict[str, Any]]:
    urdf_path = Path(__file__).resolve().parents[1] / "assets" / "urdf" / "flexiv_Rizon4_kinematics.urdf"
    limits = parse_urdf_joint_limits(urdf_path)
    if limits:
        return limits
    return [
        {"name": "joint1", "lower": -2.7925, "upper": 2.7925, "hardLower": -2.8798, "hardUpper": 2.8798},
        {"name": "joint2", "lower": -2.2689, "upper": 2.2689, "hardLower": -2.3562, "hardUpper": 2.3562},
        {"name": "joint3", "lower": -2.9671, "upper": 2.9671, "hardLower": -3.0543, "hardUpper": 3.0543},
        {"name": "joint4", "lower": -1.8675, "upper": 2.6878, "hardLower": -1.9548, "hardUpper": 2.7751},
        {"name": "joint5", "lower": -2.9671, "upper": 2.9671, "hardLower": -3.0543, "hardUpper": 3.0543},
        {"name": "joint6", "lower": -1.3963, "upper": 4.5379, "hardLower": -1.4835, "hardUpper": 4.6251},
        {"name": "joint7", "lower": -2.9671, "upper": 2.9671, "hardLower": -3.0543, "hardUpper": 3.0543},
    ]


def parse_urdf_joint_limits(urdf_path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    limits: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        if str(joint.attrib.get("type") or "") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        hard_lower = safe_float(limit.attrib.get("lower"))
        hard_upper = safe_float(limit.attrib.get("upper"))
        if hard_lower is None or hard_upper is None:
            continue
        safety = joint.find("safety_controller")
        soft_lower = safe_float(safety.attrib.get("soft_lower_limit")) if safety is not None else None
        soft_upper = safe_float(safety.attrib.get("soft_upper_limit")) if safety is not None else None
        lower = soft_lower if soft_lower is not None else hard_lower
        upper = soft_upper if soft_upper is not None else hard_upper
        limits.append(
            {
                "name": str(joint.attrib.get("name") or f"joint{len(limits) + 1}"),
                "lower": float(lower),
                "upper": float(upper),
                "hardLower": float(hard_lower),
                "hardUpper": float(hard_upper),
            }
        )
    return limits


def joint_limit_guard_state(
    joint_pose: list[float] | None,
    joint_limits: list[dict[str, Any]],
    buffer_rad: float,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "ok": True, "reason": "disabled"}
    buffer = max(0.0, float(buffer_rad))
    if not isinstance(joint_pose, list) or len(joint_pose) < len(joint_limits):
        return {
            "enabled": True,
            "ok": False,
            "reason": "missing_joint_pose",
            "bufferRad": buffer,
            "jointCount": len(joint_pose) if isinstance(joint_pose, list) else 0,
            "limitCount": len(joint_limits),
        }
    rows = []
    violations = []
    min_margin = math.inf
    for index, limit in enumerate(joint_limits):
        value = float(joint_pose[index])
        if not math.isfinite(value):
            return {
                "enabled": True,
                "ok": False,
                "reason": "invalid_joint_pose",
                "bufferRad": buffer,
                "jointIndex": index,
                "jointName": limit.get("name") or f"joint{index + 1}",
                "value": value,
            }
        lower = float(limit["lower"])
        upper = float(limit["upper"])
        safe_lower = lower + buffer
        safe_upper = upper - buffer
        if safe_lower > safe_upper:
            return {
                "enabled": True,
                "ok": False,
                "reason": "invalid_joint_limit_buffer",
                "bufferRad": buffer,
                "jointIndex": index,
                "jointName": limit.get("name") or f"joint{index + 1}",
                "lower": lower,
                "upper": upper,
                "safeLower": safe_lower,
                "safeUpper": safe_upper,
            }
        lower_margin = value - safe_lower
        upper_margin = safe_upper - value
        margin = min(lower_margin, upper_margin)
        min_margin = min(min_margin, margin)
        row = {
            "index": index,
            "name": limit.get("name") or f"joint{index + 1}",
            "value": value,
            "lower": lower,
            "upper": upper,
            "hardLower": limit.get("hardLower"),
            "hardUpper": limit.get("hardUpper"),
            "safeLower": safe_lower,
            "safeUpper": safe_upper,
            "marginRad": float(margin),
        }
        if value < safe_lower or value > safe_upper:
            row["violation"] = "lower" if value < safe_lower else "upper"
            violations.append(row)
        rows.append(row)
    return {
        "enabled": True,
        "ok": not violations,
        "reason": "ok" if not violations else "joint_limit_buffer",
        "bufferRad": buffer,
        "minMarginRad": float(min_margin) if math.isfinite(min_margin) else None,
        "joints": rows,
        "violations": violations,
    }


def compact_joint_limit_guard_for_record(guard: Any) -> dict[str, Any] | None:
    if not isinstance(guard, dict):
        return None
    compact = {
        "enabled": guard.get("enabled"),
        "ok": guard.get("ok"),
        "reason": guard.get("reason"),
        "bufferRad": guard.get("bufferRad"),
        "minMarginRad": guard.get("minMarginRad"),
    }
    violations = guard.get("violations")
    if isinstance(violations, list) and violations:
        compact["violations"] = violations
    return compact


def compact_robot_state_for_record(state: dict[str, Any]) -> dict[str, Any]:
    result = {
        "ok": state.get("ok"),
        "robotSn": state.get("robotSn"),
        "poseField": state.get("poseField"),
        "readUnixSeconds": state.get("readUnixSeconds"),
        "rawEndEffectorPoseWxyz": state.get("rawEndEffectorPoseWxyz"),
        "jointPose": state.get("jointPose"),
        "jointLimitGuard": compact_joint_limit_guard_for_record(state.get("jointLimitGuard")),
        "gripper": state.get("gripper"),
    }
    for key in ("flange_pose", "tcp_pose"):
        pose = compact_robot_pose_for_record(state.get(key))
        if pose is not None:
            result[key] = pose
    return result


def compact_robot_pose_for_record(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    pose = value.get("pose")
    if isinstance(pose, list) and len(pose) >= 7:
        try:
            result["pose"] = [float(pose[index]) for index in range(7)]
        except (TypeError, ValueError):
            pass
    matrix = transform_from_json(value.get("T_base_pose"))
    if matrix is not None:
        result["T_base_pose"] = transform_to_compact_json(matrix)
    return result or None


def save_bgr_jpeg(bgr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_bgr_jpeg(bgr, quality=92))


def encode_bgr_jpeg(bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("could not encode image")
    return bytes(buffer)


def save_rgb_jpeg(rgb: np.ndarray, path: Path) -> None:
    save_bgr_jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), path)


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    return encode_bgr_jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), quality=quality)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
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


def read_json_if_exists(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def config_from_json(payload: Any, fallback: FlexivRealSenseConfig | None = None) -> FlexivRealSenseConfig:
    config = copy.deepcopy(fallback) if fallback is not None else FlexivRealSenseConfig()
    if not isinstance(payload, dict):
        return config

    string_fields = (
        ("robotSn", "robot_sn"),
        ("poseField", "robot_pose_field"),
        ("cameraSerial", "camera_serial"),
        ("thirdCameraSerial", "third_camera_serial"),
        ("recordDepthFormat", "record_depth_format"),
        ("gripperDevice", "gripper_device"),
    )
    for key, attr in string_fields:
        if key in payload and payload[key] is not None:
            setattr(config, attr, str(payload[key]).strip())
    config.record_depth_format = normalize_depth_format(config.record_depth_format)

    if "flexivRdk" in payload:
        config.flexiv_rdk = Path(payload["flexivRdk"]) if payload["flexivRdk"] else None
    if "networkInterfaces" in payload:
        config.flexiv_network_interfaces = normalize_network_interfaces(payload["networkInterfaces"])

    int_fields = (
        ("width", "width"),
        ("height", "height"),
        ("fps", "fps"),
        ("recordDepthEveryNFrames", "record_depth_every_n_frames"),
        ("warmupFrames", "warmup_frames"),
        ("boardCheckWarmupFrames", "board_check_warmup_frames"),
        ("minHandEyeDetections", "min_hand_eye_detections"),
        ("handEyeMaxDiverseSamples", "hand_eye_max_diverse_samples"),
        ("handEyeMinDiverseSamples", "hand_eye_min_diverse_samples"),
    )
    for key, attr in int_fields:
        if key in payload and is_number(payload[key]):
            setattr(config, attr, int(payload[key]))
    config.record_depth_every_n_frames = max(1, int(config.record_depth_every_n_frames))

    required_float_fields = (
        ("robotStateHz", "robot_state_hz"),
        ("captureIntervalSeconds", "capture_interval_seconds"),
        ("squareSizeM", "square_size_m"),
        ("handEyeDiverseTranslationScaleM", "hand_eye_diverse_translation_scale_m"),
        ("handEyeDiverseRotationScaleDeg", "hand_eye_diverse_rotation_scale_deg"),
        ("handEyeDiverseMinScore", "hand_eye_diverse_min_score"),
        ("controllerTranslationScale", "controller_translation_scale"),
        ("controllerMaxStepM", "controller_max_step_m"),
        ("controllerMaxRotationStepDeg", "controller_max_rotation_step_deg"),
        ("controllerTargetUpdateHz", "controller_target_update_hz"),
        ("cartesianMaxLinearVelocityMps", "cartesian_max_linear_velocity_mps"),
        ("cartesianMaxAngularVelocityRadps", "cartesian_max_angular_velocity_radps"),
        ("cartesianMaxLinearAccelerationMps2", "cartesian_max_linear_acceleration_mps2"),
        ("cartesianMaxAngularAccelerationRadps2", "cartesian_max_angular_acceleration_radps2"),
        ("controllerJointLimitBufferRad", "controller_joint_limit_buffer_rad"),
        ("gripperOpenWidthM", "gripper_open_width_m"),
        ("gripperCloseWidthM", "gripper_close_width_m"),
        ("gripperSpeedMps", "gripper_speed_mps"),
        ("gripperForceN", "gripper_force_n"),
        ("gripperTriggerCloseThreshold", "gripper_trigger_close_threshold"),
        ("gripperTriggerOpenThreshold", "gripper_trigger_open_threshold"),
    )
    for key, attr in required_float_fields:
        if key in payload and is_number(payload[key]):
            setattr(config, attr, float(payload[key]))

    optional_float_fields = (
        ("realsenseExposure", "realsense_exposure"),
        ("realsenseGain", "realsense_gain"),
    )
    for key, attr in optional_float_fields:
        if key in payload:
            setattr(config, attr, float(payload[key]) if is_number(payload[key]) else None)

    bool_fields = (
        ("recordDepth", "record_depth"),
        ("recordDepthAlignToColor", "record_depth_align_to_color"),
        ("realsenseAutoExposure", "realsense_auto_exposure"),
        ("runHandEye", "run_hand_eye"),
        ("handEyeDisableDiverseSelection", "hand_eye_disable_diverse_selection"),
        ("controllerMotionEnabled", "controller_motion_enabled"),
        ("controllerJointLimitGuardEnabled", "controller_joint_limit_guard_enabled"),
        ("gripperEnabled", "gripper_enabled"),
        ("gripperInitOnEnable", "gripper_init_on_enable"),
    )
    for key, attr in bool_fields:
        if key in payload:
            setattr(config, attr, bool(payload[key]))

    pattern = payload.get("pattern")
    if isinstance(pattern, list) and len(pattern) >= 2 and is_number(pattern[0]) and is_number(pattern[1]):
        config.pattern_cols = int(pattern[0])
        config.pattern_rows = int(pattern[1])
    if config.third_camera_serial and config.third_camera_serial == config.camera_serial:
        config.third_camera_serial = ""
    config.cartesian_max_linear_velocity_mps = positive_cartesian_limit(
        config.cartesian_max_linear_velocity_mps,
        "cartesian max linear velocity",
    )
    config.cartesian_max_angular_velocity_radps = positive_cartesian_limit(
        config.cartesian_max_angular_velocity_radps,
        "cartesian max angular velocity",
    )
    config.cartesian_max_linear_acceleration_mps2 = positive_cartesian_limit(
        config.cartesian_max_linear_acceleration_mps2,
        "cartesian max linear acceleration",
    )
    config.cartesian_max_angular_acceleration_radps2 = positive_cartesian_limit(
        config.cartesian_max_angular_acceleration_radps2,
        "cartesian max angular acceleration",
    )
    return config


def config_to_json(config: FlexivRealSenseConfig) -> dict[str, Any]:
    return {
        "robotSn": config.robot_sn,
        "poseField": config.robot_pose_field,
        "flexivRdk": str(config.flexiv_rdk) if config.flexiv_rdk is not None else None,
        "networkInterfaces": normalize_network_interfaces(config.flexiv_network_interfaces),
        "cameraSerial": config.camera_serial,
        "thirdCameraSerial": config.third_camera_serial,
        "width": config.width,
        "height": config.height,
        "fps": config.fps,
        "robotStateHz": config.robot_state_hz,
        "recordDepth": config.record_depth,
        "recordDepthAlignToColor": config.record_depth_align_to_color,
        "recordDepthEveryNFrames": config.record_depth_every_n_frames,
        "recordDepthFormat": normalize_depth_format(config.record_depth_format),
        "warmupFrames": config.warmup_frames,
        "realsenseAutoExposure": config.realsense_auto_exposure,
        "realsenseExposure": config.realsense_exposure,
        "realsenseGain": config.realsense_gain,
        "boardCheckWarmupFrames": config.board_check_warmup_frames,
        "captureIntervalSeconds": config.capture_interval_seconds,
        "pattern": [config.pattern_cols, config.pattern_rows],
        "squareSizeM": config.square_size_m,
        "runHandEye": config.run_hand_eye,
        "minHandEyeDetections": config.min_hand_eye_detections,
        "handEyeMaxDiverseSamples": config.hand_eye_max_diverse_samples,
        "handEyeMinDiverseSamples": config.hand_eye_min_diverse_samples,
        "handEyeDiverseTranslationScaleM": config.hand_eye_diverse_translation_scale_m,
        "handEyeDiverseRotationScaleDeg": config.hand_eye_diverse_rotation_scale_deg,
        "handEyeDiverseMinScore": config.hand_eye_diverse_min_score,
        "handEyeDisableDiverseSelection": config.hand_eye_disable_diverse_selection,
        "controllerMotionEnabled": config.controller_motion_enabled,
        "controllerTranslationScale": config.controller_translation_scale,
        "controllerWorkspaceLimitEnabled": False,
        "controllerMaxOffsetM": None,
        "controllerMaxStepM": config.controller_max_step_m,
        "controllerMaxRotationDeg": None,
        "controllerMaxRotationStepDeg": config.controller_max_rotation_step_deg,
        "controllerTargetUpdateHz": config.controller_target_update_hz,
        "cartesianMaxLinearVelocityMps": config.cartesian_max_linear_velocity_mps,
        "cartesianMaxAngularVelocityRadps": config.cartesian_max_angular_velocity_radps,
        "cartesianMaxLinearAccelerationMps2": config.cartesian_max_linear_acceleration_mps2,
        "cartesianMaxAngularAccelerationRadps2": config.cartesian_max_angular_acceleration_radps2,
        "controllerJointLimitBufferRad": config.controller_joint_limit_buffer_rad,
        "controllerJointLimitGuardEnabled": config.controller_joint_limit_guard_enabled,
        "gripperEnabled": config.gripper_enabled,
        "gripperDevice": config.gripper_device,
        "gripperOpenWidthM": config.gripper_open_width_m,
        "gripperCloseWidthM": config.gripper_close_width_m,
        "gripperSpeedMps": config.gripper_speed_mps,
        "gripperForceN": config.gripper_force_n,
        "gripperTriggerCloseThreshold": config.gripper_trigger_close_threshold,
        "gripperTriggerOpenThreshold": config.gripper_trigger_open_threshold,
        "gripperInitOnEnable": config.gripper_init_on_enable,
    }


def normalize_quaternion(value: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in value[:4]))
    if not math.isfinite(norm) or norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / norm for v in value[:4]]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clamp_optional_range(value: float, lower: Any, upper: Any) -> float:
    result = float(value)
    low = safe_float(lower)
    high = safe_float(upper)
    if low is not None:
        result = max(low, result)
    if high is not None:
        result = min(high, result)
    return result


def normalize_network_interfaces(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]
