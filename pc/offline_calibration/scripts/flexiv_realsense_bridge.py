from __future__ import annotations

import json
import math
import os
import sys
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
DEFAULT_CAPTURE_INTERVAL_SECONDS = 0.35
DEFAULT_FLEXIV_RDK_ROOT: Path | None = None
DEFAULT_FLEXIV_ROBOT_SN = "Rizon4-062713"
DEFAULT_END_CAMERA_SERIAL = "750612070265"
MIN_HAND_EYE_EE_TRANSLATION_SPAN_M = 0.02
MIN_HAND_EYE_EE_ROTATION_SPAN_DEG = 2.0
DEFAULT_CONTROLLER_TRANSLATION_SCALE = 1.0
DEFAULT_CONTROLLER_MAX_OFFSET_M = 0.18
DEFAULT_CONTROLLER_MAX_STEP_M = 0.025
DEFAULT_CONTROLLER_MAX_ROTATION_DEG = 30.0
DEFAULT_CONTROLLER_MAX_ROTATION_STEP_DEG = 2.0
DEFAULT_CONTROLLER_JOINT_LIMIT_BUFFER_RAD = 0.08
DEFAULT_HAND_EYE_MAX_DIVERSE_SAMPLES = 120
DEFAULT_HAND_EYE_MIN_DIVERSE_SAMPLES = 20
DEFAULT_HAND_EYE_DIVERSE_TRANSLATION_SCALE_M = 0.02
DEFAULT_HAND_EYE_DIVERSE_ROTATION_SCALE_DEG = 3.0
DEFAULT_HAND_EYE_DIVERSE_MIN_SCORE = 0.75
DEFAULT_GRIPPER_DEVICE = "gripper"
DEFAULT_GRIPPER_OPEN_WIDTH_M = 0.08
DEFAULT_GRIPPER_CLOSE_WIDTH_M = 0.0
DEFAULT_GRIPPER_SPEED_MPS = 0.04
DEFAULT_GRIPPER_FORCE_N = 20.0
DEFAULT_GRIPPER_TRIGGER_CLOSE_THRESHOLD = 0.65
DEFAULT_GRIPPER_TRIGGER_OPEN_THRESHOLD = 0.25
MAX_HAND_EYE_CAMERA_OFFSET_M = 0.50
MAX_HAND_EYE_TRANSLATION_MEDIAN_RESIDUAL_MM = 15.0
MAX_HAND_EYE_TRANSLATION_P95_RESIDUAL_MM = 35.0
RED_ANCHOR_RADIUS_GRID_SPACING = 2.0
RED_ANCHOR_MIN_PIXELS = 16
RED_ANCHOR_MIN_RATIO = 0.002
RED_ANCHOR_MIN_BEST_SECOND_RATIO = 1.8
APPEARANCE_ANCHOR_MIN_CONTRAST = 18.0
QUEST_TO_ROBOT_UNALIGNED_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
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
DEFAULT_FREEDRIVE_PLANS = ("PLAN-FreeDriveManual", "PLAN-FreeDriveAuto")


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
    controller_max_offset_m: float = DEFAULT_CONTROLLER_MAX_OFFSET_M
    controller_max_step_m: float = DEFAULT_CONTROLLER_MAX_STEP_M
    controller_max_rotation_deg: float = DEFAULT_CONTROLLER_MAX_ROTATION_DEG
    controller_max_rotation_step_deg: float = DEFAULT_CONTROLLER_MAX_ROTATION_STEP_DEG
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


class FlexivRobotClient:
    def __init__(self) -> None:
        self.robot: Any | None = None
        self.robot_sn: str | None = None
        self.pose_field = "flange_pose"
        self.lock = threading.Lock()
        self.last_error: str | None = None
        self.motion_armed = False
        self.motion_last_target_pose: list[float] | None = None
        self.freedrive_enabled = False
        self.freedrive_method: str | None = None
        self.freedrive_plan: str | None = None
        self.gripper: Any | None = None
        self.gripper_enabled = False
        self.gripper_device: str | None = None
        self.gripper_last_error: str | None = None
        self.joint_limits = default_rizon4_joint_limits()

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
            return self.read_state_locked()

    def disconnect(self) -> None:
        self.disarm_motion_locked()
        self.disable_freedrive_locked()
        self.disable_gripper_locked()
        self.robot = None
        self.robot_sn = None

    def status(self) -> dict[str, Any]:
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
                "gripper": self.gripper_status_locked(),
            }
            if connected:
                try:
                    payload["state"] = self.read_state_locked()
                except Exception as exc:  # pragma: no cover - hardware path
                    self.last_error = str(exc)
                    payload["lastError"] = self.last_error
            return payload

    def read_state(self) -> dict[str, Any]:
        with self.lock:
            return self.read_state_locked()

    def read_state_locked(self) -> dict[str, Any]:
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

    def read_tcp_pose(self) -> list[float]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            return [float(v) for v in self.robot.states().tcp_pose]

    def arm_motion(self) -> dict[str, Any]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            self.disable_freedrive_locked()
            flexivrdk = sys.modules.get("flexivrdk") or import_flexivrdk(None)
            robot = self.robot
            if robot.fault():
                raise RuntimeError("Flexiv robot has fault; clear it before arming controller motion")
            robot.Enable()
            start = time.time()
            while not robot.operational():
                if time.time() - start > 5.0:
                    raise RuntimeError("Timed out waiting for Flexiv robot to become operational")
                time.sleep(0.05)
            robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            robot.SetForceControlAxis([False, False, False, False, False, False])
            self.motion_armed = True
            self.motion_last_target_pose = [float(v) for v in robot.states().tcp_pose]
            return self.read_state_locked()

    def enable_freedrive(self) -> dict[str, Any]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            self.disarm_motion_locked()
            flexivrdk = sys.modules.get("flexivrdk") or import_flexivrdk(None)
            robot = self.robot
            if robot.fault():
                raise RuntimeError("Flexiv robot has fault; clear it before enabling free-drag mode")
            robot.Enable()
            start = time.time()
            while not robot.operational():
                if time.time() - start > 5.0:
                    raise RuntimeError("Timed out waiting for Flexiv robot to become operational")
                time.sleep(0.05)
            plan_errors: list[str] = []
            try:
                available_plans = set(str(row) for row in robot.plan_list())
            except Exception as exc:
                available_plans = set()
                plan_errors.append(f"plan_list failed: {exc}")
            for plan_name in DEFAULT_FREEDRIVE_PLANS:
                if available_plans and plan_name not in available_plans:
                    continue
                try:
                    robot.SwitchMode(flexivrdk.Mode.NRT_PLAN_EXECUTION)
                    robot.ExecutePlan(plan_name, False, True)
                    self.freedrive_enabled = True
                    self.freedrive_method = "plan"
                    self.freedrive_plan = plan_name
                    self.motion_armed = False
                    self.motion_last_target_pose = None
                    return self.read_state_locked()
                except Exception as exc:
                    plan_errors.append(f"{plan_name}: {exc}")
            try:
                robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_IMPEDANCE)
                joint_pose = self.read_joint_pose_locked()
                dof = len(joint_pose) if joint_pose else len(self.joint_limits) or 7
                robot.SetJointImpedance([0.0] * dof)
                self.freedrive_method = "nrt_joint_impedance_zero_stiffness"
                self.freedrive_plan = None
            except Exception as exc:
                detail = "; ".join(plan_errors + [f"impedance fallback: {exc}"])
                raise RuntimeError(f"Failed to enable Flexiv free-drag mode ({detail})") from exc
            self.freedrive_enabled = True
            self.motion_armed = False
            self.motion_last_target_pose = None
            return self.read_state_locked()

    def disable_freedrive(self) -> dict[str, Any]:
        with self.lock:
            self.disable_freedrive_locked()
            return self.status_unlocked()

    def disable_freedrive_locked(self) -> None:
        if self.robot is not None and self.freedrive_enabled:
            try:
                self.robot.Stop()
            except Exception:
                pass
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
                self.robot.Stop()
            except Exception:
                pass
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
            "gripper": self.gripper_status_locked(),
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
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            if not self.motion_armed:
                return {"enabled": bool(joint_limit_guard_enabled), "ok": False, "reason": "motion_not_armed"}
            if len(target_pose_wxyz) < 7:
                raise ValueError("target pose must be [x,y,z,qw,qx,qy,qz]")
            target = [float(v) for v in target_pose_wxyz[:7]]
            joint_pose = self.read_joint_pose_locked()
            guard = joint_limit_guard_state(
                joint_pose,
                self.joint_limits,
                joint_limit_buffer_rad,
                joint_limit_guard_enabled,
            )
            if not bool(guard.get("ok")):
                return guard
            self.robot.SendCartesianMotionForce(target)
            self.motion_last_target_pose = target
            return guard

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

    def enable_gripper(self, device_name: str) -> dict[str, Any]:
        with self.lock:
            return self.enable_gripper_locked(device_name)

    def enable_gripper_locked(self, device_name: str) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Flexiv robot is not connected")
        flexivrdk = sys.modules.get("flexivrdk") or import_flexivrdk(None)
        device = str(device_name or DEFAULT_GRIPPER_DEVICE).strip() or DEFAULT_GRIPPER_DEVICE
        self.gripper = flexivrdk.Gripper(self.robot)
        self.gripper.Enable(device)
        try:
            self.gripper.Init()
        except Exception:
            # Some gripper configurations are already initialized after Enable.
            pass
        self.gripper_enabled = True
        self.gripper_device = device
        self.gripper_last_error = None
        return self.gripper_status_locked()

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

    def gripper_status_locked(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.gripper_enabled,
            "device": self.gripper_device,
            "lastError": self.gripper_last_error,
        }
        if self.gripper is not None:
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

    def gripper_status(self) -> dict[str, Any]:
        with self.lock:
            return self.gripper_status_locked()

    def move_gripper(self, width_m: float, speed_mps: float, force_n: float) -> dict[str, Any]:
        with self.lock:
            if self.robot is None:
                raise RuntimeError("Flexiv robot is not connected")
            if self.gripper is None:
                self.enable_gripper_locked(str(self.gripper_device or DEFAULT_GRIPPER_DEVICE))
            assert self.gripper is not None
            width = float(width_m)
            speed = float(speed_mps)
            force = float(force_n)
            self.gripper.Move(width, speed, force)
            self.gripper_last_error = None
            return self.gripper_status_locked()


class RealSenseColorCamera:
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.profile: Any | None = None
        self.serial: str | None = None
        self.width = DEFAULT_REALSENSE_WIDTH
        self.height = DEFAULT_REALSENSE_HEIGHT
        self.fps = DEFAULT_REALSENSE_FPS
        self.metadata: dict[str, Any] | None = None

    def start(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        auto_exposure: bool = True,
        exposure: float | None = None,
        gain: float | None = None,
    ) -> dict[str, Any]:
        if self.pipeline is not None and self.serial == serial and self.width == width and self.height == height and self.fps == fps:
            return self.metadata or {}
        self.stop()
        import pyrealsense2 as rs  # type: ignore[import-not-found]

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.rgb8, int(fps))
        profile = pipeline.start(config)
        color_options = configure_color_sensor(profile, auto_exposure, exposure, gain)
        self.pipeline = pipeline
        self.profile = profile
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.metadata = camera_metadata_from_profile(profile, serial)
        self.metadata["colorOptions"] = color_options
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

    def capture_rgb(self, warmup_frames: int) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("RealSense camera is not started")
        frame = None
        for _ in range(max(1, int(warmup_frames))):
            frames = self.pipeline.wait_for_frames(5000)
            frame = frames.get_color_frame()
        if not frame:
            raise RuntimeError("No RealSense color frame received")
        return np.asanyarray(frame.get_data()).copy()


class RealSenseStreamHub:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.cameras: dict[str, RealSenseColorCamera] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, dict[str, Any]] = {}
        self.signature: tuple[Any, ...] | None = None
        self.stop_event: threading.Event | None = None
        self.thread: threading.Thread | None = None
        self.frame_count = 0
        self.last_error: str | None = None

    def start(self, config: FlexivRealSenseConfig) -> dict[str, Any]:
        signature = self._signature(config)
        with self.lock:
            if self.thread is not None and self.thread.is_alive() and self.signature == signature:
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
        thread = threading.Thread(target=self._capture_loop, args=(stop_event,), name="realsense-stream", daemon=True)
        with self.lock:
            self.cameras = cameras
            self.metadata = metadata
            self.latest = {}
            self.signature = signature
            self.stop_event = stop_event
            self.thread = thread
            self.frame_count = 0
            self.last_error = None
        thread.start()
        return self.status()

    def stop(self) -> None:
        with self.lock:
            stop_event = self.stop_event
            thread = self.thread
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self.lock:
            cameras = self.cameras
            self.cameras = {}
            self.metadata = {}
            self.latest = {}
            self.signature = None
            self.stop_event = None
            self.thread = None
        for camera in cameras.values():
            camera.stop()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.status_locked()

    def status_locked(self) -> dict[str, Any]:
        running = self.thread is not None and self.thread.is_alive()
        return {
            "running": running,
            "roles": sorted(self.cameras.keys()),
            "metadata": self.metadata,
            "frameCount": self.frame_count,
            "latest": {
                role: {
                    "capturedAtUtc": frame.get("capturedAtUtc"),
                    "serial": frame.get("serial"),
                    "sequence": frame.get("sequence"),
                }
                for role, frame in self.latest.items()
            },
            "lastError": self.last_error,
        }

    def get_latest(self, role: str = "end", wait_timeout: float = 2.0) -> dict[str, Any]:
        deadline = time.perf_counter() + max(0.0, float(wait_timeout))
        while True:
            with self.lock:
                frame = self.latest.get(role)
                running = self.thread is not None and self.thread.is_alive()
                last_error = self.last_error
            if frame is not None:
                return {
                    **frame,
                    "rgb": frame["rgb"].copy(),
                    "jpeg": bytes(frame["jpeg"]),
                    "metadata": dict(frame["metadata"]),
                }
            if not running:
                raise RuntimeError(last_error or "RealSense stream is not running")
            if time.perf_counter() >= deadline:
                raise RuntimeError(f"No {role} RealSense frame is available yet")
            time.sleep(0.02)

    def snapshot_roles(self, roles: list[str], wait_timeout: float = 2.0) -> dict[str, dict[str, Any]]:
        return {role: self.get_latest(role, wait_timeout=wait_timeout) for role in roles}

    def _capture_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self.lock:
                cameras = list(self.cameras.items())
            for role, camera in cameras:
                if stop_event.is_set():
                    return
                try:
                    rgb = camera.capture_rgb(1)
                    jpeg = encode_rgb_jpeg(rgb, quality=86)
                    captured_at = datetime.now(timezone.utc).isoformat()
                    with self.lock:
                        self.frame_count += 1
                        self.latest[role] = {
                            "role": role,
                            "serial": camera.serial,
                            "metadata": self.metadata.get(role, {}),
                            "rgb": rgb,
                            "jpeg": jpeg,
                            "capturedAtUtc": captured_at,
                            "sequence": self.frame_count,
                        }
                        self.last_error = None
                except Exception as exc:  # pragma: no cover - hardware path
                    with self.lock:
                        self.last_error = str(exc)
                    time.sleep(0.1)

    def _signature(self, config: FlexivRealSenseConfig) -> tuple[Any, ...]:
        serials = tuple(sorted(self._camera_serials(config).items()))
        return (
            serials,
            int(config.width),
            int(config.height),
            int(config.fps),
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
        self.directory = self.root / "robot_realsense"
        self.image_dir = self.directory / "images"
        self.video_dir = self.directory / "videos"
        self.samples_path = self.directory / "samples.jsonl"
        self.motion_path = self.directory / "controller_motion.jsonl"
        self.gripper_path = self.directory / "gripper_commands.jsonl"
        self.video_frames_path = self.directory / "video_frames.jsonl"
        self.summary_path = self.directory / "session_summary.json"
        self.samples_handle: Any | None = None
        self.motion_handle: Any | None = None
        self.gripper_handle: Any | None = None
        self.video_frames_handle: Any | None = None
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.video_paths: dict[str, str] = {}
        self.video_frame_counts: dict[str, int] = {}
        self.lock = threading.Lock()
        self.next_capture_perf = 0.0
        self.sample_count = 0
        self.image_count = 0
        self.video_frame_count = 0
        self.error_count = 0
        self.closed = False
        self.last_error: str | None = None
        self.camera_metadata: dict[str, dict[str, Any]] = {}
        self.controller_anchor_world: np.ndarray | None = None
        self.controller_anchor_rotation_world: np.ndarray | None = None
        self.robot_anchor_tcp_pose: list[float] | None = None
        self.motion_command_count = 0
        self.motion_skip_count = 0
        self.motion_error_count = 0
        self.last_motion_event: dict[str, Any] | None = None
        self.gripper_command_count = 0
        self.gripper_skip_count = 0
        self.gripper_error_count = 0
        self.last_gripper_event: dict[str, Any] | None = None
        self.last_gripper_closed: bool | None = None
        self.ee_pose_history: list[np.ndarray] = []
        self.robot_alignment_result = robot_alignment_result if isinstance(robot_alignment_result, dict) else None
        self.require_controller_alignment = require_controller_alignment
        self.t_ee_end_camera = self._alignment_transform("end_camera", "T_ee_realsense")
        self.t_base_world = self._alignment_transform("questAlignment", "T_base_world")

    def start(self) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.samples_handle = self.samples_path.open("a", encoding="utf-8", newline="\n")
        self.motion_handle = self.motion_path.open("a", encoding="utf-8", newline="\n")
        self.gripper_handle = self.gripper_path.open("a", encoding="utf-8", newline="\n")
        self.video_frames_handle = self.video_frames_path.open("a", encoding="utf-8", newline="\n")
        config_payload = config_to_json(self.config)
        config_payload["recordId"] = self.record_id
        config_payload["controlMode"] = self.control_mode
        config_payload["startedAtUtc"] = datetime.now(timezone.utc).isoformat()
        write_json(config_payload, self.directory / "capture_config.json")
        stream_status = self.stream_hub.start(self.config)
        metadata = stream_status.get("metadata") if isinstance(stream_status, dict) else None
        self.camera_metadata = metadata if isinstance(metadata, dict) else {}
        write_json(self.camera_metadata, self.directory / "cameras.json")
        if self.robot_alignment_result is not None:
            write_json(self.robot_alignment_result, self.directory / "robot_hand_eye_result.json")
        if self.config.gripper_enabled:
            self._try_initialize_gripper()
        summary = self.summary("recording")
        self._publish({"type": "robot_status", "stage": "robot_realsense_recording", **summary})
        return summary

    def update_controller_motion(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
        if self.control_mode != ROBOT_SESSION_CONTROL_TELEOP:
            return None
        with self.lock:
            if self.closed:
                return None
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
        with self.lock:
            self.last_motion_event = event
            if self.motion_handle is not None:
                self.motion_handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                self.motion_handle.flush()
        if event.get("ok"):
            self.motion_command_count += 1
        else:
            self.motion_skip_count += 1
        self._publish(robot_motion_event(event))
        return event

    def update_gripper(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
        if not self.config.gripper_enabled:
            return None
        with self.lock:
            if self.closed:
                return None
        try:
            event = self._update_gripper_unlocked(quest_sample)
        except Exception as exc:  # pragma: no cover - hardware path
            self.gripper_error_count += 1
            event = {
                "ok": False,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "quest_sample_index": quest_sample.get("sampleIndex"),
                "error": str(exc),
            }
        with self.lock:
            self.last_gripper_event = event
            if self.gripper_handle is not None:
                self.gripper_handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                self.gripper_handle.flush()
        if event.get("ok") and event.get("commandSent"):
            self.gripper_command_count += 1
        else:
            self.gripper_skip_count += 1
        self._publish(robot_gripper_event(event))
        return event

    def record_sample(self, quest_sample: dict[str, Any]) -> dict[str, Any] | None:
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
            assert self.samples_handle is not None
            self.samples_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.samples_handle.flush()
            self.sample_count += 1
            if row.get("ok"):
                images = row.get("images") if isinstance(row.get("images"), dict) else {}
                self.image_count += len(images)
                t_base_ee = transform_from_json(row.get("T_base_ee"))
                if t_base_ee is not None:
                    self.ee_pose_history.append(t_base_ee)
                    row["poseDiversity"] = ee_pose_diversity(self.ee_pose_history)
            self._publish(robot_sample_event(row))
            return row

    def close(self) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return self.summary("already_closed")
            self.closed = True
            if self.samples_handle is not None:
                self.samples_handle.close()
                self.samples_handle = None
            if self.motion_handle is not None:
                self.motion_handle.close()
                self.motion_handle = None
            if self.gripper_handle is not None:
                self.gripper_handle.close()
                self.gripper_handle = None
            if self.video_frames_handle is not None:
                self.video_frames_handle.close()
                self.video_frames_handle = None
            for writer in self.video_writers.values():
                writer.release()
            self.video_writers.clear()
            summary = self.summary("closed")
            write_json(summary, self.summary_path)
            self._publish({"type": "robot_status", "stage": "robot_realsense_closed", **summary})
            return summary

    def summary(self, stage: str) -> dict[str, Any]:
        return {
            "ok": self.error_count == 0,
            "recordId": self.record_id,
            "stage": stage,
            "directory": str(self.directory),
            "cameraSerial": self.config.camera_serial,
            "thirdCameraSerial": self.config.third_camera_serial,
            "cameraRoles": sorted(self.camera_metadata.keys()),
            "samples": self.sample_count,
            "images": self.image_count,
            "videos": self.video_paths,
            "videoFrames": self.video_frame_count,
            "errors": self.error_count,
            "motionCommands": self.motion_command_count,
            "motionSkips": self.motion_skip_count,
            "motionErrors": self.motion_error_count,
            "lastMotion": self.last_motion_event,
            "controlMode": self.control_mode,
            "controllerAlignmentRequired": self.require_controller_alignment,
            "controllerAlignmentAvailable": self.t_base_world is not None,
            "gripperEnabled": self.config.gripper_enabled,
            "gripperCommands": self.gripper_command_count,
            "gripperSkips": self.gripper_skip_count,
            "gripperErrors": self.gripper_error_count,
            "lastGripper": self.last_gripper_event,
            "poseDiversity": ee_pose_diversity(self.ee_pose_history),
            "lastError": self.last_error,
        }

    def _capture_row(self, quest_sample: dict[str, Any]) -> dict[str, Any]:
        sample_index = self.sample_count
        robot_state = self.robot.read_state()
        robot_state["jointLimitGuard"] = self.robot.joint_limit_guard(
            self.config.controller_joint_limit_buffer_rad,
            self.config.controller_joint_limit_guard_enabled,
        )
        images: dict[str, str] = {}
        videos: dict[str, dict[str, Any]] = {}
        video_frames: dict[str, dict[str, Any]] = {}
        frames = self.stream_hub.snapshot_roles(sorted(self.camera_metadata.keys()), wait_timeout=2.0)
        for role, frame in frames.items():
            serial = frame.get("serial") or self.camera_serials().get(role) or role
            frame_index = self._write_video_frame(role, serial, frame)
            videos[role] = {
                "path": self.video_paths.get(role),
                "frameIndex": frame_index,
                "serial": serial,
            }
            video_frames[role] = {
                "role": role,
                "serial": serial,
                "frameIndex": frame_index,
                "capturedAtUtc": frame.get("capturedAtUtc"),
                "streamSequence": frame.get("sequence"),
            }
        t_base_ee = transform_from_json(robot_state.get("endEffectorPose"))
        t_base_tool_tcp = self._tool_tcp_transform(robot_state, t_base_ee)
        t_base_end_camera_payload = None
        t_world_tool_tcp_payload = None
        t_world_end_camera_payload = None
        t_display_tool_tcp_payload = None
        t_display_end_camera_payload = None
        if t_base_tool_tcp is not None:
            if self.t_ee_end_camera is not None:
                t_base_end_camera = t_base_tool_tcp @ self.t_ee_end_camera
                t_base_end_camera_payload = transform_to_json(t_base_end_camera)
            if self.t_base_world is not None:
                t_world_tool_tcp = invert_transform(self.t_base_world) @ t_base_tool_tcp
                t_world_tool_tcp_payload = transform_to_json(t_world_tool_tcp, PC_WORLD_FRAME)
                t_display_tool_tcp_payload = transform_to_json(t_world_tool_tcp, PC_WORLD_FRAME)
                if t_base_end_camera_payload is not None:
                    t_base_end_camera_matrix = transform_from_json(t_base_end_camera_payload)
                    if t_base_end_camera_matrix is not None:
                        t_world_end_camera = invert_transform(self.t_base_world) @ t_base_end_camera_matrix
                        t_world_end_camera_payload = transform_to_json(t_world_end_camera, PC_WORLD_FRAME)
                        t_display_end_camera_payload = transform_to_json(t_world_end_camera, PC_WORLD_FRAME)
        joint_pose = robot_state.get("jointPose")
        quest_gaze3d_pc_world = unity_vec3_to_pc(quest_sample.get("gazePoint3DWorld"))
        right_controller_pc = controller_payload_to_pc_world(quest_sample.get("rightController"))
        return {
            "sample_index": sample_index,
            "record_id": self.record_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "coordinate_frame": PC_WORLD_FRAME,
            "pose_source": f"flexiv:{self.config.robot_pose_field}:{self.config.robot_sn}",
            "quest_sample_index": quest_sample.get("sampleIndex"),
            "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
            "quest_pc_receive_perf_counter_seconds": quest_sample.get("pcReceivePerfCounterSeconds"),
            "quest_gaze3d_world": quest_sample.get("gazePoint3DWorld"),
            "quest_gaze3d_pc_world": quest_gaze3d_pc_world,
            "quest_gaze3d_source": quest_sample.get("gazePoint3DSource") or quest_sample.get("gazeSource"),
            "right_controller": visualizer_controller_payload(right_controller_pc),
            "right_controller_unity": visualizer_controller_payload(quest_sample.get("rightController")),
            "robot_state": robot_state,
            "T_base_ee": robot_state.get("endEffectorPose"),
            "T_base_tool_tcp": transform_to_json(t_base_tool_tcp) if t_base_tool_tcp is not None else None,
            "T_world_tool_tcp": t_world_tool_tcp_payload,
            "T_display_tool_tcp": t_display_tool_tcp_payload,
            "T_base_end_camera": t_base_end_camera_payload,
            "T_world_end_camera": t_world_end_camera_payload,
            "T_display_end_camera": t_display_end_camera_payload,
            "jointpose": joint_pose,
            "jointpos": joint_pose,
            "images": images,
            "videos": videos,
            "videoFrames": video_frames,
            "gripper": self.robot.gripper_status(),
        }

    def _write_video_frame(self, role: str, serial: str, frame: dict[str, Any]) -> int:
        rgb = frame["rgb"]
        if not isinstance(rgb, np.ndarray):
            raise RuntimeError(f"{role} frame has no RGB image")
        writer = self.video_writers.get(role)
        if writer is None:
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            rel_path = Path("videos") / f"{role}_{safe_filename(serial)}.mp4"
            path = self.directory / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            fps = max(1.0, 1.0 / max(1e-6, float(self.config.capture_interval_seconds)))
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"could not open robot camera video writer: {path}")
            self.video_writers[role] = writer
            self.video_paths[role] = str(rel_path).replace("\\", "/")
            self.video_frame_counts[role] = 0
        frame_index = int(self.video_frame_counts.get(role, 0))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.video_frame_counts[role] = frame_index + 1
        self.video_frame_count += 1
        row = {
            "sample_index": self.sample_count,
            "record_id": self.record_id,
            "role": role,
            "serial": serial,
            "frame_index": frame_index,
            "video": self.video_paths.get(role),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "frame_captured_at_utc": frame.get("capturedAtUtc"),
            "stream_sequence": frame.get("sequence"),
        }
        if self.video_frames_handle is not None:
            self.video_frames_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.video_frames_handle.flush()
        return frame_index

    def _tool_tcp_transform(self, robot_state: dict[str, Any], fallback: np.ndarray | None) -> np.ndarray | None:
        tcp = robot_state.get("tcp_pose")
        if isinstance(tcp, dict):
            matrix = transform_from_json(tcp.get("T_base_pose"))
            if matrix is not None:
                return matrix
        if self.config.robot_pose_field == "tcp_pose" and fallback is not None:
            return fallback
        return fallback

    def _publish(self, event: dict[str, Any]) -> None:
        if self.publish_event is not None:
            self.publish_event(event)

    def _try_initialize_gripper(self) -> None:
        try:
            status = self.robot.enable_gripper(self.config.gripper_device)
            event = {
                "ok": True,
                "commandSent": False,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "action": "initialize",
                "status": status,
            }
        except Exception as exc:  # pragma: no cover - hardware path
            self.gripper_error_count += 1
            event = {
                "ok": False,
                "commandSent": False,
                "record_id": self.record_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "action": "initialize",
                "error": str(exc),
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
        if not self.robot.motion_armed:
            self.robot.arm_motion()
            self.reset_controller_motion_anchor()
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
            mapping_mode = "unaligned_quest_y_to_robot_z"
        raw_rotation = mapping_rotation @ raw_rotation_world @ mapping_rotation.T
        offset = clamp_vector_norm(raw_offset, float(self.config.controller_max_offset_m))
        rotation_delta = clamp_rotation_angle(raw_rotation, float(self.config.controller_max_rotation_deg))
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
                "limits": {
                    "scale": self.config.controller_translation_scale,
                    "maxOffsetM": self.config.controller_max_offset_m,
                    "maxStepM": self.config.controller_max_step_m,
                    "maxRotationDeg": self.config.controller_max_rotation_deg,
                    "maxRotationStepDeg": self.config.controller_max_rotation_step_deg,
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
            "limits": {
                "scale": self.config.controller_translation_scale,
                "maxOffsetM": self.config.controller_max_offset_m,
                "maxStepM": self.config.controller_max_step_m,
                "maxRotationDeg": self.config.controller_max_rotation_deg,
                "maxRotationStepDeg": self.config.controller_max_rotation_step_deg,
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
            "status": self.robot.gripper_status(),
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
    def __init__(self, config: FlexivRealSenseConfig | None = None) -> None:
        self.config = config or FlexivRealSenseConfig()
        self.robot = FlexivRobotClient()
        self.lock = threading.RLock()
        self.stream_hub = RealSenseStreamHub()
        self.active_session: RobotRealsenseSession | None = None
        self.last_calibration: dict[str, Any] | None = None
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            active = self.active_session.summary("recording") if self.active_session is not None else None
            robot_status = self.robot.status()
            state = robot_status.get("state") if isinstance(robot_status, dict) else None
            if isinstance(state, dict) and robot_status.get("connected"):
                state["jointLimitGuard"] = self.robot.joint_limit_guard(
                    self.config.controller_joint_limit_buffer_rad,
                    self.config.controller_joint_limit_guard_enabled,
                )
            return {
                "ok": True,
                "enabled": True,
                "config": config_to_json(self.config),
                "robot": robot_status,
                "activeSession": active,
                "realsenseStream": self.stream_hub.status(),
                "lastCalibration": self.last_calibration,
                "lastError": self.last_error,
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
                self.config.controller_translation_scale = float(payload["controllerTranslationScale"])
            if "controllerMaxOffsetM" in payload and is_number(payload["controllerMaxOffsetM"]):
                self.config.controller_max_offset_m = float(payload["controllerMaxOffsetM"])
            if "controllerMaxStepM" in payload and is_number(payload["controllerMaxStepM"]):
                self.config.controller_max_step_m = float(payload["controllerMaxStepM"])
            if "controllerMaxRotationDeg" in payload and is_number(payload["controllerMaxRotationDeg"]):
                self.config.controller_max_rotation_deg = float(payload["controllerMaxRotationDeg"])
            if "controllerMaxRotationStepDeg" in payload and is_number(payload["controllerMaxRotationStepDeg"]):
                self.config.controller_max_rotation_step_deg = float(payload["controllerMaxRotationStepDeg"])
            if "controllerJointLimitBufferRad" in payload and is_number(payload["controllerJointLimitBufferRad"]):
                self.config.controller_joint_limit_buffer_rad = max(0.0, float(payload["controllerJointLimitBufferRad"]))
            if "controllerJointLimitGuardEnabled" in payload:
                self.config.controller_joint_limit_guard_enabled = bool(payload["controllerJointLimitGuardEnabled"])
            if "gripperEnabled" in payload:
                self.config.gripper_enabled = bool(payload["gripperEnabled"])
            if "gripperDevice" in payload:
                self.config.gripper_device = str(payload.get("gripperDevice") or DEFAULT_GRIPPER_DEVICE).strip() or DEFAULT_GRIPPER_DEVICE
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
            self.last_error = None
            return {"ok": True, "state": state, "status": self.status()}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def disconnect_robot(self) -> dict[str, Any]:
        self.robot.disconnect()
        return self.status()

    def arm_motion(self) -> dict[str, Any]:
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
        try:
            self.robot.disable_freedrive()
            self.config.controller_motion_enabled = False
            self.last_error = None
            return {"ok": True, "status": self.status()}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def list_cameras(self) -> dict[str, Any]:
        try:
            return {"ok": True, "cameras": list_realsense_cameras()}
        except Exception as exc:  # pragma: no cover - hardware path
            return {"ok": False, "error": str(exc), "cameras": []}

    def start_realsense_stream(self) -> dict[str, Any]:
        try:
            stream = self.stream_hub.start(self.config)
            self.last_error = None
            return {"ok": True, "stream": stream, "status": self.status()}
        except Exception as exc:  # pragma: no cover - hardware path
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error, "status": self.status()}

    def stop_realsense_stream(self) -> dict[str, Any]:
        self.stream_hub.stop()
        return {"ok": True, "status": self.status()}

    def capture_end_camera_preview_jpeg(self) -> tuple[bytes, dict[str, Any]]:
        return self.capture_camera_preview_jpeg("end")

    def capture_camera_preview_jpeg(self, role: str = "end") -> tuple[bytes, dict[str, Any]]:
        role = str(role or "end").strip().lower()
        if role not in ("end", "third"):
            raise RuntimeError(f"Unsupported RealSense camera role: {role}")
        with self.lock:
            configured_serial = self.config.third_camera_serial if role == "third" else self.config.camera_serial
            serial = str(configured_serial or "").strip()
        if not serial:
            raise RuntimeError(f"RealSense {role} camera serial is empty")
        frame = self.stream_hub.get_latest(role, wait_timeout=5.0)
        return frame["jpeg"], {
            "createdAtUtc": frame.get("capturedAtUtc") or datetime.now(timezone.utc).isoformat(),
            "camera": frame.get("metadata"),
            "sequence": frame.get("sequence"),
            "role": role,
            "stream": self.stream_hub.status(),
        }

    def capture_end_camera_overlay_jpeg(self) -> tuple[bytes, dict[str, Any]]:
        with self.lock:
            serial = str(self.config.camera_serial or "").strip()
        if not serial:
            raise RuntimeError("RealSense end camera serial is empty")
        frame = self.stream_hub.get_latest("end", wait_timeout=5.0)
        rgb = frame["rgb"]
        overlay_rgb, board = checkerboard_overlay_rgb(
            rgb,
            self.config.pattern_cols,
            self.config.pattern_rows,
        )
        meta = {
            "createdAtUtc": frame.get("capturedAtUtc") or datetime.now(timezone.utc).isoformat(),
            "camera": frame.get("metadata"),
            "sequence": frame.get("sequence"),
            "stream": self.stream_hub.status(),
            "overlay": True,
            "board": board,
        }
        return encode_rgb_jpeg(overlay_rgb, quality=86), meta

    def check_end_camera_board(self, output_root: Path) -> dict[str, Any]:
        output_dir = output_root.resolve() / "end_camera"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_hub.start(self.config)
        frame = self.stream_hub.get_latest("end", wait_timeout=2.0)
        metadata = frame.get("metadata") or {}
        rgb = frame["rgb"]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        brightness = {
            "mean": float(np.mean(gray)),
            "median": float(np.median(gray)),
            "p95": float(np.percentile(gray, 95)),
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        serial_name = safe_filename(self.config.camera_serial or "camera")
        image_path = output_dir / f"end_camera_board_check_{serial_name}_{stamp}.jpg"
        save_rgb_jpeg(rgb, image_path)
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
            cols=self.config.pattern_cols,
            rows=self.config.pattern_rows,
            square_size_m=self.config.square_size_m,
            overlay_dir=overlay_dir,
        )
        result = {
            "ok": observation is not None,
            "createdAtUtc": datetime.now(timezone.utc).isoformat(),
            "camera": metadata,
            "pattern": {
                "cols": self.config.pattern_cols,
                "rows": self.config.pattern_rows,
                "squareSizeM": self.config.square_size_m,
            },
            "brightness": brightness,
            "imagePath": str(image_path),
            "streamFrame": {
                "capturedAtUtc": frame.get("capturedAtUtc"),
                "sequence": frame.get("sequence"),
                "serial": frame.get("serial"),
            },
            "detectedCorners": self.config.pattern_cols * self.config.pattern_rows if observation is not None else 0,
            "method": observation.get("method") if observation is not None else None,
            "bestReprojectionRmsePx": None,
            "bestReprojectionMedianPx": None,
            "appearanceAnchor": observation.get("appearance_anchor") if observation is not None else None,
            "redAnchor": observation.get("red_anchor") if observation is not None else None,
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
    ) -> RobotRealsenseSession | None:
        if control_mode not in ROBOT_SESSION_CONTROL_MODES:
            control_mode = ROBOT_SESSION_CONTROL_TELEOP
        with self.lock:
            if self.active_session is not None:
                previous = self.active_session
                previous.close()
                if getattr(previous, "control_mode", ROBOT_SESSION_CONTROL_TELEOP) == ROBOT_SESSION_CONTROL_FREEDRIVE:
                    self.robot.disable_freedrive()
                else:
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
            session = RobotRealsenseSession(
                parent_directory,
                record_id,
                self.config,
                self.robot,
                self.stream_hub,
                publish_event,
                robot_alignment_result=robot_alignment_result,
                require_controller_alignment=require_controller_alignment,
                control_mode=control_mode,
            )
            try:
                session.start()
            except Exception as exc:  # pragma: no cover - hardware path
                session.close()
                self.last_error = str(exc)
                if publish_event is not None:
                    publish_event({"type": "robot_status", "ok": False, "stage": "start_failed", "error": self.last_error})
                return None
            try:
                if control_mode == ROBOT_SESSION_CONTROL_FREEDRIVE:
                    self.robot.enable_freedrive()
                    self.config.controller_motion_enabled = False
                    stage = "freedrive_enabled"
                elif control_mode == ROBOT_SESSION_CONTROL_TELEOP:
                    self.robot.arm_motion()
                    self.config.controller_motion_enabled = True
                    stage = "motion_armed"
                else:
                    self.robot.disarm_motion()
                    self.robot.disable_freedrive()
                    self.config.controller_motion_enabled = False
                    stage = "record_only"
                self.last_error = None
                if publish_event is not None:
                    publish_event({"type": "robot_status", "stage": stage, "controlMode": control_mode, **self.status()})
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
            self.active_session = session
            return session

    def stop_session(self, session: RobotRealsenseSession | None) -> dict[str, Any] | None:
        if session is None:
            return None
        summary = session.close()
        if getattr(session, "control_mode", ROBOT_SESSION_CONTROL_TELEOP) == ROBOT_SESSION_CONTROL_FREEDRIVE:
            self.robot.disable_freedrive()
        else:
            self.robot.disarm_motion()
        self.config.controller_motion_enabled = False
        with self.lock:
            if self.active_session is session:
                self.active_session = None
        return summary

    def calibrate_session(
        self,
        session_dir: Path,
        quest_calibration_event: dict[str, Any] | None,
        publish_event: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        try:
            result = calibrate_robot_realsense_run(
                session_dir,
                self.config.pattern_cols,
                self.config.pattern_rows,
                self.config.square_size_m,
                self.config.min_hand_eye_detections,
                self.config,
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
            t_base_ee = transform_from_json(sample.get("T_base_tool_tcp"))
            if t_base_ee is None:
                t_base_ee = transform_from_json(sample.get("T_base_ee"))
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
    red_anchor_global = infer_observation_red_anchor_target(observations, cols, rows)
    apply_global_red_anchor_target(observations, red_anchor_global, cols, rows)

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
            "red_anchor_ok": int(sum(bool((obs.get("red_anchor") or {}).get("ok")) for obs in observations)),
            "red_anchor_rot180": int(sum((obs.get("red_anchor") or {}).get("order") == "rot180" for obs in observations)),
            "red_anchor_global": red_anchor_global,
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
        transform = transform_from_json(sample.get("T_base_tool_tcp"))
        if transform is None:
            transform = transform_from_json(sample.get("T_base_ee"))
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
    red_anchor = detect_red_corner_anchor(image, corners.reshape(-1, 2), cols, rows)
    anchor_order = red_anchor.get("order") if red_anchor.get("ok") else None
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
                "matches_red_anchor": bool(order == anchor_order) if anchor_order in ("identity", "rot180") else None,
            }
        )
    if not candidates:
        return None
    overlay_path = overlay_dir / f"sample_{int(sample.get('sample_index', 0)):06d}_end_overlay.jpg"
    overlay = image.copy()
    cv2.drawChessboardCorners(overlay, pattern, corners, True)
    draw_red_anchor_overlay(overlay, corners.reshape(-1, 2), red_anchor, cols, rows)
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
        "red_anchor": red_anchor,
        "candidates": candidates,
        "selected": 0,
    }


def checkerboard_overlay_rgb(rgb: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    pattern = (int(cols), int(rows))
    ok = False
    corners = None
    method = "findChessboardCorners"
    try:
        scale = min(1.0, 640.0 / max(1, max(rgb.shape[0], rgb.shape[1])))
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
        red_anchor = detect_red_corner_anchor(bgr, corners.reshape(-1, 2), int(cols), int(rows))
        cv2.drawChessboardCorners(overlay, pattern, corners, True)
        draw_red_anchor_overlay(overlay, corners.reshape(-1, 2), red_anchor, int(cols), int(rows))
        draw_checkerboard_appearance_anchor_overlay(overlay, corners.reshape(-1, 2), appearance_anchor, int(cols), int(rows))
        status = f"checkerboard detected: {len(corners)}/{int(cols) * int(rows)}"
        color = (80, 220, 120)
    else:
        appearance_anchor = None
        red_anchor = None
        status = "checkerboard not detected"
        color = (80, 120, 255)
    cv2.rectangle(overlay, (10, 10), (min(520, overlay.shape[1] - 10), 46), (20, 24, 28), -1)
    cv2.putText(overlay, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), {
        "ok": bool(ok and corners is not None),
        "detectedCorners": int(len(corners)) if ok and corners is not None else 0,
        "expectedCorners": int(cols) * int(rows),
        "method": method,
        "appearanceAnchor": appearance_anchor,
        "redAnchor": red_anchor,
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
    red_anchor = obs.get("red_anchor") if isinstance(obs.get("red_anchor"), dict) else {}
    anchor_order = red_anchor.get("order") if red_anchor.get("ok") else None
    if anchor_order in ("identity", "rot180"):
        for index, candidate in enumerate(obs.get("candidates") or []):
            if candidate.get("order") == anchor_order:
                return int(index)
    return 0


def infer_observation_red_anchor_target(observations: list[dict[str, Any]], cols: int, rows: int) -> dict[str, Any]:
    total = int(cols) * int(rows)
    order_180 = np.arange(total).reshape(int(rows), int(cols))[::-1, ::-1].reshape(-1)
    counts: dict[int, int] = {}
    observed_counts: dict[int, int] = {}
    score_sum: dict[int, float] = {}
    for obs in observations:
        anchor = obs.get("red_anchor") if isinstance(obs.get("red_anchor"), dict) else {}
        if not anchor.get("ok"):
            continue
        observed = int(anchor.get("observedIndex", -1))
        if observed < 0 or observed >= total:
            continue
        target = int(min(observed, int(order_180[observed])))
        counts[target] = counts.get(target, 0) + 1
        observed_counts[observed] = observed_counts.get(observed, 0) + 1
        score_sum[target] = score_sum.get(target, 0.0) + float(anchor.get("bestScore") or 0.0)
    if not counts:
        return {
            "target_index": None,
            "source": "none_detected",
            "ok_frames": 0,
            "counts": {},
            "observed_counts": {},
            "note": "Red anchor is optional; observations without red use identity/rot180 hand-eye residual matching.",
        }
    target = max(counts, key=lambda idx: (counts[idx], score_sum.get(idx, 0.0)))
    return {
        "target_index": int(target),
        "source": "auto_from_any_reliable_red_anchor",
        "ok_frames": int(sum(counts.values())),
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "observed_counts": {str(key): int(value) for key, value in sorted(observed_counts.items())},
        "score_sum": {str(key): float(value) for key, value in sorted(score_sum.items())},
        "note": "A reliable red anchor frame orients the global 180-degree pair; observations without red remain valid and fall back to residual matching.",
    }


def apply_global_red_anchor_target(observations: list[dict[str, Any]], summary: dict[str, Any], cols: int, rows: int) -> None:
    target = summary.get("target_index") if isinstance(summary, dict) else None
    if target is None:
        return
    total = int(cols) * int(rows)
    order_180 = np.arange(total).reshape(int(rows), int(cols))[::-1, ::-1].reshape(-1)
    for obs in observations:
        anchor = obs.get("red_anchor") if isinstance(obs.get("red_anchor"), dict) else {}
        if not anchor.get("ok"):
            continue
        observed = int(anchor.get("observedIndex", -1))
        if observed < 0 or observed >= total:
            anchor["targetIndex"] = int(target)
            anchor["order"] = None
            anchor["orderReason"] = "observed_index_out_of_range"
            continue
        if observed == int(target):
            order = "identity"
        elif int(order_180[observed]) == int(target):
            order = "rot180"
        else:
            anchor["targetIndex"] = int(target)
            anchor["order"] = None
            anchor["orderReason"] = "observed_corner_not_in_global_target_180_pair"
            for candidate in obs.get("candidates") or []:
                candidate["matches_red_anchor"] = None
            continue
        anchor["targetIndex"] = int(target)
        anchor["order"] = order
        anchor["orderReason"] = "global_red_anchor_target"
        for candidate in obs.get("candidates") or []:
            candidate["matches_red_anchor"] = bool(candidate.get("order") == order)


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
    max_translation = 0.0
    max_rotation = 0.0
    pair_count = 0
    for i, a in enumerate(poses):
        for b in poses[i + 1 :]:
            pair_count += 1
            delta = invert_transform(a) @ b
            max_translation = max(max_translation, float(np.linalg.norm(delta[:3, 3])))
            max_rotation = max(max_rotation, rotation_angle_deg(delta[:3, :3]))
    axis_span = np.ptp(translations, axis=0) if translations.size else np.zeros(3, dtype=float)
    return {
        "samples": len(poses),
        "pairCount": pair_count,
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
    red_anchor = obs.get("red_anchor") if isinstance(obs.get("red_anchor"), dict) else {}
    anchor_order = red_anchor.get("order") if red_anchor.get("ok") else None
    if anchor_order in ("identity", "rot180"):
        for index, candidate in enumerate(obs.get("candidates") or []):
            if candidate.get("order") == anchor_order:
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
        "trigger": row.get("trigger"),
        "targetWidthM": row.get("target_width_m"),
        "status": row.get("status"),
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


def checkerboard_corner_indices(cols: int, rows: int) -> list[int]:
    return [0, cols - 1, (rows - 1) * cols, rows * cols - 1]


def red_anchor_order_for_observed(observed_index: int, target_index: int, order_180: np.ndarray) -> str | None:
    if observed_index < 0 or target_index < 0:
        return None
    if int(observed_index) == int(target_index):
        return "identity"
    if int(order_180[int(observed_index)]) == int(target_index):
        return "rot180"
    return None


def red_anchor_target_index_for_observed(observed_index: int, cols: int, rows: int) -> int | None:
    total = int(cols) * int(rows)
    if observed_index < 0 or observed_index >= total:
        return None
    order_180 = np.arange(total).reshape(int(rows), int(cols))[::-1, ::-1].reshape(-1)
    paired = int(order_180[int(observed_index)])
    return int(min(int(observed_index), paired))


def corner_grid_spacing_px(corners: np.ndarray, cols: int, rows: int) -> float:
    points = np.asarray(corners, dtype=float).reshape(rows, cols, 2)
    diffs: list[np.ndarray] = []
    if cols >= 2:
        diffs.append(np.linalg.norm(np.diff(points, axis=1), axis=2).reshape(-1))
    if rows >= 2:
        diffs.append(np.linalg.norm(np.diff(points, axis=0), axis=2).reshape(-1))
    if not diffs:
        return 12.0
    values = np.concatenate(diffs)
    values = values[np.isfinite(values) & (values > 1.0)]
    return float(np.median(values)) if values.size else 12.0


def red_mask_bgr(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_mask = (((h <= 10) | (h >= 170)) & (s >= 70) & (v >= 45))
    b, g, r = cv2.split(image)
    rgb_mask = (r >= 80) & (r.astype(np.float32) >= 1.25 * g.astype(np.float32)) & (r.astype(np.float32) >= 1.25 * b.astype(np.float32))
    return (hsv_mask | rgb_mask).astype(np.uint8)


def detect_red_corner_anchor(image: np.ndarray, corners: np.ndarray, cols: int, rows: int) -> dict[str, Any]:
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    corner_indices = checkerboard_corner_indices(cols, rows)
    spacing = corner_grid_spacing_px(points, cols, rows)
    radius = max(8.0, RED_ANCHOR_RADIUS_GRID_SPACING * spacing)
    radius = min(radius, 0.35 * float(min(image.shape[:2])))
    mask = red_mask_bgr(image)
    height, width = mask.shape[:2]
    scores: dict[str, float] = {}
    red_pixels: dict[str, int] = {}
    areas: dict[str, int] = {}
    centers: dict[str, list[float]] = {}
    for index in corner_indices:
        x, y = points[index]
        centers[str(index)] = [float(x), float(y)]
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(width, int(np.ceil(x + radius + 1)))
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(height, int(np.ceil(y + radius + 1)))
        if x0 >= x1 or y0 >= y1:
            scores[str(index)] = 0.0
            red_pixels[str(index)] = 0
            areas[str(index)] = 0
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - x) * (xx - x) + (yy - y) * (yy - y) <= radius * radius
        roi = mask[y0:y1, x0:x1]
        area = int(np.count_nonzero(circle))
        count = int(np.count_nonzero(roi[circle]))
        scores[str(index)] = float(count / max(1, area))
        red_pixels[str(index)] = count
        areas[str(index)] = area
    ranked = sorted(corner_indices, key=lambda idx: (scores[str(idx)], red_pixels[str(idx)]), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else best
    best_score = float(scores[str(best)])
    second_score = float(scores[str(second)]) if second != best else 0.0
    score_ratio = float(best_score / max(second_score, 1e-9))
    ok = (
        red_pixels[str(best)] >= RED_ANCHOR_MIN_PIXELS
        and best_score >= RED_ANCHOR_MIN_RATIO
        and score_ratio >= RED_ANCHOR_MIN_BEST_SECOND_RATIO
    )
    target = red_anchor_target_index_for_observed(int(best), cols, rows) if ok else None
    order_180 = np.arange(int(cols) * int(rows)).reshape(int(rows), int(cols))[::-1, ::-1].reshape(-1)
    order = red_anchor_order_for_observed(int(best), int(target), order_180) if target is not None else None
    return {
        "ok": bool(ok),
        "observedIndex": int(best) if ok else -1,
        "targetIndex": int(target) if target is not None else -1,
        "order": order,
        "bestIndex": int(best),
        "secondIndex": int(second),
        "bestScore": best_score,
        "secondScore": second_score,
        "scoreRatio": score_ratio,
        "redPixels": red_pixels,
        "areas": areas,
        "scoresByCorner": scores,
        "centersByCorner": centers,
        "radiusPx": float(radius),
        "gridSpacingPx": float(spacing),
        "cornerIndices": [int(v) for v in corner_indices],
        "reason": "ok" if ok else "weak_or_ambiguous_red_corner",
    }


def draw_red_anchor_overlay(out: np.ndarray, corners: np.ndarray, anchor: dict[str, Any] | None, cols: int, rows: int) -> None:
    if not anchor:
        return
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    radius = int(round(float(anchor.get("radiusPx") or 0.0)))
    scores = anchor.get("scoresByCorner") if isinstance(anchor.get("scoresByCorner"), dict) else {}
    observed = int(anchor.get("observedIndex", -1))
    best = int(anchor.get("bestIndex", -1))
    for index in checkerboard_corner_indices(cols, rows):
        x, y = points[index]
        color = (0, 0, 255) if index == observed else ((0, 180, 255) if index == best else (200, 200, 200))
        if radius > 0:
            cv2.circle(out, (int(round(x)), int(round(y))), radius, color, 2, cv2.LINE_AA)
        label = f"{index}:{float(scores.get(str(index), 0.0)):.3f}"
        cv2.putText(out, label, (int(round(x)) + 6, int(round(y)) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


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
                "red_anchor": obs.get("red_anchor"),
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


def save_rgb_jpeg(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_rgb_jpeg(rgb, quality=92))


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("could not encode image")
    return bytes(buffer)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "controllerMaxOffsetM": config.controller_max_offset_m,
        "controllerMaxStepM": config.controller_max_step_m,
        "controllerMaxRotationDeg": config.controller_max_rotation_deg,
        "controllerMaxRotationStepDeg": config.controller_max_rotation_step_deg,
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
