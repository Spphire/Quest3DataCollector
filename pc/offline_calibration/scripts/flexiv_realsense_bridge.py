from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


DEFAULT_PATTERN_COLS = 11
DEFAULT_PATTERN_ROWS = 8
DEFAULT_SQUARE_SIZE_M = 0.025
DEFAULT_REALSENSE_WIDTH = 1280
DEFAULT_REALSENSE_HEIGHT = 720
DEFAULT_REALSENSE_FPS = 30
DEFAULT_CAPTURE_INTERVAL_SECONDS = 0.35
DEFAULT_FLEXIV_RDK_ROOT: Path | None = None
DEFAULT_FLEXIV_ROBOT_SN = "Rizon4-H6uDOq"
DEFAULT_END_CAMERA_SERIAL = "244222073667"


@dataclass
class FlexivRealSenseConfig:
    robot_sn: str = DEFAULT_FLEXIV_ROBOT_SN
    robot_pose_field: str = "flange_pose"
    flexiv_rdk: Path | None = DEFAULT_FLEXIV_RDK_ROOT
    camera_serial: str = DEFAULT_END_CAMERA_SERIAL
    width: int = DEFAULT_REALSENSE_WIDTH
    height: int = DEFAULT_REALSENSE_HEIGHT
    fps: int = DEFAULT_REALSENSE_FPS
    warmup_frames: int = 2
    capture_interval_seconds: float = DEFAULT_CAPTURE_INTERVAL_SECONDS
    pattern_cols: int = DEFAULT_PATTERN_COLS
    pattern_rows: int = DEFAULT_PATTERN_ROWS
    square_size_m: float = DEFAULT_SQUARE_SIZE_M
    run_hand_eye: bool = True
    min_hand_eye_detections: int = 6


class FlexivRobotClient:
    def __init__(self) -> None:
        self.robot: Any | None = None
        self.robot_sn: str | None = None
        self.pose_field = "flange_pose"
        self.lock = threading.Lock()
        self.last_error: str | None = None

    def connect(
        self,
        robot_sn: str,
        pose_field: str = "flange_pose",
        flexiv_rdk: Path | None = None,
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
            self.robot = flexivrdk.Robot(robot_sn)
            self.robot_sn = robot_sn
            self.pose_field = pose_field
            self.last_error = None
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            return self.read_state_locked()

    def disconnect(self) -> None:
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


class RealSenseColorCamera:
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.profile: Any | None = None
        self.serial: str | None = None
        self.width = DEFAULT_REALSENSE_WIDTH
        self.height = DEFAULT_REALSENSE_HEIGHT
        self.fps = DEFAULT_REALSENSE_FPS
        self.metadata: dict[str, Any] | None = None

    def start(self, serial: str, width: int, height: int, fps: int) -> dict[str, Any]:
        if self.pipeline is not None and self.serial == serial and self.width == width and self.height == height and self.fps == fps:
            return self.metadata or {}
        self.stop()
        import pyrealsense2 as rs  # type: ignore[import-not-found]

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.rgb8, int(fps))
        profile = pipeline.start(config)
        self.pipeline = pipeline
        self.profile = profile
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.metadata = camera_metadata_from_profile(profile, serial)
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


class RobotRealsenseSession:
    def __init__(
        self,
        root: Path,
        record_id: str,
        config: FlexivRealSenseConfig,
        robot: FlexivRobotClient,
        publish_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.record_id = record_id
        self.config = config
        self.robot = robot
        self.publish_event = publish_event
        self.directory = self.root / "robot_realsense"
        self.image_dir = self.directory / "images"
        self.samples_path = self.directory / "samples.jsonl"
        self.summary_path = self.directory / "session_summary.json"
        self.camera = RealSenseColorCamera()
        self.samples_handle: Any | None = None
        self.lock = threading.Lock()
        self.next_capture_perf = 0.0
        self.sample_count = 0
        self.image_count = 0
        self.error_count = 0
        self.closed = False
        self.last_error: str | None = None
        self.camera_metadata: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.samples_handle = self.samples_path.open("a", encoding="utf-8", newline="\n")
        config_payload = config_to_json(self.config)
        config_payload["recordId"] = self.record_id
        config_payload["startedAtUtc"] = datetime.now(timezone.utc).isoformat()
        write_json(config_payload, self.directory / "capture_config.json")
        if self.config.camera_serial:
            self.camera_metadata = self.camera.start(
                self.config.camera_serial,
                self.config.width,
                self.config.height,
                self.config.fps,
            )
            write_json({"end": self.camera_metadata}, self.directory / "cameras.json")
        summary = self.summary("recording")
        self._publish({"type": "robot_status", "stage": "robot_realsense_recording", **summary})
        return summary

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
                self.image_count += 1
            self._publish(robot_sample_event(row))
            return row

    def close(self) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return self.summary("already_closed")
            self.closed = True
            self.camera.stop()
            if self.samples_handle is not None:
                self.samples_handle.close()
                self.samples_handle = None
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
            "samples": self.sample_count,
            "images": self.image_count,
            "errors": self.error_count,
            "lastError": self.last_error,
        }

    def _capture_row(self, quest_sample: dict[str, Any]) -> dict[str, Any]:
        sample_index = self.sample_count
        robot_state = self.robot.read_state()
        image_rel: str | None = None
        if self.camera.pipeline is not None:
            rgb = self.camera.capture_rgb(self.config.warmup_frames)
            image_rel_path = Path("images") / f"sample_{sample_index:06d}_end_{self.config.camera_serial}.jpg"
            save_rgb_jpeg(rgb, self.directory / image_rel_path)
            image_rel = str(image_rel_path).replace("\\", "/")
        return {
            "sample_index": sample_index,
            "record_id": self.record_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "ok": True,
            "pose_source": f"flexiv:{self.config.robot_pose_field}:{self.config.robot_sn}",
            "quest_sample_index": quest_sample.get("sampleIndex"),
            "quest_recording_timestamp_seconds": quest_sample.get("recordingTimestampSeconds"),
            "quest_pc_receive_perf_counter_seconds": quest_sample.get("pcReceivePerfCounterSeconds"),
            "robot_state": robot_state,
            "T_base_ee": robot_state.get("endEffectorPose"),
            "jointpose": robot_state.get("jointPose"),
            "images": {"end": image_rel} if image_rel else {},
        }

    def _publish(self, event: dict[str, Any]) -> None:
        if self.publish_event is not None:
            self.publish_event(event)


class FlexivRealSenseManager:
    def __init__(self, config: FlexivRealSenseConfig | None = None) -> None:
        self.config = config or FlexivRealSenseConfig()
        self.robot = FlexivRobotClient()
        self.lock = threading.Lock()
        self.active_session: RobotRealsenseSession | None = None
        self.last_calibration: dict[str, Any] | None = None
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            active = self.active_session.summary("recording") if self.active_session is not None else None
            return {
                "ok": True,
                "enabled": True,
                "config": config_to_json(self.config),
                "robot": self.robot.status(),
                "activeSession": active,
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
            for key, attr in (
                ("width", "width"),
                ("height", "height"),
                ("fps", "fps"),
                ("warmupFrames", "warmup_frames"),
                ("minHandEyeDetections", "min_hand_eye_detections"),
            ):
                if key in payload and is_number(payload[key]):
                    setattr(self.config, attr, int(payload[key]))
            if "captureIntervalSeconds" in payload and is_number(payload["captureIntervalSeconds"]):
                self.config.capture_interval_seconds = float(payload["captureIntervalSeconds"])
            if "runHandEye" in payload:
                self.config.run_hand_eye = bool(payload["runHandEye"])
        return self.status()

    def connect_robot(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.configure(payload)
        try:
            state = self.robot.connect(
                self.config.robot_sn,
                self.config.robot_pose_field,
                self.config.flexiv_rdk,
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

    def list_cameras(self) -> dict[str, Any]:
        try:
            return {"ok": True, "cameras": list_realsense_cameras()}
        except Exception as exc:  # pragma: no cover - hardware path
            return {"ok": False, "error": str(exc), "cameras": []}

    def start_session(
        self,
        parent_directory: Path,
        record_id: str,
        publish_event: Callable[[dict[str, Any]], None] | None,
    ) -> RobotRealsenseSession | None:
        with self.lock:
            if self.active_session is not None:
                self.active_session.close()
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
            session = RobotRealsenseSession(parent_directory, record_id, self.config, self.robot, publish_event)
            try:
                session.start()
            except Exception as exc:  # pragma: no cover - hardware path
                session.close()
                self.last_error = str(exc)
                if publish_event is not None:
                    publish_event({"type": "robot_status", "ok": False, "stage": "start_failed", "error": self.last_error})
                return None
            self.active_session = session
            self.last_error = None
            return session

    def stop_session(self, session: RobotRealsenseSession | None) -> dict[str, Any] | None:
        if session is None:
            return None
        summary = session.close()
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
            failure = {
                "type": "robot_calibration_failure",
                "ok": False,
                "runDir": str(session_dir),
                "error": str(exc),
                "createdAtUtc": datetime.now(timezone.utc).isoformat(),
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

    observations = []
    overlay_dir = run_dir / "detections"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        if not sample.get("ok"):
            continue
        t_base_ee = transform_from_json(sample.get("T_base_ee"))
        image_rel = (sample.get("images") or {}).get("end")
        if t_base_ee is None or not image_rel:
            continue
        image_path = (run_dir / image_rel).resolve()
        if not image_path.exists():
            continue
        obs = detect_end_observation(
            image_path=image_path,
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

    if len(observations) < int(min_detections):
        raise ValueError(f"Need at least {min_detections} valid end-camera detections, got {len(observations)}")

    solution = solve_end_hand_eye(observations)
    result = {
        "ok": True,
        "record_id": samples[0].get("record_id") if samples else None,
        "run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Flexiv end-mounted RealSense hand-eye calibration. T_A_B maps coordinates from B into A.",
        "pattern": {"cols": cols, "rows": rows, "square_size_m": square_size_m},
        "camera": camera,
        "counts": {
            "samples": len(samples),
            "detections": len(observations),
            "rot180_selected": int(sum(obs["selected"] == 1 for obs in observations)),
        },
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


def detect_end_observation(
    image_path: Path,
    sample: dict[str, Any],
    t_base_ee: np.ndarray,
    camera: dict[str, Any],
    cols: int,
    rows: int,
    square_size_m: float,
    overlay_dir: Path,
) -> dict[str, Any] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
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
            }
        )
    if not candidates:
        return None
    overlay_path = overlay_dir / f"sample_{int(sample.get('sample_index', 0)):06d}_end_overlay.jpg"
    overlay = image.copy()
    cv2.drawChessboardCorners(overlay, pattern, corners, True)
    cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return {
        "sample_index": int(sample.get("sample_index", 0)),
        "quest_sample_index": sample.get("quest_sample_index"),
        "quest_recording_timestamp_seconds": sample.get("quest_recording_timestamp_seconds"),
        "image": str(image_path),
        "overlay": str(overlay_path),
        "method": method,
        "T_base_ee": t_base_ee,
        "candidates": candidates,
        "selected": 0,
    }


def solve_end_hand_eye(observations: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.optimize import least_squares

    selected = [0 for _ in observations]
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
    t_world_board = transform_from_json(quest_calibration_event.get("T_world_board"))
    board_payload = robot_result.get("board") if isinstance(robot_result.get("board"), dict) else {}
    t_base_board = transform_from_json(board_payload.get("T_base_board"))
    if t_world_board is None or t_base_board is None:
        return {"ok": False, "reason": "missing_transform"}
    t_world_base = t_world_board @ invert_transform(t_base_board)
    return {
        "ok": True,
        "T_world_base": transform_to_json(t_world_base),
        "T_base_world": transform_to_json(invert_transform(t_world_base)),
    }


def compact_robot_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "recordId": result.get("record_id"),
        "counts": result.get("counts"),
        "endCamera": result.get("end_camera"),
        "board": result.get("board"),
        "questAlignment": result.get("questAlignment"),
    }


def robot_sample_event(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
    return {
        "type": "robot_sample",
        "ok": bool(row.get("ok")),
        "recordId": row.get("record_id"),
        "sampleIndex": row.get("sample_index"),
        "questSampleIndex": row.get("quest_sample_index"),
        "capturedAt": row.get("captured_at"),
        "T_base_ee": row.get("T_base_ee"),
        "jointpose": row.get("jointpose"),
        "poseField": state.get("poseField"),
        "robotSn": state.get("robotSn"),
        "error": row.get("error"),
    }


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


def transform_to_json(matrix: np.ndarray) -> dict[str, Any]:
    return {
        "matrix_4x4": [[float(value) for value in row] for row in matrix[:4, :4]],
        "translation_m": [float(value) for value in matrix[:3, 3]],
        "quaternion_wxyz": matrix_to_quaternion_wxyz(matrix),
    }


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


def object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid.astype(np.float32) * float(square_size_m)
    return points


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


def save_rgb_jpeg(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"could not write image: {path}")


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
        "cameraSerial": config.camera_serial,
        "width": config.width,
        "height": config.height,
        "fps": config.fps,
        "warmupFrames": config.warmup_frames,
        "captureIntervalSeconds": config.capture_interval_seconds,
        "pattern": [config.pattern_cols, config.pattern_rows],
        "squareSizeM": config.square_size_m,
        "runHandEye": config.run_hand_eye,
        "minHandEyeDetections": config.min_hand_eye_detections,
    }


def normalize_quaternion(value: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in value[:4]))
    if not math.isfinite(norm) or norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / norm for v in value[:4]]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
