from __future__ import annotations

import argparse
import json
import math
import random
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_CONTROLLER_POSITION = [0.20, 1.10, 0.30]
REDUNDANT_DATAGRAM_INTERVAL_SECONDS = 0.001


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Send a bounded synthetic Quest controller recording while an external "
            "watchdog checks the real Flexiv TCP against its frozen start pose."
        )
    )
    parser.add_argument("--host", default="10.128.1.95")
    parser.add_argument("--udp-port", type=int, default=9100)
    parser.add_argument("--view-port", type=int, default=8765)
    parser.add_argument("--duration-seconds", type=float, default=60.0, help="Random-walk duration, excluding settle/return.")
    parser.add_argument("--sample-hz", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--return-seconds", type=float, default=5.0)
    parser.add_argument("--walk-translation-radius-m", type=float, default=0.040)
    parser.add_argument("--walk-rotation-radius-deg", type=float, default=8.0)
    parser.add_argument("--translation-speed-mps", type=float, default=0.010)
    parser.add_argument("--rotation-speed-degps", type=float, default=3.0)
    parser.add_argument("--direction-change-seconds", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--hard-translation-limit-m", type=float, default=0.050)
    parser.add_argument("--hard-rotation-limit-deg", type=float, default=10.0)
    parser.add_argument("--watchdog-translation-limit-m", type=float, default=0.049)
    parser.add_argument("--watchdog-rotation-limit-deg", type=float, default=9.8)
    parser.add_argument("--watchdog-period-seconds", type=float, default=0.20)
    parser.add_argument("--record-id")
    parser.add_argument("--exercise-gripper", action="store_true")
    parser.add_argument(
        "--no-delayed-redundancy",
        dest="use_delayed_redundancy",
        action="store_false",
        help="Disable the Quest sender's two-packet delayed redundancy pattern.",
    )
    parser.set_defaults(use_delayed_redundancy=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_limits(args)
    record_id = args.record_id or f"record_bounded_teleop_{time.strftime('%Y%m%d_%H%M%S')}"
    report_path = args.report or Path.cwd() / f"{record_id}_probe_report.json"
    if args.dry_run:
        report = dry_run_report(args, record_id)
        write_report(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    initial_status = get_json(args.host, args.view_port, "/robot/safety-status", timeout=8)
    validate_initial_status(initial_status)
    baseline_tcp = extract_tcp_pose(initial_status)
    state = ProbeState(baseline_tcp)
    stop_event = threading.Event()
    watchdog = threading.Thread(
        target=watch_robot,
        args=(args, state, stop_event),
        name="bounded-teleop-watchdog",
        daemon=True,
    )
    watchdog.start()

    started_at = datetime.now(timezone.utc).isoformat()
    error: str | None = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = DelayedRedundantUdpSender(
        sock,
        args.host,
        args.udp_port,
        enabled=args.use_delayed_redundancy,
    )
    total_duration = max(0.0, args.settle_seconds) + max(1.0, args.duration_seconds) + max(0.0, args.return_seconds)
    sample_count = max(1, int(round(total_duration * max(1.0, args.sample_hz))))
    walker = BoundedRandomWalk(
        translation_radius_m=args.walk_translation_radius_m,
        rotation_radius_deg=args.walk_rotation_radius_deg,
        translation_speed_mps=args.translation_speed_mps,
        rotation_speed_degps=args.rotation_speed_degps,
        sample_hz=args.sample_hz,
        direction_change_seconds=args.direction_change_seconds,
        seed=args.seed,
    )
    sequence = 0
    start_wall = time.time()
    try:
        sequence += 1
        sender.send({"type": "recording_start", "recordId": record_id, "sequence": sequence})
        next_perf = time.perf_counter()
        for index in range(sample_count):
            if stop_event.is_set():
                raise RuntimeError(state.violation or "watchdog stopped the probe")
            t = index / args.sample_hz
            if t < args.settle_seconds:
                signal = zero_signal(teleop_held=t >= args.settle_seconds * 0.5, phase=t / total_duration)
            elif t < args.settle_seconds + args.duration_seconds:
                signal = walker.step(1.0 / args.sample_hz)
                signal["teleopHeld"] = True
                signal["phase"] = t / total_duration
            else:
                # Repeatedly request the frozen center. The receiver's existing
                # per-command step limits then walk the robot target back safely.
                signal = zero_signal(teleop_held=True, phase=t / total_duration)
            enforce_generated_bounds(signal, args.hard_translation_limit_m, args.hard_rotation_limit_deg)
            state.note_generated(signal)
            sequence += 1
            sender.send(
                sample_payload(
                    record_id,
                    sequence,
                    index,
                    t,
                    start_wall,
                    signal,
                    exercise_gripper=args.exercise_gripper,
                ),
            )
            next_perf += 1.0 / args.sample_hz
            now_perf = time.perf_counter()
            if next_perf <= now_perf:
                next_perf = now_perf + 1.0 / args.sample_hz
            sleep_seconds = next_perf - now_perf
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        send_release_samples(sender, args, record_id, sequence, sample_count, start_wall)
    except Exception as exc:
        error = str(exc)
        stop_event.set()
        try:
            send_release_samples(sender, args, record_id, sequence, sample_count, start_wall)
        except Exception:
            pass
    finally:
        try:
            sequence += 10
            sender.send(
                {
                    "type": "recording_stop",
                    "recordId": record_id,
                    "sequence": sequence,
                    "telemetryMode": "live_preview",
                    "recordingTimestampSeconds": total_duration,
                },
                repeat_current=True,
            )
        finally:
            sock.close()
        stop_event.set()
        watchdog.join(timeout=max(2.0, args.watchdog_period_seconds * 3.0))
        disarm_results = disarm_repeatedly(args.host, args.view_port)

    final_status: dict[str, Any] | None = None
    try:
        final_status = get_json(args.host, args.view_port, "/robot/safety-status", timeout=8)
        state.note_status(final_status)
    except Exception as exc:
        state.status_errors.append(f"final status: {exc}")
    report = {
        "ok": error is None and state.violation is None,
        "recordId": record_id,
        "startedAtUtc": started_at,
        "finishedAtUtc": datetime.now(timezone.utc).isoformat(),
        "baselineTcpPoseWxyz": baseline_tcp,
        "limits": {
            "hardTranslationM": args.hard_translation_limit_m,
            "hardRotationDeg": args.hard_rotation_limit_deg,
            "watchdogTranslationM": args.watchdog_translation_limit_m,
            "watchdogRotationDeg": args.watchdog_rotation_limit_deg,
            "walkTranslationRadiusM": args.walk_translation_radius_m,
            "walkRotationRadiusDeg": args.walk_rotation_radius_deg,
            "translationSpeedMps": args.translation_speed_mps,
            "rotationSpeedDegps": args.rotation_speed_degps,
            "seed": args.seed,
        },
        "generated": state.generated_summary(),
        "actualTcp": state.actual_summary(),
        "watchdogViolation": state.violation,
        "statusErrors": state.status_errors,
        "error": error,
        "disarmResults": disarm_results,
        "finalMotionArmed": nested(final_status, "robot", "motionArmed") if final_status else None,
        "finalTcpPoseWxyz": extract_tcp_pose(final_status) if final_status else None,
    }
    write_report(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


class ProbeState:
    def __init__(self, baseline_tcp: list[float]) -> None:
        self.baseline_tcp = baseline_tcp
        self.lock = threading.Lock()
        self.violation: str | None = None
        self.status_errors: list[str] = []
        self.status_samples = 0
        self.max_actual_translation_m = 0.0
        self.max_actual_rotation_deg = 0.0
        self.max_generated_translation_m = 0.0
        self.max_generated_rotation_deg = 0.0

    def note_generated(self, signal: dict[str, Any]) -> None:
        with self.lock:
            self.max_generated_translation_m = max(
                self.max_generated_translation_m,
                vector_norm(signal["translationUnityM"]),
            )
            self.max_generated_rotation_deg = max(
                self.max_generated_rotation_deg,
                vector_norm(signal["rotationVectorUnityDeg"]),
            )

    def note_status(self, status: dict[str, Any]) -> tuple[float, float]:
        tcp = extract_tcp_pose(status)
        translation = vector_norm([tcp[i] - self.baseline_tcp[i] for i in range(3)])
        rotation = quaternion_angle_deg(tcp[3:7], self.baseline_tcp[3:7])
        with self.lock:
            self.status_samples += 1
            self.max_actual_translation_m = max(self.max_actual_translation_m, translation)
            self.max_actual_rotation_deg = max(self.max_actual_rotation_deg, rotation)
        return translation, rotation

    def fail(self, message: str) -> None:
        with self.lock:
            if self.violation is None:
                self.violation = message

    def generated_summary(self) -> dict[str, Any]:
        return {
            "maxTranslationM": self.max_generated_translation_m,
            "maxRotationDeg": self.max_generated_rotation_deg,
        }

    def actual_summary(self) -> dict[str, Any]:
        return {
            "statusSamples": self.status_samples,
            "maxTranslationFromBaselineM": self.max_actual_translation_m,
            "maxRotationFromBaselineDeg": self.max_actual_rotation_deg,
        }


def validate_limits(args: argparse.Namespace) -> None:
    values = (
        args.walk_translation_radius_m,
        args.walk_rotation_radius_deg,
        args.translation_speed_mps,
        args.rotation_speed_degps,
        args.hard_translation_limit_m,
        args.hard_rotation_limit_deg,
        args.watchdog_translation_limit_m,
        args.watchdog_rotation_limit_deg,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
        raise ValueError("all amplitudes and limits must be finite and positive")
    if args.walk_translation_radius_m >= args.watchdog_translation_limit_m:
        raise ValueError("translation walk radius must be below the watchdog limit")
    if args.walk_rotation_radius_deg >= args.watchdog_rotation_limit_deg:
        raise ValueError("rotation walk radius must be below the watchdog limit")
    if args.watchdog_translation_limit_m >= args.hard_translation_limit_m:
        raise ValueError("translation watchdog limit must be below the hard limit")
    if args.watchdog_rotation_limit_deg >= args.hard_rotation_limit_deg:
        raise ValueError("rotation watchdog limit must be below the hard limit")


def validate_initial_status(status: dict[str, Any]) -> None:
    if status.get("formalControlMode") != "controller_teleop":
        raise RuntimeError(f"receiver is not in controller_teleop: {status.get('formalControlMode')}")
    if status.get("motionCommandsAllowed") is not True:
        raise RuntimeError("receiver does not allow controller motion commands")
    robot = status.get("robot") if isinstance(status.get("robot"), dict) else {}
    state = robot.get("state") if isinstance(robot.get("state"), dict) else {}
    if robot.get("connected") is not True:
        raise RuntimeError("Flexiv robot is not connected")
    if robot.get("lastError"):
        raise RuntimeError(f"robot reports error: {robot.get('lastError')}")
    if state.get("operationalStatus") not in (None, "READY"):
        raise RuntimeError(f"robot is not READY: {state.get('operationalStatus')}")
    guard = state.get("jointLimitGuard") if isinstance(state.get("jointLimitGuard"), dict) else {}
    if guard.get("ok") is False:
        raise RuntimeError(f"joint limit guard is not clear: {guard.get('reason')}")
    if status.get("activeSession"):
        raise RuntimeError("receiver already has an active recording session")


class BoundedRandomWalk:
    def __init__(
        self,
        *,
        translation_radius_m: float,
        rotation_radius_deg: float,
        translation_speed_mps: float,
        rotation_speed_degps: float,
        sample_hz: float,
        direction_change_seconds: float,
        seed: int,
    ) -> None:
        self.translation_radius_m = float(translation_radius_m)
        self.rotation_radius_deg = float(rotation_radius_deg)
        self.translation_speed_mps = float(translation_speed_mps)
        self.rotation_speed_degps = float(rotation_speed_degps)
        self.direction_change_steps = max(1, int(round(max(0.05, direction_change_seconds) * sample_hz)))
        self.rng = random.Random(seed)
        self.steps = 0
        self.translation = [0.0, 0.0, 0.0]
        self.rotation_vector_deg = [0.0, 0.0, 0.0]
        self.translation_velocity = scale_vector(random_unit_vector(self.rng), self.translation_speed_mps)
        self.rotation_velocity = scale_vector(random_unit_vector(self.rng), self.rotation_speed_degps)

    def step(self, dt: float) -> dict[str, Any]:
        if self.steps and self.steps % self.direction_change_steps == 0:
            self.translation_velocity = blended_direction(
                self.rng,
                self.translation_velocity,
                self.translation_speed_mps,
            )
            self.rotation_velocity = blended_direction(
                self.rng,
                self.rotation_velocity,
                self.rotation_speed_degps,
            )
        self.translation, self.translation_velocity = reflected_step(
            self.translation,
            self.translation_velocity,
            dt,
            self.translation_radius_m,
        )
        self.rotation_vector_deg, self.rotation_velocity = reflected_step(
            self.rotation_vector_deg,
            self.rotation_velocity,
            dt,
            self.rotation_radius_deg,
        )
        self.steps += 1
        return self.signal()

    def signal(self) -> dict[str, Any]:
        return {
            "translationUnityM": list(self.translation),
            "rotationVectorUnityDeg": list(self.rotation_vector_deg),
            "teleopHeld": True,
            "phase": 0.0,
        }


def reflected_step(
    position: list[float],
    velocity: list[float],
    dt: float,
    radius: float,
) -> tuple[list[float], list[float]]:
    candidate = add_vectors(position, scale_vector(velocity, dt))
    magnitude = vector_norm(candidate)
    if magnitude <= radius:
        return candidate, velocity
    normal = scale_vector(candidate, 1.0 / max(1e-12, magnitude))
    reflected = add_vectors(velocity, scale_vector(normal, -2.0 * dot_product(velocity, normal)))
    candidate = add_vectors(position, scale_vector(reflected, dt))
    magnitude = vector_norm(candidate)
    if magnitude > radius:
        candidate = scale_vector(candidate, radius * 0.999 / magnitude)
    return candidate, reflected


def blended_direction(rng: random.Random, current: list[float], speed: float) -> list[float]:
    mixed = add_vectors(scale_vector(current, 0.65), scale_vector(random_unit_vector(rng), speed * 0.35))
    magnitude = vector_norm(mixed)
    return scale_vector(mixed, speed / max(1e-12, magnitude))


def random_unit_vector(rng: random.Random) -> list[float]:
    while True:
        vector = [rng.gauss(0.0, 1.0) for _ in range(3)]
        magnitude = vector_norm(vector)
        if magnitude > 1e-9:
            return scale_vector(vector, 1.0 / magnitude)


def zero_signal(*, teleop_held: bool, phase: float) -> dict[str, Any]:
    return {
        "translationUnityM": [0.0, 0.0, 0.0],
        "rotationVectorUnityDeg": [0.0, 0.0, 0.0],
        "teleopHeld": teleop_held,
        "phase": phase,
    }


def enforce_generated_bounds(signal: dict[str, Any], max_translation_m: float, max_rotation_deg: float) -> None:
    translation = vector_norm(signal["translationUnityM"])
    rotation = vector_norm(signal["rotationVectorUnityDeg"])
    if translation > max_translation_m + 1e-12:
        raise RuntimeError(f"generated target exceeds translation limit: {translation:.6f}m")
    if rotation > max_rotation_deg + 1e-9:
        raise RuntimeError(f"generated target exceeds rotation limit: {rotation:.6f}deg")


def sample_payload(
    record_id: str,
    sequence: int,
    index: int,
    t: float,
    start_wall: float,
    signal: dict[str, Any],
    *,
    exercise_gripper: bool,
) -> dict[str, Any]:
    offset = signal["translationUnityM"]
    position = [DEFAULT_CONTROLLER_POSITION[i] + offset[i] for i in range(3)]
    rotation = axis_angle_vector_deg_to_quaternion(signal["rotationVectorUnityDeg"])
    phase = float(signal["phase"])
    index_trigger = 1.0 if exercise_gripper and 0.45 <= phase < 0.52 else 0.0
    hand_trigger = 1.0 if signal["teleopHeld"] else 0.0
    return {
        "type": "sample",
        "recordId": record_id,
        "sequence": sequence,
        "sampleIndex": index,
        "isRecording": True,
        "telemetryMode": "recording",
        "recordingTimestampSeconds": t,
        "unityTimestampSeconds": start_wall + t,
        "hasGaze": True,
        "gazeSource": "bounded_teleop_probe",
        "gazePoint3DWorld": [0.0, 1.0, 0.5],
        "gazePoint3DSource": "bounded_teleop_probe",
        "gazeRayOrigin": [0.0, 1.2, 0.0],
        "gazeRayDirection": [0.0, -0.15, 1.0],
        "headPose": pose_payload([0.0, 1.5, 0.0], [1.0, 0.0, 0.0, 0.0]),
        "leftEyePose": pose_payload([-0.03, 1.5, 0.02], [1.0, 0.0, 0.0, 0.0]),
        "rightEyePose": pose_payload([0.03, 1.5, 0.02], [1.0, 0.0, 0.0, 0.0]),
        "leftController": controller_payload([-0.2, 1.1, 0.3], [1.0, 0.0, 0.0, 0.0], 0.0, 0.0),
        "rightController": controller_payload(position, rotation, hand_trigger, index_trigger),
    }


def pose_payload(position: list[float], rotation: list[float]) -> dict[str, Any]:
    return {"hasPose": True, "position": position, "rotation": rotation, "source": "bounded_teleop_probe"}


def controller_payload(
    position: list[float],
    rotation: list[float],
    hand_trigger: float,
    index_trigger: float,
) -> dict[str, Any]:
    payload = pose_payload(position, rotation)
    payload.update(
        {
            "positionTracked": True,
            "rotationTracked": True,
            "handTrigger": hand_trigger,
            "handTriggerPressed": hand_trigger >= 0.65,
            "indexTrigger": index_trigger,
            "indexTriggerPressed": index_trigger >= 0.65,
        }
    )
    return payload


def send_release_samples(
    sender: DelayedRedundantUdpSender,
    args: argparse.Namespace,
    record_id: str,
    sequence: int,
    sample_count: int,
    start_wall: float,
) -> None:
    signal = zero_signal(teleop_held=False, phase=1.0)
    total_duration = args.settle_seconds + args.duration_seconds + args.return_seconds
    for offset in range(3):
        payload = sample_payload(
            record_id,
            sequence + offset + 1,
            sample_count + offset,
            total_duration + offset / args.sample_hz,
            start_wall,
            signal,
            exercise_gripper=False,
        )
        sender.send(payload)
        time.sleep(1.0 / args.sample_hz)


def watch_robot(args: argparse.Namespace, state: ProbeState, stop_event: threading.Event) -> None:
    consecutive_errors = 0
    while not stop_event.wait(max(0.05, args.watchdog_period_seconds)):
        try:
            status = get_json(args.host, args.view_port, "/robot/safety-status", timeout=4)
            translation, rotation = state.note_status(status)
            consecutive_errors = 0
            robot = status.get("robot") if isinstance(status.get("robot"), dict) else {}
            robot_state = robot.get("state") if isinstance(robot.get("state"), dict) else {}
            guard = robot_state.get("jointLimitGuard") if isinstance(robot_state.get("jointLimitGuard"), dict) else {}
            if robot.get("lastError"):
                state.fail(f"robot error: {robot.get('lastError')}")
            elif robot_state.get("operationalStatus") not in (None, "READY"):
                state.fail(f"robot operational status: {robot_state.get('operationalStatus')}")
            elif guard.get("ok") is False:
                state.fail(f"joint limit guard: {guard.get('reason')}")
            elif translation >= args.watchdog_translation_limit_m:
                state.fail(f"actual TCP translation watchdog: {translation:.6f}m")
            elif rotation >= args.watchdog_rotation_limit_deg:
                state.fail(f"actual TCP rotation watchdog: {rotation:.6f}deg")
            if state.violation:
                stop_event.set()
                return
        except Exception as exc:
            consecutive_errors += 1
            state.status_errors.append(str(exc))
            if consecutive_errors >= 3:
                state.fail(f"watchdog status failed {consecutive_errors} times")
                stop_event.set()
                return


def disarm_repeatedly(host: str, port: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for delay in (0.0, 0.5, 2.0):
        if delay:
            time.sleep(delay)
        try:
            response = post_json(host, port, "/robot/disarm-motion", {}, timeout=8)
            motion_armed = nested(response, "status", "robot", "motionArmed")
            if motion_armed is None:
                motion_armed = nested(response, "robot", "motionArmed")
            results.append({"ok": response.get("ok"), "motionArmed": motion_armed})
        except Exception as exc:
            results.append({"ok": False, "error": str(exc)})
    return results


def extract_tcp_pose(status: dict[str, Any] | None) -> list[float]:
    pose = nested(status, "robot", "state", "tcp_pose", "pose")
    if not isinstance(pose, list) or len(pose) < 7:
        raise RuntimeError("robot status does not contain a valid TCP pose")
    values = [float(value) for value in pose[:7]]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("robot TCP pose contains non-finite values")
    return values


def quaternion_angle_deg(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(value * value for value in a))
    nb = math.sqrt(sum(value * value for value in b))
    if na <= 1e-12 or nb <= 1e-12:
        raise ValueError("zero-length quaternion")
    dot = abs(sum(left * right for left, right in zip(a, b)) / (na * nb))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def axis_angle_vector_deg_to_quaternion(vector_deg: list[float]) -> list[float]:
    angle_deg = vector_norm(vector_deg)
    if angle_deg <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    axis = scale_vector(vector_deg, 1.0 / angle_deg)
    half = math.radians(angle_deg) * 0.5
    sine = math.sin(half)
    return [math.cos(half), axis[0] * sine, axis[1] * sine, axis[2] * sine]


def add_vectors(a: list[float], b: list[float]) -> list[float]:
    return [float(left) + float(right) for left, right in zip(a, b)]


def scale_vector(values: list[float], scale: float) -> list[float]:
    return [float(value) * float(scale) for value in values]


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(float(left) * float(right) for left, right in zip(a, b))


def nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def get_json(host: str, port: int, path: str, *, timeout: float) -> dict[str, Any]:
    with urlopen(f"http://{host}:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(host: str, port: int, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        f"http://{host}:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class DelayedRedundantUdpSender:
    def __init__(
        self,
        sock: socket.socket,
        host: str,
        port: int,
        *,
        enabled: bool = True,
        datagram_interval_seconds: float = REDUNDANT_DATAGRAM_INTERVAL_SECONDS,
    ) -> None:
        self.sock = sock
        self.remote = (host, int(port))
        self.enabled = bool(enabled)
        self.datagram_interval_seconds = max(0.0, float(datagram_interval_seconds))
        self.pending: bytes | None = None
        self.older: bytes | None = None

    def send(self, payload: dict[str, Any], *, repeat_current: bool = False) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(data) > 1472:
            raise RuntimeError(f"synthetic UDP payload exceeds MTU budget: {len(data)} bytes")
        datagrams: list[bytes] = []
        if self.enabled:
            if self.older is not None:
                datagrams.append(self.older)
            if self.pending is not None:
                datagrams.append(self.pending)
        datagrams.append(data)
        if repeat_current:
            datagrams.append(data)
        for index, datagram in enumerate(datagrams):
            self.sock.sendto(datagram, self.remote)
            if index + 1 < len(datagrams) and self.datagram_interval_seconds > 0.0:
                time.sleep(self.datagram_interval_seconds)
        if self.enabled:
            self.older = self.pending
            self.pending = data


def dry_run_report(args: argparse.Namespace, record_id: str) -> dict[str, Any]:
    total_duration = args.settle_seconds + args.duration_seconds + args.return_seconds
    samples = max(1, int(round(total_duration * args.sample_hz)))
    max_translation = 0.0
    max_rotation = 0.0
    max_payload_bytes = 0
    start_wall = time.time()
    walker = BoundedRandomWalk(
        translation_radius_m=args.walk_translation_radius_m,
        rotation_radius_deg=args.walk_rotation_radius_deg,
        translation_speed_mps=args.translation_speed_mps,
        rotation_speed_degps=args.rotation_speed_degps,
        sample_hz=args.sample_hz,
        direction_change_seconds=args.direction_change_seconds,
        seed=args.seed,
    )
    for index in range(samples):
        t = index / args.sample_hz
        if t < args.settle_seconds:
            signal = zero_signal(teleop_held=t >= args.settle_seconds * 0.5, phase=t / total_duration)
        elif t < args.settle_seconds + args.duration_seconds:
            signal = walker.step(1.0 / args.sample_hz)
            signal["teleopHeld"] = True
            signal["phase"] = t / total_duration
        else:
            signal = zero_signal(teleop_held=True, phase=t / total_duration)
        enforce_generated_bounds(signal, args.hard_translation_limit_m, args.hard_rotation_limit_deg)
        max_translation = max(max_translation, vector_norm(signal["translationUnityM"]))
        max_rotation = max(max_rotation, vector_norm(signal["rotationVectorUnityDeg"]))
        payload = sample_payload(record_id, index + 2, index, t, start_wall, signal, exercise_gripper=args.exercise_gripper)
        max_payload_bytes = max(max_payload_bytes, len(json.dumps(payload, separators=(",", ":")).encode("utf-8")))
    return {
        "ok": True,
        "dryRun": True,
        "recordId": record_id,
        "samples": samples,
        "maxGeneratedTranslationM": max_translation,
        "maxGeneratedRotationDeg": max_rotation,
        "maxUdpPayloadBytes": max_payload_bytes,
        "delayedRedundancy": bool(args.use_delayed_redundancy),
        "hardTranslationLimitM": args.hard_translation_limit_m,
        "hardRotationLimitDeg": args.hard_rotation_limit_deg,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
