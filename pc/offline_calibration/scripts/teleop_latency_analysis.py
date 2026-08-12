from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LATENCY_ANALYSIS_JSON = "teleop_latency_analysis.json"
LATENCY_ANALYSIS_SCHEMA = "teleop_latency_analysis_v1"
DEFAULT_GRID_DT_SECONDS = 0.01
DEFAULT_MIN_LAG_SECONDS = -0.20
DEFAULT_MAX_LAG_SECONDS = 1.50
DEFAULT_TARGET_SPEED_THRESHOLD_MPS = 0.003
DEFAULT_ROBOT_SPEED_THRESHOLD_MPS = 0.002
DEFAULT_CONTROLLER_SPEED_THRESHOLD_MPS = 0.02
DEFAULT_EVENT_MIN_GAP_SECONDS = 0.20
DEFAULT_EVENT_SEARCH_SECONDS = 1.50
DEFAULT_MAX_CHART_POINTS = 1600


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Quest controller / Collector command to robot TCP motion latency "
            "for a Quest3DataCollector pc_recordings session."
        )
    )
    parser.add_argument("--pc-session", type=Path, required=True, help="PC recording session directory.")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional output JSON path. Defaults to teleop_latency_analysis.json in the session folder.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even when an up-to-date JSON exists.")
    parser.add_argument("--grid-dt-seconds", type=float, default=DEFAULT_GRID_DT_SECONDS)
    parser.add_argument("--min-lag-seconds", type=float, default=DEFAULT_MIN_LAG_SECONDS)
    parser.add_argument("--max-lag-seconds", type=float, default=DEFAULT_MAX_LAG_SECONDS)
    parser.add_argument("--target-speed-threshold-mps", type=float, default=DEFAULT_TARGET_SPEED_THRESHOLD_MPS)
    parser.add_argument("--robot-speed-threshold-mps", type=float, default=DEFAULT_ROBOT_SPEED_THRESHOLD_MPS)
    parser.add_argument("--controller-speed-threshold-mps", type=float, default=DEFAULT_CONTROLLER_SPEED_THRESHOLD_MPS)
    parser.add_argument("--event-min-gap-seconds", type=float, default=DEFAULT_EVENT_MIN_GAP_SECONDS)
    parser.add_argument("--event-search-seconds", type=float, default=DEFAULT_EVENT_SEARCH_SECONDS)
    parser.add_argument("--max-chart-points", type=int, default=DEFAULT_MAX_CHART_POINTS)
    args = parser.parse_args()

    payload = load_or_build_latency_analysis(
        args.pc_session,
        output_json=args.output_json,
        force=args.force,
        grid_dt_seconds=args.grid_dt_seconds,
        min_lag_seconds=args.min_lag_seconds,
        max_lag_seconds=args.max_lag_seconds,
        target_speed_threshold_mps=args.target_speed_threshold_mps,
        robot_speed_threshold_mps=args.robot_speed_threshold_mps,
        controller_speed_threshold_mps=args.controller_speed_threshold_mps,
        event_min_gap_seconds=args.event_min_gap_seconds,
        event_search_seconds=args.event_search_seconds,
        max_chart_points=args.max_chart_points,
    )
    print_latency_summary(payload)
    return 0 if payload.get("ok") else 1


def latency_analysis_path(session_dir: Path) -> Path:
    return Path(session_dir) / DEFAULT_LATENCY_ANALYSIS_JSON


def load_or_build_latency_analysis(
    session_dir: Path,
    *,
    output_json: Path | None = None,
    force: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    output_path = Path(output_json).resolve() if output_json is not None else latency_analysis_path(session_dir)
    source_mtime = latency_source_mtime(session_dir)
    if not force and output_path.exists() and output_path.stat().st_mtime >= source_mtime:
        cached = read_json_if_exists(output_path)
        if isinstance(cached, dict) and cached.get("schema") == LATENCY_ANALYSIS_SCHEMA:
            cached["cacheHit"] = True
            cached["analysisJson"] = str(output_path)
            return cached
    payload = analyze_session(session_dir, **kwargs)
    payload["cacheHit"] = False
    payload["analysisJson"] = str(output_path)
    atomic_write_json(output_path, payload)
    return payload


def write_latency_analysis(session_dir: Path, **kwargs: Any) -> Path:
    payload = load_or_build_latency_analysis(session_dir, force=True, **kwargs)
    return Path(payload["analysisJson"])


def analyze_session(
    session_dir: Path,
    *,
    grid_dt_seconds: float = DEFAULT_GRID_DT_SECONDS,
    min_lag_seconds: float = DEFAULT_MIN_LAG_SECONDS,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    target_speed_threshold_mps: float = DEFAULT_TARGET_SPEED_THRESHOLD_MPS,
    robot_speed_threshold_mps: float = DEFAULT_ROBOT_SPEED_THRESHOLD_MPS,
    controller_speed_threshold_mps: float = DEFAULT_CONTROLLER_SPEED_THRESHOLD_MPS,
    event_min_gap_seconds: float = DEFAULT_EVENT_MIN_GAP_SECONDS,
    event_search_seconds: float = DEFAULT_EVENT_SEARCH_SECONDS,
    max_chart_points: int = DEFAULT_MAX_CHART_POINTS,
) -> dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    robot_dir = session_dir / "robot_realsense"
    motion_rows = read_jsonl_relaxed(robot_dir / "controller_motion.jsonl")
    robot_rows = read_jsonl_relaxed(robot_dir / "robot_states.jsonl")
    if not robot_rows:
        robot_rows = read_jsonl_relaxed(robot_dir / "samples.jsonl")
    pc_rows = read_jsonl_relaxed(session_dir / "pc_samples.jsonl")

    target_positions = controller_target_positions(motion_rows)
    robot_positions = robot_tcp_positions(robot_rows)
    controller_positions = quest_controller_positions(pc_rows)

    target_speed = speed_series_from_positions(target_positions, label="command target")
    robot_speed = speed_series_from_positions(robot_positions, label="robot TCP")
    controller_speed = speed_series_from_positions(controller_positions, label="Quest controller")

    kwargs_common = {
        "grid_dt_seconds": grid_dt_seconds,
        "min_lag_seconds": min_lag_seconds,
        "max_lag_seconds": max_lag_seconds,
        "event_min_gap_seconds": event_min_gap_seconds,
        "event_search_seconds": event_search_seconds,
    }
    target_to_robot = analyze_pair(
        target_speed,
        robot_speed,
        input_label="command target",
        response_label="robot TCP",
        input_threshold_mps=target_speed_threshold_mps,
        response_threshold_mps=robot_speed_threshold_mps,
        **kwargs_common,
    )
    controller_to_robot = analyze_pair(
        controller_speed,
        robot_speed,
        input_label="Quest controller",
        response_label="robot TCP",
        input_threshold_mps=controller_speed_threshold_mps,
        response_threshold_mps=robot_speed_threshold_mps,
        **kwargs_common,
    )
    primary_key = "targetToRobot" if target_to_robot.get("ok") else "controllerToRobot"
    primary = target_to_robot if target_to_robot.get("ok") else controller_to_robot

    chart = multi_speed_chart(
        target_speed,
        robot_speed,
        controller_speed,
        max_points=max_chart_points,
    )
    warnings: list[str] = []
    if not motion_rows:
        warnings.append("missing robot_realsense/controller_motion.jsonl")
    if not robot_rows:
        warnings.append("missing robot_realsense/robot_states.jsonl and samples.jsonl")
    if not target_to_robot.get("ok") and not controller_to_robot.get("ok"):
        warnings.append("not enough motion overlap to estimate latency")

    return {
        "ok": bool(primary.get("ok")),
        "schema": LATENCY_ANALYSIS_SCHEMA,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "sessionDir": str(session_dir),
        "robotDir": str(robot_dir),
        "method": {
            "primary": primary_key,
            "definition": (
                "Positive lag means robot TCP motion follows the command/controller motion. "
                "Correlation uses speed magnitudes on the Collector PC perf-counter timeline; "
                "event latency uses threshold-crossing motion onsets."
            ),
            "gridDtSeconds": float(grid_dt_seconds),
            "lagSearchSeconds": [float(min_lag_seconds), float(max_lag_seconds)],
            "targetSpeedThresholdMps": float(target_speed_threshold_mps),
            "robotSpeedThresholdMps": float(robot_speed_threshold_mps),
            "controllerSpeedThresholdMps": float(controller_speed_threshold_mps),
            "eventMinGapSeconds": float(event_min_gap_seconds),
            "eventSearchSeconds": float(event_search_seconds),
        },
        "counts": {
            "motionRows": len(motion_rows),
            "robotRows": len(robot_rows),
            "pcSamples": len(pc_rows),
            "targetPositions": len(target_positions["t"]),
            "robotPositions": len(robot_positions["t"]),
            "controllerPositions": len(controller_positions["t"]),
        },
        "primary": summarize_primary(primary_key, primary),
        "targetToRobot": target_to_robot,
        "controllerToRobot": controller_to_robot,
        "chart": chart,
        "warnings": warnings,
    }


def summarize_primary(key: str, payload: dict[str, Any]) -> dict[str, Any]:
    correlation = payload.get("correlation") if isinstance(payload.get("correlation"), dict) else {}
    events = payload.get("events") if isinstance(payload.get("events"), dict) else {}
    latency = events.get("latencySeconds") if isinstance(events.get("latencySeconds"), dict) else {}
    return {
        "source": key,
        "ok": bool(payload.get("ok")),
        "correlationLagSeconds": correlation.get("bestLagSeconds"),
        "correlation": correlation.get("bestCorrelation"),
        "eventCount": events.get("count"),
        "eventLatencySeconds": latency,
        "warnings": payload.get("warnings") if isinstance(payload.get("warnings"), list) else [],
    }


def analyze_pair(
    input_speed: dict[str, Any],
    response_speed: dict[str, Any],
    *,
    input_label: str,
    response_label: str,
    input_threshold_mps: float,
    response_threshold_mps: float,
    grid_dt_seconds: float,
    min_lag_seconds: float,
    max_lag_seconds: float,
    event_min_gap_seconds: float,
    event_search_seconds: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    correlation = best_lag_correlation(
        input_speed,
        response_speed,
        min_lag_seconds=min_lag_seconds,
        max_lag_seconds=max_lag_seconds,
        grid_dt_seconds=grid_dt_seconds,
    )
    events = onset_latency(
        input_speed,
        response_speed,
        input_threshold_mps=input_threshold_mps,
        response_threshold_mps=response_threshold_mps,
        min_gap_seconds=event_min_gap_seconds,
        search_seconds=event_search_seconds,
    )
    if len(input_speed["t"]) < 5:
        warnings.append(f"not enough {input_label} speed samples")
    if len(response_speed["t"]) < 5:
        warnings.append(f"not enough {response_label} speed samples")
    ok = bool(correlation.get("ok")) or bool(events.get("ok"))
    return {
        "ok": ok,
        "inputLabel": input_label,
        "responseLabel": response_label,
        "inputThresholdMps": float(input_threshold_mps),
        "responseThresholdMps": float(response_threshold_mps),
        "counts": {
            "inputSpeedSamples": len(input_speed["t"]),
            "responseSpeedSamples": len(response_speed["t"]),
        },
        "correlation": correlation,
        "events": events,
        "warnings": warnings,
    }


def controller_target_positions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times: list[float] = []
    positions: list[list[float]] = []
    accepted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("ok") is not True:
            continue
        if row.get("commandSent") is False:
            continue
        pose = row.get("target_tcp_pose_wxyz")
        if not (isinstance(pose, list) and len(pose) >= 3):
            continue
        t = numeric(row.get("pc_perf_counter_seconds"))
        if t is None:
            continue
        position = vec3_from_any(pose[:3])
        if position is None:
            continue
        times.append(t)
        positions.append(position)
        accepted += 1
    return sorted_position_series(times, positions, accepted=accepted)


def robot_tcp_positions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times: list[float] = []
    positions: list[list[float]] = []
    accepted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("ok") is False:
            continue
        t = numeric(row.get("pc_perf_counter_seconds")) or numeric(row.get("robot_state_pc_perf_counter_seconds"))
        if t is None:
            continue
        position = transform_translation(row.get("T_base_tool_tcp"))
        if position is None:
            position = transform_translation(row.get("T_base_ee"))
        if position is None:
            state = row.get("robot_state") if isinstance(row.get("robot_state"), dict) else {}
            tcp_pose = state.get("tcp_pose") if isinstance(state.get("tcp_pose"), dict) else None
            if isinstance(tcp_pose, dict):
                position = transform_translation(tcp_pose.get("T_base_pose"))
                if position is None:
                    position = vec3_from_any(tcp_pose.get("pose"))
        if position is None:
            continue
        times.append(t)
        positions.append(position)
        accepted += 1
    return sorted_position_series(times, positions, accepted=accepted)


def quest_controller_positions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times: list[float] = []
    positions: list[list[float]] = []
    accepted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        t = numeric(row.get("pcReceivePerfCounterSeconds"))
        controller = row.get("rightController")
        if not isinstance(controller, dict):
            continue
        if controller.get("hasPose") is False:
            continue
        position = vec3_from_any(controller.get("position"))
        if t is None or position is None:
            continue
        if not controller_teleop_held(controller):
            continue
        times.append(t)
        positions.append(position)
        accepted += 1
    return sorted_position_series(times, positions, accepted=accepted)


def sorted_position_series(times: list[float], positions: list[list[float]], *, accepted: int) -> dict[str, Any]:
    pairs = sorted(zip(times, positions), key=lambda item: item[0])
    clean_t: list[float] = []
    clean_p: list[list[float]] = []
    last_t = -math.inf
    for t, p in pairs:
        if not math.isfinite(t) or t <= last_t:
            continue
        clean_t.append(float(t))
        clean_p.append([float(v) for v in p[:3]])
        last_t = float(t)
    return {"t": clean_t, "p": clean_p, "accepted": accepted}


def speed_series_from_positions(positions: dict[str, Any], *, label: str) -> dict[str, Any]:
    times = [float(v) for v in positions.get("t", []) if is_number(v)]
    points = [np.asarray(p, dtype=float) for p in positions.get("p", []) if isinstance(p, list) and len(p) >= 3]
    if len(times) != len(points) or len(times) < 2:
        return {"label": label, "t": [], "speed": []}
    speed_t: list[float] = []
    speeds: list[float] = []
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if not math.isfinite(dt) or dt <= 1e-5:
            continue
        distance = float(np.linalg.norm(points[index][:3] - points[index - 1][:3]))
        speed_t.append(times[index])
        speeds.append(distance / dt)
    speeds = smooth_values(speeds, window=3)
    return {"label": label, "t": speed_t, "speed": speeds}


def best_lag_correlation(
    input_speed: dict[str, Any],
    response_speed: dict[str, Any],
    *,
    min_lag_seconds: float,
    max_lag_seconds: float,
    grid_dt_seconds: float,
) -> dict[str, Any]:
    x_t = np.asarray(input_speed.get("t") or [], dtype=float)
    y_t = np.asarray(response_speed.get("t") or [], dtype=float)
    x_v = np.asarray(input_speed.get("speed") or [], dtype=float)
    y_v = np.asarray(response_speed.get("speed") or [], dtype=float)
    if len(x_t) < 5 or len(y_t) < 5:
        return {"ok": False, "reason": "not_enough_samples"}
    dt = max(0.002, float(grid_dt_seconds))
    lag_min = float(min_lag_seconds)
    lag_max = float(max_lag_seconds)
    if lag_max < lag_min:
        lag_min, lag_max = lag_max, lag_min
    best: tuple[float, float, int] | None = None
    lag_count = int(math.floor((lag_max - lag_min) / dt)) + 1
    for lag in (lag_min + index * dt for index in range(max(1, lag_count))):
        start = max(float(x_t[0]), float(y_t[0]) - lag)
        end = min(float(x_t[-1]), float(y_t[-1]) - lag)
        if end - start < max(0.20, dt * 20):
            continue
        grid = np.arange(start, end, dt, dtype=float)
        if len(grid) < 20:
            continue
        x = np.interp(grid, x_t, x_v)
        y = np.interp(grid + lag, y_t, y_v)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(finite)) < 20:
            continue
        x = x[finite]
        y = y[finite]
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        if x_std <= 1e-9 or y_std <= 1e-9:
            continue
        corr = float(np.corrcoef((x - np.mean(x)) / x_std, (y - np.mean(y)) / y_std)[0, 1])
        if not math.isfinite(corr):
            continue
        if best is None or corr > best[1]:
            best = (float(lag), corr, int(len(x)))
    if best is None:
        return {"ok": False, "reason": "no_valid_overlap"}
    return {
        "ok": True,
        "bestLagSeconds": best[0],
        "bestCorrelation": best[1],
        "samples": best[2],
        "lagRangeSeconds": [lag_min, lag_max],
        "gridDtSeconds": dt,
    }


def onset_latency(
    input_speed: dict[str, Any],
    response_speed: dict[str, Any],
    *,
    input_threshold_mps: float,
    response_threshold_mps: float,
    min_gap_seconds: float,
    search_seconds: float,
) -> dict[str, Any]:
    x_t = np.asarray(input_speed.get("t") or [], dtype=float)
    y_t = np.asarray(response_speed.get("t") or [], dtype=float)
    x_v = np.asarray(input_speed.get("speed") or [], dtype=float)
    y_v = np.asarray(response_speed.get("speed") or [], dtype=float)
    if len(x_t) < 5 or len(y_t) < 5:
        return {"ok": False, "reason": "not_enough_samples", "count": 0}
    onsets = threshold_onsets(x_t, x_v, float(input_threshold_mps), float(min_gap_seconds))
    latencies: list[float] = []
    missed = 0
    already_moving = 0
    for onset in onsets:
        current = interp_scalar(y_t, y_v, onset)
        if current is not None and current >= response_threshold_mps:
            already_moving += 1
            continue
        start_idx = int(np.searchsorted(y_t, onset, side="left"))
        found: float | None = None
        for idx in range(start_idx, len(y_t)):
            dt = float(y_t[idx] - onset)
            if dt < 0:
                continue
            if dt > search_seconds:
                break
            if y_v[idx] >= response_threshold_mps:
                found = dt
                break
        if found is None:
            missed += 1
        else:
            latencies.append(found)
    return {
        "ok": bool(latencies),
        "count": len(latencies),
        "candidateOnsets": len(onsets),
        "missed": missed,
        "alreadyMoving": already_moving,
        "latencySeconds": numeric_stats(latencies),
        "latenciesSeconds": [round(float(v), 6) for v in latencies[:200]],
    }


def threshold_onsets(times: np.ndarray, values: np.ndarray, threshold: float, min_gap: float) -> list[float]:
    result: list[float] = []
    last = -math.inf
    was_active = bool(len(values) and values[0] >= threshold)
    for idx in range(1, len(times)):
        active = bool(values[idx] >= threshold)
        if active and not was_active and float(times[idx] - last) >= min_gap:
            result.append(float(times[idx]))
            last = float(times[idx])
        was_active = active
    return result


def multi_speed_chart(
    target_speed: dict[str, Any],
    robot_speed: dict[str, Any],
    controller_speed: dict[str, Any],
    *,
    max_points: int,
) -> list[dict[str, Any]]:
    series = [target_speed, robot_speed, controller_speed]
    valid = [s for s in series if len(s.get("t") or []) >= 2]
    if len(valid) < 2:
        return []
    start = min(float(s["t"][0]) for s in valid)
    end = max(float(s["t"][-1]) for s in valid)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return []
    max_points = max(20, int(max_points))
    dt = max(0.01, (end - start) / max_points)
    grid = np.arange(start, end + dt * 0.5, dt, dtype=float)
    rows: list[dict[str, Any]] = []
    for t in grid:
        row = {"t": round(float(t - start), 4)}
        add_chart_value(row, "targetSpeedMps", target_speed, t)
        add_chart_value(row, "robotSpeedMps", robot_speed, t)
        add_chart_value(row, "controllerSpeedMps", controller_speed, t)
        rows.append(row)
    return rows


def add_chart_value(row: dict[str, Any], key: str, series: dict[str, Any], t: float) -> None:
    value = interp_scalar(
        np.asarray(series.get("t") or [], dtype=float),
        np.asarray(series.get("speed") or [], dtype=float),
        float(t),
    )
    if value is not None and math.isfinite(value):
        row[key] = round(float(value), 6)


def transform_translation(payload: Any) -> list[float] | None:
    if not isinstance(payload, dict):
        return None
    translation = vec3_from_any(payload.get("translation_m"))
    if translation is not None:
        return translation
    matrix = payload.get("matrix_4x4")
    if isinstance(matrix, list) and len(matrix) >= 3:
        try:
            return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
        except Exception:
            return None
    return None


def vec3_from_any(value: Any) -> list[float] | None:
    if isinstance(value, list) and len(value) >= 3:
        try:
            result = [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            return None
        return result if all(math.isfinite(v) for v in result) else None
    if isinstance(value, dict):
        keys = ("x", "y", "z")
        if all(key in value for key in keys):
            try:
                result = [float(value[key]) for key in keys]
            except Exception:
                return None
            return result if all(math.isfinite(v) for v in result) else None
    return None


def controller_teleop_held(controller: dict[str, Any]) -> bool:
    if controller.get("teleopHeld") is not None:
        return bool(controller.get("teleopHeld"))
    for key in ("handTriggerPressed", "sideButtonPressed", "gripButton", "gripPressed"):
        if controller.get(key) is True:
            return True
    trigger = numeric(controller.get("handTrigger"))
    return bool(trigger is not None and trigger >= 0.65)


def smooth_values(values: list[float], *, window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return [float(v) for v in values]
    radius = window // 2
    result: list[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        result.append(float(sum(values[lo:hi]) / max(1, hi - lo)))
    return result


def interp_scalar(times: np.ndarray, values: np.ndarray, t: float) -> float | None:
    if len(times) < 2 or len(values) < 2:
        return None
    if t < float(times[0]) or t > float(times[-1]):
        return None
    return float(np.interp([t], times, values)[0])


def numeric_stats(values: list[float]) -> dict[str, Any]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return {"count": 0}
    arr = np.asarray(finite, dtype=float)
    return {
        "count": int(len(finite)),
        "min": float(arr[0]),
        "max": float(arr[-1]),
        "mean": float(np.mean(arr)),
        "median": percentile(finite, 50.0),
        "p90": percentile(finite, 90.0),
        "p95": percentile(finite, 95.0),
        "std": float(np.std(arr)),
    }


def percentile(sorted_values: list[float], percentile_value: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    frac = rank - lower
    return float(sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac)


def latency_source_mtime(session_dir: Path) -> float:
    robot_dir = Path(session_dir) / "robot_realsense"
    candidates = (
        Path(session_dir) / "pc_samples.jsonl",
        robot_dir / "controller_motion.jsonl",
        robot_dir / "robot_states.jsonl",
        robot_dir / "samples.jsonl",
        robot_dir / "session_summary.json",
    )
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    return max(mtimes or [0.0])


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl_relaxed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp_path.replace(path)


def print_latency_summary(payload: dict[str, Any]) -> None:
    primary = payload.get("primary") if isinstance(payload.get("primary"), dict) else {}
    if not payload.get("ok"):
        print("teleop latency: unavailable")
        for warning in payload.get("warnings", []) if isinstance(payload.get("warnings"), list) else []:
            print(f"  warning: {warning}")
        return
    lag = primary.get("correlationLagSeconds")
    corr = primary.get("correlation")
    lat = primary.get("eventLatencySeconds") if isinstance(primary.get("eventLatencySeconds"), dict) else {}
    parts = [f"source={primary.get('source')}"]
    if is_number(lag):
        parts.append(f"corr_lag={float(lag) * 1000.0:.1f}ms")
    if is_number(corr):
        parts.append(f"corr={float(corr):.3f}")
    if is_number(lat.get("median")):
        parts.append(f"event_median={float(lat['median']) * 1000.0:.1f}ms")
    if is_number(lat.get("p95")):
        parts.append(f"event_p95={float(lat['p95']) * 1000.0:.1f}ms")
    if is_number(lat.get("count")):
        parts.append(f"events={int(lat['count'])}")
    print("teleop latency: " + ", ".join(parts))


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_number(value: Any) -> bool:
    return numeric(value) is not None


if __name__ == "__main__":
    raise SystemExit(main())


