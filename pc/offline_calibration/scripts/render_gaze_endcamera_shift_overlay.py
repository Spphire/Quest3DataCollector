from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_RECORD_ROOT = Path(__file__).resolve().parents[1] / "pc_recordings"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def vec3(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        result = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(result)):
        return None
    return result


def normalize(value: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return value / norm


def matrix_from_payload(value: Any) -> np.ndarray | None:
    if not isinstance(value, dict):
        return None
    matrix = value.get("matrix_4x4")
    if not isinstance(matrix, list) or len(matrix) != 4:
        return None
    try:
        result = np.array(matrix, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        return None
    return result


def stats(values: list[float]) -> dict[str, Any]:
    finite = np.array([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def causal_median(values: list[float | None], window: int) -> list[float | None]:
    result: list[float | None] = []
    recent: list[float] = []
    for value in values:
        if value is not None and math.isfinite(float(value)):
            recent.append(float(value))
            if len(recent) > window:
                recent = recent[-window:]
            result.append(float(np.median(np.array(recent, dtype=float))))
        else:
            result.append(None)
    return result


def sample_time(row: dict[str, Any]) -> float | None:
    for key in ("recordingTimestampSeconds", "quest_recording_timestamp_seconds"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def build_filtered_gaze(
    pc_rows: list[dict[str, Any]],
    *,
    median_window: int,
    offset_m: float,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    raw_depths: list[float | None] = []
    prepared: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for row in pc_rows:
        sample_index_value = row.get("sampleIndex")
        if not isinstance(sample_index_value, int):
            raw_depths.append(None)
            prepared.append({"ok": False})
            continue

        pc_world = row.get("pcWorld") if isinstance(row.get("pcWorld"), dict) else {}
        gaze = vec3(pc_world.get("gazePoint3DWorld") or row.get("gazePoint3DWorld"))
        origin = vec3(pc_world.get("gazeRayOrigin") or row.get("gazeRayOrigin"))
        direction = vec3(pc_world.get("gazeRayDirection") or row.get("gazeRayDirection"))
        direction = normalize(direction) if direction is not None else None
        source = str(row.get("gazePoint3DSource") or row.get("gazeSource") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

        if gaze is None or origin is None or direction is None:
            raw_depths.append(None)
            prepared.append({"ok": False, "sampleIndex": sample_index_value, "source": source})
            continue
        raw_depth = float(np.dot(gaze - origin, direction))
        raw_depths.append(raw_depth)
        prepared.append(
            {
                "ok": True,
                "sampleIndex": sample_index_value,
                "t": sample_time(row),
                "source": source,
                "origin": origin,
                "direction": direction,
                "rawDepthM": raw_depth,
            }
        )

    filtered_depths = causal_median(raw_depths, median_window)
    by_index: dict[int, dict[str, Any]] = {}
    sorted_entries: list[dict[str, Any]] = []
    for item, filtered_depth in zip(prepared, filtered_depths):
        if not item.get("ok") or filtered_depth is None:
            continue
        origin = item["origin"]
        direction = item["direction"]
        filtered_point = origin + direction * float(filtered_depth)
        shifted_point = origin + direction * (float(filtered_depth) + float(offset_m))
        entry = {
            "sampleIndex": int(item["sampleIndex"]),
            "t": item.get("t"),
            "source": item.get("source"),
            "rawDepthM": float(item["rawDepthM"]),
            "filteredDepthM": float(filtered_depth),
            "origin": origin,
            "direction": direction,
            "filteredPoint": filtered_point,
            "shiftedPoint": shifted_point,
        }
        by_index[int(item["sampleIndex"])] = entry
        if entry.get("t") is not None:
            sorted_entries.append(entry)
    sorted_entries.sort(key=lambda entry: float(entry["t"]))
    return by_index, sorted_entries, source_counts


def nearest_by_time(entries: list[dict[str, Any]], target: float | None) -> dict[str, Any] | None:
    if target is None or not entries:
        return None
    times = [float(entry["t"]) for entry in entries]
    index = bisect_left(times, float(target))
    candidates = []
    if index > 0:
        candidates.append(entries[index - 1])
    if index < len(entries):
        candidates.append(entries[index])
    if not candidates:
        return None
    return min(candidates, key=lambda entry: abs(float(entry["t"]) - float(target)))


def project(point_world: np.ndarray, t_world_camera: np.ndarray, camera: dict[str, Any]) -> tuple[float, float, float] | None:
    camera_from_world = np.linalg.inv(t_world_camera)
    point_camera = camera_from_world @ np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=float)
    x, y, z = float(point_camera[0]), float(point_camera[1]), float(point_camera[2])
    if not all(math.isfinite(value) for value in (x, y, z)) or z <= 1e-9:
        return None
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    return (fx * x / z + cx, fy * y / z + cy, z)


def camera_point_from_pixel(u: float, v: float, z: float, camera: dict[str, Any]) -> np.ndarray:
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    return np.array([(float(u) - cx) * float(z) / fx, (float(v) - cy) * float(z) / fy, float(z)], dtype=float)


def transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    result = matrix @ np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=float)
    return result[:3]


def unproject_world(u: float, v: float, z: float, t_world_camera: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    return transform_point(t_world_camera, camera_point_from_pixel(u, v, z, camera))


def depth_window_median_m(depth: np.ndarray, u: float, v: float, scale_m: float | None, radius_px: int) -> float | None:
    if scale_m is None or not math.isfinite(float(scale_m)) or float(scale_m) <= 0:
        return None
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or y < 0 or x >= int(depth.shape[1]) or y >= int(depth.shape[0]):
        return None
    radius = max(0, int(radius_px))
    x0 = max(0, x - radius)
    x1 = min(int(depth.shape[1]), x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(int(depth.shape[0]), y + radius + 1)
    roi = np.asarray(depth[y0:y1, x0:x1], dtype=np.uint16)
    valid = roi[roi > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid.astype(np.float64)) * float(scale_m))


def depth_payload_for_frame(robot_dir: Path, frame_row: dict[str, Any] | None) -> tuple[np.ndarray | None, float | None, str | None]:
    if not isinstance(frame_row, dict):
        return None, None, None
    depth_info = frame_row.get("depth")
    if not isinstance(depth_info, dict):
        return None, None, None
    rel_path = depth_info.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        return None, None, None
    path = (robot_dir / rel_path).resolve()
    if not path.exists():
        return None, None, str(path)
    encoding = str(depth_info.get("encoding") or "").strip().lower()
    if encoding == "uint16_raw_aligned_to_color" or path.suffix.lower() == ".bin":
        width = int(depth_info.get("width") or 0)
        height = int(depth_info.get("height") or 0)
        if width <= 0 or height <= 0:
            return None, None, str(path)
        byte_offset = int(depth_info.get("byteOffset") or 0)
        byte_length = int(depth_info.get("byteLength") or (width * height * 2))
        expected_length = width * height * 2
        if byte_length < expected_length:
            return None, None, str(path)
        with path.open("rb") as handle:
            handle.seek(byte_offset)
            payload = handle.read(byte_length)
        if len(payload) < expected_length:
            return None, None, str(path)
        depth = np.frombuffer(payload[:expected_length], dtype="<u2").reshape(height, width).copy()
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None, None, str(path)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
    scale = depth_info.get("depthScaleM")
    try:
        scale_m = float(scale)
    except (TypeError, ValueError):
        scale_m = None
    if scale_m is not None and not math.isfinite(scale_m):
        scale_m = None
    return depth, scale_m, str(path)


def draw_crosshair(frame: np.ndarray, u: float, v: float, label: str | None = None) -> None:
    center = (int(round(u)), int(round(v)))
    cv2.circle(frame, center, 8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, center, 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(frame, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
    if label:
        x = min(max(center[0] + 12, 0), max(0, frame.shape[1] - 220))
        y = min(max(center[1] - 12, 24), max(24, frame.shape[0] - 10))
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def robot_rows_by_end_frame(robot_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in robot_rows:
        frames = row.get("videoFrames")
        end_frame = frames.get("end") if isinstance(frames, dict) else None
        frame_index = end_frame.get("frameIndex") if isinstance(end_frame, dict) else None
        if isinstance(frame_index, int):
            result[int(frame_index)] = row
    return result


def render_record(
    record_dir: Path,
    *,
    median_window: int,
    offset_m: float,
    output_name: str | None,
    depth_radius_px: int,
) -> dict[str, Any]:
    robot_dir = record_dir / "robot_realsense"
    if not robot_dir.is_dir():
        raise RuntimeError(f"{record_dir}: missing robot_realsense directory; this diagnostic needs formal robot/end-camera recording")
    cameras_path = robot_dir / "cameras.json"
    if not cameras_path.exists():
        raise RuntimeError(f"{record_dir}: missing {cameras_path}; end-camera intrinsics are required")
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    end_camera = cameras.get("end")
    if not isinstance(end_camera, dict):
        raise RuntimeError(f"{record_dir}: missing end camera intrinsics")

    pc_samples_path = record_dir / "pc_samples.jsonl"
    robot_samples_path = robot_dir / "samples.jsonl"
    video_frames_path = robot_dir / "video_frames.jsonl"
    for path, label in (
        (pc_samples_path, "PC samples"),
        (robot_samples_path, "robot quest-aligned samples"),
        (video_frames_path, "robot video frame index"),
    ):
        if not path.exists():
            raise RuntimeError(f"{record_dir}: missing {label}: {path}")
    pc_rows = read_jsonl(pc_samples_path)
    robot_rows = read_jsonl(robot_samples_path)
    video_rows = [row for row in read_jsonl(video_frames_path) if row.get("role") == "end"]
    if not video_rows:
        raise RuntimeError(f"{record_dir}: no end-camera rows in {video_frames_path}")
    robot_by_sample = {
        int(row["sample_index"]): row
        for row in robot_rows
        if isinstance(row.get("sample_index"), int)
    }
    video_by_frame = {
        int(row["frame_index"]): row
        for row in video_rows
        if isinstance(row.get("frame_index"), int)
    }
    robot_by_end_frame = robot_rows_by_end_frame(robot_rows)
    gaze_by_index, gaze_by_time, source_counts = build_filtered_gaze(
        pc_rows,
        median_window=median_window,
        offset_m=offset_m,
    )

    video_rel = video_rows[0].get("video") if video_rows else None
    if not isinstance(video_rel, str):
        video_rel = f"videos/end_{end_camera.get('serial', 'camera')}.mp4"
    input_video = robot_dir / video_rel
    if not input_video.exists():
        raise RuntimeError(f"{record_dir}: missing input video {input_video}")

    stem = output_name
    if not stem:
        stem = f"end_gaze3d_median_plus{int(round(offset_m * 1000.0))}mm_overlay.mp4"
    output_video = robot_dir / "videos" / stem
    stats_path = output_video.with_name(output_video.stem + "_stats.json")

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {input_video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or int(end_camera.get("width", 1280)))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or int(end_camera.get("height", 720)))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not open writer {output_video}")

    counts = {
        "framesWritten": 0,
        "framesWithRobotSample": 0,
        "framesWithFilteredGaze": 0,
        "framesProjectedInFront": 0,
        "framesProjectedInside": 0,
        "framesProjectedOutside": 0,
        "framesBehindCamera": 0,
        "framesMissingTransform": 0,
        "framesMissingFilteredGaze": 0,
        "framesWithDepthAtProjection": 0,
        "framesMissingDepthAtProjection": 0,
    }
    projection_u: list[float] = []
    projection_v: list[float] = []
    projection_z: list[float] = []
    pixel_shift: list[float] = []
    time_delta: list[float] = []
    end_depth_minus_gaze_z: list[float] = []
    end_depth_minus_unshifted_gaze_z: list[float] = []
    end_depth_surface_world_distance: list[float] = []
    end_depth_surface_along_gaze_ray_error: list[float] = []
    end_depth_surface_perp_gaze_ray_error: list[float] = []

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_row = video_by_frame.get(frame_index)
        robot_row = None
        if frame_index in robot_by_end_frame:
            robot_row = robot_by_end_frame[frame_index]
        elif isinstance(frame_row, dict) and isinstance(frame_row.get("sample_index"), int):
            robot_row = robot_by_sample.get(int(frame_row["sample_index"]))
        if robot_row is None and frame_index in robot_by_sample:
            robot_row = robot_by_sample[frame_index]
        if robot_row is not None:
            counts["framesWithRobotSample"] += 1

        t_world_camera = matrix_from_payload(robot_row.get("T_world_end_camera") if isinstance(robot_row, dict) else None)
        if t_world_camera is None:
            counts["framesMissingTransform"] += 1
            writer.write(frame)
            counts["framesWritten"] += 1
            frame_index += 1
            continue

        gaze_entry = None
        quest_index = robot_row.get("quest_sample_index") if isinstance(robot_row, dict) else None
        if isinstance(quest_index, int):
            gaze_entry = gaze_by_index.get(quest_index)
        if gaze_entry is None:
            gaze_entry = nearest_by_time(gaze_by_time, sample_time(robot_row))
        if gaze_entry is None:
            counts["framesMissingFilteredGaze"] += 1
            writer.write(frame)
            counts["framesWritten"] += 1
            frame_index += 1
            continue
        counts["framesWithFilteredGaze"] += 1

        shifted_projection = project(gaze_entry["shiftedPoint"], t_world_camera, end_camera)
        unshifted_projection = project(gaze_entry["filteredPoint"], t_world_camera, end_camera)
        if shifted_projection is None:
            counts["framesBehindCamera"] += 1
            writer.write(frame)
            counts["framesWritten"] += 1
            frame_index += 1
            continue
        counts["framesProjectedInFront"] += 1
        u, v, z = shifted_projection
        projection_u.append(float(u))
        projection_v.append(float(v))
        projection_z.append(float(z))
        if unshifted_projection is not None:
            pixel_shift.append(float(math.hypot(u - unshifted_projection[0], v - unshifted_projection[1])))
        depth, depth_scale_m, _depth_path = depth_payload_for_frame(robot_dir, frame_row)
        if depth is not None:
            depth_m = depth_window_median_m(depth, u, v, depth_scale_m, depth_radius_px)
            if depth_m is not None:
                counts["framesWithDepthAtProjection"] += 1
                end_depth_minus_gaze_z.append(float(depth_m - z))
                if unshifted_projection is not None:
                    end_depth_minus_unshifted_gaze_z.append(float(depth_m - unshifted_projection[2]))
                depth_surface_world = unproject_world(u, v, depth_m, t_world_camera, end_camera)
                surface_delta = depth_surface_world - gaze_entry["shiftedPoint"]
                end_depth_surface_world_distance.append(float(np.linalg.norm(surface_delta)))
                ray_dir = gaze_entry["direction"]
                along = float(np.dot(surface_delta, ray_dir))
                perp = float(np.linalg.norm(surface_delta - along * ray_dir))
                end_depth_surface_along_gaze_ray_error.append(along)
                end_depth_surface_perp_gaze_ray_error.append(perp)
            else:
                counts["framesMissingDepthAtProjection"] += 1
        else:
            counts["framesMissingDepthAtProjection"] += 1
        if gaze_entry.get("t") is not None and sample_time(robot_row) is not None:
            time_delta.append(float(sample_time(robot_row) - float(gaze_entry["t"])))

        if 0 <= u < width and 0 <= v < height:
            counts["framesProjectedInside"] += 1
            draw_crosshair(frame, u, v, "median + 1cm")
        else:
            counts["framesProjectedOutside"] += 1
        writer.write(frame)
        counts["framesWritten"] += 1
        frame_index += 1

    capture.release()
    writer.release()

    payload = {
        "record": record_dir.name,
        "inputVideo": str(input_video),
        "outputVideo": str(output_video),
        "inputFrameCount": frame_index,
        "fps": fps,
        "width": width,
        "height": height,
        "robotSamples": len(robot_rows),
        "pcSamples": len(pc_rows),
        **counts,
        "projectionU": stats(projection_u),
        "projectionV": stats(projection_v),
        "projectionZ": stats(projection_z),
        "pixelShiftFromMedianProjectionPx": stats(pixel_shift),
        "endCameraDepthMinusProjectedGazeZMm": stats([value * 1000.0 for value in end_depth_minus_gaze_z]),
        "endCameraDepthMinusUnshiftedGazeZMm": stats([value * 1000.0 for value in end_depth_minus_unshifted_gaze_z]),
        "endCameraDepthSurfaceWorldDistanceMm": stats([value * 1000.0 for value in end_depth_surface_world_distance]),
        "endCameraDepthSurfaceAlongGazeRayErrorMm": stats([value * 1000.0 for value in end_depth_surface_along_gaze_ray_error]),
        "endCameraDepthSurfacePerpGazeRayErrorMm": stats([value * 1000.0 for value in end_depth_surface_perp_gaze_ray_error]),
        "endCameraDepthSampling": {
            "radiusPx": int(depth_radius_px),
            "source": "aligned RealSense indexed depth payload at projected gaze pixel",
        },
        "robotMinusGazeTimestampSeconds": stats(time_delta),
        "filter": {
            "kind": "causal_median_depth",
            "window": median_window,
            "inputSamples": len(pc_rows),
            "filteredSamples": len(gaze_by_index),
            "sourceCounts": source_counts,
        },
        "offsetAlongRayM": offset_m,
        "coordinateFrame": "pc_world_rh_z_up_x_forward",
        "projectionChain": "pcWorld ray origin + ray direction * (median_depth + offset) -> inverse(T_world_end_camera) -> RealSense pinhole intrinsics",
        "depthDiagnosticInterpretation": {
            "depthMinusProjectedGazeZ": "positive means RealSense surface is farther from end camera than the rendered gaze point at the same pixel",
            "surfaceAlongGazeRayError": "positive means the RealSense surface lies farther along the Quest gaze ray than the shifted gaze point",
            "surfacePerpGazeRayError": "large values suggest lateral alignment/projection error, not just Quest depth error",
        },
    }
    if isinstance(payload.get("endCameraDepthSampling"), dict):
        payload["endCameraDepthSampling"]["source"] = "aligned RealSense indexed depth payload at projected gaze pixel"
    write_json(stats_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Render shifted median gaze3D reprojection onto the recorded end-camera MP4.")
    parser.add_argument("records", nargs="+", help="Record names or paths.")
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--median-window", type=int, default=7)
    parser.add_argument("--offset-m", type=float, default=0.01)
    parser.add_argument("--depth-radius-px", type=int, default=3, help="Median radius around the projected pixel in aligned RealSense depth.")
    parser.add_argument("--output-name", help="Output MP4 filename. Default includes offset in mm.")
    args = parser.parse_args()

    for item in args.records:
        record_dir = Path(item)
        if not record_dir.exists():
            record_dir = args.record_root / item
        result = render_record(
            record_dir.resolve(),
            median_window=max(1, int(args.median_window)),
            offset_m=float(args.offset_m),
            output_name=args.output_name,
            depth_radius_px=max(0, int(args.depth_radius_px)),
        )
        print(json.dumps({
            "record": result["record"],
            "outputVideo": result["outputVideo"],
            "framesWritten": result["framesWritten"],
            "inside": result["framesProjectedInside"],
            "outside": result["framesProjectedOutside"],
            "pixelShiftPx": result["pixelShiftFromMedianProjectionPx"],
            "depthMinusProjectedGazeZMm": result["endCameraDepthMinusProjectedGazeZMm"],
            "surfacePerpGazeRayErrorMm": result["endCameraDepthSurfacePerpGazeRayErrorMm"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
