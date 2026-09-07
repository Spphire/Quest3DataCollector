#!/usr/bin/env python3
"""
Convert PC Quest/RealSense recordings to a single-arm training zarr.

Expected input:
  pc_recordings/
    record_YYYYMMDD_HHMMSS/
      robot_realsense/
        samples.jsonl
        robot_states.jsonl
        gripper_commands.jsonl
        videos/end_*.mp4
        videos/third_*.mp4

The default ``umi`` output matches the legacy single-arm image datasets:
  data/timestamp
  data/left_robot_tcp_pose          [N, 9]
  data/left_robot_gripper_width     [N, 1]
  data/target                       [N, 10]
  data/action                       [N, 10]
  data/left_wrist_img               [N, H, W, 3]
  data/left_eye_img                 [N, H, W, 3]
  meta/episode_ends

The ``gaze-wam`` output writes the canonical robot contract:
  data/camera0_rgb                 [N, H, W, 3]
  data/camera1_rgb                 [N, H, W, 3], optional third-person camera
  data/gaze_xy                    [N, 2], normalized to the end-camera image
  data/action_abs_tcp             [N, 10]
  data/tcp_pose_abs               [N, 9]
  data/gripper_width              [N]
  data/{timestamp,image_timestamp,robot_state_timestamp,gaze_timestamp}
  data/{has_gaze_condition,has_gaze_label,has_heatmap_image}
  data/{gaze_world_pc,gaze_3d_source,gaze_projection_status}
  meta/episode_ends
"""

import argparse
import bisect
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logger = logging.getLogger("pc_recordings_to_zarr")

@dataclass
class FrameRequest:
    frame_index: int
    output_index: int


@dataclass
class EpisodePlan:
    record_id: str
    episode_dir: Path
    samples: List[dict]
    robot_states_by_index: Dict[int, dict]
    gripper_times: List[float]
    gripper_widths: List[float]
    start_output_index: int
    end_output_index: int
    camera: Optional[dict]
    segment_end_offsets: Optional[List[int]] = None
    gaze_stats: Optional[Dict[str, object]] = None


GAZE_3D_SOURCE_MISSING = 0
GAZE_3D_SOURCE_MEDIAN_FILTERED = 1
GAZE_3D_SOURCE_INTERPOLATED = 2

GAZE_PROJECTION_MISSING_3D = 0
GAZE_PROJECTION_VALID = 1
GAZE_PROJECTION_BEHIND_CAMERA = 2
GAZE_PROJECTION_OUT_OF_FRAME = 3
GAZE_PROJECTION_INVALID = 4

GAZE_3D_SOURCE_NAMES = {
    GAZE_3D_SOURCE_MISSING: "missing",
    GAZE_3D_SOURCE_MEDIAN_FILTERED: "causal_median_ray_depth",
    GAZE_3D_SOURCE_INTERPOLATED: "linear_interpolation_pc_world",
}

GAZE_PROJECTION_STATUS_NAMES = {
    GAZE_PROJECTION_MISSING_3D: "missing_3d",
    GAZE_PROJECTION_VALID: "valid",
    GAZE_PROJECTION_BEHIND_CAMERA: "behind_camera",
    GAZE_PROJECTION_OUT_OF_FRAME: "out_of_frame",
    GAZE_PROJECTION_INVALID: "invalid",
}

IMAGE_RESIZE_MODES = ("stretch", "letterbox")


def trim_quest_stale_endpoints(
    samples: List[dict],
    *,
    max_gaze_age_seconds: Optional[float],
    max_endpoint_reuse_frames: Optional[int] = 5,
) -> Tuple[List[dict], Dict[str, int]]:
    """Trim stale rows and long terminal Quest-source reuse runs.

    Internal stale/reused rows are deliberately retained for interpolation/masking.
    A long reuse run at an episode edge is treated as a Quest stop/idle tail, while
    short endpoint reuse (and all internal reuse) remains part of the episode.
    """
    if max_gaze_age_seconds is not None and max_gaze_age_seconds < 0.0:
        raise ValueError("max_gaze_age_seconds must be non-negative")
    if max_endpoint_reuse_frames is not None and max_endpoint_reuse_frames < 0:
        raise ValueError("max_endpoint_reuse_frames must be non-negative")

    def stale(row: dict) -> bool:
        aligned = row.get("pc_perf_counter_seconds")
        received = row.get("quest_pc_receive_perf_counter_seconds")
        return max_gaze_age_seconds is not None and (
            isinstance(aligned, (int, float))
            and isinstance(received, (int, float))
            and math.isfinite(float(aligned))
            and math.isfinite(float(received))
            and float(aligned) - float(received) > max_gaze_age_seconds
        )

    def reused(row: dict) -> bool:
        return bool(row.get("aligned_source_reused"))

    start = 0
    end = len(samples)
    stale_start = stale_end = reuse_start = reuse_end = 0
    while True:
        changed = False
        if max_endpoint_reuse_frames is not None:
            left = start
            while left < end and reused(samples[left]):
                left += 1
            right = end
            while right > start and reused(samples[right - 1]):
                right -= 1
            if left - start > max_endpoint_reuse_frames:
                reuse_start += left - start
                start = left
                changed = True
            if end - right > max_endpoint_reuse_frames:
                reuse_end += end - right
                end = right
                changed = True
        while start < end and stale(samples[start]):
            start += 1
            stale_start += 1
            changed = True
        while end > start and stale(samples[end - 1]):
            end -= 1
            stale_end += 1
            changed = True
        if not changed:
            break
    return samples[start:end], {
        "trim_start_frames": start,
        "trim_end_frames": len(samples) - end,
        "trim_start_stale_frames": stale_start,
        "trim_end_stale_frames": stale_end,
        "trim_start_reuse_frames": reuse_start,
        "trim_end_reuse_frames": reuse_end,
    }


def quest_reuse_metrics(samples: List[dict]) -> Dict[str, float | int]:
    """Return post-trim Quest-source reuse metrics for episode selection."""
    reused_count = 0
    longest = 0
    current = 0
    for sample in samples:
        if bool(sample.get("aligned_source_reused")):
            reused_count += 1
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "retained_frames": len(samples),
        "reused_frames": reused_count,
        "reuse_ratio": reused_count / len(samples) if samples else 1.0,
        "max_consecutive_reuse": longest,
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def load_record_ids(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    values = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                values.add(value)
    if not values:
        raise ValueError(f"Record allowlist is empty: {path}")
    return values


def find_record_dirs(input_dir: Path, record_ids: Optional[set[str]] = None) -> List[Path]:
    direct = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("record_"))
    if not direct:
        nested = input_dir / "pc_recordings"
        if nested.exists():
            direct = sorted(p for p in nested.iterdir() if p.is_dir() and p.name.startswith("record_"))
    if record_ids is None:
        return direct
    found = [p for p in direct if p.name in record_ids]
    missing = sorted(record_ids - {p.name for p in found})
    if missing:
        raise FileNotFoundError(f"Allowlisted record directories are missing: {missing}")
    return found


def load_robot_states(robot_states_path: Path) -> Dict[int, dict]:
    states = {}
    for row in iter_jsonl(robot_states_path):
        if not row.get("ok", False):
            continue
        sample_index = row.get("sample_index")
        if sample_index is None:
            continue
        states[int(sample_index)] = row
    return states


def load_gripper_timeline(gripper_path: Path, default_width: float) -> Tuple[List[float], List[float]]:
    times = []
    widths = []
    if gripper_path.exists():
        for row in iter_jsonl(gripper_path):
            if not row.get("ok", False):
                continue
            width = row.get("target_width_m")
            if width is None:
                width = row.get("status", {}).get("command", {}).get("width")
            if width is None:
                width = row.get("status", {}).get("states", {}).get("width")
            timestamp = row.get("pc_perf_counter_seconds")
            if width is None or timestamp is None:
                continue
            times.append(float(timestamp))
            widths.append(float(width))

    if not times:
        return [-float("inf")], [float(default_width)]

    order = np.argsort(np.asarray(times))
    times = [times[i] for i in order]
    widths = [widths[i] for i in order]
    if times[0] != -float("inf"):
        times.insert(0, -float("inf"))
        widths.insert(0, widths[0])
    return times, widths


def gripper_width_at(times: List[float], widths: List[float], timestamp: float) -> float:
    idx = bisect.bisect_right(times, timestamp) - 1
    idx = max(0, min(idx, len(widths) - 1))
    return widths[idx]


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = quat / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_payload_to_matrix(value: dict) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError("Pose payload must be an object")
    translation = np.asarray(value.get("translation_m"), dtype=np.float64)
    quat = np.asarray(value.get("quaternion_wxyz"), dtype=np.float64)
    if translation.shape != (3,) or quat.shape != (4,):
        raise ValueError("Pose payload must contain translation_m[3] and quaternion_wxyz[4]")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_wxyz_to_matrix(quat)
    matrix[:3, 3] = translation
    return matrix


def project_world_point_with_status(
    point_world: object,
    world_camera_pose: object,
    camera: dict,
) -> Tuple[Optional[np.ndarray], int]:
    try:
        point = np.asarray(point_world, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None, GAZE_PROJECTION_MISSING_3D
        world_camera = pose_payload_to_matrix(world_camera_pose)
        point_camera = np.linalg.inv(world_camera) @ np.asarray([*point, 1.0], dtype=np.float64)
        x, y, z = point_camera[:3]
        if not np.all(np.isfinite(point_camera)):
            return None, GAZE_PROJECTION_INVALID
        if z <= 1e-9:
            return None, GAZE_PROJECTION_BEHIND_CAMERA
        u = float(camera["fx"]) * float(x) / float(z) + float(camera["cx"])
        v = float(camera["fy"]) * float(y) / float(z) + float(camera["cy"])
        width = float(camera["width"])
        height = float(camera["height"])
        if not all(math.isfinite(value) for value in (u, v, width, height)):
            return None, GAZE_PROJECTION_INVALID
        if width <= 0 or height <= 0:
            return None, GAZE_PROJECTION_INVALID
        projection = np.asarray([u / width, v / height], dtype=np.float32)
        if not (0.0 <= u < width and 0.0 <= v < height):
            return projection, GAZE_PROJECTION_OUT_OF_FRAME
        return projection, GAZE_PROJECTION_VALID
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None, GAZE_PROJECTION_INVALID


def project_world_point(point_world: object, world_camera_pose: object, camera: dict) -> Optional[np.ndarray]:
    projection, status = project_world_point_with_status(point_world, world_camera_pose, camera)
    return projection if status == GAZE_PROJECTION_VALID else None


def letterbox_geometry(
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Dict[str, object]:
    source_h, source_w = (int(source_size[0]), int(source_size[1]))
    target_h, target_w = (int(target_size[0]), int(target_size[1]))
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("source and target image dimensions must be positive")

    scale = min(float(target_w) / float(source_w), float(target_h) / float(source_h))
    resized_w = min(target_w, max(1, int(round(float(source_w) * scale))))
    resized_h = min(target_h, max(1, int(round(float(source_h) * scale))))
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    return {
        "source_size": [source_h, source_w],
        "target_size": [target_h, target_w],
        "resized_size": [resized_h, resized_w],
        "scale_xy": [float(resized_w) / float(source_w), float(resized_h) / float(source_h)],
        "padding_ltrb": [
            pad_left,
            pad_top,
            target_w - resized_w - pad_left,
            target_h - resized_h - pad_top,
        ],
    }


def camera_source_size(episode_dir: Path, role: str) -> Tuple[int, int]:
    cameras_path = episode_dir / "cameras.json"
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    camera = cameras.get(role)
    if not isinstance(camera, dict):
        raise KeyError(f"Camera role {role!r} is missing from {cameras_path}")
    height = int(camera.get("height", 0))
    width = int(camera.get("width", 0))
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Camera role {role!r} has invalid source size {width}x{height} in {cameras_path}"
        )
    return height, width


def remap_normalized_gaze_xy(
    gaze_xy: np.ndarray,
    *,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    image_resize_mode: str,
) -> np.ndarray:
    point = np.asarray(gaze_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("gaze_xy must contain two finite normalized coordinates")
    if image_resize_mode == "stretch":
        return point.astype(np.float32)
    if image_resize_mode != "letterbox":
        raise ValueError(f"Unsupported image resize mode: {image_resize_mode!r}")

    geometry = letterbox_geometry(source_size, target_size)
    source_h, source_w = geometry["source_size"]
    target_h, target_w = geometry["target_size"]
    scale_x, scale_y = geometry["scale_xy"]
    pad_left, pad_top, _, _ = geometry["padding_ltrb"]
    output_x = float(point[0]) * float(source_w) * float(scale_x) + float(pad_left)
    output_y = float(point[1]) * float(source_h) * float(scale_y) + float(pad_top)
    return np.asarray([output_x / float(target_w), output_y / float(target_h)], dtype=np.float32)


def normalize_vector(value: object) -> Optional[np.ndarray]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return None
    return vector / norm


def load_quest_gaze_rays(record_dir: Path, median_window: int) -> Dict[int, dict]:
    path = record_dir / "pc_samples.jsonl"
    if not path.exists():
        return {}
    rays = {}
    recent_depths: List[float] = []
    for row in iter_jsonl(path):
        sample_index = row.get("sampleIndex")
        pc_world = row.get("pcWorld") if isinstance(row.get("pcWorld"), dict) else {}
        origin = pc_world.get("gazeRayOrigin")
        direction = normalize_vector(pc_world.get("gazeRayDirection"))
        point = pc_world.get("gazePoint3DWorld")
        if sample_index is None or direction is None:
            continue
        try:
            origin_array = np.asarray(origin, dtype=np.float64)
            point_array = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if (
            origin_array.shape != (3,)
            or point_array.shape != (3,)
            or not np.all(np.isfinite(origin_array))
            or not np.all(np.isfinite(point_array))
        ):
            continue
        depth = float(np.dot(point_array - origin_array, direction))
        if not math.isfinite(depth) or depth <= 0.0:
            continue
        recent_depths.append(depth)
        if len(recent_depths) > median_window:
            recent_depths = recent_depths[-median_window:]
        filtered_depth = float(np.median(np.asarray(recent_depths, dtype=np.float64)))
        rays[int(sample_index)] = {
            "origin": origin_array,
            "direction": direction,
            "depth": depth,
            "filtered_point": origin_array + direction * filtered_depth,
        }
    return rays


def sample_time(sample: dict, fallback_index: int) -> float:
    for key in (
        "aligned_target_perf_counter_seconds",
        "pc_perf_counter_seconds",
        "quest_pc_receive_perf_counter_seconds",
    ):
        value = sample.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return float(fallback_index) / 30.0


def build_smoothed_interpolated_gaze(
    samples: List[dict],
    quest_rays_by_index: Dict[int, dict],
    *,
    median_window: int,
    segment_end_offsets: List[int],
    max_gaze_age_seconds: Optional[float],
) -> Dict[str, int]:
    if median_window < 1:
        raise ValueError("gaze median window must be at least 1")

    fallback_xyz_count = 0
    allow_xyz_fallback = not quest_rays_by_index
    recent_xyz: List[np.ndarray] = []
    for sample in samples:
        aligned_time = sample.get("pc_perf_counter_seconds")
        gaze_time = sample.get("quest_pc_receive_perf_counter_seconds")
        gaze_is_stale = (
            max_gaze_age_seconds is not None
            and isinstance(aligned_time, (int, float))
            and isinstance(gaze_time, (int, float))
            and float(aligned_time) - float(gaze_time) > max_gaze_age_seconds
        )
        source_index = sample.get("quest_sample_index")
        # A reused aligned row carries the previous Quest payload by design. Do not
        # treat that payload as a fresh gaze label; internal short runs are filled
        # from the surrounding fresh samples below, while endpoint runs are trimmed.
        if bool(sample.get("aligned_source_reused")):
            sample["_gaze_was_reused"] = True
            source_index = None
        ray = (
            quest_rays_by_index.get(int(source_index))
            if source_index is not None and not gaze_is_stale
            else None
        )
        point = None
        if ray is not None:
            point = np.asarray(ray["filtered_point"], dtype=np.float64)
        else:
            try:
                raw_point = np.asarray(sample.get("quest_gaze3d_pc_world"), dtype=np.float64)
            except (TypeError, ValueError):
                raw_point = np.empty((0,), dtype=np.float64)
            if (
                allow_xyz_fallback
                and not gaze_is_stale
                and not sample.get("_gaze_was_reused", False)
                and raw_point.shape == (3,)
                and np.all(np.isfinite(raw_point))
            ):
                recent_xyz.append(raw_point)
                if len(recent_xyz) > median_window:
                    recent_xyz = recent_xyz[-median_window:]
                point = np.median(np.stack(recent_xyz, axis=0), axis=0)
                fallback_xyz_count += 1

        sample["_gaze_world_pc"] = point
        sample["_gaze_was_stale"] = gaze_is_stale
        sample["_gaze_3d_source"] = (
            GAZE_3D_SOURCE_MEDIAN_FILTERED if point is not None else GAZE_3D_SOURCE_MISSING
        )

    interpolated_count = 0
    segment_start = 0
    for segment_end in segment_end_offsets:
        segment = samples[segment_start:segment_end]
        valid_indices = [
            index for index, sample in enumerate(segment)
            if sample.get("_gaze_world_pc") is not None
        ]
        for left_index, right_index in zip(valid_indices, valid_indices[1:]):
            if right_index <= left_index + 1:
                continue
            left = segment[left_index]
            right = segment[right_index]
            left_time = sample_time(left, segment_start + left_index)
            right_time = sample_time(right, segment_start + right_index)
            if right_time <= left_time:
                continue
            left_point = np.asarray(left["_gaze_world_pc"], dtype=np.float64)
            right_point = np.asarray(right["_gaze_world_pc"], dtype=np.float64)
            for index in range(left_index + 1, right_index):
                target = segment[index]
                if target.get("_gaze_world_pc") is not None:
                    continue
                target_time = sample_time(target, segment_start + index)
                alpha = float(np.clip((target_time - left_time) / (right_time - left_time), 0.0, 1.0))
                target["_gaze_world_pc"] = (1.0 - alpha) * left_point + alpha * right_point
                target["_gaze_3d_source"] = GAZE_3D_SOURCE_INTERPOLATED
                interpolated_count += 1
        segment_start = segment_end

    return {
        "median_filtered_3d": sum(
            1 for sample in samples
            if sample.get("_gaze_3d_source") == GAZE_3D_SOURCE_MEDIAN_FILTERED
        ),
        "interpolated_3d": interpolated_count,
        "missing_3d": sum(
            1 for sample in samples
            if sample.get("_gaze_3d_source") == GAZE_3D_SOURCE_MISSING
        ),
        "fallback_xyz_median": fallback_xyz_count,
        "stale_input_3d": sum(1 for sample in samples if sample.get("_gaze_was_stale")),
        "reused_input_3d": sum(1 for sample in samples if sample.get("_gaze_was_reused")),
    }


def pose_7d_to_pose_9d(pose_7d: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_matrix(pose_7d[3:7])
    rot_6d = rot[:, :2].T.reshape(-1)
    return np.concatenate([pose_7d[:3], rot_6d], axis=0)


def pose_row_to_9d(row: dict, pose_frame: str) -> np.ndarray:
    pose = row.get(pose_frame)
    if pose is None:
        raise KeyError(f"Missing pose frame {pose_frame}")
    xyz = pose["translation_m"]
    quat_wxyz = pose["quaternion_wxyz"]
    pose_7d = np.asarray([*xyz, *quat_wxyz], dtype=np.float64)
    return pose_7d_to_pose_9d(pose_7d).astype(np.float32)


def resolve_video_path(episode_dir: Path, role: str, samples: List[dict]) -> Optional[Path]:
    for sample in samples:
        video_info = sample.get("videos", {}).get(role)
        if video_info and video_info.get("path"):
            path = episode_dir / video_info["path"]
            if path.exists():
                return path

    videos_dir = episode_dir / "videos"
    matches = sorted(videos_dir.glob(f"{role}_*.mp4"))
    if matches:
        return matches[0]
    return None


def collect_valid_samples(
    samples_path: Path,
    robot_states_by_index: Dict[int, dict],
    required_roles: Tuple[str, str],
    max_image_age_seconds: Optional[float],
) -> Tuple[List[dict], Dict[str, int]]:
    valid = []
    stats = {
        "total_rows": 0,
        "invalid_rows": 0,
        "missing_robot_state": 0,
        "missing_video_mapping": 0,
        "stale_image": 0,
    }
    for sample in iter_jsonl(samples_path):
        stats["total_rows"] += 1
        if not sample.get("ok", False):
            stats["invalid_rows"] += 1
            continue
        robot_state_index = sample.get("robot_state_sample_index")
        if robot_state_index is None or int(robot_state_index) not in robot_states_by_index:
            stats["missing_robot_state"] += 1
            continue
        videos = sample.get("videos", {})
        video_frames = sample.get("videoFrames", {})
        if any(role not in videos or videos[role].get("frameIndex") is None for role in required_roles):
            stats["missing_video_mapping"] += 1
            continue
        if max_image_age_seconds is not None:
            too_old = False
            for role in required_roles:
                age = video_frames.get(role, {}).get("ageSeconds")
                if age is not None and float(age) > max_image_age_seconds:
                    too_old = True
                    break
            if too_old:
                stats["stale_image"] += 1
                continue
        valid.append(sample)
    return valid, stats


def load_video_frame_timelines(
    episode_dir: Path,
    roles: Tuple[str, ...],
) -> Dict[str, List[dict]]:
    path = episode_dir / "video_frames.jsonl"
    timelines = {role: [] for role in roles}
    if not path.exists():
        return timelines
    for row in iter_jsonl(path):
        role = row.get("role")
        if role in timelines:
            timelines[role].append(row)
    for role in roles:
        timelines[role].sort(key=lambda row: int(row.get("frame_index", -1)))
    return timelines


def _finite_time_array(values: Iterable[object]) -> Optional[np.ndarray]:
    try:
        result = np.asarray(list(values), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        return None
    return result


def _best_video_window(
    sample_times: np.ndarray,
    frames: List[dict],
    *,
    max_trim_frames: int,
    max_alignment_delta_seconds: Optional[float],
    max_stream_gap_seconds: Optional[float],
) -> Optional[dict]:
    retained_frames = int(sample_times.shape[0])
    if retained_frames < 2 or len(frames) < retained_frames:
        return None
    capture_times = _finite_time_array(
        frame.get("frame_captured_perf_counter_seconds") for frame in frames
    )
    if capture_times is None:
        return None

    candidates = []
    max_frame_start = min(max_trim_frames, len(frames) - retained_frames)
    for frame_start in range(max_frame_start + 1):
        frame_end = frame_start + retained_frames
        trim_end = len(frames) - frame_end
        if trim_end > max_trim_frames:
            continue
        selected_times = capture_times[frame_start:frame_end]
        deltas = np.abs(selected_times - sample_times)
        capture_gaps = np.diff(selected_times)
        if max_alignment_delta_seconds is not None and float(np.max(deltas)) > max_alignment_delta_seconds:
            continue
        if capture_gaps.size and (
            np.any(capture_gaps <= 0.0)
            or (
                max_stream_gap_seconds is not None
                and float(np.max(capture_gaps)) > max_stream_gap_seconds
            )
        ):
            continue
        candidates.append(
            {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "trim_start_frames": frame_start,
                "trim_end_frames": trim_end,
                "max_alignment_delta_seconds": float(np.max(deltas)),
                "mean_alignment_delta_seconds": float(np.mean(deltas)),
                "p95_alignment_delta_seconds": float(np.percentile(deltas, 95)),
                "max_capture_gap_seconds": float(np.max(capture_gaps)) if capture_gaps.size else 0.0,
            }
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["trim_start_frames"] + item["trim_end_frames"],
            item["max_alignment_delta_seconds"],
            item["mean_alignment_delta_seconds"],
        ),
    )


def trim_and_attach_video_timelines(
    samples: List[dict],
    timelines: Dict[str, List[dict]],
    *,
    roles: Tuple[str, ...],
    max_image_age_seconds: Optional[float],
    max_sample_gap_seconds: Optional[float],
    max_endpoint_trim_seconds: float,
    sample_hz: float = 30.0,
) -> Tuple[List[dict], List[str], dict]:
    if max_endpoint_trim_seconds < 0.0:
        raise ValueError("max_endpoint_trim_seconds must be non-negative")
    if sample_hz <= 0.0:
        raise ValueError("sample_hz must be positive")
    max_trim_frames = int(math.floor(max_endpoint_trim_seconds * sample_hz + 1e-9))
    metrics = {
        "original_aligned_rows": len(samples),
        "max_endpoint_trim_seconds": float(max_endpoint_trim_seconds),
        "max_endpoint_trim_frames": max_trim_frames,
        "roles": {},
    }

    for role in roles:
        frames = timelines.get(role, [])
        metrics["roles"][role] = {"original_video_frames": len(frames)}
        indices = [int(row.get("frame_index", -1)) for row in frames]
        if indices != list(range(len(frames))):
            return [], [f"{role}_frame_index_not_contiguous"], metrics

    candidates = []
    max_sample_start = min(max_trim_frames, max(0, len(samples) - 2))
    for sample_start in range(max_sample_start + 1):
        max_sample_end_trim = min(max_trim_frames, len(samples) - sample_start - 2)
        for sample_end_trim in range(max_sample_end_trim + 1):
            sample_end = len(samples) - sample_end_trim
            retained = samples[sample_start:sample_end]
            sample_times = _finite_time_array(
                sample_time(sample, sample_start + index)
                for index, sample in enumerate(retained)
            )
            if sample_times is None or sample_times.shape[0] < 2:
                continue
            sample_gaps = np.diff(sample_times)
            if np.any(sample_gaps <= 0.0) or (
                max_sample_gap_seconds is not None
                and float(np.max(sample_gaps)) > max_sample_gap_seconds
            ):
                continue

            role_windows = {}
            for role in roles:
                window = _best_video_window(
                    sample_times,
                    timelines[role],
                    max_trim_frames=max_trim_frames,
                    max_alignment_delta_seconds=max_image_age_seconds,
                    max_stream_gap_seconds=max_sample_gap_seconds,
                )
                if window is None:
                    break
                role_windows[role] = window
            if len(role_windows) != len(roles):
                continue
            total_trim = sample_start + sample_end_trim + sum(
                window["trim_start_frames"] + window["trim_end_frames"]
                for window in role_windows.values()
            )
            candidates.append(
                {
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                    "sample_end_trim": sample_end_trim,
                    "sample_max_gap_seconds": float(np.max(sample_gaps)),
                    "role_windows": role_windows,
                    "score": (
                        total_trim,
                        max(window["max_alignment_delta_seconds"] for window in role_windows.values()),
                        sum(window["mean_alignment_delta_seconds"] for window in role_windows.values()),
                    ),
                }
            )

    if not candidates:
        return [], [
            "no_continuous_common_window_within_endpoint_trim_and_timing_limits"
        ], metrics

    best = min(candidates, key=lambda item: item["score"])
    retained_samples = samples[best["sample_start"]:best["sample_end"]]
    metrics.update({
        "retained_aligned_rows": len(retained_samples),
        "aligned_trim_start_frames": best["sample_start"],
        "aligned_trim_end_frames": best["sample_end_trim"],
        "aligned_max_gap_seconds": best["sample_max_gap_seconds"],
    })
    for role in roles:
        window = best["role_windows"][role]
        metrics["roles"][role].update(window)
        selected_frames = timelines[role][window["frame_start"]:window["frame_end"]]
        for sample, frame in zip(retained_samples, selected_frames):
            sample.setdefault("videos", {})[role] = {
                "path": frame.get("video"),
                "frameIndex": int(frame["frame_index"]),
                "serial": frame.get("serial"),
            }
            sample.setdefault("videoFrames", {})[role] = {
                "capturedPerfCounterSeconds": float(
                    frame["frame_captured_perf_counter_seconds"]
                ),
                "capturedAtUtc": frame.get("frame_captured_at_utc"),
                "streamSequence": frame.get("stream_sequence"),
            }
    return retained_samples, [], metrics


def find_sample_timeline_discontinuities(
    samples: List[dict],
    max_gap_seconds: Optional[float],
) -> List[dict]:
    discontinuities = []
    if max_gap_seconds is not None:
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        for index in range(1, len(samples)):
            previous_sample = samples[index - 1]
            current_sample = samples[index]
            previous_aligned = previous_sample.get("aligned_target_perf_counter_seconds")
            current_aligned = current_sample.get("aligned_target_perf_counter_seconds")
            if previous_aligned is None or current_aligned is None:
                previous_aligned = previous_sample.get("pc_perf_counter_seconds")
                current_aligned = current_sample.get("pc_perf_counter_seconds")
            timestamp_pairs = ((previous_aligned, current_aligned),)
            has_discontinuity = any(
                previous is not None
                and current is not None
                and (
                    float(current) < float(previous)
                    or float(current) - float(previous) > max_gap_seconds
                )
                for previous, current in timestamp_pairs
            )
            if has_discontinuity:
                discontinuities.append(
                    {
                        "previous_index": index - 1,
                        "current_index": index,
                        "previous_sample_index": previous_sample.get("sample_index"),
                        "current_sample_index": current_sample.get("sample_index"),
                    }
                )
    return discontinuities


def build_plans(
    input_dir: Path,
    wrist_role: str,
    eye_role: str,
    pose_frame: str,
    default_gripper_width: float,
    max_image_age_seconds: Optional[float],
    limit_records: Optional[int],
    record_ids: Optional[set[str]],
    require_camera_intrinsics: bool,
    prepare_gaze: bool,
    gaze_median_window: int,
    max_gaze_age_seconds: Optional[float],
    max_sample_gap_seconds: Optional[float],
    max_endpoint_trim_seconds: float,
    max_consecutive_reuse: Optional[int] = 5,
    max_fallback_reuse_ratio: Optional[float] = 0.20,
    selection_report: Optional[dict] = None,
) -> List[EpisodePlan]:
    plans = []
    total = 0
    record_dirs = find_record_dirs(input_dir, record_ids=record_ids)
    if limit_records is not None:
        record_dirs = record_dirs[:limit_records]
    if selection_report is not None:
        selection_report["candidate_record_ids"] = [path.name for path in record_dirs]
        selection_report["accepted"] = []
        selection_report["excluded"] = []

    def reject(record_id: str, reasons: List[str], details: Optional[dict] = None) -> None:
        if selection_report is not None:
            selection_report["excluded"].append(
                {
                    "record_id": record_id,
                    "reasons": list(reasons),
                    "details": dict(details or {}),
                }
            )

    for record_dir in record_dirs:
        episode_dir = record_dir / "robot_realsense"
        samples_path = episode_dir / "samples.jsonl"
        robot_states_path = episode_dir / "robot_states.jsonl"
        gripper_path = episode_dir / "gripper_commands.jsonl"
        cameras_path = episode_dir / "cameras.json"
        if not samples_path.exists() or not robot_states_path.exists():
            logger.warning(f"Skip {record_dir.name}: missing samples.jsonl or robot_states.jsonl")
            reject(
                record_dir.name,
                ["missing_core_telemetry"],
                {
                    "missing_samples_jsonl": not samples_path.exists(),
                    "missing_robot_states_jsonl": not robot_states_path.exists(),
                },
            )
            continue
        if require_camera_intrinsics and not cameras_path.exists():
            logger.warning(f"Skip {record_dir.name}: missing cameras.json")
            reject(record_dir.name, ["missing_camera_intrinsics_file"])
            continue
        cameras = json.loads(cameras_path.read_text(encoding="utf-8")) if cameras_path.exists() else {}
        camera = cameras.get(wrist_role)
        if require_camera_intrinsics and not isinstance(camera, dict):
            logger.warning(f"Skip {record_dir.name}: missing {wrist_role} camera intrinsics")
            reject(
                record_dir.name,
                ["missing_camera_intrinsics"],
                {"camera_role": wrist_role},
            )
            continue

        robot_states = load_robot_states(robot_states_path)
        if prepare_gaze:
            all_samples = [sample for sample in iter_jsonl(samples_path)]
            invalid_rows = sum(1 for sample in all_samples if not sample.get("ok", False))
            missing_robot_state = sum(
                1 for sample in all_samples
                if sample.get("robot_state_sample_index") is None
                or int(sample["robot_state_sample_index"]) not in robot_states
            )
            valid_samples, video_reasons, video_metrics = trim_and_attach_video_timelines(
                all_samples,
                load_video_frame_timelines(episode_dir, (wrist_role, eye_role)),
                roles=(wrist_role, eye_role),
                max_image_age_seconds=max_image_age_seconds,
                max_sample_gap_seconds=max_sample_gap_seconds,
                max_endpoint_trim_seconds=max_endpoint_trim_seconds,
            )
            valid_samples, quest_endpoint_trim = trim_quest_stale_endpoints(
                valid_samples,
                max_gaze_age_seconds=max_gaze_age_seconds,
                max_endpoint_reuse_frames=max_consecutive_reuse,
            )
            reuse_metrics = quest_reuse_metrics(valid_samples)
            if video_metrics is None:
                video_metrics = {}
            video_metrics["quest_endpoint_trim"] = quest_endpoint_trim
            video_metrics["quest_source_reuse"] = reuse_metrics
            discontinuities = find_sample_timeline_discontinuities(
                valid_samples,
                max_gap_seconds=max_sample_gap_seconds,
            )
            excessive_reuse = (
                max_consecutive_reuse is not None
                and reuse_metrics["max_consecutive_reuse"] > max_consecutive_reuse
            )
            excessive_reuse_ratio = (
                max_fallback_reuse_ratio is not None
                and reuse_metrics["reuse_ratio"] > max_fallback_reuse_ratio
            )
            if (
                invalid_rows > 0
                or missing_robot_state > 0
                or video_reasons
                or discontinuities
                or excessive_reuse
                or excessive_reuse_ratio
            ):
                reasons = []
                if invalid_rows > 0:
                    reasons.append("invalid_aligned_rows")
                if missing_robot_state > 0:
                    reasons.append("missing_robot_state")
                reasons.extend(video_reasons)
                if discontinuities:
                    reasons.append("timeline_discontinuity")
                if excessive_reuse:
                    reasons.append("aligned_quest_consecutive_reuse")
                if excessive_reuse_ratio:
                    reasons.append("aligned_quest_reuse_ratio")
                logger.warning(
                    f"Skip {record_dir.name}: strict episode integrity failed "
                    f"(invalid_rows={invalid_rows}, missing_robot_state={missing_robot_state}, "
                    f"video_reasons={video_reasons}, timeline_discontinuities={len(discontinuities)})"
                )
                reject(
                    record_dir.name,
                    reasons,
                    {
                        "invalid_rows": invalid_rows,
                        "missing_robot_state_rows": missing_robot_state,
                        "video_reasons": video_reasons,
                        "timeline_discontinuities": discontinuities,
                        "video_alignment": video_metrics,
                        "quest_source_reuse": reuse_metrics,
                    },
                )
                continue
        else:
            valid_samples, _ = collect_valid_samples(
                samples_path=samples_path,
                robot_states_by_index=robot_states,
                required_roles=(wrist_role, eye_role),
                max_image_age_seconds=max_image_age_seconds,
            )
            video_metrics = None
        segment_end_offsets = [len(valid_samples)] if valid_samples else []
        if len(valid_samples) < 2:
            logger.warning(f"Skip {record_dir.name}: only {len(valid_samples)} valid samples")
            reject(
                record_dir.name,
                ["too_few_valid_samples"],
                {"valid_samples": len(valid_samples)},
            )
            continue

        try:
            first_state = robot_states[int(valid_samples[0]["robot_state_sample_index"])]
            pose_row_to_9d(first_state, pose_frame)
        except Exception as exc:
            logger.warning(f"Skip {record_dir.name}: cannot read {pose_frame}: {exc}")
            reject(
                record_dir.name,
                ["invalid_tcp_pose"],
                {"pose_frame": pose_frame, "error": str(exc)},
            )
            continue

        wrist_video = resolve_video_path(episode_dir, wrist_role, valid_samples)
        eye_video = resolve_video_path(episode_dir, eye_role, valid_samples)
        if wrist_video is None or eye_video is None:
            logger.warning(f"Skip {record_dir.name}: missing required videos")
            reject(
                record_dir.name,
                ["missing_required_video"],
                {
                    "missing_roles": [
                        role
                        for role, video in ((wrist_role, wrist_video), (eye_role, eye_video))
                        if video is None
                    ]
                },
            )
            continue

        gripper_times, gripper_widths = load_gripper_timeline(gripper_path, default_gripper_width)
        gaze_stats = None
        if prepare_gaze:
            gaze_stats = build_smoothed_interpolated_gaze(
                valid_samples,
                load_quest_gaze_rays(record_dir, gaze_median_window),
                median_window=gaze_median_window,
                segment_end_offsets=segment_end_offsets,
                max_gaze_age_seconds=max_gaze_age_seconds,
            )
            projection_counts = {name: 0 for name in GAZE_PROJECTION_STATUS_NAMES.values()}
            for sample in valid_samples:
                state_index = sample.get("robot_state_sample_index")
                robot_state = robot_states.get(int(state_index)) if state_index is not None else None
                _, projection_status = project_world_point_with_status(
                    sample.get("_gaze_world_pc"),
                    robot_state.get("T_world_end_camera") if robot_state else None,
                    camera,
                )
                projection_counts[GAZE_PROJECTION_STATUS_NAMES[projection_status]] += 1
            gaze_stats.update({f"projection_{key}": value for key, value in projection_counts.items()})
            gaze_stats["video_alignment"] = video_metrics
        start = total
        total += len(valid_samples)
        plans.append(EpisodePlan(
            record_id=record_dir.name,
            episode_dir=episode_dir,
            samples=valid_samples,
            robot_states_by_index=robot_states,
            gripper_times=gripper_times,
            gripper_widths=gripper_widths,
            start_output_index=start,
            end_output_index=total,
            camera=camera,
            segment_end_offsets=segment_end_offsets,
            gaze_stats=gaze_stats,
        ))
        if selection_report is not None:
            selection_report["accepted"].append(
                {
                    "record_id": record_dir.name,
                    "frames": len(valid_samples),
                    "video_alignment": video_metrics,
                }
            )
        logger.info(
            f"{record_dir.name}: {len(valid_samples)} action/image samples in "
            f"one intact episode "
            f"(gaze={gaze_stats})"
        )

    return plans


def create_output_zarr(
    output_path: Path,
    total_frames: int,
    image_size: Tuple[int, int],
    overwrite: bool,
    output_format: str,
    include_eye_camera: bool = False,
):
    import zarr

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_path}")
        logger.warning(f"Overwriting {output_path}")
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.group(str(output_path))
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    h, w = image_size

    # Keep absolute performance-counter timestamps in float64.  At values around
    # 1e6 seconds, float32 has a resolution of about 0.125 s and collapses
    # consecutive 30 Hz samples to identical timestamps.
    data.create_dataset("timestamp", shape=(total_frames,), chunks=(10000,), dtype="float64", compressor=compressor)
    if output_format == "gaze-wam":
        for key in ("image_timestamp", "robot_state_timestamp", "gaze_timestamp"):
            data.create_dataset(key, shape=(total_frames,), chunks=(10000,), dtype="float64", compressor=compressor)
        data.create_dataset("tcp_pose_abs", shape=(total_frames, 9), chunks=(10000, 9), dtype="float32", compressor=compressor)
        data.create_dataset("gripper_width", shape=(total_frames,), chunks=(10000,), dtype="float32", compressor=compressor)
        data.create_dataset("action_abs_tcp", shape=(total_frames, 10), chunks=(10000, 10), dtype="float32", compressor=compressor)
        data.create_dataset("action_timestamp", shape=(total_frames,), chunks=(10000,), dtype="float64", compressor=compressor)
        data.create_dataset("gaze_xy", shape=(total_frames, 2), chunks=(10000, 2), dtype="float32", compressor=compressor)
        data.create_dataset("gaze_world_pc", shape=(total_frames, 3), chunks=(10000, 3), dtype="float32", compressor=compressor)
        data.create_dataset("gaze_3d_source", shape=(total_frames,), chunks=(10000,), dtype="uint8", compressor=compressor)
        data.create_dataset("gaze_projection_status", shape=(total_frames,), chunks=(10000,), dtype="uint8", compressor=compressor)
        data.create_dataset("has_gaze_condition", shape=(total_frames,), chunks=(10000,), dtype="bool", compressor=compressor)
        data.create_dataset("has_gaze_label", shape=(total_frames,), chunks=(10000,), dtype="bool", compressor=compressor)
        data.create_dataset("has_heatmap_image", shape=(total_frames,), chunks=(10000,), dtype="bool", compressor=compressor)
        data.create_dataset(
            "camera0_rgb",
            shape=(total_frames, h, w, 3),
            chunks=(16, h, w, 3),
            dtype="uint8",
            compressor=compressor,
        )
        if include_eye_camera:
            data.create_dataset(
                "camera1_image_timestamp",
                shape=(total_frames,),
                chunks=(10000,),
                dtype="float64",
                compressor=compressor,
            )
            data.create_dataset(
                "camera1_rgb",
                shape=(total_frames, h, w, 3),
                chunks=(16, h, w, 3),
                dtype="uint8",
                compressor=compressor,
            )
    else:
        data.create_dataset("left_robot_tcp_pose", shape=(total_frames, 9), chunks=(10000, 9), dtype="float32", compressor=compressor)
        data.create_dataset("left_robot_gripper_width", shape=(total_frames, 1), chunks=(10000, 1), dtype="float32", compressor=compressor)
        data.create_dataset("target", shape=(total_frames, 10), chunks=(10000, 10), dtype="float32", compressor=compressor)
        data.create_dataset("action", shape=(total_frames, 10), chunks=(10000, 10), dtype="float32", compressor=compressor)
        data.create_dataset("left_wrist_img", shape=(total_frames, h, w, 3), chunks=(100, h, w, 3), dtype="uint8")
        data.create_dataset("left_eye_img", shape=(total_frames, h, w, 3), chunks=(100, h, w, 3), dtype="uint8")
    return root, data, meta


def write_lowdim_arrays(
    data,
    meta,
    plans: List[EpisodePlan],
    pose_frame: str,
    output_format: str,
    image_size: Tuple[int, int],
    image_resize_mode: str,
    eye_role: str = "third",
    include_eye_camera: bool = False,
):
    episode_ends = []
    for plan in plans:
        start = plan.start_output_index
        end = plan.end_output_index
        n = end - start
        timestamps = np.zeros((n,), dtype=np.float64)
        tcp = np.zeros((n, 9), dtype=np.float32)
        gripper = np.zeros((n, 1), dtype=np.float32)
        image_timestamps = np.zeros((n,), dtype=np.float64)
        eye_image_timestamps = np.zeros((n,), dtype=np.float64)
        robot_timestamps = np.zeros((n,), dtype=np.float64)
        gaze_timestamps = np.zeros((n,), dtype=np.float64)
        gaze_xy = np.zeros((n, 2), dtype=np.float32)
        gaze_world_pc = np.zeros((n, 3), dtype=np.float32)
        gaze_3d_source = np.zeros((n,), dtype=np.uint8)
        gaze_projection_status = np.zeros((n,), dtype=np.uint8)
        has_gaze_condition = np.zeros((n,), dtype=np.bool_)
        has_gaze_label = np.zeros((n,), dtype=np.bool_)

        for i, sample in enumerate(plan.samples):
            state_idx = int(sample["robot_state_sample_index"])
            robot_state = plan.robot_states_by_index[state_idx]
            timestamps[i] = sample_time(sample, i)
            image_timestamps[i] = float(
                sample.get("videoFrames", {}).get("end", {}).get(
                    "capturedPerfCounterSeconds", timestamps[i]
                )
            )
            robot_timestamps[i] = float(
                sample.get("robot_state_pc_perf_counter_seconds", robot_state.get("pc_perf_counter_seconds", timestamps[i]))
            )
            gaze_timestamps[i] = float(sample.get("quest_pc_receive_perf_counter_seconds", timestamps[i]))
            tcp[i] = pose_row_to_9d(robot_state, pose_frame)
            gripper[i, 0] = gripper_width_at(
                plan.gripper_times,
                plan.gripper_widths,
                float(sample.get("pc_perf_counter_seconds", timestamps[i])),
            )
            point_world = sample.get("_gaze_world_pc")
            if point_world is not None:
                gaze_world_pc[i] = np.asarray(point_world, dtype=np.float32)
            gaze_3d_source[i] = int(sample.get("_gaze_3d_source", GAZE_3D_SOURCE_MISSING))
            projection, projection_status = project_world_point_with_status(
                point_world,
                robot_state.get("T_world_end_camera"),
                plan.camera,
            )
            eye_image_timestamps[i] = float(
                sample.get("videoFrames", {}).get(eye_role, {}).get(
                    "capturedPerfCounterSeconds", timestamps[i]
                )
            )
            gaze_projection_status[i] = projection_status
            if projection_status in (
                GAZE_PROJECTION_VALID,
                GAZE_PROJECTION_OUT_OF_FRAME,
            ) and projection is not None:
                gaze_xy[i] = remap_normalized_gaze_xy(
                    projection,
                    source_size=(int(plan.camera["height"]), int(plan.camera["width"])),
                    target_size=image_size,
                    image_resize_mode=image_resize_mode,
                )
                has_gaze_condition[i] = True
                has_gaze_label[i] = projection_status == GAZE_PROJECTION_VALID

        state = np.concatenate([tcp, gripper], axis=-1).astype(np.float32)
        data["timestamp"][start:end] = timestamps
        if output_format == "gaze-wam":
            data["image_timestamp"][start:end] = image_timestamps
            if include_eye_camera:
                data["camera1_image_timestamp"][start:end] = eye_image_timestamps
            data["robot_state_timestamp"][start:end] = robot_timestamps
            data["action_timestamp"][start:end] = robot_timestamps
            data["gaze_timestamp"][start:end] = gaze_timestamps
            data["tcp_pose_abs"][start:end] = tcp
            data["gripper_width"][start:end] = gripper[:, 0]
            data["action_abs_tcp"][start:end] = state
            data["gaze_xy"][start:end] = gaze_xy
            data["gaze_world_pc"][start:end] = gaze_world_pc
            data["gaze_3d_source"][start:end] = gaze_3d_source
            data["gaze_projection_status"][start:end] = gaze_projection_status
            data["has_gaze_condition"][start:end] = has_gaze_condition
            data["has_gaze_label"][start:end] = has_gaze_label
            data["has_heatmap_image"][start:end] = False
        else:
            action = state.copy()
            if n > 1:
                action[:-1] = state[1:]
                action[-1] = action[-2]
            data["left_robot_tcp_pose"][start:end] = tcp
            data["left_robot_gripper_width"][start:end] = gripper
            data["target"][start:end] = state
            data["action"][start:end] = action
        segment_end_offsets = plan.segment_end_offsets or [n]
        episode_ends.extend(start + offset for offset in segment_end_offsets)

    meta.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64), chunks=(10000,), dtype="int64")


def build_frame_requests(plan: EpisodePlan, role: str) -> List[FrameRequest]:
    requests = []
    for local_idx, sample in enumerate(plan.samples):
        frame_index = int(sample["videos"][role]["frameIndex"])
        requests.append(FrameRequest(
            frame_index=frame_index,
            output_index=plan.start_output_index + local_idx,
        ))
    return sorted(requests, key=lambda r: r.frame_index)


def right_bottom_crop_and_resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Crop the bottom-right square and resize it to (H, W)."""
    h, w = image.shape[:2]
    side = min(h, w)
    y_start = h - side
    x_start = w - side
    cropped = image[y_start:y_start + side, x_start:x_start + side]
    target_h, target_w = target_size
    return cv2.resize(cropped, (target_w, target_h))


def center_crop_and_resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    crop: bool,
) -> np.ndarray:
    target_h, target_w = target_size
    if crop:
        height, width = image.shape[:2]
        target_ratio = float(target_w) / float(target_h)
        source_ratio = float(width) / float(height)
        if source_ratio > target_ratio:
            crop_width = max(1, int(round(height * target_ratio)))
            x0 = (width - crop_width) // 2
            image = image[:, x0 : x0 + crop_width]
        elif source_ratio < target_ratio:
            crop_height = max(1, int(round(width / target_ratio)))
            y0 = (height - crop_height) // 2
            image = image[y0 : y0 + crop_height, :]
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    image_resize_mode: str,
) -> np.ndarray:
    target_h, target_w = target_size
    if image_resize_mode == "stretch":
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if image_resize_mode != "letterbox":
        raise ValueError(f"Unsupported image resize mode: {image_resize_mode!r}")

    geometry = letterbox_geometry(image.shape[:2], target_size)
    resized_h, resized_w = geometry["resized_size"]
    pad_left, pad_top, _, _ = geometry["padding_ltrb"]
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    output_shape = (target_h, target_w) + image.shape[2:]
    output = np.zeros(output_shape, dtype=image.dtype)
    output[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    return output


def flush_frame_batch(dataset, pending: List[Tuple[int, np.ndarray]]) -> None:
    if not pending:
        return
    pending.sort(key=lambda item: item[0])
    run_start = 0
    while run_start < len(pending):
        run_end = run_start + 1
        while (
            run_end < len(pending)
            and pending[run_end][0] == pending[run_end - 1][0] + 1
        ):
            run_end += 1
        output_start = pending[run_start][0]
        output_end = pending[run_end - 1][0] + 1
        dataset[output_start:output_end] = np.stack(
            [image for _, image in pending[run_start:run_end]], axis=0
        )
        run_start = run_end
    pending.clear()


def write_video_role(
    dataset,
    video_path: Path,
    requests: List[FrameRequest],
    image_size: Tuple[int, int],
    crop: bool,
    role_name: str,
    image_resize_mode: str = "stretch",
    crop_anchor: str = "center",
    allow_trailing_fill: bool = True,
):
    if not requests:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    request_idx = 0
    next_frame = requests[request_idx].frame_index
    current_frame = 0
    last_image = None
    missing = 0
    pending: List[Tuple[int, np.ndarray]] = []
    batch_size = 128

    while request_idx < len(requests):
        ok, frame = cap.read()
        if not ok:
            break

        if current_frame == next_frame:
            # OpenCV decodes video frames as BGR. Convert to canonical RGB
            # before writing either camera stream into the training Zarr.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_anchor == "right_bottom":
                image = right_bottom_crop_and_resize_image(
                    frame, target_size=image_size
                ).astype(np.uint8)
            elif crop:
                image = center_crop_and_resize_image(
                    frame, target_size=image_size, crop=crop
                ).astype(np.uint8)
            else:
                image = resize_image(
                    frame,
                    target_size=image_size,
                    image_resize_mode=image_resize_mode,
                ).astype(np.uint8)
            last_image = image
            while request_idx < len(requests) and requests[request_idx].frame_index == current_frame:
                pending.append((requests[request_idx].output_index, image))
                request_idx += 1
                if len(pending) >= batch_size:
                    flush_frame_batch(dataset, pending)
            if request_idx < len(requests):
                next_frame = requests[request_idx].frame_index
        current_frame += 1

    cap.release()
    flush_frame_batch(dataset, pending)

    while request_idx < len(requests):
        if last_image is None:
            raise RuntimeError(f"{role_name}: video ended before first requested frame in {video_path}")
        if not allow_trailing_fill:
            missing_requests = len(requests) - request_idx
            raise RuntimeError(
                f"{role_name}: video ended before {missing_requests} requested frame(s) "
                f"in {video_path}"
            )
        pending.append((requests[request_idx].output_index, last_image))
        missing += 1
        request_idx += 1
    flush_frame_batch(dataset, pending)

    if missing:
        logger.warning(f"{role_name}: filled {missing} missing trailing frames with last decoded frame")


def write_images(
    data,
    plans: List[EpisodePlan],
    wrist_role: str,
    eye_role: str,
    image_size: Tuple[int, int],
    output_format: str,
    image_resize_mode: str,
    include_eye_camera: bool = False,
):
    for plan in plans:
        wrist_video = resolve_video_path(plan.episode_dir, wrist_role, plan.samples)
        eye_video = resolve_video_path(plan.episode_dir, eye_role, plan.samples)
        wrist_key = "camera0_rgb" if output_format == "gaze-wam" else "left_wrist_img"
        logger.info(f"{plan.record_id}: writing {wrist_role} -> {wrist_key}")
        write_video_role(
            dataset=data[wrist_key],
            video_path=wrist_video,
            requests=build_frame_requests(plan, wrist_role),
            image_size=image_size,
            crop=False,
            role_name=f"{plan.record_id}/{wrist_role}",
            image_resize_mode=image_resize_mode,
            crop_anchor="center",
            allow_trailing_fill=output_format != "gaze-wam",
        )
        if output_format == "gaze-wam" and not include_eye_camera:
            continue
        eye_key = "camera1_rgb" if output_format == "gaze-wam" else "left_eye_img"
        logger.info(f"{plan.record_id}: writing {eye_role} -> {eye_key}")
        write_video_role(
            dataset=data[eye_key],
            video_path=eye_video,
            requests=build_frame_requests(plan, eye_role),
            image_size=image_size,
            crop=False,
            role_name=f"{plan.record_id}/{eye_role}",
            image_resize_mode=image_resize_mode,
        )


def write_selection_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_summary(
    output_path: Path,
    plans: List[EpisodePlan],
    args,
    selection_report: dict,
    selection_report_path: Path,
):
    summary = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_zarr": str(output_path.resolve()),
        "wrist_role": args.wrist_role,
        "eye_role": args.eye_role,
        "include_eye_camera": bool(getattr(args, "include_eye_camera", False)),
        "camera_roles": (
            {"camera0_rgb": args.wrist_role, "camera1_rgb": args.eye_role}
            if args.output_format == "gaze-wam"
            and bool(getattr(args, "include_eye_camera", False))
            else {"camera0_rgb": args.wrist_role}
            if args.output_format == "gaze-wam"
            else {"left_wrist_img": args.wrist_role, "left_eye_img": args.eye_role}
        ),
        "pose_frame": args.pose_frame,
        "image_size": args.image_size,
        "image_resize_mode": args.image_resize_mode,
        "output_format": args.output_format,
        "record_ids_file": str(Path(args.record_ids_file).resolve()) if args.record_ids_file else None,
        "selection_report": str(selection_report_path.resolve()),
        "candidate_recordings": selection_report["candidate_count"],
        "accepted_recordings": selection_report["accepted_count"],
        "excluded_recordings": selection_report["excluded_count"],
        "action_semantics": (
            "absolute executed TCP pose plus gripper at the current aligned row"
            if args.output_format == "gaze-wam"
            else "legacy one-step-shifted absolute state"
        ),
        "gaze_semantics": (
            "Quest ray depth causal-median filtered, internal missing 3D gaze linearly "
            "interpolated in pc_world, then projected into the full end-camera image"
            if args.output_format == "gaze-wam"
            else None
        ),
        "episodes": [
            {
                "record_id": plan.record_id,
                "frames": plan.end_output_index - plan.start_output_index,
                "start": plan.start_output_index,
                "end": plan.end_output_index,
                "segment_end_offsets": plan.segment_end_offsets or [
                    plan.end_output_index - plan.start_output_index
                ],
                "gaze_stats": plan.gaze_stats,
            }
            for plan in plans
        ],
        "training_episodes": sum(len(plan.segment_end_offsets or [1]) for plan in plans),
        "total_frames": plans[-1].end_output_index,
    }
    with (output_path.parent / "pc_recordings_to_zarr_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_image_size(value: str) -> Tuple[int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 2 or parts[0] <= 0 or parts[1] <= 0:
        raise argparse.ArgumentTypeError("image size must be H,W, for example 224,224")
    return parts[0], parts[1]


def convert_pc_recordings_to_zarr(args):
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    if args.image_resize_mode is None:
        args.image_resize_mode = "letterbox" if args.output_format == "gaze-wam" else "stretch"
    include_eye_camera = bool(getattr(args, "include_eye_camera", False))
    if args.output_format == "gaze-wam" and args.wrist_role != "end":
        raise ValueError(
            "The canonical Gaze-WAM converter currently projects gaze only into the "
            "calibrated end camera; use --wrist-role end."
        )
    record_ids = load_record_ids(Path(args.record_ids_file).resolve()) if args.record_ids_file else None
    selection_report_path = (
        Path(args.selection_report).resolve()
        if getattr(args, "selection_report", None)
        else output_path.with_suffix(".selection.json")
    )
    selection_report = {
        "schema_version": "collector_gaze_wam_batch_selection_v1",
        "input_dir": str(input_dir),
        "output_zarr": str(output_path),
        "record_ids_file": str(Path(args.record_ids_file).resolve()) if args.record_ids_file else None,
        "policy": {
            "selection_unit": "whole_physical_recording",
            "allow_internal_segment_extraction": False,
            "max_endpoint_trim_seconds_per_stream_end": float(args.max_endpoint_trim_seconds),
            "max_image_age_seconds": args.max_image_age_seconds,
            "max_gaze_age_seconds": args.max_gaze_age_seconds,
            "max_sample_gap_seconds": args.max_sample_gap_seconds,
            "missing_gaze_excludes_recording": False,
            "trim_quest_stale_endpoints": True,
            "max_consecutive_reuse_for_selection": getattr(args, "max_consecutive_reuse", 5),
            "max_fallback_reuse_ratio": getattr(args, "max_fallback_reuse_ratio", 0.20),
            "gaze_median_window": int(args.gaze_median_window),
            "image_resize_mode": args.image_resize_mode,
            "include_eye_camera": include_eye_camera,
            "eye_role": args.eye_role if include_eye_camera else None,
        },
    }
    plans = build_plans(
        input_dir=input_dir,
        wrist_role=args.wrist_role,
        eye_role=args.eye_role,
        pose_frame=args.pose_frame,
        default_gripper_width=args.default_gripper_width,
        max_image_age_seconds=args.max_image_age_seconds,
        limit_records=args.limit_records,
        record_ids=record_ids,
        require_camera_intrinsics=args.output_format == "gaze-wam",
        prepare_gaze=args.output_format == "gaze-wam",
        gaze_median_window=args.gaze_median_window,
        max_gaze_age_seconds=args.max_gaze_age_seconds,
        max_sample_gap_seconds=args.max_sample_gap_seconds,
        max_endpoint_trim_seconds=args.max_endpoint_trim_seconds,
        max_consecutive_reuse=getattr(args, "max_consecutive_reuse", 5),
        max_fallback_reuse_ratio=getattr(args, "max_fallback_reuse_ratio", 0.20),
        selection_report=selection_report,
    )
    selection_report["candidate_count"] = len(selection_report["candidate_record_ids"])
    selection_report["accepted_count"] = len(selection_report["accepted"])
    selection_report["excluded_count"] = len(selection_report["excluded"])
    write_selection_report(selection_report_path, selection_report)
    logger.info(f"Selection report: {selection_report_path}")
    if not plans:
        raise RuntimeError(f"No valid record_* episodes found under {input_dir}")
    total_frames = plans[-1].end_output_index
    logger.info(f"Total episodes: {len(plans)}")
    logger.info(f"Total frames: {total_frames}")

    if args.dry_run:
        return None

    root, data, meta = create_output_zarr(
        output_path=output_path,
        total_frames=total_frames,
        image_size=args.image_size,
        overwrite=not args.no_overwrite,
        output_format=args.output_format,
        include_eye_camera=include_eye_camera,
    )
    write_lowdim_arrays(
        data,
        meta,
        plans,
        pose_frame=args.pose_frame,
        output_format=args.output_format,
        image_size=args.image_size,
        image_resize_mode=args.image_resize_mode,
        eye_role=args.eye_role,
        include_eye_camera=include_eye_camera,
    )
    write_images(
        data,
        plans,
        wrist_role=args.wrist_role,
        eye_role=args.eye_role,
        image_size=args.image_size,
        output_format=args.output_format,
        image_resize_mode=args.image_resize_mode,
        include_eye_camera=include_eye_camera,
    )
    if args.output_format == "gaze-wam":
        meta.attrs.update({
            "dataset_type": "robot",
            "canonical_schema": "gaze_wam_robot_v1",
            "camera_role": args.wrist_role,
            "camera_key": "camera0_rgb",
            "camera_roles": {
                "camera0_rgb": args.wrist_role,
                **({"camera1_rgb": args.eye_role} if include_eye_camera else {}),
            },
            "camera_keys": (
                ["camera0_rgb", "camera1_rgb"]
                if include_eye_camera
                else ["camera0_rgb"]
            ),
            "image_size": list(args.image_size),
            "image_resize_mode": args.image_resize_mode,
            "gaze_is_normalized": True,
            "gaze_projection_source": "causal_median_ray_depth_then_internal_linear_interpolation_pc_world",
            "gaze_projection_camera_role": args.wrist_role,
            "gaze_projection_resize_transform": (
                "source_normalized_to_letterboxed_output"
                if args.image_resize_mode == "letterbox"
                else "source_normalized_equals_stretched_output_normalized"
            ),
            "image_padding_rgb": [0, 0, 0] if args.image_resize_mode == "letterbox" else None,
            "source_camera_size": [
                int(plans[0].camera["height"]),
                int(plans[0].camera["width"]),
            ],
            "source_camera_sizes": {
                "camera0_rgb": [
                    int(plans[0].camera["height"]),
                    int(plans[0].camera["width"]),
                ],
                **(
                    {"camera1_rgb": list(camera_source_size(plans[0].episode_dir, args.eye_role))}
                    if include_eye_camera
                    else {}
                ),
            },
            "image_resize_geometry": (
                letterbox_geometry(
                    (int(plans[0].camera["height"]), int(plans[0].camera["width"])),
                    args.image_size,
                )
                if args.image_resize_mode == "letterbox"
                else None
            ),
            "image_resize_geometry_by_camera": {
                "camera0_rgb": (
                    letterbox_geometry(
                        (int(plans[0].camera["height"]), int(plans[0].camera["width"])),
                        args.image_size,
                    )
                    if args.image_resize_mode == "letterbox"
                    else None
                ),
                **(
                    {
                        "camera1_rgb": letterbox_geometry(
                            camera_source_size(plans[0].episode_dir, args.eye_role),
                            args.image_size,
                        )
                    }
                    if include_eye_camera and args.image_resize_mode == "letterbox"
                    else {}
                ),
            },
            "gaze_median_filter": {
                "kind": "causal_median_ray_depth",
                "window": int(args.gaze_median_window),
                "replay_aligned": True,
            },
            "gaze_interpolation": {
                "kind": "linear_pc_world",
                "scope": "internal_missing_rows_within_episode_only",
                "extrapolate_edges": False,
            },
            "gaze_3d_source_values": GAZE_3D_SOURCE_NAMES,
            "gaze_projection_status_values": GAZE_PROJECTION_STATUS_NAMES,
            "action_representation": "absolute",
            "action_semantics": "executed_tcp_pose_plus_gripper_at_current_aligned_row",
            "timestamp_key": "timestamp",
            "timestamp_stream_keys": {
                "image_timestamp": {"output_key": "image_timestamp"},
                **(
                    {
                        "camera1_image_timestamp": {
                            "output_key": "camera1_image_timestamp"
                        }
                    }
                    if include_eye_camera
                    else {}
                ),
                "robot_state_timestamp": {"output_key": "robot_state_timestamp"},
                "action_timestamp": {"output_key": "action_timestamp"},
                "gaze_timestamp": {"output_key": "gaze_timestamp"},
            },
            "presence_mask_keys": [
                "has_gaze_condition",
                "has_gaze_label",
                "has_heatmap_image",
            ],
        })
    write_summary(output_path, plans, args, selection_report, selection_report_path)
    logger.info(f"Saved zarr: {output_path}")
    logger.info(f"Summary: {output_path.parent / 'pc_recordings_to_zarr_summary.json'}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert pc_recordings to single-arm replay_buffer.zarr")
    parser.add_argument(
        "--input-dir",
        default="/mnt/workspace/zhengkai/pc_recordings_test/pc_recordings",
        help="Directory containing record_* folders, or its parent containing pc_recordings/",
    )
    parser.add_argument(
        "--output-format",
        choices=("umi", "gaze-wam"),
        default="umi",
        help="Write the legacy UMI schema or the canonical Gaze-WAM robot schema.",
    )
    parser.add_argument(
        "--record-ids-file",
        default=None,
        help="Optional newline-separated record allowlist. Missing allowlisted directories fail.",
    )
    parser.add_argument(
        "--selection-report",
        default=None,
        help=(
            "Write structured candidate/accepted/excluded results here. Defaults to "
            "<output>.selection.json and is written for dry runs too."
        ),
    )
    parser.add_argument(
        "--output",
        default="/mnt/workspace/zhengkai/pc_recordings_test/zarr/replay_buffer.zarr",
        help="Output replay_buffer.zarr path",
    )
    parser.add_argument("--wrist-role", default="end", help="Camera role to write as left_wrist_img")
    parser.add_argument("--eye-role", default="third", help="Camera role to write as left_eye_img")
    parser.add_argument(
        "--include-eye-camera",
        action="store_true",
        help=(
            "For gaze-wam output, also write --eye-role as data/camera1_rgb. "
            "The stream uses the same image size and resize mode as camera0_rgb."
        ),
    )
    parser.add_argument(
        "--pose-frame",
        default="T_base_tool_tcp",
        choices=["T_base_tool_tcp", "T_world_tool_tcp", "T_display_tool_tcp"],
        help="Robot TCP pose frame to convert into left_robot_tcp_pose",
    )
    parser.add_argument("--image-size", type=parse_image_size, default=(224, 224), help="Output image size as H,W")
    parser.add_argument(
        "--image-resize-mode",
        choices=IMAGE_RESIZE_MODES,
        default=None,
        help=(
            "Resize geometry for output images and normalized gaze. Defaults to "
            "letterbox for gaze-wam and stretch for legacy UMI output."
        ),
    )
    parser.add_argument("--default-gripper-width", type=float, default=0.08)
    parser.add_argument(
        "--max-image-age-seconds",
        type=float,
        default=None,
        help="Optional filter for stale sample/video alignments",
    )
    parser.add_argument(
        "--max-gaze-age-seconds",
        type=float,
        default=0.060,
        help=(
            "For gaze-wam output, treat Quest gaze older than this as a missing 3D "
            "sample eligible for internal interpolation. The action row is retained. "
            "Default: 0.060."
        ),
    )
    parser.add_argument(
        "--max-consecutive-reuse",
        type=int,
        default=5,
        help=(
            "Documented quality threshold for consecutive Quest-source reuse. "
            "Short internal runs are interpolated; endpoint stale runs are trimmed. Default: 5."
        ),
    )
    parser.add_argument(
        "--max-fallback-reuse-ratio",
        type=float,
        default=0.20,
        help="Maximum cumulative Quest-source reuse ratio after endpoint trim. Default: 0.20.",
    )
    parser.add_argument(
        "--gaze-median-window",
        type=int,
        default=7,
        help=(
            "Causal median window for gaze ray depth, matching Collector replay. "
            "Default: 7."
        ),
    )
    parser.add_argument(
        "--max-sample-gap-seconds",
        type=float,
        default=0.060,
        help=(
            "For gaze-wam output, reject a physical recording if the retained common "
            "window contains a larger aligned or camera gap. Default: 0.060."
        ),
    )
    parser.add_argument(
        "--max-endpoint-trim-seconds",
        type=float,
        default=1.0,
        help=(
            "Allow at most this much trimming from each start/end of aligned and "
            "camera timelines to form one continuous episode. Internal segments are "
            "never extracted. Default: 1.0."
        ),
    )
    parser.add_argument("--limit-records", type=int, default=None, help="Convert only the first N records")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect records; do not write zarr")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output already exists")
    args = parser.parse_args()

    convert_pc_recordings_to_zarr(args)


if __name__ == "__main__":
    main()
