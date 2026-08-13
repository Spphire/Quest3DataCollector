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
  data/gaze_xy                    [N, 2], normalized to the end-camera image
  data/action_abs_tcp             [N, 10]
  data/tcp_pose_abs               [N, 9]
  data/gripper_width              [N]
  data/{timestamp,image_timestamp,robot_state_timestamp,gaze_timestamp}
  data/{has_gaze_label,has_heatmap_image}
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
    dropped_stale_gaze: int = 0
    dropped_invalid_gaze_projection: int = 0
    dropped_short_segment_samples: int = 0


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


def project_world_point(point_world: object, world_camera_pose: object, camera: dict) -> Optional[np.ndarray]:
    try:
        point = np.asarray(point_world, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None
        world_camera = pose_payload_to_matrix(world_camera_pose)
        point_camera = np.linalg.inv(world_camera) @ np.asarray([*point, 1.0], dtype=np.float64)
        x, y, z = point_camera[:3]
        if not np.all(np.isfinite(point_camera)) or z <= 1e-9:
            return None
        u = float(camera["fx"]) * float(x) / float(z) + float(camera["cx"])
        v = float(camera["fy"]) * float(y) / float(z) + float(camera["cy"])
        width = float(camera["width"])
        height = float(camera["height"])
        if not all(math.isfinite(value) for value in (u, v, width, height)):
            return None
        if width <= 0 or height <= 0 or not (0.0 <= u < width and 0.0 <= v < height):
            return None
        return np.asarray([u / width, v / height], dtype=np.float32)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


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
) -> List[dict]:
    valid = []
    for sample in iter_jsonl(samples_path):
        if not sample.get("ok", False):
            continue
        robot_state_index = sample.get("robot_state_sample_index")
        if robot_state_index is None or int(robot_state_index) not in robot_states_by_index:
            continue
        videos = sample.get("videos", {})
        video_frames = sample.get("videoFrames", {})
        if any(role not in videos or videos[role].get("frameIndex") is None for role in required_roles):
            continue
        if max_image_age_seconds is not None:
            too_old = False
            for role in required_roles:
                age = video_frames.get(role, {}).get("ageSeconds")
                if age is not None and float(age) > max_image_age_seconds:
                    too_old = True
                    break
            if too_old:
                continue
        valid.append(sample)
    return valid


def split_samples_at_time_gaps(
    samples: List[dict],
    max_gap_seconds: Optional[float],
    min_segment_frames: int,
) -> Tuple[List[dict], List[int], int]:
    if not samples:
        return [], [], 0
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be at least 1")

    segments = []
    segment_start = 0
    if max_gap_seconds is not None:
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        for index in range(1, len(samples)):
            previous_sample = samples[index - 1]
            current_sample = samples[index]
            timestamp_pairs = (
                (
                    previous_sample.get("pc_perf_counter_seconds"),
                    current_sample.get("pc_perf_counter_seconds"),
                ),
                (
                    previous_sample.get("videoFrames", {}).get("end", {}).get("capturedPerfCounterSeconds"),
                    current_sample.get("videoFrames", {}).get("end", {}).get("capturedPerfCounterSeconds"),
                ),
                (
                    previous_sample.get("robot_state_pc_perf_counter_seconds"),
                    current_sample.get("robot_state_pc_perf_counter_seconds"),
                ),
                (
                    previous_sample.get("quest_pc_receive_perf_counter_seconds"),
                    current_sample.get("quest_pc_receive_perf_counter_seconds"),
                ),
            )
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
                segments.append(samples[segment_start:index])
                segment_start = index
    segments.append(samples[segment_start:])

    kept_samples = []
    segment_end_offsets = []
    dropped_short_segment_samples = 0
    for segment in segments:
        if len(segment) < min_segment_frames:
            dropped_short_segment_samples += len(segment)
            continue
        kept_samples.extend(segment)
        segment_end_offsets.append(len(kept_samples))
    return kept_samples, segment_end_offsets, dropped_short_segment_samples


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
    require_valid_gaze: bool,
    max_gaze_age_seconds: Optional[float],
    max_sample_gap_seconds: Optional[float],
    min_segment_frames: int,
) -> List[EpisodePlan]:
    plans = []
    total = 0
    record_dirs = find_record_dirs(input_dir, record_ids=record_ids)
    if limit_records is not None:
        record_dirs = record_dirs[:limit_records]

    for record_dir in record_dirs:
        episode_dir = record_dir / "robot_realsense"
        samples_path = episode_dir / "samples.jsonl"
        robot_states_path = episode_dir / "robot_states.jsonl"
        gripper_path = episode_dir / "gripper_commands.jsonl"
        cameras_path = episode_dir / "cameras.json"
        if not samples_path.exists() or not robot_states_path.exists():
            logger.warning(f"Skip {record_dir.name}: missing samples.jsonl or robot_states.jsonl")
            continue
        if require_camera_intrinsics and not cameras_path.exists():
            logger.warning(f"Skip {record_dir.name}: missing cameras.json")
            continue
        cameras = json.loads(cameras_path.read_text(encoding="utf-8")) if cameras_path.exists() else {}
        camera = cameras.get(wrist_role)
        if require_camera_intrinsics and not isinstance(camera, dict):
            logger.warning(f"Skip {record_dir.name}: missing {wrist_role} camera intrinsics")
            continue

        robot_states = load_robot_states(robot_states_path)
        valid_samples = collect_valid_samples(
            samples_path=samples_path,
            robot_states_by_index=robot_states,
            required_roles=(wrist_role, eye_role),
            max_image_age_seconds=max_image_age_seconds,
        )
        dropped_stale_gaze = 0
        dropped_invalid_gaze_projection = 0
        if require_valid_gaze:
            projected_samples = []
            for sample in valid_samples:
                aligned_time = sample.get("pc_perf_counter_seconds")
                gaze_time = sample.get("quest_pc_receive_perf_counter_seconds")
                if (
                    max_gaze_age_seconds is not None
                    and isinstance(aligned_time, (int, float))
                    and isinstance(gaze_time, (int, float))
                    and float(aligned_time) - float(gaze_time) > max_gaze_age_seconds
                ):
                    dropped_stale_gaze += 1
                    continue
                state_index = sample.get("robot_state_sample_index")
                robot_state = robot_states.get(int(state_index)) if state_index is not None else None
                projection = project_world_point(
                    sample.get("quest_gaze3d_pc_world"),
                    robot_state.get("T_world_end_camera") if robot_state else None,
                    camera,
                )
                if projection is None:
                    dropped_invalid_gaze_projection += 1
                    continue
                projected_samples.append(sample)
            valid_samples = projected_samples
        valid_samples, segment_end_offsets, dropped_short_segment_samples = split_samples_at_time_gaps(
            samples=valid_samples,
            max_gap_seconds=max_sample_gap_seconds if require_valid_gaze else None,
            min_segment_frames=min_segment_frames if require_valid_gaze else 1,
        )
        if len(valid_samples) < 2:
            logger.warning(f"Skip {record_dir.name}: only {len(valid_samples)} valid samples")
            continue

        try:
            first_state = robot_states[int(valid_samples[0]["robot_state_sample_index"])]
            pose_row_to_9d(first_state, pose_frame)
        except Exception as exc:
            logger.warning(f"Skip {record_dir.name}: cannot read {pose_frame}: {exc}")
            continue

        wrist_video = resolve_video_path(episode_dir, wrist_role, valid_samples)
        eye_video = resolve_video_path(episode_dir, eye_role, valid_samples)
        if wrist_video is None or eye_video is None:
            logger.warning(f"Skip {record_dir.name}: missing required videos")
            continue

        gripper_times, gripper_widths = load_gripper_timeline(gripper_path, default_gripper_width)
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
            dropped_stale_gaze=dropped_stale_gaze,
            dropped_invalid_gaze_projection=dropped_invalid_gaze_projection,
            dropped_short_segment_samples=dropped_short_segment_samples,
        ))
        logger.info(
            f"{record_dir.name}: {len(valid_samples)} valid samples in "
            f"{len(segment_end_offsets)} segment(s) "
            f"(dropped stale gaze={dropped_stale_gaze}, "
            f"invalid projection={dropped_invalid_gaze_projection}, "
            f"short segment samples={dropped_short_segment_samples})"
        )

    if not plans:
        raise RuntimeError(f"No valid record_* episodes found under {input_dir}")
    return plans


def create_output_zarr(
    output_path: Path,
    total_frames: int,
    image_size: Tuple[int, int],
    overwrite: bool,
    output_format: str,
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
        data.create_dataset("has_gaze_label", shape=(total_frames,), chunks=(10000,), dtype="bool", compressor=compressor)
        data.create_dataset("has_heatmap_image", shape=(total_frames,), chunks=(10000,), dtype="bool", compressor=compressor)
        data.create_dataset(
            "camera0_rgb",
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


def write_lowdim_arrays(data, meta, plans: List[EpisodePlan], pose_frame: str, output_format: str):
    episode_ends = []
    for plan in plans:
        start = plan.start_output_index
        end = plan.end_output_index
        n = end - start
        timestamps = np.zeros((n,), dtype=np.float64)
        tcp = np.zeros((n, 9), dtype=np.float32)
        gripper = np.zeros((n, 1), dtype=np.float32)
        image_timestamps = np.zeros((n,), dtype=np.float64)
        robot_timestamps = np.zeros((n,), dtype=np.float64)
        gaze_timestamps = np.zeros((n,), dtype=np.float64)
        gaze_xy = np.zeros((n, 2), dtype=np.float32)
        has_gaze = np.zeros((n,), dtype=np.bool_)

        for i, sample in enumerate(plan.samples):
            state_idx = int(sample["robot_state_sample_index"])
            robot_state = plan.robot_states_by_index[state_idx]
            timestamps[i] = float(sample.get("pc_perf_counter_seconds", i / 30.0))
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
            projection = project_world_point(
                sample.get("quest_gaze3d_pc_world"),
                robot_state.get("T_world_end_camera"),
                plan.camera,
            )
            if projection is not None:
                gaze_xy[i] = projection
                has_gaze[i] = True

        state = np.concatenate([tcp, gripper], axis=-1).astype(np.float32)
        data["timestamp"][start:end] = timestamps
        if output_format == "gaze-wam":
            data["image_timestamp"][start:end] = image_timestamps
            data["robot_state_timestamp"][start:end] = robot_timestamps
            data["action_timestamp"][start:end] = robot_timestamps
            data["gaze_timestamp"][start:end] = gaze_timestamps
            data["tcp_pose_abs"][start:end] = tcp
            data["gripper_width"][start:end] = gripper[:, 0]
            data["action_abs_tcp"][start:end] = state
            data["gaze_xy"][start:end] = gaze_xy
            data["has_gaze_label"][start:end] = has_gaze
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


def write_video_role(
    dataset,
    video_path: Path,
    requests: List[FrameRequest],
    image_size: Tuple[int, int],
    crop: bool,
    role_name: str,
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
            else:
                image = center_crop_and_resize_image(
                    frame, target_size=image_size, crop=crop
                ).astype(np.uint8)
            last_image = image
            while request_idx < len(requests) and requests[request_idx].frame_index == current_frame:
                dataset[requests[request_idx].output_index] = image
                request_idx += 1
            if request_idx < len(requests):
                next_frame = requests[request_idx].frame_index
        current_frame += 1

    cap.release()

    while request_idx < len(requests):
        if last_image is None:
            raise RuntimeError(f"{role_name}: video ended before first requested frame in {video_path}")
        if not allow_trailing_fill:
            missing_requests = len(requests) - request_idx
            raise RuntimeError(
                f"{role_name}: video ended before {missing_requests} requested frame(s) "
                f"in {video_path}"
            )
        dataset[requests[request_idx].output_index] = last_image
        missing += 1
        request_idx += 1

    if missing:
        logger.warning(f"{role_name}: filled {missing} missing trailing frames with last decoded frame")


def write_images(
    data,
    plans: List[EpisodePlan],
    wrist_role: str,
    eye_role: str,
    image_size: Tuple[int, int],
    output_format: str,
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
            crop_anchor="center",
            allow_trailing_fill=output_format != "gaze-wam",
        )
        if output_format == "gaze-wam":
            continue
        logger.info(f"{plan.record_id}: writing {eye_role} -> left_eye_img")
        write_video_role(
            dataset=data["left_eye_img"],
            video_path=eye_video,
            requests=build_frame_requests(plan, eye_role),
            image_size=image_size,
            crop=False,
            role_name=f"{plan.record_id}/{eye_role}",
        )


def write_summary(output_path: Path, plans: List[EpisodePlan], args):
    summary = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_zarr": str(output_path.resolve()),
        "wrist_role": args.wrist_role,
        "eye_role": args.eye_role,
        "pose_frame": args.pose_frame,
        "image_size": args.image_size,
        "output_format": args.output_format,
        "record_ids_file": str(Path(args.record_ids_file).resolve()) if args.record_ids_file else None,
        "action_semantics": (
            "absolute executed TCP pose plus gripper at the current aligned row"
            if args.output_format == "gaze-wam"
            else "legacy one-step-shifted absolute state"
        ),
        "gaze_semantics": (
            "quest_gaze3d_pc_world projected into the full end-camera image"
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
                "dropped_stale_gaze": plan.dropped_stale_gaze,
                "dropped_invalid_gaze_projection": plan.dropped_invalid_gaze_projection,
                "dropped_short_segment_samples": plan.dropped_short_segment_samples,
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
    if args.output_format == "gaze-wam" and args.wrist_role != "end":
        raise ValueError(
            "The canonical Gaze-WAM converter currently projects gaze only into the "
            "calibrated end camera; use --wrist-role end."
        )
    record_ids = load_record_ids(Path(args.record_ids_file).resolve()) if args.record_ids_file else None
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
        require_valid_gaze=args.output_format == "gaze-wam",
        max_gaze_age_seconds=args.max_gaze_age_seconds,
        max_sample_gap_seconds=args.max_sample_gap_seconds,
        min_segment_frames=args.min_segment_frames,
    )
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
    )
    write_lowdim_arrays(data, meta, plans, pose_frame=args.pose_frame, output_format=args.output_format)
    write_images(
        data,
        plans,
        wrist_role=args.wrist_role,
        eye_role=args.eye_role,
        image_size=args.image_size,
        output_format=args.output_format,
    )
    if args.output_format == "gaze-wam":
        meta.attrs.update({
            "dataset_type": "robot",
            "canonical_schema": "gaze_wam_robot_v1",
            "camera_role": args.wrist_role,
            "camera_key": "camera0_rgb",
            "image_size": list(args.image_size),
            "image_resize_mode": "stretch",
            "gaze_is_normalized": True,
            "gaze_projection_source": "quest_gaze3d_pc_world",
            "gaze_projection_camera_role": args.wrist_role,
            "action_representation": "absolute",
            "action_semantics": "executed_tcp_pose_plus_gripper_at_current_aligned_row",
            "timestamp_key": "timestamp",
            "timestamp_stream_keys": {
                "image_timestamp": {"output_key": "image_timestamp"},
                "robot_state_timestamp": {"output_key": "robot_state_timestamp"},
                "action_timestamp": {"output_key": "action_timestamp"},
                "gaze_timestamp": {"output_key": "gaze_timestamp"},
            },
            "presence_mask_keys": ["has_gaze_label", "has_heatmap_image"],
        })
    write_summary(output_path, plans, args)
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
        "--output",
        default="/mnt/workspace/zhengkai/pc_recordings_test/zarr/replay_buffer.zarr",
        help="Output replay_buffer.zarr path",
    )
    parser.add_argument("--wrist-role", default="end", help="Camera role to write as left_wrist_img")
    parser.add_argument("--eye-role", default="third", help="Camera role to write as left_eye_img")
    parser.add_argument(
        "--pose-frame",
        default="T_base_tool_tcp",
        choices=["T_base_tool_tcp", "T_world_tool_tcp", "T_display_tool_tcp"],
        help="Robot TCP pose frame to convert into left_robot_tcp_pose",
    )
    parser.add_argument("--image-size", type=parse_image_size, default=(224, 224), help="Output image size as H,W")
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
            "For gaze-wam output, drop rows whose aligned sample uses a Quest gaze "
            "source older than this many seconds. Default: 0.060."
        ),
    )
    parser.add_argument(
        "--max-sample-gap-seconds",
        type=float,
        default=0.200,
        help=(
            "For gaze-wam output, split a physical recording into separate training "
            "episodes when retained aligned rows are farther apart than this. Default: 0.200."
        ),
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=16,
        help=(
            "For gaze-wam output, discard temporal segments shorter than this many "
            "frames after gaze filtering. Default: 16."
        ),
    )
    parser.add_argument("--limit-records", type=int, default=None, help="Convert only the first N records")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect records; do not write zarr")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output already exists")
    args = parser.parse_args()

    convert_pc_recordings_to_zarr(args)


if __name__ == "__main__":
    main()
