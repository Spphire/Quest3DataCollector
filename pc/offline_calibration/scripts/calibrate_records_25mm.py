from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from quest_coordinate_frames import (
    PC_WORLD_FRAME,
    UNITY_WORLD_FRAME,
    WORLD_FRAME_CONVERSION,
    unity_transform_matrix_to_pc,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
DEFAULT_OUTPUT = ROOT / "outputs" / "record_calibration_25mm"
DEFAULT_RECORDS = [
    "record_20260615_113806",
    "record_20260615_113838",
    "record_20260615_202510",
    "record_20260615_211626",
]
PATTERN_COLS = 11
PATTERN_ROWS = 8
SQUARE_SIZE_M = 0.025
IMAGE_SIZE = (640, 480)
RED_ANCHOR_RADIUS_GRID_SPACING = 2.0
RED_ANCHOR_MIN_PIXELS = 16
RED_ANCHOR_MIN_RATIO = 0.002
RED_ANCHOR_MIN_BEST_SECOND_RATIO = 1.8
DEFAULT_DIVERSE_DETECTION_FRAMES_PER_SIDE = 220
DEFAULT_DIVERSE_FIT_FRAMES_PER_SIDE = 180
DEFAULT_DIVERSE_MIN_FRAMES_PER_SIDE = 28
DEFAULT_DIVERSE_TRANSLATION_SCALE_M = 0.025
DEFAULT_DIVERSE_ROTATION_SCALE_DEG = 3.0


@dataclass
class PoseSeries:
    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation
    slerp: Slerp

    @classmethod
    def from_jsonl(cls, path: Path) -> "PoseSeries":
        rows = read_jsonl(path)
        times = np.asarray([row["unityTimestampSeconds"] for row in rows], dtype=float)
        positions = np.asarray([row["pose"][:3] for row in rows], dtype=float)
        rotations = Rotation.from_quat([[row["pose"][4], row["pose"][5], row["pose"][6], row["pose"][3]] for row in rows])
        return cls(times=times, positions=positions, rotations=rotations, slerp=Slerp(times, rotations))

    def sample(self, query_times: np.ndarray) -> tuple[np.ndarray, Rotation, np.ndarray]:
        query_times = np.asarray(query_times, dtype=float)
        valid = (query_times >= self.times[0]) & (query_times <= self.times[-1])
        tq = query_times[valid]
        positions = np.column_stack([np.interp(tq, self.times, self.positions[:, axis]) for axis in range(3)])
        return positions, self.slerp(tq), valid


@dataclass
class Batch:
    record: str
    side: str
    frame_indices: np.ndarray
    times: np.ndarray
    corners: np.ndarray
    camera_positions: np.ndarray
    camera_rotations: Rotation
    red_anchor_observed_indices: np.ndarray
    red_anchor_best_scores: np.ndarray
    red_anchor_score_ratios: np.ndarray
    red_anchor_target_indices: np.ndarray
    red_anchor_orders: np.ndarray


class CalibrationFailure(RuntimeError):
    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


def diverse_pose_indices(
    positions: np.ndarray,
    rotations: Rotation,
    max_count: int,
    min_count: int,
    translation_scale_m: float,
    rotation_scale_deg: float,
    min_score: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = int(len(positions))
    if count <= 0:
        return np.asarray([], dtype=int), {
            "enabled": True,
            "input_count": 0,
            "selected_count": 0,
            "reason": "empty",
        }
    max_count = int(max(1, max_count))
    min_count = int(max(1, min_count))
    if count <= max_count:
        return np.arange(count, dtype=int), {
            "enabled": True,
            "input_count": count,
            "selected_count": count,
            "reason": "input_below_limit",
        }

    positions = np.asarray(positions, dtype=float).reshape(count, 3)
    trans_scale = max(1e-6, float(translation_scale_m))
    rot_scale = max(1e-6, np.deg2rad(float(rotation_scale_deg)))
    selected: list[int] = [0]
    center = np.median(positions, axis=0)
    selected.append(int(np.argmax(np.linalg.norm(positions - center[None, :], axis=1))))
    selected = sorted(set(selected))

    min_dist = pose_distance_to_set(positions, rotations, selected, trans_scale, rot_scale)
    stop_score = float(min_score)
    while len(selected) < max_count:
        candidate_scores = min_dist.copy()
        candidate_scores[selected] = -np.inf
        candidate = int(np.argmax(candidate_scores))
        score = float(candidate_scores[candidate])
        if len(selected) >= min_count and score < stop_score:
            break
        if not np.isfinite(score):
            break
        selected.append(candidate)
        new_dist = pose_distance_to_set(positions, rotations, [candidate], trans_scale, rot_scale)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[selected] = 0.0

    selected_arr = np.asarray(sorted(selected), dtype=int)
    selected_positions = positions[selected_arr]
    selected_rotations = rotations[selected_arr]
    return selected_arr, {
        "enabled": True,
        "input_count": count,
        "selected_count": int(len(selected_arr)),
        "max_count": max_count,
        "min_count": min_count,
        "min_score": stop_score,
        "translation_scale_m": trans_scale,
        "rotation_scale_deg": float(rotation_scale_deg),
        "selected_fraction": float(len(selected_arr) / count),
        "pose_span": pose_selection_span(selected_positions, selected_rotations),
        "last_best_score": float(np.max(min_dist)) if len(min_dist) else 0.0,
    }


def pose_distance_to_set(
    positions: np.ndarray,
    rotations: Rotation,
    selected: list[int],
    translation_scale_m: float,
    rotation_scale_rad: float,
) -> np.ndarray:
    distances = np.full(len(positions), np.inf, dtype=float)
    for index in selected:
        translation = np.linalg.norm(positions - positions[index][None, :], axis=1) / translation_scale_m
        delta = rotations[index].inv() * rotations
        rotation = delta.magnitude() / rotation_scale_rad
        distances = np.minimum(distances, np.sqrt(translation * translation + rotation * rotation))
    return distances


def pose_selection_span(positions: np.ndarray, rotations: Rotation) -> dict[str, Any]:
    if len(positions) <= 0:
        return {
            "translation_span_m": 0.0,
            "rotation_span_deg": 0.0,
            "axis_span_m": [0.0, 0.0, 0.0],
        }
    max_translation = 0.0
    max_rotation = 0.0
    for i in range(len(positions)):
        if i + 1 >= len(positions):
            break
        translation = np.linalg.norm(positions[i + 1 :] - positions[i][None, :], axis=1)
        if len(translation):
            max_translation = max(max_translation, float(np.max(translation)))
        delta = rotations[i].inv() * rotations[i + 1 :]
        if len(delta):
            max_rotation = max(max_rotation, float(np.rad2deg(np.max(delta.magnitude()))))
    axis_span = np.ptp(positions, axis=0) if len(positions) else np.zeros(3, dtype=float)
    return {
        "translation_span_m": float(max_translation),
        "rotation_span_deg": float(max_rotation),
        "axis_span_m": [float(v) for v in axis_span],
    }


def metadata_pose_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, Rotation, list[int]]:
    positions = []
    quats_xyzw = []
    frame_indices = []
    for row in rows:
        pose = row.get("pose")
        if not isinstance(pose, list) or len(pose) < 7:
            continue
        try:
            positions.append([float(pose[0]), float(pose[1]), float(pose[2])])
            quats_xyzw.append([float(pose[4]), float(pose[5]), float(pose[6]), float(pose[3])])
            frame_indices.append(int(row["frameIndex"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not positions:
        return np.zeros((0, 3), dtype=float), Rotation.identity(0), []
    return np.asarray(positions, dtype=float), Rotation.from_quat(quats_xyzw), frame_indices


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-record Quest/Unity world to a fixed 11x8 checkerboard from recorded "
            "passthrough videos and PassthroughCameraAccess.GetCameraPose trajectories. "
            "The physical checkerboard square size is 25mm."
        )
    )
    parser.add_argument("--records", nargs="*", default=DEFAULT_RECORDS, help="Record directory names under raw/.")
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern-cols", type=int, default=PATTERN_COLS)
    parser.add_argument("--pattern-rows", type=int, default=PATTERN_ROWS)
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE_M)
    parser.add_argument("--max-reproj-px", type=float, default=4.0)
    parser.add_argument("--model", choices=["pinhole", "radial2"], default="pinhole")
    parser.add_argument(
        "--image-y-axis",
        choices=["auto", "down", "up"],
        default="auto",
        help=(
            "Direction of increasing pixel row in the recorded video relative to the passthrough camera local +Y axis. "
            "'down' is the usual top-left image convention; 'up' matches vertically flipped/bottom-left recordings. "
            "Default: auto, fit both and select the lower reprojection error."
        ),
    )
    parser.add_argument("--skip-detection", action="store_true", help="Reuse existing NPZ detections in output-root/detections.")
    parser.add_argument("--lag-min-ms", type=float, default=-120.0)
    parser.add_argument("--lag-max-ms", type=float, default=120.0)
    parser.add_argument("--coarse-step-ms", type=float, default=1000.0 / 30.0)
    parser.add_argument("--fine-window-ms", type=float, default=25.0)
    parser.add_argument("--fine-step-ms", type=float, default=1000.0 / 120.0)
    parser.add_argument("--coarse-stride", type=int, default=8)
    parser.add_argument("--fine-stride", type=int, default=5)
    parser.add_argument("--order-iterations", type=int, default=3)
    parser.add_argument("--order-keep-threshold-px", type=float, default=20.0)
    parser.add_argument("--max-final-nfev", type=int, default=300)
    parser.add_argument(
        "--disable-red-anchor",
        action="store_true",
        help="Disable red-near-corner anchor detection for resolving 180-degree checkerboard ambiguity.",
    )
    parser.add_argument(
        "--red-anchor-target-index",
        type=int,
        default=None,
        help=(
            "Optional physical red-corner target index. Default: auto, choose the smaller index from the detected "
            "180-degree corner pair, e.g. 0 for 0/87 or 10 for 10/77."
        ),
    )
    parser.add_argument(
        "--disable-diverse-frame-selection",
        action="store_true",
        help="Use every recorded frame instead of pose-diverse frame selection.",
    )
    parser.add_argument(
        "--max-diverse-detection-frames-per-side",
        type=int,
        default=DEFAULT_DIVERSE_DETECTION_FRAMES_PER_SIDE,
        help=(
            "Maximum pose-diverse frames to run checkerboard detection on per record/side. "
            f"Default: {DEFAULT_DIVERSE_DETECTION_FRAMES_PER_SIDE}."
        ),
    )
    parser.add_argument(
        "--max-diverse-fit-frames-per-side",
        type=int,
        default=DEFAULT_DIVERSE_FIT_FRAMES_PER_SIDE,
        help=(
            "Maximum pose-diverse detected frames used by optimizer per record/side. "
            f"Default: {DEFAULT_DIVERSE_FIT_FRAMES_PER_SIDE}."
        ),
    )
    parser.add_argument(
        "--min-diverse-frames-per-side",
        type=int,
        default=DEFAULT_DIVERSE_MIN_FRAMES_PER_SIDE,
        help=(
            "Minimum frames retained before early-stopping the pose-diverse selector. "
            f"Default: {DEFAULT_DIVERSE_MIN_FRAMES_PER_SIDE}."
        ),
    )
    parser.add_argument(
        "--diverse-translation-scale-m",
        type=float,
        default=DEFAULT_DIVERSE_TRANSLATION_SCALE_M,
        help=(
            "Camera translation difference that counts as roughly one diversity unit. "
            f"Default: {DEFAULT_DIVERSE_TRANSLATION_SCALE_M:g} m."
        ),
    )
    parser.add_argument(
        "--diverse-rotation-scale-deg",
        type=float,
        default=DEFAULT_DIVERSE_ROTATION_SCALE_DEG,
        help=(
            "Camera rotation difference that counts as roughly one diversity unit. "
            f"Default: {DEFAULT_DIVERSE_ROTATION_SCALE_DEG:g} deg."
        ),
    )
    parser.add_argument(
        "--diverse-min-score",
        type=float,
        default=0.75,
        help="Stop selecting extra frames once the best remaining pose distance is below this score. Default: 0.75.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    detections_root = output_root / "detections"
    output_root.mkdir(parents=True, exist_ok=True)
    detections_root.mkdir(parents=True, exist_ok=True)

    records = [record for record in args.records if (raw_root / record).is_dir()]
    missing = [record for record in args.records if record not in records]
    if missing:
        raise SystemExit(f"Missing record directories: {missing}")

    detection_summary = []
    emit_progress("start", 0.0, records=records, outputRoot=str(output_root))
    if not args.skip_detection:
        total_scans = max(1, len(records) * 2)
        completed_scans = 0
        for record in records:
            record_dir = raw_root / record
            metadata = json.loads((record_dir / "quest_camera_metadata.json").read_text(encoding="utf-8"))
            for side in ("left", "right"):
                emit_progress(
                    "detecting",
                    0.05 + 0.55 * completed_scans / total_scans,
                    record=record,
                    side=side,
                )
                detection_summary.append(scan_video(record_dir, side, metadata, detections_root, args))
                completed_scans += 1
                emit_progress(
                    "detected",
                    0.05 + 0.55 * completed_scans / total_scans,
                    record=record,
                    side=side,
                )
        write_detection_summaries(detection_summary, output_root)
    else:
        emit_progress("loading_detections", 0.55, records=records)
        detection_summary = read_existing_detection_summary(records, detections_root, output_root)

    red_anchor_global = infer_global_red_anchor(records, detections_root, args)
    setattr(args, "resolved_red_anchor_target_index", red_anchor_global.get("target_index"))
    setattr(args, "red_anchor_global_summary", red_anchor_global)
    emit_progress("fitting", 0.65, records=records)
    try:
        result = fit_per_record(records, raw_root, detections_root, args)
    except CalibrationFailure as exc:
        diagnostics = dict(exc.diagnostics)
        diagnostics.setdefault("reason", exc.reason)
        diagnostics.setdefault("records", records)
        diagnostics.setdefault("square_size_m", args.square_size)
        diagnostics.setdefault("pattern", [args.pattern_cols, args.pattern_rows])
        diagnostics["detection_summary"] = detection_summary
        failure_path = output_root / "calibration_failure_25mm.json"
        write_json(diagnostics, failure_path)
        emit_progress(
            "failed",
            1.0,
            records=records,
            reason=diagnostics.get("reason"),
            reasonCode=diagnostics.get("reason_code"),
            failurePath=str(failure_path),
            diagnostics=diagnostics,
        )
        print("CALIBRATION_FAILURE_JSON " + json.dumps(compact_failure(diagnostics), ensure_ascii=False, separators=(",", ":")))
        raise SystemExit(f"Calibration failed: {diagnostics.get('reason')}")
    result["description"] = (
        "Per-record calibration from recorded passthrough videos and recorded passthrough camera poses. "
        "The fit runs in the raw Unity trajectory frame, then T_world_board is exported in the PC "
        "right-handed world frame; T_unity_world_board preserves the raw Unity result."
    )
    result["records"] = records
    result["pattern"] = [args.pattern_cols, args.pattern_rows]
    result["square_size_m"] = args.square_size
    result["detection_summary"] = detection_summary
    result["red_anchor_global"] = red_anchor_global
    result["coordinate_frame"] = PC_WORLD_FRAME
    result["raw_trajectory_frame"] = UNITY_WORLD_FRAME
    result["world_frame_conversion"] = WORLD_FRAME_CONVERSION

    emit_progress("writing", 0.95, records=records)
    stale_failure_path = output_root / "calibration_failure_25mm.json"
    if stale_failure_path.exists():
        stale_failure_path.unlink()
    write_json(result, output_root / "calibration_result_25mm.json")
    write_report(result, output_root / "calibration_report_25mm.md")
    emit_progress(
        "done",
        1.0,
        records=records,
        resultPath=str(output_root / "calibration_result_25mm.json"),
        reportPath=str(output_root / "calibration_report_25mm.md"),
    )
    print(json.dumps(compact_result(result), indent=2))
    print(f"Wrote {output_root / 'calibration_result_25mm.json'}")
    print(f"Wrote {output_root / 'calibration_report_25mm.md'}")
    return 0


def emit_progress(stage: str, progress: float, **extra: Any) -> None:
    if os.environ.get("QUEST_CALIB_PROGRESS") != "1":
        return
    payload = {
        "stage": stage,
        "progress": max(0.0, min(1.0, float(progress))),
        **extra,
    }
    print("PROGRESS_JSON " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def camera_matrix(intrinsics: list[float]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def image_y_sign(args: argparse.Namespace) -> float:
    axis = getattr(args, "image_y_axis", "down")
    if axis == "auto":
        raise ValueError("image_y_axis must be resolved before projection")
    return -1.0 if axis == "down" else 1.0


def object_points(args: argparse.Namespace, order: str = "identity") -> np.ndarray:
    cols, rows = args.pattern_cols, args.pattern_rows
    xy = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64)
    if order == "rot180":
        xy[:, 0] = cols - 1 - xy[:, 0]
        xy[:, 1] = rows - 1 - xy[:, 1]
    elif order != "identity":
        raise ValueError(order)
    obj = np.zeros((cols * rows, 3), dtype=np.float64)
    obj[:, :2] = xy * args.square_size
    return obj


def rot180_index(args: argparse.Namespace) -> np.ndarray:
    return np.arange(args.pattern_rows * args.pattern_cols).reshape(args.pattern_rows, args.pattern_cols)[::-1, ::-1].reshape(-1)


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[bool, np.ndarray | None, str]:
    flags_sb = 0
    for name in ("CALIB_CB_NORMALIZE_IMAGE", "CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY"):
        flags_sb |= int(getattr(cv2, name, 0))
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags_sb)
        if ok:
            return True, corners.reshape(-1, 2).astype(np.float64), "sb"

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return False, None, "none"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    return True, corners.reshape(-1, 2).astype(np.float64), "classic"


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


def red_anchor_target_index_for_observed(observed_index: int, args: argparse.Namespace, order_180: np.ndarray) -> int | None:
    total = args.pattern_cols * args.pattern_rows
    if observed_index < 0 or observed_index >= total:
        return None
    configured = getattr(args, "resolved_red_anchor_target_index", None)
    if configured is None:
        configured = getattr(args, "red_anchor_target_index", None)
    if configured is not None:
        target = int(configured)
        return target if 0 <= target < total else None
    paired = int(order_180[int(observed_index)])
    return int(min(int(observed_index), paired))


def infer_global_red_anchor(records: list[str], detections_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    order_180 = rot180_index(args)
    total = args.pattern_cols * args.pattern_rows
    configured = getattr(args, "red_anchor_target_index", None)
    if getattr(args, "disable_red_anchor", False):
        return {"enabled": False, "target_index": None, "source": "disabled", "ok_frames": 0, "counts": {}}
    if configured is not None:
        target = int(configured)
        return {
            "enabled": True,
            "target_index": target if 0 <= target < total else None,
            "source": "configured",
            "ok_frames": 0,
            "counts": {},
        }
    counts: dict[int, int] = {}
    observed_counts: dict[int, int] = {}
    score_sum: dict[int, float] = {}
    for record in records:
        for side in ("left", "right"):
            path = detections_root / f"{record}_{side}_checkerboard_25mm.npz"
            if not path.exists():
                continue
            det = dict(np.load(path, allow_pickle=True))
            ok_values = np.asarray(det.get("red_anchor_ok", []), dtype=bool)
            observed_values = np.asarray(det.get("red_anchor_observed_indices", []), dtype=int)
            best_scores = np.asarray(det.get("red_anchor_best_scores", []), dtype=float)
            for i, ok in enumerate(ok_values):
                if not bool(ok) or i >= len(observed_values):
                    continue
                observed = int(observed_values[i])
                if observed < 0 or observed >= total:
                    continue
                target = int(min(observed, int(order_180[observed])))
                counts[target] = counts.get(target, 0) + 1
                observed_counts[observed] = observed_counts.get(observed, 0) + 1
                score_sum[target] = score_sum.get(target, 0.0) + (float(best_scores[i]) if i < len(best_scores) else 0.0)
    if not counts:
        return {
            "enabled": True,
            "target_index": None,
            "source": "none_detected",
            "ok_frames": 0,
            "counts": {},
            "observed_counts": {},
        }
    target = max(counts, key=lambda idx: (counts[idx], score_sum.get(idx, 0.0)))
    return {
        "enabled": True,
        "target_index": int(target),
        "source": "auto_from_any_reliable_red_anchor",
        "ok_frames": int(sum(counts.values())),
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "observed_counts": {str(key): int(value) for key, value in sorted(observed_counts.items())},
        "score_sum": {str(key): float(value) for key, value in sorted(score_sum.items())},
        "note": "Red anchor is a global optional orientation hint; frames without red fall back to identity/rot180 reprojection matching.",
    }


def detection_red_anchor_order(det: dict[str, Any], det_index: int, args: argparse.Namespace, order_180: np.ndarray) -> tuple[str | None, int | None, int | None, float, float]:
    if getattr(args, "disable_red_anchor", False):
        return None, None, None, 0.0, 0.0
    ok_values = det.get("red_anchor_ok")
    observed_values = det.get("red_anchor_observed_indices")
    if ok_values is None or observed_values is None or det_index >= len(observed_values):
        return None, None, None, 0.0, 0.0
    if not bool(np.asarray(ok_values)[det_index]):
        return None, None, None, 0.0, 0.0
    observed = int(np.asarray(observed_values, dtype=int)[det_index])
    target = red_anchor_target_index_for_observed(observed, args, order_180)
    if target is None:
        return None, observed, None, 0.0, 0.0
    order = red_anchor_order_for_observed(observed, target, order_180)
    best_scores = det.get("red_anchor_best_scores")
    ratios = det.get("red_anchor_score_ratios")
    best_score = float(np.asarray(best_scores, dtype=float)[det_index]) if best_scores is not None and det_index < len(best_scores) else 0.0
    ratio = float(np.asarray(ratios, dtype=float)[det_index]) if ratios is not None and det_index < len(ratios) else 0.0
    return order, observed, target, best_score, ratio


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
    reason = "ok" if ok else "weak_or_ambiguous_red_corner"
    return {
        "ok": bool(ok),
        "observedIndex": int(best) if ok else -1,
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
        "reason": reason,
    }


def solve_pose(corners: np.ndarray, k: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dist = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    err = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
    return rvec.reshape(3), tvec.reshape(3), float(np.sqrt(np.mean(err * err)))


def scan_video(record_dir: Path, side: str, metadata: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    pattern = (args.pattern_cols, args.pattern_rows)
    obj = object_points(args)
    k = camera_matrix(metadata[f"{side}Intrinsics"])
    frames_meta = read_jsonl(record_dir / f"{side}_frames.jsonl")
    by_index = {int(row["frameIndex"]): row for row in frames_meta}
    video_path = record_dir / f"{side}_recording.mp4"
    candidate_frame_indices: set[int] | None = None
    selection_summary: dict[str, Any] = {
        "enabled": False,
        "input_count": len(frames_meta),
        "selected_count": len(frames_meta),
        "reason": "disabled",
    }
    if not getattr(args, "disable_diverse_frame_selection", False):
        pose_positions, pose_rotations, pose_frame_indices = metadata_pose_arrays(frames_meta)
        selected_local, selection_summary = diverse_pose_indices(
            pose_positions,
            pose_rotations,
            int(args.max_diverse_detection_frames_per_side),
            int(args.min_diverse_frames_per_side),
            float(args.diverse_translation_scale_m),
            float(args.diverse_rotation_scale_deg),
            float(args.diverse_min_score),
        )
        if len(pose_frame_indices) == 0:
            candidate_frame_indices = None
            selection_summary = {
                "enabled": True,
                "input_count": len(frames_meta),
                "selected_count": len(frames_meta),
                "reason": "missing_pose_metadata_fallback_all_frames",
            }
        else:
            candidate_frame_indices = {int(pose_frame_indices[i]) for i in selected_local}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    frame_indices: list[int] = []
    unity_times: list[float] = []
    camera_times: list[float] = []
    corners_all: list[np.ndarray] = []
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    reproj_errors: list[float] = []
    methods: list[str] = []
    red_anchor_ok: list[bool] = []
    red_anchor_observed_indices: list[int] = []
    red_anchor_best_indices: list[int] = []
    red_anchor_second_indices: list[int] = []
    red_anchor_best_scores: list[float] = []
    red_anchor_second_scores: list[float] = []
    red_anchor_score_ratios: list[float] = []
    red_anchor_radius_px: list[float] = []
    red_anchor_grid_spacing_px: list[float] = []
    red_anchor_scores_by_corner: list[np.ndarray] = []
    red_anchor_red_pixels_by_corner: list[np.ndarray] = []
    best_payload = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    red_corner_indices = checkerboard_corner_indices(args.pattern_cols, args.pattern_rows)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        meta = by_index.get(frame_index)
        if meta is not None and (candidate_frame_indices is None or frame_index in candidate_frame_indices):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners, method = detect_corners(gray, pattern)
            if found and corners is not None:
                red_anchor = (
                    detect_red_corner_anchor(frame, corners, args.pattern_cols, args.pattern_rows)
                    if not args.disable_red_anchor
                    else {"ok": False, "observedIndex": -1, "reason": "disabled"}
                )
                rvec, tvec, reproj = solve_pose(corners, k, obj)
                frame_indices.append(frame_index)
                unity_times.append(float(meta["unityTimestampSeconds"]))
                camera_times.append(float(meta["cameraTimestampSeconds"]))
                corners_all.append(corners)
                rvecs.append(rvec)
                tvecs.append(tvec)
                reproj_errors.append(reproj)
                methods.append(method)
                red_anchor_ok.append(bool(red_anchor.get("ok")))
                red_anchor_observed_indices.append(int(red_anchor.get("observedIndex", -1)))
                red_anchor_best_indices.append(int(red_anchor.get("bestIndex", -1)))
                red_anchor_second_indices.append(int(red_anchor.get("secondIndex", -1)))
                red_anchor_best_scores.append(float(red_anchor.get("bestScore", 0.0)))
                red_anchor_second_scores.append(float(red_anchor.get("secondScore", 0.0)))
                red_anchor_score_ratios.append(float(red_anchor.get("scoreRatio", 0.0)))
                red_anchor_radius_px.append(float(red_anchor.get("radiusPx", 0.0)))
                red_anchor_grid_spacing_px.append(float(red_anchor.get("gridSpacingPx", 0.0)))
                scores = red_anchor.get("scoresByCorner") if isinstance(red_anchor.get("scoresByCorner"), dict) else {}
                pixels = red_anchor.get("redPixels") if isinstance(red_anchor.get("redPixels"), dict) else {}
                red_anchor_scores_by_corner.append(np.asarray([float(scores.get(str(idx), 0.0)) for idx in red_corner_indices], dtype=np.float64))
                red_anchor_red_pixels_by_corner.append(np.asarray([int(pixels.get(str(idx), 0)) for idx in red_corner_indices], dtype=np.int32))
                if best_payload is None or reproj < best_payload["reproj"]:
                    best_payload = {
                        "frame": frame.copy(),
                        "frame_index": frame_index,
                        "corners": corners,
                        "rvec": rvec,
                        "tvec": tvec,
                        "reproj": reproj,
                        "red_anchor": red_anchor,
                    }
        frame_index += 1
    cap.release()

    npz_path = out_dir / f"{record_dir.name}_{side}_checkerboard_25mm.npz"
    np.savez_compressed(
        npz_path,
        pattern_size=np.asarray(pattern, dtype=np.int32),
        square_size_m=np.asarray([args.square_size], dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        unity_times=np.asarray(unity_times, dtype=np.float64),
        camera_times=np.asarray(camera_times, dtype=np.float64),
        corners=np.asarray(corners_all, dtype=np.float64),
        rvecs=np.asarray(rvecs, dtype=np.float64),
        tvecs=np.asarray(tvecs, dtype=np.float64),
        reproj_errors=np.asarray(reproj_errors, dtype=np.float64),
        methods=np.asarray(methods),
        red_anchor_corner_indices=np.asarray(red_corner_indices, dtype=np.int32),
        red_anchor_ok=np.asarray(red_anchor_ok, dtype=np.bool_),
        red_anchor_observed_indices=np.asarray(red_anchor_observed_indices, dtype=np.int32),
        red_anchor_best_indices=np.asarray(red_anchor_best_indices, dtype=np.int32),
        red_anchor_second_indices=np.asarray(red_anchor_second_indices, dtype=np.int32),
        red_anchor_best_scores=np.asarray(red_anchor_best_scores, dtype=np.float64),
        red_anchor_second_scores=np.asarray(red_anchor_second_scores, dtype=np.float64),
        red_anchor_score_ratios=np.asarray(red_anchor_score_ratios, dtype=np.float64),
        red_anchor_radius_px=np.asarray(red_anchor_radius_px, dtype=np.float64),
        red_anchor_grid_spacing_px=np.asarray(red_anchor_grid_spacing_px, dtype=np.float64),
        red_anchor_scores_by_corner=np.asarray(red_anchor_scores_by_corner, dtype=np.float64),
        red_anchor_red_pixels_by_corner=np.asarray(red_anchor_red_pixels_by_corner, dtype=np.int32),
        camera_matrix=k,
        dist_coeffs=np.zeros(5, dtype=np.float64),
    )

    overlay_path = None
    if best_payload is not None:
        overlay = draw_overlay(
            best_payload["frame"],
            best_payload["corners"],
            best_payload["rvec"],
            best_payload["tvec"],
            k,
            pattern,
            best_payload.get("red_anchor"),
        )
        label = f"{record_dir.name} {side} 25mm frame {best_payload['frame_index']} err {best_payload['reproj']:.2f}px"
        cv2.putText(overlay, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        overlay_path = out_dir / f"{record_dir.name}_{side}_best_overlay_25mm.jpg"
        cv2.imwrite(str(overlay_path), overlay)

    reproj = np.asarray(reproj_errors, dtype=float)
    anchor_ok_arr = np.asarray(red_anchor_ok, dtype=bool)
    observed_arr = np.asarray(red_anchor_observed_indices, dtype=int)
    anchor_counts = {
        str(index): int(np.sum((observed_arr == int(index)) & anchor_ok_arr))
        for index in red_corner_indices
    }
    return {
        "record": record_dir.name,
        "side": side,
        "video_frame_count": total,
        "frame_metadata_count": len(frames_meta),
        "detections": len(frame_indices),
        "detection_ratio": len(frame_indices) / total if total else 0.0,
        "first_detection_frame": int(frame_indices[0]) if frame_indices else None,
        "last_detection_frame": int(frame_indices[-1]) if frame_indices else None,
        "reprojection_error_px": stats(reproj),
        "diverse_frame_selection": selection_summary,
        "red_anchor": {
            "enabled": not args.disable_red_anchor,
            "ok_count": int(np.sum(anchor_ok_arr)),
            "ok_ratio": float(np.mean(anchor_ok_arr)) if len(anchor_ok_arr) else 0.0,
            "observed_index_counts": anchor_counts,
            "corner_indices": [int(v) for v in red_corner_indices],
        },
        "npz": display_path(npz_path),
        "best_overlay": display_path(overlay_path) if overlay_path else None,
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


def draw_overlay(
    frame: np.ndarray,
    corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    k: np.ndarray,
    pattern: tuple[int, int],
    red_anchor: dict[str, Any] | None = None,
) -> np.ndarray:
    out = frame.copy()
    cv2.drawChessboardCorners(out, pattern, corners.reshape(-1, 1, 2).astype(np.float32), True)
    draw_red_anchor_overlay(out, corners, red_anchor, pattern[0], pattern[1])
    axis = np.float32([[0, 0, 0], [0.20, 0, 0], [0, 0.15, 0], [0, 0, 0.15]])
    pts, _ = cv2.projectPoints(axis, rvec, tvec, k, np.zeros(5))
    pts = np.round(pts.reshape(-1, 2)).astype(int)
    o, x, y, z = pts
    cv2.line(out, tuple(o), tuple(x), (0, 0, 255), 3)
    cv2.line(out, tuple(o), tuple(y), (0, 255, 0), 3)
    cv2.line(out, tuple(o), tuple(z), (255, 0, 0), 3)
    return out


def read_existing_detection_summary(records: list[str], detections_root: Path, output_root: Path) -> list[dict[str, Any]]:
    summary_path = output_root / "checkerboard_detection_summary_25mm.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for record in records:
        for side in ("left", "right"):
            path = detections_root / f"{record}_{side}_checkerboard_25mm.npz"
            if not path.exists():
                raise SystemExit(f"Missing detection file with --skip-detection: {path}")
            det = np.load(path, allow_pickle=True)
            reproj = np.asarray(det["reproj_errors"], dtype=float)
            rows.append(
                {
                    "record": record,
                    "side": side,
                    "detections": int(len(det["frame_indices"])),
                    "first_detection_frame": int(det["frame_indices"][0]) if len(det["frame_indices"]) else None,
                    "last_detection_frame": int(det["frame_indices"][-1]) if len(det["frame_indices"]) else None,
                    "reprojection_error_px": stats(reproj),
                    "npz": display_path(path),
                    "best_overlay": None,
                }
            )
    write_detection_summaries(rows, output_root)
    return rows


def write_detection_summaries(rows: list[dict[str, Any]], output_root: Path) -> None:
    write_json(rows, output_root / "checkerboard_detection_summary_25mm.json")
    csv_path = output_root / "checkerboard_detection_summary_25mm.csv"
    fieldnames = [
        "record",
        "side",
        "video_frame_count",
        "frame_metadata_count",
        "detections",
        "detection_ratio",
        "first_detection_frame",
        "last_detection_frame",
        "red_anchor",
        "npz",
        "best_overlay",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_detection(record: str, side: str, detections_root: Path) -> dict[str, Any]:
    return dict(np.load(detections_root / f"{record}_{side}_checkerboard_25mm.npz", allow_pickle=True))


def build_batches(
    records: list[str],
    raw_root: Path,
    detections_root: Path,
    lag_seconds: float,
    args: argparse.Namespace,
    stride: int,
    corrected_orders: dict[tuple[str, str, int], str] | None = None,
    corrected_keep: set[tuple[str, str, int]] | None = None,
) -> list[Batch]:
    batches = []
    order_180 = rot180_index(args)
    for record in records:
        for side in ("left", "right"):
            det = load_detection(record, side, detections_root)
            reproj = np.asarray(det["reproj_errors"], dtype=float)
            keep = reproj <= args.max_reproj_px
            idx = np.flatnonzero(keep)[::stride]
            if corrected_keep is not None:
                idx = np.asarray(
                    [
                        i
                        for i in idx
                        if (record, side, int(det["frame_indices"][i])) in corrected_keep
                    ],
                    dtype=int,
                )
            if len(idx) == 0:
                continue
            times = np.asarray(det["unity_times"], dtype=float)[idx]
            pose_series = PoseSeries.from_jsonl(raw_root / record / f"{side}_frames.jsonl")
            cam_pos, cam_rot, valid = pose_series.sample(times + lag_seconds)
            idx = idx[valid]
            times = times[valid]
            if len(idx) == 0:
                continue
            corners = np.asarray(det["corners"], dtype=float)[idx].copy()
            anchor_observed: list[int] = []
            anchor_best_scores: list[float] = []
            anchor_score_ratios: list[float] = []
            anchor_targets: list[int] = []
            anchor_orders: list[str] = []
            for local_i, global_i in enumerate(idx):
                frame = int(det["frame_indices"][global_i])
                key = (record, side, frame)
                anchor_order, observed, target, best_score, score_ratio = detection_red_anchor_order(det, int(global_i), args, order_180)
                order = corrected_orders.get(key) if corrected_orders is not None and key in corrected_orders else anchor_order
                if order == "rot180":
                    corners[local_i] = corners[local_i, order_180, :]
                anchor_observed.append(int(observed) if observed is not None else -1)
                anchor_best_scores.append(float(best_score))
                anchor_score_ratios.append(float(score_ratio))
                anchor_targets.append(int(target) if target is not None else -1)
                anchor_orders.append(anchor_order or "")
            batches.append(
                Batch(
                    record=record,
                    side=side,
                    frame_indices=np.asarray(det["frame_indices"], dtype=int)[idx],
                    times=times,
                    corners=corners,
                    camera_positions=cam_pos,
                    camera_rotations=cam_rot,
                    red_anchor_observed_indices=np.asarray(anchor_observed, dtype=np.int32),
                    red_anchor_best_scores=np.asarray(anchor_best_scores, dtype=np.float64),
                    red_anchor_score_ratios=np.asarray(anchor_score_ratios, dtype=np.float64),
                    red_anchor_target_indices=np.asarray(anchor_targets, dtype=np.int32),
                    red_anchor_orders=np.asarray(anchor_orders, dtype=object),
                )
            )
    return batches


def select_diverse_batch_frames(batch: Batch, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if getattr(args, "disable_diverse_frame_selection", False):
        idx = np.arange(len(batch.frame_indices), dtype=int)
        return idx, {
            "enabled": False,
            "record": batch.record,
            "side": batch.side,
            "input_frames": int(len(idx)),
            "selected_frames": int(len(idx)),
            "reason": "disabled",
        }
    selected, summary = diverse_pose_indices(
        batch.camera_positions,
        batch.camera_rotations,
        int(args.max_diverse_fit_frames_per_side),
        int(args.min_diverse_frames_per_side),
        float(args.diverse_translation_scale_m),
        float(args.diverse_rotation_scale_deg),
        float(args.diverse_min_score),
    )
    summary.update(
        {
            "record": batch.record,
            "side": batch.side,
            "input_frames": int(len(batch.frame_indices)),
            "selected_frames": int(len(selected)),
        }
    )
    return selected, summary


def apply_diverse_batch_selection(batches: list[Batch], args: argparse.Namespace) -> tuple[list[Batch], dict[str, Any]]:
    selected_batches: list[Batch] = []
    rows = []
    input_frames = 0
    selected_frames = 0
    for batch in batches:
        input_frames += int(len(batch.frame_indices))
        idx, row = select_diverse_batch_frames(batch, args)
        rows.append(row)
        selected_frames += int(len(idx))
        if len(idx) == 0:
            continue
        selected_batches.append(
            Batch(
                record=batch.record,
                side=batch.side,
                frame_indices=batch.frame_indices[idx],
                times=batch.times[idx],
                corners=batch.corners[idx],
                camera_positions=batch.camera_positions[idx],
                camera_rotations=batch.camera_rotations[idx],
                red_anchor_observed_indices=batch.red_anchor_observed_indices[idx],
                red_anchor_best_scores=batch.red_anchor_best_scores[idx],
                red_anchor_score_ratios=batch.red_anchor_score_ratios[idx],
                red_anchor_target_indices=batch.red_anchor_target_indices[idx],
                red_anchor_orders=batch.red_anchor_orders[idx],
            )
        )
    return selected_batches, {
        "enabled": not getattr(args, "disable_diverse_frame_selection", False),
        "input_frames": int(input_frames),
        "selected_frames": int(selected_frames),
        "selected_fraction": float(selected_frames / input_frames) if input_frames else 0.0,
        "max_fit_frames_per_side": int(args.max_diverse_fit_frames_per_side),
        "min_frames_per_side": int(args.min_diverse_frames_per_side),
        "rows": rows,
    }


def initial_params(records: list[str], raw_root: Path, detections_root: Path, args: argparse.Namespace, intr_scale: float) -> np.ndarray:
    metadata = json.loads((raw_root / records[0] / "quest_camera_metadata.json").read_text(encoding="utf-8"))
    left = np.asarray(metadata["leftIntrinsics"], dtype=float)
    right = np.asarray(metadata["rightIntrinsics"], dtype=float)
    left[:2] *= intr_scale
    right[:2] *= intr_scale
    params: list[float] = [*left.tolist(), *right.tolist()]
    if args.model == "radial2":
        params += [0.0, 0.0, 0.0, 0.0]
    for record in records:
        rvec, t = triangulate_initial_pose(record, raw_root, detections_root, args)
        params += [*rvec.tolist(), *t.tolist()]
    return np.asarray(params, dtype=float)


def triangulate_initial_pose(record: str, raw_root: Path, detections_root: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((raw_root / record / "quest_camera_metadata.json").read_text(encoding="utf-8"))
    k_left = camera_matrix(metadata["leftIntrinsics"])
    k_right = camera_matrix(metadata["rightIntrinsics"])
    ray_map = np.diag([1.0, image_y_sign(args), 1.0])
    order_180 = rot180_index(args)
    left = load_detection(record, "left", detections_root)
    right = load_detection(record, "right", detections_root)
    li = {int(value): i for i, value in enumerate(left["frame_indices"])}
    ri = {int(value): i for i, value in enumerate(right["frame_indices"])}
    common = sorted(set(li).intersection(ri))
    if len(common) > 80:
        common = [common[i] for i in np.linspace(0, len(common) - 1, 80).round().astype(int)]

    left_series = PoseSeries.from_jsonl(raw_root / record / "left_frames.jsonl")
    right_series = PoseSeries.from_jsonl(raw_root / record / "right_frames.jsonl")
    per_corner: list[list[np.ndarray]] = [[] for _ in range(args.pattern_cols * args.pattern_rows)]

    def rays(corners: np.ndarray, k: np.ndarray) -> np.ndarray:
        x = (corners[:, 0] - k[0, 2]) / k[0, 0]
        y = (corners[:, 1] - k[1, 2]) / k[1, 1]
        cv = np.column_stack([x, y, np.ones_like(x)])
        local = cv @ ray_map.T
        return local / np.linalg.norm(local, axis=1, keepdims=True)

    for frame_index in common:
        il = li[frame_index]
        ir = ri[frame_index]
        lp, lr, lv = left_series.sample(np.asarray([left["unity_times"][il]], dtype=float))
        rp, rr, rv = right_series.sample(np.asarray([right["unity_times"][ir]], dtype=float))
        if not bool(lv[0]) or not bool(rv[0]):
            continue
        left_corners = np.asarray(left["corners"][il], dtype=float)
        right_corners = np.asarray(right["corners"][ir], dtype=float)
        left_order, _, _, _, _ = detection_red_anchor_order(left, il, args, order_180)
        right_order, _, _, _, _ = detection_red_anchor_order(right, ir, args, order_180)
        if left_order == "rot180":
            left_corners = left_corners[order_180]
        if right_order == "rot180":
            right_corners = right_corners[order_180]
        left_dirs = lr.apply(rays(left_corners, k_left))
        right_dirs = rr.apply(rays(right_corners, k_right))
        left_orig = np.repeat(lp, len(left_dirs), axis=0)
        right_orig = np.repeat(rp, len(right_dirs), axis=0)
        points, dist = closest_ray_midpoints(left_orig, left_dirs, right_orig, right_dirs)
        if np.median(dist) > 0.03:
            continue
        for corner_idx, point in enumerate(points):
            per_corner[corner_idx].append(point)

    dst = []
    for samples in per_corner:
        if len(samples) < 5:
            dst.append([np.nan, np.nan, np.nan])
        else:
            dst.append(np.median(np.asarray(samples), axis=0))
    dst_arr = np.asarray(dst)
    obj = object_points(args)
    valid = np.all(np.isfinite(dst_arr), axis=1)
    if int(valid.sum()) < 20:
        return Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_rotvec(), np.asarray([0.2, 1.35, -0.2], dtype=float)

    src = obj[valid]
    dst_valid = dst_arr[valid]
    src_center = src.mean(axis=0)
    dst_center = dst_valid.mean(axis=0)
    h = (src - src_center).T @ (dst_valid - dst_center)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    t = dst_center - r @ src_center
    return Rotation.from_matrix(r).as_rotvec(), t


def closest_ray_midpoints(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w0 = o1 - o2
    a = np.einsum("ij,ij->i", d1, d1)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d2, d2)
    d = np.einsum("ij,ij->i", d1, w0)
    e = np.einsum("ij,ij->i", d2, w0)
    denom = a * c - b * b
    denom = np.where(np.abs(denom) < 1e-10, np.nan, denom)
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p1 = o1 + s[:, None] * d1
    p2 = o2 + t[:, None] * d2
    return 0.5 * (p1 + p2), np.linalg.norm(p1 - p2, axis=1)


def bounds(args: argparse.Namespace, record_count: int) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for _ in range(2):
        lower += [150.0, 150.0, 0.0, 0.0]
        upper += [900.0, 900.0, IMAGE_SIZE[0], IMAGE_SIZE[1]]
    if args.model == "radial2":
        lower += [-2.0, -2.0, -2.0, -2.0]
        upper += [2.0, 2.0, 2.0, 2.0]
    for _ in range(record_count):
        lower += [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        upper += [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def parse_params(params: np.ndarray, records: list[str], args: argparse.Namespace) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "left_intr": params[0:4],
        "right_intr": params[4:8],
    }
    offset = 8
    if args.model == "radial2":
        parsed["left_dist"] = params[8:10]
        parsed["right_dist"] = params[10:12]
        offset = 12
    else:
        parsed["left_dist"] = np.zeros(2, dtype=float)
        parsed["right_dist"] = np.zeros(2, dtype=float)
    record_poses = {}
    for record in records:
        record_poses[record] = {
            "board_rvec": params[offset : offset + 3],
            "board_t": params[offset + 3 : offset + 6],
        }
        offset += 6
    parsed["records"] = record_poses
    return parsed


def project_points(points_camera: np.ndarray, intr: np.ndarray, dist: np.ndarray, y_sign: float) -> np.ndarray:
    z = points_camera[..., 2]
    z_safe = np.where(z <= 1e-4, 1e-4, z)
    x = points_camera[..., 0] / z_safe
    y = y_sign * points_camera[..., 1] / z_safe
    k1, k2 = dist
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    fx, fy, cx, cy = intr
    return np.stack([fx * x * radial + cx, fy * y * radial + cy], axis=-1)


def residuals(params: np.ndarray, batches: list[Batch], records: list[str], args: argparse.Namespace, obj: np.ndarray) -> np.ndarray:
    parsed = parse_params(params, records, args)
    y_sign = image_y_sign(args)
    world_points_by_record = {}
    for record, pose in parsed["records"].items():
        r_w_b = Rotation.from_rotvec(pose["board_rvec"]).as_matrix()
        world_points_by_record[record] = obj @ r_w_b.T + pose["board_t"]

    chunks = []
    behind_chunks = []
    for batch in batches:
        p_world = world_points_by_record[batch.record]
        r_c_w = batch.camera_rotations.inv().as_matrix()
        rel = p_world[None, :, :] - batch.camera_positions[:, None, :]
        p_camera = np.einsum("nij,nmj->nmi", r_c_w, rel)
        pred = project_points(p_camera, parsed[f"{batch.side}_intr"], parsed[f"{batch.side}_dist"], y_sign)
        chunks.append((pred - batch.corners).reshape(-1))
        behind = np.minimum(p_camera[..., 2] - 0.05, 0.0)
        behind_chunks.append(behind.reshape(-1) * 1000.0)
    chunks.extend(behind_chunks)
    return np.concatenate(chunks)


def reprojection_errors(params: np.ndarray, batches: list[Batch], records: list[str], args: argparse.Namespace, obj: np.ndarray) -> list[dict[str, Any]]:
    parsed = parse_params(params, records, args)
    y_sign = image_y_sign(args)
    rows = []
    for batch in batches:
        pose = parsed["records"][batch.record]
        r_w_b = Rotation.from_rotvec(pose["board_rvec"]).as_matrix()
        p_world = obj @ r_w_b.T + pose["board_t"]
        r_c_w = batch.camera_rotations.inv().as_matrix()
        rel = p_world[None, :, :] - batch.camera_positions[:, None, :]
        p_camera = np.einsum("nij,nmj->nmi", r_c_w, rel)
        pred = project_points(p_camera, parsed[f"{batch.side}_intr"], parsed[f"{batch.side}_dist"], y_sign)
        err = np.linalg.norm(pred - batch.corners, axis=-1)
        for frame_index, time_s, frame_err in zip(batch.frame_indices, batch.times, err):
            rows.append(
                {
                    "record": batch.record,
                    "side": batch.side,
                    "frame_index": int(frame_index),
                    "unity_time": float(time_s),
                    "median_px": float(np.median(frame_err)),
                    "mean_px": float(np.mean(frame_err)),
                    "p90_px": float(np.percentile(frame_err, 90)),
                }
            )
    return rows


def stats_for_params(params: np.ndarray, batches: list[Batch], records: list[str], args: argparse.Namespace, obj: np.ndarray) -> dict[str, Any]:
    parsed = parse_params(params, records, args)
    y_sign = image_y_sign(args)
    all_err = []
    per_batch = []
    for batch in batches:
        pose = parsed["records"][batch.record]
        r_w_b = Rotation.from_rotvec(pose["board_rvec"]).as_matrix()
        p_world = obj @ r_w_b.T + pose["board_t"]
        r_c_w = batch.camera_rotations.inv().as_matrix()
        rel = p_world[None, :, :] - batch.camera_positions[:, None, :]
        p_camera = np.einsum("nij,nmj->nmi", r_c_w, rel)
        pred = project_points(p_camera, parsed[f"{batch.side}_intr"], parsed[f"{batch.side}_dist"], y_sign)
        err = np.linalg.norm(pred - batch.corners, axis=-1)
        all_err.append(err.reshape(-1))
        frame_median = np.median(err, axis=1)
        per_batch.append(
            {
                "record": batch.record,
                "side": batch.side,
                "frames": int(len(batch.frame_indices)),
                "corners": int(err.size),
                "mean_px": float(err.mean()),
                "median_px": float(np.median(err)),
                "p90_px": float(np.percentile(err, 90)),
                "p95_px": float(np.percentile(err, 95)),
                "max_px": float(err.max()),
                "frame_median_px_p90": float(np.percentile(frame_median, 90)),
                "points_behind_camera": int(np.sum(p_camera[..., 2] <= 0.0)),
            }
        )
    err = np.concatenate(all_err)
    return {
        "overall": {
            "corners": int(err.size),
            "mean_px": float(err.mean()),
            "median_px": float(np.median(err)),
            "p90_px": float(np.percentile(err, 90)),
            "p95_px": float(np.percentile(err, 95)),
            "max_px": float(err.max()),
        },
        "per_batch": per_batch,
    }


def solve_at_lag(
    records: list[str],
    raw_root: Path,
    detections_root: Path,
    lag_seconds: float,
    stride: int,
    x0: np.ndarray,
    args: argparse.Namespace,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    batches = build_batches(records, raw_root, detections_root, lag_seconds, args, stride=stride)
    if not batches:
        raise CalibrationFailure(
            "No usable checkerboard/pose batches were available for fitting.",
            {
                "reason_code": "no_batches_at_lag",
                "phase": "lag_scan",
                "lag_seconds": lag_seconds,
                "stride": stride,
                "max_reproj_px": args.max_reproj_px,
            },
        )
    obj = object_points(args)
    result = least_squares(
        residuals,
        x0,
        args=(batches, records, args, obj),
        bounds=bounds(args, len(records)),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=0,
    )
    st = stats_for_params(result.x, batches, records, args, obj)
    return result.x, {
        "lag_seconds": float(lag_seconds),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "median_px": st["overall"]["median_px"],
        "p90_px": st["overall"]["p90_px"],
        "stats": st,
    }


def fit_per_record(records: list[str], raw_root: Path, detections_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    axis = getattr(args, "image_y_axis", "down")
    if axis != "auto":
        return _fit_per_record_fixed_y_axis(records, raw_root, detections_root, args)

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate_axis in ("down", "up"):
        candidate_args = copy.copy(args)
        candidate_args.image_y_axis = candidate_axis
        emit_progress("fitting_image_y_" + candidate_axis, 0.65, records=records, imageYAxis=candidate_axis)
        try:
            result = _fit_per_record_fixed_y_axis(records, raw_root, detections_root, candidate_args)
        except CalibrationFailure as exc:
            failures.append(
                {
                    "image_y_axis": candidate_axis,
                    "reason": exc.reason,
                    "reason_code": exc.diagnostics.get("reason_code"),
                    "phase": exc.diagnostics.get("phase"),
                    "best_median_px_stats": exc.diagnostics.get("best_median_px_stats"),
                    "threshold_keep_counts": exc.diagnostics.get("threshold_keep_counts"),
                }
            )
            continue
        overall = result["stats"]["overall"]
        score = float(overall["median_px"]) + 0.25 * float(overall["p90_px"])
        result["image_y_axis"] = candidate_axis
        candidates.append({"score": score, "image_y_axis": candidate_axis, "result": result})

    if not candidates:
        raise CalibrationFailure(
            "Both image y-axis conventions failed to produce a calibration.",
            {
                "reason_code": "image_y_axis_auto_all_failed",
                "phase": "image_y_axis_auto",
                "records": records,
                "image_y_axis_failures": failures,
            },
        )

    candidates.sort(key=lambda row: row["score"])
    best_result = candidates[0]["result"]
    best_result["image_y_axis_auto_candidates"] = [
        {
            "image_y_axis": row["image_y_axis"],
            "score": row["score"],
            "median_px": row["result"]["stats"]["overall"]["median_px"],
            "p90_px": row["result"]["stats"]["overall"]["p90_px"],
            "best_lag_seconds": row["result"]["best_lag_seconds"],
            "kept_frames": row["result"]["order_summary"]["kept_frames"],
            "input_frames": row["result"]["order_summary"]["input_frames"],
        }
        for row in candidates
    ]
    if failures:
        best_result["image_y_axis_auto_failures"] = failures
    return best_result


def _fit_per_record_fixed_y_axis(records: list[str], raw_root: Path, detections_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    starts = [
        initial_params(records, raw_root, detections_root, args, intr_scale=1.0),
        initial_params(records, raw_root, detections_root, args, intr_scale=0.90),
        initial_params(records, raw_root, detections_root, args, intr_scale=0.78),
    ]
    lag_values = np.round(np.arange(args.lag_min_ms, args.lag_max_ms + 0.5 * args.coarse_step_ms, args.coarse_step_ms) / 1000.0, 6)
    best = None
    coarse_rows = []
    coarse_skipped = []
    for start_idx, start in enumerate(starts):
        x = start
        for lag in lag_values:
            try:
                x_fit, row = solve_at_lag(records, raw_root, detections_root, float(lag), args.coarse_stride, x, args, max_nfev=80)
            except CalibrationFailure as exc:
                if exc.diagnostics.get("reason_code") == "no_batches_at_lag":
                    coarse_skipped.append({"start_index": start_idx, "lag_seconds": float(lag), **exc.diagnostics})
                    continue
                raise
            row["start_index"] = start_idx
            coarse_rows.append(row)
            score = row["median_px"] + 0.25 * row["p90_px"]
            if best is None or score < best["score"]:
                best = {"score": score, "lag": float(lag), "x": x_fit, "row": row}
            x = x_fit
    if best is None:
        raise CalibrationFailure(
            "No scanned lag produced any usable frame batches.",
            {
                "reason_code": "no_overlap_lag",
                "phase": "lag_scan",
                "records": records,
                "coarse_stride": args.coarse_stride,
                "fine_stride": args.fine_stride,
                "lag_min_ms": args.lag_min_ms,
                "lag_max_ms": args.lag_max_ms,
                "coarse_step_ms": args.coarse_step_ms,
                "coarse_scan_rows": coarse_rows,
                "coarse_scan_skipped": coarse_skipped,
            },
        )

    fine_lags = np.round(
        np.arange(
            best["lag"] - args.fine_window_ms / 1000.0,
            best["lag"] + args.fine_window_ms / 1000.0 + 0.5 * args.fine_step_ms / 1000.0,
            args.fine_step_ms / 1000.0,
        ),
        6,
    )
    fine_rows = []
    fine_skipped = []
    x = best["x"]
    for lag in fine_lags:
        try:
            x_fit, row = solve_at_lag(records, raw_root, detections_root, float(lag), args.fine_stride, x, args, max_nfev=100)
        except CalibrationFailure as exc:
            if exc.diagnostics.get("reason_code") == "no_batches_at_lag":
                fine_skipped.append({"lag_seconds": float(lag), **exc.diagnostics})
                continue
            raise
        fine_rows.append(row)
        score = row["median_px"] + 0.25 * row["p90_px"]
        if score < best["score"]:
            best = {"score": score, "lag": float(lag), "x": x_fit, "row": row}
        x = x_fit

    lag = float(best["lag"])
    corrected_orders: dict[tuple[str, str, int], str] | None = None
    corrected_keep: set[tuple[str, str, int]] | None = None
    order_summaries = []
    params = best["x"]
    obj = object_points(args)
    for iteration in range(args.order_iterations):
        all_batches = build_batches(records, raw_root, detections_root, lag, args, stride=1)
        if not all_batches:
            raise CalibrationFailure(
                "No usable frames survived the initial reprojection and pose-time filters.",
                order_failure_diagnostics(
                    records,
                    args,
                    lag,
                    phase="corner_order_initial",
                    iteration=iteration,
                    all_batches=all_batches,
                    corrected_batches=[],
                    order_rows=[],
                    order_summary={"input_frames": 0, "kept_frames": 0, "rot180_frames": 0, "identity_frames": 0, "discarded_frames": 0},
                    selected_row=best.get("row") if isinstance(best, dict) else None,
                ),
            )
        order_rows = classify_frame_orders(params, all_batches, records, args, obj)
        corrected_orders, corrected_keep, order_summary = summarize_frame_orders(order_rows, args.order_keep_threshold_px)
        corrected_batches = build_batches(
            records,
            raw_root,
            detections_root,
            lag,
            args,
            stride=1,
            corrected_orders=corrected_orders,
            corrected_keep=corrected_keep,
        )
        if not corrected_batches:
            raise CalibrationFailure(
                "All frames were rejected by the corner-order/reprojection quality gate.",
                order_failure_diagnostics(
                    records,
                    args,
                    lag,
                    phase="corner_order_refit",
                    iteration=iteration,
                    all_batches=all_batches,
                    corrected_batches=corrected_batches,
                    order_rows=order_rows,
                    order_summary=order_summary,
                    selected_row=best.get("row") if isinstance(best, dict) else None,
                ),
            )
        result = least_squares(
            residuals,
            params,
            args=(corrected_batches, records, args, obj),
            bounds=bounds(args, len(records)),
            loss="soft_l1",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=args.max_final_nfev,
            verbose=0,
        )
        stats_after = stats_for_params(result.x, corrected_batches, records, args, obj)
        order_summary.update(
            {
                "iteration": iteration,
                "cost": float(result.cost),
                "nfev": int(result.nfev),
                "median_px": stats_after["overall"]["median_px"],
                "p90_px": stats_after["overall"]["p90_px"],
            }
        )
        order_summaries.append(order_summary)
        params = result.x

    all_batches = build_batches(records, raw_root, detections_root, lag, args, stride=1)
    if not all_batches:
        raise CalibrationFailure(
            "No usable frames survived the initial reprojection and pose-time filters.",
            order_failure_diagnostics(
                records,
                args,
                lag,
                phase="final_corner_order_initial",
                iteration=None,
                all_batches=all_batches,
                corrected_batches=[],
                order_rows=[],
                order_summary={"input_frames": 0, "kept_frames": 0, "rot180_frames": 0, "identity_frames": 0, "discarded_frames": 0},
                selected_row=best.get("row") if isinstance(best, dict) else None,
            ),
        )
    final_order_rows = classify_frame_orders(params, all_batches, records, args, obj)
    corrected_orders, corrected_keep, final_order_summary = summarize_frame_orders(final_order_rows, args.order_keep_threshold_px)
    final_batches = build_batches(
        records,
        raw_root,
        detections_root,
        lag,
        args,
        stride=1,
        corrected_orders=corrected_orders,
        corrected_keep=corrected_keep,
    )
    final_batches, diverse_fit_summary = apply_diverse_batch_selection(final_batches, args)
    if not final_batches:
        raise CalibrationFailure(
            "All frames were rejected by the final corner-order/reprojection quality gate or diverse frame selector.",
            order_failure_diagnostics(
                records,
                args,
                lag,
                phase="final_corner_order_refit",
                iteration=None,
                all_batches=all_batches,
                corrected_batches=final_batches,
                order_rows=final_order_rows,
                order_summary=final_order_summary,
                selected_row=best.get("row") if isinstance(best, dict) else None,
            ),
        )
    final = least_squares(
        residuals,
        params,
        args=(final_batches, records, args, obj),
        bounds=bounds(args, len(records)),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=args.max_final_nfev,
        verbose=0,
    )
    final_stats = stats_for_params(final.x, final_batches, records, args, obj)
    final_reproj_rows = reprojection_errors(final.x, final_batches, records, args, obj)
    parsed = parse_params(final.x, records, args)
    per_record = {}
    for record in records:
        pose = parsed["records"][record]
        t_unity_world_board = pose_to_matrix(pose["board_rvec"], pose["board_t"])
        t_board_unity_world = np.linalg.inv(t_unity_world_board)
        t_world_board = unity_transform_matrix_to_pc(t_unity_world_board)
        t_board_world = np.linalg.inv(t_world_board)
        twb = transform_doc(t_world_board, PC_WORLD_FRAME)
        tbw = transform_doc(t_board_world, PC_WORLD_FRAME)
        twb_unity = transform_doc(t_unity_world_board, UNITY_WORLD_FRAME)
        tbw_unity = transform_doc(t_board_unity_world, UNITY_WORLD_FRAME)
        per_record[record] = {
            "coordinate_frame": PC_WORLD_FRAME,
            "raw_trajectory_frame": UNITY_WORLD_FRAME,
            "world_frame_conversion": WORLD_FRAME_CONVERSION,
            "T_world_board": twb,
            "T_board_world": tbw,
            "quest_world_origin_in_board_m": tbw["translation_m"],
            "T_unity_world_board": twb_unity,
            "T_board_unity_world": tbw_unity,
            "unity_quest_world_origin_in_board_m": tbw_unity["translation_m"],
        }

    final_order_summary.update(
        {
            "keep_threshold_px": args.order_keep_threshold_px,
            "iterations": args.order_iterations,
            "iteration_summaries": order_summaries,
        }
    )
    return {
        "model": args.model,
        "coordinate_frame": PC_WORLD_FRAME,
        "raw_trajectory_frame": UNITY_WORLD_FRAME,
        "world_frame_conversion": WORLD_FRAME_CONVERSION,
        "image_y_axis": getattr(args, "image_y_axis", "down"),
        "image_y_projection": (
            "pixel_y = -camera_y / camera_z for top-left/y-down video rows"
            if getattr(args, "image_y_axis", "down") == "down"
            else "pixel_y = +camera_y / camera_z for bottom-left/y-up or vertically flipped video rows"
        ),
        "best_lag_seconds": lag,
        "left_intrinsics_fxfycxcy": parsed["left_intr"].tolist(),
        "right_intrinsics_fxfycxcy": parsed["right_intr"].tolist(),
        "left_radial_k1k2": parsed["left_dist"].tolist(),
        "right_radial_k1k2": parsed["right_dist"].tolist(),
        "corner_order_policy": (
            "per-frame prefer red-near-corner anchor to resolve identity vs rot180; "
            "fallback to lower median reprojection error when the red anchor is missing or ambiguous; discard weak frames; refit"
        ),
        "red_anchor_policy": {
            "enabled": not getattr(args, "disable_red_anchor", False),
            "target_index": getattr(args, "red_anchor_target_index", None),
            "resolved_target_index": getattr(args, "resolved_red_anchor_target_index", None),
            "auto_target": getattr(args, "red_anchor_target_index", None) is None,
            "global_summary": getattr(args, "red_anchor_global_summary", None),
            "frame_requirement": "optional; any reliable red anchor frame can orient the global 180-degree pair, and frames without red fall back to reprojection/trajectory matching",
            "radius_grid_spacing": RED_ANCHOR_RADIUS_GRID_SPACING,
            "min_pixels": RED_ANCHOR_MIN_PIXELS,
            "min_ratio": RED_ANCHOR_MIN_RATIO,
            "min_best_second_ratio": RED_ANCHOR_MIN_BEST_SECOND_RATIO,
        },
        "order_summary": final_order_summary,
        "diverse_frame_selection": {
            "enabled": not getattr(args, "disable_diverse_frame_selection", False),
            "detection_max_frames_per_side": int(args.max_diverse_detection_frames_per_side),
            "fit": diverse_fit_summary,
            "translation_scale_m": float(args.diverse_translation_scale_m),
            "rotation_scale_deg": float(args.diverse_rotation_scale_deg),
            "min_score": float(args.diverse_min_score),
        },
        "final_cost": float(final.cost),
        "final_status": int(final.status),
        "final_message": final.message,
        "final_nfev": int(final.nfev),
        "stats": final_stats,
        "per_record_calibration": per_record,
        "coarse_scan": coarse_rows,
        "coarse_scan_skipped": coarse_skipped,
        "fine_scan": fine_rows,
        "fine_scan_skipped": fine_skipped,
        "frame_order_classification": final_order_rows,
        "frame_reprojection_errors": final_reproj_rows,
    }


def classify_frame_orders(
    params: np.ndarray,
    batches: list[Batch],
    records: list[str],
    args: argparse.Namespace,
    obj: np.ndarray,
) -> list[dict[str, Any]]:
    parsed = parse_params(params, records, args)
    y_sign = image_y_sign(args)
    order_180 = rot180_index(args)
    rows = []
    for batch in batches:
        pose = parsed["records"][batch.record]
        r_w_b = Rotation.from_rotvec(pose["board_rvec"]).as_matrix()
        p_world = obj @ r_w_b.T + pose["board_t"]
        r_c_w = batch.camera_rotations.inv().as_matrix()
        rel = p_world[None, :, :] - batch.camera_positions[:, None, :]
        p_camera = np.einsum("nij,nmj->nmi", r_c_w, rel)
        pred = project_points(p_camera, parsed[f"{batch.side}_intr"], parsed[f"{batch.side}_dist"], y_sign)
        err_identity = np.linalg.norm(pred - batch.corners, axis=-1)
        err_rot180 = np.linalg.norm(pred - batch.corners[:, order_180, :], axis=-1)
        med_identity = np.median(err_identity, axis=1)
        med_rot180 = np.median(err_rot180, axis=1)
        for local_i, (frame_index, time_s, e_i, e_r) in enumerate(zip(batch.frame_indices, batch.times, med_identity, med_rot180)):
            reproj_order = "rot180" if e_r < e_i else "identity"
            anchor_order = str(batch.red_anchor_orders[local_i]) if local_i < len(batch.red_anchor_orders) else ""
            order_source = "red_anchor" if anchor_order in ("identity", "rot180") else "reprojection"
            order = anchor_order if order_source == "red_anchor" else reproj_order
            rows.append(
                {
                    "record": batch.record,
                    "side": batch.side,
                    "frame_index": int(frame_index),
                    "unity_time": float(time_s),
                    "identity_median_px": float(e_i),
                    "rot180_median_px": float(e_r),
                    "best_order": order,
                    "best_median_px": float(e_r if order == "rot180" else e_i),
                    "improvement_px": float(e_i - e_r),
                    "order_source": order_source,
                    "reprojection_order": reproj_order,
                    "red_anchor_observed_index": int(batch.red_anchor_observed_indices[local_i]) if local_i < len(batch.red_anchor_observed_indices) else -1,
                    "red_anchor_target_index": int(batch.red_anchor_target_indices[local_i]) if local_i < len(batch.red_anchor_target_indices) else -1,
                    "red_anchor_best_score": float(batch.red_anchor_best_scores[local_i]) if local_i < len(batch.red_anchor_best_scores) else 0.0,
                    "red_anchor_score_ratio": float(batch.red_anchor_score_ratios[local_i]) if local_i < len(batch.red_anchor_score_ratios) else 0.0,
                }
            )
    return rows


def summarize_frame_orders(
    rows: list[dict[str, Any]],
    keep_threshold_px: float,
) -> tuple[dict[tuple[str, str, int], str], set[tuple[str, str, int]], dict[str, int]]:
    orders = {}
    keep = set()
    summary = {
        "input_frames": len(rows),
        "kept_frames": 0,
        "rot180_frames": 0,
        "identity_frames": 0,
        "discarded_frames": 0,
        "red_anchor_frames": 0,
        "reprojection_frames": 0,
    }
    for row in rows:
        key = (row["record"], row["side"], int(row["frame_index"]))
        if row["best_median_px"] > keep_threshold_px:
            summary["discarded_frames"] += 1
            continue
        keep.add(key)
        orders[key] = row["best_order"]
        summary["kept_frames"] += 1
        if row.get("order_source") == "red_anchor":
            summary["red_anchor_frames"] += 1
        else:
            summary["reprojection_frames"] += 1
        if row["best_order"] == "rot180":
            summary["rot180_frames"] += 1
        else:
            summary["identity_frames"] += 1
    return orders, keep, summary


def order_failure_diagnostics(
    records: list[str],
    args: argparse.Namespace,
    lag_seconds: float,
    phase: str,
    iteration: int | None,
    all_batches: list[Batch],
    corrected_batches: list[Batch],
    order_rows: list[dict[str, Any]],
    order_summary: dict[str, int],
    selected_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best_values = np.asarray([row["best_median_px"] for row in order_rows], dtype=float) if order_rows else np.asarray([], dtype=float)
    identity_values = np.asarray([row["identity_median_px"] for row in order_rows], dtype=float) if order_rows else np.asarray([], dtype=float)
    rot180_values = np.asarray([row["rot180_median_px"] for row in order_rows], dtype=float) if order_rows else np.asarray([], dtype=float)
    thresholds = [10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0]
    keep_counts = {
        f"{threshold:.0f}": int(np.sum(best_values <= threshold))
        for threshold in thresholds
    }
    sorted_rows = sorted(order_rows, key=lambda row: (row["best_median_px"], row["record"], row["side"], row["frame_index"])) if order_rows else []
    preview = sorted_rows[:12]
    preview_rows = [
        {
            "record": row["record"],
            "side": row["side"],
            "frame_index": row["frame_index"],
            "unity_time": row["unity_time"],
                    "best_order": row["best_order"],
                    "order_source": row.get("order_source"),
                    "reprojection_order": row.get("reprojection_order"),
                    "best_median_px": row["best_median_px"],
                    "identity_median_px": row["identity_median_px"],
                    "rot180_median_px": row["rot180_median_px"],
                    "improvement_px": row["improvement_px"],
                    "red_anchor_observed_index": row.get("red_anchor_observed_index"),
                    "red_anchor_target_index": row.get("red_anchor_target_index"),
                    "red_anchor_best_score": row.get("red_anchor_best_score"),
                    "red_anchor_score_ratio": row.get("red_anchor_score_ratio"),
                }
                for row in preview
            ]
    per_batch = []
    for batch in all_batches:
        per_batch.append(
            {
                "record": batch.record,
                "side": batch.side,
                "frames": int(len(batch.frame_indices)),
                "time_span_seconds": float(batch.times[-1] - batch.times[0]) if len(batch.times) >= 2 else 0.0,
                "first_frame_index": int(batch.frame_indices[0]) if len(batch.frame_indices) else None,
                "last_frame_index": int(batch.frame_indices[-1]) if len(batch.frame_indices) else None,
            }
        )
    reason_code = "order_gate_rejected_all_frames" if order_summary.get("kept_frames", 0) == 0 else "order_gate_rejected_remaining_frames"
    diagnostics: dict[str, Any] = {
        "reason_code": reason_code,
        "phase": phase,
        "iteration": iteration,
        "records": records,
        "lag_seconds": lag_seconds,
        "square_size_m": args.square_size,
        "pattern": [args.pattern_cols, args.pattern_rows],
        "image_y_axis": getattr(args, "image_y_axis", "down"),
        "max_reproj_px": args.max_reproj_px,
        "order_keep_threshold_px": args.order_keep_threshold_px,
        "selected_fit": selected_row,
        "order_summary": {
            **order_summary,
            "kept_ratio": (order_summary["kept_frames"] / order_summary["input_frames"]) if order_summary.get("input_frames") else 0.0,
        },
        "threshold_keep_counts": keep_counts,
        "best_median_px_stats": stats(best_values),
        "identity_median_px_stats": stats(identity_values),
        "rot180_median_px_stats": stats(rot180_values),
        "frame_preview": preview_rows,
        "batch_preview": per_batch,
        "corrected_batch_count": len(corrected_batches),
    }
    return diagnostics


def compact_failure(failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "reason": failure.get("reason"),
        "reason_code": failure.get("reason_code"),
        "phase": failure.get("phase"),
        "lag_seconds": failure.get("lag_seconds"),
        "image_y_axis": failure.get("image_y_axis"),
        "order_summary": failure.get("order_summary"),
        "threshold_keep_counts": failure.get("threshold_keep_counts"),
        "best_median_px_stats": failure.get("best_median_px_stats"),
        "frame_preview": failure.get("frame_preview"),
    }


def pose_to_matrix(rvec: np.ndarray, t: np.ndarray) -> np.ndarray:
    rot = Rotation.from_rotvec(rvec)
    t_world_board = np.eye(4)
    t_world_board[:3, :3] = rot.as_matrix()
    t_world_board[:3, 3] = t
    return t_world_board


def pose_to_matrices(rvec: np.ndarray, t: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    t_world_board = pose_to_matrix(rvec, t)
    t_board_world = np.linalg.inv(t_world_board)
    return transform_doc(t_world_board, UNITY_WORLD_FRAME), transform_doc(t_board_world, UNITY_WORLD_FRAME)


def transform_doc(mat: np.ndarray, coordinate_frame: str | None = None) -> dict[str, Any]:
    rotation = Rotation.from_matrix(mat[:3, :3])
    qx, qy, qz, qw = [float(value) for value in rotation.as_quat()]
    payload = {
        "matrix_4x4": mat.tolist(),
        "translation_m": mat[:3, 3].tolist(),
        "quaternion_xyzw": [qx, qy, qz, qw],
        "quaternion_wxyz": [qw, qx, qy, qz],
        "rotation_matrix": mat[:3, :3].tolist(),
    }
    if coordinate_frame:
        payload["coordinateFrame"] = coordinate_frame
    return payload


def stats(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"min": None, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_report(final: dict[str, Any], path: Path) -> None:
    overall = final["stats"]["overall"]
    order = final["order_summary"]
    lines = [
        "# 25mm Record Calibration",
        "",
        "Inputs: recorded passthrough videos, PassthroughCameraAccess.GetCameraPose trajectories, fixed checkerboard.",
        "",
        f"Records: {', '.join(final['records'])}",
        f"Checkerboard: {final['pattern'][0]}x{final['pattern'][1]} inner corners, {final['square_size_m'] * 1000:.1f} mm squares.",
        f"Model: `{final['model']}`",
        f"Image y-axis: `{final.get('image_y_axis', 'down')}`",
        f"Best lag: {final['best_lag_seconds'] * 1000:.3f} ms",
        f"Frames kept after corner-order check: {order['kept_frames']} / {order['input_frames']}",
        f"Rot180 corrected frames: {order['rot180_frames']}",
        f"Red-anchor ordered frames: {order.get('red_anchor_frames', 0)}",
        "",
        "## Reprojection",
        "",
        f"- Median: {overall['median_px']:.2f} px",
        f"- P90: {overall['p90_px']:.2f} px",
        f"- P95: {overall['p95_px']:.2f} px",
        "",
        "## Intrinsics",
        "",
        f"- Left fxfycxcy: {json.dumps(final['left_intrinsics_fxfycxcy'])}",
        f"- Right fxfycxcy: {json.dumps(final['right_intrinsics_fxfycxcy'])}",
        "",
    ]
    diverse = final.get("diverse_frame_selection") if isinstance(final.get("diverse_frame_selection"), dict) else {}
    fit_diverse = diverse.get("fit") if isinstance(diverse.get("fit"), dict) else {}
    if diverse:
        lines += [
            "## Diverse Frame Selection",
            "",
            f"- Enabled: {bool(diverse.get('enabled'))}",
            f"- Detection max per side: {diverse.get('detection_max_frames_per_side')}",
            f"- Fit frames: {fit_diverse.get('selected_frames')} / {fit_diverse.get('input_frames')}",
            f"- Pose scales: {diverse.get('translation_scale_m')} m, {diverse.get('rotation_scale_deg')} deg",
            "",
        ]
    if final.get("image_y_axis_auto_candidates"):
        lines += [
            "## Image Y-Axis Auto Candidates",
            "",
        ]
        for row in final["image_y_axis_auto_candidates"]:
            lines.append(
                "- "
                f"{row['image_y_axis']}: score {row['score']:.2f}, "
                f"median {row['median_px']:.2f} px, p90 {row['p90_px']:.2f} px, "
                f"kept {row['kept_frames']} / {row['input_frames']}"
            )
        lines.append("")
    lines += [
        "## Coordinate Frame",
        "",
        f"- Exported world frame: `{final.get('coordinate_frame') or PC_WORLD_FRAME}`",
        f"- Raw trajectory frame: `{final.get('raw_trajectory_frame') or UNITY_WORLD_FRAME}`",
        "- Conversion: `pc = [unity.x, unity.y, -unity.z]`",
        "",
        "## Per-Record T_world_board",
        "",
        "`T_world_board` is in the exported PC right-handed world frame. "
        "`T_unity_world_board` keeps the raw Unity fit for diagnostics.",
        "",
    ]
    for record, row in final["per_record_calibration"].items():
        lines += [
            f"### {record}",
            "",
            "```json",
            json.dumps(row["T_world_board"], indent=2),
            "```",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_result(final: dict[str, Any]) -> dict[str, Any]:
    return {
        "records": final["records"],
        "square_size_m": final["square_size_m"],
        "model": final["model"],
        "coordinate_frame": final.get("coordinate_frame"),
        "raw_trajectory_frame": final.get("raw_trajectory_frame"),
        "world_frame_conversion": final.get("world_frame_conversion"),
        "image_y_axis": final.get("image_y_axis"),
        "best_lag_seconds": final["best_lag_seconds"],
        "left_intrinsics_fxfycxcy": final["left_intrinsics_fxfycxcy"],
        "right_intrinsics_fxfycxcy": final["right_intrinsics_fxfycxcy"],
        "stats_overall": final["stats"]["overall"],
        "order_summary": {
            key: final["order_summary"][key]
            for key in (
                "input_frames",
                "kept_frames",
                "rot180_frames",
                "identity_frames",
                "discarded_frames",
                "red_anchor_frames",
                "reprojection_frames",
            )
            if key in final["order_summary"]
        },
        "diverse_frame_selection": final.get("diverse_frame_selection"),
        "per_record_t_world_board_translation_m": {
            record: row["T_world_board"]["translation_m"]
            for record, row in final["per_record_calibration"].items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
