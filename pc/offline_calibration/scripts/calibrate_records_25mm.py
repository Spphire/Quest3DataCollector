from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


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


class CalibrationFailure(RuntimeError):
    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


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
        "Each recording may have its own Unity world origin, so each record gets its own T_world_board."
    )
    result["records"] = records
    result["pattern"] = [args.pattern_cols, args.pattern_rows]
    result["square_size_m"] = args.square_size
    result["detection_summary"] = detection_summary

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
    best_payload = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        meta = by_index.get(frame_index)
        if meta is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners, method = detect_corners(gray, pattern)
            if found and corners is not None:
                rvec, tvec, reproj = solve_pose(corners, k, obj)
                frame_indices.append(frame_index)
                unity_times.append(float(meta["unityTimestampSeconds"]))
                camera_times.append(float(meta["cameraTimestampSeconds"]))
                corners_all.append(corners)
                rvecs.append(rvec)
                tvecs.append(tvec)
                reproj_errors.append(reproj)
                methods.append(method)
                if best_payload is None or reproj < best_payload["reproj"]:
                    best_payload = {
                        "frame": frame.copy(),
                        "frame_index": frame_index,
                        "corners": corners,
                        "rvec": rvec,
                        "tvec": tvec,
                        "reproj": reproj,
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
        camera_matrix=k,
        dist_coeffs=np.zeros(5, dtype=np.float64),
    )

    overlay_path = None
    if best_payload is not None:
        overlay = draw_overlay(best_payload["frame"], best_payload["corners"], best_payload["rvec"], best_payload["tvec"], k, pattern)
        label = f"{record_dir.name} {side} 25mm frame {best_payload['frame_index']} err {best_payload['reproj']:.2f}px"
        cv2.putText(overlay, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        overlay_path = out_dir / f"{record_dir.name}_{side}_best_overlay_25mm.jpg"
        cv2.imwrite(str(overlay_path), overlay)

    reproj = np.asarray(reproj_errors, dtype=float)
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
        "npz": str(npz_path.relative_to(ROOT)),
        "best_overlay": str(overlay_path.relative_to(ROOT)) if overlay_path else None,
    }


def draw_overlay(frame: np.ndarray, corners: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    out = frame.copy()
    cv2.drawChessboardCorners(out, pattern, corners.reshape(-1, 1, 2).astype(np.float32), True)
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
                    "npz": str(path.relative_to(ROOT)),
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
            if corrected_orders is not None:
                for local_i, global_i in enumerate(idx):
                    frame = int(det["frame_indices"][global_i])
                    if corrected_orders.get((record, side, frame)) == "rot180":
                        corners[local_i] = corners[local_i, order_180, :]
            batches.append(
                Batch(
                    record=record,
                    side=side,
                    frame_indices=np.asarray(det["frame_indices"], dtype=int)[idx],
                    times=times,
                    corners=corners,
                    camera_positions=cam_pos,
                    camera_rotations=cam_rot,
                )
            )
    return batches


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
        left_dirs = lr.apply(rays(np.asarray(left["corners"][il], dtype=float), k_left))
        right_dirs = rr.apply(rays(np.asarray(right["corners"][ir], dtype=float), k_right))
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
        lower += [-np.inf, -np.inf, -np.inf, -5.0, 0.0, -5.0]
        upper += [np.inf, np.inf, np.inf, 5.0, 3.0, 5.0]
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
    if not final_batches:
        raise CalibrationFailure(
            "All frames were rejected by the final corner-order/reprojection quality gate.",
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
        twb, tbw = pose_to_matrices(pose["board_rvec"], pose["board_t"])
        per_record[record] = {
            "T_world_board": twb,
            "T_board_world": tbw,
            "quest_world_origin_in_board_m": tbw["translation_m"],
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
        "corner_order_policy": "per-frame choose identity vs rot180 by lower median reprojection error; discard weak frames; refit",
        "order_summary": final_order_summary,
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
        for frame_index, time_s, e_i, e_r in zip(batch.frame_indices, batch.times, med_identity, med_rot180):
            order = "rot180" if e_r < e_i else "identity"
            rows.append(
                {
                    "record": batch.record,
                    "side": batch.side,
                    "frame_index": int(frame_index),
                    "unity_time": float(time_s),
                    "identity_median_px": float(e_i),
                    "rot180_median_px": float(e_r),
                    "best_order": order,
                    "best_median_px": float(min(e_i, e_r)),
                    "improvement_px": float(e_i - e_r),
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
    }
    for row in rows:
        key = (row["record"], row["side"], int(row["frame_index"]))
        if row["best_median_px"] > keep_threshold_px:
            summary["discarded_frames"] += 1
            continue
        keep.add(key)
        orders[key] = row["best_order"]
        summary["kept_frames"] += 1
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
            "best_median_px": row["best_median_px"],
            "identity_median_px": row["identity_median_px"],
            "rot180_median_px": row["rot180_median_px"],
            "improvement_px": row["improvement_px"],
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


def pose_to_matrices(rvec: np.ndarray, t: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    rot = Rotation.from_rotvec(rvec)
    t_world_board = np.eye(4)
    t_world_board[:3, :3] = rot.as_matrix()
    t_world_board[:3, 3] = t
    t_board_world = np.linalg.inv(t_world_board)
    return transform_doc(t_world_board), transform_doc(t_board_world)


def transform_doc(mat: np.ndarray) -> dict[str, Any]:
    rotation = Rotation.from_matrix(mat[:3, :3])
    return {
        "matrix_4x4": mat.tolist(),
        "translation_m": mat[:3, 3].tolist(),
        "quaternion_xyzw": rotation.as_quat().tolist(),
        "rotation_matrix": mat[:3, :3].tolist(),
    }


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
        "## Per-Record T_world_board",
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
        "image_y_axis": final.get("image_y_axis"),
        "best_lag_seconds": final["best_lag_seconds"],
        "left_intrinsics_fxfycxcy": final["left_intrinsics_fxfycxcy"],
        "right_intrinsics_fxfycxcy": final["right_intrinsics_fxfycxcy"],
        "stats_overall": final["stats"]["overall"],
        "order_summary": {
            key: final["order_summary"][key]
            for key in ("input_frames", "kept_frames", "rot180_frames", "identity_frames", "discarded_frames")
        },
        "per_record_t_world_board_translation_m": {
            record: row["T_world_board"]["translation_m"]
            for record, row in final["per_record_calibration"].items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
