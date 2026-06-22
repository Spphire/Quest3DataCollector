from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


APPEARANCE_ANCHOR_MIN_CONTRAST = 18.0


def checkerboard_corner_indices(cols: int, rows: int) -> list[int]:
    return [0, int(cols) - 1, (int(rows) - 1) * int(cols), int(rows) * int(cols) - 1]


def rot180_index(cols: int, rows: int) -> np.ndarray:
    return np.arange(int(rows) * int(cols)).reshape(int(rows), int(cols))[::-1, ::-1].reshape(-1)


def appearance_anchor_order_for_observed(observed_index: int, target_index: int, cols: int, rows: int) -> str | None:
    if observed_index < 0 or target_index < 0:
        return None
    if int(observed_index) == int(target_index):
        return "identity"
    order_180 = rot180_index(cols, rows)
    if int(order_180[int(observed_index)]) == int(target_index):
        return "rot180"
    return None


def detect_checkerboard_appearance_anchor(
    image: np.ndarray,
    corners: np.ndarray,
    cols: int,
    rows: int,
    *,
    min_contrast: float = APPEARANCE_ANCHOR_MIN_CONTRAST,
) -> dict[str, Any]:
    points = np.asarray(corners, dtype=float).reshape(int(rows), int(cols), 2)
    corner_indices = checkerboard_corner_indices(cols, rows)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else np.asarray(image, dtype=np.uint8)
    samples = _corner_inner_square_samples(gray, points, int(cols), int(rows))
    luminance = {str(index): float(samples[index]["luminance"]) for index in corner_indices}
    centers = {str(index): samples[index]["center"] for index in corner_indices}
    valid = {str(index): bool(samples[index]["valid"]) for index in corner_indices}
    pixels = {str(index): int(samples[index]["pixels"]) for index in corner_indices}
    primary_pair = [0, int(rows) * int(cols) - 1]
    secondary_pair = [int(cols) - 1, (int(rows) - 1) * int(cols)]
    pair_choice = _best_contrast_pair(samples, [primary_pair, secondary_pair], float(min_contrast))
    if pair_choice is None:
        return {
            "ok": False,
            "observedIndex": -1,
            "targetIndex": -1,
            "order": None,
            "bestIndex": -1,
            "secondIndex": -1,
            "contrast": 0.0,
            "minContrast": float(min_contrast),
            "luminanceByCorner": luminance,
            "validByCorner": valid,
            "pixelsByCorner": pixels,
            "centersByCorner": centers,
            "cornerIndices": [int(v) for v in corner_indices],
            "primaryPair": [int(v) for v in primary_pair],
            "secondaryPair": [int(v) for v in secondary_pair],
            "reason": "weak_or_incomplete_checker_appearance",
        }
    first, second, contrast = pair_choice
    observed = int(first if samples[first]["luminance"] <= samples[second]["luminance"] else second)
    other = int(second if observed == first else first)
    turn_signs = _corner_turn_signs(points, int(cols), int(rows))
    target_categories = _target_corner_categories(int(cols), int(rows))
    observed_category = {
        "turnSign": int(turn_signs.get(observed, 0)),
        "innerSquare": "dark",
    }
    target = _target_index_for_category(target_categories, observed_category)
    order = appearance_anchor_order_for_observed(observed, target, cols, rows)
    return {
        "ok": bool(order in ("identity", "rot180")),
        "observedIndex": observed,
        "targetIndex": target,
        "order": order,
        "orderReason": "appearance_dark_inner_square_in_turn_pair",
        "bestIndex": observed,
        "secondIndex": other,
        "contrast": float(contrast),
        "minContrast": float(min_contrast),
        "luminanceByCorner": luminance,
        "validByCorner": valid,
        "pixelsByCorner": pixels,
        "centersByCorner": centers,
        "cornerIndices": [int(v) for v in corner_indices],
        "primaryPair": [int(v) for v in primary_pair],
        "secondaryPair": [int(v) for v in secondary_pair],
        "selectedPair": [int(first), int(second)],
        "observedCategory": observed_category,
        "turnSignByCorner": {str(index): int(turn_signs.get(index, 0)) for index in corner_indices},
        "targetCategoryByCorner": {
            str(index): {"turnSign": int(value["turnSign"]), "innerSquare": value["innerSquare"]}
            for index, value in target_categories.items()
        },
        "polarity": "darker_inner_square",
        "reason": "ok",
    }


def draw_checkerboard_appearance_anchor_overlay(
    out: np.ndarray,
    corners: np.ndarray,
    anchor: dict[str, Any] | None,
    cols: int,
    rows: int,
) -> None:
    if not anchor:
        return
    points = np.asarray(corners, dtype=float).reshape(-1, 2)
    luminance = anchor.get("luminanceByCorner") if isinstance(anchor.get("luminanceByCorner"), dict) else {}
    centers = anchor.get("centersByCorner") if isinstance(anchor.get("centersByCorner"), dict) else {}
    observed = int(anchor.get("observedIndex", -1))
    selected_pair = {int(v) for v in anchor.get("selectedPair", []) if isinstance(v, (int, float))}
    for index in checkerboard_corner_indices(cols, rows):
        x, y = points[index]
        color = (255, 80, 40) if index == observed else ((255, 180, 60) if index in selected_pair else (180, 180, 180))
        cv2.circle(out, (int(round(x)), int(round(y))), 8, color, 2, cv2.LINE_AA)
        center = centers.get(str(index))
        if isinstance(center, list) and len(center) >= 2 and all(math.isfinite(float(v)) for v in center[:2]):
            cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
            cv2.circle(out, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.line(out, (int(round(x)), int(round(y))), (cx, cy), color, 1, cv2.LINE_AA)
        label = f"a{index}:{float(luminance.get(str(index), 0.0)):.0f}"
        cv2.putText(out, label, (int(round(x)) + 6, int(round(y)) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)


def _best_contrast_pair(
    samples: dict[int, dict[str, Any]],
    pairs: list[list[int]],
    min_contrast: float,
) -> tuple[int, int, float] | None:
    best: tuple[int, int, float] | None = None
    for pair in pairs:
        first, second = int(pair[0]), int(pair[1])
        if not samples[first]["valid"] or not samples[second]["valid"]:
            continue
        contrast = abs(float(samples[first]["luminance"]) - float(samples[second]["luminance"]))
        if contrast < min_contrast:
            continue
        if best is None or contrast > best[2]:
            best = (first, second, float(contrast))
    return best


def _corner_inner_square_samples(gray: np.ndarray, points: np.ndarray, cols: int, rows: int) -> dict[int, dict[str, Any]]:
    specs = {
        0: ((0, 0), (0, 1), (1, 0)),
        cols - 1: ((0, cols - 1), (0, cols - 2), (1, cols - 1)),
        (rows - 1) * cols: ((rows - 1, 0), (rows - 1, 1), (rows - 2, 0)),
        rows * cols - 1: ((rows - 1, cols - 1), (rows - 1, cols - 2), (rows - 2, cols - 1)),
    }
    result: dict[int, dict[str, Any]] = {}
    for index, (origin_rc, x_rc, y_rc) in specs.items():
        origin = points[origin_rc]
        x_vec = points[x_rc] - origin
        y_vec = points[y_rc] - origin
        center = origin + 0.5 * (x_vec + y_vec)
        radius = max(2.0, 0.18 * min(float(np.linalg.norm(x_vec)), float(np.linalg.norm(y_vec))))
        luminance, pixels = _sample_circle_mean(gray, center, radius)
        result[int(index)] = {
            "valid": bool(pixels > 0 and math.isfinite(luminance)),
            "luminance": float(luminance) if math.isfinite(luminance) else float("nan"),
            "pixels": int(pixels),
            "center": [float(center[0]), float(center[1])],
            "radiusPx": float(radius),
        }
    return result


def _corner_turn_signs(points: np.ndarray, cols: int, rows: int) -> dict[int, int]:
    specs = {
        0: ((0, 0), (0, 1), (1, 0)),
        cols - 1: ((0, cols - 1), (0, cols - 2), (1, cols - 1)),
        (rows - 1) * cols: ((rows - 1, 0), (rows - 1, 1), (rows - 2, 0)),
        rows * cols - 1: ((rows - 1, cols - 1), (rows - 1, cols - 2), (rows - 2, cols - 1)),
    }
    horizontal_is_long = int(cols) >= int(rows)
    result: dict[int, int] = {}
    for index, (origin_rc, horizontal_rc, vertical_rc) in specs.items():
        origin = points[origin_rc]
        horizontal = points[horizontal_rc] - origin
        vertical = points[vertical_rc] - origin
        long_vec = horizontal if horizontal_is_long else vertical
        short_vec = vertical if horizontal_is_long else horizontal
        det = float(long_vec[0] * short_vec[1] - long_vec[1] * short_vec[0])
        result[int(index)] = 1 if det >= 0 else -1
    return result


def _target_corner_categories(cols: int, rows: int) -> dict[int, dict[str, Any]]:
    grid = np.asarray([[[c, r] for c in range(cols)] for r in range(rows)], dtype=float)
    signs = _corner_turn_signs(grid, cols, rows)
    inner_square_coords = {
        0: (1, 1),
        cols - 1: (cols - 1, 1),
        (rows - 1) * cols: (1, rows - 1),
        rows * cols - 1: (cols - 1, rows - 1),
    }
    reference_parity = sum(inner_square_coords[0]) % 2
    return {
        int(index): {
            "turnSign": int(signs[index]),
            "innerSquare": "dark" if (sum(coord) % 2) == reference_parity else "light",
        }
        for index, coord in inner_square_coords.items()
    }


def _target_index_for_category(target_categories: dict[int, dict[str, Any]], category: dict[str, Any]) -> int:
    for index, target_category in target_categories.items():
        if (
            int(target_category.get("turnSign", 0)) == int(category.get("turnSign", 0))
            and target_category.get("innerSquare") == category.get("innerSquare")
        ):
            return int(index)
    return -1


def _sample_circle_mean(gray: np.ndarray, center: np.ndarray, radius: float) -> tuple[float, int]:
    height, width = gray.shape[:2]
    x, y = float(center[0]), float(center[1])
    if not math.isfinite(x) or not math.isfinite(y) or radius <= 0:
        return float("nan"), 0
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(width, int(math.ceil(x + radius + 1)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(height, int(math.ceil(y + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return float("nan"), 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) * (xx - x) + (yy - y) * (yy - y) <= radius * radius
    values = np.asarray(gray[y0:y1, x0:x1][mask], dtype=np.float32)
    if values.size == 0:
        return float("nan"), 0
    return float(np.mean(values)), int(values.size)
