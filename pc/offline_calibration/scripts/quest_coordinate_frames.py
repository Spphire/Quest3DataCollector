from __future__ import annotations

import math
from typing import Any

import numpy as np


UNITY_WORLD_FRAME = "unity_world_lh_y_up_z_forward"
PC_WORLD_FRAME = "pc_world_rh_y_up_z_back"
WORLD_FRAME_CONVERSION = {
    "name": "unity_lh_to_pc_rh_z_flip",
    "position": "pc = [unity.x, unity.y, -unity.z]",
    "quaternion_wxyz": "pc = [w, -x, -y, z]",
    "transform": "T_pc = diag(1,1,-1,1) @ T_unity @ diag(1,1,-1,1)",
}

UNITY_TO_PC_REFLECTION_3X3 = np.diag([1.0, 1.0, -1.0])
UNITY_TO_PC_REFLECTION_4X4 = np.diag([1.0, 1.0, -1.0, 1.0])


def is_pc_world_frame(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text and ("pc_world" in text or "right_handed" in text or "rh" in text))


def is_unity_world_frame(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text and ("unity" in text or "left_handed" in text or "lh" in text))


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def vec3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = [finite_float(value[index]) for index in range(3)]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result]


def quat_wxyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    result = [finite_float(value[index]) for index in range(4)]
    if any(item is None for item in result):
        return None
    return normalize_quaternion_wxyz([float(item) for item in result])


def normalize_quaternion_wxyz(value: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in value[:4]))
    if norm <= 1e-12 or not math.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [float(item) / norm for item in value[:4]]


def unity_vec3_to_pc(value: Any) -> list[float] | None:
    parsed = vec3(value)
    if parsed is None:
        return None
    return [parsed[0], parsed[1], -parsed[2]]


def pc_vec3_to_unity(value: Any) -> list[float] | None:
    return unity_vec3_to_pc(value)


def unity_quaternion_wxyz_to_pc(value: Any) -> list[float] | None:
    parsed = quat_wxyz(value)
    if parsed is None:
        return None
    return normalize_quaternion_wxyz([parsed[0], -parsed[1], -parsed[2], parsed[3]])


def pc_quaternion_wxyz_to_unity(value: Any) -> list[float] | None:
    return unity_quaternion_wxyz_to_pc(value)


def unity_pose_array_to_pc(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 7:
        return None
    position = unity_vec3_to_pc(value[:3])
    rotation = unity_quaternion_wxyz_to_pc(value[3:7])
    if position is None or rotation is None:
        return None
    return [position[0], position[1], position[2], rotation[0], rotation[1], rotation[2], rotation[3]]


def matrix_from_transform_payload(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    matrix = payload.get("matrix_4x4")
    if isinstance(matrix, list) and len(matrix) >= 4:
        try:
            return np.asarray([[float(row[col]) for col in range(4)] for row in matrix[:4]], dtype=float)
        except Exception:
            return None
    translation = vec3(payload.get("translation_m"))
    quat = quat_wxyz(payload.get("quaternion_wxyz"))
    if quat is None and isinstance(payload.get("quaternion_xyzw"), list):
        qxyzw = payload.get("quaternion_xyzw")
        quat = quat_wxyz([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]) if len(qxyzw) >= 4 else None
    if translation is None or quat is None:
        return None
    result = np.eye(4, dtype=float)
    result[:3, :3] = quaternion_wxyz_to_matrix(quat)
    result[:3, 3] = np.asarray(translation, dtype=float)
    return result


def transform_payload_from_matrix(matrix: np.ndarray, coordinate_frame: str | None = None) -> dict[str, Any]:
    mat = np.asarray(matrix, dtype=float).reshape(4, 4)
    qwxyz = matrix_to_quaternion_wxyz(mat[:3, :3])
    payload: dict[str, Any] = {
        "matrix_4x4": [[float(value) for value in row] for row in mat[:4, :4]],
        "translation_m": [float(value) for value in mat[:3, 3]],
        "rotation_matrix": [[float(value) for value in row] for row in mat[:3, :3]],
        "quaternion_wxyz": qwxyz,
        "quaternion_xyzw": [qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]],
    }
    if coordinate_frame:
        payload["coordinateFrame"] = coordinate_frame
    return payload


def unity_transform_matrix_to_pc(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float).reshape(4, 4)
    return UNITY_TO_PC_REFLECTION_4X4 @ mat @ UNITY_TO_PC_REFLECTION_4X4


def pc_transform_matrix_to_unity(matrix: np.ndarray) -> np.ndarray:
    return unity_transform_matrix_to_pc(matrix)


def unity_transform_payload_to_pc(payload: Any) -> dict[str, Any] | None:
    matrix = matrix_from_transform_payload(payload)
    if matrix is None:
        return None
    return transform_payload_from_matrix(unity_transform_matrix_to_pc(matrix), PC_WORLD_FRAME)


def ensure_pc_transform_payload(payload: Any, source_frame: Any = None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    frame = payload.get("coordinateFrame") or source_frame
    matrix = matrix_from_transform_payload(payload)
    if matrix is None:
        return None
    if is_pc_world_frame(frame):
        return transform_payload_from_matrix(matrix, PC_WORLD_FRAME)
    return transform_payload_from_matrix(unity_transform_matrix_to_pc(matrix), PC_WORLD_FRAME)


def quaternion_wxyz_to_matrix(q: list[float]) -> np.ndarray:
    qw, qx, qy, qz = normalize_quaternion_wxyz(q)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return normalize_quaternion_wxyz([float(qw), float(qx), float(qy), float(qz)])
