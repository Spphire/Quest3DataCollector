from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from quest_coordinate_frames import (
    PC_WORLD_FRAME,
    UNITY_WORLD_FRAME,
    WORLD_FRAME_CONVERSION,
    ensure_pc_transform_payload,
    matrix_from_transform_payload,
    transform_payload_from_matrix,
    unity_quaternion_wxyz_to_pc,
    unity_pose_array_to_pc,
    unity_vec3_to_pc,
)


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def vec3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def quat(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return normalize_quat([float(value[0]), float(value[1]), float(value[2]), float(value[3])])
    except (TypeError, ValueError):
        return None


def normalize_quat(q: list[float] | None) -> list[float] | None:
    if q is None:
        return None
    norm = sum(value * value for value in q) ** 0.5
    if norm <= 1e-12:
        return None
    return [value / norm for value in q]


def average_quat(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None:
        return normalize_quat(b)
    if b is None:
        return normalize_quat(a)
    if sum(left * right for left, right in zip(a, b)) < 0:
        b = [-value for value in b]
    return normalize_quat([(left + right) * 0.5 for left, right in zip(a, b)])


def pose_from_list(value: Any) -> dict[str, Any]:
    converted = unity_pose_array_to_pc(value)
    position = vec3(converted[:3]) if isinstance(converted, list) and len(converted) >= 3 else None
    rotation = quat(converted[3:7]) if isinstance(converted, list) and len(converted) >= 7 else None
    return {
        "ok": position is not None,
        "p": position,
        "q": rotation,
        "source": "recorded",
    }


def controller_rotation(controller: dict[str, Any]) -> list[float] | None:
    return quat(unity_quaternion_wxyz_to_pc(controller.get("rotation")))


def subtract_origin(p: list[float] | None, origin: list[float]) -> list[float] | None:
    if p is None:
        return None
    return [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]


def translated_matrix(matrix: Any, origin: list[float]) -> list[list[float]]:
    result: list[list[float]] = []
    if not isinstance(matrix, list) or len(matrix) < 4:
        raise SystemExit("Calibration snapshot has no 4x4 T_world_board matrix")
    for row in matrix[:4]:
        if not isinstance(row, list) or len(row) < 4:
            raise SystemExit("Calibration snapshot has a malformed T_world_board matrix")
        result.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    for index in range(3):
        result[index][3] -= origin[index]
    return result


def load_session(session_dir: Path) -> dict[str, Any]:
    samples_path = session_dir / "pc_samples.jsonl"
    summary_path = session_dir / "pc_session_summary.json"
    snapshot_path = session_dir / "pc_calibration_snapshot.json"
    samples = read_jsonl(samples_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else summary.get("calibrationSnapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("T_world_board"):
        raise SystemExit(f"Missing calibration snapshot in {session_dir}")

    frame = snapshot.get("coordinateFrame") or snapshot.get("coordinate_frame")
    t_world_board = ensure_pc_transform_payload(snapshot.get("T_world_board"), frame)
    if t_world_board is None:
        raise SystemExit(f"Calibration snapshot has no usable T_world_board in {session_dir}")
    board_matrix_world = t_world_board["matrix_4x4"]
    board_matrix = matrix_from_transform_payload(t_world_board)
    if board_matrix is not None:
        snapshot = dict(snapshot)
        snapshot["coordinateFrame"] = PC_WORLD_FRAME
        snapshot["rawTrajectoryFrame"] = snapshot.get("rawTrajectoryFrame") or UNITY_WORLD_FRAME
        snapshot["worldFrameConversion"] = snapshot.get("worldFrameConversion") or WORLD_FRAME_CONVERSION
        snapshot["T_world_board"] = t_world_board
        snapshot["T_board_world"] = transform_payload_from_matrix(np.linalg.inv(board_matrix), PC_WORLD_FRAME)
    board_origin_world = [
        float(board_matrix_world[0][3]),
        float(board_matrix_world[1][3]),
        float(board_matrix_world[2][3]),
    ]
    board_matrix_display = translated_matrix(board_matrix_world, board_origin_world)

    display_samples: list[dict[str, Any]] = []
    for row in samples:
        left_eye = pose_from_list(row.get("leftEyePose"))
        right_eye = pose_from_list(row.get("rightEyePose"))
        head: dict[str, Any] = {
            "ok": left_eye["ok"] and right_eye["ok"],
            "p": None,
            "q": None,
            "source": "eye_midpoint",
        }
        if left_eye["ok"] and right_eye["ok"]:
            head["p"] = [
                (left_eye["p"][0] + right_eye["p"][0]) * 0.5,
                (left_eye["p"][1] + right_eye["p"][1]) * 0.5,
                (left_eye["p"][2] + right_eye["p"][2]) * 0.5,
            ]
            head["q"] = average_quat(left_eye["q"], right_eye["q"])
        elif left_eye["ok"]:
            head = dict(left_eye)
        elif right_eye["ok"]:
            head = dict(right_eye)

        left_controller = row.get("leftController") if isinstance(row.get("leftController"), dict) else {}
        right_controller = row.get("rightController") if isinstance(row.get("rightController"), dict) else {}
        left_controller_p = unity_vec3_to_pc(left_controller.get("position"))
        right_controller_p = unity_vec3_to_pc(right_controller.get("position"))
        gaze = unity_vec3_to_pc(row.get("gazePoint3DWorld"))

        display_samples.append(
            {
                "sampleIndex": row.get("sampleIndex"),
                "recordingTimestampSeconds": row.get("recordingTimestampSeconds"),
                "isRecording": bool(row.get("isRecording", True)),
                "recordId": row.get("recordId"),
                "head": {
                    "p": subtract_origin(head["p"], board_origin_world),
                    "q": head["q"],
                    "ok": bool(head["ok"]),
                    "source": head["source"],
                },
                "leftEye": {
                    "p": subtract_origin(left_eye["p"], board_origin_world),
                    "q": left_eye["q"],
                    "ok": bool(left_eye["ok"]),
                    "source": left_eye["source"],
                },
                "rightEye": {
                    "p": subtract_origin(right_eye["p"], board_origin_world),
                    "q": right_eye["q"],
                    "ok": bool(right_eye["ok"]),
                    "source": right_eye["source"],
                },
                "left": {
                    "p": subtract_origin(left_controller_p, board_origin_world),
                    "q": controller_rotation(left_controller),
                    "ok": bool(left_controller.get("hasPose")) and left_controller_p is not None,
                    "source": left_controller.get("source") or "missing",
                },
                "right": {
                    "p": subtract_origin(right_controller_p, board_origin_world),
                    "q": controller_rotation(right_controller),
                    "ok": bool(right_controller.get("hasPose")) and right_controller_p is not None,
                    "source": right_controller.get("source") or "missing",
                },
                "gaze": {
                    "p": subtract_origin(gaze, board_origin_world),
                    "ok": gaze is not None,
                    "source": row.get("gazePoint3DSource") or row.get("gazeSource"),
                },
            }
        )

    return {
        "sessionDir": str(session_dir),
        "summary": summary,
        "snapshot": snapshot,
        "coordinateMode": "pc_right_handed_world_axes_translated_to_board_origin",
        "coordinateFrame": PC_WORLD_FRAME,
        "rawTrajectoryFrame": UNITY_WORLD_FRAME,
        "worldFrameConversion": WORLD_FRAME_CONVERSION,
        "boardOriginWorld": board_origin_world,
        "boardMatrix": board_matrix_display,
        "samples": display_samples,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quest Recording Replay</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0d1012;
  --panel: #121820;
  --line: #2a3544;
  --text: #e8edf3;
  --muted: #91a0b0;
  --board: #f2c94c;
  --head: #f1ecd0;
  --eye: #a8ff9a;
  --left: #21c7e8;
  --right: #ff62b8;
  --gaze: #ffd65c;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text); font: 13px/1.35 system-ui, sans-serif; }
#app { display: grid; grid-template-columns: minmax(0, 1fr) 340px; height: 100vh; }
canvas { width: 100%; height: 100%; display: block; background: #080a0c; touch-action: none; }
aside { border-left: 1px solid var(--line); padding: 12px; overflow: auto; background: #10161d; }
h1 { margin: 0 0 6px; font-size: 18px; }
button { background: #172232; color: var(--text); border: 1px solid #2b3a4f; border-radius: 6px; padding: 6px 10px; }
input[type=range], input[type=checkbox] { accent-color: #83aefc; }
.sub, .kv, .tiny { color: var(--muted); }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
.scrub { width: 100%; }
.pill { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; }
.dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
.section { margin-top: 10px; border-top: 1px solid var(--line); padding-top: 10px; }
.kv { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; margin-top: 8px; }
.ok { color: #9df09d; font-weight: 700; }
.rec { color: #ff8c96; font-weight: 700; }
@media (max-width: 900px) {
  #app { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) auto; }
  aside { border-left: 0; border-top: 1px solid var(--line); max-height: 44vh; }
}
</style>
</head>
<body>
<div id="app">
  <canvas id="view"></canvas>
  <aside>
    <h1>Quest Recording Replay</h1>
    <div class="sub" id="sessionLabel"></div>
    <div class="row">
      <button id="playBtn">Pause</button>
      <button id="resetBtn">Reset</button>
      <label class="pill"><input id="centerBoard" type="checkbox" checked> center board</label>
    </div>
    <input id="scrub" class="scrub" type="range" min="0" max="0" value="0" step="1">
    <div class="row tiny">
      <span id="timeLabel">0.000s</span>
      <span id="sampleLabel">sample 0</span>
      <span id="recordingLabel"></span>
    </div>
    <div class="row">
      <span class="pill"><span class="dot" style="background: var(--head)"></span>Head</span>
      <span class="pill"><span class="dot" style="background: var(--eye)"></span>Eyes</span>
      <span class="pill"><span class="dot" style="background: var(--left)"></span>Left</span>
      <span class="pill"><span class="dot" style="background: var(--right)"></span>Right</span>
      <span class="pill"><span class="dot" style="background: var(--gaze)"></span>Gaze</span>
    </div>
    <div class="section">
      <div class="ok">Calibration snapshot</div>
      <div class="kv" id="snapKv"></div>
    </div>
    <div class="section tiny">Drag to orbit, wheel to zoom. Coordinates keep Quest world axes and are translated by the board origin.</div>
  </aside>
</div>
<script>
const DATA = __DATA__;
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const scrub = document.getElementById('scrub');
const playBtn = document.getElementById('playBtn');
const resetBtn = document.getElementById('resetBtn');
const centerBoard = document.getElementById('centerBoard');
const sessionLabel = document.getElementById('sessionLabel');
const timeLabel = document.getElementById('timeLabel');
const sampleLabel = document.getElementById('sampleLabel');
const recordingLabel = document.getElementById('recordingLabel');
const snapKv = document.getElementById('snapKv');

const state = {
  playing: true,
  speed: 1,
  t: 0,
  idx: 0,
  yaw: -0.82,
  pitch: -0.34,
  distance: 1.45,
  target: [0, 0, 0],
  dragging: false,
  lastPointer: [0, 0],
  trails: { head: [], left: [], right: [], gaze: [] }
};

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function currentSample() {
  return DATA.samples[Math.max(0, Math.min(DATA.samples.length - 1, state.idx))];
}

function updateScrub() {
  scrub.max = Math.max(0, DATA.samples.length - 1);
  scrub.value = state.idx;
}

function setSnapshotInfo() {
  const s = DATA.snapshot || {};
  const origin = DATA.boardOriginWorld || [];
  const kv = [
    ['record', s.recordId || 'n/a'],
    ['mode', DATA.coordinateMode || 'n/a'],
    ['origin world', origin.length >= 3 ? origin.map(v => Number(v).toFixed(3)).join(', ') + ' m' : 'n/a'],
    ['image y', s.imageYAxis || 'n/a'],
    ['lag', Number.isFinite(s.bestLagSeconds) ? (s.bestLagSeconds * 1000).toFixed(1) + ' ms' : 'n/a'],
    ['median', Number.isFinite(s.medianReprojectionPx) ? s.medianReprojectionPx.toFixed(2) + ' px' : 'n/a'],
    ['board Z vs Y', Number.isFinite(s.boardNormalAbsAngleToWorldYDeg) ? s.boardNormalAbsAngleToWorldYDeg.toFixed(1) + ' deg' : 'n/a']
  ];
  snapKv.innerHTML = kv.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');
  sessionLabel.textContent = DATA.sessionDir || '';
}

function resetView() {
  state.yaw = -0.82;
  state.pitch = -0.34;
  state.target = [0, 0, 0];
  const points = boardPoints();
  for (const sample of DATA.samples) {
    for (const key of ['head', 'left', 'right', 'leftEye', 'rightEye', 'gaze']) {
      if (sample[key]?.p) points.push(sample[key].p);
    }
  }
  if (!points.length) {
    state.distance = 1.45;
    return;
  }
  const min = points[0].slice();
  const max = points[0].slice();
  for (const p of points) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  const center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
  if (!centerBoard.checked) state.target = center;
  let radius = 0.12;
  for (const p of points) radius = Math.max(radius, length(sub(p, state.target)));
  state.distance = Math.max(0.35, radius * 3.1);
}

function boardSize() {
  const snap = DATA.snapshot || {};
  const cols = Math.max(1, (snap.pattern?.[0] || 11) - 1);
  const rows = Math.max(1, (snap.pattern?.[1] || 8) - 1);
  const square = snap.squareSizeM || 0.025;
  return { width: cols * square, height: rows * square, cols, rows, square };
}

function boardPoints() {
  const m = DATA.boardMatrix;
  if (!m) return [];
  const size = boardSize();
  return [
    boardPoint(m, 0, 0, 0),
    boardPoint(m, size.width, 0, 0),
    boardPoint(m, size.width, size.height, 0),
    boardPoint(m, 0, size.height, 0)
  ];
}

function updateLabels() {
  const sample = currentSample();
  if (!sample) return;
  timeLabel.textContent = `${Number(sample.recordingTimestampSeconds || 0).toFixed(3)}s`;
  sampleLabel.textContent = `sample ${sample.sampleIndex ?? state.idx}`;
  recordingLabel.textContent = sample.isRecording ? 'REC' : 'LIVE';
  recordingLabel.className = sample.isRecording ? 'rec' : 'ok';
}

function render() {
  const rect = canvas.getBoundingClientRect();
  if (centerBoard.checked) state.target = [0, 0, 0];
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#080a0c';
  ctx.fillRect(0, 0, rect.width, rect.height);
  drawGroundGrid();
  drawWorldAxes();
  drawBoard();
  drawTrails();

  const sample = currentSample();
  if (!sample) return;
  const head = sample.head?.p;
  const gaze = sample.gaze?.p;
  if (head && gaze) drawLine(head, gaze, 'rgba(242,201,76,0.72)', 1.6);
  if (gaze) drawPoint(gaze, '#f2c94c', 4, 'gaze');
  drawPose(sample.head, '#f1ecd0', 'head', 0.06);
  drawPose(sample.leftEye, '#a8ff9a', 'L-eye', 0.04);
  drawPose(sample.rightEye, '#a8ff9a', 'R-eye', 0.04);
  drawPose(sample.left, '#21c7e8', 'L', 0.06);
  drawPose(sample.right, '#ff62b8', 'R', 0.06);
}

function drawGroundGrid() {
  const extent = 1.0;
  const step = 0.1;
  for (let v = -extent; v <= extent + 1e-6; v += step) {
    const alpha = Math.abs(v) < 1e-6 ? 0.24 : 0.10;
    drawLine([-extent, 0, v], [extent, 0, v], `rgba(255,255,255,${alpha})`, 1);
    drawLine([v, 0, -extent], [v, 0, extent], `rgba(255,255,255,${alpha})`, 1);
  }
}

function drawWorldAxes() {
  const origin = [0, 0, 0];
  drawLine(origin, [0.12, 0, 0], '#ff4545', 2.4);
  drawLine(origin, [0, 0.12, 0], '#42e875', 2.4);
  drawLine(origin, [0, 0, 0.12], '#4b7cff', 2.4);
}

function drawBoard() {
  const m = DATA.boardMatrix;
  if (!m) return;
  const size = boardSize();
  const corners = boardPoints();
  const projected = corners.map(project);
  if (projected.every(p => p.visible)) {
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.beginPath();
    ctx.moveTo(projected[0].x, projected[0].y);
    for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i].x, projected[i].y);
    ctx.closePath();
    ctx.fillStyle = '#f2c94c';
    ctx.fill();
    ctx.restore();
  }
  for (let col = 0; col <= size.cols; col++) {
    const x = col * size.square;
    drawLine(boardPoint(m, x, 0, 0), boardPoint(m, x, size.height, 0), 'rgba(242,201,76,0.42)', 1);
  }
  for (let row = 0; row <= size.rows; row++) {
    const y = row * size.square;
    drawLine(boardPoint(m, 0, y, 0), boardPoint(m, size.width, y, 0), 'rgba(242,201,76,0.42)', 1);
  }
  const o = boardPoint(m, 0, 0, 0);
  drawPoint(o, '#f2c94c', 5, 'board');
  drawLine(o, boardPoint(m, 0.08, 0, 0), '#ff4545', 3);
  drawLine(o, boardPoint(m, 0, 0.08, 0), '#42e875', 3);
  drawLine(o, boardPoint(m, 0, 0, 0.08), '#4b7cff', 3);
}

function drawTrails() {
  drawTrail(state.trails.head, 'rgba(241,236,208,0.35)');
  drawTrail(state.trails.left, 'rgba(33,199,232,0.45)');
  drawTrail(state.trails.right, 'rgba(255,98,184,0.45)');
  drawTrail(state.trails.gaze, 'rgba(242,201,76,0.35)');
}

function drawTrail(points, color) {
  let prev = null;
  for (const p of points) {
    if (prev) drawLine(prev, p, color, 1.1);
    prev = p;
  }
}

function drawPose(pose, color, label, axisScale) {
  if (!pose?.ok || !pose.p) return;
  const q = pose.q || [1, 0, 0, 0];
  drawLine(pose.p, add(pose.p, quatRotate(q, [axisScale, 0, 0])), '#ff4545', 2.1);
  drawLine(pose.p, add(pose.p, quatRotate(q, [0, axisScale, 0])), '#42e875', 2.1);
  drawLine(pose.p, add(pose.p, quatRotate(q, [0, 0, axisScale])), '#4b7cff', 2.1);
  drawPoint(pose.p, color, label === 'head' ? 6 : 5, label);
}

function drawPoint(p, color, radius, label) {
  const s = project(p);
  if (!s.visible) return;
  ctx.beginPath();
  ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth = 1;
  ctx.stroke();
  if (label) {
    ctx.fillStyle = '#dce4ec';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(label, s.x + radius + 4, s.y - radius - 2);
  }
}

function drawLine(a, b, color, width) {
  if (!a || !b) return;
  const pa = project(a);
  const pb = project(b);
  if (!pa.visible || !pb.visible) return;
  ctx.beginPath();
  ctx.moveTo(pa.x, pa.y);
  ctx.lineTo(pb.x, pb.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function project(p) {
  const rect = canvas.getBoundingClientRect();
  const x = p[0] - state.target[0];
  const y = p[1] - state.target[1];
  const z = p[2] - state.target[2];
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x1 = cy * x - sy * z;
  const z1 = sy * x + cy * z;
  const y2 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1 + state.distance;
  const focal = Math.min(rect.width, rect.height) * 0.92;
  return {
    x: rect.width * 0.5 + x1 * focal / Math.max(0.03, z2),
    y: rect.height * 0.5 - y2 * focal / Math.max(0.03, z2),
    visible: z2 > 0.03
  };
}

function boardPoint(m, x, y, z) {
  return [
    m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
    m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
    m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
  ];
}

function quatRotate(q, v) {
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx)
  ];
}

function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function length(a) { return Math.hypot(a[0], a[1], a[2]); }

canvas.addEventListener('pointerdown', event => {
  state.dragging = true;
  state.lastPointer = [event.clientX, event.clientY];
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (!state.dragging) return;
  const dx = event.clientX - state.lastPointer[0];
  const dy = event.clientY - state.lastPointer[1];
  state.lastPointer = [event.clientX, event.clientY];
  state.yaw += dx * 0.006;
  state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * 0.006));
});
canvas.addEventListener('pointerup', () => { state.dragging = false; });
canvas.addEventListener('pointercancel', () => { state.dragging = false; });
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  state.distance = Math.max(0.06, state.distance * Math.exp(event.deltaY * 0.001));
}, { passive: false });

playBtn.addEventListener('click', () => {
  state.playing = !state.playing;
  playBtn.textContent = state.playing ? 'Pause' : 'Play';
});
resetBtn.addEventListener('click', resetView);
centerBoard.addEventListener('change', resetView);
scrub.addEventListener('input', () => {
  state.idx = Math.max(0, Math.min(DATA.samples.length - 1, Number(scrub.value) || 0));
  state.t = DATA.samples[state.idx]?.recordingTimestampSeconds || 0;
  updateLabels();
});
window.addEventListener('resize', () => { resize(); resetView(); });

for (const sample of DATA.samples) {
  if (sample.head?.p) state.trails.head.push(sample.head.p);
  if (sample.left?.p) state.trails.left.push(sample.left.p);
  if (sample.right?.p) state.trails.right.push(sample.right.p);
  if (sample.gaze?.p) state.trails.gaze.push(sample.gaze.p);
}

setSnapshotInfo();
updateScrub();
resize();
resetView();
state.t = DATA.samples[0]?.recordingTimestampSeconds || 0;
updateLabels();
render();

let lastFrame = performance.now();
function tick(now) {
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if (state.playing && DATA.samples.length) {
    state.t += dt * state.speed;
    if (state.t > (DATA.samples[DATA.samples.length - 1].recordingTimestampSeconds || 0)) {
      state.t = DATA.samples[0].recordingTimestampSeconds || 0;
      state.idx = 0;
    }
    while (state.idx < DATA.samples.length - 1 && DATA.samples[state.idx + 1].recordingTimestampSeconds <= state.t) state.idx++;
    while (state.idx > 0 && DATA.samples[state.idx].recordingTimestampSeconds > state.t) state.idx--;
    scrub.value = state.idx;
  }
  updateLabels();
  render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def build_replay_html(session_dir: Path, output_path: Path) -> None:
    data = load_session(session_dir)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local HTML replay for a Quest PC recording.")
    parser.add_argument("--session", type=Path, required=True, help="PC recording folder.")
    parser.add_argument("--output", type=Path, help="Output HTML file.")
    args = parser.parse_args()
    session_dir = args.session.resolve()
    output = (args.output or (session_dir / "replay.html")).resolve()
    build_replay_html(session_dir, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
