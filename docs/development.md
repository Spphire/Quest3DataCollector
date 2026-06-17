# Development Notes

## Working model

This repository is a thin packaging of the Quest 3 chessboard calibration workflow:

- Quest app records telemetry and local trajectory/video.
- PC receiver writes a synchronized copy and optional replay artifacts.
- Replay and diagnostics live in the PC receiver web UI.

## Key invariants

- `recordId` identifies one logical recording.
- `sampleIndex` is the primary join key between Quest trajectory and PC samples.
- `gazePoint3DWorld` is considered a world-space gaze point, not a camera reprojection result.
- PC replay is read-only.

## Runtime paths

- PC recordings: `pc_recordings/<recordId>/`
- Pulled Quest records: `raw/<recordId>/`
- Calibration outputs: `outputs/pc_live_calibration/`
- Quest B-button PC calibration raw records: `raw/record_pc_calib_*/`
- Optional Flexiv/RealSense capture inside a B-button record: `raw/<recordId>/robot_realsense/`

## Common commands

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py receive --host 0.0.0.0 --port 9100 --visualize --visualize-port 8765 --no-open-browser
py pc\offline_calibration\scripts\quest_pc_receiver.py analyze --quest-record pc\offline_calibration\raw\record_YYYYMMDD_HHMMSS --pc-session pc\offline_calibration\pc_recordings\record_YYYYMMDD_HHMMSS
py pc\offline_calibration\scripts\quest_pc_receiver.py diagnose-gaze-depth --pc-session pc\offline_calibration\pc_recordings\record_YYYYMMDD_HHMMSS --quest-record pc\offline_calibration\raw\record_YYYYMMDD_HHMMSS
```

## Notes for future edits

- Keep replay payloads bounded to one record.
- Keep artifact serving constrained to safe leaf paths and small files.
- Avoid changing Unity source conventions without matching PC-side readers.

## Flexiv + RealSense Integration

The `quest3-chessboard-flexiv` branch adds an optional PC-side bridge for the table-fixed checkerboard plus Flexiv robot plus end-mounted RealSense workflow.

### Remote deployment

The project is deployed on the lab PC at:

```bash
/ssd1/shenyibo/Quest3DataCollector
```

Use the Python 3.12 virtual environment:

```bash
/ssd1/shenyibo/Quest3DataCollector/.venv312/bin/python
```

Installed runtime packages include:

- `numpy`
- `scipy`
- `opencv-python`
- `pyrealsense2`
- `pillow`
- `flexivrdk==1.7.0`

The Flexiv robot software observed through Flexiv Elements is `RobotControlApp v3.9.3`. Flexiv's compatibility table maps robot software `v3.9` to RDK `v1.7`, so the project venv should import the pip package `flexivrdk==1.7.0`. Do not add the old `/home/lvjun/flexiv_rdk/lib_py` RDK v1.4 path to this venv, because that version targets older robot software and can shadow the correct pip package.

Verification:

```bash
/ssd1/shenyibo/Quest3DataCollector/.venv312/bin/python - <<'PY'
import flexivrdk
print(flexivrdk.__version__)
print(flexivrdk.__file__)
PY
```

Expected:

```text
1.7.0
.../.venv312/lib/python3.12/site-packages/flexivrdk/__init__.py
```

### Hardware state found on 2026-06-17

RealSense devices detected on the remote PC:

- `244222073667`: Intel RealSense D435I, intended default for the end-mounted camera.
- `750612070265`: Intel RealSense D435, previously used as a fixed/third-view camera.

Robot serial candidates tested:

- `Rizon4-H6uDOq`
- `Rizon4-062713`

With RDK v1.7 loaded, both still stopped at robot discovery (`Searching for [...]`). At that point the remaining likely causes are Flexiv Remote mode/Ethernet not enabled, wrong serial string, firewall, or network path. The receiver UI keeps the robot SN editable so the operator can retry without changing code.

### Receiver command

```bash
/ssd1/shenyibo/Quest3DataCollector/.venv312/bin/python \
  /ssd1/shenyibo/Quest3DataCollector/pc/offline_calibration/scripts/quest_pc_receiver.py \
  receive \
  --host 0.0.0.0 \
  --port 9100 \
  --visualize \
  --visualize-host 0.0.0.0 \
  --visualize-port 8765 \
  --no-open-browser
```

Useful options:

- `--realsense-serial <serial>` sets the default end-mounted RealSense.
- `--flexiv-robot-sn <sn>` sets the default robot serial shown in the UI.
- `--flexiv-pose-field flange_pose|tcp_pose` selects the robot state pose used as `T_base_ee`.
- `--robot-capture-interval 0.35` controls how often robot/RealSense samples are taken during B-button calibration.
- `--no-flexiv-realsense` hides/disables the robot bridge.
- `--no-robot-hand-eye` records robot/RealSense data but skips the automatic hand-eye solve.

### Web workflow

Open:

```text
http://<pc-ip>:8765/
```

Then:

1. Select the end-mounted RealSense camera in the `Flexiv / RealSense` panel.
2. Enter the Flexiv robot SN.
3. Click `Connect Robot`.
4. Press Quest B once to start PC calibration capture.
5. Move the Quest for Quest/checkerboard pose diversity.
6. Move the robot/end camera for robot/checkerboard pose diversity.
7. Press Quest B again to stop.

The PC receiver writes:

- Quest videos/frame metadata/trajectory under `raw/<recordId>/`.
- Robot states, `jointpose`, `T_base_ee`, and RealSense images under `raw/<recordId>/robot_realsense/`.
- Quest/checkerboard result under `outputs/pc_live_calibration/<recordId>/`.
- Robot hand-eye result under `raw/<recordId>/robot_realsense/robot_hand_eye_result.json`.

When both Quest/checkerboard and robot hand-eye succeed, the bridge computes:

- `T_world_board` from the Quest calibration.
- `T_base_board` from the robot/RealSense hand-eye solve.
- `T_world_base = T_world_board * inverse(T_base_board)`.

This unifies Quest world, checkerboard, and robot base in the live/replay visualization.

### Implementation notes

- `flexiv_realsense_bridge.py` is intentionally PC-only; Unity does not need to know about Flexiv or RealSense.
- The bridge uses `flexivrdk.Robot(sn).states()` and records the configured `flange_pose` or `tcp_pose` as `[x, y, z, qw, qx, qy, qz]`.
- RealSense intrinsics come from `pyrealsense2` color stream metadata.
- The board is fixed at 11x8 inner corners, 25 mm square size.
- Hand-eye calibration solves `T_ee_realsense` and `T_base_board` from repeated end-camera observations of the fixed board.
- Replay keeps Quest axes and translates the view near the board origin; robot EE samples are drawn as white points with local RGB axes.
