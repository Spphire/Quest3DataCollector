# Quest3DataCollector

Quest 3 chessboard calibration and PC-side recording tools.

## Layout

- `pc/offline_calibration/scripts/quest_pc_receiver.py`: PC UDP receiver, live 3D viewer, PC record replay, gaze diagnostics, Quest B-button calibration receiver.
- `pc/offline_calibration/scripts/calibrate_records_25mm.py`: 11x8 inner-corner, 25 mm square Quest/checkerboard calibration.
- `pc/offline_calibration/scripts/flexiv_realsense_bridge.py`: optional Flexiv + end-mounted RealSense capture and hand-eye calibration bridge.
- `unity/Assets/EyeTracking/`: Quest recording, telemetry, and calibration scripts.
- `docs/usage.md`: operator quick start for recording, calibration, replay, and robot capture.
- `docs/development.md`: setup notes, runtime paths, and development workflow.

## PC Receiver

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py receive --host 0.0.0.0 --port 9100 --visualize --visualize-port 8765 --no-open-browser
```

Live viewer:

- `http://127.0.0.1:8765/`
- `http://127.0.0.1:8765/recordings`

Runtime artifacts are written under `pc/offline_calibration/pc_recordings/`, `pc/offline_calibration/raw/`, and `pc/offline_calibration/outputs/`.

For a step-by-step operator flow, see [docs/usage.md](docs/usage.md).

## Flexiv / RealSense

The receiver includes an optional Flexiv/RealSense panel. Select the end-mounted RealSense camera, enter the robot serial number, click `Connect Robot`, then use Quest B to start/stop PC calibration capture.

During B-button calibration, the PC records Quest frames/trajectory plus Flexiv robot state and RealSense images. After Quest/checkerboard calibration succeeds, it attempts the robot hand-eye solve and writes the result into the calibration record.

Right-controller robot motion is armed automatically when a robot session starts, and right-controller displacement can command bounded TCP offsets during B-button calibration. `Arm Motion` remains a fallback/debug action, and stopping the calibration disables motion commands.

Robot replay exposes both `T_display_tool_tcp` and `T_display_end_camera` in the live/replay viewer. `T_display_*` keeps Quest world rotation and translates the calibrated board near the origin for easier inspection.

The recorded robot session includes:

- Quest `gazePoint3DWorld`
- robot end-camera trajectory
- robot tool TCP trajectory
- end-camera MP4 video
- camera serials and intrinsics in `robot_realsense/cameras.json` and `capture_config.json`

The checkerboard calibration uses 11x8 inner corners with 25 mm squares. A single red marker near one board corner can be used as a global 180-degree orientation hint, but it is optional per frame and may appear in only one frame of a record.
