# Quest3DataCollector

Quest 3 chessboard calibration and PC-side recording tools.

## Layout

- `pc/offline_calibration/scripts/quest_pc_receiver.py`: PC UDP receiver, live 3D viewer, PC record replay, gaze diagnostics, Quest B-button calibration receiver.
- `pc/offline_calibration/scripts/calibrate_records_25mm.py`: 11x8 inner-corner, 25 mm square Quest/checkerboard calibration.
- `pc/offline_calibration/scripts/flexiv_realsense_bridge.py`: optional Flexiv + end-mounted RealSense capture and hand-eye calibration bridge.
- `unity/Assets/EyeTracking/`: Quest recording, telemetry, and calibration scripts.
- `docs/development.md`: setup notes, runtime paths, and development workflow.

## PC Receiver

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py receive --host 0.0.0.0 --port 9100 --visualize --visualize-port 8765 --no-open-browser
```

Live viewer:

- `http://127.0.0.1:8765/`
- `http://127.0.0.1:8765/recordings`

Runtime artifacts are written under `pc/offline_calibration/pc_recordings/`, `pc/offline_calibration/raw/`, and `pc/offline_calibration/outputs/`.

## Flexiv / RealSense

The receiver includes an optional Flexiv/RealSense panel. Select the end-mounted RealSense camera, enter the robot serial number, click `Connect Robot`, then use Quest B to start/stop PC calibration capture.

During B-button calibration, the PC records Quest frames/trajectory plus Flexiv robot state and RealSense images. After Quest/checkerboard calibration succeeds, it attempts the robot hand-eye solve and writes the result into the calibration record.
