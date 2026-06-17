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

