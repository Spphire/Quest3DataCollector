# Quest3DataCollector

Quest 3 chessboard calibration and PC-side recording tools.

## Layout

- `pc/offline_calibration/scripts/quest_pc_receiver.py`: PC UDP receiver, live 3D viewer, PC record replay, gaze diagnostics, teleop latency analysis, Quest B-button calibration receiver.
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

On the lab PC at `10.128.1.95`, use:

```bash
cd /ssd1/shenyibo/Quest3DataCollector
pc/offline_calibration/scripts/start_lab_receiver.sh --restart
```

## Flexiv / RealSense

The receiver includes an optional Flexiv/RealSense panel. Select the end-mounted RealSense camera, enter the robot serial number, click `Connect Robot`, then use Quest B to start/stop PC calibration capture.

During B-button calibration, the PC records Quest frames/trajectory plus Flexiv robot state and RealSense images. After Quest/checkerboard calibration succeeds, it attempts the robot hand-eye solve and writes the result into the calibration record.

Right-controller robot motion is armed automatically when a robot session starts, and right-controller displacement can command bounded TCP offsets during B-button calibration. `Arm Motion` remains a fallback/debug action, and stopping the calibration disables motion commands.

Robot replay exposes both `T_display_tool_tcp` and `T_display_end_camera` in the live/replay viewer. `T_display_*` keeps Quest world rotation and translates the calibrated board near the origin for easier inspection. Formal recordings with robot teleop also get `teleop_latency_analysis.json`, and the replay page shows command/controller to robot TCP latency.

The recorded robot session includes:

- Quest `gazePoint3DWorld`
- robot end-camera trajectory
- robot tool TCP trajectory
- end-camera MP4 video
- camera serials and intrinsics in `robot_realsense/cameras.json` and `capture_config.json`

## Recording Rate Contract

- `robot_realsense/robot_states.jsonl`: fixed-deadline Flexiv state sampling at `90 Hz` by default.
- `robot_realsense/video_frames.jsonl` and camera videos: `30 Hz` per RealSense role.
- `robot_realsense/samples.jsonl`: fixed-deadline `30 Hz` aligned training timeline using the latest Quest sample, 90 Hz robot state, and 30 Hz camera frames.
- `pc_samples.jsonl`: preserves every received formal Quest sample.
- `pc_telemetry_raw.jsonl` and `pc_controllers.csv`: preserve every received formal Quest sample; there is no second receiver-side rate limiter.

Audit a completed recording:

```bash
.venv312/bin/python pc/offline_calibration/scripts/quest_pc_receiver.py audit-performance \
  --pc-session pc/offline_calibration/pc_recordings/<record_id>
```

The default audit requires at least 95% of the 90/30 Hz targets, zero camera queue drops, zero fixed-sampler missed ticks, robot p95 scheduling lateness at most 6 ms, aligned-sample p95 lateness at most 16 ms, Quest source-age p95 at most 60 ms, aligned Quest source reuse at most 10%, Quest UDP sequence loss at most 0.1%, and UDP datagrams no larger than 1472 bytes.

The checkerboard calibration uses 11x8 inner corners with 25 mm squares. The 180-degree corner-order ambiguity is resolved from the board's black/white corner appearance first, then by identity-vs-rot180 reprojection or hand-eye residual matching when appearance is inconclusive.
