# Usage

## Start the PC receiver

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py receive --host 0.0.0.0 --port 9100 --visualize --visualize-port 8765 --no-open-browser
```

Open:

- `http://127.0.0.1:8765/`
- `http://127.0.0.1:8765/recordings`

## Quest controls

- `A`: start/stop a normal Quest recording
- `B`: start/stop a calibration recording
- The headset dot turns on while recording is active

## Calibration recording

Calibration recording writes the Quest frames and trajectory, and it can also attach PC-side robot/camera data when the receiver is connected to the Flexiv/RealSense stack.

Recommended flow:

1. Start the PC receiver.
2. Open the live viewer and confirm telemetry is flowing.
3. Connect the robot in the `Flexiv / RealSense` panel.
4. Press `B` to start calibration recording.
5. Move the Quest and the checkerboard through the capture volume.
6. Press `B` again to stop.

What gets saved in the calibration record:

- Quest trajectory and left/right videos
- PC robot states
- end-camera MP4 and frame metadata
- third-camera MP4 and frame metadata
- calibration outputs under `outputs/pc_live_calibration/<recordId>/`

## Formal recording

Use `A` for the regular recording flow. When robot capture is enabled, the PC session also stores:

- robot end-effector pose
- joint pose
- gripper state
- end-camera pose
- end-camera MP4
- fixed third-camera MP4

The replay page shows the record list, a 3D view, the robot poses, and the recorded videos.

## Replay

Open `http://127.0.0.1:8765/recordings`, select a record, then:

- drag the canvas to orbit
- use the mouse wheel to zoom
- scrub the timeline to inspect poses and gaze samples

## Output folders

- `pc/offline_calibration/pc_recordings/`
- `pc/offline_calibration/raw/`
- `pc/offline_calibration/outputs/`

