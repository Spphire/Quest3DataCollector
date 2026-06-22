# Usage

This is the operator guide for the Quest 3 PC receiver, live viewer, calibration
recording, robot/RealSense capture, and replay pages.

## Quick Start on the Lab PC

The lab PC currently used for the live receiver is:

```text
10.128.0.227
```

Remote project root:

```bash
/ssd1/shenyibo/Quest3DataCollector
```

Open the live viewer from another PC on the same network:

```text
http://10.128.0.227:8765/
```

If the browser appears stale after a deployment, add a cache-busting query:

```text
http://10.128.0.227:8765/?reload=1
```

The current lab receiver command is:

```bash
cd /ssd1/shenyibo/Quest3DataCollector
nohup .venv312/bin/python pc/offline_calibration/scripts/quest_pc_receiver.py receive \
  --host 0.0.0.0 \
  --port 9100 \
  --visualize \
  --visualize-host 0.0.0.0 \
  --visualize-port 8765 \
  --no-open-browser \
  --flexiv-network-interface 192.168.2.108 \
  --realsense-serial 244222073667 \
  --third-realsense-serial 750612070265 \
  >> receiver.log 2>&1 &
```

Check the remote process:

```bash
pgrep -af 'pc/offline_calibration/scripts/quest_pc_receiver.py receive'
ss -ltnp | grep ':8765'
```

## Start the PC receiver

From the repository root:

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py receive --host 0.0.0.0 --port 9100 --visualize --visualize-port 8765 --no-open-browser
```

Open:

- `http://127.0.0.1:8765/` for the live 3D viewer and device controls
- `http://127.0.0.1:8765/recordings` for PC record replay

Useful receiver options:

- `--calibration-raw-root <path>` chooses where B-button calibration raw records are written.
- `--calibration-output-root <path>` chooses where Quest/checkerboard calibration outputs are written.
- `--pc-record-root <path>` chooses where A-button PC recordings are written.
- `--no-flexiv-realsense` hides and disables the robot/RealSense bridge.
- `--no-calibrate-after-pc-recording` records B-button raw data but skips automatic calibration after stop.

## Install the Quest App

Use this when installing the Unity app on a new Quest.

1. Connect the Quest by USB.
2. Put on the headset and accept the USB debugging / RSA fingerprint prompt.
3. Confirm ADB sees the headset as `device`, not `unauthorized`:

```powershell
adb devices -l
```

4. Install the APK:

```powershell
adb -s <quest-serial> install -r -d W:\lasertag-projs\Build\EyeTrackingBuild\EyeTrackingTest.apk
```

Current package name:

```text
com.Apricity.EyeTrackingTest
```

Optional launch command:

```powershell
adb -s <quest-serial> shell monkey -p com.Apricity.EyeTrackingTest 1
```

If the device stays `unauthorized`, reconnect the USB cable and accept the prompt
inside the headset. If no prompt appears, toggle developer mode / USB debugging
for the Quest and restart the ADB server:

```powershell
adb kill-server
adb start-server
adb devices -l
```

## Quest Controls

- `A`: start/stop a normal recording.
- `B`: start/stop a calibration recording.
- Right hand trigger: gripper open/close command when gripper control is enabled.
- Right hand side/grip trigger: hold to enable robot TCP teleoperation during an
  `A` normal recording after Quest-robot calibration is available.
- The headset recording indicator turns on while recording is active.

For `B` calibration recording, the PC first tries Flexiv's native
`FloatingCartesian()` free-drive primitive. If that primitive is unavailable or
not licensed, it falls back to the same Cartesian motion-force compliance loop
used by the iPhone calibration tool: low Cartesian impedance while recording,
then high-stiffness hold when recording stops.

For `A` normal recording, robot motion is gated by the right hand side/grip
trigger. Moving the right controller without holding that trigger records
controller poses but does not command the robot. Controller TCP commands are
mapped through the calibrated Quest-world-to-robot-base transform, and they pass
through a joint-limit guard: by default the PC stops sending teleop commands
when any Flexiv joint enters the configured buffer around the URDF soft joint
limits.

## Live Viewer

The live page shows Quest head pose, left/right controllers, gaze point, robot
poses, camera streams, calibration status, and preflight checks.

Before recording:

1. Confirm Quest telemetry is updating in the 3D view.
2. If using the robot, select the end-mounted RealSense and connect Flexiv.
3. Confirm both camera streams are available if they are needed for the run.
4. Confirm the preflight panel has no blocking failures.

The 3D view can be dragged to orbit and scrolled to zoom. Unity/Quest telemetry
arrives in raw Unity world coordinates (`x` right, `y` up, `z` forward). The PC
viewer, replay, and robot bridge display a derived right-handed Z-up frame with
`pc = [unity.z, -unity.x, unity.y]`; raw Unity samples are still stored for audit.
After calibration, the display translates the checkerboard near the origin while
keeping the converted world axes.

### Robot live visualization

Click `Connect Robot` in the `Flexiv / RealSense` panel before a robot run.
The live viewer polls `/robot/status` independently of the RealSense stream, so
the robot skeleton/mesh follows the latest Flexiv `jointPose` even when camera
streaming is stopped.

The robot status text should show:

- `robot: connected Rizon4-062713`
- `last sample: live robot status`
- `joint age: <small value>s`
- `joints: ...`
- `URDF FK vs flange: ...mm`

Use `joint age` as the quick health check. If it grows continuously, the page is
not receiving fresh robot state. Refresh the page, reconnect the robot, and
check `/robot/status`.

The robot model is served by:

```text
http://<pc-ip>:8765/robot/model
```

The lab build can be used to try different Flexiv URDF variants. At the time of
writing, the receiver is set to load `flexiv_Rizon4R_kinematics.urdf` from
`pc/offline_calibration/assets/urdf/` as a visualization experiment. The original
Rizon4 URDF is also kept in the same folder. A large `URDF FK vs flange` value
means the selected URDF does not match the robot state/pose well, even if the
mesh is visible.

## Calibration Recording

Use `B` for the calibration flow. A calibration record writes Quest frames and
trajectory, and it can also attach PC-side Flexiv/RealSense data when the bridge
is connected.

Recommended flow:

1. Start the PC receiver and open the live viewer.
2. Connect the robot in the `Flexiv / RealSense` panel when robot hand-eye is needed.
3. Press `B` once to start calibration recording.
4. Move the Quest/head through varied viewpoints of the 11x8 checkerboard.
5. If the robot bridge is connected, free-drag the robot/end camera through
   varied viewpoints while keeping the board visible.
6. Press `B` again to stop.
7. Wait for the PC-side calibration status to finish.

The checkerboard is 11x8 inner corners with 25 mm square size. A red marker near
one board corner is optional; one reliable red-anchored observation can resolve
the global 180-degree corner-order ambiguity, and frames without the red marker
can still be used by residual matching.

After a calibration recording stops, the PC pipeline uses pose-diverse frame
selection by default. It does not process every nearly identical frame. Quest
calibration selects passthrough frames with different camera poses, while robot
hand-eye calibration selects samples with different Flexiv TCP/end-camera poses.

What gets saved in the calibration record:

- Quest trajectory, metadata, and left/right MP4 videos.
- PC robot states, joint poses, and gripper state.
- End-camera MP4 and frame metadata.
- Third-camera MP4 and frame metadata when available.
- Quest/checkerboard outputs under `outputs/pc_live_calibration/<recordId>/`.
- Robot hand-eye result under `raw/<recordId>/robot_realsense/robot_hand_eye_result.json`.

Useful speed/diversity options:

- `--max-diverse-detection-frames-per-side`: max Quest video frames checked for checkerboard per eye.
- `--max-diverse-fit-frames-per-side`: max detected Quest frames used by the final optimizer per eye.
- `--min-diverse-frames-per-side`: minimum Quest frames kept before early stopping can happen.
- `--hand-eye-max-diverse-samples`: max robot samples checked for end-camera hand-eye.
- `--hand-eye-min-diverse-samples`: minimum robot samples kept before early stopping can happen.
- `--disable-diverse-frame-selection`: use all Quest frames for comparison runs.
- `--disable-hand-eye-diverse-selection`: use all robot samples for comparison runs.

## Formal Recording

Use `A` for the regular data-recording flow. When robot capture is enabled, the
PC session stores the Quest stream plus:

- robot tool/end-effector pose
- robot joint pose
- gripper state and gripper command events
- end-camera pose
- end-camera MP4
- fixed third-camera MP4 when available

The A-button flow is intended for task data collection after calibration. Robot
teleoperation requires a successful Quest-robot calibration and still requires
holding the right side/grip trigger during an active robot session.

## Replay

Open `http://127.0.0.1:8765/recordings`, select a record, then:

- drag the canvas to orbit
- use the mouse wheel to zoom
- scrub the timeline to inspect poses, gaze samples, robot state, and videos

Replay can show Quest gaze/head/controller samples, checkerboard placement,
robot tool/end-camera poses, joint-based robot skeleton, end-camera video, and
third-camera video when those artifacts exist in the record.

## Output Folders

- `pc/offline_calibration/pc_recordings/`: A-button PC receiver sessions.
- `pc/offline_calibration/raw/`: B-button calibration raw records and pulled Quest records.
- `pc/offline_calibration/outputs/`: calibration outputs and compact summaries.

## Troubleshooting

- No Quest data in live view: check the Quest app is running and sending UDP to the PC receiver host/port.
- Quest is visible in ADB but install fails: `unauthorized` means the headset has not accepted USB debugging.
- A/B buttons do not start recording: confirm Touch controllers are active; hand tracking alone does not fire these hotkeys.
- Robot connected but does not move during `A`: confirm the latest calibration is loaded, then hold the right side/grip trigger during the active recording/session.
- Robot cannot be dragged during `B`: check the live robot status for `control: freedrive` and `free-drag: enabled`. If it says `floating_cartesian_primitive`, verify the arm is in a state where Flexiv native free-drive is allowed. If it falls back to `cartesian_compliance`, verify `cartesian loop: running`, clear robot faults, reconnect Flexiv, and restart the receiver.
- Robot mesh does not follow the real arm: check that `joint age` stays low and that `joints:` changes when the real robot moves. If not, refresh the page and reconnect the robot.
- Robot mesh is visible but kinematically wrong: check `URDF FK vs flange`. A large value usually means the active URDF variant is not the correct one for the physical arm.
- Robot motion skips with `joint_limit_buffer`: move the arm away from the reported joint limit or reduce the teleop target direction; the guard is intentionally stopping TCP commands before Flexiv reaches its own limit stop.
- Hand-eye fails with too few detections: move the end camera so the checkerboard is visible in at least the required number of selected samples.
- Calibration is slow: lower the max diverse frame/sample limits, or disable automatic calibration and run the script manually.
