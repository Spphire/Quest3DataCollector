# Usage

This is the operator guide for the Quest 3 PC receiver, live viewer, calibration
recording, robot/RealSense capture, and replay pages.

## Quick Start on the Lab PC

The lab PC currently used for the live receiver is:

```text
10.128.1.95
```

Remote project root:

```bash
/ssd1/shenyibo/Quest3DataCollector
```

Open the live viewer from another PC on the same network:

```text
http://10.128.1.95:8765/
```

If the browser appears stale after a deployment, add a cache-busting query:

```text
http://10.128.1.95:8765/?reload=1
```

The current lab receiver command is:

```bash
cd /ssd1/shenyibo/Quest3DataCollector
pc/offline_calibration/scripts/start_lab_receiver.sh --restart
```

The script stops only existing `quest_pc_receiver.py receive` processes, starts
the receiver under `nohup`, writes logs to `receiver.log`, and uses the current
lab defaults: UDP `9100`, viewer `8765`, Flexiv interface `192.168.2.108`,
end RealSense `244222073667`, third RealSense `750612070265`, robot state
`90 Hz`, and depth every `3` RGB frames.

With `--enable-gripper`, the default gripper device is `auto`: the receiver
asks the connected Flexiv robot for the Elements device list, prefers online
Robotiq devices, then tries other gripper-looking names before falling back to
common Robotiq names. If needed, pass
`--gripper-device <exact Elements device name>`.

Check the remote process:

```bash
pc/offline_calibration/scripts/start_lab_receiver.sh --status
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
- `--record-realsense-depth-every-n-frames <N>` keeps RGB at full frame rate but only appends depth to the indexed raw depth stream every `N` frames during A-button formal recordings. Default: `3`.
- `--flush-every <N>` and `--flush-interval-seconds <T>` control how often the PC telemetry JSONL/CSV files are flushed; the default is batched instead of every packet.
- `--sample-log-interval-seconds <T>` limits the console sample-status print rate during high-frequency runs. Default: `1`.
- `--udp-receive-buffer-bytes <N>` requests a larger UDP receive buffer for Quest telemetry bursts. The effective value may still be capped by the remote OS socket limits.
- `--recording-idle-timeout-seconds <T>` closes an active PC recording if its own recording datagrams stop arriving for too long, which helps recover when a `recording_stop` packet is lost.
- `--formal-control-mode record_only` keeps A-button formal recording in a no-arm-motion mode for safe performance benchmarks. Robot state, cameras, Quest data, and optional gripper handling remain available.

Formal A-button recording rates:

- Flexiv state is sampled on a fixed `90 Hz` deadline into `robot_realsense/robot_states.jsonl`.
- RealSense RGB is recorded at `30 Hz` per camera role.
- `robot_realsense/samples.jsonl` is the final fixed `30 Hz` aligned stream for training conversion.
- `pc_samples.jsonl`, `pc_telemetry_raw.jsonl`, and `pc_controllers.csv` preserve every received formal Quest sample; the fixed 30 Hz training timeline is `robot_realsense/samples.jsonl`.

After recording, enforce the rate and jitter gates:

```bash
.venv312/bin/python pc/offline_calibration/scripts/quest_pc_receiver.py audit-performance \
  --pc-session pc/offline_calibration/pc_recordings/<record_id>
```

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
- Right index trigger: gripper open/close command when gripper control is enabled.
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

The checkerboard is 11x8 inner corners with 25 mm square size. The 180-degree
corner-order ambiguity is resolved from black/white corner appearance first,
then by identity-vs-rot180 reprojection or hand-eye residual matching when
appearance is inconclusive.

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

The A-button flow is intended for task data collection after calibration, and it
does not reuse the low-frequency calibration sampling policy. It runs as an
independent timestamped capture pipeline:

- Quest telemetry is recorded whenever the Quest sends a sample.
- Flexiv robot state is polled independently into
  `robot_realsense/robot_states.jsonl`, default `60 Hz`.
- RealSense videos are recorded independently at the stream frame rate, with
  frame timing written to `robot_realsense/video_frames.jsonl`.
- `robot_realsense/samples.jsonl` keeps one Quest-aligned compatibility row per
  Quest sample, pointing at the latest robot state and latest video frames.

This means the formal record should be synchronized by timestamps during replay
or downstream processing, rather than by assuming every modality shares the same
sample clock. The high-rate `robot_states.jsonl` rows intentionally store the
raw Flexiv pose/joint state in a lightweight form; replay and analysis derive
display transforms from those raw fields when needed.

When the machine is under unusually heavy disk/CPU load, the live viewer exposes
`Depth every` as the same `recordDepthEveryNFrames` knob. The current default
is `3`, which keeps RGB video at full frame rate while recording depth at about
10 Hz on a 30 Hz stream. Setting it to `1` preserves every depth frame, but that
is significantly heavier and should be reserved for short, depth-critical
captures.

On the current lab machine, the requested UDP receive buffer may be higher than
the effective kernel-accepted buffer. The receiver prints the actual bound
buffer in its startup log as `rcvbuf=...`.

Robot teleoperation still requires a successful Quest-robot calibration and
still requires holding the right side/grip trigger during an active robot
session.

After a formal recording, run the performance audit when you want a quick
pass/fail on capture health:

```bash
cd /ssd1/shenyibo/Quest3DataCollector
.venv312/bin/python pc/offline_calibration/scripts/quest_pc_receiver.py audit-performance \
  --pc-session pc/offline_calibration/pc_recordings/<record_id>
```

The default thresholds require robot state to reach at least 80% of the
configured target rate, each RealSense stream to reach at least 80% of target
FPS, camera queue drops to stay below 5%, p95 capture-to-write latency to stay
below 1 second, and the PC writer/save path to finish without drops or close
errors. The lab default is 60 Hz because the current Python single-process
pipeline can show long-tail scheduling gaps at a requested 90 Hz when dual
RGB-D recording is active. A failed audit means the record is still often
replayable, but it should not be treated as a healthy high-frequency data
capture.

To evaluate teleoperation responsiveness, run the latency analysis after a
formal recording with robot teleop enabled:

```bash
cd /ssd1/shenyibo/Quest3DataCollector
.venv312/bin/python pc/offline_calibration/scripts/quest_pc_receiver.py teleop-latency \
  --pc-session pc/offline_calibration/pc_recordings/<record_id>
```

The receiver also writes `teleop_latency_analysis.json` automatically at the
end of each formal recording, and the replay page shows a `Teleop latency`
panel. The primary metric is `command target -> robot`: it compares
`robot_realsense/controller_motion.jsonl` target TCP motion against
`robot_realsense/robot_states.jsonl` TCP motion on the same PC perf-counter
timeline. Positive lag means the robot follows the command after that delay.
When target commands are missing, the analyzer falls back to `Quest controller
-> robot` using right-controller motion in `pc_samples.jsonl`.

Interpret the two displayed numbers differently: `corr lag` is the best
cross-correlation lag for continuous following, while `event p50/p95` measures
threshold-crossing motion onset delay. The chart overlays command target speed,
robot TCP speed, and controller speed. This is a system responsiveness metric,
not an absolute one-way network latency measurement.

To check receiver performance without wearing the Quest, use the synthetic
formal-recording probe on the lab machine:

```bash
cd /ssd1/shenyibo/Quest3DataCollector
.venv312/bin/python pc/offline_calibration/scripts/perf_probe_receiver.py --duration-seconds 10
```

The probe sends A-recording UDP samples, waits for the saving phase to finish,
runs the same audit, and loads compact replay. It intentionally does not press
teleop or gripper controls.

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
- Right index trigger does not move the gripper: check the live `gripper:` and
  `devices:` status. If `devices` does not list a Robotiq/gripper device, fix
  the Flexiv Elements device configuration or pass the exact
  `--gripper-device` name. If a recording already happened, inspect
  `robot_realsense/gripper_commands.jsonl`; `trigger` proves Quest/PC trigger
  delivery, while `enableAttempts` shows device-name failures.
- Robot cannot be dragged during `B`: check the live robot status for `control: freedrive` and `free-drag: enabled`. If it says `floating_cartesian_primitive`, verify the arm is in a state where Flexiv native free-drive is allowed. If it falls back to `cartesian_compliance`, verify `cartesian loop: running`, clear robot faults, reconnect Flexiv, and restart the receiver.
- Robot mesh does not follow the real arm: check that `joint age` stays low and that `joints:` changes when the real robot moves. If not, refresh the page and reconnect the robot.
- Robot mesh is visible but kinematically wrong: check `URDF FK vs flange`. A large value usually means the active URDF variant is not the correct one for the physical arm.
- Robot motion skips with `joint_limit_buffer`: move the arm away from the reported joint limit or reduce the teleop target direction; the guard is intentionally stopping TCP commands before Flexiv reaches its own limit stop. The default guard buffer is `0.04` rad. If the arm feels artificially constrained but the status does not report `joint_limit_buffer`, check the teleop step limits (`Max step m` and `Rot step deg`) in the live UI; teleop does not clamp cumulative workspace offset.
- Hand-eye fails with too few detections: move the end camera so the checkerboard is visible in at least the required number of selected samples.
- Calibration is slow: lower the max diverse frame/sample limits, or disable automatic calibration and run the script manually.
