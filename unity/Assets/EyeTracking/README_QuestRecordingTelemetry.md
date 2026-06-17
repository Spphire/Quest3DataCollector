# Quest Recording Telemetry

`QuestCameraRecorder` can send realtime telemetry while recording. The sender is
`QuestRecordingTelemetrySender` on the `QuestCameraRecorder` GameObject.

Transport:

- UDP
- default target: `10.128.0.227:9100`
- encoding: UTF-8 JSON, one datagram per message, newline terminated
- protocol name: `quest_recording_telemetry_v1`

Message types:

- `recording_start`
- `sample`
- `recording_stop`

The `sample` message is emitted from the same update path that writes
`trajectory.jsonl`, so gaze and camera timestamps match the local recording.
The same left/right controller pose payload is now also written into
`trajectory.jsonl`, so Quest-local and PC-received controller samples can be
matched by `sampleIndex`. Controller poses are expressed in Unity world frame.
HMD `leftEyePose`/`rightEyePose` and `leftEyePosition`/`rightEyePosition` are
also included so the PC side can compare the two controllers plus both eyes as
four world-space check points.
The sender first tries `OVRInput`, then Unity XR `InputDevice`, then the
Interaction SDK `ControllerRef`, then the `OVRCameraRig`
`LeftControllerAnchor`/`RightControllerAnchor` transform as a fallback. The
`source` field records which path produced the pose. When no pose is available,
`source` is `missing` and `missingReason` summarizes what each source reported.

Controller `position` arrays use Unity world meters. `rotation` arrays are
quaternions in `[qw, qx, qy, qz]` order. The legacy `pose` field keeps the
existing recorder convention:

```text
[x, y, z, qw, qx, qy, qz]
```

Important `sample` fields:

```json
{
  "protocol": "quest_recording_telemetry_v1",
  "type": "sample",
  "sequence": 12,
  "recordId": "record_20260616_123456",
  "sampleIndex": 12,
  "unityTimestampSeconds": 123.45,
  "recordingTimestampSeconds": 0.21,
  "hasGaze": true,
  "hasGazeHit": true,
  "gazePoint3DWorld": [0.1, 1.2, 0.8],
  "gazePoint3DSource": "environment_mapper_hit",
  "gazeRayOrigin": [0.0, 1.6, 0.0],
  "gazeRayDirection": [0.0, -0.2, 0.98],
  "hasLeftEyePose": true,
  "hasRightEyePose": true,
  "leftEyePoseSource": "OVRCameraRig.leftEyeAnchor",
  "rightEyePoseSource": "OVRCameraRig.rightEyeAnchor",
  "leftEyePose": [-0.032, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0],
  "rightEyePose": [0.032, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0],
  "leftEyePosition": [-0.032, 1.6, 0.0],
  "rightEyePosition": [0.032, 1.6, 0.0],
  "leftController": {
    "handedness": "left",
    "hasPose": true,
    "source": "OVRInput",
    "missingReason": null,
    "positionTracked": true,
    "rotationTracked": true,
    "position": [-0.2, 1.2, 0.4],
    "rotation": [1.0, 0.0, 0.0, 0.0],
    "pose": [-0.2, 1.2, 0.4, 1.0, 0.0, 0.0, 0.0]
  },
  "rightController": {
    "handedness": "right",
    "hasPose": true,
    "source": "OVRInput",
    "missingReason": null,
    "positionTracked": true,
    "rotationTracked": true,
    "position": [0.2, 1.2, 0.4],
    "rotation": [1.0, 0.0, 0.0, 0.0],
    "pose": [0.2, 1.2, 0.4, 1.0, 0.0, 0.0, 0.0]
  }
}
```

Run the sample receiver from the PC:

```powershell
py tools\quest_recording_udp_receiver.py --port 9100 --output telemetry.jsonl
```

For full synchronized PC recording plus Quest-local comparison, use the
offline calibration receiver:

```powershell
py W:\QuestCalib\offline_calibration\scripts\quest_pc_receiver.py receive --host 0.0.0.0 --port 9100 --single-session --pull-quest-record
```

It writes `pc_telemetry_raw.jsonl`, `pc_samples.jsonl`,
`pc_controllers.csv`, and `pc_session_summary.json` under
`W:\QuestCalib\offline_calibration\pc_recordings\<recordId>\`. With
`--pull-quest-record`, it pulls the matching Quest output directory into
`W:\QuestCalib\offline_calibration\raw\` and writes
`quest_pc_alignment_summary.json` with controller pose error and PC receive
timing statistics. It also reports a four-point alignment check over left/right
controller positions plus left/right eye positions. If the Quest recording has
already been pulled manually, run the analyzer directly:

```powershell
py W:\QuestCalib\offline_calibration\scripts\quest_pc_receiver.py analyze --quest-record W:\QuestCalib\offline_calibration\raw\record_YYYYMMDD_HHMMSS --pc-session W:\QuestCalib\offline_calibration\pc_recordings\record_YYYYMMDD_HHMMSS
```

For a short smoke test or automation, the receiver can exit after a fixed number
of valid JSON messages:

```powershell
py tools\quest_recording_udp_receiver.py --port 9100 --output telemetry.jsonl --max-messages 3 --timeout 30
```

To make a smoke test fail unless realtime gaze and both controller poses are
actually present:

```powershell
py tools\quest_recording_udp_receiver.py --host 0.0.0.0 --port 9100 --output telemetry.jsonl --timeout 120 --wait-for-requirements --require-gaze3d --require-left-controller-pose --require-right-controller-pose
```

The receiver prints a final `Telemetry summary` with message/sample counts,
whether each required payload was seen, and the observed controller pose
sources. On the Quest side, `QuestRecordingTelemetrySender` logs recording
telemetry start/stop, UDP target opening, the first sample, periodic sample
status, and controller source changes to logcat.

For controller-pose checks, wake and connect both Quest controllers before or
soon after launching the app. If the headset reports the controllers as
inactive/disconnected, samples will still include gaze, but
`leftController.hasPose` and `rightController.hasPose` will remain false.

Before a Quest test, confirm the sender target matches the PC address on the
same Wi-Fi network. The default scene value is `10.128.0.227:9100`.
