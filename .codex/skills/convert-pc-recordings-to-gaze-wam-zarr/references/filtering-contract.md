# Filtering And Conversion Contract

## Repository Boundary

`Quest3DataCollector` owns `pc_recordings`, performance audit semantics, RealSense roles, calibration references, and Collector-to-canonical-Zarr conversion. `gaze-dp` owns the canonical schema validator, dataset reader, and training behavior.

The offline converter uses the self-contained pinned packages in `pc/offline_calibration/requirements-zarr.txt`. Keep this environment separate from the live Collector receiver and its hardware SDKs.

## 2026-08-13 Selection

- Source: records whose IDs are at or after `record_20260813_162120`.
- Candidate count: 62.
- Selected count: 50.
- Excluded count: 12.
- Calibration: `record_pc_calib_20260813_161447`.
- Calibration quality: `quality.ok=true`, `quality.accepted=true`.
- Policy/end camera serial: `244222073667`.
- Third camera serial: `750612070265`.

Qualify only these stability metrics:

- `robot_state_rate` effective rate / 90 Hz must be at least 0.95.
- `aligned_sample_rate` effective rate / 30 Hz must be at least 0.95.
- `camera_end_rate` effective rate / 30 Hz must be at least 0.95.
- `camera_third_rate` effective rate / 30 Hz must be at least 0.95.
- `alignedReusedSourceSamples / alignedSamples.count` must be no more than 0.10.

Do not exclude a record merely because the strict all-check audit reports isolated sampler missed ticks when all five dataset-stability metrics above pass. The committed allowlist and exclusion TSV are the reproducible result of this narrower dataset filter.

Gaze temporarily leaving the end-camera image is also not a record-level exclusion. The point-gaze-supervised canonical Zarr omits those individual rows and splits episode boundaries around resulting temporal gaps. A future no-gaze training mode must use an explicit presence-mask/config contract instead of fabricated image-bound gaze labels.

## Audit Command

```bash
.venv312/bin/python \
  pc/offline_calibration/scripts/quest_pc_receiver.py \
  audit-performance \
  --pc-session pc/offline_calibration/pc_recordings/record_YYYYMMDD_HHMMSS \
  --output-json /tmp/gaze_record_audit_20260813_162120/record_YYYYMMDD_HHMMSS.json
```

## Transfer From Collector To H200-5041

PowerShell/OpenSSH requires legacy SCP mode for this remote-to-remote path:

```powershell
scp -O -3 -r `
  lvjun@10.128.1.95:/ssd1/shenyibo/Quest3DataCollector/pc/offline_calibration/pc_recordings/record_YYYYMMDD_HHMMSS `
  H200-5041:/mnt/workspace/zhengkai/gaze_wam_robot_20260813/pc_recordings/
```

Transfers are restartable per record. Verify all allowlisted directory names before conversion.

## Gaze-DP Validation

From the gaze-dp repository:

```bash
PYTHONPATH=$PWD python scripts/validate_gaze_wam_zarr.py \
  --dataset-path /path/to/gaze_wam_robot.zarr \
  --dataset-type robot \
  --camera-key camera0_rgb \
  --gaze-key gaze_xy \
  --heatmap-key none \
  --action-abs-key action_abs_tcp \
  --tcp-pose-key tcp_pose_abs \
  --gripper-key gripper_width \
  --n-obs-steps 2 \
  --action-horizon 16 \
  --image-size 256 256 \
  --image-resize-mode stretch \
  --heatmap-token-grid 16 16 \
  --heatmap-dim 16 \
  --action-dim 10 \
  --timestamp-key timestamp \
  --image-timestamp-key image_timestamp \
  --robot-state-timestamp-key robot_state_timestamp \
  --action-timestamp-key action_timestamp \
  --gaze-timestamp-key gaze_timestamp \
  --require-timestamps \
  --timestamp-max-delta 0.060 \
  --timestamp-max-step 0.200
```

The validator must compute timestamp intervals within `meta/episode_ends`; physical recordings and gap-split segments may have discontinuous absolute timestamps at episode boundaries.
