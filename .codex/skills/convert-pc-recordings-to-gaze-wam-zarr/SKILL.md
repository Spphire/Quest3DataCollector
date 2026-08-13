---
name: convert-pc-recordings-to-gaze-wam-zarr
description: Select stable Quest3DataCollector pc_recordings from audit-performance reports, transfer the allowlisted recordings, convert them into canonical Gaze-WAM robot Zarr, and validate the result against gaze-dp. Use for Collector-to-Zarr dataset preparation, frame-rate filtering, Quest gaze reprojection, or reproducing the 2026-08-13 robot dataset selection.
---

# Convert Collector Recordings To Gaze-WAM Zarr

Own this workflow in `Quest3DataCollector`. The input schema, stream quality metrics, RealSense roles, calibration references, and gaze reprojection are Collector contracts. Keep `gaze-dp` responsible for canonical Zarr validation, loading, and training; do not copy this converter into that repo.

Read [references/filtering-contract.md](references/filtering-contract.md) before changing thresholds, camera geometry, action semantics, or the dataset manifest.

## Workflow

1. Run `quest_pc_receiver.py audit-performance` once per source record and retain each JSON report.
2. Run `pc/offline_calibration/scripts/select_pc_recordings_for_gaze_wam.py` to generate an explicit allowlist and exclusion TSV.
3. Transfer only allowlisted full record directories. On Windows, use `scp -O -3` for remote-to-remote transfer through the workstation.
4. Run `pc/offline_calibration/scripts/pc_recordings_to_zarr.py --output-format gaze-wam` with the allowlist.
5. Run gaze-dp `scripts/validate_gaze_wam_zarr.py` with timestamps required and a 60 ms alignment gate.
6. Smoke-load at least one sample through `GazeWAMDataset` before training.

Never modify or restart the live Collector receiver for this offline workflow. Preserve source recordings and write Zarr to a separate destination.

Use a separate offline conversion venv with `pc/offline_calibration/requirements-zarr.txt`. Do not install the Collector hardware stack or add Zarr to the live receiver environment solely for this workflow.

## Canonical Commands

Create the offline environment:

```bash
python3 -m venv .venv-zarr
.venv-zarr/bin/python -m pip install -r pc/offline_calibration/requirements-zarr.txt
```

Generate the manifest:

```bash
python pc/offline_calibration/scripts/select_pc_recordings_for_gaze_wam.py \
  --audit-dir /tmp/gaze_record_audit_20260813_162120 \
  --start-record-id record_20260813_162120 \
  --selected-output pc/offline_calibration/manifests/gaze_wam_robot_20260813_after_162120.txt \
  --excluded-output pc/offline_calibration/manifests/gaze_wam_robot_20260813_after_162120_excluded.tsv
```

Convert the allowlist:

```bash
python pc/offline_calibration/scripts/pc_recordings_to_zarr.py \
  --input-dir /path/to/pc_recordings \
  --record-ids-file pc/offline_calibration/manifests/gaze_wam_robot_20260813_after_162120.txt \
  --output /path/to/gaze_wam_robot.zarr \
  --output-format gaze-wam \
  --wrist-role end \
  --image-size 256,256 \
  --max-image-age-seconds 0.060 \
  --max-gaze-age-seconds 0.060 \
  --max-sample-gap-seconds 0.200 \
  --min-segment-frames 16
```

Use the exact gaze-dp validator command in the reference. A conversion is not complete until it returns `"valid": true`.

## Invariants

- Use end camera serial `244222073667` as `camera0_rgb` for this dataset.
- Stretch the full end image to `256x256`; do not crop unless gaze coordinates are remapped through the same crop.
- Project `quest_gaze3d_pc_world` with each row's `T_world_end_camera` and the end-camera intrinsics.
- Gaze leaving the end-camera field of view is valid behavior and does not exclude the physical record. Drop only those rows from this point-gaze-supervised Zarr.
- Split retained rows whenever the aligned, image, robot/action, or gaze source timeline has a gap over 200 ms. Never allow an action window to cross missing time.
- Store the current aligned executed TCP and gripper row in `action_abs_tcp`. Do not pre-shift actions; gaze-dp constructs future horizons.
- Fail on missing requested video frames. Do not duplicate the last decoded frame in canonical output.
- Require every allowlisted record directory. Silent partial conversion is forbidden.
