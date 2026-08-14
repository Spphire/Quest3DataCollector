---
name: convert-pc-recordings-to-gaze-wam-zarr
description: Filter any user-specified batch of Quest3DataCollector pc_recordings, convert retained whole recordings into canonical Gaze-WAM robot Zarr, and validate the result. Use when Codex is asked to select recordings by directory, explicit IDs, or record-ID time bounds; reject unstable or incomplete episodes; explain exclusions; rebuild robot Zarr; or check Collector data for gaze-dp training readiness.
---

# Build A Filtered Collector Zarr

Treat each invocation as an independent batch job. Never silently reuse the historical 2026-08-13 manifests or their accepted record set.

Read [references/filtering-contract.md](references/filtering-contract.md) before changing thresholds, camera roles, image geometry, gaze handling, or action semantics.

## Required Inputs

Resolve these values from the request or deployment context:

- Collector `pc_recordings` source directory.
- Batch boundary: all records in that directory, an explicit record-ID file, or inclusive start/end record IDs.
- Output Zarr path and an artifact directory for the batch manifest, selection report, logs, and validation report.
- Gaze-dp checkout used to run the canonical validator.

Do not infer a date range from an old manifest. Do not modify source recordings.

## Workflow

1. Create an immutable manifest for exactly the requested batch:

```bash
python .codex/skills/convert-pc-recordings-to-gaze-wam-zarr/scripts/build_recording_batch_manifest.py \
  --input-dir <pc_recordings-or-parent> \
  --start-record-id <optional-inclusive-start> \
  --end-record-id <optional-inclusive-end> \
  --requested-record-ids-file <optional-explicit-input-list> \
  --output <artifact-dir>/batch.record_ids.txt
```

Omit selectors that the user did not request. The script fails if an explicit record directory is missing or the batch is empty.

2. Run a dry selection pass before writing images:

```bash
python pc/offline_calibration/scripts/pc_recordings_to_zarr.py \
  --input-dir <pc_recordings-or-parent> \
  --record-ids-file <artifact-dir>/batch.record_ids.txt \
  --selection-report <artifact-dir>/batch.selection.json \
  --output <output.zarr> \
  --output-format gaze-wam \
  --wrist-role end \
  --eye-role third \
  --image-size 256,256 \
  --image-resize-mode letterbox \
  --max-image-age-seconds 0.060 \
  --max-gaze-age-seconds 0.060 \
  --max-sample-gap-seconds 0.060 \
  --max-endpoint-trim-seconds 1.0 \
  --gaze-median-window 7 \
  --dry-run
```

Inspect `batch.selection.json`. Report the candidate, accepted, and excluded counts plus every exclusion reason. Stop if no recording survives.

3. Run the same command without `--dry-run`. Capture stdout/stderr in the artifact directory. Do not change thresholds between dry-run and conversion.

4. Validate the resulting Zarr from the intended gaze-dp checkout:

```bash
PYTHONPATH=$PWD python scripts/validate_gaze_wam_zarr.py \
  --dataset-path <output.zarr> \
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
  --image-resize-mode letterbox \
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
  --timestamp-max-step 0.060
```

Save validator JSON beside the selection report. Smoke-load at least one sample through `GazeWAMDataset` before declaring the batch training-ready.

## Optional Rate Prequalification

Use `quest_pc_receiver.py audit-performance` plus `select_pc_recordings_for_gaze_wam.py` only when the user explicitly requests rate/reuse prequalification. Put only this batch's audit JSON files in an isolated directory. The default reusable workflow starts from the complete requested batch and lets strict episode-integrity filtering decide acceptance.

## Completion Report

Always report:

- Exact source, manifest, Zarr, selection report, conversion summary, log, and validator paths.
- Candidate/accepted/excluded recording counts and retained frame count.
- Every excluded record with structured reasons.
- Effective filter parameters, image geometry, gaze policy, and action semantics.
- Validator result and any remaining training-readiness blocker.

Do not claim success from file existence alone. Require completed conversion, `batch.selection.json`, and validator `"valid": true`.
