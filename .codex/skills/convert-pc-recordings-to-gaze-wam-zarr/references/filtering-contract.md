# Reusable Filtering And Conversion Contract

## Ownership

`Quest3DataCollector` owns batch selection, recording integrity, stream alignment, camera calibration, gaze reprojection, and Collector-to-Zarr conversion. `gaze-dp` owns canonical schema validation, dataset loading, and training behavior.

Use the isolated environment from `pc/offline_calibration/requirements-zarr.txt`. Do not install conversion dependencies into the live Collector environment and do not restart the receiver for offline conversion.

## Batch Boundary

A batch is an immutable, sorted, newline-separated list of recording directory names. Resolve it from one of:

- Every `record_*` directory under a specified source directory.
- An explicit user-provided record-ID list.
- Inclusive `record_YYYYMMDD_HHMMSS` start/end boundaries.

Require every listed directory. Never silently process a partial batch. Historical manifests are provenance artifacts, not defaults for a future batch.

## Default Episode Filter

Apply these rules to each physical recording independently:

- Require aligned samples, robot states, both required videos, and end-camera intrinsics.
- Require every retained aligned row to be valid and linked to a robot state.
- Require contiguous camera frame indexes and monotonically increasing aligned/capture timelines.
- Require image alignment age and internal sample/camera gaps to be at most 60 ms.
- Search only for a common continuous window obtainable by trimming at most 1.0 second from each start/end of each aligned or camera stream.
- Keep the retained window as one episode. Never extract an internal good segment around a bad middle interval.
- Reject the entire physical recording when no valid common window exists or an internal discontinuity remains.
- Require at least two retained samples and readable TCP pose/video paths.

The converter writes `<output>.selection.json` even during `--dry-run` and even when every candidate is rejected. Its `candidate_count`, `accepted_count`, and `excluded_count` must reconcile, and every exclusion must include stable reason codes and details.

## Gaze Policy

- Treat gaze leaving the end-camera image as valid behavior, not a recording-level failure.
- Apply a causal median filter of window 7 to Quest gaze-ray depth.
- Linearly interpolate internal missing 3D gaze in PC world coordinates.
- Do not extrapolate missing gaze at episode edges.
- Reproject with each row's `T_world_end_camera` and that recording's end-camera intrinsics.
- Preserve all action rows. Encode gaze availability with `has_gaze_label` and projection status instead of deleting action frames.

## Image Geometry

Use the full end-camera frame. For a 1280x720 source and 256x256 output, resize proportionally to 256x144 and add 56 black pixels above and below. Apply the exact same letterbox transform to normalized `gaze_xy`.

Do not stretch or crop unless the user explicitly selects another geometry and the validator/training config uses the same setting.

## Action Contract

Write the current aligned executed TCP pose plus gripper width to `action_abs_tcp`. Do not shift action rows in the converter. Gaze-dp constructs each future action chunk relative to the observation time and episode boundary.

Required canonical keys include:

- `data/camera0_rgb`
- `data/gaze_xy`
- `data/has_gaze_label`
- `data/gaze_world_pc`
- `data/gaze_3d_source`
- `data/gaze_projection_status`
- `data/action_abs_tcp`
- `data/tcp_pose_abs`
- `data/gripper_width`
- aligned timestamp arrays
- `meta/episode_ends`

## Optional Audit Prefilter

Do not apply the historical rate/reuse filter by default. When explicitly requested, require:

- Flexiv robot-state effective rate / 90 Hz at least 0.95.
- Aligned sample effective rate / 30 Hz at least 0.95.
- End and third camera effective rates / 30 Hz at least 0.95.
- `alignedReusedSourceSamples / alignedSamples.count` at most 0.10.

Generate audit reports and prefilter manifests only from the current batch so unrelated recordings cannot enter by timestamp ordering.

## Acceptance

A batch is complete only when:

1. The immutable candidate manifest exists.
2. Selection counts reconcile and all exclusions are explained.
3. Zarr conversion completes without trailing-frame substitution.
4. The gaze-dp validator returns `"valid": true` with timestamps required.
5. `GazeWAMDataset` successfully loads a sample using the intended training config.
