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
- Require contiguous camera frame indexes within each source and monotonically increasing capture timelines.
- Require image alignment age and internal sample/camera gaps to be at most 60 ms.
- Use the end/wrist camera as the master capture timeline. For every retained master frame, match the nearest third-camera frame by capture timestamp, requiring monotonic third frame indexes and a maximum timestamp delta of 60 ms. Do not require equal frame counts or equal frame indexes across devices.
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

## Quest Source Reuse And Endpoint Recovery

Quest source reuse means reuse of the Quest telemetry/gaze/pose source while
aligning PC rows. It does not mean reusing robot-state rows or end/third camera
frames. Track Quest reuse, UDP sequence loss/ordering, source age, robot timing,
and camera timing as separate metrics.

Apply the following order when raw `samples.jsonl` is available:

1. Identify leading/trailing rows whose Quest source age exceeds
   `max_gaze_age_seconds` (default 60 ms), plus terminal reuse runs. Evaluate a
   terminal reuse run before applying stale-age trim so a partly stale run is
   removed in full.
2. If a leading/trailing run of `aligned_source_reused` is longer than
   `max_consecutive_reuse` (default 5 frames), treat it as a Quest stop/idle tail
   and trim it. Do not trim short endpoint reuse unconditionally.
3. On the retained internal window, accept runs up to 5 consecutive frames. For
   internal reused rows, never use the repeated Quest payload as a fresh gaze label:
   interpolate from surrounding fresh gaze in PC-world coordinates when both sides
   exist; otherwise keep the action row and set `has_gaze_condition=false` and
   `has_gaze_label=false`.
4. Exclude when an internal run exceeds the configured consecutive threshold.
5. Use post-trim cumulative reuse ratio only as a configurable fallback guard
   (default 20%), never as the sole historical 10% hard gate.

Endpoint trimming may recover a usable episode such as a recording whose final
~2 seconds were written after Quest telemetry stopped. It must not hide independent
robot, camera, or UDP/system failures; those checks remain recording-level gates.
Quest endpoint recovery is a source-timeline policy and is not capped by the
camera-only `max_endpoint_trim_seconds` limit.

## Optional Audit Prefilter

Do not apply the historical rate/reuse filter by default. When explicitly requested, require:

- Flexiv robot-state effective rate / 90 Hz at least 0.95.
- Aligned sample effective rate / 30 Hz at least 0.95.
- End and third camera effective rates / 30 Hz at least 0.95.
- When raw timelines are unavailable, retain the legacy cumulative reuse check.
  When raw timelines are available, report the post-trim reuse ratio and apply the
  configured fallback guard instead of treating 10% as a hard gate.

Generate audit reports and prefilter manifests only from the current batch so unrelated recordings cannot enter by timestamp ordering. The selection report
must include both pre-trim and post-trim reuse metrics and stable exclusion reasons.

## Acceptance

A batch is complete only when:

1. The immutable candidate manifest exists.
2. Selection counts reconcile and all exclusions are explained.
3. Zarr conversion completes without trailing-frame substitution.
4. The gaze-dp validator returns `"valid": true` with timestamps required.
5. `GazeWAMDataset` successfully loads a sample using the intended training config.
