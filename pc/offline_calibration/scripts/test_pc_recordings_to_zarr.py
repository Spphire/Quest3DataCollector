from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("pc_recordings_to_zarr.py")
SPEC = importlib.util.spec_from_file_location("pc_recordings_to_zarr", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _MetaGroup:
    def __init__(self):
        self.values = {}

    def create_dataset(self, key, data, **_kwargs):
        self.values[key] = np.asarray(data)


class PcRecordingsToZarrTest(unittest.TestCase):
    def test_flush_frame_batch_writes_contiguous_runs_in_bulk(self):
        target = np.zeros((7, 2, 2, 3), dtype=np.uint8)
        first = np.full((2, 2, 3), 11, dtype=np.uint8)
        second = np.full((2, 2, 3), 22, dtype=np.uint8)
        third = np.full((2, 2, 3), 33, dtype=np.uint8)
        pending = [(4, third), (1, first), (2, second)]

        MODULE.flush_frame_batch(target, pending)

        self.assertEqual(pending, [])
        self.assertTrue(np.all(target[1] == 11))
        self.assertTrue(np.all(target[2] == 22))
        self.assertTrue(np.all(target[4] == 33))
        self.assertFalse(np.any(target[0]))
        self.assertFalse(np.any(target[3]))

    def test_project_world_point_normalizes_and_rejects_invalid_points(self):
        camera = {
            "width": 100,
            "height": 80,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 40.0,
        }
        pose = {
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        self.assertTrue(
            np.allclose(
                MODULE.project_world_point([0.0, 0.0, 1.0], pose, camera),
                [0.5, 0.5],
            )
        )
        self.assertIsNone(MODULE.project_world_point([0.0, 0.0, -1.0], pose, camera))
        self.assertIsNone(MODULE.project_world_point([2.0, 0.0, 1.0], pose, camera))

        _, behind = MODULE.project_world_point_with_status([0.0, 0.0, -1.0], pose, camera)
        outside_xy, outside = MODULE.project_world_point_with_status([2.0, 0.0, 1.0], pose, camera)
        self.assertEqual(behind, MODULE.GAZE_PROJECTION_BEHIND_CAMERA)
        self.assertEqual(outside, MODULE.GAZE_PROJECTION_OUT_OF_FRAME)
        self.assertTrue(np.allclose(outside_xy, [2.5, 0.5]))

    def test_letterbox_geometry_and_gaze_remap_match_1280x720_to_256_square(self):
        geometry = MODULE.letterbox_geometry((720, 1280), (256, 256))

        self.assertEqual(geometry["resized_size"], [144, 256])
        self.assertEqual(geometry["padding_ltrb"], [0, 56, 0, 56])
        self.assertTrue(np.allclose(geometry["scale_xy"], [0.2, 0.2]))

        remapped = MODULE.remap_normalized_gaze_xy(
            np.asarray([0.25, 0.75], dtype=np.float32),
            source_size=(720, 1280),
            target_size=(256, 256),
            image_resize_mode="letterbox",
        )
        self.assertTrue(np.allclose(remapped, [0.25, 0.640625]))
        self.assertTrue(
            np.allclose(
                MODULE.remap_normalized_gaze_xy(
                    np.asarray([0.25, 0.75], dtype=np.float32),
                    source_size=(720, 1280),
                    target_size=(256, 256),
                    image_resize_mode="stretch",
                ),
                [0.25, 0.75],
            )
        )

    def test_letterbox_image_has_expected_black_padding(self):
        image = np.full((720, 1280, 3), [10, 20, 30], dtype=np.uint8)
        output = MODULE.resize_image(
            image,
            target_size=(256, 256),
            image_resize_mode="letterbox",
        )

        self.assertEqual(output.shape, (256, 256, 3))
        self.assertFalse(np.any(output[:56]))
        self.assertFalse(np.any(output[200:]))
        self.assertTrue(np.all(output[56:200] == [10, 20, 30]))

    def test_replay_aligned_ray_depth_median_and_internal_interpolation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            record_dir = Path(temp_dir)
            rows = []
            for index, depth in enumerate((1.0, 100.0, 2.0)):
                rows.append(
                    {
                        "sampleIndex": index,
                        "pcWorld": {
                            "gazePoint3DWorld": [0.0, 0.0, depth],
                            "gazeRayOrigin": [0.0, 0.0, 0.0],
                            "gazeRayDirection": [0.0, 0.0, 1.0],
                        },
                    }
                )
            (record_dir / "pc_samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            rays = MODULE.load_quest_gaze_rays(record_dir, median_window=3)
            self.assertTrue(np.allclose(rays[2]["filtered_point"], [0.0, 0.0, 2.0]))

            samples = [
                {
                    "quest_sample_index": source_index,
                    "aligned_target_perf_counter_seconds": index / 30.0,
                    "pc_perf_counter_seconds": index / 30.0,
                    "quest_pc_receive_perf_counter_seconds": index / 30.0,
                    "action_marker": index,
                }
                for index, source_index in enumerate((0, 99, 99, 2))
            ]
            stats = MODULE.build_smoothed_interpolated_gaze(
                samples,
                rays,
                median_window=3,
                segment_end_offsets=[4],
                max_gaze_age_seconds=0.060,
            )

            self.assertEqual([sample["action_marker"] for sample in samples], [0, 1, 2, 3])
            self.assertEqual(stats["median_filtered_3d"], 2)
            self.assertEqual(stats["interpolated_3d"], 2)
            self.assertTrue(np.allclose(samples[1]["_gaze_world_pc"], [0.0, 0.0, 4.0 / 3.0]))
            self.assertTrue(np.allclose(samples[2]["_gaze_world_pc"], [0.0, 0.0, 5.0 / 3.0]))
            self.assertEqual(samples[1]["_gaze_3d_source"], MODULE.GAZE_3D_SOURCE_INTERPOLATED)

    def test_allowlist_requires_every_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "record_20260813_162120").mkdir()
            found = MODULE.find_record_dirs(root, {"record_20260813_162120"})
            self.assertEqual([path.name for path in found], ["record_20260813_162120"])
            with self.assertRaises(FileNotFoundError):
                MODULE.find_record_dirs(root, {"record_missing"})

    def test_time_gap_marks_the_whole_episode_as_discontinuous(self):
        samples = [
            {"pc_perf_counter_seconds": value}
            for value in (0.0, 0.033, 0.066, 1.0, 1.033, 2.0)
        ]
        discontinuities = MODULE.find_sample_timeline_discontinuities(
            samples,
            max_gap_seconds=0.2,
        )

        self.assertEqual(
            [(row["previous_index"], row["current_index"]) for row in discontinuities],
            [(2, 3), (4, 5)],
        )

    def test_video_timeline_trims_only_endpoints_with_a_hard_limit(self):
        samples = [
            {"aligned_target_perf_counter_seconds": index / 30.0}
            for index in range(3)
        ]
        frames = [
            {
                "frame_index": index,
                "frame_captured_perf_counter_seconds": index / 30.0 + 0.005,
                "video": "videos/end.mp4",
                "serial": "camera",
            }
            for index in range(3)
        ]
        retained, reasons, metrics = MODULE.trim_and_attach_video_timelines(
            samples,
            {"end": frames},
            roles=("end",),
            max_image_age_seconds=0.060,
            max_sample_gap_seconds=0.060,
            max_endpoint_trim_seconds=1.0,
            sample_hz=30.0,
        )
        self.assertEqual(reasons, [])
        self.assertEqual([row["videos"]["end"]["frameIndex"] for row in retained], [0, 1, 2])
        self.assertLess(metrics["roles"]["end"]["max_alignment_delta_seconds"], 0.060)

        long_samples = [
            {"aligned_target_perf_counter_seconds": index / 30.0}
            for index in range(100)
        ]
        too_short_frames = [
            {
                "frame_index": index,
                "frame_captured_perf_counter_seconds": (index + 32) / 30.0,
                "video": "videos/end.mp4",
            }
            for index in range(69)
        ]
        retained, reasons, _ = MODULE.trim_and_attach_video_timelines(
            long_samples,
            {"end": too_short_frames},
            roles=("end",),
            max_image_age_seconds=0.060,
            max_sample_gap_seconds=0.060,
            max_endpoint_trim_seconds=1.0,
            sample_hz=30.0,
        )
        self.assertEqual(retained, [])
        self.assertTrue(any("no_continuous_common_window" in reason for reason in reasons))

    def test_video_timeline_rejects_internal_gap_instead_of_extracting_a_segment(self):
        samples = [
            {"aligned_target_perf_counter_seconds": index / 30.0}
            for index in range(90)
        ]
        frames = []
        for index in range(90):
            capture_time = index / 30.0
            if 30 <= index < 60:
                capture_time += 0.2
            frames.append(
                {
                    "frame_index": index,
                    "frame_captured_perf_counter_seconds": capture_time,
                    "video": "videos/end.mp4",
                }
            )
        retained, reasons, _ = MODULE.trim_and_attach_video_timelines(
            samples,
            {"end": frames},
            roles=("end",),
            max_image_age_seconds=0.060,
            max_sample_gap_seconds=0.060,
            max_endpoint_trim_seconds=1.0,
            sample_hz=30.0,
        )
        self.assertEqual(retained, [])
        self.assertTrue(reasons)

    def test_aligned_timeline_is_not_rejected_for_camera_capture_phase_changes(self):
        samples = []
        for index in range(4):
            samples.append(
                {
                    "pc_perf_counter_seconds": index * 0.033,
                    "robot_state_pc_perf_counter_seconds": index * 0.033,
                    "quest_pc_receive_perf_counter_seconds": index * 0.033,
                    "videoFrames": {
                        "end": {"capturedPerfCounterSeconds": index * 0.033}
                    },
                }
            )
        samples[2]["videoFrames"]["end"]["capturedPerfCounterSeconds"] = 0.30
        samples[3]["videoFrames"]["end"]["capturedPerfCounterSeconds"] = 0.333

        discontinuities = MODULE.find_sample_timeline_discontinuities(
            samples,
            max_gap_seconds=0.2,
        )

        self.assertEqual(discontinuities, [])

    def test_gaze_wam_lowdim_rows_keep_current_absolute_state(self):
        total = 2
        data = {
            "timestamp": np.zeros(total, dtype=np.float64),
            "image_timestamp": np.zeros(total, dtype=np.float64),
            "robot_state_timestamp": np.zeros(total, dtype=np.float64),
            "action_timestamp": np.zeros(total, dtype=np.float64),
            "gaze_timestamp": np.zeros(total, dtype=np.float64),
            "tcp_pose_abs": np.zeros((total, 9), dtype=np.float32),
            "gripper_width": np.zeros(total, dtype=np.float32),
            "action_abs_tcp": np.zeros((total, 10), dtype=np.float32),
            "gaze_xy": np.zeros((total, 2), dtype=np.float32),
            "gaze_world_pc": np.zeros((total, 3), dtype=np.float32),
            "gaze_3d_source": np.zeros(total, dtype=np.uint8),
            "gaze_projection_status": np.zeros(total, dtype=np.uint8),
            "has_gaze_label": np.zeros(total, dtype=np.bool_),
            "has_heatmap_image": np.ones(total, dtype=np.bool_),
        }
        camera = {
            "width": 100,
            "height": 80,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 40.0,
        }
        camera_pose = {
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        states = {}
        samples = []
        for index, x_value in enumerate((0.1, 0.2)):
            states[index] = {
                "pc_perf_counter_seconds": 10.0 + index,
                "T_base_tool_tcp": {
                    "translation_m": [x_value, 0.0, 0.5],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "T_world_end_camera": camera_pose,
            }
            samples.append(
                {
                    "robot_state_sample_index": index,
                    "aligned_target_perf_counter_seconds": 30.0 + index / 30.0,
                    "pc_perf_counter_seconds": 20.0 + index,
                    "robot_state_pc_perf_counter_seconds": 10.0 + index,
                    "quest_pc_receive_perf_counter_seconds": 19.0 + index,
                    "quest_gaze3d_pc_world": [0.0, 0.0, 1.0],
                    "_gaze_world_pc": np.asarray([0.0, 0.0, 1.0]),
                    "_gaze_3d_source": MODULE.GAZE_3D_SOURCE_MEDIAN_FILTERED,
                    "videoFrames": {
                        "end": {"capturedPerfCounterSeconds": 18.0 + index}
                    },
                }
            )
        plan = MODULE.EpisodePlan(
            record_id="record_test",
            episode_dir=Path("."),
            samples=samples,
            robot_states_by_index=states,
            gripper_times=[-float("inf")],
            gripper_widths=[0.08],
            start_output_index=0,
            end_output_index=2,
            camera=camera,
            segment_end_offsets=[1, 2],
        )
        meta = _MetaGroup()
        MODULE.write_lowdim_arrays(
            data,
            meta,
            [plan],
            pose_frame="T_base_tool_tcp",
            output_format="gaze-wam",
            image_size=(100, 100),
            image_resize_mode="letterbox",
        )

        self.assertTrue(np.allclose(data["tcp_pose_abs"][:, 0], [0.1, 0.2]))
        self.assertTrue(np.allclose(data["timestamp"], [30.0, 30.0 + 1.0 / 30.0]))
        self.assertTrue(np.allclose(data["action_abs_tcp"][:, 0], [0.1, 0.2]))
        self.assertTrue(np.allclose(data["action_abs_tcp"][:, 9], [0.08, 0.08]))
        self.assertTrue(np.allclose(data["gaze_xy"], [[0.5, 0.5], [0.5, 0.5]]))
        self.assertTrue(np.all(data["has_gaze_label"]))
        self.assertTrue(np.allclose(data["gaze_world_pc"], [[0.0, 0.0, 1.0]] * 2))
        self.assertTrue(np.all(data["gaze_projection_status"] == MODULE.GAZE_PROJECTION_VALID))
        self.assertFalse(np.any(data["has_heatmap_image"]))
        self.assertTrue(np.array_equal(meta.values["episode_ends"], [1, 2]))


if __name__ == "__main__":
    unittest.main()
