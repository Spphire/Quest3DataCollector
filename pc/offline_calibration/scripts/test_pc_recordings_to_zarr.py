from __future__ import annotations

import importlib.util
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

    def test_allowlist_requires_every_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "record_20260813_162120").mkdir()
            found = MODULE.find_record_dirs(root, {"record_20260813_162120"})
            self.assertEqual([path.name for path in found], ["record_20260813_162120"])
            with self.assertRaises(FileNotFoundError):
                MODULE.find_record_dirs(root, {"record_missing"})

    def test_time_gaps_become_episode_boundaries_and_short_segments_are_removed(self):
        samples = [
            {"pc_perf_counter_seconds": value}
            for value in (0.0, 0.033, 0.066, 1.0, 1.033, 2.0)
        ]
        kept, segment_ends, dropped = MODULE.split_samples_at_time_gaps(
            samples,
            max_gap_seconds=0.2,
            min_segment_frames=2,
        )

        self.assertEqual([row["pc_perf_counter_seconds"] for row in kept], [0.0, 0.033, 0.066, 1.0, 1.033])
        self.assertEqual(segment_ends, [3, 5])
        self.assertEqual(dropped, 1)

    def test_source_timestamp_gap_splits_even_when_aligned_timeline_is_continuous(self):
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

        kept, segment_ends, dropped = MODULE.split_samples_at_time_gaps(
            samples,
            max_gap_seconds=0.2,
            min_segment_frames=2,
        )

        self.assertEqual(kept, samples)
        self.assertEqual(segment_ends, [2, 4])
        self.assertEqual(dropped, 0)

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
                    "pc_perf_counter_seconds": 20.0 + index,
                    "robot_state_pc_perf_counter_seconds": 10.0 + index,
                    "quest_pc_receive_perf_counter_seconds": 19.0 + index,
                    "quest_gaze3d_pc_world": [0.0, 0.0, 1.0],
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
        )

        self.assertTrue(np.allclose(data["tcp_pose_abs"][:, 0], [0.1, 0.2]))
        self.assertTrue(np.allclose(data["action_abs_tcp"][:, 0], [0.1, 0.2]))
        self.assertTrue(np.allclose(data["action_abs_tcp"][:, 9], [0.08, 0.08]))
        self.assertTrue(np.allclose(data["gaze_xy"], [[0.5, 0.5], [0.5, 0.5]]))
        self.assertTrue(np.all(data["has_gaze_label"]))
        self.assertFalse(np.any(data["has_heatmap_image"]))
        self.assertTrue(np.array_equal(meta.values["episode_ends"], [1, 2]))


if __name__ == "__main__":
    unittest.main()
