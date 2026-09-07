from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("select_pc_recordings_for_gaze_wam.py")
SPEC = importlib.util.spec_from_file_location("select_pc_recordings_for_gaze_wam", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _report(robot=1.0, aligned=1.0, end=1.0, third=1.0, reused=10, count=100):
    checks = []
    for check_id, ratio in (
        ("robot_state_rate", robot),
        ("aligned_sample_rate", aligned),
    ):
        checks.append({"id": check_id, "performance": {"targetRatio": ratio}})
    for check_id, ratio in (("camera_end_rate", end), ("camera_third_rate", third)):
        checks.append({"id": check_id, "performance": {"video": {"targetRatio": ratio}}})
    return {
        "checks": checks,
        "performance": {
            "alignedReusedSourceSamples": reused,
            "alignedSamples": {"count": count},
        },
    }


class SelectPcRecordingsTest(unittest.TestCase):
    def _recording(self, flags, stale_tail=0):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        samples = root / "robot_realsense"
        samples.mkdir()
        rows = []
        for index, reused in enumerate(flags):
            rows.append({
                "aligned_source_reused": reused,
                "pc_perf_counter_seconds": index / 30.0,
                "quest_pc_receive_perf_counter_seconds": (
                    index / 30.0 - 0.1 if index >= len(flags) - stale_tail else index / 30.0
                ),
            })
        (samples / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return temp_dir, root

    def test_reuse_profile_accepts_five_rejects_six_internal_frames(self):
        holder, root = self._recording([False] * 20 + [True] * 5 + [False] * 75)
        try:
            profile = MODULE._reuse_profile(root, max_consecutive_reuse=5)
            self.assertEqual(profile["max_internal_consecutive_reuse"], 5)
            reasons, _ = MODULE.qualify_report(_report(reused=5, count=100), recording_dir=root)
            self.assertEqual(reasons, [])
        finally:
            holder.cleanup()

        holder, root = self._recording([False] * 20 + [True] * 6 + [False] * 74)
        try:
            profile = MODULE._reuse_profile(root, max_consecutive_reuse=5)
            self.assertEqual(profile["max_internal_consecutive_reuse"], 6)
            reasons, _ = MODULE.qualify_report(_report(reused=6, count=100), recording_dir=root)
            self.assertIn("aligned_quest_consecutive_reuse", reasons)
        finally:
            holder.cleanup()

    def test_long_terminal_reuse_is_trimmed_before_selection(self):
        holder, root = self._recording([False, False] + [True] * 60, stale_tail=60)
        try:
            profile = MODULE._reuse_profile(root, max_consecutive_reuse=5)
            self.assertEqual(profile["trim_end_frames"], 60)
            self.assertEqual(profile["max_internal_consecutive_reuse"], 0)
        finally:
            holder.cleanup()

    def test_thresholds_are_inclusive(self):
        reasons, metrics = MODULE.qualify_report(
            _report(robot=0.95, aligned=0.95, end=0.95, third=0.95, reused=10, count=100)
        )
        self.assertEqual(reasons, [])
        self.assertAlmostEqual(metrics["aligned_quest_source_reuse"], 0.10)

    def test_reports_every_failed_contract_metric(self):
        reasons, _metrics = MODULE.qualify_report(
            _report(robot=0.94, aligned=0.93, end=0.92, third=0.91, reused=11, count=100)
        )
        self.assertEqual(
            reasons,
            [
                "robot_state_rate",
                "aligned_sample_rate",
                "camera_end_rate",
                "camera_third_rate",
                "aligned_quest_source_reuse",
            ],
        )


if __name__ == "__main__":
    unittest.main()
