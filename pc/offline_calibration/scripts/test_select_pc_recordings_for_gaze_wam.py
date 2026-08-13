from __future__ import annotations

import importlib.util
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
