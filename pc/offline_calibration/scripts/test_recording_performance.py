from __future__ import annotations

import io
import gzip
import json
import threading
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from flexiv_realsense_bridge import (
    ColorStreamWriter,
    DEFAULT_FFMPEG_NICE_LEVEL,
    DepthStreamWriter,
    FlexivRealSenseConfig,
    FlexivRealSenseManager,
    FlexivRobotClient,
    OnlineNumericStats,
    OnlineTimeSeriesStats,
    ROBOT_SESSION_CONTROL_RECORD_ONLY,
    RobotRealsenseSession,
    ee_pose_diversity,
    ffmpeg_encoder_command_prefix,
)
from bounded_teleop_probe import DelayedRedundantUdpSender
from quest_pc_receiver import (
    FixedRateLatestSampler,
    build_performance_audit,
    decode_quest_udp_datagram,
    note_quest_udp_wire_datagram,
    quest_udp_transport_summary,
    recording_chronology_key,
    robot_realsense_performance_summary,
)


class BufferSink:
    def __init__(self) -> None:
        self.byte_count = 0
        self.last_type: type[object] | None = None

    def write(self, data: object) -> int:
        view = memoryview(data)
        self.byte_count += view.nbytes
        self.last_type = type(data)
        return view.nbytes


class FakeProcess:
    def __init__(self, sink: BufferSink) -> None:
        self.stdin = sink


class RecordingPerformanceTests(unittest.TestCase):
    def test_recording_chronology_uses_record_id_instead_of_mtime_or_sample_count(self) -> None:
        records = [
            {"recordId": "record_20260722_190001", "source": "pc", "mtime": 300.0, "samples": 9000},
            {"recordId": "record_pc_calib_20260722_203000", "source": "raw", "mtime": 100.0, "samples": 10},
            {"recordId": "record_bounded_v3_20260722_200000", "source": "pc", "mtime": 200.0, "samples": 100},
        ]

        ordered = sorted(records, key=recording_chronology_key, reverse=True)

        self.assertEqual(
            [record["recordId"] for record in ordered],
            [
                "record_pc_calib_20260722_203000",
                "record_bounded_v3_20260722_200000",
                "record_20260722_190001",
            ],
        )

    def test_delayed_redundancy_sends_old_packets_before_current(self) -> None:
        class FakeSocket:
            def __init__(self) -> None:
                self.payloads: list[bytes] = []

            def sendto(self, data: bytes, _remote: object) -> int:
                self.payloads.append(data)
                return len(data)

        sock = FakeSocket()
        sender = DelayedRedundantUdpSender(
            sock,
            "127.0.0.1",
            9100,
            datagram_interval_seconds=0.0,
        )
        sender.send({"sequence": 1})
        sender.send({"sequence": 2})
        sender.send({"sequence": 3})

        sequences = [json.loads(data.decode("utf-8"))["sequence"] for data in sock.payloads]
        self.assertEqual(sequences, [1, 1, 2, 1, 2, 3])

    def test_unsent_near_target_does_not_advance_cartesian_dedup_baseline(self) -> None:
        client = FlexivRobotClient()
        baseline = [0.4, -0.1, 0.5, 1.0, 0.0, 0.0, 0.0]
        client.robot = object()
        client.motion_armed = True
        client.motion_last_target_pose = list(baseline)
        client.freedrive_hold_pose = list(baseline)
        client.mode_name_locked = lambda: "NRT_CARTESIAN_MOTION_FORCE"
        client.read_joint_pose_locked = lambda: [0.0] * 7

        sent_targets: list[list[float]] = []

        def send_target(_robot: object, target: list[float]) -> None:
            sent_targets.append(list(target))
            client.cartesian_last_send = {
                "durationSeconds": 0.001,
                "slow": False,
                "signature": "test",
            }

        client.send_cartesian_motion_force_compat = send_target
        first = list(baseline)
        first[0] += 0.0006
        second = list(baseline)
        second[0] += 0.0012

        first_result = client.send_cartesian_target(first, 0.04, True)
        second_result = client.send_cartesian_target(second, 0.04, True)

        self.assertFalse(first_result["commandSent"])
        self.assertEqual(client.motion_last_target_pose, second)
        self.assertTrue(second_result["commandSent"])
        self.assertEqual(sent_targets, [second])

    def test_ffmpeg_encoder_runs_at_lower_priority_on_linux(self) -> None:
        with (
            patch("flexiv_realsense_bridge.os", SimpleNamespace(name="posix")),
            patch("flexiv_realsense_bridge.shutil.which", return_value="/usr/bin/nice"),
        ):
            prefix = ffmpeg_encoder_command_prefix("/usr/bin/ffmpeg")

        self.assertEqual(
            prefix,
            ["/usr/bin/nice", "-n", str(DEFAULT_FFMPEG_NICE_LEVEL), "/usr/bin/ffmpeg"],
        )

    def test_ffmpeg_encoder_priority_falls_back_without_nice(self) -> None:
        with (
            patch("flexiv_realsense_bridge.os", SimpleNamespace(name="posix")),
            patch("flexiv_realsense_bridge.shutil.which", return_value=None),
        ):
            prefix = ffmpeg_encoder_command_prefix("/usr/bin/ffmpeg")

        self.assertEqual(prefix, ["/usr/bin/ffmpeg"])

    def test_gzip_udp_datagram_round_trip_stays_below_mtu(self) -> None:
        message = {
            "protocol": "quest_recording_telemetry_v1",
            "type": "sample",
            "sequence": 17,
            "pose": [0.123456789] * 140,
        }
        json_bytes = json.dumps(message, separators=(",", ":")).encode("utf-8")
        datagram = b"QGZ1" + gzip.compress(json_bytes, compresslevel=1)

        decoded, transport = decode_quest_udp_datagram(datagram)

        self.assertEqual(decoded, message)
        self.assertEqual(transport["encoding"], "gzip-json")
        self.assertEqual(transport["jsonBytes"], len(json_bytes))
        self.assertLess(transport["wireBytes"], 1472)

    def test_udp_transport_summary_counts_loss_order_and_wire_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            rows = [
                {
                    "udpTransport": {"encoding": "gzip-json", "wireBytes": 620, "jsonBytes": 2900},
                    "message": {"type": "recording_start", "sequence": 0},
                },
                {
                    "udpTransport": {"encoding": "gzip-json", "wireBytes": 710, "jsonBytes": 2950},
                    "message": {"type": "sample", "sequence": 2},
                },
            ]
            (session_dir / "pc_telemetry_raw.jsonl").write_text(
                "".join(f"{json.dumps(row)}\n" for row in rows),
                encoding="utf-8",
            )

            summary = quest_udp_transport_summary(session_dir)

            self.assertEqual(summary["missingSequenceCount"], 1)
            self.assertEqual(summary["firstMissingSequences"], [1])
            self.assertEqual(summary["duplicateSequenceCount"], 0)
            self.assertEqual(summary["reorderedSequenceCount"], 0)
            self.assertEqual(summary["wireBytes"]["max"], 710.0)

    def test_udp_deduplicator_ignores_delayed_redundant_copy(self) -> None:
        recent: dict[tuple[object, ...], float] = {}
        datagram = b"QGZ1" + gzip.compress(b'{"type":"sample","sequence":42}')

        self.assertFalse(note_quest_udp_wire_datagram(datagram, "10.0.0.2", 100.0, recent))
        self.assertTrue(note_quest_udp_wire_datagram(datagram, "10.0.0.2", 100.033, recent))
        self.assertFalse(note_quest_udp_wire_datagram(datagram + b"x", "10.0.0.2", 100.034, recent))

    def test_pose_diversity_bounds_pairwise_work_for_long_records(self) -> None:
        poses = []
        for index in range(600):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = index / 1000.0
            poses.append(pose)

        diversity = ee_pose_diversity(poses)

        self.assertEqual(diversity["samples"], 600)
        self.assertEqual(diversity["analyzedSamples"], 512)
        self.assertEqual(diversity["pairCount"], 512 * 511 // 2)
        self.assertEqual(diversity["totalPairCount"], 600 * 599 // 2)
        self.assertTrue(diversity["pairwiseDownsampled"])
        self.assertAlmostEqual(diversity["eeTranslationSpanM"], 0.599, places=6)

    def test_record_only_manager_never_arms_or_enters_freedrive(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.robot = object()
                self.arm_calls = 0
                self.freedrive_calls = 0

            def connect(self, *_args, **_kwargs):
                return {"connected": True}

            def arm_motion(self):
                self.arm_calls += 1
                return {}

            def enable_freedrive(self):
                self.freedrive_calls += 1
                return {}

            def status(self, **_kwargs):
                return {"connected": False}

        manager = FlexivRealSenseManager(
            FlexivRealSenseConfig(gripper_enabled=False),
            formal_control_mode=ROBOT_SESSION_CONTROL_RECORD_ONLY,
        )
        fake_robot = FakeRobot()
        manager.robot = fake_robot
        manager.warm_realsense_stream = lambda _timeout: {"ok": True}

        connected = manager.connect_robot({"waitSeconds": 0})

        self.assertTrue(connected["ok"])
        self.assertFalse(connected["motionCommandsAllowed"])
        self.assertIsNone(connected["motionWarmup"])
        self.assertEqual(fake_robot.arm_calls, 0)
        self.assertTrue(manager.arm_motion()["blocked"])
        self.assertTrue(manager.enable_freedrive()["blocked"])
        self.assertEqual(fake_robot.arm_calls, 0)
        self.assertEqual(fake_robot.freedrive_calls, 0)

    def test_record_only_session_close_does_not_restore_arm(self) -> None:
        class FakeRobot:
            def __init__(self) -> None:
                self.arm_calls = 0

            def arm_motion(self):
                self.arm_calls += 1
                return {}

        manager = FlexivRealSenseManager(formal_control_mode=ROBOT_SESSION_CONTROL_RECORD_ONLY)
        fake_robot = FakeRobot()
        manager.robot = fake_robot
        session = SimpleNamespace(
            control_mode=ROBOT_SESSION_CONTROL_RECORD_ONLY,
            close=lambda: {"ok": True, "closedReason": "recording_stop"},
        )
        manager.active_session = session

        summary = manager.stop_session(session)

        self.assertTrue(summary["ok"])
        self.assertEqual(fake_robot.arm_calls, 0)
        self.assertFalse(manager.config.controller_motion_enabled)

    def test_fixed_rate_latest_sampler_targets_30_hz(self) -> None:
        ticks: list[dict[str, object]] = []

        def callback(value: object, tick: dict[str, object]) -> object:
            ticks.append(dict(tick))
            return value

        sampler = FixedRateLatestSampler(
            30.0,
            callback,
            name="test-fixed-rate-sampler",
            initial_delay_seconds=0.008,
            fresh_source_wait_seconds=0.015,
        )
        sampler.start()
        submit_period = 1.0 / 90.0
        deadline = time.perf_counter() + 1.2
        next_submit = time.perf_counter()
        version = 0
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now >= next_submit:
                sampler.submit({"version": version})
                version += 1
                next_submit += submit_period
            time.sleep(0.001)
        sampler.stop(1.0)

        summary = sampler.summary()
        self.assertGreaterEqual(len(ticks), 32)
        self.assertLessEqual(len(ticks), 40)
        self.assertAlmostEqual(float(summary["scheduledHz"]), 30.0, places=6)
        self.assertGreater(float(summary["effectiveHz"]), 27.0)
        self.assertLess(float(summary["effectiveHz"]), 33.0)
        self.assertLessEqual(int(summary["missedTicks"]), 1)
        self.assertEqual(summary["initialDelaySeconds"], 0.008)
        self.assertEqual(summary["freshSourceWaitSeconds"], 0.015)

    def test_color_writer_uses_buffer_view_without_tobytes_copy(self) -> None:
        sink = BufferSink()
        writer = ColorStreamWriter.__new__(ColorStreamWriter)
        writer.width = 8
        writer.height = 4
        writer.process = FakeProcess(sink)
        writer.frame_count = 0
        frame = np.zeros((4, 8, 3), dtype=np.uint8)

        writer.write(frame)

        self.assertIs(sink.last_type, memoryview)
        self.assertEqual(sink.byte_count, frame.nbytes)
        self.assertEqual(writer.frame_count, 1)

    def test_depth_writer_uses_buffer_view_without_tobytes_copy(self) -> None:
        sink = BufferSink()
        writer = DepthStreamWriter.__new__(DepthStreamWriter)
        writer.width = 8
        writer.height = 4
        writer.encoding = "raw"
        writer.process = None
        writer.handle = sink
        writer.frame_count = 0
        writer.byte_count = 0
        frame = np.zeros((4, 8), dtype=np.uint16)

        info = writer.write(frame)

        self.assertIs(sink.last_type, memoryview)
        self.assertEqual(sink.byte_count, frame.nbytes)
        self.assertEqual(info["sourceByteLength"], frame.nbytes)
        self.assertEqual(writer.frame_count, 1)

    def test_robot_state_loop_targets_90_hz_fixed_deadlines(self) -> None:
        session = RobotRealsenseSession.__new__(RobotRealsenseSession)
        session.stop_event = threading.Event()
        session.config = SimpleNamespace(robot_state_hz=90.0)
        session.lock = threading.RLock()
        session.closed = False
        session.robot_states_handle = io.StringIO()
        session.robot_state_count = 0
        session.latest_robot_state_row = None
        session.robot_state_perf_stats = OnlineTimeSeriesStats()
        session.robot_state_target_perf_stats = OnlineTimeSeriesStats()
        session.robot_state_lateness_stats = OnlineNumericStats()
        session.robot_state_missed_tick_count = 0
        session.robot_state_error_count = 0
        session.last_robot_state_error = None
        session.last_error = None
        session._note_text_write_locked = lambda _key: None
        session._build_robot_state_row = lambda **kwargs: {
            "pc_perf_counter_seconds": kwargs["pc_perf_counter_seconds"],
            "pc_unix_seconds": kwargs["pc_unix_seconds"],
            "captured_at": kwargs["captured_at"],
        }

        thread = threading.Thread(target=session._robot_state_loop, daemon=True)
        thread.start()
        time.sleep(0.55)
        session.stop_event.set()
        thread.join(timeout=1.0)

        summary = session.robot_state_perf_stats.summary(90.0)
        target_summary = session.robot_state_target_perf_stats.summary(90.0)
        self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(session.robot_state_count, 44)
        self.assertLessEqual(session.robot_state_count, 54)
        self.assertAlmostEqual(float(target_summary["effectiveHz"]), 90.0, places=5)
        self.assertGreater(float(summary["effectiveHz"]), 85.0)
        self.assertLessEqual(session.robot_state_missed_tick_count, 1)

    def test_camera_rate_uses_capture_timestamps_instead_of_bursty_write_times(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            robot_dir = Path(temp_dir)
            rows = []
            for index in range(60):
                rows.append(
                    {
                        "role": "end",
                        "frame_index": index,
                        "frame_captured_perf_counter_seconds": index / 30.0,
                        "pc_perf_counter_seconds": 10.0 + index / 1000.0,
                        "write_duration_seconds": 0.001,
                    }
                )
            (robot_dir / "video_frames.jsonl").write_text(
                "".join(f"{json.dumps(row)}\n" for row in rows),
                encoding="utf-8",
            )
            (robot_dir / "robot_states.jsonl").write_text("", encoding="utf-8")
            (robot_dir / "samples.jsonl").write_text("", encoding="utf-8")

            performance = robot_realsense_performance_summary(
                robot_dir,
                {},
                {"fps": 30, "robotStateHz": 90, "recordDepth": False},
            )

            camera = performance["cameras"]["end"]["video"]
            self.assertAlmostEqual(float(camera["effectiveHz"]), 30.0, places=6)

    def test_performance_audit_accepts_stable_90_and_30_hz_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "record_test"
            session_dir.mkdir()
            replay_cache = session_dir / "replay_visualization.json"
            replay_cache.write_text("{}", encoding="utf-8")
            raw_rows = [
                {
                    "udpTransport": {"encoding": "gzip-json", "wireBytes": 640, "jsonBytes": 2920},
                    "message": {"type": "recording_start", "sequence": 0},
                },
                {
                    "udpTransport": {"encoding": "gzip-json", "wireBytes": 700, "jsonBytes": 2940},
                    "message": {"type": "sample", "sequence": 1},
                },
            ]
            (session_dir / "pc_telemetry_raw.jsonl").write_text(
                "".join(f"{json.dumps(row)}\n" for row in raw_rows),
                encoding="utf-8",
            )
            pc_summary = {
                "closedReason": "recording_stop",
                "writerDroppedMessages": 0,
                "writerError": None,
                "closeErrors": [],
                "replayVisualizationJson": str(replay_cache),
                "alignedSampler": {"missedTicks": 0},
            }
            robot_summary = {
                "status": "closed",
                "closeErrors": [],
                "cameraWriterThreadsAliveOnClose": {"end": False},
                "performance": {
                    "robotState": {
                        "count": 900,
                        "effectiveHz": 89.8,
                        "targetHz": 90.0,
                        "targetRatio": 89.8 / 90.0,
                        "gapSeconds": {"p95": 0.012},
                    },
                    "robotStateLatenessSeconds": {"p95": 0.002},
                    "robotStateMissedTicks": 0,
                    "alignedSamples": {
                        "count": 300,
                        "effectiveHz": 29.95,
                        "targetHz": 30.0,
                        "targetRatio": 29.95 / 30.0,
                        "gapSeconds": {"p95": 0.034},
                    },
                    "alignedSampleLatenessSeconds": {"p95": 0.004},
                    "alignedQuestSourceAgeSeconds": {"p95": 0.020},
                    "alignedReusedSourceSamples": 5,
                    "cameras": {
                        "end": {
                            "video": {
                                "count": 300,
                                "effectiveHz": 29.9,
                                "targetHz": 30.0,
                                "targetRatio": 29.9 / 30.0,
                            },
                            "queueDrops": 0,
                            "queueDropRatio": 0.0,
                            "captureToWriteLatencySeconds": {"p95": 0.030},
                        }
                    },
                },
            }
            thresholds = {
                "minRobotTargetRatio": 0.95,
                "maxRobotLatenessP95Seconds": 0.006,
                "minAlignedTargetRatio": 0.95,
                "maxAlignedLatenessP95Seconds": 0.016,
                "maxQuestSourceAgeP95Seconds": 0.060,
                "maxAlignedReusedSourceRatio": 0.10,
                "maxUdpSequenceLossRatio": 0.001,
                "maxUdpDatagramBytes": 1472,
                "minCameraTargetRatio": 0.95,
                "maxCameraDropRatio": 0.0,
                "maxCameraLatencyP95Seconds": 0.10,
            }

            audit = build_performance_audit(
                session_dir,
                robot_summary,
                thresholds,
                pc_summary=pc_summary,
            )

            self.assertTrue(audit["ok"], audit["summary"])
            checks = {check["id"]: check for check in audit["checks"]}
            self.assertTrue(checks["quest_udp_sequence_loss"]["ok"])
            self.assertTrue(checks["quest_udp_mtu"]["ok"])
            self.assertTrue(checks["aligned_quest_source_reuse"]["ok"])


if __name__ == "__main__":
    unittest.main()
