# Quest3DataCollector

Quest 3 棋盘标定与录制工具集。

## 目录

- `pc/offline_calibration/scripts/`
  - `quest_pc_receiver.py`: PC 端 UDP receiver、web viewer、record 回放、gaze 深度诊断、Quest 原始 record 对齐分析
  - `calibrate_records_25mm.py`: 25mm 棋盘标定
  - `build_pc_recording_replay.py`: 独立回放构建脚本
- `unity/Assets/EyeTracking/`
  - Quest 端录制、telemetry、标定相关脚本和场景
- `docs/`
  - 开发说明和运行流程

## PC receiver

启动：

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py receive --host 0.0.0.0 --port 9100 --visualize --visualize-port 8765 --no-open-browser
```

它会写入：

- `pc_recordings/<recordId>/pc_telemetry_raw.jsonl`
- `pc_recordings/<recordId>/pc_samples.jsonl`
- `pc_recordings/<recordId>/pc_controllers.csv`
- `pc_recordings/<recordId>/pc_session_summary.json`

## Web viewer

- Live viewer: `http://127.0.0.1:8765/`
- PC record replay: `http://127.0.0.1:8765/recordings`

回放页支持：

- head / eyes / controllers / gaze3D / hit
- raw / median-depth / board-plane gaze 对比
- 拖拽旋转、滚轮缩放
- scrub 播放

## Quest record 对齐

手动拉 Quest 原始 record 后：

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py analyze --quest-record pc\offline_calibration\raw\record_YYYYMMDD_HHMMSS --pc-session pc\offline_calibration\pc_recordings\record_YYYYMMDD_HHMMSS
```

## Gaze 深度诊断

```powershell
py pc\offline_calibration\scripts\quest_pc_receiver.py diagnose-gaze-depth --pc-session pc\offline_calibration\pc_recordings\record_YYYYMMDD_HHMMSS --quest-record pc\offline_calibration\raw\record_YYYYMMDD_HHMMSS
```

这会比较：

- raw gaze 深度
- board 平面误差
- 逐帧深度跳变
- Quest 原始 `trajectory.jsonl` 与 PC `pc_samples.jsonl` 是否一致

## Quest 端

Unity 端 telemetry 由 `QuestRecordingTelemetrySender` 发送。当前约定：

- `gazePoint3DWorld` 直接来自 `EyeGazePose.vizObj.transform.position`
- `gazePoint3DSource` 标记实际来源
- controller / eye pose 以 Unity world meter 为准

## 注意

- `pc_recordings/`、`raw/`、`outputs/` 是运行产物，不要提交。
- `_001` 这类碎片 session 已在 receiver 侧做了过滤。
- 深度毛刺诊断优先看 `quest_pc_receiver.py diagnose-gaze-depth` 输出。

