#!/usr/bin/env bash
set -euo pipefail

ROOT="${QUEST3_DATA_COLLECTOR_ROOT:-/ssd1/shenyibo/Quest3DataCollector}"
PYTHON="${QUEST3_RECEIVER_PYTHON:-$ROOT/.venv312/bin/python}"
LOG="${QUEST3_RECEIVER_LOG:-$ROOT/receiver.log}"

HOST="${QUEST3_RECEIVER_HOST:-0.0.0.0}"
UDP_PORT="${QUEST3_RECEIVER_UDP_PORT:-9100}"
VIEW_HOST="${QUEST3_RECEIVER_VIEW_HOST:-0.0.0.0}"
VIEW_PORT="${QUEST3_RECEIVER_VIEW_PORT:-8765}"
FLEXIV_IFACE="${QUEST3_FLEXIV_NETWORK_INTERFACE:-192.168.2.108}"
END_CAMERA="${QUEST3_END_REALSENSE_SERIAL:-244222073667}"
THIRD_CAMERA="${QUEST3_THIRD_REALSENSE_SERIAL:-750612070265}"
ROBOT_STATE_HZ="${QUEST3_ROBOT_STATE_HZ:-90}"
DEPTH_EVERY="${QUEST3_RECORD_DEPTH_EVERY_N_FRAMES:-3}"
DEPTH_FORMAT="${QUEST3_RECORD_DEPTH_FORMAT:-ffv1}"
GRIPPER_DEVICE="${QUEST3_GRIPPER_DEVICE:-Robotiq-2F-85}"
GRIPPER_FORCE="${QUEST3_GRIPPER_FORCE_N:-40}"
GRIPPER_INIT_ON_ENABLE="${QUEST3_GRIPPER_INIT_ON_ENABLE:-0}"

usage() {
  cat <<'EOF'
Usage: start_lab_receiver.sh [--restart] [--status] [--foreground]

Starts the lab PC Quest receiver with the Flexiv + dual RealSense defaults.

Options:
  --restart     Stop only existing quest_pc_receiver.py receive processes first.
  --status      Print matching receiver processes and listening ports, then exit.
  --foreground  Run in the foreground instead of nohup background mode.

Environment overrides:
  QUEST3_DATA_COLLECTOR_ROOT
  QUEST3_RECEIVER_PYTHON
  QUEST3_RECEIVER_LOG
  QUEST3_FLEXIV_NETWORK_INTERFACE
  QUEST3_END_REALSENSE_SERIAL
  QUEST3_THIRD_REALSENSE_SERIAL
  QUEST3_ROBOT_STATE_HZ
  QUEST3_RECORD_DEPTH_EVERY_N_FRAMES
  QUEST3_RECORD_DEPTH_FORMAT
  QUEST3_GRIPPER_DEVICE
  QUEST3_GRIPPER_FORCE_N
  QUEST3_GRIPPER_INIT_ON_ENABLE
EOF
}

receiver_pids() {
  pgrep -f 'pc/offline_calibration/scripts/quest_pc_receiver.py receive' || true
}

print_status() {
  echo "Receiver processes:"
  pgrep -af 'pc/offline_calibration/scripts/quest_pc_receiver.py receive' || true
  echo
  echo "Listening ports:"
  ss -ltnup 2>/dev/null | grep -E ':8765|:9100|:9101' || true
}

stop_existing() {
  local pids
  pids="$(receiver_pids)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "$pids" | xargs -r kill
  for _ in {1..30}; do
    if [[ -z "$(receiver_pids)" ]]; then
      return 0
    fi
    sleep 0.2
  done
  echo "$pids" | xargs -r kill -9
}

restart=false
foreground=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart)
      restart=true
      ;;
    --status)
      print_status
      exit 0
      ;;
    --foreground)
      foreground=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$ROOT"

if [[ "$restart" == true ]]; then
  stop_existing
elif [[ -n "$(receiver_pids)" ]]; then
  echo "Receiver is already running. Use --restart to replace it." >&2
  print_status >&2
  exit 1
fi

cmd=(
  "$PYTHON"
  pc/offline_calibration/scripts/quest_pc_receiver.py
  receive
  --host "$HOST"
  --port "$UDP_PORT"
  --visualize
  --visualize-host "$VIEW_HOST"
  --visualize-port "$VIEW_PORT"
  --no-open-browser
  --flexiv-network-interface "$FLEXIV_IFACE"
  --realsense-serial "$END_CAMERA"
  --third-realsense-serial "$THIRD_CAMERA"
  --robot-state-hz "$ROBOT_STATE_HZ"
  --record-realsense-depth-every-n-frames "$DEPTH_EVERY"
  --record-realsense-depth-format "$DEPTH_FORMAT"
  --enable-gripper
  --gripper-device "$GRIPPER_DEVICE"
  --gripper-force "$GRIPPER_FORCE"
)

case "${GRIPPER_INIT_ON_ENABLE,,}" in
  1|true|yes|on)
    cmd+=(--gripper-init-on-enable)
    ;;
esac

if [[ "$foreground" == true ]]; then
  exec "${cmd[@]}"
fi

mkdir -p "$(dirname "$LOG")"
nohup "${cmd[@]}" >> "$LOG" 2>&1 < /dev/null &
echo "Started Quest receiver pid=$! log=$LOG"
