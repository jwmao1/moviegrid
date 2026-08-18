#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <video_list_file> <out_root> <session_prefix> [gpu_count]" >&2
  exit 1
fi

VIDEO_LIST_FILE="$1"
OUT_ROOT="$2"
SESSION_PREFIX="$3"
GPU_COUNT="${4:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET_ROOT:?Set DATASET_ROOT to the generated VLM-video directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/total_vlm.py}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

pids=()
for i in $(seq 0 $((GPU_COUNT - 1))); do
  log="${LOG_ROOT}/${SESSION_PREFIX}_g${i}.log"
  DATASET_ROOT="$DATASET_ROOT" \
  VIDEO_LIST_FILE="$VIDEO_LIST_FILE" \
  OUT_ROOT="$OUT_ROOT" \
  SHARD_COUNT="$GPU_COUNT" \
  SHARD_INDEX="$i" \
  OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}" \
  TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES="$i" \
  "$PYTHON_BIN" -u "$SCRIPT_PATH" >"$log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  echo "Started shard $i on GPU $i (PID $pid), log: $log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
