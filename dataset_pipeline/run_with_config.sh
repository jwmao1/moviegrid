#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG_FILE COMMAND [ARG ...]" >&2
  exit 2
fi

CONFIG_FILE="$1"
shift

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Configuration file not found: ${CONFIG_FILE}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

required=(GRID_ROWS GRID_COLS CELL_WIDTH CELL_HEIGHT SHOT_FRAMES CHUNK_FRAMES GRID_DIR_NAME)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required configuration value: ${name}" >&2
    exit 2
  fi
done

if (( GRID_ROWS <= 0 || GRID_COLS <= 0 || CELL_WIDTH <= 0 || CELL_HEIGHT <= 0 || SHOT_FRAMES <= 0 || CHUNK_FRAMES <= 0 )); then
  echo "Grid, cell, and frame values must be positive integers" >&2
  exit 2
fi

expected_chunk_frames=$((GRID_ROWS * GRID_COLS * SHOT_FRAMES))
if (( CHUNK_FRAMES != expected_chunk_frames )); then
  echo "CHUNK_FRAMES=${CHUNK_FRAMES} must equal GRID_ROWS * GRID_COLS * SHOT_FRAMES (${expected_chunk_frames})" >&2
  exit 2
fi

exec "$@"
