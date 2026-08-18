#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${ROOT:?Set ROOT to the directory containing the input case folders}"
: "${OUT_ROOT:?Set OUT_ROOT to the output directory}"

START_CASE_ID="${START_CASE_ID:-}"
END_CASE_ID="${END_CASE_ID:-}"
MAX_PARALLEL_CASES="${MAX_PARALLEL_CASES:-8}"
FFMPEG_EXE="${FFMPEG_EXE:-ffmpeg}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/grid_videos_runner.py}"

ROOT="$ROOT" \
OUT_ROOT="$OUT_ROOT" \
START_CASE_ID="$START_CASE_ID" \
END_CASE_ID="$END_CASE_ID" \
MAX_WORKERS="$MAX_PARALLEL_CASES" \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
FFMPEG_EXE="$FFMPEG_EXE" \
exec "$PYTHON_BIN" -u "$RUNNER"
