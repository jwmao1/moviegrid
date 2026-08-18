#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 ADAPTER_DIR PROMPT_FILE OUTPUT_DIR [extra sampler arguments...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
ADAPTER_DIR="$1"
PROMPT_FILE="$2"
OUTPUT_DIR="$3"
shift 3

: "${WAN_REPO:?Set WAN_REPO to the official Wan2.2 source checkout}"
: "${WAN_MODEL_DIR:?Set WAN_MODEL_DIR to the Wan2.2-TI2V-5B checkpoint directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/env/sample_wan22_ti2v_lora.py" \
  --wan_repo "${WAN_REPO}" \
  --ckpt_dir "${WAN_MODEL_DIR}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --prompt_file "${PROMPT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
