#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${WAN_MODEL_DIR:?Set WAN_MODEL_DIR to the Wan2.2-TI2V-5B checkpoint directory}"
: "${DATASET_DIR:?Set DATASET_DIR to the cached 16-grid dataset directory}"

export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/16grid_lora}"
export PREFIX_EMBEDDING_PATH="${PREFIX_EMBEDDING_PATH:-${REPO_ROOT}/env/prefix_embeddings/wan22_16grid_prefix.pt}"
CONFIG="${CONFIG:-${REPO_ROOT}/env/configs/train_16grid_lora.toml}"
NUM_GPUS="${NUM_GPUS:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

if [[ ! -d "${WAN_MODEL_DIR}" ]]; then
  echo "WAN model directory does not exist: ${WAN_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory does not exist: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PREFIX_EMBEDDING_PATH}" ]]; then
  echo "Prefix embedding does not exist: ${PREFIX_EMBEDDING_PATH}" >&2
  echo "Generate it with env/generate_wan22_prefix_embedding.py first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}/diffusion-pipe"
exec "${DEEPSPEED_BIN}" --num_gpus "${NUM_GPUS}" train.py --deepspeed --config "${CONFIG}" "$@"
