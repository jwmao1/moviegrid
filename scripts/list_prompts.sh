#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_DIR="$(cd "${SCRIPT_DIR}/../env/inference_prompts" && pwd)"

find "${PROMPT_DIR}" -mindepth 2 -maxdepth 2 -type f -name '*.txt' -printf '%P\n' | sort
