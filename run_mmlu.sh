#!/usr/bin/env bash
set -euo pipefail

SUITE="default"
TYPE="non"
DATA="data1"

run_model () {
  local MODEL="$1"
  local MODEL_DIR="$2"
  local SIZE="$3"

  echo "=== Running ${MODEL} on built-in MMLU TASKS ==="
  python get_answer_logits.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --model_dir "${MODEL_DIR}" \
    --size "${SIZE}" \
    --type "${TYPE}" \
    --ans_file "answer_orig_mmlu_cot" \
    --suite "${SUITE}" \
    --cot
}

run_model "qwen3" "Qwen/Qwen3-8B" "8B"
run_model "mistral" "mistralai/Mistral-7B-Instruct-v0.3" "7B"