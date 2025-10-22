#!/usr/bin/env bash
set -euo pipefail

SUITE="default"
TYPE="non"
DATA="data1"

run_model () {
  local MODEL="$1"
  local MODEL_DIR="$2"
  local SIZE="$3"
  local EDIT="$4"

  echo "=== Running ${MODEL} on built-in MMLU TASKS ==="
  python get_answer_logits.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --model_dir "${MODEL_DIR}" \
    --size "${SIZE}" \
    --type "${TYPE}" \
    --ans_file "answer_orig_mmlu" \
    --suite "${SUITE}" \
  
  python get_answer_regenerate_logits.py \
    --data ${DATA}" \
    --model "${MODEL}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${MODEL}" \
    --size "${SIZE}" \
    --type ${TYPE}" \
    --percentage 0.5 \
    --configs ${EDIT}" \
    --suite "${SUITE}" \
    --mask_type nmd \
    --ans_file answer_mdf_mmlu \
    --tail_len 1 \

}

run_model "qwen3" "Qwen/Qwen3-8B" "8B" "3-17-26 4-17-26"
run_model "mistral" "mistralai/Mistral-7B-Instruct-v0.3" "7B" "3-14-22 4-14-22"

