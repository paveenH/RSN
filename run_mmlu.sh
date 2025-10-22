#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Common datasets
# ------------------------
DATASETS=(
  "medqa/medqa_source_test.json"
  "pubmedqa/pubmedqa_labeled_train.json"
)

SUITE="default"
TYPE="non"
DATA="data1"

run_model () {
  local MODEL="$1"
  local MODEL_DIR="$2"
  local SIZE="$3"

  for FILE in "${DATASETS[@]}"; do
    NAME="$(echo "$FILE" | cut -d'/' -f1)"   # e.g., medqa / pubmedqa

    echo "=== Running ${MODEL} on ${NAME} ==="
    python get_answer_logits.py \
      --data "${DATA}" \
      --model "${MODEL}" \
      --model_dir "${MODEL_DIR}" \
      --size "${SIZE}" \
      --type "${TYPE}" \
      --test_file "${FILE}" \
      --ans_file "answer/answer_orig_${NAME}_cot" \
      --suite "${SUITE}" \
      --cot
  done
}

# ------------------------
# Qwen3-8B
# ------------------------
run_model "qwen3" "Qwen/Qwen3-8B" "8B"

# ------------------------
# Mistral-7B
# ------------------------
run_model "mistral" "mistralai/Mistral-7B-Instruct-v0.3" "7B"
