#!/bin/bash
# run_all.sh

MODEL="qwen3"
MODEL_DIR="Qwen/Qwen3-8B"
SIZE="8B"
TYPE="non"
SUITE="default"

DATASETS=(
  "medqa/medqa_source_test.json"
  "pubmedqa/pubmedqa_labeled_train.json"
  "factor/factor_mc.json"
  "gpqa/gpqa_train.json"
  "arlsat/arlsat_all.json"
  "logiqa/logiqa_mrc.json"
)

for FILE in "${DATASETS[@]}"; do
  NAME=$(echo "$FILE" | cut -d'/' -f1)

  echo "=== Running $NAME ==="
  python get_answer_logits_mmlupro.py \
    --data data1 \
    --model "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --type "$TYPE" \
    --test_file "$FILE" \
    --ans_file "answer/answer_orig_${NAME}_cot" \
    --suite "$SUITE" \
    --cot
done