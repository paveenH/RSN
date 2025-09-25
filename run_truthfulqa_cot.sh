#!/bin/bash
# run_all_tqa.sh

# model information
declare -A MODEL_DIRS
declare -A MODEL_SIZES

MODEL_DIRS["llama3"]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZES["llama3"]="8B"

MODEL_DIRS["qwen3"]="Qwen/Qwen3-8B"
MODEL_SIZES["qwen3"]="8B"

MODEL_DIRS["mistral"]="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SIZES["mistral"]="7B"

TYPE="non"
SUITE="default"

MODES=("mc1" "mc2")
MODELS=("llama3" "qwen3" "mistral")

for mode in "${MODES[@]}"; do
  for model in "${MODELS[@]}"; do
    MODEL_DIR=${MODEL_DIRS[$model]}
    SIZE=${MODEL_SIZES[$model]}
    echo "=== Running model=$model (size=$SIZE, dir=$MODEL_DIR) with mode=$mode ==="

    python get_answer_logits_tqa.py \
      --mode "$mode" \
      --model "$model" \
      --model_dir "$MODEL_DIR" \
      --size "$SIZE" \
      --type "$TYPE" \
      --ans_file "answer/answer_orig_${model}_cot" \
      --suite "$SUITE" \
      --cot
  done
done