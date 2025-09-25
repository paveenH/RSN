#!/bin/bash
# run_actions.sh
# Requirements:
# 1) llama3 + qwen3 + mistral: Effect of COT on action (mmlu & mmlupro)
# 2) qwen3, mistral: Original (no COT; mmlu & mmlupro)
# 3) qwen3, mistral: Modified (mdf; mmlu & mmlupro)

set -euo pipefail

# ========= Common parameters =========
TYPE="non"
SUITE="default"
MMLUPRO_TEST="mmlupro/mmlupro_test.json"

# Model configs
declare -A MODEL_DIRS
declare -A MODEL_SIZES
declare -A MDF_CONFIGS  # MDF configs per model (different start/end)

MODEL_DIRS["llama3"]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZES["llama3"]="8B"
MDF_CONFIGS["llama3"]="4-11-20 3-11-20 neg4-11-20"

MODEL_DIRS["qwen3"]="Qwen/Qwen3-8B"
MODEL_SIZES["qwen3"]="8B"
MDF_CONFIGS["qwen3"]="4-17-26 3-17-26 neg4-17-26"

MODEL_DIRS["mistral"]="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SIZES["mistral"]="7B"
MDF_CONFIGS["mistral"]="4-14-22 3-14-22 neg4-14-22"

# Models participating in each part
MODELS_ALL=("llama3" "qwen3" "mistral")
MODELS_QM=("qwen3" "mistral")

# Optional: unified output directory
mkdir -p answer

# ========= Part 1: All models with COT (mmlu & mmlupro) =========
echo "=== Part 1: ALL models with COT (mmlu & mmlupro) ==="
for model in "${MODELS_ALL[@]}"; do
  MODEL_DIR=${MODEL_DIRS[$model]}
  SIZE=${MODEL_SIZES[$model]}
  echo "[COT] $model → mmlu"
  python get_action_logits.py \
    --model "$model" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --type "$TYPE" \
    --ans_file action_mmlu_cot \
    --cot

  echo "[COT] $model → mmlupro"
  python get_action_logits_pro.py \
    --model "$model" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --type "$TYPE" \
    --test_file "$MMLUPRO_TEST" \
    --ans_file action_mmlupro_cot \
    --cot
done

# ========= Part 2: qwen3 & mistral originals (no COT; mmlu & mmlupro) =========
echo "=== Part 2: qwen3 & mistral originals (no COT) ==="
for model in "${MODELS_QM[@]}"; do
  MODEL_DIR=${MODEL_DIRS[$model]}
  SIZE=${MODEL_SIZES[$model]}
  echo "[ORIG] $model → mmlu"
  python get_action_logits.py \
    --model "$model" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --type "$TYPE" \
    --ans_file action_mmlu

  echo "[ORIG] $model → mmlupro"
  python get_action_logits_pro.py \
    --model "$model" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --type "$TYPE" \
    --test_file "$MMLUPRO_TEST" \
    --ans_file action_mmlupro
done

# ========= Part 3: qwen3 & mistral MDF (edits; mmlu & mmlupro) =========
# Notes:
# - ans_file is fixed as action_mdf_mmlu / action_mdf_mmlupro (your code already separates by model path, no extra suffix needed)
# - Use percentage=0.5, mask_type=nmd, tail_len=1
echo "=== Part 3: qwen3 & mistral MDF (edits) ==="
for model in "${MODELS_QM[@]}"; do
  MODEL_DIR=${MODEL_DIRS[$model]}
  SIZE=${MODEL_SIZES[$model]}
  CONFIGS=${MDF_CONFIGS[$model]}

  echo "[MDF] $model → mmlu  (configs: $CONFIGS)"
  python get_action_regenerate_logits.py \
    --model "$model" \
    --model_dir "$MODEL_DIR" \
    --hs "$model" \
    --size "$SIZE" \
    --type "$TYPE" \
    --percentage 0.5 \
    --configs $CONFIGS \
    --mask_type nmd \
    --ans_file action_mdf_mmlu \
    --tail_len 1

  echo "[MDF] $model → mmlupro (configs: $CONFIGS)"
  python get_action_regenerate_logits_pro.py \
    --model "$model" \
    --model_dir "$MODEL_DIR" \
    --hs "$model" \
    --size "$SIZE" \
    --type "$TYPE" \
    --percentage 0.5 \
    --configs $CONFIGS \
    --mask_type nmd \
    --test_file "$MMLUPRO_TEST" \
    --ans_file action_mdf_mmlupro
done

echo "✅ All runs finished."