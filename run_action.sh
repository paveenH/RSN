#!/bin/bash
# run_actions.sh
# Requirements:
# 1) Only llama3: Effect of COT on action (mmlu & mmlupro)
# 2) qwen3, mistral: Add original action logits (no COT; mmlu & mmlupro)
# 3) qwen3, mistral: Add “modified” action logits (mdf; mmlu & mmlupro)

set -euo pipefail

# Prepare directories (remove if your scripts create them automatically)
mkdir -p answer

############################
# Common parameters (edit if needed)
TYPE="non"
SUITE="default"
MMLUPRO_TEST="mmlupro/mmlupro_test.json"

# Model definitions
LLAMA3_DIR="meta-llama/Llama-3.1-8B-Instruct"
LLAMA3_SIZE="8B"

QWEN3_DIR="Qwen/Qwen3-8B"
QWEN3_SIZE="8B"

MISTRAL_DIR="mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_SIZE="7B"

############################
# Part 1: llama3 only — COT (mmlu & mmlupro)
echo "=== Part 1: llama3 with COT (mmlu & mmlupro) ==="

python get_action_logits.py \
  --model llama3 \
  --model_dir "${LLAMA3_DIR}" \
  --size "${LLAMA3_SIZE}" \
  --type "${TYPE}" \
  --ans_file action_mmlu_cot \
  --cot

python get_action_logits_pro.py \
  --model llama3 \
  --model_dir "${LLAMA3_DIR}" \
  --size "${LLAMA3_SIZE}" \
  --type "${TYPE}" \
  --test_file "${MMLUPRO_TEST}" \
  --ans_file action_mmlupro_cot \
  --cot

############################
# Part 2: qwen3, mistral — original (no COT; mmlu & mmlupro)
echo "=== Part 2: qwen3 & mistral originals (no COT) ==="

# qwen3 — mmlu
python get_action_logits.py \
  --model qwen3 \
  --model_dir "${QWEN3_DIR}" \
  --size "${QWEN3_SIZE}" \
  --type "${TYPE}" \
  --ans_file action_mmlu

# qwen3 — mmlupro
python get_action_logits_pro.py \
  --model qwen3 \
  --model_dir "${QWEN3_DIR}" \
  --size "${QWEN3_SIZE}" \
  --type "${TYPE}" \
  --test_file "${MMLUPRO_TEST}" \
  --ans_file action_mmlupro

# mistral — mmlu
python get_action_logits.py \
  --model mistral \
  --model_dir "${MISTRAL_DIR}" \
  --size "${MISTRAL_SIZE}" \
  --type "${TYPE}" \
  --ans_file action_mmlu

# mistral — mmlupro
python get_action_logits_pro.py \
  --model mistral \
  --model_dir "${MISTRAL_DIR}" \
  --size "${MISTRAL_SIZE}" \
  --type "${TYPE}" \
  --test_file "${MMLUPRO_TEST}" \
  --ans_file action_mmlupro

############################
# Part 3: qwen3, mistral — Modified (mdf; mmlu & mmlupro)
# Notes:
# - ans_file is fixed as action_mdf_mmlu / action_mdf_mmlupro (per your request, no extra info added)
# - Using your specified configs, mask_type, percentage, tail_len
echo "=== Part 3: qwen3 & mistral MDF (edits) ==="

# qwen3 — mmlu
python get_action_regenerate_logits.py \
  --model qwen3 \
  --model_dir "${QWEN3_DIR}" \
  --hs qwen3 \
  --size "${QWEN3_SIZE}" \
  --type "${TYPE}" \
  --percentage 0.5 \
  --configs 4-17-26 3-17-26 neg4-17-26 \
  --mask_type nmd \
  --ans_file action_mdf_mmlu \
  --tail_len 1

# mistral — mmlu
python get_action_regenerate_logits.py \
  --model mistral \
  --model_dir "${MISTRAL_DIR}" \
  --hs mistral \
  --size "${MISTRAL_SIZE}" \
  --type "${TYPE}" \
  --percentage 0.5 \
  --configs 4-14-22 3-14-22 neg4-14-22 \
  --mask_type nmd \
  --ans_file action_mdf_mmlu \
  --tail_len 1

# qwen3 — mmlupro
python get_action_regenerate_logits_pro.py \
  --model qwen3 \
  --model_dir "${QWEN3_DIR}" \
  --hs qwen3 \
  --size "${QWEN3_SIZE}" \
  --type "${TYPE}" \
  --percentage 0.5 \
  --configs 4-17-26 3-17-26 neg4-17-26 \
  --mask_type nmd \
  --test_file "${MMLUPRO_TEST}" \
  --ans_file action_mdf_mmlupro

# mistral — mmlupro
python get_action_regenerate_logits_pro.py \
  --model mistral \
  --model_dir "${MISTRAL_DIR}" \
  --hs mistral \
  --size "${MISTRAL_SIZE}" \
  --type "${TYPE}" \
  --percentage 0.5 \
  --configs 4-14-22 3-14-22 neg4-14-22 \
  --mask_type nmd \
  --test_file "${MMLUPRO_TEST}" \
  --ans_file action_mdf_mmlupro

echo "✅ All runs finished."