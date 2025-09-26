#!/bin/bash
# run_all.sh — Batch run for Hermes Original & MDF (7 datasets)
set -euo pipefail

# ====== Model and common parameters ======
MODEL="hermes"
MODEL_DIR="NousResearch/Hermes-3-Llama-3.1-8B"
SIZE="8B"
TYPE="non"

DATA_ORIG="data2"   # dataset setting for original runs
DATA_MDF="data2"    # dataset setting for MDF runs (change to data2 if needed)
HS="hermes"         # Hermes is based on Llama3

# ====== Configs: factor dataset uses different configs ======
CFG_DEFAULT=("4-11-20" "3-11-20" "neg4-11-20")

# ====== 7 datasets ======
FILE="mmlupro/mmlupro_test.json"
mkdir -p answer

NAME="$(echo "$FILE" | cut -d'/' -f1)"     # top-level directory name: medqa/pubmedqa/factor/...
SAFE_NAME="$NAME"                          # can customize if more fine-grained naming is needed

echo "=============================="
echo "=== Running dataset: $FILE"
echo "=============================="

: <<'EOF'
# ---------- ORIGINAL: no COT ----------
echo "[ORIG no-cot] $SAFE_NAME"
python get_answer_logits_mmlupro.py \
  --data "$DATA_ORIG" \
  --model "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --size "$SIZE" \
  --type "$TYPE" \
  --test_file "$FILE" \
  --ans_file "answer/answer_orig_${SAFE_NAME}"

# ---------- ORIGINAL: with COT ----------
echo "[ORIG with COT] $SAFE_NAME"
python get_answer_logits_mmlupro.py \
  --data "$DATA_ORIG" \
  --model "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --size "$SIZE" \
  --type "$TYPE" \
  --test_file "$FILE" \
  --ans_file "answer/answer_orig_${SAFE_NAME}_cot" \
  --cot

# ---------- MDF (edited version) ----------
CONFIGS=("${CFG_DEFAULT[@]}")

echo "[MDF] $SAFE_NAME  (configs: ${CONFIGS[*]})"
python get_answer_regenerate_logits_mmlupro.py \
  --data "$DATA_MDF" \
  --model "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --hs "$HS" \
  --size "$SIZE" \
  --type "$TYPE" \
  --percentage 0.5 \
  --configs "${CONFIGS[@]}" \
  --mask_type nmd \
  --test_file "$FILE" \
  --ans_file "answer/answer_mdf_${SAFE_NAME}" \
  --tail_len 1
EOF

# ===== MMLU: original =====
echo "=== MMLU original ==="
python get_answer_logits.py \
  --data "$DATA_ORIG" \
  --model "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --size "$SIZE" \
  --type "$TYPE" \
  --ans_file answer/answer_orig_mmlu


# ===== MMLU: MDF (edits) =====
# Configs used: neg4-11-20 neg3-11-20
echo "=== MMLU MDF (edits) ==="
python get_answer_regenerate_logits.py \
  --data "$DATA_MDF" \
  --model "$MODEL" \
  --model_dir "$MODEL_DIR" \
  --hs "$HS" \
  --size "$SIZE" \
  --type "$TYPE" \
  --percentage 0.5 \
  --configs "${CFG_DEFAULT[@]}" \
  --mask_type nmd \
  --ans_file answer/answer_mdf_mmlu \
  --tail_len 1
echo "✅ All Hermes runs finished."