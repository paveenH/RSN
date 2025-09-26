#!/bin/bash
# run_all.sh — Batch run for Hermes Original & MDF (7 datasets)
set -euo pipefail

# ====== Model and common parameters ======
MODEL="hermes"
MODEL_DIR="NousResearch/Hermes-3-Llama-3.1-8B"
SIZE="8B"
TYPE="non"

DATA_ORIG="data1"   # dataset setting for original runs
DATA_MDF="data1"    # dataset setting for MDF runs (change to data2 if needed)
HS="hermes"         # Hermes is based on Llama3

# ====== Configs: factor dataset uses different configs ======
CFG_FACTOR=("4-11-20" "3-11-20" "neg4-11-20" "neg3-11-20")
CFG_DEFAULT=("4-11-20" "3-11-20" "neg4-11-20")

# ====== 7 datasets ======
DATASETS=(
  "medqa/medqa_source_test.json"
  "pubmedqa/pubmedqa_labeled_train.json"
  "factor/factor_mc.json"
  "gpqa/gpqa_train.json"
  "arlsat/arlsat_all.json"
  "logiqa/logiqa_mrc.json"
)

mkdir -p answer
: <<'EOF'
for FILE in "${DATASETS[@]}"; do
  NAME="$(echo "$FILE" | cut -d'/' -f1)"     # top-level directory name: medqa/pubmedqa/factor/...
  SAFE_NAME="$NAME"                          # can customize if more fine-grained naming is needed

  echo "=============================="
  echo "=== Running dataset: $FILE"
  echo "=============================="

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
  # Select configs: factor dataset uses CFG_FACTOR, others use CFG_DEFAULT
  if [[ "$NAME" == "factor" ]]; then
    CONFIGS=("${CFG_FACTOR[@]}")
  else
    CONFIGS=("${CFG_DEFAULT[@]}")
  fi

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
done
EOF

########################################
# Extra runs (not in the 7 datasets)   #
# All for Hermes: TQA (mc1/mc2) + MMLU #
########################################

# ===== TQA: original (mc1 & mc2) =====
# Note: keep the original style, write all original results
#       into the same ans_file (answer/answer_orig_tqa)
MODES=("mc1" "mc2")
: <<'EOF'
for mode in "${MODES[@]}"; do

  echo "=== TQA original (mode=$mode) ==="
  python get_answer_logits_tqa.py \
    --data "$DATA_ORIG" \
    --mode "$mode" \
    --model "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --ans_file answer/answer_orig_tqa 
    
  echo "=== TQA original (mode=$mode) ==="
  python get_answer_logits_tqa.py \
    --data "$DATA_ORIG" \
    --mode "$mode" \
    --model "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --size "$SIZE" \
    --ans_file answer/answer_orig_tqa_cot \
    --cot
EOF

for mode in "${MODES[@]}"; do
  echo "=== TQA MDF (edits) ==="
  python get_answer_regenerate_logits_tqa.py \
    --data "$DATA_MDF" \
    --mode "$mode" \
    --model "$MODEL" \
    --model_dir "$MODEL_DIR" \
    --hs "$HS" \
    --size "$SIZE" \
    --type "$TYPE" \
    --percentage 0.5 \
    --configs "${CFG_FACTOR[@]}" \
    --mask_type nmd \
    --ans_file answer/answer_mdf_tqa
done

echo "✅ All Hermes runs finished."