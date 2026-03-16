#!/bin/bash
# ==================== MMLU Local Run Script ====================
# For running on lab server (not NCHC)
# Model: Mistral3-8B-Reasoning (mistralai/Ministral-3-8B-Reasoning-2512)
#
# Usage: bash run_mmlu_local.sh

# ==================== Configuration ====================
MODEL_NAME="mistral3"
MODEL_DIR="mistralai/Ministral-3-8B-Reasoning-2512"  # Direct HuggingFace download
MODEL_SIZE="8B"
TYPE="non"
ROLES="{task} expert,non {task} expert"  # Expert/non-expert roles
ANS_FILE="answer_non_logits"             # Folder name for output answers
SUITE="default"
SAVE_HS="--save"                         # Save hidden states
USE_E="--use_E"                          # With E option (A-E)
DATA="data1"

# ==================== Paths ====================
# Adjust these paths for your local environment
WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Environment ====================
echo "=================================================="
echo "Start time: $(date)"
echo "=================================================="

# Check environment
# echo "Python: $(which python)"
# echo "Python version: $(python --version)"
# nvidia-smi

# ==================== Run ====================
cd ${WORK_DIR}

echo "[Running] get_answer_logits.py"
echo "Model: ${MODEL_NAME} (${MODEL_DIR})"
echo "Roles: ${ROLES}"
echo "Type: ${TYPE}, Suite: ${SUITE}"
echo "Save HS: ${SAVE_HS}, Use E: ${USE_E}"

python get_answer_logits.py \
    --data "${DATA}" \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --roles "${ROLES}" \
    --ans_file "${ANS_FILE}" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    ${SAVE_HS} \
    ${USE_E}

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
