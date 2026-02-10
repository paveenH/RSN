#!/bin/bash
# ==================== MMLU Regenerate Local Run Script ====================
# For running on lab server (not NCHC)
# Model: Mistral3-8B-Reasoning (mistralai/Ministral-3-8B-Reasoning-2512)
#
# Usage: bash run_regenerate_mmlu_local.sh

# ==================== Configuration ====================
MODEL_NAME="mistral3"
MODEL_DIR="mistralai/Ministral-3-8B-Reasoning-2512"  # Direct HuggingFace download
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="mistral3"                            # Hidden state folder prefix
DATA="data1"

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5                                  # Must match the mask file

# Alpha and layer range configurations (format: alpha-start-end)
# Mistral3-8B layer range: [8,19)
CONFIGS="4-8-19 4-11-22 1-1-35"

# Roles
ROLES="{task} expert,non {task} expert"

# Output
ANS_FILE="answer_mdf_mmlue"
SUITE="default"
USE_E="--E"                                      # With E option (A-E)

# ==================== Paths ====================
WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Environment ====================
echo "=================================================="
echo "Start time: $(date)"
echo "=================================================="

# ==================== Run ====================
cd ${WORK_DIR}

echo "[Running] get_answer_regenerate_logits.py"
echo "Model: ${MODEL_NAME} (${MODEL_DIR})"
echo "Mask: ${MASK_TYPE}, Percentage: ${PERCENTAGE}%"
echo "Configs: ${CONFIGS}"
echo "Roles: ${ROLES}"

python get_answer_regenerate_logits.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --ans_file "${ANS_FILE}" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    ${USE_E}

echo "=================================================="
echo "Finished at: $(date)"
echo "=================================================="
