#!/bin/bash
# ==================== Capitulation Rate Experiment ====================
# Round 1: pre-computed original answers (answer_neutral)
# Round 2: RSN steering applied, pressure prompt appended
# Three conditions: no steering (0) / +4 / -4
#
# Usage: bash run_capitulation.sh

# ==================== Config ====================
MODEL="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
HS="llama3"
SIZE="8B"
TYPE="non"
PERCENTAGE=0.5
MASK_TYPE="nmd"
SUITE="default"
TAIL_LEN=1
CONFIGS="0-11-20 4-11-20 neg4-11-20"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"
ORIG_DIR="${BASE_DIR}/mmlupro/llama3"
ANS_FILE="answer_cap_mmlupro"

# ==================== Run ====================
echo "=================================================="
echo "Capitulation Rate — ${MODEL} ${SIZE}"
echo "Configs: ${CONFIGS}"
echo "Orig dir: ${ORIG_DIR}"
echo "Start time: $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_capitulation.py \
    --model      "${MODEL}" \
    --model_dir  "${MODEL_DIR}" \
    --hs         "${HS}" \
    --size       "${SIZE}" \
    --type       "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type  "${MASK_TYPE}" \
    --configs    ${CONFIGS} \
    --orig_dir   "${ORIG_DIR}" \
    --ans_file   "${ANS_FILE}" \
    --suite      "${SUITE}" \
    --tail_len   "${TAIL_LEN}" \
    --base_dir   "${BASE_DIR}"

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "[✓ Done] Capitulation Rate finished at: $(date)"
    echo "=================================================="
else
    echo ""
    echo "[✗ Failed] Capitulation Rate — check logs above"
    exit 1
fi
