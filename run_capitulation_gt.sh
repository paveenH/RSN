#!/bin/bash
# ==================== Capitulation Rate Experiment (Gold R1) ====================
# Round 1: ground-truth label used as R1 answer (model sees same prompt format)
# Round 2: RSN steering applied, pressure prompt appended
# Three conditions: no steering (0) / +4 / -4
#
# Key difference from run_capitulation.sh: --gold_r1 flag
# Output goes to answer_cap_mmlupro_gold/ (separate from original results)
#
# Models  : llama3-8B, qwen3-8B, mistral-7B
# Configs : llama3  → 0-11-20 / 4-11-20 / neg4-11-20
#           qwen3   → 0-17-26 / 4-17-26 / neg4-17-26
#           mistral → 0-14-22 / 4-14-22 / neg4-14-22
#
# Usage: bash run_capitulation_gt.sh

# ==================== Shared config ====================
TYPE="non"
PERCENTAGE=0.5
MASK_TYPE="nmd"
SUITE="default"
TAIL_LEN=1
ANS_FILE="answer_cap_mmlupro"

# ==================== Model configs ====================
# Format: MODEL|MODEL_DIR|HS|SIZE|CONFIGS
MODELS=(
    # "llama3|meta-llama/Llama-3.1-8B-Instruct|llama3|8B|0-11-20 4-11-20 neg4-11-20"
    "qwen3|Qwen/Qwen3-8B|qwen3|8B|0-17-26 4-17-26 neg4-17-26"
    "mistral|mistralai/Mistral-7B-Instruct-v0.3|mistral|7B|0-14-22 4-14-22 neg4-14-22"
)

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Run ====================
echo "=================================================="
echo "Capitulation Rate Experiment (Gold R1)"
echo "Start time: $(date)"
echo "=================================================="

cd "${WORK_DIR}"

for MODEL_CFG in "${MODELS[@]}"; do
    IFS="|" read -r MODEL MODEL_DIR HS SIZE CONFIGS <<< "${MODEL_CFG}"
    ORIG_DIR="${BASE_DIR}/mmlupro/${MODEL}"

    echo ""
    echo "####################################################"
    echo "# Model: ${MODEL} (${SIZE})"
    echo "# Configs: ${CONFIGS}"
    echo "# Orig dir: ${ORIG_DIR}"
    echo "# Mode: Gold R1 (ground-truth as Round 1 answer)"
    echo "####################################################"

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
        --base_dir   "${BASE_DIR}" \
        --gold_r1

    if [ $? -eq 0 ]; then
        echo ""
        echo "[✓ Done] Gold R1 Capitulation — ${MODEL} ${SIZE} finished at: $(date)"
    else
        echo ""
        echo "[✗ Failed] Gold R1 Capitulation — ${MODEL} ${SIZE}"
        exit 1
    fi

done

echo ""
echo "=================================================="
echo "All models (Gold R1) finished at: $(date)"
echo "=================================================="
