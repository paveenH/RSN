#!/bin/bash

# ==================== Save MMLU Hidden States ====================
# Runs get_answer_logits.py --save for all 7 roles on MMLU tasks.
# Used to build classifier training data in ConfSteer.
#
# Roles (7):
#   neutral, {task} expert, non {task} expert,
#   confident, unconfident, student, person
#
# Output:
#   components/hidden_states_non/{model}/  — per-role per-task .npy
#   components/{model}/answer_hs_mmlu/     — answer JSON with logits
#
# Usage: bash run_save_hs_mmlu.sh [llama3|qwen3|both]
#   default: both

TARGET="${1:-both}"

# ==================== Paths ====================
WORK_DIR="/data1/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"
DATA="data1"
TYPE="non"
SUITE="default"

# 7 roles for MMLU (no {task} placeholder — MMLU tasks use underscore names)
ROLES="neutral,{task} expert,non {task} expert,confident,unconfident,student,person"

cd "${WORK_DIR}"

# ==================== llama3 ====================
run_llama3() {
    echo ""
    echo "=================================================="
    echo "  llama3 — Save MMLU Hidden States"
    echo "  Start: $(date)"
    echo "=================================================="

    python get_answer_logits.py \
        --model      "llama3" \
        --model_dir  "meta-llama/Llama-3.1-8B-Instruct" \
        --size       "8B" \
        --type       "${TYPE}" \
        --ans_file   "answer_hs_mmlu" \
        --suite      "${SUITE}" \
        --data       "${DATA}" \
        --base_dir   "${BASE_DIR}" \
        --roles      "${ROLES}" \
        --save

    if [ $? -eq 0 ]; then
        echo "[✓ Done] llama3 MMLU hidden states saved"
    else
        echo "[✗ Failed] llama3 MMLU hidden states"
        exit 1
    fi
}

# ==================== qwen3 ====================
run_qwen3() {
    echo ""
    echo "=================================================="
    echo "  qwen3 — Save MMLU Hidden States"
    echo "  Start: $(date)"
    echo "=================================================="

    python get_answer_logits.py \
        --model      "qwen3" \
        --model_dir  "Qwen/Qwen3-8B" \
        --size       "8B" \
        --type       "${TYPE}" \
        --ans_file   "answer_hs_mmlu" \
        --suite      "${SUITE}" \
        --data       "${DATA}" \
        --base_dir   "${BASE_DIR}" \
        --roles      "${ROLES}" \
        --save

    if [ $? -eq 0 ]; then
        echo "[✓ Done] qwen3 MMLU hidden states saved"
    else
        echo "[✗ Failed] qwen3 MMLU hidden states"
        exit 1
    fi
}

# ==================== Dispatch ====================
case "${TARGET}" in
    llama3) run_llama3 ;;
    qwen3)  run_qwen3  ;;
    both)
        run_llama3
        run_qwen3
        ;;
    *)
        echo "Usage: bash run_save_hs_mmlu.sh [llama3|qwen3|both]"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "All done: $(date)"
echo "=================================================="
