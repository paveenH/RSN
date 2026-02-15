#!/bin/bash
# ==================== Mean Diff Calculation Script ====================
# This script computes mean hidden states from inconsistent samples.
# Can be run directly on login node (no GPU required).
#
# Usage: bash run_mean_diff.sh

# ==================== Configuration ====================
MODEL="qwen3"
SIZE="32B"                          # Model size: 1B, 7B, 14B, 70B
TYPE="non"

# ==================== Paths ====================
USERNAME="d12922004"
WORK_DIR="/work/${USERNAME}/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Run ====================
cd ${WORK_DIR}/mean

echo "=================================================="
echo "Computing mean hidden states"
echo "Model: ${MODEL}, Size: ${SIZE}, Type: ${TYPE}"
echo "=================================================="

python mean_diff.py \
    --model "${MODEL}" \
    --size "${SIZE}" \
    --type "${TYPE}" \
    --base_dir "${BASE_DIR}"

echo "=================================================="
echo "Done!"
echo "=================================================="
