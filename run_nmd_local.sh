#!/bin/bash
# ==================== NMD Mask Generation Script (Local) ====================
# This script generates NMD masks from mean hidden states.
# For local server (not NCHC)
#
# Usage: bash run_nmd_local.sh

# ==================== Configuration ====================
MODEL="mistral3"
SIZE="8B"
TYPE="non"
HS_PREFIX="mistral3"        # Hidden state folder prefix
DATA="data1"

# Percentage of neurons to keep per layer
PERCENTAGE=0.5              # 0.5% of hidden_dim

# Mask type: nmd / random / diff_random / sparse_fv
MASK_TYPE="nmd"

# Random seed (for random/diff_random)
SEED=42

# Layer range configs to generate masks for
# Mistral3-8B: 35 hidden layers
LAYER_CONFIGS=(
    "8 19"
    "11 22"
    "1 35"
)

# ==================== Paths ====================
WORK_DIR="/${DATA}/paveen/RolePlaying"

# ==================== Run ====================
cd ${WORK_DIR}/detection

for CFG in "${LAYER_CONFIGS[@]}"; do
    read -r START_LAYER END_LAYER <<< "${CFG}"
    echo "=================================================="
    echo "Generating ${MASK_TYPE} mask"
    echo "Model: ${MODEL}, Size: ${SIZE}, Type: ${TYPE}"
    echo "Layer range: [${START_LAYER}, ${END_LAYER})"
    echo "Percentage: ${PERCENTAGE}%"
    echo "=================================================="

    python nmd.py \
        --model "${MODEL}" \
        --size "${SIZE}" \
        --type "${TYPE}" \
        --hs "${HS_PREFIX}" \
        --percentage "${PERCENTAGE}" \
        --start_layer "${START_LAYER}" \
        --end_layer "${END_LAYER}" \
        --mask_type "${MASK_TYPE}" \
        --seed "${SEED}" \
        --base_dir "${WORK_DIR}/components" \
        --logits

    echo "[Done] Mask for layers [${START_LAYER}, ${END_LAYER})"
    echo ""
done

echo "=================================================="
echo "All masks generated!"
echo "=================================================="
