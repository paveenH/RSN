#!/bin/bash
# ==================== NMD Mask Generation Script ====================
# This script generates NMD masks from mean hidden states.
# Can be run directly on login node (no GPU required).
#
# Usage: bash run_nmd.sh

# ==================== Configuration ====================
MODEL="qwen3"
SIZE="32B"
TYPE="non"
HS_PREFIX="qwen3_32b"       # Hidden state folder prefix

# Percentage of neurons to keep per layer
PERCENTAGE=0.5              # 0.5% of hidden_dim

# Mask type: nmd / random / diff_random / sparse_fv
MASK_TYPE="nmd"

# Random seed (for random/diff_random)
SEED=42

# Layer range configs to generate masks for
# Qwen3-32B: 64 hidden layers + 1 embedding = 65 total layers
# Layer range candidates
LAYER_CONFIGS=(
    "16 48"
    "32 48"
    "1 65"
)

# ==================== Run ====================
cd /work/d12922004/RolePlaying/detection

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
        --base_dir "/work/d12922004/RolePlaying/components" \
        --logits

    echo "[Done] Mask for layers [${START_LAYER}, ${END_LAYER})"
    echo ""
done

echo "=================================================="
echo "All masks generated!"
echo "=================================================="
