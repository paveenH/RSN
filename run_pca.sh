#!/bin/bash

MODEL="llama3"
SIZE="8B"
TYPE="non"
PCT="0.5"

BASE_ANS="python get_answer_regenerate_logits.py"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"

DATA="data1"

# ------------------------------
# Mask list
# ------------------------------
MASKS=(
    "ttest"
    "ttest_abs"
    "sparse_pca"
    "pca_selection"
    "dense_pca"
)

# ------------------------------
# Layer ranges for each mask
# ------------------------------
SPARSE_LAYERS=("11-20" "1-33")
DENSE_LAYERS=("11-20" "1-33")

# ------------------------------
# CONFIG mapping
# ------------------------------
declare -A CONFIGS
CONFIGS["11-20"]="4-11-20 3-11-20"
CONFIGS["1-33"]="1-1-33"

echo "========================= RUNNING MMLU STEERING ========================="

# ---------------------------------------------------------------------
# 1) Run sparse masks (ttest, ttest_abs, sparse_pca, pca_selection)
# ---------------------------------------------------------------------
for MASK in "sparse_pca" "pca_selection" "ttest" "ttest_abs" ; do
    for LAYER in "${SPARSE_LAYERS[@]}"; do

        CONFIG_SET=${CONFIGS[$LAYER]}

        for CFG in $CONFIG_SET; do
            echo ""
            echo ">>> [$MASK] LAYER=$LAYER  CONFIG=$CFG"

            $BASE_ANS \
                --data $DATA \
                --model $MODEL \
                --model_dir $MODEL_DIR \
                --hs $MODEL \
                --size $SIZE \
                --type $TYPE \
                --percentage $PCT \
                --configs $CFG \
                --mask_type $MASK \
                --ans_file mmlu_${MASK} \
                --suite default \
                --E
        done
    done
done

# ---------------------------------------------------------------------
# 2) Run dense_pca masks (percentage = 100)
# ---------------------------------------------------------------------
for LAYER in "${DENSE_LAYERS[@]}"; do

    CONFIG_SET=${CONFIGS[$LAYER]}

    for CFG in $CONFIG_SET; do
        echo ""
        echo ">>> [dense_pca] LAYER=$LAYER  CONFIG=$CFG"

        $BASE_ANS \
            --data $DATA \
            --model $MODEL \
            --model_dir $MODEL_DIR \
            --hs $MODEL \
            --size $SIZE \
            --type $TYPE \ d
            --percentage 100 \
            --configs $CFG \
            --mask_type dense_pca \
            --ans_file mmlu_dense_pca \
            --suite default \
            --E
    done
done

echo ""
echo "====================== ALL MMLU MASK RUNS FINISHED ======================"