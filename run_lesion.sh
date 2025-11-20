#!/bin/bash

PCTS=(0.5 1 3 5 10)

for P in "${PCTS[@]}"
do
    echo "=============================================="
    echo " Running complement ablation with percentage = $P"
    echo "=============================================="

    python get_answer_regenerate_logits_lesion_complement.py \
        --data data2 \
        --model llama3 \
        --model_dir meta-llama/Llama-3.1-8B-Instruct \
        --hs llama3 \
        --size 8B \
        --type non \
        --percentage $P \
        --configs 1-1-33 \
        --mask_type nmd \
        --ans_file mmlu_lesion_complement \
        --suite default \
        --E \
        --tail_len 1
done