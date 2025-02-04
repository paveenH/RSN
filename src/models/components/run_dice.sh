#!/bin/bash

# model and size
declare -A model_sizes
model_sizes["llama3"]="1B 3B 8B"
model_sizes["mistral"]="7B"
model_sizes["qwen2.5"]="0.5B 3B 7B"
model_sizes["phi"]="3.8B"

for model in "${!model_sizes[@]}"; do
    for size in ${model_sizes[$model]}; do
        echo "Running ks_task.py for model: $model, size: $size"
        python3 ks_task.py "$model" "$size"
    done
done