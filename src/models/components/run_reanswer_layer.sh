#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024 (Updated on $(date))
# Author: paveenhuang

TASKS=(
"professional_law"
"professional_medicine"
"professional_psychology"
"public_relations"
"security_studies"
"sociology"
"us_foreign_policy"
"virology"
"world_religions"
)


SIZES=("7B")
MODELS=("mistral")
TOPS=("20")
ALPHAS=("1")
START_END_PAIRS=("22 32")

 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the combinations: TASK MODEL SIZE TOP ALPHA start end
COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SIZE in "${SIZES[@]}"; do
            for TOP in "${TOPS[@]}"; do
                for ALPHA in "${ALPHAS[@]}"; do
                    for PAIR in "${START_END_PAIRS[@]}"; do
                        COMBINATIONS+=("$TASK $MODEL $SIZE $TOP $ALPHA $PAIR")
                    done
                done
            done
        done
    done
done

# Print combinations for debugging
echo "Task-Model-Size-Top-Alpha-Start-End Combinations to Process:"
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "$COMBINATION"
done

# Execute the Python script for each combination sequentially
echo "Starting sequential execution..."
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "Processing: $COMBINATION"
    python3 /data2/paveen/RolePlaying/src/models/components/get_answer_regenerate_layer.py "$COMBINATION"
    if [ $? -ne 0 ]; then
        echo "Error processing: $COMBINATION"
        exit 1
    fi
done

echo "All tasks have been processed successfully."