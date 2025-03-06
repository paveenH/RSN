
#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024 (Updated on $(date))
# Author: paveenhuang
# This script calls get_answer_regenerate_layer.py and calculates each individual layer (start-end) pairs.

TASKS=(
"abstract_algebra"
"anatomy"
"econometrics"
"global_facts"
"jurisprudence"
)

SIZES=("8B")
MODELS=("llama3")
TOPS=("20")
ALPHAS=("1")
START=0
END=31

# Define start-end layer pairs: [(0-1), (1-2), ..., (30-31)]
START_END_PAIRS=()
for ((i=START; i<END; i++)); do
    START_END_PAIRS+=("$i $((i+1))")
done

JOBS=2

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

# Execute using GNU parallel for better efficiency
echo "Starting parallel execution with $JOBS jobs..."
parallel -j "$JOBS" python3 /data2/paveen/RolePlaying/src/models/components/get_answer_regenerate_layer.py ::: "${COMBINATIONS[@]}"

# Check if parallel execution is successful
if [ $? -eq 0 ]; then
    echo "All tasks have been processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi