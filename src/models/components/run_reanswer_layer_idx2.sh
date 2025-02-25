#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024 (Updated on $(date))
# Author: paveenhuang
# This script calls get_answer_regenerate_index.py and calculates specified layers based on pre-defined start-end pairs.
# We directly define the start and end values as follows:
#   Group 1: 1 to 5, Group 2: 5 to 9, ..., Group 8: 29 to 33.

TASKS=(
"abstract_algebra"
"anatomy"
"astronomy"
"econometrics"
"global_facts"
"jurisprudence"
)

SIZES=("8B")
MODELS=("llama3")
TOPS=("20")
ALPHAS=("1")

# Define explicit start-end pairs (1-based indexing)
START_END_PAIRS=("0 31")

JOBS=1

# Full ablation list (frequency-based sorted order)
# ABLATION_FULL="2629,2692,4055,1731,373,133,291,873,2352,1298,2646,1189,3695,3585,2932,630,1421,3516,2265,3076,384,761,2977"
ABLATION_FULL="2977,761,384,3076,2265,3516,1421,630,2932,3585,3695,1189,2646,1298,2352,873,291,133,373,1731,4055,2692,2629"
# Split ABLATION_FULL into an array
IFS=',' read -ra ABLATION_ARR <<< "$ABLATION_FULL"
NUM_ABLATION=${#ABLATION_ARR[@]}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the base combinations: TASK MODEL SIZE TOP ALPHA start end
BASE_COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SIZE in "${SIZES[@]}"; do
            for TOP in "${TOPS[@]}"; do
                for ALPHA in "${ALPHAS[@]}"; do
                    for PAIR in "${START_END_PAIRS[@]}"; do
                        BASE_COMBINATIONS+=("$TASK $MODEL $SIZE $TOP $ALPHA $PAIR")
                    done
                done
            done
        done
    done
done

# Now, for each base combination, generate new combinations with ablation_list variants.
COMBINATIONS=()
for BASE in "${BASE_COMBINATIONS[@]}"; do
    # For each possible number from 1 to NUM_ABLATION, construct an ablation_list
    for ((i=1; i<=NUM_ABLATION; i++)); do
        # Generate ablation_list string consisting of the first i elements
        ablation_list=$(IFS=','; echo "${ABLATION_ARR[*]:0:i}")
        # Append the ablation_list as the eighth parameter to the BASE combination
        COMBINATIONS+=("$BASE $ablation_list")
    done
done

# Print combinations for debugging
echo "Task-Model-Size-Top-Alpha-Start-End-AblationList Combinations to Process:"
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "$COMBINATION"
done

# Execute using GNU parallel
echo "Starting parallel execution with $JOBS jobs..."
parallel -j "$JOBS" python3 /data2/paveen/RolePlaying/src/models/components/get_answer_regenerate_index.py ::: "${COMBINATIONS[@]}"

# Check if parallel execution is successful
if [ $? -eq 0 ]; then
    echo "All tasks have been processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi