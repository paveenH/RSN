#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024
# Author: paveenhuang

# Define the list of tasks
TASKS=(
"abstract_algebra"
"anatomy"
"astronomy"
"business_ethics"
"clinical_knowledge"
"college_biology"
"college_chemistry"
"college_computer_science"
"college_medicine"
"college_mathematics"
"college_physics"
"computer_security"
"conceptual_physics"
"econometrics"
"electrical_engineering"
"elementary_mathematics"
"formal_logic"
"global_facts"
"high_school_biology"
"high_school_chemistry"
"high_school_computer_science"
"high_school_european_history"
"high_school_geography"
"high_school_government_and_politics"
"high_school_macroeconomics"
"high_school_mathematics"
"high_school_microeconomics"
"high_school_physics"
"high_school_psychology"
"high_school_statistics"
"high_school_us_history"
"high_school_world_history"
"human_aging"
"human_sexuality"
"international_law"
"jurisprudence"
"logical_fallacies"
"machine_learning"
"management"
"marketing"
"medical_genetics"
"miscellaneous"
"moral_disputes"
"moral_scenarios"
"nutrition"
"philosophy"
"prehistory"
"professional_accounting"
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

# Define model and size parameters
MODEL="llama3"
SIZE="8B"
START=1
END=31

# Define multiple neuron index lists (这里以单个 index 为例)
NEURON_LISTS=("4055" "2692" "2629" "1731")

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Automatically set number of parallel jobs based on available CPU cores
JOBS=1

# Prepare all combinations: for each TASK and each neuron index, 
# the combination string has six parts: task, model, size, start, end, neuron_indices.
COMBINATIONS=()
for TASK in "${TASKS[@]}"; do
    for INDEX in "${NEURON_LISTS[@]}"; do
        COMBINATIONS+=("$TASK $MODEL $SIZE $START $END $INDEX")
    done
done

# Debug: Print combinations
echo "Task-Model-Size-Start-End-NeuronIndices combinations to process:"
for COMBINATION in "${COMBINATIONS[@]}"; do
    echo "$COMBINATION"
done

# Ensure GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel could not be found. Please install it."
    exit 1
fi

# Execute using GNU parallel, calling the get_answer_lesion_sample.py script.
echo "Starting parallel execution with $JOBS job(s)..."
parallel -j "$JOBS" python3 /data2/paveen/RolePlaying/src/models/components/get_answer_lesion.py ::: "${COMBINATIONS[@]}"if [ $? -eq 0 ]; then
    echo "All tasks processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi