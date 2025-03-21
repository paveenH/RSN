#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024 (Updated on $(date))
# Author: paveenhuang

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

SIZES=("8B")
MODELS=("llama3")
TOPS=("10" "20")
ALPHAS=("1.1" "1.3" "1.5" "2" "2.5" "3" )

# Define explicit start-end pairs (1-based indexing)
START_END_PAIRS=("11 32")

JOBS=1

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

# Execute using GNU parallel
echo "Starting parallel execution with $JOBS jobs..."
parallel -j "$JOBS" python3 /data2/paveen/RolePlaying/src/models/components/get_answer_regenerate_layer.py ::: "${COMBINATIONS[@]}"

# Check if parallel execution is successful
if [ $? -eq 0 ]; then
    echo "All tasks have been processed successfully."
else
    echo "An error occurred during parallel execution."
    exit 1
fi