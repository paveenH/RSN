#!/bin/bash

# Define task list
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

# Define model parameters
MODEL="llama3"    # Your model name, adjustable
SIZE="8B"        # Model size, adjustable
TOP=20            # Select top-K largest hidden states dimensions
ALPHA=1       # Impact factor
START=0          # Starting layer for modification
END=1            # Ending layer for modification

# Define the path to `get_hidden_states_mdf.py`
SCRIPT_PATH="/data2/paveen/RolePlaying/src/models/components/get_hidden_states_mdf.py"

# Loop through all tasks and run the Python script
for TASK in "${TASKS[@]}"; do
    echo "Running task: $TASK with model: $MODEL size: $SIZE"

    # Call Python script
    python "$SCRIPT_PATH" "$TASK $MODEL $SIZE $TOP $ALPHA $START $END"

    echo "Finished processing $TASK"
done

echo "All tasks completed!"