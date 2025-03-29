#!/bin/bash
# Example: run multiple tasks and append results to CSV files per role

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

# You can also set other parameters if needed:
MODEL="llama3"
SIZE="8B"

# Loop through each task and run the Python script
for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK"
    python3 get_logits.py "$TASK $MODEL $SIZE"
    if [ $? -ne 0 ]; then
        echo "Error processing task: $TASK"
        exit 1
    fi
done

echo "All tasks processed."