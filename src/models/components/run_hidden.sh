#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024
# Author: paveenhuang

# === Define the list of tasks ===
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

# === Define the list of model sizes ===
# SIZES=("0.5B" "3B" "7B")
SIZES=("3B")

# === Define the model name ===
MODEL="llama3"

# === Get the directory where the script is located ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === Change to the script's directory ===
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }

# === Define the Python script's path ===
PYTHON_SCRIPT="get_hidden_states.py"

# === Check if the Python script exists ===
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in $SCRIPT_DIR."
    exit 1
fi

# === Check if Python3 is installed ===
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 could not be found. Please install Python3."
    exit 1
fi

# === Print task-size-model combinations ===
echo "Task-Size-Model Combinations to Process:"
for TASK in "${TASKS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        echo "  Task: $TASK, Size: $SIZE, Model: $MODEL"
    done
done

# === Execute the Python script sequentially ===
echo "Starting hidden state extraction sequentially..."

for TASK in "${TASKS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        echo "--------------------------------------------"
        echo "Processing Task: '$TASK' with Size: '$SIZE' and Model: '$MODEL'..."
        
        # Run the Python script with task, size, and model as arguments
        python3 "$PYTHON_SCRIPT" "$TASK" "$SIZE" "$MODEL"
        
        # Check if the Python script executed successfully
        if [ $? -eq 0 ]; then
            echo "Successfully extracted hidden states for Task: '$TASK', Size: '$SIZE', Model: '$MODEL'."
        else
            echo "An error occurred while extracting hidden states for Task: '$TASK', Size: '$SIZE', Model: '$MODEL'."
            exit 1
        fi
    done
done

echo "All hidden states have been extracted and saved successfully."
