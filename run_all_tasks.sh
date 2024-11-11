#!/bin/bash

# Define all tasks
tasks=(
    "high_school_european_history"
    "business_ethics"
    "clinical_knowledge"
    "medical_genetics"
    "high_school_us_history"
    "high_school_physics"
    "high_school_world_history"
    "virology"
    "high_school_microeconomics"
    "econometrics"
    "college_computer_science"
    "high_school_biology"
    "abstract_algebra"
    "professional_accounting"
    "philosophy"
    "professional_medicine"
    "nutrition"
    "global_facts"
    "machine_learning"
    "security_studies"
    "public_relations"
    "professional_psychology"
    "prehistory"
    "anatomy"
    "human_sexuality"
    "college_medicine"
    "high_school_government_and_politics"
    "college_chemistry"
    "logical_fallacies"
    "high_school_geography"
    "elementary_mathematics"
    "human_aging"
    "college_mathematics"
    "high_school_psychology"
    "formal_logic"
    "high_school_statistics"
    "international_law"
    "high_school_mathematics"
    "high_school_computer_science"
    "conceptual_physics"
    "miscellaneous"
    "high_school_chemistry"
    "marketing"
    "professional_law"
    "management"
    "college_physics"
    "jurisprudence"
    "world_religions"
    "sociology"
    "us_foreign_policy"
    "high_school_macroeconomics"
    "computer_security"
    "moral_scenarios"
    "moral_disputes"
    "electrical_engineering"
    "astronomy"
    "college_biology"
)

# Define collection path
COLLECTED_DIR="./logs/collected_metrics_3b/"
mkdir -p "$COLLECTED_DIR"

# Defining log files
LOG_FILE="./logs/eval/collection.log"
touch "$LOG_FILE"

# Define the function to run the task
run_task() {
    task=$1
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Running task: $task" | tee -a "$LOG_FILE"
    
    # Get a directory listing before running
    before_run=$(ls -td /data2/paveen/RolePlaying/logs/eval/runs/*)
    
    # Run Python scripts
    python3 src/eval.py model=text_otf data=mmlu data.dataset_partial.task="$task"
    if [ $? -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Error running task: $task" | tee -a "$LOG_FILE"
        return 1
    fi

    # Make sure all files are written
    sleep 2

    # Get the directory listing after running
    after_run=$(ls -td /data2/paveen/RolePlaying/logs/eval/runs/*)
    
    # Find the newly created directory
    new_run_dir=$(comm -13 <(echo "$before_run") <(echo "$after_run") | head -n 1)
    
    if [ -z "$new_run_dir" ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] No new run directory found for task: $task" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] New run directory: $new_run_dir" | tee -a "$LOG_FILE"
    
    metrics_path="$new_run_dir/csv/version_0/metrics.csv"
    
    # Check if metrics.csv exists
    if [ -f "$metrics_path" ]; then
        # Copy and rename to [task].csv
        cp "$metrics_path" "$COLLECTED_DIR/$task.csv"
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Saved: $COLLECTED_DIR/$task.csv" | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] metrics.csv not found: $metrics_path" | tee -a "$LOG_FILE"
    fi
}

# Iterate over all tasks and run
for task in "${tasks[@]}"
do
    run_task "$task"
    if [ $? -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Skipping to next task due to error." | tee -a "$LOG_FILE"
    fi
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Completed collection of metrics.csv files." | tee -a "$LOG_FILE"