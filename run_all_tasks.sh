#!/bin/bash
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

# Define the log storage directory
COLLECTED_DIR="./logs/eval/collected_metrics/"
mkdir -p "$COLLECTED_DIR"

run_task() {
    task=$1
    echo "Running tasks: $task"
    python3 src/eval.py model=text_otf data=mmlu data.dataset_partial.task="$task"
    
    # Find the run directory corresponding to the task
    run_dir=$(ls -td /data2/paveen/RolePlaying/logs/eval/runs/* | head -n 1)
    
    metrics_path="$run_dir/csv/version_0/metrics.csv"
    
    # Check if metrics.csv exists
    if [ -f "$metrics_path" ]; then
        # Copy and rename to [task].csv
        cp "$metrics_path" "$COLLECTED_DIR/$task.csv"
        echo "Saved: $COLLECTED_DIR/$task.csv"
    else
        echo "metrics.csv lsot: $metrics_path"
    fi
}

export -f run_task
export COLLECTED_DIR

# Run each task in a loop
for task in "${tasks[@]}"
do
    run_task "$task" &
    current_jobs=$((current_jobs + 1))
    
    # If the maximum number of concurrent tasks is reached, wait for any background task to complete
    if [ "$current_jobs" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
done

wait

echo "All tasks have completed and collected the metrics.csv file."