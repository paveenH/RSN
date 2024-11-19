#!/bin/bash

# 1. Define the list of tasks to run
TASKS=(
    "abstract_algebra"
    "anatomy"
    "college_biology"
    "college_chemistry"
    "college_mathematics"
    "computer_security"
    "conceptual_physics"
    "electrical_engineering"
    "elementary_mathematics"
    "high_school_macroeconomics"
    "high_school_microeconomics"
    "high_school_world_history"
    "international_law"
    "jurisprudence"
    "machine_learning"
    "management"
    "marketing"
    "medical_genetics"
    "philosophy"
    "professional_psychology"
    "sociology"
)


# 2. Number of repetitions per task
REPEATS=10

# 3. Base directories
BASE_LOG_DIR="./logs"
BASE_EVAL_DIR="/data2/paveen/RolePlaying/logs/eval/runs"

# 4. Ensure base directories exist
mkdir -p "$BASE_LOG_DIR/eval"
mkdir -p "$BASE_LOG_DIR/collected_metrics"

# 5. Define the function to run a task
run_task() {
    local TASK_NAME=$1
    local RUN_INDEX=$2

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Running task: ${TASK_NAME}, iteration: ${RUN_INDEX}" | tee -a "$LOG_FILE"

    # Get a directory listing before running
    before_run=$(ls -td "$BASE_EVAL_DIR"/*)

    # Run Python evaluation script
    python3 src/eval.py model=text_otf data=mmlu data.dataset_partial.task="$TASK_NAME"
    if [ $? -ne 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Error running task: ${TASK_NAME}, iteration: ${RUN_INDEX}" | tee -a "$LOG_FILE"
        return 1
    fi

    # Ensure all files are written
    sleep 2

    # Get the directory listing after running
    after_run=$(ls -td "$BASE_EVAL_DIR"/*)

    # Find the newly created directory
    new_run_dir=$(comm -13 <(echo "$before_run") <(echo "$after_run") | head -n 1)

    if [ -z "$new_run_dir" ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] No new run directory found for task: ${TASK_NAME}, iteration: ${RUN_INDEX}" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] New run directory: $new_run_dir" | tee -a "$LOG_FILE"

    metrics_path="$new_run_dir/csv/version_0/metrics.csv"

    # Check if metrics.csv exists
    if [ -f "$metrics_path" ]; then
        # Define collected metrics directory for the task
        COLLECTED_DIR="${BASE_LOG_DIR}/collected_metrics_${TASK_NAME}"
        mkdir -p "$COLLECTED_DIR"

        # Copy and rename to [task]_[iteration].csv
        cp "$metrics_path" "$COLLECTED_DIR/${TASK_NAME}_${RUN_INDEX}.csv"
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Saved: $COLLECTED_DIR/${TASK_NAME}_${RUN_INDEX}.csv" | tee -a "$LOG_FILE"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] metrics.csv not found: $metrics_path" | tee -a "$LOG_FILE"
    fi

    # Copy all files from the run directory into a subfolder
    subfolder="${COLLECTED_DIR}/run_${RUN_INDEX}"
    mkdir -p "$subfolder"
    cp -r "$new_run_dir"/* "$subfolder"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] All files copied to: $subfolder" | tee -a "$LOG_FILE"
}

# 6. Iterate over each task and perform repetitions
for TASK in "${TASKS[@]}"; do
    echo "-------------------------------------------"
    echo "Starting task: ${TASK}"
    echo "-------------------------------------------"

    # Define collection path for the current task
    COLLECTED_DIR="${BASE_LOG_DIR}/collected_metrics_${TASK}"
    mkdir -p "$COLLECTED_DIR"

    # Define log file for the current task
    LOG_FILE="${BASE_LOG_DIR}/eval/${TASK}_repeats.log"
    touch "$LOG_FILE"

    # Iterate and run the task multiple times
    for i in $(seq 1 $REPEATS); do
        run_task "$TASK" "$i"
        if [ $? -ne 0 ]; then
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] Skipping to next iteration due to error." | tee -a "$LOG_FILE"
        fi
    done

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Completed ${REPEATS} repetitions for task: ${TASK}." | tee -a "$LOG_FILE"
done

echo "[$(date +"%Y-%m-%d %H:%M:%S")] All tasks completed."