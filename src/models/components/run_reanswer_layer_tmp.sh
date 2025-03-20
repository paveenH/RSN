
#!/bin/bash
# Created on Tue Dec 24 10:18:42 2024 (Updated on $(date))
# Author: paveenhuang

TASKS=( ... )  # 任务列表
SIZES=("8B")
MODELS=("llama3")
TOPS=("20")
ALPHAS=("1")
START_END_PAIRS=("11 32" "21 32" "21 32" "11 21" "1 11")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 顺序执行所有组合
echo "Starting sequential execution..."
for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SIZE in "${SIZES[@]}"; do
            for TOP in "${TOPS[@]}"; do
                for ALPHA in "${ALPHAS[@]}"; do
                    for PAIR in "${START_END_PAIRS[@]}"; do
                        echo "Processing: $TASK $MODEL $SIZE $TOP $ALPHA $PAIR"
                        python3 /data2/paveen/RolePlaying/src/models/components/get_answer_regenerate_layer.py \
                            "$TASK" "$MODEL" "$SIZE" "$TOP" "$ALPHA" $PAIR
                        if [ $? -ne 0 ]; then
                            echo "Error in processing $TASK $MODEL $SIZE $TOP $ALPHA $PAIR"
                            exit 1
                        fi
                    done
                done
            done
        done
    done
done

echo "All tasks have been processed successfully."