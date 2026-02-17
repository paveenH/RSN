#!/bin/bash
# ==================== TriviaQA Debug Script (Local) ====================
# Runs first 2 TriviaQA samples with both generate and regenerate
# Model: llama3-8B
# For local development/debugging - compare baseline vs neuron editing
#
# Usage: bash run_triviaqa_debug_local.sh

# ==================== Configuration ====================
MODEL="llama3"
SIZE="8B"
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
DATA="data1"
HS_PREFIX="llama3"
TYPE="non"

# Role and template configuration
ROLES="neutral"
SUITE="default"

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5
CONFIGS="4-11-20 neg4-11-20"

# Generation parameters
MAX_NEW_TOKENS=64
TEMPERATURE=0.0

# TriviaQA data file
TRIVIAQA_FILE="benchmark/triviaqa_validation.json"

# ==================== Paths ====================
WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

# ==================== Environment ====================
echo "=================================================="
echo "TriviaQA Debug Script (Local)"
echo "Start time: $(date)"
echo "=================================================="
echo "Model: ${MODEL} (${SIZE})"
echo "Model path: ${MODEL_PATH}"
echo "Roles: ${ROLES}"
echo "Configs: ${CONFIGS}"
echo "Samples: First 2 only (debug mode)"
echo "=================================================="

cd ${WORK_DIR}

# ==================== Create debug data ====================
python -c "
import json
from pathlib import Path

data_path = Path('${BASE_DIR}') / '${TRIVIAQA_FILE}'
with open(data_path, 'r') as f:
    all_samples = json.load(f)

debug_samples = all_samples[:2]

debug_file = Path('${BASE_DIR}') / 'benchmark/triviaqa_debug.json'
with open(debug_file, 'w') as f:
    json.dump(debug_samples, f, indent=2, ensure_ascii=False)

print(f'Created debug file with {len(debug_samples)} samples')
for i, s in enumerate(debug_samples):
    print(f'  [{i}] Q: {s[\"question\"][:80]}...')
    print(f'      A: {s[\"answer\"]}')
    print(f'      Aliases: {s.get(\"aliases\", [])[:3]}...')
"

# ==================== 1. Baseline (generate) ====================
echo ""
echo "=========================================="
echo "[1/2] TriviaQA Baseline (generate) - 2 samples"
echo "=========================================="

python get_answer_triviaqa.py \
    --model "${MODEL}" \
    --model_dir "${MODEL_PATH}" \
    --size "${SIZE}" \
    --test_file "benchmark/triviaqa_debug.json" \
    --ans_file "answer_triviaqa_debug" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE}

echo "[Done] TriviaQA baseline"

# ==================== 2. Regenerate (neuron editing) ====================
echo ""
echo "=========================================="
echo "[2/2] TriviaQA Regenerate (neuron editing) - 2 samples"
echo "=========================================="

python get_answer_regenerate_triviaqa.py \
    --model "${MODEL}" \
    --model_dir "${MODEL_PATH}" \
    --hs "${HS_PREFIX}" \
    --size "${SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --configs ${CONFIGS} \
    --mask_type "${MASK_TYPE}" \
    --test_file "benchmark/triviaqa_debug.json" \
    --ans_file "answer_mdf_triviaqa_debug" \
    --suite "${SUITE}" \
    --base_dir "${BASE_DIR}" \
    --roles "${ROLES}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE}

echo "[Done] TriviaQA regenerate"

# ==================== Compare results ====================
echo ""
echo "=========================================="
echo "Comparing baseline vs regenerate results"
echo "=========================================="

python -c "
import json, os

base_dir = '${BASE_DIR}'
model = '${MODEL}'

# Load baseline results
baseline_path = os.path.join(base_dir, model, 'answer_triviaqa_debug', 'orig', 'triviaqa_${SIZE}_answers.json')
if os.path.exists(baseline_path):
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    print('=== BASELINE (generate) ===')
    for i, s in enumerate(baseline['data']):
        print(f'\\n--- Sample {i} ---')
        print(f'Q: {s[\"question\"]}')
        print(f'Gold: {s[\"answer\"]}')
        print(f'Generated: {s.get(\"generated_neutral\", \"N/A\")[:200]}')
        print(f'Pred: {s.get(\"pred_answer_neutral\", \"N/A\")}')
        print(f'Correct: {s.get(\"correct_neutral\", \"N/A\")}')
else:
    print(f'Baseline file not found: {baseline_path}')

# Load regenerate results (find mdf_* dirs)
regen_root = os.path.join(base_dir, model, 'answer_mdf_triviaqa_debug')
if os.path.exists(regen_root):
    for mdf_dir in sorted(os.listdir(regen_root)):
        mdf_path = os.path.join(regen_root, mdf_dir)
        if not os.path.isdir(mdf_path):
            continue
        for fname in sorted(os.listdir(mdf_path)):
            if fname.endswith('.json'):
                fpath = os.path.join(mdf_path, fname)
                with open(fpath, 'r') as f:
                    regen = json.load(f)
                print(f'\\n=== REGENERATE ({mdf_dir} / {fname}) ===')
                if 'accuracy' in regen:
                    for role, acc in regen['accuracy'].items():
                        print(f'  {role}: {acc}')
                for i, s in enumerate(regen['data']):
                    print(f'\\n--- Sample {i} ---')
                    print(f'Q: {s[\"question\"]}')
                    print(f'Gold: {s[\"answer\"]}')
                    print(f'Generated: {s.get(\"generated_neutral\", \"N/A\")[:200]}')
                    print(f'Pred: {s.get(\"pred_answer_neutral\", \"N/A\")}')
                    print(f'Correct: {s.get(\"correct_neutral\", \"N/A\")}')
else:
    print(f'Regenerate dir not found: {regen_root}')
"

echo ""
echo "=================================================="
echo "Debug run finished at: $(date)"
echo "=================================================="
echo ""
echo "Output files:"
echo "  Baseline: ${BASE_DIR}/${MODEL}/answer_triviaqa_debug/"
echo "  Regenerate: ${BASE_DIR}/${MODEL}/answer_mdf_triviaqa_debug/"
