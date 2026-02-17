#!/bin/bash

#SBATCH -A MST114558                                # Account ID
#SBATCH --job-name=gsm8k_debug                      # Job name
#SBATCH --output=./execution/output_%j.log          # Standard output log
#SBATCH --error=./execution/error_%j.log            # Error output log
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks-per-node=1                         # Tasks per node
#SBATCH --gres=gpu:1                                # Number of GPUs
#SBATCH --cpus-per-task=4                           # Number of CPUs
#SBATCH --time=1:00:00                              # Maximum runtime
#SBATCH --partition=normal                          # Partition
#SBATCH --mail-type=ALL                             # Email notification
#SBATCH --mail-user=paveenhuang@gmail.com           # Email address

# ==================== GSM8K Debug (N5) ====================
# Runs first 2 samples only, both llama3-8B and qwen3-8B
# Both baseline (generate) and regenerate
# Purpose: Check output format and debug errors
#
# Usage: sbatch run_gsm8k_debug.sh

# ==================== Configuration ====================
USERNAME="d12922004"

# Role and template configuration
ROLES="neutral"
SUITE="default"

# Mask configuration
MASK_TYPE="nmd"
PERCENTAGE=0.5

# Generation parameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.0

# GSM8K data file
GSM8K_FILE="benchmark/gsm8k_test.json"

# ==================== Model configurations ====================
# Format: MODEL_NAME|MODEL_DIR|MODEL_SIZE|HS_PREFIX|CONFIGS
MODELS=(
    "llama3|/work/${USERNAME}/models/Llama3-8B|8B|llama3|4-11-20 neg4-11-20"
    "qwen3|/work/${USERNAME}/models/Qwen3-8B|8B|qwen3|4-17-26 neg4-17-26"
)

# ==================== Paths ====================
WORK_DIR="/work/${USERNAME}/RolePlaying"
BASE_DIR="${WORK_DIR}/components"
CONDA_ENV="roleplaying"

# ==================== Environment ====================
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="
echo "Models: llama3-8B, qwen3-8B"
echo "Task: GSM8K DEBUG (2 samples only)"
echo "Roles: ${ROLES}"
echo "=================================================="

# Load miniconda
ml load miniconda3

# Activate conda environment
conda activate ${CONDA_ENV}

# Check environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi

cd ${WORK_DIR}

# ==================== Create debug data (2 samples) ====================
echo ""
echo "=========================================="
echo "Creating debug data (first 2 samples)"
echo "=========================================="

python -c "
import json
from pathlib import Path

data_path = Path('${BASE_DIR}') / '${GSM8K_FILE}'
with open(data_path, 'r') as f:
    all_samples = json.load(f)

debug_samples = all_samples[:2]

debug_file = Path('${BASE_DIR}') / 'benchmark/gsm8k_debug.json'
with open(debug_file, 'w') as f:
    json.dump(debug_samples, f, indent=2, ensure_ascii=False)

print(f'Created debug file with {len(debug_samples)} samples')
for i, s in enumerate(debug_samples):
    print(f'  [{i}] Q: {s[\"question\"][:80]}...')
    print(f'      A: {s[\"answer\"]}')
"

# ==================== Run each model ====================
STEP=1
TOTAL=$(( ${#MODELS[@]} * 2 ))

for MODEL_CFG in "${MODELS[@]}"; do
    IFS="|" read -r MODEL_NAME MODEL_DIR MODEL_SIZE HS_PREFIX CONFIGS <<< "${MODEL_CFG}"
    TYPE="non"

    echo ""
    echo "####################################################"
    echo "# Model: ${MODEL_NAME} (${MODEL_SIZE})"
    echo "# Model dir: ${MODEL_DIR}"
    echo "# Configs: ${CONFIGS}"
    echo "####################################################"

    # ==================== Baseline ====================
    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] GSM8K baseline (generate) - ${MODEL_NAME}"
    echo "=========================================="

    python get_answer_gsm8k.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --size "${MODEL_SIZE}" \
        --test_file "benchmark/gsm8k_debug.json" \
        --ans_file "answer_gsm8k_debug" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE}

    echo "[Done] GSM8K baseline - ${MODEL_NAME}"
    STEP=$((STEP + 1))

    # ==================== Regenerate ====================
    echo ""
    echo "=========================================="
    echo "[${STEP}/${TOTAL}] GSM8K regenerate - ${MODEL_NAME}"
    echo "=========================================="

    python get_answer_regenerate_gsm8k.py \
        --model "${MODEL_NAME}" \
        --model_dir "${MODEL_DIR}" \
        --hs "${HS_PREFIX}" \
        --size "${MODEL_SIZE}" \
        --type "${TYPE}" \
        --percentage "${PERCENTAGE}" \
        --configs ${CONFIGS} \
        --mask_type "${MASK_TYPE}" \
        --test_file "benchmark/gsm8k_debug.json" \
        --ans_file "answer_mdf_gsm8k_debug" \
        --suite "${SUITE}" \
        --base_dir "${BASE_DIR}" \
        --roles "${ROLES}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --temperature ${TEMPERATURE}

    echo "[Done] GSM8K regenerate - ${MODEL_NAME}"
    STEP=$((STEP + 1))

    # ==================== Print results for this model ====================
    echo ""
    echo "=========================================="
    echo "Results for ${MODEL_NAME}:"
    echo "=========================================="

    python -c "
import json, os

base_dir = '${BASE_DIR}'
model = '${MODEL_NAME}'
size = '${MODEL_SIZE}'

# Baseline
bp = os.path.join(base_dir, model, 'answer_gsm8k_debug', 'orig', f'gsm8k_{size}_answers.json')
if os.path.exists(bp):
    with open(bp) as f:
        bl = json.load(f)
    print('=== BASELINE (generate) ===')
    if 'accuracy' in bl:
        for r, a in bl['accuracy'].items():
            print(f'  {r}: {a}')
    for i, s in enumerate(bl['data']):
        print(f'\\n--- Sample {i} ---')
        print(f'Q: {s[\"question\"]}')
        print(f'Gold: {s[\"answer\"]}')
        print(f'Generated: {s.get(\"generated_neutral\", \"N/A\")[:300]}')
        print(f'Pred: {s.get(\"pred_answer_neutral\", \"N/A\")}')
        print(f'Correct: {s.get(\"correct_neutral\", \"N/A\")}')
else:
    print(f'Baseline not found: {bp}')

# Regenerate
rr = os.path.join(base_dir, model, 'answer_mdf_gsm8k_debug')
if os.path.exists(rr):
    for md in sorted(os.listdir(rr)):
        mp = os.path.join(rr, md)
        if not os.path.isdir(mp):
            continue
        for fn in sorted(os.listdir(mp)):
            if fn.endswith('.json'):
                with open(os.path.join(mp, fn)) as f:
                    rg = json.load(f)
                print(f'\\n=== REGENERATE ({md}/{fn}) ===')
                if 'accuracy' in rg:
                    for r, a in rg['accuracy'].items():
                        print(f'  {r}: {a}')
                for i, s in enumerate(rg['data']):
                    print(f'\\n--- Sample {i} ---')
                    print(f'Q: {s[\"question\"]}')
                    print(f'Gold: {s[\"answer\"]}')
                    print(f'Generated: {s.get(\"generated_neutral\", \"N/A\")[:300]}')
                    print(f'Pred: {s.get(\"pred_answer_neutral\", \"N/A\")}')
                    print(f'Correct: {s.get(\"correct_neutral\", \"N/A\")}')
else:
    print(f'Regenerate not found: {rr}')
"
done

echo ""
echo "=================================================="
echo "GSM8K debug finished at: $(date)"
echo "=================================================="
