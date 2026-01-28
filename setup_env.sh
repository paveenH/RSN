#!/bin/bash
# ==================== NCHC Nano5 Environment Setup Script ====================
# This script should be executed on the login node (not submitted via sbatch) to initialize the environment
# Usage: bash setup_env.sh

set -e

# ===== Modify here =====
USERNAME="d12922004"
# =======================

WORK_DIR="/work/${USERNAME}/RolePlaying"
DATA_DIR="${WORK_DIR}/components/mmlu"
CONDA_ENV="roleplaying"

echo "=================================================="
echo "Setting up RolePlaying environment on NCHC Nano5"
echo "Work directory: ${WORK_DIR}"
echo "=================================================="

# Step 1: Create directory structure
echo "[1/4] Creating directory structure..."
mkdir -p "${DATA_DIR}/mmlu"
mkdir -p "${DATA_DIR}/hidden_states_non"

echo "  Created: ${DATA_DIR}/mmlu"
echo "  Created: ${DATA_DIR}/hidden_states_non"
echo "  Created: ${DATA_DIR}/mmlu_fewshot"

# Step 2: Load miniconda and create conda environment
echo ""
echo "[2/4] Setting up conda environment..."
ml load miniconda3

if conda env list | grep -q "${CONDA_ENV}"; then
    echo "  Environment '${CONDA_ENV}' already exists, skipping creation."
else
    echo "  Creating conda environment: ${CONDA_ENV}"
    conda create -n "${CONDA_ENV}" python=3.12 -y
fi

conda activate "${CONDA_ENV}"

# Step 3: Install dependencies
echo ""
echo "[3/4] Installing dependencies..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers accelerate bitsandbytes tqdm numpy lm-eval

# If encountering CUDA NOT FOUND error, uncomment the line below
# conda install cuda-nvcc -c nvidia -y

echo ""
echo "[4/4] Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "=================================================="
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Upload code:     scp -r *.py detection/ ${USERNAME}@nano5.nchc.org.tw:${CODE_DIR}/"
echo "  2. Upload data:     scp -r <local_mmlu_dir>/* ${USERNAME}@nano5.nchc.org.tw:${DATA_DIR}/mmlu/"
echo "  3. Upload script:   scp run_logits.sh ${USERNAME}@nano5.nchc.org.tw:${CODE_DIR}/"
echo "  4. Edit run_logits.sh: fill in USERNAME and email address"
echo "  5. Submit job:      sbatch run_logits.sh"
echo "=================================================="
