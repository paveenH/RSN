#!/bin/bash
# ==================== NCHC Nano5 Environment Setup Script ====================
# This script should be executed on the login node (not submitted via sbatch) to initialize the environment
# Usage: bash setup_env.sh

set -e

# ===== Modify here =====
USERNAME="d12922004"
# =======================

WORK_DIR="/work/${USERNAME}/RolePlaying"
CODE_DIR="${WORK_DIR}/code"
DATA_DIR="${WORK_DIR}/components"
CONDA_ENV="roleplaying"

echo "=================================================="
echo "Setting up RolePlaying environment on NCHC Nano5"
echo "Work directory: ${WORK_DIR}"
echo "=================================================="

# 1. 创建目录结构
echo "[1/4] Creating directory structure..."
mkdir -p "${CODE_DIR}"
mkdir -p "${DATA_DIR}/mmlu"
mkdir -p "${DATA_DIR}/hidden_states_non"
mkdir -p "${DATA_DIR}/mmlu_fewshot"

echo "  Created: ${CODE_DIR}"
echo "  Created: ${DATA_DIR}/mmlu"
echo "  Created: ${DATA_DIR}/hidden_states_non"
echo "  Created: ${DATA_DIR}/mmlu_fewshot"

# 2. 加载 miniconda 并创建 conda 环境
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

# 3. 安装依赖
echo ""
echo "[3/4] Installing dependencies..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers accelerate bitsandbytes tqdm numpy lm-eval

# 如果遇到 CUDA NOT FOUND，取消下面这行的注释
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
echo "  1. Upload code:  scp -r *.py detection/ ${USERNAME}@nano5.nchc.org.tw:${CODE_DIR}/"
echo "  2. Upload data:  scp -r <local_mmlu_dir>/* ${USERNAME}@nano5.nchc.org.tw:${DATA_DIR}/mmlu/"
echo "  3. Upload script: scp run_logits.sh ${USERNAME}@nano5.nchc.org.tw:${CODE_DIR}/"
echo "  4. Edit run_logits.sh: fill in USERNAME and ACCOUNT_ID"
echo "  5. Submit job:   sbatch run_logits.sh"
echo "=================================================="
