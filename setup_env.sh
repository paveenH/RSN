#!/bin/bash
# ==================== RolePlaying Environment Setup ====================
# This script creates and configures the conda environment for RolePlaying
#
# Usage: bash setup_env.sh
#
# Note: Run this script in an interactive shell (not via sbatch)

ENV_NAME="roleplaying"
PYTHON_VERSION="3.10"

echo "=================================================="
echo "Setting up RolePlaying environment"
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "=================================================="

# Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "[2/4] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install PyTorch with CUDA support
echo "[3/4] Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install required packages
echo "[4/4] Installing Python packages..."
pip install --upgrade pip

# Core packages
pip install transformers accelerate datasets

# For Mistral3 models (requires latest transformers from main)
pip install git+https://github.com/huggingface/transformers
pip install "mistral-common>=1.8.6"

# Data processing & analysis
pip install numpy pandas scikit-learn scipy

# Visualization
pip install matplotlib seaborn

# Hugging Face utilities
pip install huggingface_hub

# Progress bars
pip install tqdm

# Optional: for notebook support
# pip install jupyter ipykernel

echo "=================================================="
echo "Environment setup complete!"
echo ""
echo "To activate: conda activate ${ENV_NAME}"
echo "To verify: python -c 'import torch; print(torch.cuda.is_available())'"
echo "=================================================="
