#!/usr/bin/env bash
# Shell script to automatically set up python, install PyTorch, and run the VRAM test.
set -e

echo "=========================================================="
# Export CUDA and NVIDIA library paths to prevent CUDA driver loading failures
CLEANED_LD_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "cuda/compat" | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:$CLEANED_LD_PATH"

echo ">>> [1/3] Setting up Python virtual environment with uv..."
if ! command -v uv &> /dev/null; then
    echo "    - uv not found, installing via pip..."
    pip install uv --quiet
fi

uv venv .venv
source .venv/bin/activate

echo ">>> [2/3] Installing PyTorch with CUDA 12.1 support..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

echo ">>> [3/3] Executing VRAM isolation experiment..."
python scripts/vram_test.py
