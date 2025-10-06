#!/bin/bash
# AI4DB Environment Setup Script
# Quick setup for eclab machine (CPU-only)

set -e  # Exit on error

echo "=========================================="
echo "AI4DB Environment Setup"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^ai4db "; then
    echo "⚠️  Environment 'ai4db' already exists."
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating environment..."
        conda env update -f environment.yml --prune
        echo "✓ Environment updated!"
    else
        echo "Skipping environment creation."
    fi
else
    echo "Creating new environment 'ai4db'..."
    conda env create -f environment.yml
    echo "✓ Environment created!"
fi

echo ""
echo "=========================================="
echo "Testing Installation"
echo "=========================================="
echo ""

# Activate and test
eval "$(conda shell.bash hook)"
conda activate ai4db

echo "Python version:"
python --version

echo ""
echo "Testing core packages..."

python << EOF
try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except ImportError as e:
    print("❌ NumPy:", e)

try:
    import pandas as pd
    print("✓ Pandas:", pd.__version__)
except ImportError as e:
    print("❌ Pandas:", e)

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  - CUDA available:", torch.cuda.is_available())
except ImportError as e:
    print("❌ PyTorch:", e)

try:
    import sdv
    print("✓ SDV:", sdv.__version__)
except ImportError as e:
    print("❌ SDV:", e)

try:
    import transformers
    print("✓ Transformers:", transformers.__version__)
except ImportError as e:
    print("❌ Transformers:", e)

try:
    import sentence_transformers
    print("✓ Sentence Transformers: installed")
except ImportError as e:
    print("❌ Sentence Transformers:", e)

try:
    import sqlparse
    print("✓ SQLParse:", sqlparse.__version__)
except ImportError as e:
    print("❌ SQLParse:", e)

print("\n✅ Core packages installed successfully!")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ai4db"
echo ""
echo "To run Stage 1:"
echo "  python stage1_enhanced_generator_stratified.py 200 100"
echo ""
echo "To run Stage 2 (GaussianCopula - Fast):"
echo "  python stage2_sdv_pipeline_eclab.py 50000"
echo ""
echo "To run Stage 2 (CTGAN - High Quality):"
echo "  nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2.log 2>&1 &"
echo ""
echo "To run Stage 3 (Ollama):"
echo "  python stage3_augmentation_pipeline_eclab.py --multiplier 5"
echo ""
echo "To run Stage 3 (OpenRouter):"
echo "  export OPENROUTER_API_KEY='your-key'"
echo "  python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8"
echo ""
echo "For more details, see INSTALLATION_GUIDE.md"
echo ""

