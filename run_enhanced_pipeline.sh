#!/bin/bash
# Quick start script for enhanced pipeline with secure API key management

echo "========================================================================"
echo "AI4DB Enhanced Pipeline - Complete Run"
echo "========================================================================"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo ""
    echo "Setup steps:"
    echo "1. Copy the template: cp .env.example .env"
    echo "2. Edit .env and add your OPENROUTER_API_KEY"
    echo "3. Run this script again"
    echo ""
    exit 1
fi

# Check if API key is set
source .env
if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "sk-or-v1-your-key-here" ]; then
    echo "⚠️  OPENROUTER_API_KEY not set in .env file!"
    echo ""
    echo "Edit .env and replace 'sk-or-v1-your-key-here' with your actual API key"
    echo ""
    exit 1
fi

echo "✓ API key loaded from .env"
echo "✓ Model: ${OPENROUTER_MODEL:-openai/gpt-4-turbo-preview}"
echo ""

# Parse arguments
MULTIPLIER=8
while [[ $# -gt 0 ]]; do
    case $1 in
        --multiplier)
            MULTIPLIER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--multiplier N]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  - Target multiplier: ${MULTIPLIER}x"
echo "  - Estimated time: ~3-5 hours total"
echo "  - Estimated cost: \$10-30"
echo ""

# Confirm
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "========================================================================"
echo "Stage 1: Enhanced Generator (7-13 min)"
echo "========================================================================"
python stage1_enhanced_generator_stratified.py 200 100

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Stage 2: CTGAN Synthetic SQL Generation (~20 min)"
echo "========================================================================"
python stage2_sdv_pipeline_eclab_ctgan.py 50000 300

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Stage 3: Enhanced NL Question + Instruction Generation (1-2 hours)"
echo "========================================================================"
python stage3_augmentation_pipeline_eclab_openrouter_enhanced.py --multiplier ${MULTIPLIER}

if [ $? -ne 0 ]; then
    echo "❌ Stage 3 failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ Complete Pipeline Finished Successfully!"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  - Stage 1: training_datasets/stage1_enhanced_dataset.jsonl"
echo "  - Stage 2: training_datasets/stage2_synthetic_dataset_eclab_ctgan.jsonl"
echo "  - Stage 3: training_datasets/stage3_augmented_dataset_eclab_openrouter_enhanced.jsonl"
echo ""
echo "Statistics:"
ls -lh training_datasets/stage1_enhanced_dataset_stats.json 2>/dev/null
ls -lh training_datasets/stage2_synthetic_dataset_eclab_ctgan_stats.json 2>/dev/null
ls -lh training_datasets/stage3_augmented_dataset_eclab_openrouter_enhanced_stats.json 2>/dev/null
echo ""
echo "🎉 Ready to fine-tune your model!"
echo ""

