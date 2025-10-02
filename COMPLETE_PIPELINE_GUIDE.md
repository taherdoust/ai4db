# Complete 3-Stage Pipeline Execution Guide

## 🎯 Overview

This guide walks you through executing the complete Text-to-Spatial-SQL dataset generation pipeline:

**Stage 1**: Rule-Based Template Generation (Foundation)  
**Stage 2**: SDV Synthetic SQL Generation (Multiplication)  
**Stage 3**: NL Question Augmentation (Diversification)

**Final Dataset**: 500K-1M high-quality (SQL, NL) pairs for LLM fine-tuning

---

## 📦 Prerequisites

### 1. Environment Setup

```bash
# Create Conda environment
conda create -n ai4db-advanced python=3.10 -y
conda activate ai4db-advanced

# Navigate to project directory
cd /home/eclab/Desktop/ai4db
```

### 2. Install Dependencies

```bash
# Core dependencies (all stages)
pip install pandas==2.1.0 numpy==1.24.0 sqlparse==0.4.4

# Stage 2: SDV
pip install sdv==1.9.0

# Stage 3: NLP augmentation
pip install sentence-transformers==2.2.2
pip install transformers==4.36.0
pip install torch==2.1.0  # or torch==2.1.0+cu118 for CUDA 11.8
pip install sacremoses==0.0.53
pip install language-tool-python==2.7.1
pip install nltk==3.8.1

# Download NLTK data
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet omw-1.4
```

### 3. Verify Installation

```bash
python -c "import sdv; import sentence_transformers; import transformers; print('✓ All dependencies installed')"
```

---

## 🚀 Stage 1: Rule-Based Template Generation

### Objective
Generate 10,000+ schema-aware SQL pairs using templates and realistic parameters.

### Quick Test (10 variations)

```bash
python stage1_enhanced_generator.py
```

**Expected output**: 
- `training_datasets/stage1_enhanced_dataset.jsonl` (~2000 samples)
- `training_datasets/stage1_enhanced_dataset_eval.jsonl` (100 evaluation samples)
- `training_datasets/stage1_enhanced_dataset_stats.json`

### Full Production Run

To generate more samples, modify the script:

```python
# In stage1_enhanced_generator.py, line ~286
samples, stats = generate_stage1_enhanced_dataset(
    num_variations=500,  # Change from 200 to 500 for ~10K samples
    output_file="training_datasets/stage1_enhanced_dataset.jsonl",
    evaluation_sample_size=100
)
```

Then run:
```bash
python stage1_enhanced_generator.py
```

**Execution Time**: 10-20 minutes  
**Output Size**: ~10,000 samples

### Verify Output

```bash
# Count samples
wc -l training_datasets/stage1_enhanced_dataset.jsonl

# View sample
head -n 1 training_datasets/stage1_enhanced_dataset.jsonl | python -m json.tool

# Check statistics
cat training_datasets/stage1_enhanced_dataset_stats.json
```

---

## 🤖 Stage 2: SDV Synthetic SQL Generation

### Objective
Use CTGAN to generate 50,000 synthetic SQL structures, achieving 5x dataset expansion.

### Configuration Options

**Option A: Fast Mode (GaussianCopula, 5-10 minutes)**
```bash
python stage2_sdv_pipeline.py gaussian 5000
```

**Option B: High Quality Mode (CTGAN, 2-4 hours, recommended)**
```bash
# CPU only
python stage2_sdv_pipeline.py ctgan 50000

# With GPU (if available)
# Edit stage2_sdv_pipeline.py, line ~688, set use_gpu=True
python stage2_sdv_pipeline.py ctgan 50000
```

### Expected Output

- `training_datasets/stage2_synthetic_dataset.jsonl` (~50,000 samples)
- `training_datasets/stage2_synthetic_dataset_stats.json`
- `training_datasets/stage2_synthetic_dataset_model.pkl` (saved CTGAN model)

### Monitor Progress

```bash
# Stage 2 will print progress:
# [1/6] Loading Stage 1 data...
# [2/6] Extracting features for CTGAN training...
# [3/6] Training synthesizer... (this takes 2-4 hours)
# [4/6] Generating synthetic structures...
# [5/6] Assembling SQL queries...
# [6/6] Filtering by quality...
```

### Verify Quality

```bash
# Check statistics
cat training_datasets/stage2_synthetic_dataset_stats.json

# Sample quality scores
head -n 10 training_datasets/stage2_synthetic_dataset.jsonl | \
  python -c "import sys, json; [print(f\"Quality: {json.loads(line)['quality_score']:.3f}\") for line in sys.stdin]"
```

**Expected Quality Score**: 0.70-0.85 average

---

## 🌈 Stage 3: NL Question Augmentation

### Objective
Generate 10x diverse natural language questions for each SQL query.

### Configuration Options

**Option A: Template + Compositional Only (Fast, no GPU needed, ~1 hour)**
```bash
python stage3_augmentation_pipeline.py --no-paraphrase --no-backtrans --multiplier 5
```

**Option B: Template + Paraphrase + Back-Translation (Medium quality, CPU, ~20 hours)**
```bash
python stage3_augmentation_pipeline.py --multiplier 10
```

**Option C: Full Pipeline with LLM (Highest quality, GPU required, ~40 hours)**
```bash
# Requires GPU with 16GB+ VRAM
python stage3_augmentation_pipeline.py --llm --multiplier 10
```

### Expected Output

- `training_datasets/stage3_augmented_dataset.jsonl` (~500K samples for 10x multiplier)
- `training_datasets/stage3_augmented_dataset_stats.json`

### Monitor Progress

```bash
# Stage 3 will print:
# [1/5] Initializing augmentation strategies...
# [2/5] Loading Stage 2 data...
# [3/5] Generating augmented questions...
#   Progress: 1000/50000 samples processed...
# [4/5] Saving augmented dataset...
# [5/5] Generating statistics...
```

### Verify Diversity

```bash
# Check average multiplier
cat training_datasets/stage3_augmented_dataset_stats.json | grep average_multiplier

# Sample questions
head -n 5 training_datasets/stage3_augmented_dataset.jsonl | \
  python -c "import sys, json; [print(f\"{json.loads(line)['id']}: {json.loads(line)['question'][:80]}...\") for line in sys.stdin]"
```

---

## 📊 Complete Pipeline Summary

### Execution Timeline

| Stage | Method | Time (CPU) | Time (GPU) | Output Size |
|-------|--------|-----------|-----------|-------------|
| Stage 1 | Rule-based | 15 min | 15 min | ~10K samples |
| Stage 2 | CTGAN | 3-4 hours | 1-2 hours | ~50K samples |
| Stage 3 | Multi-augment | 20-30 hours | 5-8 hours | ~500K samples |
| **Total** | | **~25-35 hours** | **~7-10 hours** | **~500K samples** |

### Resource Requirements

**CPU Mode:**
- RAM: 16GB minimum, 32GB recommended
- Disk: 5GB for models + 2GB for datasets
- Time: 25-35 hours total

**GPU Mode (Recommended):**
- GPU: 16GB+ VRAM (e.g., RTX 4090, A100)
- RAM: 32GB
- Disk: 10GB for models + 2GB for datasets
- Time: 7-10 hours total

---

## 🔍 Quality Validation

### After Each Stage

**Stage 1 Validation:**
```bash
python -c "
import json
with open('training_datasets/stage1_enhanced_dataset.jsonl') as f:
    samples = [json.loads(line) for line in f]
    print(f'Total samples: {len(samples)}')
    print(f'Unique SQL types: {len(set(s[\"sql_type\"] for s in samples))}')
    print(f'Avg spatial functions: {sum(len(s[\"spatial_functions\"]) for s in samples) / len(samples):.2f}')
"
```

**Stage 2 Validation:**
```bash
python -c "
import json
with open('training_datasets/stage2_synthetic_dataset.jsonl') as f:
    samples = [json.loads(line) for line in f]
    quality_scores = [s['quality_score'] for s in samples]
    print(f'Total samples: {len(samples)}')
    print(f'Average quality: {sum(quality_scores) / len(quality_scores):.3f}')
    print(f'High quality (>0.8): {sum(1 for q in quality_scores if q > 0.8)} ({sum(1 for q in quality_scores if q > 0.8)/len(quality_scores)*100:.1f}%)')
"
```

**Stage 3 Validation:**
```bash
python -c "
import json
from collections import Counter
with open('training_datasets/stage3_augmented_dataset.jsonl') as f:
    samples = [json.loads(line) for line in f]
    tones = Counter(s['question_tone'] for s in samples)
    print(f'Total samples: {len(samples)}')
    print(f'Question tone distribution:')
    for tone, count in tones.most_common(5):
        print(f'  {tone}: {count} ({count/len(samples)*100:.1f}%)')
"
```

---

## 🎓 Training Preparation

### 1. Split Train/Eval

```python
import json
import random

# Load final dataset
with open('training_datasets/stage3_augmented_dataset.jsonl') as f:
    samples = [json.loads(line) for line in f]

# Shuffle
random.seed(42)
random.shuffle(samples)

# Split 90/10
split_idx = int(len(samples) * 0.9)
train = samples[:split_idx]
eval_set = samples[split_idx:]

# Save
with open('training_datasets/train.jsonl', 'w') as f:
    for s in train:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

with open('training_datasets/eval.jsonl', 'w') as f:
    for s in eval_set:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f"Train: {len(train):,} samples")
print(f"Eval: {len(eval_set):,} samples")
```

### 2. Convert to Training Format

For LLM fine-tuning (e.g., using Hugging Face Transformers):

```python
import json

def convert_to_training_format(input_file, output_file):
    """Convert to instruction-following format"""
    
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            sample = json.loads(line)
            
            # Create instruction-following format
            training_sample = {
                "instruction": f"Convert this natural language question to PostGIS spatial SQL for the CIM Wizard database.",
                "input": sample['question'],
                "output": sample['sql_postgis'],
                "metadata": {
                    "database_id": sample['database_id'],
                    "sql_type": sample['sql_type'],
                    "difficulty": sample['difficulty']['overall_difficulty'],
                    "spatial_functions": sample['spatial_functions']
                }
            }
            
            f_out.write(json.dumps(training_sample, ensure_ascii=False) + '\n')

# Convert train and eval
convert_to_training_format('training_datasets/train.jsonl', 'training_datasets/train_formatted.jsonl')
convert_to_training_format('training_datasets/eval.jsonl', 'training_datasets/eval_formatted.jsonl')
```

---

## 🐛 Troubleshooting

### Issue: Stage 2 CTGAN training is very slow

**Solution**: Use GaussianCopula for testing, or reduce epochs:
```python
# In stage2_sdv_pipeline.py
epochs=100  # Instead of 300
```

### Issue: Stage 3 LLM runs out of memory

**Solution**: Disable LLM augmentation or use smaller model:
```bash
python stage3_augmentation_pipeline.py --no-llm --multiplier 8
```

Or use a smaller model:
```python
# In stage3_augmentation_pipeline.py
llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Issue: Back-translation downloads fail

**Solution**: Pre-download models:
```python
from transformers import MarianMTModel, MarianTokenizer

# Pre-download
for pair in ['en-fr', 'fr-en', 'en-de', 'de-en']:
    model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{pair}')
    tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{pair}')
```

### Issue: Grammar check is slow

**Solution**: Disable grammar checking in Stage 3:
```python
# Comment out grammar check in stage3_augmentation_pipeline.py
# GRAMMAR_CHECK_AVAILABLE = False
```

---

## 📈 Next Steps After Dataset Creation

1. **Validate Sample Quality**: Manually review 100 random samples
2. **Execute Evaluation Queries**: Run SQL queries on CIM Wizard database to fill `results` field
3. **Calculate Baseline Metrics**: Evaluate existing LLMs (GPT-4, Claude) on eval set
4. **Fine-tune LLM**: Use dataset to fine-tune Code-Llama-7B or StarCoder
5. **Evaluate Fine-tuned Model**: Measure Execution Accuracy (EX) on test set
6. **Iterate**: Refine dataset based on model performance

---

## 📚 Academic References

This pipeline is based on:

- **BIRD**: A comprehensive Text-to-SQL benchmark with execution-based evaluation
- **Spider**: Multi-domain semantic parsing and Text-to-SQL dataset
- **OmniSQL**: Question tone classification for diverse NL generation
- **SpatialSQL**: Empirical analysis of spatial SQL function usage
- **SDV**: Synthetic Data Vault for generating realistic tabular data
- **CTGAN**: Conditional Tabular GAN for high-fidelity synthetic data generation

---

## 🎉 Success Checklist

- [ ] Stage 1 complete: ~10K schema-aware SQL pairs
- [ ] Stage 2 complete: ~50K synthetic SQL with quality >0.70
- [ ] Stage 3 complete: ~500K diverse (SQL, NL) pairs
- [ ] Train/eval split created (90/10)
- [ ] Training format conversion complete
- [ ] Sample validation passed (manual review)
- [ ] Ready for LLM fine-tuning

**Congratulations!** You now have a comprehensive Text-to-Spatial-SQL dataset for fine-tuning LLMs.

---

## 💡 Tips for Best Results

1. **Start Small**: Run each stage with small samples first to verify everything works
2. **Use GPU**: Stage 2 (CTGAN) and Stage 3 (LLM) benefit greatly from GPU acceleration
3. **Monitor Quality**: Check quality metrics after each stage
4. **Parallel Execution**: Stage 3 can be parallelized by splitting Stage 2 output
5. **Save Checkpoints**: Stage 2 and 3 save models/checkpoints for resuming
6. **Disk Space**: Ensure 10GB+ free space for models and datasets

---

For questions or issues, refer to:
- `README.md`: Project overview
- `STAGE2_SDV_PLAN.md`: Stage 2 technical details
- `STAGE3_AUGMENTATION_PLAN.md`: Stage 3 augmentation strategies
- `txt2ssql_paper/`: Academic methodology and justifications

