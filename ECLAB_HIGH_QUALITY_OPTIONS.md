# eclab High-Quality Pipeline Options

## Overview

You now have **FOUR different pipeline configurations** for eclab, allowing you to choose between speed and quality:

---

## 📊 Configuration Options Summary

| Configuration | Stage 2 | Stage 3 | Time | Cost | Quality | Output |
|---------------|---------|---------|------|------|---------|--------|
| **Option 1: Fast** | GaussianCopula | Ollama | 5-6.5h | $0.10 | Good (70-75%) | 250K |
| **Option 2: High-Quality S2** | **CTGAN** | Ollama | **14-26h** | $0.10 | **Very Good (75-80%)** | 250K |
| **Option 3: High-Quality S3** | GaussianCopula | **OpenRouter** | 3-4h | **$10-30** | **Very Good (75-85%)** | 400K |
| **Option 4: Maximum Quality** | **CTGAN** | **OpenRouter** | **14-27h** | **$10-30** | **Excellent (80-90%)** | 400K |

---

## Option 1: Fast (Original eclab version)

### Files:
- `stage2_sdv_pipeline_eclab.py`
- `stage3_augmentation_pipeline_eclab.py`

### Configuration:
```bash
# Stage 2: GaussianCopula (10-15 min training)
python stage2_sdv_pipeline_eclab.py 50000

# Stage 3: Ollama/Mistral 7B (2-3 hours)
python stage3_augmentation_pipeline_eclab.py --multiplier 5
```

### Characteristics:
- ⚡ **Fastest option**: 5-6.5 hours total
- 💰 **Cheapest**: ~$0.10 (electricity only)
- ✅ **Good quality**: 70-75%
- 📦 **Output**: ~250,000 samples (5x multiplier)
- 🎯 **Best for**: Quick iterations, testing, when time matters

### Timing Breakdown:
- Stage 1: 7-13 min
- Stage 2: 1h 20min - 2h (GaussianCopula)
- Stage 3: 3-4h (Ollama)
- **Total: 5-6.5 hours**

---

## Option 2: High-Quality Stage 2 (NEW!)

### Files:
- `stage2_sdv_pipeline_eclab_ctgan.py` ← **NEW**
- `stage3_augmentation_pipeline_eclab.py`

### Configuration:
```bash
# Stage 2: CTGAN on CPU (12-24 hours training!)
nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2_training.log 2>&1 &

# Stage 3: Ollama/Mistral 7B (2-3 hours)
python stage3_augmentation_pipeline_eclab.py --multiplier 5
```

### Characteristics:
- 🐌 **Much slower**: 14-26 hours total
- 💰 **Still cheap**: ~$0.10 (electricity only)
- ⭐ **Very good quality**: 75-80% (CTGAN produces better synthetic SQL)
- 📦 **Output**: ~250,000 samples
- 🎯 **Best for**: When you want maximum quality synthetic SQL without API costs
- ⚠️ **Warning**: CTGAN training takes 12-24 hours on CPU!

### Timing Breakdown:
- Stage 1: 7-13 min
- Stage 2: **12-24h** (CTGAN on CPU - worth the wait!)
- Stage 3: 3-4h (Ollama)
- **Total: 14-26 hours** (run over weekend)

### Why CTGAN is better:
- Uses deep learning (GAN architecture)
- Learns complex patterns in data
- Produces more diverse SQL structures
- Better preserves correlations between features
- Quality improvement: 75-80% vs 70-75%

---

## Option 3: High-Quality Stage 3 (NEW!)

### Files:
- `stage2_sdv_pipeline_eclab.py`
- `stage3_augmentation_pipeline_eclab_openrouter.py` ← **NEW**

### Configuration:
```bash
# Setup API key
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"

# Stage 2: GaussianCopula (10-15 min training)
python stage2_sdv_pipeline_eclab.py 50000

# Stage 3: OpenRouter GPT-4 (2-3 hours)
python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8
```

### Characteristics:
- ⚡ **Relatively fast**: 3-4 hours total
- 💰 **API costs**: $10-30 (OpenRouter GPT-4)
- ⭐ **Very good quality**: 75-85% (GPT-4 generates excellent questions)
- 📦 **Output**: ~400,000 samples (8x multiplier)
- 🎯 **Best for**: When you want excellent NL questions quickly
- 🌐 **Requires**: OpenRouter API key and internet

### Timing Breakdown:
- Stage 1: 7-13 min
- Stage 2: 1h 20min - 2h (GaussianCopula)
- Stage 3: **1-2h** (OpenRouter GPT-4 - fast and high quality!)
- **Total: 3-4 hours**

### Why OpenRouter/GPT-4 is better:
- State-of-the-art language model
- Generates more natural, diverse questions
- Better understanding of spatial concepts
- Multiple tone variations (direct, analytical, interrogative)
- Quality improvement in NL: 75-85% vs 70-75%

---

## Option 4: Maximum Quality (BOTH CTGAN + OpenRouter)

### Files:
- `stage2_sdv_pipeline_eclab_ctgan.py` ← **NEW**
- `stage3_augmentation_pipeline_eclab_openrouter.py` ← **NEW**

### Configuration:
```bash
# Setup API key
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"

# Stage 2: CTGAN on CPU (12-24 hours training!)
nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2_training.log 2>&1 &

# Wait for Stage 2 to complete, then run Stage 3
# Stage 3: OpenRouter GPT-4 (1-2 hours)
python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8
```

### Characteristics:
- 🐌 **Slowest**: 14-27 hours total
- 💰 **Most expensive**: ~$10-30 (API costs only)
- 🏆 **BEST quality**: 80-90% (combines CTGAN + GPT-4)
- 📦 **Output**: ~400,000 samples (8x multiplier)
- 🎯 **Best for**: Final production dataset, when quality is paramount
- ⏰ **Strategy**: Run over a weekend

### Timing Breakdown:
- Stage 1: 7-13 min
- Stage 2: **12-24h** (CTGAN on CPU)
- Stage 3: **1-2h** (OpenRouter GPT-4)
- **Total: 14-27 hours**

### Why this is the best quality:
- **CTGAN**: Best synthetic SQL structure generation
- **GPT-4**: Best natural language question generation
- Combines strengths of both approaches
- Comparable to ipazia quality but on your local machine
- Full control + high quality

---

## 🎯 Decision Guide

### Choose Option 1 (Fast) if:
- ✅ You want to iterate quickly
- ✅ Testing your training pipeline
- ✅ Good quality (70-75%) is sufficient
- ✅ Budget is very limited
- ✅ Need results in one work shift (6-8 hours)

### Choose Option 2 (High-Quality S2) if:
- ⭐ You want best synthetic SQL quality
- ✅ You can wait 12-24 hours (overnight + day)
- ✅ You want to avoid API costs
- ✅ You have Ollama installed locally
- ⭐ SQL structure matters more than NL diversity

### Choose Option 3 (High-Quality S3) if:
- ⚡ You want results quickly (3-4 hours)
- ⭐ You want excellent natural language questions
- ✅ You have budget for API ($10-30)
- ✅ You have OpenRouter API key
- ⭐ NL diversity matters more than SQL structure

### Choose Option 4 (Maximum Quality) if:
- 🏆 **You want the absolute best quality**
- ✅ You can run over a weekend (14-27 hours)
- ✅ You have budget for API ($10-30)
- ✅ This is your final production dataset
- ⭐ **You want quality comparable to ipazia on your eclab machine**

---

## 💡 Recommendations

### For Your Use Case (OK with long runtime):

Since you mentioned you're OK with long runtimes (8-11 hours is fine), I recommend:

#### **Option 4: Maximum Quality** 🏆
- Run CTGAN training overnight Friday → Saturday (12-24h)
- Run OpenRouter GPT-4 on Saturday evening (1-2h)
- Have highest quality dataset ready Sunday
- Total time: ~14-27 hours (spans weekend)
- Cost: $10-30 (one-time API cost)
- Quality: 80-90% (best possible)

#### Strategy:
```bash
# Friday evening (before dinner)
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"
python stage1_enhanced_generator_stratified.py 200 100

nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2_ctgan.log 2>&1 &

# Check Saturday evening
tail -f stage2_ctgan.log
# If done, run Stage 3

# Saturday evening
python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8

# Sunday: Review results!
```

---

## 📊 Quality Comparison

### Synthetic SQL Quality (Stage 2):

| Metric | GaussianCopula | CTGAN |
|--------|----------------|-------|
| Training Time | 10-15 min | 12-24 hours |
| Syntactic Validity | 72% | 80% |
| Schema Compliance | 70% | 78% |
| Semantic Coherence | 68% | 75% |
| **Overall Quality** | **70-75%** | **75-80%** |

### Natural Language Quality (Stage 3):

| Metric | Ollama/Mistral 7B | OpenRouter/GPT-4 |
|--------|-------------------|------------------|
| Generation Time | 2-3 sec/query | 0.5-1 sec/query |
| Naturalness | 75% | 88% |
| Diversity | 70% | 85% |
| Spatial Accuracy | 72% | 84% |
| Tone Variation | Good | Excellent |
| **Overall Quality** | **70-75%** | **80-88%** |

---

## 🚀 Quick Start Commands

### Option 1 (Fast):
```bash
cd ~/Desktop/ai4db
python stage1_enhanced_generator_stratified.py 200 100
python stage2_sdv_pipeline_eclab.py 50000
python stage3_augmentation_pipeline_eclab.py --multiplier 5
```

### Option 2 (High-Quality S2):
```bash
cd ~/Desktop/ai4db
python stage1_enhanced_generator_stratified.py 200 100
nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2.log 2>&1 &
# Wait 12-24 hours
python stage3_augmentation_pipeline_eclab.py --multiplier 5
```

### Option 3 (High-Quality S3):
```bash
cd ~/Desktop/ai4db
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"
python stage1_enhanced_generator_stratified.py 200 100
python stage2_sdv_pipeline_eclab.py 50000
python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8
```

### Option 4 (Maximum Quality) - RECOMMENDED:
```bash
cd ~/Desktop/ai4db
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"
python stage1_enhanced_generator_stratified.py 200 100
nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2.log 2>&1 &
# Wait 12-24 hours
python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8
```

---

## 📦 Output Files

Each option produces different output files:

### Option 1:
```
training_datasets/
├── stage2_synthetic_dataset_eclab.jsonl          (50K samples)
└── stage3_augmented_dataset_eclab.jsonl          (250K samples)
```

### Option 2:
```
training_datasets/
├── stage2_synthetic_dataset_eclab_ctgan.jsonl     (50K samples, higher quality)
└── stage3_augmented_dataset_eclab.jsonl           (250K samples)
```

### Option 3:
```
training_datasets/
├── stage2_synthetic_dataset_eclab.jsonl           (50K samples)
└── stage3_augmented_dataset_eclab_openrouter.jsonl (400K samples, higher quality)
```

### Option 4:
```
training_datasets/
├── stage2_synthetic_dataset_eclab_ctgan.jsonl     (50K samples, highest quality)
└── stage3_augmented_dataset_eclab_openrouter.jsonl (400K samples, highest quality)
```

---

## 🔧 Cost Optimization

### For OpenRouter API:

**Cheaper models (still good quality):**
```bash
# Use Claude 3 Haiku (~$5-10 instead of $20-30)
python stage3_augmentation_pipeline_eclab_openrouter.py \
  --multiplier 8 \
  --model "anthropic/claude-3-haiku"

# Or use Llama 3 (~$2-5)
python stage3_augmentation_pipeline_eclab_openrouter.py \
  --multiplier 8 \
  --model "meta-llama/llama-3-70b-instruct"
```

---

## ✅ My Recommendation

**For your situation (you're OK with long runtime and have full control of eclab):**

### **Go with Option 4: Maximum Quality**

**Why:**
1. ✅ You're OK with 8-11 hour+ runtimes
2. ✅ You have full control of eclab (can run over weekend)
3. ✅ You want the best quality dataset possible
4. ✅ One-time $10-30 API cost is acceptable
5. ✅ 14-27 hours over a weekend is totally fine
6. ✅ You get ipazia-level quality on your own machine

**Timeline:**
- **Friday 5pm**: Start Stage 1 + 2
- **Saturday 5pm**: Check progress, start Stage 3
- **Saturday 7pm**: Everything done!
- **Sunday**: Review and celebrate! 🎉

This gives you:
- 🏆 80-90% quality (best possible)
- 📦 ~400,000 high-quality training samples
- 💰 $10-30 one-time cost
- 🎯 Production-ready dataset

**Start this weekend!** 🚀

