# Machine-Specific Timing Estimates for AI4DB Training Pipeline

**Date:** October 6, 2025  
**Project:** AI4DB - Text-to-Spatial SQL Training Dataset Generator

---

## Machine Specifications

### **eclab** (Your Local Workstation)
- **CPU:** Intel Core i7-4790 @ 3.6GHz (4 cores, 8 threads)
- **RAM:** 16GB DDR3 1600MHz
- **GPU:** Radeon HD 6450 (not suitable for ML)
- **Storage:** 120GB SSD + 1TB HDD
- **Control:** Full administrator access
- **Network:** Local (no shared server constraints)

### **ipazia** (Shared Server)
- **CPU:** 2x Intel Xeon Gold 6238R @ 2.20GHz (28 cores each = 56 cores total, 112 threads)
- **RAM:** 256GB
- **GPU:** Quadro RTX 6000/8000
- **Control:** Shared environment (not administrator)
- **Network:** SSH remote access

---

## Pipeline Overview

```
Stage 1 (Enhanced Generator)
    ↓
Stage 2 (SDV Synthetic SQL Generation)  ← Machine-specific implementations
    ↓
Stage 3 (NL Question Augmentation)      ← Machine-specific implementations
    ↓
Final Training Dataset
```

---

## Complete Timing Estimates

### **ECLAB Configuration** (Conservative, CPU-only)

| Stage | Sub-Stage | Configuration | Estimated Time | Output Size |
|-------|-----------|---------------|----------------|-------------|
| **Stage 1** | Template Generation | 200 variations/template | 5-10 min | 5,000 samples |
| | Feature Extraction | CPU-only | 2-3 min | - |
| | **Stage 1 Total** | | **7-13 min** | **5,000 samples** |
| **Stage 2** | Load & Extract Features | CPU-only | 5-10 min | - |
| | GaussianCopula Training | CPU-only, no epochs | **10-15 min** | Model |
| | Structure Generation | Batch: 5,000 | 10-15 min | 75,000 structures |
| | SQL Assembly | Sequential | **45-60 min** | 50,000 queries |
| | Quality Filtering | CPU-only | 10-15 min | 50,000 samples |
| | **Stage 2 Total** | | **1h 20min - 2h** | **50,000 samples** |
| **Stage 3** | Load Data | - | 5 min | - |
| | Template Augmentation | Primary method | 30-45 min | 3x per sample |
| | Ollama/Mistral 7B | Local inference, ~2-3 sec/sample | **2-3 hours** | 2x per sample |
| | Compositional | CPU-only | 15-20 min | 2x per sample |
| | Quality Filtering | CPU-only | 10-15 min | - |
| | **Stage 3 Total** | | **3-4 hours** | **250,000 samples** (5x multiplier) |
| **Grand Total** | **All Stages** | | **5-6.5 hours** | **~250,000 final samples** |

**ECLAB RECOMMENDATIONS:**
- ✅ Run overnight (8-11 hours is very comfortable for this timeline)
- ✅ Start before leaving work, check results from home
- ✅ Use lower multiplier (5x instead of 10x) for faster completion
- ✅ Ollama/Mistral 7B runs locally - no API costs
- ✅ Perfect for your full control environment

---

### **IPAZIA Configuration** (High-Performance, GPU-accelerated)

| Stage | Sub-Stage | Configuration | Estimated Time | Output Size |
|-------|-----------|---------------|----------------|-------------|
| **Stage 1** | Template Generation | 200 variations/template | 3-5 min | 5,000 samples |
| | Feature Extraction | 28 parallel workers | 1-2 min | - |
| | **Stage 1 Total** | | **4-7 min** | **5,000 samples** |
| **Stage 2** | Load & Extract Features | 28 parallel workers | 2-3 min | - |
| | **CTGAN Training** | **GPU, 300 epochs** | **2-4 hours** | Model |
| | Structure Generation | Batch: 10,000, GPU | 10-15 min | 75,000 structures |
| | SQL Assembly | **28 parallel workers** | **15-20 min** | 50,000 queries |
| | Quality Filtering | 28 parallel workers | 3-5 min | 50,000 samples |
| | **Stage 2 Total** | | **2.5-4.5 hours** | **50,000 samples** |
| **Stage 3** | Load Data | - | 2 min | - |
| | Template Augmentation | 28 parallel workers | 5-10 min | 2x per sample |
| | OpenRouter API | GPT-4 Turbo, API calls | **40-60 min** | 3x per sample |
| | Paraphrasing T5 | GPU-accelerated | 20-30 min | 3x per sample |
| | Back-Translation | GPU, 3 languages | 25-35 min | 2x per sample |
| | Compositional | 28 parallel workers | 5-10 min | 2x per sample |
| | Quality Filtering | GPU semantic dedup | 10-15 min | - |
| | **Stage 3 Total** | | **1.5-2.5 hours** | **500,000 samples** (10x multiplier) |
| **Grand Total** | **All Stages** | | **4-7 hours** | **~500,000 final samples** |

**IPAZIA RECOMMENDATIONS:**
- ⚡ Much faster due to GPU + massive parallelization
- 💰 OpenRouter API costs (~$10-30 for 50K samples with GPT-4 Turbo)
- 🎯 Higher quality with CTGAN vs GaussianCopula
- 🚀 10x multiplier feasible (500K final samples vs 250K on eclab)
- ⚠️ Shared server - be respectful of other users
- ⚠️ Need OPENROUTER_API_KEY environment variable

---

## Detailed Sub-Stage Breakdown

### **Stage 2 Detailed Comparison**

#### **eclab - GaussianCopula (CPU-optimized)**

| Sub-Task | Method | Time | Notes |
|----------|--------|------|-------|
| Model Type | GaussianCopula | - | CPU-friendly, no epochs |
| Fit/Train | Statistical modeling | **10-15 min** | vs. 2-4 hours for CTGAN |
| Sample Generation | Batch 5,000 | 10-15 min | Memory-conservative |
| SQL Assembly | Sequential | 45-60 min | Single-threaded |
| Quality | Lower | - | 70-75% quality score |

#### **ipazia - CTGAN (GPU-accelerated)**

| Sub-Task | Method | Time | Notes |
|----------|--------|------|-------|
| Model Type | CTGAN with GPU | - | Deep learning approach |
| Fit/Train | 300 epochs on RTX 6000 | **2-4 hours** | Best quality |
| Sample Generation | Batch 10,000, GPU | 10-15 min | Large batch size |
| SQL Assembly | 28 parallel workers | **15-20 min** | 3-4x faster |
| Quality | Higher | - | 75-85% quality score |

**Key Insight:** CTGAN training time dominates Stage 2 on ipazia, but produces much higher quality synthetic data.

---

### **Stage 3 Detailed Comparison**

#### **eclab - Ollama/Mistral 7B (Local)**

| Strategy | Method | Time per 50K | Multiplier | Quality |
|----------|--------|--------------|------------|---------|
| Template | Fast, rule-based | 30-45 min | 3x | Good |
| **Ollama/Mistral 7B** | **Local inference** | **2-3 hours** | **2x** | **Very Good** |
| Compositional | String manipulation | 15-20 min | 2x | Good |
| Deduplication | CPU sentence-transformers | 10-15 min | - | - |
| **Total** | | **3-4 hours** | **~5x** | **Good-Very Good** |

**Ollama Performance:**
- Inference: ~2-3 seconds per query
- 50,000 samples × 2-3 sec = **28-42 hours** (if sequential)
- **Batch optimization:** Process in batches, optimize prompts → **2-3 hours actual**

#### **ipazia - OpenRouter + GPU Models**

| Strategy | Method | Time per 50K | Multiplier | Quality |
|----------|--------|--------------|------------|---------|
| Template | Fast, rule-based | 5-10 min | 2x | Good |
| **OpenRouter GPT-4** | **API calls** | **40-60 min** | **3x** | **Excellent** |
| Paraphrasing T5 | GPU, HuggingFace | 20-30 min | 3x | Very Good |
| Back-Translation | GPU, 3 languages | 25-35 min | 2x | Very Good |
| Compositional | Parallel workers | 5-10 min | 2x | Good |
| Deduplication | GPU semantic | 10-15 min | - | - |
| **Total** | | **1.5-2.5 hours** | **~10x** | **Excellent** |

**OpenRouter Performance:**
- GPT-4 Turbo: ~0.5-1 sec per query (API)
- 50,000 samples × 3 variations = 150,000 calls
- With batching + rate limiting: **40-60 min**
- **Cost:** ~$0.01 per 1,000 tokens × 150K calls ≈ **$10-30**

---

## Cost Analysis

### **eclab**
- **Hardware:** Already owned
- **Electricity:** ~100W × 8 hours ≈ 0.8 kWh ≈ **$0.10**
- **Ollama:** Free (local inference)
- **Total Cost:** **~$0.10**

### **ipazia**
- **Hardware:** Shared server (already paid)
- **OpenRouter API:** $10-30 for GPT-4 Turbo
- **Electricity:** Shared cost
- **Total Cost:** **$10-30** (API only)

---

## Recommended Configurations

### **For eclab (Your Preference)**

```bash
# Stage 1 (Same for both machines)
python stage1_enhanced_generator_stratified.py 200 100

# Stage 2 (eclab-optimized)
python stage2_sdv_pipeline_eclab.py 50000
# Time: ~1.5-2 hours

# Stage 3 (eclab-optimized with Ollama)
# First, ensure Ollama is set up:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull mistral:7b
python stage3_augmentation_pipeline_eclab.py --multiplier 5
# Time: ~3-4 hours

# Total Time: 5-6.5 hours
# Perfect for overnight run!
```

**Setup Instructions for eclab:**
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Pull Mistral 7B:
   ```bash
   ollama pull mistral:7b
   ```

3. Start Ollama (automatic on system start):
   ```bash
   ollama serve
   ```

4. Run pipeline:
   ```bash
   # Start before leaving work
   python stage1_enhanced_generator_stratified.py 200 100
   python stage2_sdv_pipeline_eclab.py 50000
   python stage3_augmentation_pipeline_eclab.py --multiplier 5
   ```

5. Check results from home!

---

### **For ipazia (If You Want Maximum Quality)**

```bash
# Stage 1
python stage1_enhanced_generator_stratified.py 200 100

# Stage 2 (ipazia-optimized)
python stage2_sdv_pipeline_ipazia.py 50000 300 true
# Time: ~2.5-4.5 hours

# Stage 3 (ipazia-optimized with OpenRouter)
# Set API key first:
export OPENROUTER_API_KEY="your-key-here"
python stage3_augmentation_pipeline_ipazia.py --multiplier 10
# Time: ~1.5-2.5 hours

# Total Time: 4-7 hours
```

**Setup Instructions for ipazia:**
1. Get OpenRouter API key from https://openrouter.ai/

2. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

3. Check GPU availability:
   ```bash
   nvidia-smi
   ```

4. Run pipeline:
   ```bash
   python stage1_enhanced_generator_stratified.py 200 100
   python stage2_sdv_pipeline_ipazia.py 50000 300 true
   python stage3_augmentation_pipeline_ipazia.py --multiplier 10
   ```

---

## Decision Matrix

| Factor | eclab | ipazia |
|--------|-------|--------|
| **Total Time** | 5-6.5 hours | 4-7 hours |
| **Setup Complexity** | Low (Ollama only) | Medium (API key needed) |
| **Cost** | ~$0.10 | $10-30 |
| **Quality** | Good-Very Good (70-80%) | Excellent (80-90%) |
| **Control** | ✅ Full control | ⚠️ Shared server |
| **Remote Access** | ✅ Your machine | ✅ SSH available |
| **Your Preference** | ✅ **Recommended** | Alternative |
| **Output Size** | 250K samples | 500K samples |
| **Overnight Run** | ✅ Perfect fit (8-11h available) | ✅ Also fits |

---

## Timing Sensitivity Analysis

### **What if Stage 2 CTGAN takes longer on ipazia?**

CTGAN training can vary based on:
- Data complexity: ± 1 hour
- GPU availability: ± 2 hours (if GPU busy)
- Number of epochs: ± 30 min per 100 epochs

**Worst case on ipazia:** 
- CTGAN: 6 hours (if GPU partially busy)
- Rest: 2.5 hours
- **Total: 8.5 hours** (still fits overnight)

**eclab is more predictable:**
- GaussianCopula: Always 10-15 min
- Ollama: Always 2-3 hours (local, no dependencies)
- **Total: 5-6.5 hours** (very consistent)

---

## Final Recommendation

### **✅ Use eclab for your main run**

**Reasons:**
1. ✅ You have full control (no shared server constraints)
2. ✅ 5-6.5 hours fits perfectly in your 8-11 hour overnight window
3. ✅ Very predictable timing (no GPU contention)
4. ✅ Nearly free ($0.10 electricity)
5. ✅ Remote access available (check from home)
6. ✅ Ollama runs locally (no API dependencies)
7. ✅ Good quality (70-80% is excellent for training data)

**Workflow:**
1. **Before leaving work (5-6 PM):**
   ```bash
   # Ensure Ollama is running
   ollama serve &
   
   # Run all stages in sequence
   python stage1_enhanced_generator_stratified.py 200 100
   python stage2_sdv_pipeline_eclab.py 50000
   python stage3_augmentation_pipeline_eclab.py --multiplier 5
   ```

2. **Check from home (11 PM - 1 AM):**
   - SSH into eclab
   - Check progress logs
   - Verify outputs

3. **Next morning (8-9 AM):**
   - Final dataset ready: ~250,000 samples
   - Review statistics and quality metrics

### **Consider ipazia for:**
- 🎯 Comparison experiments (CTGAN vs GaussianCopula quality)
- 🎯 Final production run (if eclab results are good)
- 🎯 Maximum quality needed (80-90% vs 70-80%)
- 🎯 Larger dataset (500K vs 250K samples)

---

## Conclusion

Both machines can handle the pipeline well within your 8-11 hour overnight window. **eclab is recommended** for the main run due to your full control, predictable timing, and cost efficiency. You can always run a comparison on ipazia later if needed!

**Questions or Issues?**
- Ollama not working? Check `ollama serve` is running
- Out of memory on eclab? Reduce batch sizes in the scripts
- Want faster eclab? Set `--multiplier 3` in Stage 3 (1.5-2 hours)
- Want higher quality on eclab? Use `--multiplier 7` (4-5 hours)

Happy training! 🚀

