# Quick Reference - Machine-Specific Pipelines

## 📋 Summary

You now have **machine-optimized pipelines** for both eclab and ipazia!

---

## 🚀 Quick Commands

### **eclab (RECOMMENDED)** - 5-6.5 hours total

```bash
# One-time setup
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b

# Run complete pipeline
cd ~/Desktop/ai4db
python stage1_enhanced_generator_stratified.py 200 100
python stage2_sdv_pipeline_eclab.py 50000
python stage3_augmentation_pipeline_eclab.py --multiplier 5
```

**Output:** ~250,000 training samples  
**Cost:** ~$0.10  
**Quality:** Good-Very Good (70-80%)

---

### **ipazia (ALTERNATIVE)** - 4-7 hours total

```bash
# Setup
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"

# Run complete pipeline
cd ~/path/to/ai4db
python stage1_enhanced_generator_stratified.py 200 100
python stage2_sdv_pipeline_ipazia.py 50000 300 true
python stage3_augmentation_pipeline_ipazia.py --multiplier 10
```

**Output:** ~500,000 training samples  
**Cost:** $10-30 (OpenRouter API)  
**Quality:** Excellent (80-90%)

---

## 📊 Timing Breakdown

| Stage | eclab | ipazia |
|-------|-------|--------|
| Stage 1 | 7-13 min | 4-7 min |
| Stage 2 | 1h 20min - 2h | 2.5-4.5h |
| Stage 3 | 3-4h | 1.5-2.5h |
| **TOTAL** | **5-6.5h** | **4-7h** |

---

## 📁 New Files Created

### Pipeline Scripts
- `stage2_sdv_pipeline_eclab.py` - CPU-optimized Stage 2 (GaussianCopula)
- `stage2_sdv_pipeline_ipazia.py` - GPU-optimized Stage 2 (CTGAN)
- `stage3_augmentation_pipeline_eclab.py` - Ollama/Mistral 7B integration
- `stage3_augmentation_pipeline_ipazia.py` - OpenRouter API integration

### Documentation
- `MACHINE_TIMING_ESTIMATES.md` - Comprehensive timing analysis
- `SETUP_INSTRUCTIONS.md` - Detailed setup guide
- `QUICK_REFERENCE.md` - This file!

---

## ⚙️ Key Differences

### Stage 2
| Feature | eclab | ipazia |
|---------|-------|--------|
| Model | GaussianCopula | CTGAN |
| Training | 10-15 min (CPU) | 2-4h (GPU, 300 epochs) |
| Batch Size | 5,000 | 10,000 |
| Parallelization | Sequential | 28 workers |
| Quality | 70-75% | 75-85% |

### Stage 3
| Feature | eclab | ipazia |
|---------|-------|--------|
| LLM | Ollama/Mistral 7B (local) | OpenRouter GPT-4 (API) |
| Speed | 2-3 sec/query | 0.5-1 sec/query |
| Cost | Free | $10-30 |
| Paraphrasing | Optional (CPU) | GPU-accelerated |
| Back-Translation | Not included | GPU-accelerated (3 languages) |
| Multiplier | 5x | 10x |

---

## 🎯 Decision Guide

**Choose eclab if:**
- ✅ You want full control (your machine)
- ✅ You prefer free/low-cost solution
- ✅ You're OK with 8-11 hours overnight
- ✅ 250K samples is sufficient
- ✅ Good quality (70-80%) is acceptable

**Choose ipazia if:**
- ⚡ You want maximum quality (80-90%)
- ⚡ You need 500K+ samples
- ⚡ You have OpenRouter API budget ($10-30)
- ⚡ You want faster GPU processing
- ⚠️ You're comfortable on shared server

---

## 🔧 Command Options

### Stage 2

**eclab:**
```bash
python stage2_sdv_pipeline_eclab.py [num_samples]
# Default: 50000
# Example: python stage2_sdv_pipeline_eclab.py 30000
```

**ipazia:**
```bash
python stage2_sdv_pipeline_ipazia.py [num_samples] [epochs] [use_gpu]
# Default: 50000 300 true
# Example: python stage2_sdv_pipeline_ipazia.py 50000 200 true
```

### Stage 3

**eclab:**
```bash
python stage3_augmentation_pipeline_eclab.py [options]
# --multiplier N     : Variations per sample (default: 5)
# --no-ollama        : Skip Ollama, use template-only

# Examples:
python stage3_augmentation_pipeline_eclab.py --multiplier 3
python stage3_augmentation_pipeline_eclab.py --no-ollama
```

**ipazia:**
```bash
python stage3_augmentation_pipeline_ipazia.py [options]
# --multiplier N         : Variations per sample (default: 10)
# --no-openrouter       : Skip OpenRouter API
# --no-paraphrase       : Skip paraphrasing
# --no-backtrans        : Skip back-translation

# Examples:
python stage3_augmentation_pipeline_ipazia.py --multiplier 8
python stage3_augmentation_pipeline_ipazia.py --no-paraphrase --no-backtrans
```

---

## 📈 Performance Tuning

### Faster on eclab (reduce time to 3-4 hours):
```bash
# Reduce multiplier
python stage3_augmentation_pipeline_eclab.py --multiplier 3
# Output: 150K samples instead of 250K
# Time: 1.5-2h instead of 3-4h for Stage 3

# OR skip Ollama
python stage3_augmentation_pipeline_eclab.py --no-ollama
# Time: 1-1.5h instead of 3-4h for Stage 3
```

### Cheaper on ipazia (reduce cost to $2-5):
```bash
# Use cheaper model
python stage3_augmentation_pipeline_ipazia.py \
  --multiplier 10 \
  --openrouter-model "anthropic/claude-3-haiku"
# Cost: ~$5 instead of $20-30
```

---

## 🐛 Quick Troubleshooting

### eclab

**Ollama not found:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b
ollama serve &
```

**Out of memory:**
```bash
# Edit stage2_sdv_pipeline_eclab.py line 346
# Change: batch_size=5000 → batch_size=2500
```

### ipazia

**API key error:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"
```

**GPU busy:**
```bash
# Use CPU mode
python stage2_sdv_pipeline_ipazia.py 50000 300 false
```

---

## 📞 Quick Help

| Issue | Solution |
|-------|----------|
| Ollama slow | Reduce `--multiplier` or use `--no-ollama` |
| Out of memory | Reduce batch sizes in scripts |
| GPU not available | Run CPU mode (see above) |
| API rate limit | Use cheaper model or reduce multiplier |
| Wrong output path | Check current directory with `pwd` |

---

## ✅ Verification Checklist

After pipeline completes:

```bash
# 1. Check all files exist
ls -lh training_datasets/*.jsonl

# 2. Verify sample counts
wc -l training_datasets/stage3_augmented_dataset_eclab.jsonl
# Should be ~250,000 lines for eclab

# 3. Check statistics
cat training_datasets/stage3_augmented_dataset_eclab_stats.json | jq .

# 4. Sample quality check
head -3 training_datasets/stage3_augmented_dataset_eclab.jsonl | jq '.question, .sql_postgis'

# 5. Disk space used
du -sh training_datasets/
```

---

## 🏁 Final Output Location

```
training_datasets/
├── stage1_enhanced_dataset.jsonl              (5,000 samples)
├── stage1_enhanced_dataset_eval.jsonl         (100 samples)
├── stage2_synthetic_dataset_eclab.jsonl       (50,000 samples)
└── stage3_augmented_dataset_eclab.jsonl       (250,000 samples) ← FINAL OUTPUT
```

or for ipazia:
```
training_datasets/
└── stage3_augmented_dataset_ipazia.jsonl      (500,000 samples) ← FINAL OUTPUT
```

---

## 🎉 Success!

Once complete, you'll have a high-quality training dataset for text-to-spatial-SQL!

**Next steps:**
1. Review quality statistics
2. Split into train/val/test sets
3. Train your model
4. Celebrate! 🎊

---

**Need more details?**
- See `MACHINE_TIMING_ESTIMATES.md` for comprehensive timing analysis
- See `SETUP_INSTRUCTIONS.md` for detailed setup and troubleshooting
- Check the original `README.md` for project overview

**Ready to start?** Run the commands above and check back in 6-8 hours! 🚀

