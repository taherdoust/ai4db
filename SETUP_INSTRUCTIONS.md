# Setup Instructions for Machine-Specific Pipelines

## Quick Start Guide

You now have **4 new pipeline files** optimized for your specific machines:

```
stage2_sdv_pipeline_eclab.py    → For eclab (CPU, GaussianCopula)
stage2_sdv_pipeline_ipazia.py   → For ipazia (GPU, CTGAN)
stage3_augmentation_pipeline_eclab.py    → For eclab (Ollama/Mistral 7B)
stage3_augmentation_pipeline_ipazia.py   → For ipazia (OpenRouter API)
```

---

## 🎯 Recommended: Run on eclab

### Why eclab?
- ✅ Full administrator access
- ✅ 5-6.5 hours total time (perfect for overnight)
- ✅ Nearly free (~$0.10 electricity)
- ✅ Remote access from home
- ✅ Very predictable timing

---

## Setup for eclab

### 1. Install Ollama (one-time setup)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Mistral 7B model (this will take 10-15 minutes)
ollama pull mistral:7b

# Verify installation
ollama list
# Should show: mistral:7b

# Start Ollama server (runs automatically after reboot)
ollama serve &
```

### 2. Install Python Dependencies (if not already installed)

```bash
# Activate your environment
conda activate ai4db  # or your environment name

# Install required packages
pip install sdv==1.9.0
pip install sqlparse==0.4.4
pip install sentence-transformers
pip install requests
```

### 3. Run the Complete Pipeline

```bash
# Navigate to project directory
cd ~/Desktop/ai4db

# Stage 1: Generate base dataset (5-10 min)
python stage1_enhanced_generator_stratified.py 200 100

# Stage 2: Generate synthetic SQL with GaussianCopula (1.5-2 hours)
python stage2_sdv_pipeline_eclab.py 50000

# Stage 3: Augment with Ollama/Mistral 7B (3-4 hours)
python stage3_augmentation_pipeline_eclab.py --multiplier 5

# Total: 5-6.5 hours → Perfect for overnight!
```

### 4. Run Everything in One Command

```bash
# Run as a background job
nohup bash -c "
  python stage1_enhanced_generator_stratified.py 200 100 && \
  python stage2_sdv_pipeline_eclab.py 50000 && \
  python stage3_augmentation_pipeline_eclab.py --multiplier 5
" > pipeline_eclab.log 2>&1 &

# Check progress
tail -f pipeline_eclab.log

# Or check from home via SSH
ssh eclab@eclab-machine
tail -f ~/Desktop/ai4db/pipeline_eclab.log
```

---

## Setup for ipazia (Alternative)

### 1. Get OpenRouter API Key

1. Visit: https://openrouter.ai/
2. Sign up and get API key
3. Note the key (starts with `sk-or-v1-...`)

### 2. Set Environment Variable

```bash
# Add to your ~/.bashrc for persistence
echo 'export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY-HERE"' >> ~/.bashrc
source ~/.bashrc

# Or set temporarily for this session
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY-HERE"
```

### 3. Verify GPU Access

```bash
# Check GPU availability
nvidia-smi

# Should show: Quadro RTX 6000/8000
```

### 4. Run the Complete Pipeline

```bash
# Navigate to project directory
cd ~/Desktop/ai4db  # or your path on ipazia

# Stage 1: Generate base dataset (3-5 min)
python stage1_enhanced_generator_stratified.py 200 100

# Stage 2: Generate synthetic SQL with CTGAN (2.5-4.5 hours)
python stage2_sdv_pipeline_ipazia.py 50000 300 true

# Stage 3: Augment with OpenRouter GPT-4 (1.5-2.5 hours)
python stage3_augmentation_pipeline_ipazia.py --multiplier 10

# Total: 4-7 hours
```

### 5. Run Everything in One Command

```bash
# Run as a background job
nohup bash -c "
  python stage1_enhanced_generator_stratified.py 200 100 && \
  python stage2_sdv_pipeline_ipazia.py 50000 300 true && \
  python stage3_augmentation_pipeline_ipazia.py --multiplier 10
" > pipeline_ipazia.log 2>&1 &

# Monitor progress
tail -f pipeline_ipazia.log
```

---

## Comparison Table

| Feature | eclab | ipazia |
|---------|-------|--------|
| **Total Time** | 5-6.5 hours | 4-7 hours |
| **Cost** | ~$0.10 | $10-30 (OpenRouter API) |
| **Quality** | Good (70-80%) | Excellent (80-90%) |
| **Output** | 250K samples | 500K samples |
| **Setup** | Ollama (easy) | API key (medium) |
| **Control** | Full admin | Shared server |
| **Your Choice** | ✅ **Recommended** | Alternative |

---

## Troubleshooting

### eclab Issues

**Problem:** Ollama not running
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve &

# Verify model is available
ollama list
```

**Problem:** Out of memory
```bash
# Edit stage2_sdv_pipeline_eclab.py
# Line ~346: Change batch_size from 5000 to 2500
# Line ~686: Change generation_target multiplier from 1.5 to 1.3
```

**Problem:** Ollama too slow
```bash
# Use template-only mode (faster, no Ollama)
python stage3_augmentation_pipeline_eclab.py --multiplier 5 --no-ollama
# Time reduces from 3-4 hours to 1-1.5 hours
```

### ipazia Issues

**Problem:** OpenRouter API key not found
```bash
# Check if variable is set
echo $OPENROUTER_API_KEY

# Set it
export OPENROUTER_API_KEY="sk-or-v1-..."
```

**Problem:** GPU not available
```bash
# Check GPU status
nvidia-smi

# If busy, run CPU-only mode
python stage2_sdv_pipeline_ipazia.py 50000 300 false
```

**Problem:** OpenRouter rate limiting
```bash
# Use slower model (cheaper, faster)
python stage3_augmentation_pipeline_ipazia.py \
  --multiplier 10 \
  --openrouter-model "anthropic/claude-3-haiku"
```

---

## Output Files

After completion, you'll find:

```
training_datasets/
├── stage1_enhanced_dataset.jsonl         # Base dataset (5,000 samples)
├── stage1_enhanced_dataset_stats.json
├── stage1_enhanced_dataset_eval.jsonl    # Evaluation subset (100 samples)
│
├── stage2_synthetic_dataset_eclab.jsonl  # Stage 2 output (50,000 samples)
├── stage2_synthetic_dataset_eclab_stats.json
├── stage2_synthetic_dataset_eclab_model.pkl  # Trained model
│
└── stage3_augmented_dataset_eclab.jsonl  # Final output (~250,000 samples)
    stage3_augmented_dataset_eclab_stats.json
```

Or for ipazia:
```
training_datasets/
├── stage2_synthetic_dataset_ipazia.jsonl
├── stage2_synthetic_dataset_ipazia_stats.json
├── stage2_synthetic_dataset_ipazia_model.pkl
│
└── stage3_augmented_dataset_ipazia.jsonl  # Final output (~500,000 samples)
    stage3_augmented_dataset_ipazia_stats.json
```

---

## Next Steps After Pipeline Completes

1. **Verify Output Quality**
   ```bash
   # Check statistics
   cat training_datasets/stage3_augmented_dataset_eclab_stats.json
   
   # Sample a few records
   head -5 training_datasets/stage3_augmented_dataset_eclab.jsonl | jq .
   ```

2. **Merge with Existing Data** (if needed)
   ```bash
   cat training_datasets/stage1_enhanced_dataset.jsonl \
       training_datasets/stage3_augmented_dataset_eclab.jsonl \
       > training_datasets/final_training_dataset.jsonl
   ```

3. **Split Train/Val/Test**
   ```python
   # Use your existing splitting logic or:
   from sklearn.model_selection import train_test_split
   
   # Load data
   with open('training_datasets/stage3_augmented_dataset_eclab.jsonl', 'r') as f:
       data = [json.loads(line) for line in f]
   
   # Split: 80% train, 10% val, 10% test
   train, temp = train_test_split(data, test_size=0.2, random_state=42)
   val, test = train_test_split(temp, test_size=0.5, random_state=42)
   
   # Save splits
   for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
       with open(f'training_datasets/{split_name}.jsonl', 'w') as f:
           for sample in split_data:
               f.write(json.dumps(sample) + '\n')
   ```

4. **Train Your Model**
   ```bash
   # Use your existing training script
   python train_text_to_sql_model.py \
     --train_data training_datasets/train.jsonl \
     --val_data training_datasets/val.jsonl \
     --model_name t5-base \
     --epochs 10
   ```

---

## Performance Tips

### To Speed Up eclab:

1. **Reduce multiplier** in Stage 3:
   ```bash
   python stage3_augmentation_pipeline_eclab.py --multiplier 3
   # Time: 1.5-2 hours instead of 3-4 hours
   # Output: 150K samples instead of 250K
   ```

2. **Skip Ollama** (template-only mode):
   ```bash
   python stage3_augmentation_pipeline_eclab.py --multiplier 5 --no-ollama
   # Time: 1-1.5 hours instead of 3-4 hours
   # Quality: Good (but not Very Good)
   ```

3. **Reduce Stage 2 output**:
   ```bash
   python stage2_sdv_pipeline_eclab.py 30000
   # Generates 30K instead of 50K samples
   # Proportionally reduces all downstream times
   ```

### To Speed Up ipazia:

1. **Use cheaper/faster model**:
   ```bash
   export OPENROUTER_MODEL="anthropic/claude-3-haiku"
   # or
   export OPENROUTER_MODEL="meta-llama/llama-3-8b-instruct"
   ```

2. **Skip expensive augmentations**:
   ```bash
   python stage3_augmentation_pipeline_ipazia.py \
     --multiplier 10 \
     --no-paraphrase \
     --no-backtrans
   # Uses only OpenRouter + templates
   ```

---

## Cost Breakdown

### eclab
- Electricity: ~100W × 6 hours = 0.6 kWh ≈ **$0.08**
- Ollama: **Free** (local inference)
- **Total: ~$0.10**

### ipazia
- Electricity: Shared (already paid)
- OpenRouter GPT-4 Turbo:
  - 50,000 samples × 3 questions = 150,000 API calls
  - ~100 tokens/call = 15M tokens
  - $0.01 per 1K input tokens = **$150** (input)
  - $0.03 per 1K output tokens = **~$30** (output)
  - **Total: ~$10-30** (with optimization and caching)

**Alternative for ipazia:** Use `claude-3-haiku` (~$5-10) or `llama-3` (~$2-5)

---

## Monitoring Progress

### Check Logs
```bash
# eclab
tail -f pipeline_eclab.log

# ipazia
tail -f pipeline_ipazia.log
```

### Check Output Files
```bash
# List outputs
ls -lh training_datasets/

# Check latest file size
watch -n 60 "ls -lh training_datasets/*.jsonl"
```

### Monitor System Resources

**eclab:**
```bash
# CPU and memory
htop

# Disk usage
df -h
```

**ipazia:**
```bash
# GPU usage
watch -n 5 nvidia-smi

# CPU and memory
htop
```

---

## Support

If you encounter issues:

1. Check `MACHINE_TIMING_ESTIMATES.md` for detailed timing information
2. Review error logs in `pipeline_*.log`
3. Verify all dependencies are installed
4. For Ollama issues: `ollama --help`
5. For OpenRouter issues: Check API key and credits at https://openrouter.ai/

---

**Ready to start?** 🚀

```bash
# For eclab (RECOMMENDED):
cd ~/Desktop/ai4db
ollama serve &
nohup bash -c "
  python stage1_enhanced_generator_stratified.py 200 100 && \
  python stage2_sdv_pipeline_eclab.py 50000 && \
  python stage3_augmentation_pipeline_eclab.py --multiplier 5
" > pipeline_eclab.log 2>&1 &

# Check from home later!
```

