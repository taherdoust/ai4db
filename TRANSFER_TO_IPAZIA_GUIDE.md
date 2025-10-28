# Transfer Files to ipazia126 for Stage 2

## Files to Transfer

You need to transfer these files from **eclab (local)** to **ipazia126**:

1. **stage1_cim_dataset_annotated.jsonl** - Annotated Stage 1 with no_error labels (6,800 good samples)
2. **stage1_cim.py** - Updated schema definitions (fixes applied)
3. **stage2_sdv_pipeline_ipazia.py** - Updated Stage 2 script (now filters by no_error)

## Transfer Commands

### From Local Machine (eclab)

```bash
# Navigate to the ai4db directory
cd /home/ali/Desktop/HDD_Volume/000products/coesi/ai4db

# Transfer all three files in one command
scp training_datasets/stage1_cim_dataset_annotated.jsonl \
    stage1_cim.py \
    stage2_sdv_pipeline_ipazia.py \
    castangia@ipazia126.polito.it:/media/space/castangia/Ali_workspace/ai4db/

# Enter password when prompted
```

### Alternative: Transfer One by One

If you prefer to transfer files individually:

```bash
cd /home/ali/Desktop/HDD_Volume/000products/coesi/ai4db

# 1. Transfer annotated dataset
scp training_datasets/stage1_cim_dataset_annotated.jsonl \
    castangia@ipazia126.polito.it:/media/space/castangia/Ali_workspace/ai4db/training_datasets/

# 2. Transfer updated stage1_cim.py (for schema definitions)
scp stage1_cim.py \
    castangia@ipazia126.polito.it:/media/space/castangia/Ali_workspace/ai4db/

# 3. Transfer updated stage2 script
scp stage2_sdv_pipeline_ipazia.py \
    castangia@ipazia126.polito.it:/media/space/castangia/Ali_workspace/ai4db/
```

## Verify Transfer on ipazia126

After transfer, SSH to ipazia126 and verify:

```bash
# SSH to ipazia126
ssh castangia@ipazia126.polito.it

# Navigate to workspace
cd /media/space/castangia/Ali_workspace/ai4db

# Check files exist
ls -lh training_datasets/stage1_cim_dataset_annotated.jsonl
ls -lh stage1_cim.py
ls -lh stage2_sdv_pipeline_ipazia.py

# Verify annotated dataset has no_error field
head -1 training_datasets/stage1_cim_dataset_annotated.jsonl | python -m json.tool | grep no_error

# Expected output: "no_error": true  (or false)
```

## Run Stage 2 on ipazia126

Once files are transferred, run Stage 2:

```bash
# On ipazia126
cd /media/space/castangia/Ali_workspace/ai4db
conda activate ai4cimdb

# Run Stage 2 (2-3 hours with GPU)
python stage2_sdv_pipeline_ipazia.py 50000 300 true

# Or run with nohup to prevent disconnection
nohup python stage2_sdv_pipeline_ipazia.py 50000 300 true > stage2.log 2>&1 &

# Monitor progress
tail -f stage2.log
```

## What Stage 2 Will Do (with NoErr filtering)

1. **Load Stage 1 annotated dataset**: 7,600 samples
2. **Filter by no_error=True**: Keep only 6,800 working samples (89.5%)
3. **Train CTGAN**: Learn patterns from 6,800 high-quality samples
4. **Generate 75,000 synthetic structures**: 1.5x overgeneration
5. **Quality filter (threshold=0.70)**: Keep ~50,000 best samples
6. **Output**: `stage2_synthetic_dataset_ipazia.jsonl`

## Expected Stage 2 Results

- **Input**: 6,800 working Stage 1 samples (NoErr=True only)
- **Training time**: 20-30 minutes (GPU-accelerated CTGAN)
- **Assembly time**: 30-60 minutes (parallel processing)
- **Output**: 50,000 synthetic SQL samples
- **Expected NoErr rate**: 85-92% (learned from clean data)
- **Overall quality**: 89-93% (syntactic + schema + semantic)

## Key Improvements with NoErr Filtering

✓ **Before**: Stage 2 learned from 71% good + 29% bad samples  
✓ **After**: Stage 2 learns from 89.5% verified good samples only  
✓ **Impact**: Higher Stage 2 quality, fewer propagated errors  
✓ **Result**: Better Stage 3 augmentation input

## Troubleshooting

**Problem: Permission denied**
```bash
# Make sure you're using the correct path on ipazia126
ls /media/space/castangia/Ali_workspace/
```

**Problem: Module not found on ipazia126**
```bash
# Activate correct conda environment
conda activate ai4cimdb

# Check installed packages
pip list | grep -E "sdv|torch|pandas"
```

**Problem: GPU not available**
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, run with CPU mode (slower):
python stage2_sdv_pipeline_ipazia.py 50000 300 false
```

## File Sizes

- stage1_cim_dataset_annotated.jsonl: ~25 MB
- stage1_cim.py: ~100 KB
- stage2_sdv_pipeline_ipazia.py: ~50 KB
- **Total transfer**: ~25 MB (fast even on slow connection)

## Next Steps After Stage 2

Once Stage 2 completes on ipazia126:

1. **Transfer back to eclab** (for evaluation):
   ```bash
   # On local machine
   scp castangia@ipazia126.polito.it:/media/space/castangia/Ali_workspace/ai4db/training_datasets/stage2_synthetic_dataset_ipazia.jsonl \
       /home/ali/Desktop/HDD_Volume/000products/coesi/ai4db/training_datasets/
   ```

2. **Evaluate Stage 2 quality** (optional):
   ```bash
   cd /home/ali/Desktop/HDD_Volume/000products/coesi/ai4db
   conda activate aienv
   python evaluate_generation_quality.py \
     --input training_datasets/stage2_synthetic_dataset_ipazia.jsonl \
     --output training_datasets/stage2_quality_report.json \
     --stage 2
   ```

3. **Run Stage 3** (on ipazia126 or eclab):
   - Stage 3 uses OpenRouter API (LLM augmentation)
   - Can run on either machine (doesn't need GPU)
   - Recommended: Run on ipazia126 to keep full pipeline there

---

**Ready to transfer?** Just run the commands above!

