# Environment Quick Reference

## 🚀 One-Command Setup

```bash
cd ~/Desktop/ai4db
./setup_environment.sh
```

This will:
- ✅ Create conda environment "ai4db"
- ✅ Install all required packages
- ✅ Test all installations
- ✅ Show you what to run next

---

## 📦 Files Created

| File | Purpose |
|------|---------|
| `requirements.txt` | Pip packages with versions |
| `environment.yml` | Complete conda environment |
| `setup_environment.sh` | Automated setup script |
| `INSTALLATION_GUIDE.md` | Detailed installation guide |

---

## ⚡ Quick Commands

### Setup (First Time):
```bash
# Automated setup
./setup_environment.sh

# Or manual
conda env create -f environment.yml
conda activate ai4db
```

### Update Environment:
```bash
conda activate ai4db
conda env update -f environment.yml --prune
```

### Activate:
```bash
conda activate ai4db
```

### Verify Installation:
```bash
conda activate ai4db
python -c "import sdv, torch, transformers; print('✅ Ready!')"
```

---

## 📋 Package List (Key Dependencies)

### Stage 1 (Built-in only):
- ✅ Python 3.10
- ✅ json, random, datetime (built-in)

### Stage 2 (SDV):
- ✅ `sdv==1.9.0` - Synthetic Data Vault
- ✅ `torch>=2.0.0` - Deep learning (for CTGAN)
- ✅ `sqlparse==0.4.4` - SQL parsing
- ✅ `numpy, pandas` - Data manipulation

### Stage 3 (NLP):
- ✅ `sentence-transformers>=2.2.2` - Semantic similarity
- ✅ `transformers>=4.35.0` - Paraphrasing, back-translation
- ✅ `requests>=2.31.0` - API calls

### Optional:
- ✅ `jupyter` - Interactive development
- ✅ `matplotlib, seaborn` - Visualization
- ✅ `tqdm` - Progress bars

---

## 🔧 Configuration by Machine

### eclab (CPU-only) - DEFAULT:
```yaml
# environment.yml already configured for CPU
- cpuonly  # ← Already set
```

### ipazia (GPU):
```yaml
# Edit environment.yml:
# Comment out: - cpuonly
# Uncomment: - pytorch-cuda=11.8
```

---

## 🧪 Verification Tests

### Quick Test:
```bash
conda activate ai4db
python -c "import sdv, torch; print('SDV:', sdv.__version__, '| CUDA:', torch.cuda.is_available())"
```

### Full Test:
```bash
conda activate ai4db
python << EOF
import numpy, pandas, torch, sdv, transformers, sentence_transformers, sqlparse
print("✅ NumPy:", numpy.__version__)
print("✅ Pandas:", pandas.__version__)
print("✅ PyTorch:", torch.__version__)
print("✅ SDV:", sdv.__version__)
print("✅ Transformers:", transformers.__version__)
print("✅ SQLParse:", sqlparse.__version__)
print("✅ CUDA:", torch.cuda.is_available())
print("✅ All packages working!")
EOF
```

### Test Stage 1:
```bash
conda activate ai4db
python stage1_enhanced_generator_stratified.py 10 5
# Should create small test dataset
```

---

## 🐛 Common Issues & Fixes

### "conda: command not found"
```bash
# Add conda to PATH
export PATH="$HOME/anaconda3/bin:$PATH"
# Or
export PATH="$HOME/miniconda3/bin:$PATH"
```

### "SDV installation failed"
```bash
pip install numpy pandas scikit-learn
pip install torch torchvision
pip install sdv==1.9.0
```

### "CUDA not available" (on ipazia)
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "Out of memory during CTGAN"
```bash
# Edit pipeline file, reduce batch_size:
# eclab: batch_size=500 → batch_size=250
# ipazia: batch_size=1000 → batch_size=500
```

---

## 📦 Disk Space

Required space:
- Base environment: **~5 GB**
- Sentence Transformers: **~500 MB**
- Ollama + Mistral 7B: **~4 GB**
- Training datasets: **~1-2 GB**

**Total**: ~10-12 GB (have at least **15 GB free**)

---

## 🔄 Environment Management

### List environments:
```bash
conda env list
```

### Export current environment:
```bash
conda env export > environment_backup.yml
```

### Remove environment:
```bash
conda deactivate
conda env remove -n ai4db
```

### Clone environment:
```bash
conda create --name ai4db_backup --clone ai4db
```

---

## 📝 Adding New Packages

### Add to environment.yml:
```yaml
dependencies:
  - your-package>=1.0.0
```

Then update:
```bash
conda env update -f environment.yml --prune
```

### Add with pip:
```bash
conda activate ai4db
pip install your-package
```

To persist:
```bash
pip freeze > requirements_new.txt
```

---

## 🎯 Ready to Run!

After setup, you can immediately run:

### Option 1: Fast (5-6.5 hours, free):
```bash
conda activate ai4db
python stage1_enhanced_generator_stratified.py 200 100
python stage2_sdv_pipeline_eclab.py 50000
python stage3_augmentation_pipeline_eclab.py --multiplier 5
```

### Option 2: High-Quality S2 (14-26 hours, free):
```bash
conda activate ai4db
python stage1_enhanced_generator_stratified.py 200 100
nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2.log 2>&1 &
python stage3_augmentation_pipeline_eclab.py --multiplier 5
```

### Option 3: High-Quality S3 (3-4 hours, $10-30):
```bash
conda activate ai4db
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"
python stage1_enhanced_generator_stratified.py 200 100
python stage2_sdv_pipeline_eclab.py 50000
python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8
```

### Option 4: Maximum Quality (14-27 hours, $10-30):
```bash
conda activate ai4db
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY"
python stage1_enhanced_generator_stratified.py 200 100
nohup python stage2_sdv_pipeline_eclab_ctgan.py 50000 300 > stage2.log 2>&1 &
# Wait for Stage 2, then:
python stage3_augmentation_pipeline_eclab_openrouter.py --multiplier 8
```

---

## 📚 More Information

- **Detailed installation**: See `INSTALLATION_GUIDE.md`
- **Machine comparison**: See `MACHINE_TIMING_ESTIMATES.md`
- **Quick reference**: See `QUICK_REFERENCE.md`
- **High-quality options**: See `ECLAB_HIGH_QUALITY_OPTIONS.md`

---

## ✅ Checklist

Before running pipelines:

- [ ] Environment created: `conda env list | grep ai4db`
- [ ] Environment activated: `conda activate ai4db`
- [ ] Packages verified: `python -c "import sdv; print('OK')"`
- [ ] Ollama installed (if using): `ollama list`
- [ ] OpenRouter key set (if using): `echo $OPENROUTER_API_KEY`
- [ ] Disk space checked: `df -h` (need 15+ GB free)

**All checked?** 🎉 **You're ready to run!**

```bash
conda activate ai4db
python stage1_enhanced_generator_stratified.py 200 100
```

