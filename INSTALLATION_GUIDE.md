# Installation Guide for AI4DB Environment

## Quick Start

Choose the installation method that works best for you:

---

## Option 1: Using Conda (Recommended)

### For eclab (CPU-only):

```bash
# Navigate to project directory
cd ~/Desktop/ai4db

# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate ai4db

# Verify installation
python -c "import sdv, torch, transformers; print('✅ All packages installed!')"
```

### For ipazia (GPU):

```bash
# Navigate to project directory
cd ~/Desktop/ai4db

# Edit environment.yml first:
# 1. Comment out line: - cpuonly
# 2. Uncomment lines: - pytorch-cuda=11.8

# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate ai4db

# Verify GPU access
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Option 2: Using pip with requirements.txt

### Basic Installation (all machines):

```bash
# Activate your conda environment (or create new one)
conda create -n ai4db python=3.10
conda activate ai4db

# Install from requirements.txt
pip install -r requirements.txt
```

### CPU-only Installation (eclab):

```bash
conda create -n ai4db python=3.10
conda activate ai4db

# Install CPU-only PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other packages
pip install -r requirements.txt
```

### GPU Installation (ipazia):

```bash
conda create -n ai4db python=3.10
conda activate ai4db

# Install GPU PyTorch first (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other packages
pip install -r requirements.txt
```

---

## Option 3: Manual Installation (Step-by-Step)

If you prefer to install packages one by one:

```bash
# Create environment
conda create -n ai4db python=3.10
conda activate ai4db

# Install core packages
conda install numpy pandas scipy scikit-learn matplotlib seaborn

# Install PyTorch (choose CPU or GPU version)
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# For GPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SDV (Stage 2)
pip install sdv==1.9.0

# Install SQL parser
pip install sqlparse==0.4.4

# Install transformers (Stage 3)
pip install sentence-transformers
pip install transformers tokenizers huggingface-hub

# Install utilities
pip install requests tqdm jupyter ipython
```

---

## Updating Existing Environment

If you already have the `ai4db` environment and want to update it:

```bash
# Activate environment
conda activate ai4db

# Update using environment.yml
conda env update -f environment.yml --prune

# Or update using pip
pip install -r requirements.txt --upgrade
```

---

## Verification Tests

### Test Core Packages:

```bash
conda activate ai4db

python << EOF
import numpy as np
import pandas as pd
import torch
import sdv
import transformers
import sentence_transformers
import sqlparse

print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ PyTorch version:", torch.__version__)
print("✅ SDV version:", sdv.__version__)
print("✅ Transformers version:", transformers.__version__)
print("✅ SQLParse version:", sqlparse.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ All core packages working!")
EOF
```

### Test Stage 1:

```bash
conda activate ai4db
python stage1_enhanced_generator_stratified.py 10 5
# Should generate small test dataset without errors
```

### Test Stage 2 (GaussianCopula):

```bash
conda activate ai4db
python -c "
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

# Create sample data
df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Test synthesizer
synth = GaussianCopulaSynthesizer(metadata)
synth.fit(df)
print('✅ GaussianCopula working!')
"
```

### Test Stage 2 (CTGAN):

```bash
conda activate ai4db
python -c "
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

# Create sample data
df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Test synthesizer
synth = CTGANSynthesizer(metadata, epochs=1, verbose=False)
synth.fit(df)
print('✅ CTGAN working!')
"
```

### Test Stage 3 (Sentence Transformers):

```bash
conda activate ai4db
python -c "
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embedding = model.encode('test sentence')
print('✅ Sentence Transformers working!')
print('✅ Embedding shape:', embedding.shape)
"
```

---

## Package Details

### Required for All Stages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing

### Required for Stage 1:
- Built-in Python libraries only
- `cim_wizard_sql_generator.py` (included in repo)
- `rule_based_ssql_generator.py` (included in repo)

### Required for Stage 2:
- `sdv==1.9.0` - Synthetic Data Vault (GaussianCopula + CTGAN)
- `sqlparse==0.4.4` - SQL parsing and validation
- `torch` - Deep learning framework (for CTGAN)

### Required for Stage 3:
- `sentence-transformers` - Semantic similarity
- `transformers` - Paraphrasing, back-translation
- `requests` - API calls (Ollama, OpenRouter)

### Optional but Recommended:
- `jupyter` - Interactive development
- `matplotlib`, `seaborn` - Data visualization
- `tqdm` - Progress bars
- `black`, `flake8` - Code quality

---

## Special Setup: Ollama (for eclab Stage 3)

If using the Ollama version of Stage 3:

```bash
# Install Ollama (one-time)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Mistral 7B model
ollama pull mistral:7b

# Verify
ollama list
# Should show: mistral:7b

# Start Ollama server (runs automatically after reboot)
ollama serve &
```

---

## Special Setup: OpenRouter API (for Stage 3 with GPT-4)

If using OpenRouter version of Stage 3:

```bash
# Get API key from https://openrouter.ai/

# Set environment variable (add to ~/.bashrc for persistence)
export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY-HERE"

# Or add to ~/.bashrc
echo 'export OPENROUTER_API_KEY="sk-or-v1-YOUR-KEY-HERE"' >> ~/.bashrc
source ~/.bashrc

# Verify
python -c "import os; print('API Key set:', bool(os.environ.get('OPENROUTER_API_KEY')))"
```

---

## Troubleshooting

### Issue: SDV installation fails

```bash
# Try installing dependencies first
pip install numpy pandas scikit-learn
pip install torch torchvision

# Then install SDV
pip install sdv==1.9.0
```

### Issue: PyTorch CUDA version mismatch

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Sentence Transformers download fails

```bash
# Download models manually
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print('Model downloaded successfully!')
"
```

### Issue: Out of memory during CTGAN training

```bash
# Edit the pipeline file and reduce batch_size
# For eclab: batch_size=500 → batch_size=250
# For ipazia: batch_size=1000 → batch_size=500
```

### Issue: Import errors after installation

```bash
# Make sure you're in the right environment
conda activate ai4db

# Verify environment
conda list | grep -E "sdv|torch|transformers"

# Reinstall if needed
pip install --force-reinstall sdv==1.9.0
```

---

## Environment Management

### Export your environment:

```bash
# Export conda environment
conda env export > environment_backup.yml

# Export pip packages only
pip freeze > requirements_backup.txt
```

### Create backup:

```bash
# Backup before major updates
conda env export > environment_$(date +%Y%m%d).yml
```

### Remove environment:

```bash
# Deactivate first
conda deactivate

# Remove environment
conda env remove -n ai4db
```

### Clone environment:

```bash
# Clone existing environment
conda create --name ai4db_backup --clone ai4db
```

---

## Disk Space Requirements

Approximate disk space needed:

- **Base environment**: ~5 GB
- **Sentence Transformers models**: ~500 MB
- **Ollama + Mistral 7B**: ~4 GB
- **Training datasets**: ~1-2 GB
- **CTGAN models**: ~500 MB

**Total**: ~10-12 GB

Make sure you have at least **15 GB free** on your system.

---

## Next Steps

After installation:

1. **Verify installation**: Run verification tests above
2. **Download models**: Sentence transformers will download on first use
3. **Setup Ollama** (if using): Install and pull Mistral 7B
4. **Setup OpenRouter** (if using): Get API key and set environment variable
5. **Run test**: Try Stage 1 with small dataset
6. **Run full pipeline**: Choose your configuration and start!

---

## Quick Reference

### Activate environment:
```bash
conda activate ai4db
```

### Update environment:
```bash
conda env update -f environment.yml --prune
```

### Check installed packages:
```bash
conda list
```

### Check Python version:
```bash
python --version
```

### Check PyTorch GPU support:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify all packages are installed: `conda list`
3. Check Python version: `python --version` (should be 3.10.x)
4. Check CUDA availability (if GPU): `nvidia-smi`
5. Try reinstalling specific package: `pip install --force-reinstall <package>`

**Ready to start!** 🚀

```bash
conda activate ai4db
python stage1_enhanced_generator_stratified.py 200 100
```

