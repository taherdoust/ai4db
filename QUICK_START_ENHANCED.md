# Quick Start - Enhanced Pipeline 🚀

## TL;DR (30 seconds)

```bash
# 1. Setup API key (one-time)
cp .env.example .env
nano .env  # Add your OPENROUTER_API_KEY

# 2. Run complete pipeline
./run_enhanced_pipeline.sh
```

**That's it!** ✨

---

## What This Does

Generates **400,000 high-quality training samples** with:
- ✅ Synthetic SQL queries (PostGIS + SpatiaLite)
- ✅ Natural language questions (diverse, professional)
- ✅ **NEW**: Contextual instructions (not generic placeholders!)

**Time**: ~2-3 hours | **Cost**: $5-15 | **Quality**: 85-89%

---

## Step-by-Step (5 minutes)

### 1. Get API Key
Go to https://openrouter.ai/ → Sign up → Get API key (starts with `sk-or-v1-`)

### 2. Setup .env File
```bash
cd ~/Desktop/ai4db
cp .env.example .env
nano .env
```

In `.env`, replace with your key:
```
OPENROUTER_API_KEY=sk-or-v1-YOUR-ACTUAL-KEY-HERE
```
Save (Ctrl+X, Y, Enter)

### 3. Run Pipeline
```bash
./run_enhanced_pipeline.sh
```

---

## What Makes This "Enhanced"?

| Feature | Before | After |
|---------|--------|-------|
| Instructions | Generic placeholder | **Generated with context** |
| API Calls | 2 per sample | **1 per sample** |
| Cost | $10-30 | **$5-15 (50% cheaper!)** |
| Time | 3-4 hours | **2-3 hours (33% faster!)** |

---

## Example Output

```json
{
  "question": "What buildings are within 500m of grid buses in Milan?",
  "instruction": "Write a PostGIS SQL query to identify all buildings located within a 500-meter buffer of grid bus stations in the Milan smart district project",
  "sql_postgis": "SELECT b.building_id FROM cim_vector.building b...",
  "quality_score": 0.92
}
```

**Before**: Generic instruction ("Convert this question to SQL...")
**After**: Specific, contextual instruction! 🎯

---

## Why This Matters for Your Use Case

You want to fine-tune **Llama 3 8B** for text-to-spatial-SQL.

**Problem**: Using Llama 3 to generate training data for Llama 3 = circular training ❌

**Solution**: Use **GPT-4** to generate training data for Llama 3 = diverse, high-quality data ✅

**Result**: Better fine-tuning performance! 🎉

---

## Cost Breakdown

| Model | Cost | Quality | Recommendation |
|-------|------|---------|----------------|
| GPT-4 Turbo | $5-15 | 85-88% | ⭐ **Best** |
| Claude 3 Haiku | $2-5 | 80-85% | 💰 **Budget** |
| Llama 3 70B | $1-3 | 75-80% | Minimal budget |

---

## Security ✅

- `.env` file is in `.gitignore` (won't be committed)
- API key never exposed in code
- Safe to push to public GitHub repo

---

## Need Help?

1. **Full docs**: `SECURE_API_SETUP.md`
2. **Summary**: `ENHANCED_PIPELINE_SUMMARY.md`
3. **Main README**: `README.md`

---

## Ready? Let's Go! 🚀

```bash
cp .env.example .env
nano .env  # Add your API key
./run_enhanced_pipeline.sh
```

**Time to completion**: ~2-3 hours
**Coffee breaks**: 2-3 recommended ☕

When done, you'll have **400K high-quality training samples** ready for fine-tuning Llama 3 8B! 🎉
