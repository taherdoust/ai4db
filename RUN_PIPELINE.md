# Three-Stage Dataset Creation Pipeline - Quick Start Guide

## 📋 Overview

This guide walks you through the complete three-stage pipeline for creating a comprehensive Text-to-Spatial-SQL dataset.

```
Stage 1 (Rule-Based) → 10,000 samples → 3 hours
Stage 2 (SDV)        → 50,000 samples → 1-2 days  
Stage 3 (NL Aug)     → 200,000 samples → 6 hours
────────────────────────────────────────────────
Total: ~260,000 samples in ~3 days
```

---

## 🚀 Stage 1: Enhanced Rule-Based Generation

### Installation

```bash
cd /home/eclab/Desktop/ai4db
conda activate ai4db

# No additional packages needed - Python standard library only!
```

### Run Stage 1

```bash
# Default: 200 variations per template, 100 evaluation samples
python stage1_enhanced_generator.py

# Custom configuration
python stage1_enhanced_generator.py 200 100

# Quick test (10 variations)
python stage1_enhanced_generator.py 10 5
```

### Expected Output

```
training_datasets/
├── stage1_enhanced_dataset.jsonl       # Main dataset (~10,000 samples)
├── stage1_enhanced_dataset_eval.jsonl  # Evaluation subset (~100 samples)
└── stage1_enhanced_dataset_stats.json  # Comprehensive statistics
```

### Output Schema (Each Sample)

```json
{
  "id": "cim_stage1_000001",
  "database_id": 1,
  "database_name": "cim_wizard",
  
  "question": "Find buildings of specific type with area above threshold...",
  "question_tone": "DIRECT",
  
  "sql_postgis": "SELECT b.building_id, ...",
  "sql_spatialite": "SELECT b.building_id, ...",
  
  "sql_type": "SPATIAL_JOIN",
  "sql_taxonomy": {
    "operation_type": "SPATIAL_JOIN",
    "has_cte": false,
    "has_subquery": false,
    ...
  },
  
  "difficulty": {
    "query_complexity": "MEDIUM",
    "spatial_complexity": "INTERMEDIATE",
    "schema_complexity": "SINGLE_SCHEMA",
    "function_count": "3-5",
    "join_count": "1-2",
    "overall_difficulty": "MEDIUM",
    "complexity_score": 4
  },
  
  "usage_frequency": "VERY_HIGH",
  
  "database_schema": {
    "schemas": ["cim_vector"],
    "tables": ["cim_vector.building", "cim_vector.building_properties"],
    "columns": ["building_id", "building_geometry", "type", "area"],
    "geometry_columns": ["building_geometry"]
  },
  
  "spatial_functions": ["ST_Area", "ST_Intersects"],
  "spatial_function_categories": {
    "predicates": ["ST_Intersects"],
    "measurements": ["ST_Area"],
    "processing": [],
    ...
  },
  
  "evidence": {...},
  "instruction": "Convert this natural language question to PostGIS...",
  
  "results": null,  # Will be filled for evaluation samples
  "has_results": true,
  
  "stage": "stage1_enhanced",
  "template_id": "CIM_A1_buildings_by_type_area_var_1",
  "complexity_level": "A",
  "tags": ["cim_building", "area_filter", ...],
  "generation_params": {...},
  "generated_at": "2025-01-02T10:30:00"
}
```

### Verify Stage 1 Output

```bash
# Check number of samples
wc -l training_datasets/stage1_enhanced_dataset.jsonl

# View statistics
cat training_datasets/stage1_enhanced_dataset_stats.json | python -m json.tool | head -50

# Sample a few examples
head -3 training_datasets/stage1_enhanced_dataset.jsonl | python -m json.tool
```

---

## 🤖 Stage 2: SDV Synthetic Generation

### Installation

```bash
# Install SDV and dependencies
pip install sdv==1.9.0 sqlparse==0.4.4 torch==2.0.0

# Verify installation
python -c "import sdv; print(f'SDV version: {sdv.__version__}')"
```

### Implementation Status

⚠️ **Stage 2 implementation is in progress** - see `STAGE2_SDV_PLAN.md` for detailed design.

**Key Design Decisions:**
- **Model Selected:** CTGANSynthesizer (best quality for complex structured data)
- **Alternative:** GaussianCopulaSynthesizer (if CTGAN too slow)
- **Approach:** Hybrid (CTGAN for structure + Rule-based for tokens)
- **Target:** 50,000 synthetic samples with 75%+ quality score

**Next Steps:**
1. Review `STAGE2_SDV_PLAN.md` for complete design
2. Implement `stage2_sdv_pipeline.py` based on the plan
3. Train CTGAN on Stage 1 features
4. Generate and validate synthetic SQL

---

## 💬 Stage 3: Natural Language Augmentation

### Installation

```bash
# Install NL augmentation tools
pip install transformers==4.35.0 sentence-transformers==2.2.0 sacremoses==0.1.0

# Download NLTK data
python -m nltk.downloader punkt averaged_perceptron_tagger
```

### Implementation Status

⚠️ **Stage 3 implementation is planned** - will create after Stage 2 is complete.

**Planned Augmentation Methods:**
1. Template-based paraphrasing (rule-based)
2. Back-translation (multilingual models)
3. LLM-based paraphrasing (T5/BART)
4. Difficulty-level variations
5. Question tone variations

**Target:** 10 NL variations per SQL → 200,000+ final samples

---

## 📊 Pipeline Progress Tracking

### Current Status (2025-01-02)

- ✅ **Stage 1: Enhanced Rule-Based Generator** - COMPLETE
  - File: `stage1_enhanced_generator.py`
  - Output: Comprehensive metadata schema
  - Features: Question tone, SQL taxonomy, multi-dimensional difficulty
  - Status: Ready to run

- 🚧 **Stage 2: SDV Synthetic Generation** - PLANNING COMPLETE
  - File: `STAGE2_SDV_PLAN.md`
  - Design: CTGANSynthesizer-based hybrid approach
  - Features: Schema-aware generation, quality filtering
  - Status: Ready for implementation

- 📋 **Stage 3: NL Augmentation** - PLANNED
  - Design: Multi-method augmentation
  - Target: 10× multiplication
  - Status: Awaiting Stage 2 completion

---

## 🧪 Testing & Validation

### Stage 1 Quick Test

```python
# Test Stage 1 with minimal samples
python stage1_enhanced_generator.py 10 5

# Verify output
import json

# Load and inspect
with open('training_datasets/stage1_enhanced_dataset.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

print(f"Total samples: {len(samples)}")
print(f"First sample keys: {list(samples[0].keys())}")
print(f"SQL types: {set(s['sql_type'] for s in samples)}")
print(f"Difficulty levels: {set(s['difficulty']['overall_difficulty'] for s in samples)}")
print(f"Question tones: {set(s['question_tone'] for s in samples)}")
```

### Validate Schema Compliance

```python
# Verify all samples have required fields
required_fields = [
    'id', 'database_id', 'question', 'question_tone',
    'sql_postgis', 'sql_spatialite', 'sql_type',
    'difficulty', 'usage_frequency', 'database_schema',
    'spatial_functions', 'instruction', 'results'
]

for i, sample in enumerate(samples):
    missing = [f for f in required_fields if f not in sample]
    if missing:
        print(f"Sample {i} missing fields: {missing}")
    else:
        print(f"Sample {i}: ✓ All required fields present")
```

---

## 📈 Expected Timeline

### Complete Pipeline Execution

| Stage | Task | Duration | Output |
|-------|------|----------|--------|
| **Stage 1** | Rule-based generation | 3 hours | 10,000 samples |
| **Stage 2** | CTGAN training | 2-4 hours | - |
| **Stage 2** | Synthetic generation | 30 min | 50,000 samples |
| **Stage 2** | Quality filtering | 1 hour | - |
| **Stage 3** | NL augmentation | 6 hours | 200,000 samples |
| **Stage 3** | Quality control | 2 hours | - |
| **Total** | End-to-end | ~3 days | 260,000 samples |

### Recommended Execution Schedule

**Day 1 (Morning):**
- ✅ Run Stage 1 (3 hours)
- ✅ Verify output quality
- ✅ Start Stage 2 CTGAN training (2-4 hours)

**Day 1 (Afternoon):**
- ⏳ Wait for CTGAN training
- 📊 Analyze Stage 1 statistics
- 📝 Review Stage 2 outputs

**Day 2 (Morning):**
- ✅ Generate Stage 2 synthetic samples (30 min)
- ✅ Quality filtering (1 hour)
- ✅ Validation and statistics

**Day 2 (Afternoon):**
- ✅ Start Stage 3 NL augmentation (6 hours)

**Day 3 (Morning):**
- ✅ Complete Stage 3
- ✅ Final quality control
- ✅ Generate final dataset statistics

---

## 🔍 Troubleshooting

### Stage 1 Issues

**Problem:** Import errors
```bash
# Solution: Ensure you're in the project directory
cd /home/eclab/Desktop/ai4db
python stage1_enhanced_generator.py
```

**Problem:** Out of memory
```bash
# Solution: Reduce variations
python stage1_enhanced_generator.py 50 25  # Smaller dataset
```

**Problem:** Slow generation
```bash
# Solution: This is normal - 10K samples takes ~3 hours
# Use 'top' or 'htop' to monitor progress
```

### Stage 2 Issues (Future)

**Problem:** CTGAN training too slow
```
Solution 1: Use GPU (cuda=True)
Solution 2: Reduce epochs (300 → 150)
Solution 3: Fallback to GaussianCopulaSynthesizer
```

**Problem:** Low quality synthetic SQL
```
Solution 1: Increase quality threshold (0.70 → 0.80)
Solution 2: Strengthen schema constraints
Solution 3: Retrain CTGAN with adjusted hyperparameters
```

---

## 📚 Additional Resources

- **Stage 1 Code:** `stage1_enhanced_generator.py`
- **Stage 2 Plan:** `STAGE2_SDV_PLAN.md`
- **CIM Schema:** `database_schemas/CIM_WIZARD_DATABASE_METADATA.md`
- **Papers:** `txt2ssql_paper/ref/` directory

---

## 🎯 Success Criteria

### Stage 1 ✅
- [x] Generate 10,000 samples
- [x] Comprehensive metadata (13+ fields)
- [x] Question tone classification
- [x] Multi-dimensional difficulty
- [x] SQL taxonomy
- [x] Evaluation subset (100 samples)

### Stage 2 (Target)
- [ ] Generate 50,000 samples
- [ ] Quality score ≥ 0.75
- [ ] Schema compliance ≥ 85%
- [ ] Novel patterns ≥ 40%

### Stage 3 (Target)
- [ ] Generate 200,000+ samples
- [ ] 10 NL variations per SQL
- [ ] Diversity score ≥ 0.85
- [ ] Grammaticality ≥ 85%

---

## 🆘 Need Help?

1. **Check logs:** All stages print detailed progress
2. **Verify data:** Inspect JSONL files with `head`, `jq`, or Python
3. **Review plans:** See `STAGE2_SDV_PLAN.md` for Stage 2 details
4. **Test incrementally:** Start with small samples (10 variations)

**Ready to start? Run Stage 1 now:**
```bash
python stage1_enhanced_generator.py
```

