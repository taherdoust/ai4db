# Pipeline Improvements Quick Reference
**For Fine-Tuning 14B Parameter Models**

## Schema Priority (CIM Wizard Database)

| Priority | Focus | Target % |
|----------|-------|----------|
| **P1** | `cim_vector` internal | 35-40% |
| **P2** | `cim_vector` + `cim_raster` | 15-20% |
| **P2** | `cim_vector` + `cim_census` | 10-15% |
| **P2** | `cim_vector` + `cim_network` | 10-15% |
| **P3** | Other schemas internal | 5-8% each |
| **P4** | Cross non-vector | <5% |

## Immediate Actions (Week 1)

### 1. Analyze Current Distribution
```bash
cd /mnt/HDD_Data/000products/coesi1/ai4db
python analyze_schema_distribution.py \
  --input training_datasets/stage2_synthetic_dataset_ipazia.jsonl
```

### 2. Continue Stage 3 to 10K
```bash
# On ipazia
ssh castangia@ipazia126.polito.it
cd /media/space/castangia/Ali_workspace
python stage3_augmentation_pipeline_eclab_openrouter_enhanced.py
```
**Result:** 38K samples total (~$2.50, 16 more hours)

### 3. Start Fine-Tuning Experiments
Once you have 38K samples:
- Dataset A: Question → SQL (38K samples)
- Dataset B: Question → Instruction (38K samples)
- Split: 80% train, 10% val, 10% test

## Quality-First Improvements (Priority Order)

### CRITICAL (Do First)
1. **Schema distribution analysis** - Understand current state
2. **Continue Stage 3** - Get 38K samples
3. **Fine-tune initial models** - Establish baseline performance

### HIGH (Week 2-3)
1. **Stage 1 enhancements:**
   - Add schema priority weighting
   - Increase variations to 300
   - Focus 60% on cim_vector
   - Generate 10K-15K samples

2. **Stage 2 retraining:**
   - Add schema-aware features
   - Stratified sampling by priority
   - Generate 30K-50K samples

3. **Stage 3 enhancements:**
   - Schema-aware prompts
   - Instruction quality scoring
   - Process 50K Stage 2 samples

### MEDIUM (Week 4+)
1. Advanced deduplication
2. GPT-4o upgrade (partial, for quality boost)
3. Validation pipeline
4. Generate 100K+ total samples

## Expected Investment

| Phase | Cost | Time | Samples | Quality |
|-------|------|------|---------|---------|
| Current checkpoint | $1.25 | 16h | 19K | ⭐⭐⭐⭐⭐ |
| Complete Stage 3 | +$1.25 | +16h | 38K | ⭐⭐⭐⭐⭐ |
| Enhanced pipeline | +$10-25 | +60h | 100K+ | ⭐⭐⭐⭐⭐ |

## Key Quality Metrics for 14B Models

- Total samples: 50K-100K
- Unique questions: >90%
- Unique instructions: >85%
- Schema distribution: Matches priorities
- Reasoning depth: >95%
- SQL validity: >99%

## Decision: Quality vs Speed

**Recommended Path (Quality First):**
1. Complete current Stage 3 → 38K samples
2. Fine-tune initial models → Establish baseline
3. Analyze results → Identify weaknesses
4. Implement enhancements → Targeted improvements
5. Generate enhanced dataset → 100K samples
6. Fine-tune final models → Production quality

**Fast Path (If Time-Critical):**
1. Complete current Stage 3 → 38K samples
2. Fine-tune directly → Good results expected
3. Iterate based on performance

**Current recommendation:** Quality-first path for best 14B model performance.

