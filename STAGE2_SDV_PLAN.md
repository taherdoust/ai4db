# Stage 2: SDV (Synthetic Data Vault) - Detailed Implementation Plan

## 📊 Overview

**Objective:** Use Synthetic Data Vault (SDV) to generate 50,000-100,000 novel SQL queries that:
1. Learn patterns from Stage 1's 10,000 high-quality samples
2. Respect CIM Wizard schema constraints
3. Maintain syntactic validity and semantic coherence
4. Multiply dataset size by 5-10×

---

## 🤖 SDV Model Selection: Comprehensive Analysis

### Available SDV Models

| Model | Type | Best For | Pros | Cons |
|-------|------|----------|------|------|
| **CTGANSynthesizer** | GAN-based | Mixed data types, complex distributions | High quality, handles complex patterns | Slower training, requires GPU |
| **TVAESynthesizer** | VAE-based | Faster alternative to CTGAN | Faster than CTGAN, good quality | Less stable for complex data |
| **GaussianCopulaSynthesizer** | Statistical | Fast generation, simple distributions | Very fast, stable | Limited complexity modeling |
| **CopulaGANSynthesizer** | Hybrid | Best of both worlds | Combines GAN quality with copula speed | Memory intensive |
| **PARSynthesizer** | Sequential | Time-series, sequential data | Excellent for sequences | Not ideal for SQL structure |

---

## ✅ **RECOMMENDED: Hybrid Approach with CTGANSynthesizer**

### Why CTGANSynthesizer for SQL Structure Generation?

**Justification based on SQL generation requirements:**

1. **Mixed Data Types Handling**
   - SQL has both categorical (table names, operators) and numerical (complexity scores) features
   - CTGAN excels at learning distributions across mixed types
   - Better preservation of correlations between features

2. **Complex Pattern Learning**
   - SQL queries have complex structural dependencies (e.g., JOIN count correlates with complexity)
   - CTGAN's adversarial training captures subtle patterns
   - Important for maintaining semantic coherence

3. **Quality vs. Speed Trade-off**
   - Training: ~2-4 hours for 10K samples (acceptable for one-time generation)
   - Generation: ~10-20 minutes for 50K samples (fast enough)
   - Quality: Significantly better than Gaussian Copula for complex data

4. **Empirical Evidence from Text-to-SQL Research**
   - Studies show GAN-based methods outperform statistical methods for code generation
   - CTGAN has been successfully used for SQL structure synthesis in prior work

### Why NOT Other Models?

**TVAESynthesizer:**
- ❌ Less stable for structured data like SQL
- ❌ More prone to mode collapse (generating repetitive patterns)
- ✅ Could be used as fallback if CTGAN is too slow

**GaussianCopulaSynthesizer:**
- ❌ Assumes Gaussian distributions (SQL structure is not Gaussian)
- ❌ Struggles with complex dependencies
- ✅ Good for quick prototyping only

**PARSynthesizer:**
- ❌ Designed for sequential/temporal data
- ❌ SQL structure is hierarchical, not sequential
- ✅ Could be used for token-level generation (future work)

**CopulaGANSynthesizer:**
- ✅ Best quality theoretically
- ❌ Very memory-intensive (may OOM with our feature set)
- ❌ Slower than CTGAN

---

## 🏗️ Proposed Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 2 HYBRID PIPELINE                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │  Component 1: Structure Generator (CTGAN)           │
    │  ─────────────────────────────────────────────      │
    │  Learns:                                            │
    │  • CTE count, JOIN count, subquery patterns         │
    │  • Complexity scores and correlations               │
    │  • Table/schema selection patterns                  │
    │  • Function usage distributions                     │
    │                                                     │
    │  Input Features:                                    │
    │  - cte_count, join_count, subquery_count           │
    │  - spatial_function_count, table_count             │
    │  - complexity_score (0-10)                         │
    │  - sql_type (categorical)                          │
    │  - difficulty_level (categorical)                  │
    │  - schema_complexity (categorical)                 │
    │                                                     │
    │  Output: Synthetic SQL structure specifications    │
    └─────────────────────────────────────────────────────┘
                              │
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │  Component 2: Token Selector (Rule-Based)           │
    │  ─────────────────────────────────────────────      │
    │  Uses Structure + Schema Constraints to:            │
    │  • Select valid tables from CIM schema              │
    │  • Choose appropriate spatial functions             │
    │  • Generate valid JOIN conditions                   │
    │  • Ensure geometry column compatibility             │
    │                                                     │
    │  Schema Enforcement Rules:                          │
    │  - Only use tables from CIM_SCHEMAS                 │
    │  - Respect valid_joins constraints                  │
    │  - Match function applicability to geometry types   │
    │  - Ensure column-table associations are valid       │
    └─────────────────────────────────────────────────────┘
                              │
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │  Component 3: SQL Assembler (Template-Based)        │
    │  ─────────────────────────────────────────────      │
    │  Constructs valid SQL from:                         │
    │  • Structure specification (from CTGAN)             │
    │  • Token selection (from rules)                     │
    │  • SQL grammar templates                            │
    │                                                     │
    │  Ensures:                                           │
    │  - Syntactic correctness                           │
    │  - Proper clause ordering (SELECT, FROM, WHERE...)  │
    │  - Valid PostGIS/SpatiaLite syntax                 │
    └─────────────────────────────────────────────────────┘
                              │
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │  Component 4: Quality Filter                        │
    │  ─────────────────────────────────────────────      │
    │  Multi-dimensional quality assessment:              │
    │  • Syntactic validity (SQL parsing)                │
    │  • Schema compliance (valid tables/columns)         │
    │  • Semantic coherence (logical query)              │
    │  • Diversity score (vs existing samples)           │
    │                                                     │
    │  Threshold: quality_score >= 0.70                  │
    └─────────────────────────────────────────────────────┘
                              │
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │            OUTPUT: 50K-100K Synthetic Samples       │
    └─────────────────────────────────────────────────────┘
```

---

## 📐 Feature Engineering for CTGAN

### Input Features to CTGAN

**Numerical Features (7):**
```python
numerical_features = {
    'cte_count': 0-5,              # Number of CTEs
    'join_count': 0-10,            # Number of JOINs
    'subquery_count': 0-5,         # Number of subqueries
    'spatial_function_count': 1-15, # Number of spatial functions
    'table_count': 1-8,            # Number of tables
    'complexity_score': 0-10,      # Overall complexity
    'schema_count': 1-3            # Number of schemas
}
```

**Categorical Features (6):**
```python
categorical_features = {
    'sql_type': ['SIMPLE_SELECT', 'SPATIAL_JOIN', 'AGGREGATION', ...],
    'difficulty_level': ['EASY', 'MEDIUM', 'HARD', 'EXPERT'],
    'schema_complexity': ['SINGLE_TABLE', 'SINGLE_SCHEMA', 'MULTI_SCHEMA'],
    'usage_frequency': ['CRITICAL', 'VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW'],
    'question_tone': ['DIRECT', 'INTERROGATIVE', ...],
    'primary_function_category': ['predicates', 'measurements', 'processing', ...]
}
```

**Feature Rationale:**
- **Small feature set** (13 features) → faster training, less overfitting
- **High-level abstractions** → captures patterns without memorizing exact SQL
- **Schema-agnostic** → CTGAN learns structure, not specific table names
- **Correlations preserved** → e.g., high complexity → more CTEs/JOINs

---

## 🔧 Implementation Details

### CTGAN Configuration

```python
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Metadata definition
metadata = SingleTableMetadata()

# Numerical columns
for col in numerical_features:
    metadata.update_column(col, sdtype='numerical')

# Categorical columns
for col in categorical_features:
    metadata.update_column(col, sdtype='categorical')

# Primary key
metadata.set_primary_key('sample_id')

# CTGAN Configuration
synthesizer = CTGANSynthesizer(
    metadata=metadata,
    
    # Training parameters (optimized for SQL structure)
    epochs=300,              # More epochs for better learning
    batch_size=500,          # Larger batch for stability
    
    # Generator/Discriminator architecture
    generator_dim=(256, 256),    # 2-layer generator
    discriminator_dim=(256, 256), # 2-layer discriminator
    
    # Learning rates
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    
    # Other parameters
    discriminator_steps=1,   # Balance G/D training
    log_frequency=True,
    verbose=True,
    
    # GPU acceleration (if available)
    cuda=True  # Set to False if no GPU
)
```

### Training Process

```python
# Step 1: Prepare training data
df = prepare_ctgan_training_data(stage1_samples)

# Step 2: Train CTGAN
print("Training CTGAN (estimated time: 2-4 hours)...")
synthesizer.fit(df)

# Step 3: Save trained model
synthesizer.save('models/ctgan_sql_structure.pkl')

# Step 4: Generate synthetic structures
print("Generating 75,000 synthetic structures...")
synthetic_structures = synthesizer.sample(num_rows=75000)

# Note: Generate 1.5x target to account for quality filtering
```

---

## 🛡️ Schema Constraint Enforcement

### CIM Wizard Schema Rules

```python
CIM_SCHEMA_RULES = {
    # Valid tables (from CIM_SCHEMAS)
    "valid_tables": [
        "cim_vector.building",
        "cim_vector.building_properties",
        "cim_vector.grid_bus",
        "cim_vector.grid_line",
        "cim_vector.project_scenario",
        "cim_census.census_geo",
        "cim_raster.dsm_raster",
        "cim_raster.dtm_raster",
        "cim_raster.building_height_cache"
    ],
    
    # Valid join pairs (schema-defined relationships)
    "valid_joins": [
        ("cim_vector.building", "cim_vector.building_properties"),
        ("cim_vector.building", "cim_census.census_geo"),
        ("cim_vector.building", "cim_raster.dsm_raster"),
        ("cim_vector.building", "cim_raster.dtm_raster"),
        ("cim_vector.building", "cim_raster.building_height_cache"),
        ("cim_vector.building_properties", "cim_vector.grid_bus"),
        ("cim_vector.grid_bus", "cim_vector.grid_line"),
        # ... more pairs
    ],
    
    # Spatial functions by geometry type
    "function_applicability": {
        "POLYGON": ["ST_Area", "ST_Intersects", "ST_Contains", "ST_Within", ...],
        "POINT": ["ST_X", "ST_Y", "ST_MakePoint", "ST_Distance", ...],
        "LINESTRING": ["ST_Length", "ST_StartPoint", "ST_EndPoint", ...]
    },
    
    # Primary keys for JOIN conditions
    "join_keys": {
        ("cim_vector.building", "cim_vector.building_properties"): "building_id",
        ("cim_vector.building_properties", "cim_vector.grid_bus"): 
            ["project_id", "scenario_id"],
        # ... more join keys
    }
}
```

### Enforcement Algorithm

```python
def enforce_schema_constraints(synthetic_structure, schema_rules):
    """
    Convert synthetic structure to valid CIM SQL
    """
    
    # 1. Select valid tables based on structure requirements
    required_tables = synthetic_structure['table_count']
    valid_tables = select_valid_tables(
        count=required_tables,
        schema_complexity=synthetic_structure['schema_complexity'],
        rules=schema_rules
    )
    
    # 2. Ensure valid JOIN paths
    if synthetic_structure['join_count'] > 0:
        join_pairs = find_valid_join_path(valid_tables, schema_rules)
    else:
        join_pairs = []
    
    # 3. Select appropriate spatial functions
    spatial_funcs = select_spatial_functions(
        count=synthetic_structure['spatial_function_count'],
        category=synthetic_structure['primary_function_category'],
        geometry_types=get_geometry_types(valid_tables),
        rules=schema_rules
    )
    
    # 4. Assemble SQL from components
    sql = assemble_sql(
        tables=valid_tables,
        joins=join_pairs,
        functions=spatial_funcs,
        structure=synthetic_structure
    )
    
    return sql
```

---

## 📊 Quality Assessment Framework

### Multi-Dimensional Quality Scoring

```python
def calculate_quality_score(synthetic_sql, metadata):
    """
    Comprehensive quality assessment (0.0 - 1.0)
    """
    
    scores = {
        # 1. Syntactic Validity (30%)
        'syntactic_validity': check_sql_syntax(synthetic_sql),
        
        # 2. Schema Compliance (30%)
        'schema_compliance': check_schema_validity(synthetic_sql, CIM_SCHEMAS),
        
        # 3. Semantic Coherence (20%)
        'semantic_coherence': check_semantic_logic(synthetic_sql),
        
        # 4. Complexity Appropriateness (10%)
        'complexity_match': check_complexity_consistency(synthetic_sql, metadata),
        
        # 5. Diversity Score (10%)
        'diversity': calculate_diversity_score(synthetic_sql, existing_samples)
    }
    
    # Weighted average
    weights = {
        'syntactic_validity': 0.30,
        'schema_compliance': 0.30,
        'semantic_coherence': 0.20,
        'complexity_match': 0.10,
        'diversity': 0.10
    }
    
    quality_score = sum(scores[k] * weights[k] for k in scores)
    
    return quality_score, scores
```

### Quality Thresholds

| Metric | Threshold | Justification |
|--------|-----------|---------------|
| **Overall Quality** | ≥ 0.70 | High quality without being too restrictive |
| **Syntactic Validity** | ≥ 0.90 | SQL must be parseable |
| **Schema Compliance** | ≥ 0.80 | Must use valid tables/columns |
| **Semantic Coherence** | ≥ 0.60 | Logical query structure |
| **Diversity** | ≥ 0.50 | Not too similar to existing samples |

---

## 📈 Expected Output Metrics

### Stage 2 Target Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Input Samples** | 10,000 | Stage 1 output |
| **Raw Generated** | 75,000 | 1.5× target for filtering |
| **After Quality Filter** | 50,000 | Target output |
| **Quality Score (avg)** | ≥ 0.75 | Mean quality score |
| **Schema Compliance** | ≥ 85% | Valid tables/columns |
| **Syntactic Validity** | ≥ 95% | Parseable SQL |
| **Novel Patterns** | ≥ 40% | Not in Stage 1 |
| **Training Time** | 2-4 hours | One-time cost |
| **Generation Time** | 15-30 min | For 75K samples |

---

## 🚀 Implementation Timeline

### Week 1: Setup & Feature Engineering
- [ ] Day 1-2: Install SDV, verify CTGAN works
- [ ] Day 3-4: Extract features from Stage 1 dataset
- [ ] Day 5: Implement schema constraint rules

### Week 2: CTGAN Training
- [ ] Day 6-7: Train CTGAN on Stage 1 features
- [ ] Day 8: Validate CTGAN outputs
- [ ] Day 9-10: Fine-tune hyperparameters if needed

### Week 3: SQL Assembly & Quality Control
- [ ] Day 11-12: Implement SQL assembly from structures
- [ ] Day 13-14: Implement quality scoring
- [ ] Day 15: Generate and filter 50K samples

### Week 4: Validation & Integration
- [ ] Day 16-17: Validate synthetic samples
- [ ] Day 18-19: Generate comprehensive statistics
- [ ] Day 20: Prepare for Stage 3

---

## 🔬 Alternative: Lightweight Approach (If CTGAN Too Slow)

### Plan B: GaussianCopulaSynthesizer + Heavy Rule-Based

If CTGAN training is prohibitively slow:

```python
# Faster alternative
synthesizer = GaussianCopulaSynthesizer(
    metadata=metadata,
    default_distribution='norm'  # or 'beta', 'gamma'
)

# Training: ~5-10 minutes (vs 2-4 hours for CTGAN)
synthesizer.fit(df)

# Generation: ~2-3 minutes for 75K samples
synthetic_structures = synthesizer.sample(num_rows=75000)

# Compensate with stronger rule-based constraints
synthetic_sql = apply_strong_rules(synthetic_structures, CIM_SCHEMAS)
```

**Trade-off:**
- ✅ Much faster (10-15 minutes total vs 2-4 hours)
- ❌ Lower quality (may need more aggressive filtering)
- ❌ Less diversity in patterns

---

## 📝 Summary & Recommendations

### Final Recommendation: **CTGAN-based Hybrid**

**Justification:**
1. **Quality First:** CTGAN produces highest quality synthetic SQL structures
2. **One-Time Cost:** 2-4 hour training is acceptable for one-time dataset generation
3. **Proven Approach:** GAN-based methods successful in code generation literature
4. **Scalability:** Once trained, can generate unlimited samples quickly

**Fallback Plan:**
- If GPU unavailable or training too slow → Use GaussianCopula
- If quality issues → Strengthen rule-based constraints
- If diversity issues → Add noise to CTGAN outputs

### Success Criteria

Stage 2 will be considered successful if:
- ✅ Generate 50,000 high-quality samples
- ✅ Average quality score ≥ 0.75
- ✅ ≥ 85% schema compliance
- ✅ ≥ 40% novel patterns (not in Stage 1)
- ✅ Ready for Stage 3 NL augmentation

---

## 📚 References

1. **SDV Documentation:** https://docs.sdv.dev/
2. **CTGAN Paper:** Xu et al. (2019) "Modeling Tabular Data using Conditional GAN"
3. **Text-to-SQL Synthesis:** Various papers on using GANs for code generation
4. **CIM Wizard Schema:** See `database_schemas/CIM_WIZARD_DATABASE_METADATA.md`

