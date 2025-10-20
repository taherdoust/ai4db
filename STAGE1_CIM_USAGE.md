# Stage 1 CIM Wizard Dataset Generator - Usage Guide

## Overview

`stage1_cim.py` is a consolidated spatial SQL generator focused exclusively on the **CIM Wizard database**. It integrates and replaces the previous three-file approach with a unified, streamlined implementation.

## Key Features

### 1. Spatial Function Classification

Functions are classified by **four dimensions**:

#### Data Type
- **vector_only**: Functions that work only with vector geometries (ST_Intersects, ST_Area, ST_Distance, etc.)
- **raster_only**: Functions that work only with raster data (ST_Value, ST_SummaryStats)
- **vector_raster**: Functions that work with both (ST_Clip, ST_Intersection with raster)

#### Usage Frequency (Consolidated)
- **most_frequent**: Critical + Very High usage (ST_Intersects, ST_Area, ST_Distance, ST_Contains, ST_Within, ST_Buffer, ST_Transform, ST_IsValid, ST_MakePoint, ST_Intersection, ST_Length, ST_Touches)
- **frequent**: High + Medium usage (ST_DWithin, ST_Overlaps, ST_Union, ST_X, ST_Y, ST_Centroid, etc.)
- **low_frequent**: Low usage (ST_ClusterDBSCAN, ST_ClusterKMeans, ST_Collect, ST_Perimeter)

#### Difficulty Level
- **basic**: Simple operations (predicates, basic measurements, accessors)
- **intermediate**: Complex operations (processing, transforms, buffers)
- **advanced**: Expert operations (clustering, raster analysis)

#### Category
- predicate: Spatial relationship tests
- measurement: Area, distance, length calculations
- processing: Buffer, union, intersection
- accessor: Coordinate extraction, centroid
- constructor: Creating geometries
- transform: CRS transformations
- validation: Geometry validation
- clustering: Spatial clustering
- raster_accessor: Reading raster values
- raster_analysis: Analyzing raster statistics
- raster_processing: Clipping and processing rasters

### 2. Simplified Difficulty Dimensions

**No more EXPERT level - maximum is HARD**

#### Query Complexity: EASY, MEDIUM, HARD
- Based on CTEs, joins, subqueries, window functions

#### Spatial Complexity: BASIC, INTERMEDIATE, ADVANCED
- Based on function sophistication and count

#### Schema Complexity: SINGLE_TABLE, SINGLE_SCHEMA, MULTI_SCHEMA
- Based on number of tables and schemas involved

#### Function Count: 1, 2, 3+
- Simplified from previous 6+ categories

#### Join Count: 0, 1, 2+
- Simplified from previous 6+ categories

### 3. Complexity Level (A, B, C)

**Automatically derived** from difficulty dimensions:

- **Level A**: Easy queries + Basic spatial + Single table/schema
  - Simple selects with filters
  - Basic measurements and predicates
  - Single table operations
  
- **Level B**: Medium queries + Intermediate spatial + Joins
  - Aggregations and grouping
  - Spatial joins
  - CTEs with single level
  - Buffer and processing operations
  
- **Level C**: Hard queries + Advanced spatial + Multi-schema
  - Multi-schema integration
  - Spatial clustering
  - Raster-vector operations
  - Complex CTEs with multiple levels
  - Window functions

### 4. Question Tone Classification (9 Types)

1. **DIRECT**: "Show me", "Find", "Get", "List"
2. **INTERROGATIVE**: "What", "Which", "Where", "How many"
3. **DESCRIPTIVE**: "I need", "I want to know"
4. **ANALYTICAL**: "Analyze", "Calculate", "Determine"
5. **COMPARATIVE**: "Compare", "Find difference"
6. **AGGREGATE**: "Count", "Sum", "Average"
7. **CONDITIONAL**: "If X then Y", "Where X matches Y"
8. **TEMPORAL**: "Latest", "Recent", "Between dates"
9. **SPATIAL_SPECIFIC**: "within", "near", "intersecting"

## CIM Wizard Database Schemas

The generator focuses on **4 main schemas**:

### 1. cim_vector (Most Important)
- **building**: Geometry and metadata
- **building_properties**: Physical properties (height, area, type, HVAC, etc.)
- **project_scenario**: Project boundaries and metadata
- **grid_bus**: Electrical grid bus stations
- **grid_line**: Electrical grid lines

### 2. cim_census
- **census_geo**: Italian census data (demographics, employment, education, housing)

### 3. cim_raster
- **dtm_raster**: Digital Terrain Model (ground elevation)
- **dsm_raster**: Digital Surface Model (surface elevation)
- **building_height_cache**: Pre-computed building heights from raster analysis

### 4. cim_network
- Referenced in grid connectivity analysis

## Template Pool (16 Templates)

### Level A - Basic Operations (6 templates)
1. **CIM_A1_building_by_type**: Filter buildings by type and area
2. **CIM_A2_project_at_location**: Find projects at point location
3. **CIM_A3_grid_buses_by_voltage**: Filter grid buses by voltage
4. **CIM_A4_census_population**: Query census population data
5. **CIM_A5_building_height_cache**: Retrieve cached building heights
6. **CIM_A6_building_distance**: Calculate distance from point to buildings

### Level B - Intermediate Operations (5 templates)
1. **CIM_B1_building_stats_by_type**: Aggregate statistics by building type
2. **CIM_B2_buildings_near_grid**: Spatial join buildings to grid infrastructure
3. **CIM_B3_building_census_aggregation**: Aggregate buildings by census boundaries
4. **CIM_B4_grid_line_connectivity**: Analyze grid network connectivity
5. **CIM_B5_building_buffer_analysis**: Create and analyze building buffers
6. **CIM_B6_census_employment**: Census employment analysis

### Level C - Advanced Operations (5 templates)
1. **CIM_C1_building_height_validation**: Validate heights using raster data
2. **CIM_C2_building_clustering**: DBSCAN spatial clustering
3. **CIM_C3_multi_schema_integration**: Integrate vector, census, and grid data
4. **CIM_C4_raster_value_extraction**: Extract raster values at building centroids
5. **CIM_C5_census_demographic_transition**: Complex demographic analysis

## Usage

### Basic Usage

```bash
# Generate dataset with default settings (200 variations, 100 eval samples)
python3 stage1_cim.py

# Custom variations and evaluation size
python3 stage1_cim.py 500 150

# Arguments: num_variations eval_size
```

### Expected Output

```
training_datasets/
  ├── stage1_cim_dataset.jsonl          # Main training dataset
  ├── stage1_cim_dataset_eval.jsonl     # Evaluation subset (stratified)
  └── stage1_cim_dataset_stats.json     # Comprehensive statistics
```

### Output Statistics

The generator produces comprehensive statistics including:
- SQL type distribution
- Complexity level distribution (A, B, C)
- Difficulty distribution (EASY, MEDIUM, HARD)
- Usage frequency distribution (most_frequent, frequent, low_frequent)
- Function data types (vector_only, raster_only, vector_raster)
- Top spatial functions by frequency
- Schema complexity distribution
- Question tone distribution

## Sample Sizes

### Recommended Configurations

```bash
# Quick test (10 samples per template = ~160 samples)
python3 stage1_cim.py 10 10

# Small dataset (50 variations = ~800 samples)
python3 stage1_cim.py 50 50

# Medium dataset (200 variations = ~3,200 samples) - DEFAULT
python3 stage1_cim.py 200 100

# Large dataset (500 variations = ~8,000 samples)
python3 stage1_cim.py 500 200

# Production dataset (1000 variations = ~16,000 samples)
python3 stage1_cim.py 1000 500
```

## Stratified Evaluation Sampling

Evaluation samples are selected using **stratified sampling** to ensure representative distribution across:

1. SQL Type (10 types)
2. Query Complexity (EASY, MEDIUM, HARD)
3. Usage Frequency (most_frequent, frequent, low_frequent)
4. Complexity Level (A, B, C)

This ensures the evaluation set covers all important dimensions proportionally.

## Output Format

Each sample in the JSONL file contains:

```json
{
  "id": "cim_000001",
  "database_name": "cim_wizard",
  "question": "Find buildings of specific type...",
  "question_tone": "DIRECT",
  "sql_postgis": "SELECT...",
  "sql_spatialite": "SELECT...",
  "sql_type": "SPATIAL_JOIN",
  "difficulty": {
    "query_complexity": "EASY",
    "spatial_complexity": "BASIC",
    "schema_complexity": "SINGLE_SCHEMA",
    "function_count": "2",
    "join_count": "1",
    "overall_difficulty": "EASY",
    "complexity_level": "A"
  },
  "usage_frequency": "most_frequent",
  "spatial_functions": ["ST_Intersects", "ST_Area"],
  "spatial_function_details": [
    {
      "name": "ST_Intersects",
      "data_type": "vector_only",
      "usage_frequency": "most_frequent",
      "difficulty": "basic"
    },
    {
      "name": "ST_Area",
      "data_type": "vector_only",
      "usage_frequency": "most_frequent",
      "difficulty": "basic"
    }
  ],
  "database_schema": {
    "schemas": ["cim_vector"],
    "tables": ["cim_vector.building", "cim_vector.building_properties"],
    "table_count": 2,
    "schema_count": 1
  },
  "has_results": false,
  "stage": "stage1_cim",
  "complexity_level": "A",
  "generated_at": "2025-10-20T..."
}
```

## Integration with Pipeline

This dataset is designed to feed into:

1. **Stage 2**: SDV CTGAN synthetic SQL generation
2. **Stage 3**: NL question augmentation with OpenRouter

The comprehensive metadata enables:
- Fine-grained difficulty balancing in Stage 2
- Context-aware NL generation in Stage 3
- Stratified evaluation for model testing

## Differences from Previous Implementation

### Removed
- ❌ General spatial templates (non-CIM)
- ❌ EXPERT difficulty level
- ❌ Complex function count categories (6+)
- ❌ Complex join count categories (6+)
- ❌ Separate CRITICAL and VERY_HIGH frequency tiers
- ❌ Three separate Python files

### Added
- ✅ Consolidated frequency tiers (most_frequent, frequent, low_frequent)
- ✅ Function data type classification (vector/raster/both)
- ✅ Unified CIM-focused templates
- ✅ Automatic complexity level derivation
- ✅ Single consolidated file
- ✅ Comprehensive function metadata

## Next Steps

After generating Stage 1 dataset:

```bash
# 1. Verify output
ls -lh training_datasets/stage1_cim_dataset*

# 2. Check statistics
cat training_datasets/stage1_cim_dataset_stats.json | jq '.dataset_info'

# 3. Run Stage 2 (SDV synthetic generation)
python3 stage2_sdv_pipeline_eclab_ctgan.py

# 4. Run Stage 3 (NL augmentation)
python3 stage3_augmentation_pipeline_eclab_openrouter_enhanced.py
```

## Troubleshooting

### Issue: Missing parameters in template
**Solution**: Check that all template placeholders match keys in `generate_realistic_values()`

### Issue: Too few/many evaluation samples
**Solution**: Adjust `evaluation_sample_size` parameter or increase `num_variations`

### Issue: Imbalanced complexity distribution
**Solution**: Add more templates at underrepresented complexity levels

## Performance

- **Generation Speed**: ~5,000 samples/minute
- **Memory Usage**: ~50MB per 10,000 samples
- **Disk Space**: ~10MB per 1,000 samples (JSONL compressed)

## Citation

If using this dataset generator in research, please cite:

```
CIM Wizard Spatial SQL Dataset Generator (2025)
Stage 1: Rule-based generation with comprehensive metadata
https://github.com/your-repo/ai4db
```

