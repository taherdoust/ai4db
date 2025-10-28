# Stage 1 Template Schema Fixes - October 28, 2025

## Problem Discovered
NoErr rate dropped from expected ~100% to **71.05%** (2200 errors out of 7600 samples).

### Error Breakdown by SQL Type
- SPATIAL_MEASUREMENT: **100%** ✓
- AGGREGATION: **100%** ✓
- SPATIAL_CLUSTERING: **100%** ✓
- SIMPLE_SELECT: **81.8%** (some issues)
- RASTER_VECTOR: **75.0%** (ROUND function)
- MULTI_JOIN: **71.4%** (mixed issues)
- NESTED_QUERY: **16.7%** (ROUND function)
- SPATIAL_JOIN: **0.0%** (census column case)

## Root Causes Identified

### Issue 1: Census Column Case Sensitivity (28.95% of errors)

**Problem:**
- PostgreSQL column names are case-insensitive when **unquoted**, stored as lowercase
- When you use **quoted identifiers** like `"SEZ2011"`, it becomes case-sensitive
- Templates used: `c."SEZ2011"`, `c."P1"`, `c."P2"`, etc.
- Actual database columns: `sez2011`, `p1`, `p2`, etc. (lowercase)

**Error Example:**
```
column c.SEZ2011 does not exist
HINT: Perhaps you meant to reference the column "c.sez2011".
```

**Fix Applied:**
- Changed ALL census column references from uppercase quoted to lowercase unquoted
- Before: `c."SEZ2011"`, `c."P1"`, `cg.COMUNE`
- After: `c.sez2011`, `c.p1`, `cg.comune`
- Affected columns: SEZ2011, P1-P66, P128-P140, ST1-ST15, E1-E31, A2-A48, PF1-PF9, REGIONE, PROVINCIA, COMUNE

**Affected Templates:**
- CIM_A4_census_population
- CIM_A8_buildings_in_census
- CIM_A9_census_in_project
- CIM_B3_building_census_aggregation
- CIM_B6_census_employment
- CIM_C3_multi_schema_integration
- CIM_C5_demographic_analysis
- CIM_C6_merge_census_zones

### Issue 2: ROUND() Function Type Mismatch

**Problem:**
- PostgreSQL's `ROUND(value, digits)` requires first argument to be `numeric` type
- Templates passed `double precision` or `float` values without casting
- This is a PostgreSQL-specific requirement

**Error Example:**
```
function round(double precision, integer) does not exist
HINT: No function matches the given name and argument types
```

**Fix Applied:**
- Added `::numeric` cast to all ROUND() function calls
- Before: `ROUND(avg_height, 2)`
- After: `ROUND((avg_height)::numeric, 2)`

**Affected Templates:**
- CIM_C4_nested_building_analysis (NESTED_QUERY type)
- CIM_C5_demographic_analysis (census ratios)
- CIM_C7_project_comparison
- CIM_C9_raster_vector_comparison (RASTER_VECTOR type)

## Expected Improvements After Fixes

| SQL Type | Before | After (Expected) |
|----------|--------|------------------|
| SPATIAL_JOIN | 0.0% | ~98-100% |
| NESTED_QUERY | 16.7% | ~98-100% |
| RASTER_VECTOR | 75.0% | ~98-100% |
| MULTI_JOIN | 71.4% | ~95-98% |
| SIMPLE_SELECT | 81.8% | ~98-100% |
| **Overall** | **71.05%** | **~98-100%** |

## Action Required

**Regenerate Stage 1 dataset:**
```bash
cd /home/ali/Desktop/HDD_Volume/000products/coesi/ai4db
conda activate aienv
python stage1_cim.py 200 100
```

**Re-evaluate quality:**
```bash
python evaluate_generation_quality.py \
  --input training_datasets/stage1_cim_dataset.jsonl \
  --output training_datasets/stage1_quality_report_v2.json \
  --stage 1
```

## Why These Errors Were Missed Initially

1. **Case sensitivity**: PostgreSQL unquoted identifiers are folded to lowercase, but the `populate_censusgeo.sql` script used unquoted `SEZ2011` in CREATE TABLE (stored as lowercase), while templates used quoted `"SEZ2011"` (case-sensitive lookup)

2. **ROUND() function**: This is a PostgreSQL-specific behavior where ROUND() requires numeric type. Other databases (MySQL, SQLite) are more lenient.

3. **Template development**: Templates were likely developed/tested on a different PostgreSQL instance with different settings or using a database tool that auto-casts types.

## PostGIS Extension Verification

The user confirmed PostGIS extensions are installed on the `public` schema. All spatial functions in templates correctly use the `public.ST_*` prefix format, so no changes were needed there.

Example (correct):
```sql
public.ST_Within(public.ST_Centroid(b.building_geometry), c.geometry)
public.ST_Intersects(c.geometry, ps.project_boundary)
public.ST_Area(b.building_geometry)
```

## Files Modified

- `ai4db/stage1_cim.py`:
  - Fixed census column case (Python regex replacement)
  - Fixed ROUND() function calls (Python regex replacement)
  - 41 total fixes applied

## Next Steps

1. **Regenerate Stage 1 dataset** with fixes
2. **Re-evaluate quality** (expect ~98-100% NoErr)
3. **Update Stage 2** (use fixed Stage 1 as input)
4. **Update Stage 3** (use fixed Stage 2 as input)
5. **Re-curate** training data
6. **Re-train** models with corrected data

