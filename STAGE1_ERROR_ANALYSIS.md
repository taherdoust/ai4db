# Stage 1 Template Error Analysis

## Executive Summary

**Current Performance:** 45.75% NoErr rate (1,853/4,050 passed)
**Expected Performance:** ~100% NoErr rate (rule-based templates)
**Main Issue:** 95.6% of errors are SCHEMA_ERROR (2,100/2,197)

## Root Cause Analysis

### 1. Column Name Mismatches

The templates were written based on outdated or assumed schema definitions. The actual database schema (from `vector-raster-census-network.sql`) differs significantly:

**cim_network.network_buses:**
- Template uses: `name`
- Actual column: `bus_name`

**cim_network.network_lines:**
- Template uses: `name`, `from_bus`, `to_bus`
- Actual columns: `line_name`, `from_bus_id`, `to_bus_id`

**cim_vector.cim_wizard_building_properties:**
- Template uses: `gross_floor_area`, `heating`, `cooling`, `hvac_type`
- Actual schema: These columns DO NOT EXIST
- Available columns: `scenario_id`, `building_id`, `project_id`, `lod`, `height`, `area`, `volume`, `number_of_floors`, `const_period_census`, `n_family`, `n_people`, `type`, `const_tabula`, `const_year`

### 2. Error Breakdown by SQL Type

| SQL Type | Total | Passed | NoErr Rate | Issue |
|----------|-------|--------|-----------|-------|
| SPATIAL_CLUSTERING | 150 | 150 | 100% | ✓ Perfect |
| RASTER_VECTOR | 600 | 450 | 75% | Moderate schema issues |
| SPATIAL_MEASUREMENT | 450 | 353 | 78.4% | Some schema mismatches |
| AGGREGATION | 300 | 150 | 50% | Schema + join issues |
| MULTI_JOIN | 1,050 | 450 | 42.9% | High failure in network joins |
| SIMPLE_SELECT | 450 | 150 | 33.3% | Basic schema mismatches |
| NESTED_QUERY | 900 | 150 | 16.7% | Complex queries hit multiple schema errors |
| SPATIAL_JOIN | 150 | 0 | 0% | CRITICAL - All failed |

### 3. Why Evaluation Works (Answers User's Question)

The evaluation script processes SQL queries **one at a time** in a loop, not "100 at once":

```python
for idx, sample in enumerate(samples, 1):
    sql_query = sample.get('sql_postgis', '')
    eval_result = evaluate_query(sql_query, engine, timeout=30)
    # Processes each query individually with its own database connection
```

Each query:
1. Opens a new database connection
2. Sets statement timeout (30 seconds)
3. Executes the SQL
4. Fetches results
5. Records success/failure
6. Closes connection

This is the correct approach for validation. The issue is NOT with the evaluation - it's working perfectly and exposing real schema mismatches in our templates.

## Fixes Required

### Priority 1: Fix Schema Mismatches
1. Update all `network_buses.name` → `network_buses.bus_name`
2. Update all `network_lines.name` → `network_lines.line_name`
3. Update all `from_bus`/`to_bus` → `from_bus_id`/`to_bus_id`
4. Remove non-existent columns from building_properties queries

### Priority 2: Reorganize Template Focus
Based on user request:
1. **Priority 1:** Inner cim_vector schema (buildings, properties, projects)
2. **Priority 2:** cim_vector + cim_census/cim_network/cim_raster
3. **Priority 3:** Inner cim_census/cim_raster/cim_network
4. **Priority 4:** Cross-schema without cim_vector

### Priority 3: Validate All Templates
- Test each template against actual database
- Ensure all referenced columns exist
- Verify join conditions are correct
- Check data types match

## Action Plan

1. Fix stage1_cim.py with correct column names
2. Reorganize templates by priority
3. Remove templates with non-existent columns
4. Add validation step in template generation
5. Re-run evaluation to achieve ~100% NoErr rate

## Expected Outcome

After fixes:
- NoErr rate: 98-100% (allowing for rare edge cases)
- SPATIAL_JOIN should reach 100% (currently 0%)
- MULTI_JOIN should reach 95%+ (currently 42.9%)
- All template should reference only existing columns

