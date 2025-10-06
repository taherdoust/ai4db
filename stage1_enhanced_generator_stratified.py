#!/usr/bin/env python3
"""
stage1_enhanced_generator.py - WITH STRATIFIED SAMPLING
Enhanced Stage 1: Rule-Based Generation with Comprehensive Metadata
Supports full pipeline with SDV Stage 2 and NL Augmentation Stage 3
"""

import json
import random
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
from cim_wizard_sql_generator import (
    generate_comprehensive_cim_dataset,
    CIM_SCHEMAS,
    generate_realistic_values
)
from rule_based_ssql_generator import generate_sql_pairs

# ============================================================================
# TAXONOMY DEFINITIONS (Based on BIRD, Spider, OmniSQL Research)
# ============================================================================

class SpatialSQLTaxonomy:
    """
    Comprehensive taxonomy based on state-of-the-art Text-to-SQL research
    References:
    - BIRD: Execution accuracy benchmark
    - Spider: Cross-domain benchmark  
    - OmniSQL: Universal benchmarking with tone classification
    - SpatialSQL (Gao et al. 2024): Spatial function usage analysis
    """
    
    # SQL Operation Types (adapted from BIRD/Spider for spatial SQL)
    SQL_TYPES = {
        "SIMPLE_SELECT": "Single table selection with optional WHERE",
        "SPATIAL_JOIN": "Join with spatial predicate (ST_Intersects, ST_Within, etc.)",
        "AGGREGATION": "GROUP BY with aggregate functions (COUNT, SUM, AVG)",
        "NESTED_QUERY": "Subquery or CTE (WITH clause)",
        "SPATIAL_MEASUREMENT": "Measurement functions (ST_Area, ST_Distance, ST_Length)",
        "SPATIAL_PROCESSING": "Geometry processing (ST_Buffer, ST_Union, ST_Intersection)",
        "SPATIAL_CLUSTERING": "Spatial clustering (ST_ClusterDBSCAN, ST_ClusterKMeans)",
        "RASTER_VECTOR": "Raster-vector integration (ST_Value, ST_SummaryStats)",
        "MULTI_JOIN": "Multiple table joins (3+ tables)",
        "WINDOW_FUNCTION": "Window functions (ROW_NUMBER, RANK, PARTITION BY)",
        "CROSS_SCHEMA": "Cross-schema queries (multiple database schemas)"
    }
    
    # Question Tone/Style (from OmniSQL paper - Section 3.2)
    # OmniSQL identifies different question formulation styles
    QUESTION_TONES = {
        "DIRECT": "Direct imperative (Show me, Find, Get, List)",
        "INTERROGATIVE": "Question form (What, Which, Where, How many, How much)",
        "DESCRIPTIVE": "Descriptive request (I need, I want to know, Give me)",
        "ANALYTICAL": "Analytical request (Analyze, Calculate, Determine, Evaluate)",
        "COMPARATIVE": "Comparative request (Compare, Find difference between)",
        "AGGREGATE": "Aggregation request (Count, Sum, Average, Total)",
        "CONDITIONAL": "Conditional request (If X then Y, Where X matches Y)",
        "TEMPORAL": "Temporal request (Latest, Recent, Historical, Between dates)",
        "SPATIAL_SPECIFIC": "Spatial-specific language (within, near, intersecting, adjacent)"
    }
    
    # Difficulty Levels (Multi-dimensional from BIRD benchmark)
    DIFFICULTY_DIMENSIONS = {
        "query_complexity": ["EASY", "MEDIUM", "HARD", "EXPERT"],
        "spatial_complexity": ["BASIC", "INTERMEDIATE", "ADVANCED"],
        "schema_complexity": ["SINGLE_TABLE", "SINGLE_SCHEMA", "MULTI_SCHEMA"],
        "function_count": ["1-2", "3-5", "6+"],
        "join_count": ["0", "1-2", "3-5", "6+"]
    }
    
    # Usage Frequency (Empirical from SpatialSQL paper - Gao et al. 2024)
    # Top 5 functions account for 75.2% of all spatial operations
    USAGE_FREQUENCY = {
        "CRITICAL": ["ST_Intersects", "ST_Area", "ST_Distance", "ST_Contains", "ST_Within"],
        "VERY_HIGH": ["ST_Buffer", "ST_MakePoint", "ST_Transform", "ST_X", "ST_Y", "ST_IsValid", "ST_Length"],
        "HIGH": ["ST_Union", "ST_Touches", "ST_Overlaps", "ST_SetSRID", "ST_Centroid", "ST_GeomFromText", "ST_Envelope", "ST_DWithin"],
        "MEDIUM": ["ST_Difference", "ST_Intersection", "ST_Crosses", "ST_Disjoint", "ST_Simplify", "ST_ConvexHull", "ST_NumPoints", "ST_StartPoint", "ST_EndPoint", "ST_MakeValid"],
        "LOW": []  # All other functions
    }


def classify_sql_type(sql: str, metadata: Dict) -> str:
    """
    Classify SQL query type based on structural analysis
    Uses pattern matching on SQL components
    """
    sql_upper = sql.upper()
    
    # Priority-based classification (most specific first)
    
    # Raster operations (highest priority - most specialized)
    if 'ST_VALUE' in sql_upper or 'ST_SUMMARYSTATS' in sql_upper or 'ST_INTERSECTION' in sql_upper and 'RAST' in sql_upper:
        return "RASTER_VECTOR"
    
    # Spatial clustering
    if 'ST_CLUSTERDBSCAN' in sql_upper or 'ST_CLUSTERKMEANS' in sql_upper:
        return "SPATIAL_CLUSTERING"
    
    # Window functions
    if 'ROW_NUMBER' in sql_upper or 'RANK(' in sql_upper or 'PARTITION BY' in sql_upper:
        return "WINDOW_FUNCTION"
    
    # Nested queries (CTEs)
    if 'WITH' in sql_upper and 'AS (' in sql_upper:
        cte_count = sql_upper.count('WITH')
        if cte_count >= 2:
            return "NESTED_QUERY"
    
    # Multiple joins
    join_count = sql_upper.count('JOIN')
    if join_count >= 3:
        return "MULTI_JOIN"
    
    # Spatial joins
    spatial_predicates = ['ST_INTERSECTS', 'ST_WITHIN', 'ST_CONTAINS', 'ST_TOUCHES', 'ST_OVERLAPS', 'ST_DWITHIN']
    if join_count >= 1 and any(pred in sql_upper for pred in spatial_predicates):
        return "SPATIAL_JOIN"
    
    # Aggregation queries
    if 'GROUP BY' in sql_upper:
        return "AGGREGATION"
    
    # Spatial processing
    spatial_processing = ['ST_BUFFER', 'ST_UNION', 'ST_INTERSECTION', 'ST_DIFFERENCE', 'ST_SYMDIFFERENCE', 'ST_CONVEXHULL']
    if any(func in sql_upper for func in spatial_processing):
        return "SPATIAL_PROCESSING"
    
    # Spatial measurement
    spatial_measurement = ['ST_AREA', 'ST_LENGTH', 'ST_DISTANCE', 'ST_PERIMETER', 'ST_3DDISTANCE']
    if any(func in sql_upper for func in spatial_measurement):
        return "SPATIAL_MEASUREMENT"
    
    # Default to simple select
    return "SIMPLE_SELECT"


def classify_question_tone(natural_language: str) -> str:
    """
    Classify question tone based on linguistic patterns
    Inspired by OmniSQL paper's question style taxonomy
    """
    nl_lower = natural_language.lower()
    
    # Check patterns in priority order
    
    # Spatial-specific language (highest priority for spatial SQL)
    spatial_keywords = ['within', 'near', 'intersecting', 'overlapping', 'adjacent', 'touching', 
                       'inside', 'outside', 'contains', 'contained by', 'distance from']
    if any(kw in nl_lower for kw in spatial_keywords):
        return "SPATIAL_SPECIFIC"
    
    # Temporal requests
    temporal_keywords = ['latest', 'recent', 'historical', 'between', 'before', 'after', 
                        'current', 'past', 'previous', 'last', 'first']
    if any(kw in nl_lower for kw in temporal_keywords):
        return "TEMPORAL"
    
    # Comparative requests
    comparative_keywords = ['compare', 'difference between', 'versus', 'vs', 'vs.', 'better than', 
                          'worse than', 'more than', 'less than', 'greater', 'smaller']
    if any(kw in nl_lower for kw in comparative_keywords):
        return "COMPARATIVE"
    
    # Analytical requests
    analytical_keywords = ['analyze', 'examine', 'evaluate', 'assess', 'calculate', 'compute',
                          'determine', 'measure', 'estimate', 'derive']
    if any(kw in nl_lower for kw in analytical_keywords):
        return "ANALYTICAL"
    
    # Aggregation requests
    aggregate_keywords = ['count', 'sum', 'average', 'total', 'maximum', 'minimum', 'mean',
                         'how many', 'how much', 'number of', 'amount of']
    if any(kw in nl_lower for kw in aggregate_keywords):
        return "AGGREGATE"
    
    # Conditional requests
    conditional_keywords = ['if', 'when', 'where', 'that match', 'satisfying', 'meeting']
    if any(kw in nl_lower for kw in conditional_keywords):
        return "CONDITIONAL"
    
    # Interrogative form (question words at start)
    interrogative_starters = ['what', 'which', 'where', 'who', 'whom', 'whose', 'how']
    if any(nl_lower.startswith(starter) for starter in interrogative_starters):
        return "INTERROGATIVE"
    
    # Direct imperative
    direct_keywords = ['show', 'find', 'get', 'list', 'display', 'return', 'give', 
                      'provide', 'retrieve', 'fetch', 'select']
    if any(nl_lower.startswith(kw) for kw in direct_keywords):
        return "DIRECT"
    
    # Descriptive requests
    descriptive_phrases = ['i need', 'i want', 'i would like', 'i am looking for', 
                          'i require', 'we need', 'we want']
    if any(phrase in nl_lower for phrase in descriptive_phrases):
        return "DESCRIPTIVE"
    
    # Default to interrogative
    return "INTERROGATIVE"


def calculate_difficulty_score(sql: str, metadata: Dict) -> Dict[str, str]:
    """
    Multi-dimensional difficulty scoring based on query complexity metrics
    Adapted from BIRD benchmark difficulty classification
    """
    sql_upper = sql.upper()
    
    # Extract structural components
    cte_count = sql_upper.count('WITH')
    join_count = sql_upper.count('JOIN')
    subquery_count = sql.count('(SELECT')
    spatial_func_count = len(metadata.get('spatial_functions', []))
    table_count = len(metadata.get('tables', []))
    
    # Calculate query complexity score (0-10 scale)
    complexity_score = 0
    
    # CTEs contribute to complexity
    if cte_count >= 3:
        complexity_score += 3
    elif cte_count >= 2:
        complexity_score += 2
    elif cte_count == 1:
        complexity_score += 1
    
    # Joins add complexity
    if join_count >= 5:
        complexity_score += 3
    elif join_count >= 3:
        complexity_score += 2
    elif join_count >= 1:
        complexity_score += 1
    
    # Subqueries increase complexity
    if subquery_count >= 3:
        complexity_score += 2
    elif subquery_count >= 1:
        complexity_score += 1
    
    # Window functions are advanced
    if 'WINDOW' in sql_upper or 'PARTITION BY' in sql_upper or 'ROW_NUMBER' in sql_upper:
        complexity_score += 2
    
    # Set operations
    if 'UNION' in sql_upper or 'INTERSECT' in sql_upper or 'EXCEPT' in sql_upper:
        complexity_score += 1
    
    # Map score to difficulty level
    if complexity_score >= 7:
        query_complexity = "EXPERT"
    elif complexity_score >= 5:
        query_complexity = "HARD"
    elif complexity_score >= 3:
        query_complexity = "MEDIUM"
    else:
        query_complexity = "EASY"
    
    # Spatial complexity assessment
    advanced_spatial = ['ST_CLUSTER', 'ST_SUMMARYSTATS', 'ST_VALUE', 'ST_INTERSECTION', 
                       'ST_DIFFERENCE', 'ST_UNION', 'ST_CONVEXHULL']
    intermediate_spatial = ['ST_BUFFER', 'ST_TRANSFORM', 'ST_CENTROID', 'ST_ENVELOPE',
                          'ST_MAKEVALID', 'ST_SIMPLIFY']
    
    if any(func in sql_upper for func in advanced_spatial) or spatial_func_count >= 5:
        spatial_complexity = "ADVANCED"
    elif any(func in sql_upper for func in intermediate_spatial) or spatial_func_count >= 3:
        spatial_complexity = "INTERMEDIATE"
    else:
        spatial_complexity = "BASIC"
    
    # Schema complexity
    schema_count = len(set(table.split('.')[0] for table in metadata.get('tables', []) if '.' in table))
    if schema_count >= 2:
        schema_complexity = "MULTI_SCHEMA"
    elif table_count >= 2:
        schema_complexity = "SINGLE_SCHEMA"
    else:
        schema_complexity = "SINGLE_TABLE"
    
    # Function count category
    if spatial_func_count >= 6:
        function_count = "6+"
    elif spatial_func_count >= 3:
        function_count = "3-5"
    else:
        function_count = "1-2"
    
    # Join count category
    if join_count >= 6:
        join_count_cat = "6+"
    elif join_count >= 3:
        join_count_cat = "3-5"
    elif join_count >= 1:
        join_count_cat = "1-2"
    else:
        join_count_cat = "0"
    
    return {
        "query_complexity": query_complexity,
        "spatial_complexity": spatial_complexity,
        "schema_complexity": schema_complexity,
        "function_count": function_count,
        "join_count": join_count_cat,
        "overall_difficulty": query_complexity,  # Primary difficulty for sorting
        "complexity_score": complexity_score  # Numeric score for analysis
    }


def extract_spatial_functions(sql: str) -> List[str]:
    """Extract all spatial functions from SQL query"""
    functions = re.findall(r'ST_\w+', sql, re.IGNORECASE)
    # Normalize to uppercase and deduplicate
    return list(set([f.upper() for f in functions]))


def classify_usage_frequency(spatial_functions: List[str]) -> str:
    """
    Classify usage frequency based on empirical data from SpatialSQL paper
    Top 5 functions account for 75.2% of usage
    """
    if not spatial_functions:
        return "NONE"
    
    # Check highest priority functions first
    for func in spatial_functions:
        if func in SpatialSQLTaxonomy.USAGE_FREQUENCY["CRITICAL"]:
            return "CRITICAL"
    
    for func in spatial_functions:
        if func in SpatialSQLTaxonomy.USAGE_FREQUENCY["VERY_HIGH"]:
            return "VERY_HIGH"
    
    for func in spatial_functions:
        if func in SpatialSQLTaxonomy.USAGE_FREQUENCY["HIGH"]:
            return "HIGH"
    
    for func in spatial_functions:
        if func in SpatialSQLTaxonomy.USAGE_FREQUENCY["MEDIUM"]:
            return "MEDIUM"
    
    return "LOW"


def categorize_spatial_functions(spatial_functions: List[str]) -> Dict[str, List[str]]:
    """Categorize spatial functions by operation type"""
    categories = {
        "predicates": [],
        "measurements": [],
        "processing": [],
        "clustering": [],
        "raster": [],
        "transforms": [],
        "accessors": [],
        "constructors": []
    }
    
    for func in spatial_functions:
        func_upper = func.upper()
        
        if func_upper in ['ST_INTERSECTS', 'ST_CONTAINS', 'ST_WITHIN', 'ST_TOUCHES', 'ST_OVERLAPS', 
                         'ST_CROSSES', 'ST_DISJOINT', 'ST_EQUALS', 'ST_COVERS', 'ST_COVEREDBY', 'ST_DWITHIN']:
            categories["predicates"].append(func)
        elif func_upper in ['ST_AREA', 'ST_LENGTH', 'ST_DISTANCE', 'ST_PERIMETER', 'ST_3DDISTANCE']:
            categories["measurements"].append(func)
        elif func_upper in ['ST_BUFFER', 'ST_UNION', 'ST_INTERSECTION', 'ST_DIFFERENCE', 'ST_SYMDIFFERENCE', 
                           'ST_CONVEXHULL', 'ST_ENVELOPE', 'ST_SIMPLIFY']:
            categories["processing"].append(func)
        elif 'CLUSTER' in func_upper:
            categories["clustering"].append(func)
        elif func_upper in ['ST_VALUE', 'ST_SUMMARYSTATS']:
            categories["raster"].append(func)
        elif func_upper in ['ST_TRANSFORM', 'ST_SETSRID', 'ST_FLIPCOORDINATES']:
            categories["transforms"].append(func)
        elif func_upper in ['ST_X', 'ST_Y', 'ST_Z', 'ST_CENTROID', 'ST_STARTPOINT', 'ST_ENDPOINT']:
            categories["accessors"].append(func)
        elif func_upper in ['ST_MAKEPOINT', 'ST_GEOMFROMTEXT', 'ST_COLLECT', 'ST_MAKELINE']:
            categories["constructors"].append(func)
    
    return categories


def create_comprehensive_sample(
    sample_id: str,
    sql_pair,
    values: Dict,
    database_id: int = 1,
    include_results: bool = False
) -> Dict:
    """
    Create comprehensive training sample with all required metadata
    Follows the enhanced schema design for SDV Stage 2 and NL Augmentation Stage 3
    """
    
    # Extract spatial functions
    spatial_functions = extract_spatial_functions(sql_pair.postgis_sql)
    
    # Get table and schema information
    tables = list(sql_pair.evidence.get('tables', []))
    schemas = list(sql_pair.evidence.get('schemas', []))
    columns = list(sql_pair.evidence.get('columns', []))
    
    # Create metadata for classification
    metadata = {
        'spatial_functions': spatial_functions,
        'tables': tables,
        'schemas': schemas,
        'columns': columns
    }
    
    # Perform classifications
    sql_type = classify_sql_type(sql_pair.postgis_sql, metadata)
    question_tone = classify_question_tone(sql_pair.natural_language_desc)
    difficulty = calculate_difficulty_score(sql_pair.postgis_sql, metadata)
    usage_frequency = classify_usage_frequency(spatial_functions)
    function_categories = categorize_spatial_functions(spatial_functions)
    
    # Identify geometry columns
    geometry_columns = []
    for table in tables:
        if 'geometry' in table.lower() or 'geom' in table.lower():
            geometry_columns.append(table)
    for col in columns:
        if 'geometry' in col.lower() or 'geom' in col.lower():
            geometry_columns.append(col)
    
    # Create comprehensive sample following the specified schema
    comprehensive_sample = {
        # === Core Identifiers ===
        "id": sample_id,
        "database_id": database_id,
        "database_name": "cim_wizard",
        
        # === Natural Language Question ===
        "question": sql_pair.natural_language_desc,
        "question_tone": question_tone,
        
        # === SQL Queries (Dual Dialect Support) ===
        "sql_postgis": sql_pair.postgis_sql,
        "sql_spatialite": sql_pair.spatialite_sql,
        
        # === SQL Classification & Taxonomy ===
        "sql_type": sql_type,
        "sql_taxonomy": {
            "operation_type": sql_type,
            "has_cte": "WITH" in sql_pair.postgis_sql.upper(),
            "has_subquery": "(SELECT" in sql_pair.postgis_sql,
            "has_aggregation": "GROUP BY" in sql_pair.postgis_sql.upper(),
            "has_window_function": "PARTITION BY" in sql_pair.postgis_sql.upper(),
            "join_type": "spatial" if sql_type == "SPATIAL_JOIN" else "standard" if "JOIN" in sql_pair.postgis_sql.upper() else "none"
        },
        
        # === Difficulty Levels (Multi-dimensional) ===
        "difficulty": difficulty,
        "difficulty_level": difficulty['overall_difficulty'],  # For quick filtering
        
        # === Usage Frequency Classification ===
        "usage_frequency": usage_frequency,
        "usage_frequency_class": usage_frequency,  # Alias for compatibility
        
        # === Database Schema Information ===
        "database_schema": {
            "schemas": schemas,
            "tables": tables,
            "columns": columns,
            "geometry_columns": geometry_columns,
            "primary_schema": schemas[0] if schemas else None,
            "table_count": len(tables),
            "schema_count": len(set(schemas))
        },
        
        # === Spatial Functions ===
        "spatial_functions": spatial_functions,
        "spatial_function_count": len(spatial_functions),
        "spatial_function_categories": function_categories,
        
        # === Evidence (Original metadata from generator) ===
        "evidence": sql_pair.evidence,
        
        # === Instruction for LLM Training ===
        "instruction": f"Convert this natural language question to PostGIS spatial SQL for the CIM Wizard database: {sql_pair.natural_language_desc}",
        
        # === Results for Execution Accuracy (EX) Evaluation ===
        "results": None if include_results else [],  # None = to be filled; [] = not evaluation sample
        "has_results": include_results,
        
        # === Pipeline Metadata ===
        "stage": "stage1_enhanced",
        "generation_method": "rule_based_template",
        "template_id": sql_pair.template_id,
        "complexity_level": sql_pair.complexity,  # A, B, or C
        "tags": list(sql_pair.tags),
        "generation_params": values,
        "generated_at": datetime.now().isoformat()
    }
    
    return comprehensive_sample


# ============================================================================
# STRATIFIED SAMPLING FOR EVALUATION SET
# ============================================================================

def stratified_evaluation_sampling(
    enhanced_samples: List[Dict],
    evaluation_sample_size: int = 100,
    random_seed: int = 42
) -> List[int]:
    """
    Perform stratified sampling to ensure evaluation set is representative
    
    Stratification Dimensions:
    1. SQL Type (11 types)
    2. Difficulty Level (EASY, MEDIUM, HARD, EXPERT)
    3. Usage Frequency (CRITICAL, VERY_HIGH, HIGH, MEDIUM, LOW)
    4. Complexity Level (A, B, C)
    
    Args:
        enhanced_samples: All generated samples with metadata
        evaluation_sample_size: Target number of evaluation samples
        random_seed: Random seed for reproducibility
    
    Returns:
        List of indices to use for evaluation set
    """
    
    random.seed(random_seed)
    
    print(f"\n[STRATIFIED SAMPLING] Creating representative evaluation set")
    print(f"Target size: {evaluation_sample_size} samples")
    
    # Group samples by stratification key
    strata = defaultdict(list)
    
    for idx, sample in enumerate(enhanced_samples):
        # Create stratification key: (sql_type, difficulty, usage_freq, complexity_level)
        key = (
            sample['sql_type'],
            sample['difficulty']['overall_difficulty'],
            sample['usage_frequency'],
            sample['complexity_level']
        )
        strata[key].append(idx)
    
    print(f"  Found {len(strata)} unique strata combinations")
    
    # Calculate proportional allocation
    total_samples = len(enhanced_samples)
    selected_indices = []
    
    # First pass: Allocate proportionally
    allocation = {}
    for stratum_key, stratum_indices in strata.items():
        proportion = len(stratum_indices) / total_samples
        allocated_count = max(1, int(proportion * evaluation_sample_size))  # At least 1 sample per stratum
        allocation[stratum_key] = allocated_count
    
    # Adjust if over-allocated
    total_allocated = sum(allocation.values())
    if total_allocated > evaluation_sample_size:
        # Reduce largest strata first
        sorted_strata = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
        excess = total_allocated - evaluation_sample_size
        for stratum_key, count in sorted_strata:
            if excess == 0:
                break
            reduction = min(count - 1, excess)  # Keep at least 1
            allocation[stratum_key] -= reduction
            excess -= reduction
    
    # Under-allocated: add to largest strata
    total_allocated = sum(allocation.values())
    if total_allocated < evaluation_sample_size:
        sorted_strata = sorted(allocation.items(), key=lambda x: len(strata[x[0]]), reverse=True)
        deficit = evaluation_sample_size - total_allocated
        for stratum_key, count in sorted_strata:
            if deficit == 0:
                break
            available = len(strata[stratum_key]) - count
            addition = min(available, deficit)
            allocation[stratum_key] += addition
            deficit -= addition
    
    # Sample from each stratum
    print(f"\n  Stratification Summary:")
    print(f"  {'Stratum (SQL_Type, Difficulty, Freq, Level)':<50} {'Total':<8} {'Selected':<10}")
    print(f"  {'-'*70}")
    
    for stratum_key, stratum_indices in sorted(strata.items()):
        allocated_count = allocation[stratum_key]
        # Randomly sample from this stratum
        sampled = random.sample(stratum_indices, min(allocated_count, len(stratum_indices)))
        selected_indices.extend(sampled)
        
        sql_type, difficulty, freq, complexity = stratum_key
        print(f"  {sql_type:<20} {difficulty:<8} {freq:<12} {complexity:<3} {len(stratum_indices):<8} {len(sampled):<10}")
    
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':<53} {total_samples:<8} {len(selected_indices):<10}")
    
    return selected_indices


def generate_stage1_enhanced_dataset(
    num_variations: int = 200,
    output_file: str = "training_datasets/stage1_enhanced_dataset.jsonl",
    evaluation_sample_size: int = 100,
    random_seed: int = 42,
    use_stratified_sampling: bool = True
):
    """
    Generate Stage 1 enhanced dataset with comprehensive metadata
    
    Args:
        num_variations: Number of parameter variations per template
        output_file: Output JSONL file path
        evaluation_sample_size: Number of samples reserved for evaluation (with results)
        random_seed: Random seed for reproducibility
        use_stratified_sampling: Use stratified sampling (recommended) vs random sampling
    
    Returns:
        Tuple of (enhanced_samples, statistics)
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    print("="*80)
    print("STAGE 1: Enhanced CIM Wizard Dataset Generation")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Variations per template: {num_variations}")
    print(f"  - Evaluation samples: {evaluation_sample_size}")
    print(f"  - Random seed: {random_seed}")
    print(f"  - Sampling method: {'STRATIFIED' if use_stratified_sampling else 'RANDOM'}")
    print(f"  - Output file: {output_file}")
    
    # Generate base dataset using existing generators
    print("\n[1/5] Generating base SQL pairs from templates...")
    dataset = generate_comprehensive_cim_dataset(base_variations=num_variations)
    print(f"      ✓ Generated {len(dataset)} base samples")
    
    # Create enhanced samples with comprehensive metadata FIRST
    print(f"\n[2/5] Creating enhanced samples with comprehensive metadata...")
    enhanced_samples = []
    
    for i, pair in enumerate(dataset):
        # Generate realistic parameter values
        values = generate_realistic_values()
        
        # Create unique sample ID
        sample_id = f"cim_stage1_{i:06d}"
        
        # Create comprehensive sample (WITHOUT evaluation flag yet)
        enhanced_sample = create_comprehensive_sample(
            sample_id=sample_id,
            sql_pair=pair,
            values=values,
            database_id=1,
            include_results=False  # Will be set later
        )
        
        enhanced_samples.append(enhanced_sample)
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"      Progress: {i + 1}/{len(dataset)} samples processed...")
    
    print(f"      ✓ Created {len(enhanced_samples)} enhanced samples")
    
    # Select evaluation samples using stratified or random sampling
    print(f"\n[3/5] Selecting evaluation samples...")
    total_samples = len(enhanced_samples)
    eval_size = min(evaluation_sample_size, total_samples)
    
    if use_stratified_sampling:
        eval_indices = set(stratified_evaluation_sampling(
            enhanced_samples, 
            eval_size, 
            random_seed
        ))
    else:
        print(f"  Using RANDOM sampling (not recommended)")
        eval_indices = set(random.sample(range(total_samples), eval_size))
    
    print(f"      ✓ Selected {len(eval_indices)} samples for evaluation")
    
    # Update evaluation flags
    for idx in eval_indices:
        enhanced_samples[idx]['has_results'] = True
        enhanced_samples[idx]['results'] = None  # To be filled with actual results
    
    # Save main dataset
    print(f"\n[4/5] Saving datasets...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in enhanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      ✓ Main dataset: {output_file}")
    
    # Save evaluation subset separately
    eval_samples = [s for s in enhanced_samples if s['has_results']]
    eval_file = output_file.replace('.jsonl', '_eval.jsonl')
    with open(eval_file, 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      ✓ Evaluation subset: {eval_file} ({len(eval_samples)} samples)")
    
    # Generate comprehensive statistics
    print(f"\n[5/5] Generating statistics...")
    stats = generate_comprehensive_statistics(enhanced_samples)
    
    # Save statistics
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"      ✓ Statistics: {stats_file}")
    
    # Print summary
    print_summary_statistics(stats)
    
    return enhanced_samples, stats


def generate_comprehensive_statistics(samples: List[Dict]) -> Dict:
    """Generate comprehensive statistics for the dataset"""
    
    stats = {
        "dataset_info": {
            "total_samples": len(samples),
            "evaluation_samples": len([s for s in samples if s['has_results']]),
            "training_samples": len([s for s in samples if not s['has_results']]),
            "generation_date": datetime.now().isoformat()
        },
        
        "sql_types": {},
        "question_tones": {},
        "difficulty_levels": {},
        "usage_frequency": {},
        "spatial_functions": {},
        "function_categories": {},
        "schema_complexity": {},
        "template_distribution": {},
        "complexity_levels": {}  # A, B, C
    }
    
    # Collect statistics
    for sample in samples:
        # SQL types
        sql_type = sample['sql_type']
        stats['sql_types'][sql_type] = stats['sql_types'].get(sql_type, 0) + 1
        
        # Question tones
        tone = sample['question_tone']
        stats['question_tones'][tone] = stats['question_tones'].get(tone, 0) + 1
        
        # Difficulty levels
        difficulty = sample['difficulty']['overall_difficulty']
        stats['difficulty_levels'][difficulty] = stats['difficulty_levels'].get(difficulty, 0) + 1
        
        # Usage frequency
        freq = sample['usage_frequency']
        stats['usage_frequency'][freq] = stats['usage_frequency'].get(freq, 0) + 1
        
        # Spatial functions
        for func in sample['spatial_functions']:
            stats['spatial_functions'][func] = stats['spatial_functions'].get(func, 0) + 1
        
        # Function categories
        for cat, funcs in sample['spatial_function_categories'].items():
            if funcs:
                stats['function_categories'][cat] = stats['function_categories'].get(cat, 0) + len(funcs)
        
        # Schema complexity
        schema_complexity = sample['difficulty']['schema_complexity']
        stats['schema_complexity'][schema_complexity] = stats['schema_complexity'].get(schema_complexity, 0) + 1
        
        # Template distribution
        template_id = sample['template_id']
        stats['template_distribution'][template_id] = stats['template_distribution'].get(template_id, 0) + 1
        
        # Complexity levels (A, B, C)
        complexity = sample['complexity_level']
        stats['complexity_levels'][complexity] = stats['complexity_levels'].get(complexity, 0) + 1
    
    # Sort by frequency
    for key in ['sql_types', 'question_tones', 'difficulty_levels', 'usage_frequency', 
                'spatial_functions', 'function_categories', 'schema_complexity']:
        stats[key] = dict(sorted(stats[key].items(), key=lambda x: -x[1]))
    
    return stats


def print_summary_statistics(stats: Dict):
    """Print formatted summary statistics"""
    
    print("\n" + "="*80)
    print("STAGE 1 ENHANCED DATASET - SUMMARY STATISTICS")
    print("="*80)
    
    info = stats['dataset_info']
    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {info['total_samples']:,}")
    print(f"   Training samples: {info['training_samples']:,}")
    print(f"   Evaluation samples: {info['evaluation_samples']:,}")
    
    print(f"\n🔧 SQL Type Distribution (Top 5):")
    for i, (sql_type, count) in enumerate(list(stats['sql_types'].items())[:5], 1):
        percentage = (count / info['total_samples']) * 100
        print(f"   {i}. {sql_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n💬 Question Tone Distribution (Top 5):")
    for i, (tone, count) in enumerate(list(stats['question_tones'].items())[:5], 1):
        percentage = (count / info['total_samples']) * 100
        print(f"   {i}. {tone}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n⚡ Difficulty Distribution:")
    for difficulty in ['EASY', 'MEDIUM', 'HARD', 'EXPERT']:
        count = stats['difficulty_levels'].get(difficulty, 0)
        percentage = (count / info['total_samples']) * 100 if count > 0 else 0
        print(f"   {difficulty}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📈 Usage Frequency Distribution:")
    for freq in ['CRITICAL', 'VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW']:
        count = stats['usage_frequency'].get(freq, 0)
        percentage = (count / info['total_samples']) * 100 if count > 0 else 0
        print(f"   {freq}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🌍 Spatial Functions (Top 10):")
    for i, (func, count) in enumerate(list(stats['spatial_functions'].items())[:10], 1):
        print(f"   {i}. {func}: {count:,}")
    
    print(f"\n🗄️  Schema Complexity:")
    for complexity, count in stats['schema_complexity'].items():
        percentage = (count / info['total_samples']) * 100
        print(f"   {complexity}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n✅ Dataset Generation Complete!")
    print(f"   Ready for Stage 2 (SDV Synthetic Generation)")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_variations = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    eval_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    use_stratified = sys.argv[3].lower() != 'false' if len(sys.argv) > 3 else True
    
    # Generate enhanced Stage 1 dataset
    samples, stats = generate_stage1_enhanced_dataset(
        num_variations=num_variations,
        output_file="training_datasets/stage1_enhanced_dataset.jsonl",
        evaluation_sample_size=eval_size,
        random_seed=42,
        use_stratified_sampling=use_stratified
    )
    
    print(f"\n🎉 Stage 1 Enhanced Dataset Successfully Created!")
    print(f"   Total samples: {len(samples):,}")
    print(f"   Output: training_datasets/stage1_enhanced_dataset.jsonl")
    print(f"\n▶️  Next step: Run Stage 2 (SDV Synthetic Generation)")

