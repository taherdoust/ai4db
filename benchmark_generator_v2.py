#!/usr/bin/env python3
"""
Taxonomy-Based Benchmark Generator v2
======================================

Generates evaluation benchmarks with specific task and domain taxonomy distributions.
Uses the taxonomy classification from evaluator_v2.py to ensure accurate categorization.

Features:
- Target-based sampling for task and domain types
- Automatic taxonomy classification
- Ground truth execution results
- Quality filtering
- Taxonomy metadata in output

Usage:
    python benchmark_generator_v2.py \
        --input ../ai4db/training_datasets/downloaded/stage3_augmented_dataset_FINAL_annotated.jsonl \
        --output benchmark_v2.jsonl \
        --size 100

Author: Ali Taherdoust
Date: November 2025
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
from collections import defaultdict, Counter
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import time
from datetime import datetime


# ============================================================================
# TASK TAXONOMY (from evaluator_v2.py)
# ============================================================================

TASK_TAXONOMY = {
    "SIMPLE_SELECT": {"complexity": 1, "frequency": 1, "description": "Simple SELECT with WHERE, no spatial operations"},
    "SQL_AGGREGATION": {"complexity": 1, "frequency": 1, "description": "Aggregation (COUNT, SUM, AVG) with GROUP BY, no spatial"},
    "SQL_JOIN": {"complexity": 2, "frequency": 2, "description": "Standard table join without spatial predicates"},
    "MULTI_SQL_JOIN": {"complexity": 3, "frequency": 3, "description": "Multiple table joins (3+ tables)"},
    "NESTED_QUERY": {"complexity": 3, "frequency": 3, "description": "CTEs, subqueries, nested SELECT"},
    "SPATIAL_PREDICATE": {"complexity": 1, "frequency": 1, "description": "Spatial predicates (ST_Intersects, ST_Contains, ST_Within, ST_Touches)"},
    "SPATIAL_PREDICATE_DISTANCE": {"complexity": 2, "frequency": 2, "description": "Distance-based predicates (ST_DWithin, ST_Overlaps, ST_Crosses, ST_Disjoint)"},
    "SPATIAL_MEASUREMENT": {"complexity": 1, "frequency": 1, "description": "Basic measurements (ST_Area, ST_Distance, ST_Length, ST_Perimeter)"},
    "SPATIAL_PROCESSING": {"complexity": 2, "frequency": 1, "description": "Spatial processing (ST_Buffer, ST_Union, ST_Intersection, ST_Difference)"},
    "SPATIAL_ACCESSOR": {"complexity": 1, "frequency": 2, "description": "Coordinate extraction (ST_X, ST_Y, ST_Centroid, ST_Envelope)"},
    "SPATIAL_CONSTRUCTOR": {"complexity": 1, "frequency": 1, "description": "Geometry construction (ST_MakePoint, ST_GeomFromText, ST_Collect)"},
    "SPATIAL_TRANSFORM": {"complexity": 2, "frequency": 1, "description": "Coordinate transformation (ST_Transform, ST_SetSRID)"},
    "SPATIAL_VALIDATION": {"complexity": 2, "frequency": 1, "description": "Geometry validation (ST_IsValid, ST_MakeValid)"},
    "SPATIAL_JOIN": {"complexity": 2, "frequency": 1, "description": "Join using spatial predicates"},
    "MULTI_SPATIAL_JOIN": {"complexity": 3, "frequency": 3, "description": "Multiple spatial joins with complex predicates"},
    "SPATIAL_CLUSTERING": {"complexity": 3, "frequency": 3, "description": "Spatial clustering (ST_ClusterDBSCAN, ST_ClusterKMeans)"},
    "RASTER_ANALYSIS": {"complexity": 3, "frequency": 2, "description": "Raster analysis and raster_accessor functions (ST_Value, ST_SummaryStats)"},
    "RASTER_VECTOR": {"complexity": 3, "frequency": 3, "description": "Raster-vector integration (ST_Clip, ST_Intersection with raster)"}
}

DOMAIN_TAXONOMY = {
    "SINGLE_SCHEMA_CIM_VECTOR": {"complexity": 1, "frequency": 1, "description": "Single schema cim_vector only"},
    "MULTI_SCHEMA_WITH_CIM_VECTOR": {"complexity": 2, "frequency": 2, "description": "cim_vector + one other schema (census/network/raster)"},
    "SINGLE_SCHEMA_OTHER": {"complexity": 1, "frequency": 2, "description": "Single non-vector schema (census/network/raster only)"},
    "MULTI_SCHEMA_WITHOUT_CIM_VECTOR": {"complexity": 2, "frequency": 3, "description": "Multiple schemas without cim_vector"},
    "MULTI_SCHEMA_COMPLEX": {"complexity": 3, "frequency": 3, "description": "Three or more schemas combined"}
}

# Spatial function patterns
SPATIAL_PATTERNS = {
    "predicates": [r'ST_Intersects', r'ST_Contains', r'ST_Within', r'ST_Touches', r'ST_Equals', r'ST_Covers', r'ST_CoveredBy'],
    "predicates_distance": [r'ST_DWithin', r'ST_Overlaps', r'ST_Crosses', r'ST_Disjoint'],
    "measurements": [r'ST_Area', r'ST_Distance', r'ST_Length', r'ST_Perimeter', r'ST_3DDistance', r'ST_MaxDistance'],
    "processing": [r'ST_Buffer', r'ST_Union', r'ST_Intersection(?!_Raster)', r'ST_Difference', r'ST_SymDifference', r'ST_ConvexHull', r'ST_Simplify'],
    "accessors": [r'ST_X', r'ST_Y', r'ST_Z', r'ST_Centroid', r'ST_Envelope', r'ST_StartPoint', r'ST_EndPoint', r'ST_PointN', r'ST_GeometryN', r'ST_NumGeometries', r'ST_NumPoints', r'ST_SRID'],
    "constructors": [r'ST_MakePoint', r'ST_GeomFromText', r'ST_Collect', r'ST_MakeLine', r'ST_MakePolygon', r'ST_GeomFromGeoJSON', r'ST_Point', r'ST_Polygon'],
    "transforms": [r'ST_Transform', r'ST_SetSRID', r'ST_FlipCoordinates'],
    "validation": [r'ST_IsValid', r'ST_MakeValid', r'ST_IsSimple', r'ST_IsClosed'],
    "clustering": [r'ST_ClusterDBSCAN', r'ST_ClusterKMeans', r'ST_ClusterWithin'],
    "raster_analysis": [r'ST_Value', r'ST_SummaryStats', r'ST_Histogram', r'ST_Band', r'ST_BandMetaData', r'ST_RasterToWorldCoord'],
    "raster_vector": [r'ST_Clip', r'ST_Intersection_Raster', r'ST_AsRaster', r'ST_Resample']
}

SCHEMA_PATTERNS = {
    "cim_vector": [r'cim_vector\.', r'FROM\s+cim_vector', r'JOIN\s+cim_vector'],
    "cim_census": [r'cim_census\.', r'FROM\s+cim_census', r'JOIN\s+cim_census'],
    "cim_network": [r'cim_network\.', r'FROM\s+cim_network', r'JOIN\s+cim_network'],
    "cim_raster": [r'cim_raster\.', r'FROM\s+cim_raster', r'JOIN\s+cim_raster']
}


# ============================================================================
# TAXONOMY CLASSIFICATION
# ============================================================================

def classify_task_type(sql: str) -> Tuple[str, Dict[str, Any]]:
    """Classify SQL query into task taxonomy."""
    sql_upper = sql.upper()
    
    # Check for raster operations first
    raster_vector_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["raster_vector"])
    raster_analysis_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["raster_analysis"])
    
    if raster_vector_found:
        return "RASTER_VECTOR", TASK_TAXONOMY["RASTER_VECTOR"]
    if raster_analysis_found:
        return "RASTER_ANALYSIS", TASK_TAXONOMY["RASTER_ANALYSIS"]
    
    # Check for clustering
    clustering_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["clustering"])
    if clustering_found:
        return "SPATIAL_CLUSTERING", TASK_TAXONOMY["SPATIAL_CLUSTERING"]
    
    # Count joins and check for spatial predicates
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    spatial_predicates = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["predicates"])
    spatial_predicates_dist = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["predicates_distance"])
    
    # Check for spatial joins
    if join_count >= 2 and (spatial_predicates or spatial_predicates_dist):
        return "MULTI_SPATIAL_JOIN", TASK_TAXONOMY["MULTI_SPATIAL_JOIN"]
    if join_count >= 1 and (spatial_predicates or spatial_predicates_dist):
        return "SPATIAL_JOIN", TASK_TAXONOMY["SPATIAL_JOIN"]
    
    # Check other spatial operations
    validation_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["validation"])
    if validation_found:
        return "SPATIAL_VALIDATION", TASK_TAXONOMY["SPATIAL_VALIDATION"]
    
    transform_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["transforms"])
    if transform_found:
        return "SPATIAL_TRANSFORM", TASK_TAXONOMY["SPATIAL_TRANSFORM"]
    
    constructor_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["constructors"])
    if constructor_found:
        return "SPATIAL_CONSTRUCTOR", TASK_TAXONOMY["SPATIAL_CONSTRUCTOR"]
    
    accessor_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["accessors"])
    if accessor_found:
        return "SPATIAL_ACCESSOR", TASK_TAXONOMY["SPATIAL_ACCESSOR"]
    
    processing_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["processing"])
    if processing_found:
        return "SPATIAL_PROCESSING", TASK_TAXONOMY["SPATIAL_PROCESSING"]
    
    measurement_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["measurements"])
    if measurement_found:
        return "SPATIAL_MEASUREMENT", TASK_TAXONOMY["SPATIAL_MEASUREMENT"]
    
    if spatial_predicates_dist:
        return "SPATIAL_PREDICATE_DISTANCE", TASK_TAXONOMY["SPATIAL_PREDICATE_DISTANCE"]
    
    if spatial_predicates:
        return "SPATIAL_PREDICATE", TASK_TAXONOMY["SPATIAL_PREDICATE"]
    
    # Non-spatial operations
    has_cte = 'WITH' in sql_upper and 'AS' in sql_upper
    has_subquery = sql_upper.count('SELECT') > 1
    if has_cte or has_subquery:
        return "NESTED_QUERY", TASK_TAXONOMY["NESTED_QUERY"]
    
    if join_count >= 2:
        return "MULTI_SQL_JOIN", TASK_TAXONOMY["MULTI_SQL_JOIN"]
    
    if join_count == 1:
        return "SQL_JOIN", TASK_TAXONOMY["SQL_JOIN"]
    
    aggregation_pattern = r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\('
    has_aggregation = re.search(aggregation_pattern, sql_upper)
    has_group_by = 'GROUP BY' in sql_upper
    if has_aggregation or has_group_by:
        return "SQL_AGGREGATION", TASK_TAXONOMY["SQL_AGGREGATION"]
    
    return "SIMPLE_SELECT", TASK_TAXONOMY["SIMPLE_SELECT"]


def classify_domain_type(sql: str) -> Tuple[str, Dict[str, Any]]:
    """Classify SQL query into domain taxonomy."""
    schemas_used = set()
    
    for schema_name, patterns in SCHEMA_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                schemas_used.add(schema_name)
                break
    
    num_schemas = len(schemas_used)
    has_cim_vector = "cim_vector" in schemas_used
    
    if num_schemas >= 3:
        return "MULTI_SCHEMA_COMPLEX", DOMAIN_TAXONOMY["MULTI_SCHEMA_COMPLEX"]
    
    if num_schemas == 2:
        if has_cim_vector:
            return "MULTI_SCHEMA_WITH_CIM_VECTOR", DOMAIN_TAXONOMY["MULTI_SCHEMA_WITH_CIM_VECTOR"]
        else:
            return "MULTI_SCHEMA_WITHOUT_CIM_VECTOR", DOMAIN_TAXONOMY["MULTI_SCHEMA_WITHOUT_CIM_VECTOR"]
    
    if num_schemas == 1:
        if has_cim_vector:
            return "SINGLE_SCHEMA_CIM_VECTOR", DOMAIN_TAXONOMY["SINGLE_SCHEMA_CIM_VECTOR"]
        else:
            return "SINGLE_SCHEMA_OTHER", DOMAIN_TAXONOMY["SINGLE_SCHEMA_OTHER"]
    
    return "SINGLE_SCHEMA_CIM_VECTOR", DOMAIN_TAXONOMY["SINGLE_SCHEMA_CIM_VECTOR"]


def classify_question_tone(question: str) -> str:
    """Classify question tone."""
    question_lower = question.lower().strip()
    
    interrogative_starts = ['what', 'which', 'where', 'who', 'how', 'why', 'when', 'is', 'are', 'can', 'do', 'does']
    if any(question_lower.startswith(start) for start in interrogative_starts) or question_lower.endswith('?'):
        return "INTERROGATIVE"
    
    direct_starts = ['find', 'get', 'list', 'show', 'select', 'calculate', 'compute', 'return', 'retrieve', 'identify']
    if any(question_lower.startswith(start) for start in direct_starts):
        return "DIRECT"
    
    return "DESCRIPTIVE"


def get_spatial_functions(sql: str) -> List[str]:
    """Extract spatial functions from SQL."""
    return re.findall(r'ST_\w+', sql, re.IGNORECASE)


def classify_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a sample with full taxonomy."""
    sql = sample.get('sql') or sample.get('sql_postgis', '')
    question = sample.get('question', '')
    
    task_type, task_meta = classify_task_type(sql)
    domain_type, domain_meta = classify_domain_type(sql)
    question_tone = classify_question_tone(question)
    spatial_funcs = get_spatial_functions(sql)
    
    return {
        "task_type": task_type,
        "task_complexity": task_meta["complexity"],
        "task_frequency": task_meta["frequency"],
        "task_description": task_meta["description"],
        "domain_type": domain_type,
        "domain_complexity": domain_meta["complexity"],
        "domain_frequency": domain_meta["frequency"],
        "domain_description": domain_meta["description"],
        "question_tone": question_tone,
        "spatial_functions": spatial_funcs,
        "spatial_function_count": len(spatial_funcs)
    }


# ============================================================================
# DATA LOADING AND FILTERING
# ============================================================================

def load_dataset(input_file: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    print(f"Loading dataset from: {input_file}")
    samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}")
    
    print(f"Loaded {len(samples)} samples")
    return samples


def classify_all_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add taxonomy classification to all samples."""
    print("Classifying samples by taxonomy...")
    
    for sample in samples:
        classification = classify_sample(sample)
        sample['taxonomy'] = classification
    
    return samples


def group_by_taxonomy(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group samples by task and domain types."""
    groups = {
        'task': defaultdict(list),
        'domain': defaultdict(list)
    }
    
    for sample in samples:
        task_type = sample['taxonomy']['task_type']
        domain_type = sample['taxonomy']['domain_type']
        
        groups['task'][task_type].append(sample)
        groups['domain'][domain_type].append(sample)
    
    return groups


# ============================================================================
# SAMPLING STRATEGY
# ============================================================================

def stratified_sample_by_targets(
    samples: List[Dict[str, Any]],
    task_targets: Dict[str, float],
    domain_targets: Dict[str, float],
    total_size: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample to achieve target distributions.
    Priority: task_type > domain_type
    """
    print(f"\nStratified sampling for {total_size} samples...")
    print(f"Task type targets: {task_targets}")
    print(f"Domain type targets: {domain_targets}")
    
    random.seed(seed)
    
    # Group by task type
    task_groups = defaultdict(list)
    for sample in samples:
        task_type = sample['taxonomy']['task_type']
        task_groups[task_type].append(sample)
    
    # Calculate how many samples needed per task type
    task_allocations = {}
    for task_type, ratio in task_targets.items():
        target_count = int(total_size * ratio)
        available = len(task_groups.get(task_type, []))
        
        if available == 0:
            print(f"Warning: No samples available for task type '{task_type}'")
            task_allocations[task_type] = 0
        elif available < target_count:
            print(f"Warning: Only {available} samples available for '{task_type}' (target: {target_count})")
            task_allocations[task_type] = available
        else:
            task_allocations[task_type] = target_count
    
    # Adjust if we can't meet targets
    total_allocated = sum(task_allocations.values())
    if total_allocated < total_size:
        print(f"Adjusting allocation: {total_allocated} < {total_size}")
        # Distribute remaining across available groups
        remaining = total_size - total_allocated
        for task_type in task_allocations:
            available = len(task_groups.get(task_type, []))
            if available > task_allocations[task_type]:
                can_add = min(remaining, available - task_allocations[task_type])
                task_allocations[task_type] += can_add
                remaining -= can_add
                if remaining == 0:
                    break
    
    # Sample from each task group, considering domain distribution
    selected = []
    selected_ids = set()
    
    for task_type, target_count in task_allocations.items():
        if target_count == 0:
            continue
        
        candidates = task_groups.get(task_type, [])
        if not candidates:
            continue
        
        # Within this task type, try to match domain distribution
        domain_groups = defaultdict(list)
        for sample in candidates:
            domain_type = sample['taxonomy']['domain_type']
            domain_groups[domain_type].append(sample)
        
        # Allocate by domain within task
        task_selected = []
        for domain_type, domain_ratio in domain_targets.items():
            domain_target = int(target_count * domain_ratio)
            domain_candidates = domain_groups.get(domain_type, [])
            
            # Filter out already selected
            domain_candidates = [s for s in domain_candidates if id(s) not in selected_ids]
            
            if len(domain_candidates) >= domain_target:
                sampled = random.sample(domain_candidates, domain_target)
            else:
                sampled = domain_candidates
            
            for s in sampled:
                task_selected.append(s)
                selected_ids.add(id(s))
        
        # If we haven't reached target_count, add more from any domain
        if len(task_selected) < target_count:
            remaining_candidates = [s for s in candidates if id(s) not in selected_ids]
            needed = target_count - len(task_selected)
            if len(remaining_candidates) >= needed:
                additional = random.sample(remaining_candidates, needed)
            else:
                additional = remaining_candidates
            
            for s in additional:
                task_selected.append(s)
                selected_ids.add(id(s))
        
        # If we have too many, trim randomly
        if len(task_selected) > target_count:
            task_selected = random.sample(task_selected, target_count)
        
        selected.extend(task_selected)
    
    print(f"Selected {len(selected)} samples")
    
    # Print actual distribution
    task_counts = Counter(s['taxonomy']['task_type'] for s in selected)
    domain_counts = Counter(s['taxonomy']['domain_type'] for s in selected)
    
    print("\nActual task type distribution:")
    for task_type, count in sorted(task_counts.items()):
        pct = count / len(selected) * 100
        print(f"  {task_type:<30}: {count:>3} ({pct:>5.1f}%)")
    
    print("\nActual domain type distribution:")
    for domain_type, count in sorted(domain_counts.items()):
        pct = count / len(selected) * 100
        print(f"  {domain_type:<30}: {count:>3} ({pct:>5.1f}%)")
    
    return selected


# ============================================================================
# DATABASE EXECUTION
# ============================================================================

def convert_to_json_serializable(obj):
    """Convert non-JSON-serializable objects."""
    from uuid import UUID
    from decimal import Decimal
    from datetime import datetime, date
    
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    else:
        return obj


def execute_query(query: str, engine, timeout: int = 30) -> Dict[str, Any]:
    """Execute SQL query and return results."""
    start_time = time.time()
    
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {timeout * 1000};"))
            result = conn.execute(text(query))
            rows = result.fetchall()
            
            result_data = []
            for row in rows:
                converted_row = [convert_to_json_serializable(val) for val in row]
                result_data.append(converted_row)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': result_data,
                'rowcount': len(result_data),
                'execution_time': execution_time,
                'error': None
            }
    
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'success': False,
            'result': None,
            'rowcount': 0,
            'execution_time': execution_time,
            'error': str(e)
        }


def create_benchmark_with_execution(
    samples: List[Dict[str, Any]],
    db_uri: Optional[str],
    skip_execution: bool = False
) -> List[Dict[str, Any]]:
    """Create benchmark with ground truth execution results."""
    print(f"\nCreating benchmark from {len(samples)} samples...")
    
    engine = None
    if not skip_execution and db_uri:
        print("Connecting to database...")
        try:
            engine = create_engine(db_uri, poolclass=NullPool, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT version();"))
            print("Connected successfully")
        except Exception as e:
            print(f"Warning: Cannot connect to database: {e}")
            print("Continuing without execution...")
            skip_execution = True
    
    benchmark = []
    executed = 0
    failed = 0
    
    for idx, sample in enumerate(samples, 1):
        if idx % 20 == 0:
            print(f"  Processing {idx}/{len(samples)}...")
        
        sql = sample.get('sql') or sample.get('sql_postgis', '')
        question = sample.get('question', '')
        
        # Execute query
        exec_result = {'success': None, 'result': None, 'rowcount': None, 'execution_time': None, 'error': None}
        
        if not skip_execution and sql and engine:
            exec_result = execute_query(sql, engine, timeout=30)
            if exec_result['success']:
                executed += 1
            else:
                failed += 1
        
        # Create benchmark item
        benchmark_item = {
            'benchmark_id': idx,
            'original_id': sample.get('id', f'sample_{idx}'),
            
            # Question and SQL
            'question': question,
            'sql_postgis': sql,
            
            # Taxonomy classification
            'task_type': sample['taxonomy']['task_type'],
            'task_complexity': sample['taxonomy']['task_complexity'],
            'task_frequency': sample['taxonomy']['task_frequency'],
            'task_description': sample['taxonomy']['task_description'],
            'domain_type': sample['taxonomy']['domain_type'],
            'domain_complexity': sample['taxonomy']['domain_complexity'],
            'domain_frequency': sample['taxonomy']['domain_frequency'],
            'domain_description': sample['taxonomy']['domain_description'],
            'question_tone': sample['taxonomy']['question_tone'],
            
            # Spatial metadata
            'spatial_functions': sample['taxonomy']['spatial_functions'],
            'spatial_function_count': sample['taxonomy']['spatial_function_count'],
            
            # Execution results
            'expected_result': exec_result['result'],
            'expected_rowcount': exec_result['rowcount'],
            'executable': exec_result['success'],
            'execution_time': exec_result['execution_time'],
            'execution_error': exec_result['error'],
            
            # Additional metadata from original sample
            'original_metadata': {
                k: v for k, v in sample.items()
                if k not in ['sql', 'sql_postgis', 'question', 'taxonomy']
            }
        }
        
        benchmark.append(benchmark_item)
    
    print(f"\nBenchmark creation complete:")
    print(f"  Total samples: {len(benchmark)}")
    if not skip_execution:
        print(f"  Successfully executed: {executed}")
        print(f"  Failed execution: {failed}")
        if len(benchmark) > 0:
            print(f"  Success rate: {executed/len(benchmark)*100:.1f}%")
    
    return benchmark


# ============================================================================
# SAVE BENCHMARK
# ============================================================================

def save_benchmark(benchmark: List[Dict[str, Any]], output_file: Path):
    """Save benchmark to JSONL file."""
    print(f"\nSaving benchmark to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in benchmark:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(benchmark)} benchmark items")
    
    # Generate metadata
    task_counts = Counter(item['task_type'] for item in benchmark)
    domain_counts = Counter(item['domain_type'] for item in benchmark)
    complexity_counts = Counter(item['task_complexity'] for item in benchmark)
    
    metadata = {
        'benchmark_version': 'v2_taxonomy_based',
        'generated_at': datetime.now().isoformat(),
        'total_samples': len(benchmark),
        'executable_queries': sum(1 for item in benchmark if item.get('executable') is True),
        'failed_queries': sum(1 for item in benchmark if item.get('executable') is False),
        'task_type_distribution': dict(task_counts),
        'domain_type_distribution': dict(domain_counts),
        'task_complexity_distribution': dict(complexity_counts),
        'evaluation_modes': ['Q2SQL', 'EA (Eventual Accuracy)'],
        'metrics': ['EM', 'EX', 'EA', 'Deep EM', 'SC']
    }
    
    metadata_file = output_file.with_suffix('.meta.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to: {metadata_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate taxonomy-based evaluation benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL dataset')
    parser.add_argument('--output', type=str, default='benchmark_v2.jsonl',
                       help='Output benchmark file')
    parser.add_argument('--size', type=int, default=100,
                       help='Target benchmark size (default: 100)')
    parser.add_argument('--db_uri', type=str,
                       default="postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated",
                       help='Database URI for query execution')
    parser.add_argument('--skip_execution', action='store_true',
                       help='Skip query execution')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Task type distribution targets
    parser.add_argument('--task_simple_select', type=float, default=0.60,
                       help='SIMPLE_SELECT ratio (default: 0.60)')
    parser.add_argument('--task_sql_aggregation', type=float, default=0.10,
                       help='SQL_AGGREGATION ratio (default: 0.10)')
    parser.add_argument('--task_sql_join', type=float, default=0.05,
                       help='SQL_JOIN ratio (default: 0.05)')
    parser.add_argument('--task_spatial_measurement', type=float, default=0.10,
                       help='SPATIAL_MEASUREMENT ratio (default: 0.10)')
    parser.add_argument('--task_spatial_join', type=float, default=0.10,
                       help='SPATIAL_JOIN ratio (default: 0.10)')
    parser.add_argument('--task_spatial_accessor', type=float, default=0.05,
                       help='SPATIAL_ACCESSOR ratio (default: 0.05)')
    
    # Domain type distribution targets
    parser.add_argument('--domain_cim_vector', type=float, default=0.70,
                       help='SINGLE_SCHEMA_CIM_VECTOR ratio (default: 0.70)')
    parser.add_argument('--domain_other', type=float, default=0.30,
                       help='SINGLE_SCHEMA_OTHER ratio (default: 0.30)')
    
    parser.add_argument('--quality_threshold', type=float, default=0.0,
                       help='Minimum quality score (default: 0.0, no filtering)')
    
    args = parser.parse_args()
    
    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    
    # Build target distributions
    task_targets = {
        'SIMPLE_SELECT': args.task_simple_select,
        'SQL_AGGREGATION': args.task_sql_aggregation,
        'SQL_JOIN': args.task_sql_join,
        'SPATIAL_MEASUREMENT': args.task_spatial_measurement,
        'SPATIAL_JOIN': args.task_spatial_join,
        'SPATIAL_ACCESSOR': args.task_spatial_accessor
    }
    
    domain_targets = {
        'SINGLE_SCHEMA_CIM_VECTOR': args.domain_cim_vector,
        'SINGLE_SCHEMA_OTHER': args.domain_other
    }
    
    # Validate targets sum to ~1.0
    task_sum = sum(task_targets.values())
    domain_sum = sum(domain_targets.values())
    
    if abs(task_sum - 1.0) > 0.01:
        print(f"Warning: Task type ratios sum to {task_sum:.2f}, not 1.0")
    
    if abs(domain_sum - 1.0) > 0.01:
        print(f"Warning: Domain type ratios sum to {domain_sum:.2f}, not 1.0")
    
    print("="*70)
    print("TAXONOMY-BASED BENCHMARK GENERATOR v2")
    print("="*70)
    
    # Load dataset
    samples = load_dataset(input_path)
    
    # Quality filtering
    if args.quality_threshold > 0:
        print(f"\nFiltering by quality score >= {args.quality_threshold}...")
        samples = [s for s in samples if s.get('quality_score', 0) >= args.quality_threshold]
        print(f"Remaining samples: {len(samples)}")
    
    # Classify all samples
    samples = classify_all_samples(samples)
    
    # Stratified sampling
    selected = stratified_sample_by_targets(
        samples,
        task_targets,
        domain_targets,
        args.size,
        args.seed
    )
    
    if len(selected) == 0:
        print("Error: No samples selected")
        sys.exit(1)
    
    # Create benchmark with execution
    benchmark = create_benchmark_with_execution(
        selected,
        args.db_uri,
        args.skip_execution
    )
    
    # Save benchmark
    save_benchmark(benchmark, output_path)
    
    print("\n" + "="*70)
    print("BENCHMARK GENERATION COMPLETE")
    print("="*70)
    print(f"\nBenchmark file: {output_path}")
    print(f"Metadata file: {output_path.with_suffix('.meta.json')}")
    print(f"\nUsage:")
    print(f"  python evaluator_v2.py \\")
    print(f"    --benchmark {output_path} \\")
    print(f"    --model <model_spec> \\")
    print(f"    --mode Q2SQL \\")
    print(f"    --max_iterations 5")


if __name__ == '__main__':
    main()

