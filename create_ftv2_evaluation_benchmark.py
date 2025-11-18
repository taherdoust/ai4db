#!/usr/bin/env python3
"""
FTv2 Evaluation Benchmark Generator
====================================

Creates small, high-quality evaluation benchmarks for three FTv2 training modes:
1. Q2SQL: Question → SQL
2. QInst2SQL: Question + Instruction → SQL  
3. Q2Inst: Question → Instruction

Features:
- Weighted stratified sampling by difficulty and SQL type
- Small size (<100 samples) for manual review
- Multi-task evaluation support
- Ground truth results from database execution

Usage:
    python create_ftv2_evaluation_benchmark.py \
        --input training_datasets/stage3_augmented_dataset_FINAL_checkpoint.jsonl \
        --output ftv2_evaluation_benchmark.jsonl \
        --size 80

Author: Ali Taherdoust
Date: October 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import random
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import time

# Importance weights for stratification
IMPORTANCE_WEIGHTS = {
    'difficulty': {
        'SIMPLE': 0.30,      # 30% - basic queries, must work
        'MEDIUM': 0.40,      # 40% - most common in production
        'HARD': 0.20,        # 20% - complex but important
        'VERY_HARD': 0.10    # 10% - edge cases
    },
    
    'sql_type': {
        'AGGREGATION': 1.5,
        'SPATIAL_JOIN': 1.4,
        'SPATIAL_MEASUREMENT': 1.3,
        'SIMPLE_SELECT': 1.2,
        'MULTI_JOIN': 1.1,
        'NESTED_QUERY': 1.0,
        'RASTER_VECTOR': 0.9,
        'WINDOW_FUNCTION': 0.8
    }
}


def calculate_sample_importance(sample: Dict[str, Any]) -> float:
    """Calculate importance score for weighted sampling."""
    difficulty = sample.get('difficulty_level', 'MEDIUM')
    sql_type = sample.get('sql_type', 'SIMPLE_SELECT')
    
    difficulty_weight = IMPORTANCE_WEIGHTS['difficulty'].get(difficulty, 0.25)
    sql_type_weight = IMPORTANCE_WEIGHTS['sql_type'].get(sql_type, 1.0)
    
    importance = difficulty_weight * sql_type_weight
    
    if 'quality_score' in sample:
        quality_bonus = sample['quality_score'] * 0.2
        importance += quality_bonus
    
    return importance


def load_dataset(input_file: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    print(f"Loading dataset from: {input_file}")
    
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(samples)} samples")
    return samples


def weighted_stratified_sample(
    samples: List[Dict[str, Any]], 
    target_size: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Perform weighted stratified sampling by difficulty."""
    print(f"\nPerforming weighted stratified sampling (target: {target_size})...")
    
    random.seed(seed)
    
    difficulty_groups = defaultdict(list)
    for sample in samples:
        difficulty = sample.get('difficulty_level', 'MEDIUM')
        difficulty_groups[difficulty].append(sample)
    
    selected = []
    
    for difficulty, group_samples in difficulty_groups.items():
        weight = IMPORTANCE_WEIGHTS['difficulty'].get(difficulty, 0.25)
        difficulty_target = max(1, int(target_size * weight))
        
        weighted_samples = [
            (sample, calculate_sample_importance(sample))
            for sample in group_samples
        ]
        
        weighted_samples.sort(key=lambda x: x[1], reverse=True)
        
        if len(weighted_samples) <= difficulty_target:
            selected.extend([s for s, _ in weighted_samples])
        else:
            selected.extend([s for s, _ in weighted_samples[:difficulty_target]])
    
    if len(selected) > target_size:
        weighted_all = [
            (sample, calculate_sample_importance(sample))
            for sample in selected
        ]
        weighted_all.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in weighted_all[:target_size]]
    
    print(f"Selected {len(selected)} samples")
    
    dist = defaultdict(int)
    for s in selected:
        dist[s.get('difficulty_level', 'UNKNOWN')] += 1
    
    print("\nDifficulty distribution:")
    for difficulty in ['SIMPLE', 'MEDIUM', 'HARD', 'VERY_HARD']:
        count = dist.get(difficulty, 0)
        if count > 0:
            pct = count / len(selected) * 100
            print(f"  {difficulty:12s}: {count:3d} ({pct:5.1f}%)")
    
    return selected


def calculate_difficulty_dimensions(sql: str) -> Dict[str, Any]:
    """
    Calculate difficulty across all dimensions (matching stage1 methodology).
    
    Dimensions:
    - Query Complexity: EASY, MEDIUM, HARD (based on CTEs, joins, subqueries)
    - Spatial Complexity: BASIC, INTERMEDIATE, ADVANCED (based on PostGIS functions)
    - Schema Complexity: SINGLE_TABLE, SINGLE_SCHEMA, MULTI_SCHEMA
    - Function Count: 0, 1, 2, 3+
    - Join Count: 0, 1, 2+
    - Overall Complexity Level: A, B, C
    """
    sql_upper = sql.upper()
    
    # Extract spatial functions
    spatial_functions = re.findall(r'ST_\w+', sql, re.IGNORECASE)
    spatial_func_count = len(spatial_functions)
    
    # Extract tables (FROM and JOIN clauses)
    tables = []
    # FROM clause
    from_matches = re.findall(r'FROM\s+(\w+\.\w+|\w+)', sql, re.IGNORECASE)
    tables.extend(from_matches)
    # JOIN clauses
    join_matches = re.findall(r'JOIN\s+(\w+\.\w+|\w+)', sql, re.IGNORECASE)
    tables.extend(join_matches)
    
    table_count = len(set(tables))
    
    # Calculate complexity score
    cte_count = sql_upper.count('WITH')
    join_count = sql_upper.count('JOIN')
    subquery_count = sql.count('(SELECT')
    
    complexity_score = 0
    if cte_count >= 2:
        complexity_score += 2
    elif cte_count == 1:
        complexity_score += 1
    
    if join_count >= 2:
        complexity_score += 2
    elif join_count == 1:
        complexity_score += 1
    
    if subquery_count >= 2:
        complexity_score += 2
    elif subquery_count == 1:
        complexity_score += 1
    
    if 'PARTITION BY' in sql_upper or 'ROW_NUMBER' in sql_upper:
        complexity_score += 2
    
    # Map to EASY, MEDIUM, HARD
    if complexity_score >= 5:
        query_complexity = "HARD"
    elif complexity_score >= 3:
        query_complexity = "MEDIUM"
    else:
        query_complexity = "EASY"
    
    # Spatial complexity
    advanced_spatial = ['ST_CLUSTER', 'ST_SUMMARYSTATS', 'ST_VALUE', 'ST_MAPALGEBRA']
    intermediate_spatial = ['ST_BUFFER', 'ST_TRANSFORM', 'ST_UNION', 'ST_INTERSECTION', 
                           'ST_DIFFERENCE', 'ST_SYMDIFFERENCE', 'ST_CONVEXHULL']
    
    if any(func in sql_upper for func in advanced_spatial):
        spatial_complexity = "ADVANCED"
    elif any(func in sql_upper for func in intermediate_spatial) or spatial_func_count >= 2:
        spatial_complexity = "INTERMEDIATE"
    elif spatial_func_count >= 1:
        spatial_complexity = "BASIC"
    else:
        spatial_complexity = "NONE"
    
    # Schema complexity
    schema_count = len(set(t.split('.')[0] for t in tables if '.' in t))
    if schema_count >= 2:
        schema_complexity = "MULTI_SCHEMA"
    elif table_count >= 2:
        schema_complexity = "SINGLE_SCHEMA"
    else:
        schema_complexity = "SINGLE_TABLE"
    
    # Function count categorization
    if spatial_func_count >= 3:
        function_count = "3+"
    elif spatial_func_count == 2:
        function_count = "2"
    elif spatial_func_count == 1:
        function_count = "1"
    else:
        function_count = "0"
    
    # Join count categorization
    if join_count >= 2:
        join_count_cat = "2+"
    elif join_count == 1:
        join_count_cat = "1"
    else:
        join_count_cat = "0"
    
    # Overall complexity level (A, B, C)
    if query_complexity == "HARD" or spatial_complexity == "ADVANCED" or schema_complexity == "MULTI_SCHEMA":
        overall_complexity = "C"
    elif query_complexity == "MEDIUM" or spatial_complexity == "INTERMEDIATE" or join_count >= 1:
        overall_complexity = "B"
    else:
        overall_complexity = "A"
    
    return {
        "query_complexity": query_complexity,
        "spatial_complexity": spatial_complexity,
        "schema_complexity": schema_complexity,
        "function_count": function_count,
        "join_count": join_count_cat,
        "overall_difficulty": query_complexity,
        "complexity_level": overall_complexity,
        "complexity_score": complexity_score,
        "spatial_functions": spatial_functions,
        "spatial_function_count": spatial_func_count,
        "table_count": table_count
    }


def convert_to_json_serializable(obj):
    """Convert non-JSON-serializable objects to serializable types."""
    from uuid import UUID
    from decimal import Decimal
    from datetime import datetime, date, time as dt_time
    
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date, dt_time)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.hex()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    else:
        return obj


def execute_query(query: str, engine, timeout: int = 30) -> Dict[str, Any]:
    """Execute SQL query and capture results."""
    start_time = time.time()
    
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {timeout * 1000};"))
            result = conn.execute(text(query))
            rows = result.fetchall()
            
            # Convert rows to JSON-serializable format
            result_data = []
            for row in rows:
                converted_row = [convert_to_json_serializable(val) for val in row]
                result_data.append(converted_row)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'result': result_data,
                'row_count': len(result_data),
                'execution_time': execution_time,
                'error': None
            }
    
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'success': False,
            'result': None,
            'row_count': 0,
            'execution_time': execution_time,
            'error': str(e)
        }


def create_ftv2_benchmark(
    samples: List[Dict[str, Any]],
    db_uri: str,
    skip_execution: bool = False
) -> List[Dict[str, Any]]:
    """Create FTv2 benchmark with fields for all three task types."""
    print(f"\nCreating FTv2 benchmark from {len(samples)} samples...")
    
    engine = None
    if not skip_execution:
        print(f"Connecting to database...")
        try:
            engine = create_engine(db_uri, poolclass=NullPool, echo=False)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                print(f"Connected successfully")
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
        
        sql_query = sample.get('sql_postgis', '')
        
        exec_result = {
            'success': None, 
            'result': None, 
            'row_count': None,
            'execution_time': None, 
            'error': None
        }
        
        if not skip_execution and sql_query and engine:
            exec_result = execute_query(sql_query, engine, timeout=30)
            if exec_result['success']:
                executed += 1
            else:
                failed += 1
        
        # Calculate difficulty dimensions from SQL
        difficulty_dims = calculate_difficulty_dimensions(sql_query)
        
        benchmark_item = {
            'benchmark_id': idx,
            'original_id': sample.get('id', f'sample_{idx}'),
            
            'question': sample.get('question', ''),
            'instruction': sample.get('instruction', ''),
            'sql_postgis': sql_query,
            
            'expected_result': exec_result['result'],
            'expected_row_count': exec_result['row_count'],
            
            # Difficulty dimensions (calculated from SQL)
            'difficulty_level': difficulty_dims['overall_difficulty'],
            'query_complexity': difficulty_dims['query_complexity'],
            'spatial_complexity': difficulty_dims['spatial_complexity'],
            'schema_complexity': difficulty_dims['schema_complexity'],
            'complexity_level': difficulty_dims['complexity_level'],
            'complexity_score': difficulty_dims['complexity_score'],
            
            # SQL metadata
            'sql_type': sample.get('sql_type', 'UNKNOWN'),
            'spatial_functions': difficulty_dims['spatial_functions'],
            'spatial_function_count': difficulty_dims['spatial_function_count'],
            'function_count': difficulty_dims['function_count'],
            'join_count': difficulty_dims['join_count'],
            'table_count': difficulty_dims['table_count'],
            
            'importance_score': calculate_sample_importance(sample),
            
            'executable': exec_result['success'],
            'execution_time': exec_result['execution_time'],
            'error': exec_result['error']
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


def save_benchmark(benchmark: List[Dict[str, Any]], output_file: Path):
    """Save benchmark and metadata."""
    print(f"\nSaving benchmark to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in benchmark:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(benchmark)} benchmark items")
    
    metadata = {
        'benchmark_version': 'FTv2',
        'benchmark_size': len(benchmark),
        'tasks_supported': ['Q2SQL', 'QInst2SQL', 'Q2Inst'],
        'executable_queries': sum(1 for item in benchmark if item.get('executable') is True),
        'failed_queries': sum(1 for item in benchmark if item.get('executable') is False),
        'difficulty_distribution': {},
        'query_complexity_distribution': {},
        'spatial_complexity_distribution': {},
        'schema_complexity_distribution': {},
        'complexity_level_distribution': {},
        'sql_type_distribution': {},
        'function_count_distribution': {},
        'join_count_distribution': {},
        'average_importance': sum(item['importance_score'] for item in benchmark) / len(benchmark) if benchmark else 0,
        'average_complexity_score': sum(item.get('complexity_score', 0) for item in benchmark) / len(benchmark) if benchmark else 0,
        'evaluation_metrics': {
            'Q2SQL': ['EM (Exact Match)', 'EX (Execution Accuracy)'],
            'QInst2SQL': ['EM (Exact Match)', 'EX (Execution Accuracy)'],
            'Q2Inst': ['Semantic Similarity', 'Downstream SQL Accuracy', 'Manual Review']
        }
    }
    
    # Calculate distributions
    for item in benchmark:
        diff = item.get('difficulty_level', 'UNKNOWN')
        sql_type = item.get('sql_type', 'UNKNOWN')
        query_complexity = item.get('query_complexity', 'UNKNOWN')
        spatial_complexity = item.get('spatial_complexity', 'UNKNOWN')
        schema_complexity = item.get('schema_complexity', 'UNKNOWN')
        complexity_level = item.get('complexity_level', 'UNKNOWN')
        function_count = item.get('function_count', '0')
        join_count = item.get('join_count', '0')
        
        metadata['difficulty_distribution'][diff] = \
            metadata['difficulty_distribution'].get(diff, 0) + 1
        metadata['query_complexity_distribution'][query_complexity] = \
            metadata['query_complexity_distribution'].get(query_complexity, 0) + 1
        metadata['spatial_complexity_distribution'][spatial_complexity] = \
            metadata['spatial_complexity_distribution'].get(spatial_complexity, 0) + 1
        metadata['schema_complexity_distribution'][schema_complexity] = \
            metadata['schema_complexity_distribution'].get(schema_complexity, 0) + 1
        metadata['complexity_level_distribution'][complexity_level] = \
            metadata['complexity_level_distribution'].get(complexity_level, 0) + 1
        metadata['sql_type_distribution'][sql_type] = \
            metadata['sql_type_distribution'].get(sql_type, 0) + 1
        metadata['function_count_distribution'][function_count] = \
            metadata['function_count_distribution'].get(function_count, 0) + 1
        metadata['join_count_distribution'][join_count] = \
            metadata['join_count_distribution'].get(join_count, 0) + 1
    
    metadata_file = output_file.with_suffix('.meta.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create FTv2 evaluation benchmark with weighted stratified sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input dataset (stage3_augmented_dataset_FINAL_checkpoint.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('ftv2_evaluation_benchmark.jsonl'),
        help='Output benchmark file (default: ftv2_evaluation_benchmark.jsonl)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=80,
        help='Target benchmark size (default: 80, recommended 50-100)'
    )
    
    parser.add_argument(
        '--db_uri',
        type=str,
        default="postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated",
        help='Database URI for query execution'
    )
    
    parser.add_argument(
        '--skip_execution',
        action='store_true',
        help='Skip query execution (faster, no ground truth)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    samples = load_dataset(args.input)
    
    if len(samples) == 0:
        print("Error: No samples loaded")
        sys.exit(1)
    
    print(f"\nFiltering to high-quality samples (quality_score >= 0.75, no_error = True)...")
    quality_filtered = [
        s for s in samples 
        if s.get('quality_score', 0) >= 0.75 and s.get('no_error', True)
    ]
    
    print(f"Quality filtered: {len(quality_filtered)} samples")
    
    if len(quality_filtered) < args.size:
        print(f"Warning: Only {len(quality_filtered)} samples available, using all")
        args.size = len(quality_filtered)
    
    benchmark_samples = weighted_stratified_sample(quality_filtered, args.size, args.seed)
    
    benchmark = create_ftv2_benchmark(benchmark_samples, args.db_uri, args.skip_execution)
    
    save_benchmark(benchmark, args.output)
    
    print("\n" + "="*70)
    print("FTv2 EVALUATION BENCHMARK CREATED")
    print("="*70)
    print(f"\nBenchmark file: {args.output}")
    print(f"Metadata file: {args.output.with_suffix('.meta.json')}")
    print(f"\nUsage:")
    print(f"  python ../assist_cim/evaluate_ftv2_models.py \\")
    print(f"    --benchmark {args.output} \\")
    print(f"    --model <model_name> \\")
    print(f"    --mode Q2SQL|QInst2SQL|Q2Inst")


if __name__ == '__main__':
    main()

