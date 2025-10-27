#!/usr/bin/env python3
"""
Evaluation Benchmark Generator
==============================

Creates a stratified evaluation benchmark from AI4DB generated datasets.
Executes queries on the CIM database to capture ground truth results.

Usage:
    python create_evaluation_benchmark.py \
        --input training_datasets/stage3_augmented_dataset_final_checkpoint.jsonl \
        --output evaluation_benchmark.jsonl \
        --size 500 \
        --db_uri "postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated"

Features:
- Stratified sampling by difficulty, SQL type, spatial function usage
- Executes queries to capture expected results
- Validates query executability
- Creates reproducible benchmark with fixed seed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import random
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import time

def load_dataset(input_file: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    print(f"\nLoading dataset from: {input_file}")
    
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


def stratify_samples(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group samples into strata based on multiple criteria.
    
    Stratification keys:
    - difficulty_level (SIMPLE, MEDIUM, HARD, VERY_HARD)
    - sql_type (SELECT, SPATIAL_JOIN, AGGREGATION, etc.)
    - spatial_function_usage (CRITICAL, HIGH, MEDIUM, LOW)
    """
    print("\nStratifying samples...")
    
    strata = defaultdict(list)
    
    for sample in samples:
        # Extract stratification keys
        difficulty = sample.get('difficulty_level', 'UNKNOWN')
        sql_type = sample.get('sql_type', 'UNKNOWN')
        spatial_usage = sample.get('spatial_function_usage', 'UNKNOWN')
        
        # Create composite key
        stratum_key = f"{difficulty}|{sql_type}|{spatial_usage}"
        strata[stratum_key].append(sample)
    
    print(f"Created {len(strata)} strata")
    
    # Print stratum distribution
    print("\nStratum distribution:")
    sorted_strata = sorted(strata.items(), key=lambda x: len(x[1]), reverse=True)
    for key, samples_in_stratum in sorted_strata[:15]:
        difficulty, sql_type, spatial_usage = key.split('|')
        print(f"  {difficulty:12s} | {sql_type:20s} | {spatial_usage:10s} | {len(samples_in_stratum):5d} samples")
    
    if len(sorted_strata) > 15:
        remaining = sum(len(s) for _, s in sorted_strata[15:])
        print(f"  ... and {len(sorted_strata) - 15} more strata with {remaining} samples")
    
    return strata


def proportional_stratified_sample(
    strata: Dict[str, List[Dict[str, Any]]], 
    target_size: int, 
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Perform proportional stratified sampling.
    
    Args:
        strata: Dictionary mapping stratum keys to sample lists
        target_size: Target number of samples in benchmark
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled benchmark items
    """
    print(f"\nPerforming proportional stratified sampling (target size: {target_size})...")
    
    random.seed(seed)
    
    total_samples = sum(len(samples) for samples in strata.values())
    benchmark = []
    
    # Calculate samples per stratum proportionally
    for stratum_key, stratum_samples in strata.items():
        stratum_proportion = len(stratum_samples) / total_samples
        stratum_target = max(1, int(target_size * stratum_proportion))
        
        # Sample from this stratum
        if len(stratum_samples) <= stratum_target:
            # Take all samples if stratum is small
            selected = stratum_samples
        else:
            # Random sample
            selected = random.sample(stratum_samples, stratum_target)
        
        benchmark.extend(selected)
    
    # If we exceeded target, randomly downsample
    if len(benchmark) > target_size:
        benchmark = random.sample(benchmark, target_size)
    
    print(f"Sampled {len(benchmark)} samples for benchmark")
    
    return benchmark


def execute_query(query: str, engine, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute SQL query and capture results.
    
    Returns:
        Dictionary with:
        - success: bool
        - result: list of rows (if success)
        - error: error message (if failure)
        - execution_time: float
    """
    start_time = time.time()
    
    try:
        with engine.connect() as conn:
            # Set statement timeout
            conn.execute(text(f"SET statement_timeout = {timeout * 1000};"))
            
            # Execute query
            result = conn.execute(text(query))
            
            # Fetch results
            rows = result.fetchall()
            
            # Convert to list of tuples
            result_data = [tuple(row) for row in rows]
            
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


def create_benchmark(
    samples: List[Dict[str, Any]], 
    db_uri: str,
    skip_execution: bool = False
) -> List[Dict[str, Any]]:
    """
    Create benchmark by executing queries and capturing results.
    
    Args:
        samples: List of samples to convert to benchmark
        db_uri: Database connection string
        skip_execution: If True, skip query execution (faster but no ground truth)
    
    Returns:
        List of benchmark items with expected results
    """
    print(f"\nCreating benchmark from {len(samples)} samples...")
    
    if skip_execution:
        print("Skipping query execution (no ground truth results will be captured)")
        benchmark = []
        for idx, sample in enumerate(samples, 1):
            benchmark_item = {
                'benchmark_id': idx,
                'id': sample.get('id', f'sample_{idx}'),
                'question': sample.get('question', ''),
                'instruction': sample.get('instruction', ''),
                'sql_postgis': sample.get('sql_postgis', ''),
                'difficulty_level': sample.get('difficulty_level', 'UNKNOWN'),
                'sql_type': sample.get('sql_type', 'UNKNOWN'),
                'spatial_function_usage': sample.get('spatial_function_usage', 'UNKNOWN'),
                'expected_result': None,
                'expected_row_count': None,
                'executable': None,
                'execution_time': None,
                'error': None
            }
            benchmark.append(benchmark_item)
        return benchmark
    
    # Connect to database
    print(f"Connecting to database: {db_uri.split('@')[1] if '@' in db_uri else db_uri}")
    engine = create_engine(db_uri, poolclass=NullPool, echo=False)
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"Connected to: {version[:60]}...")
    except Exception as e:
        print(f"Error: Cannot connect to database: {e}")
        print("Continuing without execution...")
        skip_execution = True
    
    # Process samples
    benchmark = []
    executed = 0
    failed = 0
    
    for idx, sample in enumerate(samples, 1):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(samples)} samples ({executed} executed, {failed} failed)")
        
        sql_query = sample.get('sql_postgis', '')
        
        # Execute query
        if not skip_execution and sql_query:
            exec_result = execute_query(sql_query, engine, timeout=30)
        else:
            exec_result = {
                'success': None,
                'result': None,
                'row_count': None,
                'execution_time': None,
                'error': None
            }
        
        if exec_result['success']:
            executed += 1
        elif exec_result['success'] is False:
            failed += 1
        
        # Create benchmark item
        benchmark_item = {
            'benchmark_id': idx,
            'id': sample.get('id', f'sample_{idx}'),
            'question': sample.get('question', ''),
            'instruction': sample.get('instruction', ''),
            'sql_postgis': sql_query,
            'difficulty_level': sample.get('difficulty_level', 'UNKNOWN'),
            'sql_type': sample.get('sql_type', 'UNKNOWN'),
            'spatial_function_usage': sample.get('spatial_function_usage', 'UNKNOWN'),
            'expected_result': exec_result['result'],
            'expected_row_count': exec_result['row_count'],
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
        print(f"  Success rate: {executed/len(benchmark)*100:.1f}%")
    
    return benchmark


def save_benchmark(benchmark: List[Dict[str, Any]], output_file: Path):
    """Save benchmark to JSONL file."""
    print(f"\nSaving benchmark to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in benchmark:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(benchmark)} benchmark items")
    
    # Save metadata
    metadata_file = output_file.with_suffix('.meta.json')
    metadata = {
        'benchmark_size': len(benchmark),
        'executable_queries': sum(1 for item in benchmark if item.get('executable') is True),
        'failed_queries': sum(1 for item in benchmark if item.get('executable') is False),
        'difficulty_distribution': {},
        'sql_type_distribution': {},
        'spatial_usage_distribution': {}
    }
    
    # Calculate distributions
    for item in benchmark:
        difficulty = item.get('difficulty_level', 'UNKNOWN')
        sql_type = item.get('sql_type', 'UNKNOWN')
        spatial_usage = item.get('spatial_function_usage', 'UNKNOWN')
        
        metadata['difficulty_distribution'][difficulty] = \
            metadata['difficulty_distribution'].get(difficulty, 0) + 1
        metadata['sql_type_distribution'][sql_type] = \
            metadata['sql_type_distribution'].get(sql_type, 0) + 1
        metadata['spatial_usage_distribution'][spatial_usage] = \
            metadata['spatial_usage_distribution'].get(spatial_usage, 0) + 1
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create stratified evaluation benchmark from AI4DB dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSONL file (e.g., stage3_augmented_dataset_final_checkpoint.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('evaluation_benchmark.jsonl'),
        help='Output benchmark JSONL file (default: evaluation_benchmark.jsonl)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=500,
        help='Target benchmark size (default: 500)'
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
        help='Skip query execution (faster, but no ground truth results)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load dataset
    samples = load_dataset(args.input)
    
    if len(samples) == 0:
        print("Error: No samples loaded")
        sys.exit(1)
    
    # Stratify samples
    strata = stratify_samples(samples)
    
    # Perform stratified sampling
    benchmark_samples = proportional_stratified_sample(strata, args.size, args.seed)
    
    # Create benchmark with execution
    benchmark = create_benchmark(benchmark_samples, args.db_uri, args.skip_execution)
    
    # Save benchmark
    save_benchmark(benchmark, args.output)
    
    print("\nBenchmark generation complete!")
    print(f"\nUsage:")
    print(f"  python evaluate_models.py --benchmark {args.output} --model <model_name>")


if __name__ == '__main__':
    main()

