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
        
        benchmark_item = {
            'benchmark_id': idx,
            'original_id': sample.get('id', f'sample_{idx}'),
            
            'question': sample.get('question', ''),
            'instruction': sample.get('instruction', ''),
            'sql_postgis': sql_query,
            
            'expected_result': exec_result['result'],
            'expected_row_count': exec_result['row_count'],
            
            'difficulty_level': sample.get('difficulty_level', 'UNKNOWN'),
            'sql_type': sample.get('sql_type', 'UNKNOWN'),
            'spatial_function_usage': sample.get('spatial_function_usage', 'UNKNOWN'),
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
        'sql_type_distribution': {},
        'average_importance': sum(item['importance_score'] for item in benchmark) / len(benchmark) if benchmark else 0,
        'evaluation_metrics': {
            'Q2SQL': ['EM (Exact Match)', 'EX (Execution Accuracy)'],
            'QInst2SQL': ['EM (Exact Match)', 'EX (Execution Accuracy)'],
            'Q2Inst': ['Semantic Similarity', 'Downstream SQL Accuracy', 'Manual Review']
        }
    }
    
    for item in benchmark:
        diff = item['difficulty_level']
        sql_type = item['sql_type']
        
        metadata['difficulty_distribution'][diff] = \
            metadata['difficulty_distribution'].get(diff, 0) + 1
        metadata['sql_type_distribution'][sql_type] = \
            metadata['sql_type_distribution'].get(sql_type, 0) + 1
    
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

