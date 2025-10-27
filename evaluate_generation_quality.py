#!/usr/bin/env python3
"""
AI4DB Generation Quality Evaluator
===================================

Evaluates the quality of AI4DB Stage 1 and Stage 2 generated SQL queries
using the NoErr (No Error) metric - validates that SQL is syntactically correct
and executable on the target database.

Features:
- Evaluates SQL executability on target database
- Adds NoErr labels to each sample (no_error, error_category, error_message)
- Optionally saves annotated dataset with labels
- Optionally saves filtered dataset with only passing samples

Usage:
    # Basic evaluation with report only
    python evaluate_generation_quality.py \
        --input training_datasets/stage1_cim_dataset.jsonl \
        --output stage1_quality_report.json \
        --stage 1

    # Evaluation with annotated dataset output
    python evaluate_generation_quality.py \
        --input training_datasets/stage2_synthetic_dataset_ipazia.jsonl \
        --output stage2_quality_report.json \
        --output_annotated stage2_annotated.jsonl \
        --stage 2

    # Evaluation with filtered dataset (NoErr only)
    python evaluate_generation_quality.py \
        --input training_datasets/stage2_synthetic_dataset_ipazia.jsonl \
        --output stage2_quality_report.json \
        --output_filtered stage2_filtered.jsonl \
        --stage 2

    # Full evaluation with all outputs
    python evaluate_generation_quality.py \
        --input training_datasets/stage2_synthetic_dataset_ipazia.jsonl \
        --output stage2_quality_report.json \
        --output_annotated stage2_annotated.jsonl \
        --output_filtered stage2_filtered.jsonl \
        --stage 2

Metrics:
- NoErr: Percentage of queries that execute without errors
- Syntax errors: Percentage with syntax issues
- Schema errors: Percentage with invalid table/column references
- Timeout errors: Percentage that exceed time limit
- Success rate by SQL type, difficulty, etc.

Outputs:
- Quality report (JSON): Statistics and breakdown
- Annotated dataset (JSONL): Original samples + NoErr labels
- Filtered dataset (JSONL): Only samples with no_error=True
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
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


def categorize_error(error_msg: str) -> str:
    """
    Categorize error type based on error message.
    
    Categories:
    - SYNTAX_ERROR: SQL syntax issues
    - SCHEMA_ERROR: Invalid table/column references
    - PERMISSION_ERROR: Access denied
    - TIMEOUT_ERROR: Query timeout
    - OTHER_ERROR: Other errors
    """
    error_lower = error_msg.lower()
    
    if 'syntax error' in error_lower or 'parse error' in error_lower:
        return 'SYNTAX_ERROR'
    elif 'does not exist' in error_lower or 'no such table' in error_lower or 'no such column' in error_lower:
        return 'SCHEMA_ERROR'
    elif 'permission denied' in error_lower or 'access denied' in error_lower:
        return 'PERMISSION_ERROR'
    elif 'timeout' in error_lower or 'time limit' in error_lower:
        return 'TIMEOUT_ERROR'
    elif 'function' in error_lower and 'does not exist' in error_lower:
        return 'FUNCTION_ERROR'
    else:
        return 'OTHER_ERROR'


def evaluate_query(query: str, engine, timeout: int = 30) -> Dict[str, Any]:
    """
    Evaluate a single SQL query using NoErr metric.
    
    Returns:
        Dictionary with:
        - no_error: bool (True if query executes without error)
        - error: error message (if any)
        - error_category: error category
        - execution_time: float
    """
    if not query or not query.strip():
        return {
            'no_error': False,
            'error': 'Empty query',
            'error_category': 'SYNTAX_ERROR',
            'execution_time': 0.0
        }
    
    start_time = time.time()
    
    try:
        with engine.connect() as conn:
            # Set statement timeout
            conn.execute(text(f"SET statement_timeout = {timeout * 1000};"))
            
            # Execute query
            result = conn.execute(text(query))
            
            # Fetch results (to ensure query completes)
            _ = result.fetchall()
            
            execution_time = time.time() - start_time
            
            return {
                'no_error': True,
                'error': None,
                'error_category': None,
                'execution_time': execution_time
            }
    
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        error_category = categorize_error(error_msg)
        
        return {
            'no_error': False,
            'error': error_msg,
            'error_category': error_category,
            'execution_time': execution_time
        }


def evaluate_dataset(
    samples: List[Dict[str, Any]], 
    db_uri: str,
    stage: int
) -> Dict[str, Any]:
    """
    Evaluate entire dataset using NoErr metric.
    
    Args:
        samples: List of samples to evaluate
        db_uri: Database connection string
        stage: Dataset stage (1 or 2)
    
    Returns:
        Evaluation results dictionary
    """
    print(f"\nEvaluating Stage {stage} dataset with {len(samples)} samples...")
    
    # Connect to database
    print(f"Connecting to database...")
    engine = create_engine(db_uri, poolclass=NullPool, echo=False)
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"Connected to: {version[:60]}...")
    except Exception as e:
        print(f"Error: Cannot connect to database: {e}")
        sys.exit(1)
    
    # Evaluate each sample
    results = {
        'total_samples': len(samples),
        'stage': stage,
        'no_error_count': 0,
        'error_count': 0,
        'error_breakdown': defaultdict(int),
        'sql_type_breakdown': defaultdict(lambda: {'total': 0, 'no_error': 0}),
        'difficulty_breakdown': defaultdict(lambda: {'total': 0, 'no_error': 0}),
        'sample_results': []
    }
    
    for idx, sample in enumerate(samples, 1):
        if idx % 100 == 0:
            no_err_rate = results['no_error_count'] / idx * 100
            print(f"  Processed {idx}/{len(samples)} samples (NoErr: {no_err_rate:.1f}%)")
        
        sql_query = sample.get('sql_postgis', '')
        sql_type = sample.get('sql_type', 'UNKNOWN')
        difficulty = sample.get('difficulty_level', 'UNKNOWN')
        
        # Evaluate query
        eval_result = evaluate_query(sql_query, engine, timeout=30)
        
        # Update statistics
        if eval_result['no_error']:
            results['no_error_count'] += 1
            results['sql_type_breakdown'][sql_type]['no_error'] += 1
            results['difficulty_breakdown'][difficulty]['no_error'] += 1
        else:
            results['error_count'] += 1
            results['error_breakdown'][eval_result['error_category']] += 1
        
        results['sql_type_breakdown'][sql_type]['total'] += 1
        results['difficulty_breakdown'][difficulty]['total'] += 1
        
        # Store sample result
        sample_result = {
            'id': sample.get('id', f'sample_{idx}'),
            'sql_type': sql_type,
            'difficulty_level': difficulty,
            'no_error': eval_result['no_error'],
            'error_category': eval_result['error_category'],
            'error_message': eval_result['error'][:200] if eval_result['error'] else None,
            'execution_time': eval_result['execution_time']
        }
        
        results['sample_results'].append(sample_result)
    
    # Calculate overall NoErr rate
    results['no_error_rate'] = results['no_error_count'] / results['total_samples']
    results['error_rate'] = results['error_count'] / results['total_samples']
    
    # Calculate breakdown rates
    results['sql_type_breakdown'] = dict(results['sql_type_breakdown'])
    results['difficulty_breakdown'] = dict(results['difficulty_breakdown'])
    
    for sql_type, stats in results['sql_type_breakdown'].items():
        stats['no_error_rate'] = stats['no_error'] / stats['total'] if stats['total'] > 0 else 0
    
    for difficulty, stats in results['difficulty_breakdown'].items():
        stats['no_error_rate'] = stats['no_error'] / stats['total'] if stats['total'] > 0 else 0
    
    results['error_breakdown'] = dict(results['error_breakdown'])
    
    return results


def print_report(results: Dict[str, Any]):
    """Print evaluation report to console."""
    print("\n" + "="*80)
    print(f"STAGE {results['stage']} QUALITY EVALUATION REPORT")
    print("="*80)
    
    print(f"\nOverall Results:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  NoErr (No Error): {results['no_error_count']} ({results['no_error_rate']*100:.2f}%)")
    print(f"  Errors: {results['error_count']} ({results['error_rate']*100:.2f}%)")
    
    print(f"\nError Breakdown:")
    if results['error_breakdown']:
        sorted_errors = sorted(results['error_breakdown'].items(), 
                              key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors:
            percentage = count / results['total_samples'] * 100
            print(f"  {error_type:20s}: {count:5d} ({percentage:5.2f}%)")
    else:
        print("  No errors detected!")
    
    print(f"\nNoErr Rate by SQL Type:")
    sorted_sql_types = sorted(results['sql_type_breakdown'].items(), 
                             key=lambda x: x[1]['no_error_rate'], reverse=True)
    for sql_type, stats in sorted_sql_types[:10]:
        print(f"  {sql_type:25s}: {stats['no_error']:4d}/{stats['total']:4d} "
              f"({stats['no_error_rate']*100:5.1f}%)")
    
    if len(sorted_sql_types) > 10:
        print(f"  ... and {len(sorted_sql_types) - 10} more SQL types")
    
    print(f"\nNoErr Rate by Difficulty:")
    sorted_difficulties = sorted(results['difficulty_breakdown'].items(), 
                                key=lambda x: x[1]['no_error_rate'], reverse=True)
    for difficulty, stats in sorted_difficulties:
        print(f"  {difficulty:15s}: {stats['no_error']:4d}/{stats['total']:4d} "
              f"({stats['no_error_rate']*100:5.1f}%)")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    no_err_rate = results['no_error_rate']
    if no_err_rate >= 0.95:
        quality = "EXCELLENT"
    elif no_err_rate >= 0.90:
        quality = "VERY GOOD"
    elif no_err_rate >= 0.80:
        quality = "GOOD"
    elif no_err_rate >= 0.70:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print(f"  Stage {results['stage']} Quality: {quality}")
    print(f"  NoErr Rate: {no_err_rate*100:.2f}%")
    
    if results['stage'] == 1:
        print(f"\n  Stage 1 (Rule-based templates) should achieve ~100% NoErr rate.")
        if no_err_rate < 0.99:
            print(f"  WARNING: Lower than expected. Check template definitions.")
    elif results['stage'] == 2:
        print(f"\n  Stage 2 (CTGAN synthesis) target NoErr rate: >85%")
        if no_err_rate >= 0.85:
            print(f"  Status: PASSED - Ready for Stage 3 augmentation")
        else:
            print(f"  Status: NEEDS IMPROVEMENT - Consider:")
            print(f"    - Increase CTGAN training epochs")
            print(f"    - Improve quality filtering threshold")
            print(f"    - Enhance schema-aware SQL assembly")
    
    print("\n" + "="*80)


def save_report(results: Dict[str, Any], output_file: Path):
    """Save evaluation report to JSON file."""
    print(f"\nSaving report to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Report saved successfully")


def save_annotated_dataset(
    original_samples: List[Dict[str, Any]],
    results: Dict[str, Any],
    output_file: Path
):
    """
    Save dataset with NoErr labels added to each sample.
    
    Args:
        original_samples: Original dataset samples
        results: Evaluation results containing sample_results
        output_file: Output JSONL file path
    """
    print(f"\nSaving annotated dataset to: {output_file}")
    
    # Create a mapping of sample id to evaluation result
    eval_map = {}
    for sample_result in results['sample_results']:
        sample_id = sample_result['id']
        eval_map[sample_id] = {
            'no_error': sample_result['no_error'],
            'error_category': sample_result['error_category'],
            'error_message': sample_result['error_message'],
            'execution_time': sample_result['execution_time']
        }
    
    # Add NoErr labels to original samples
    annotated_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(original_samples, 1):
            # Get sample ID
            sample_id = sample.get('id', f'sample_{idx}')
            
            # Add NoErr evaluation results
            if sample_id in eval_map:
                sample['no_error'] = eval_map[sample_id]['no_error']
                sample['error_category'] = eval_map[sample_id]['error_category']
                sample['error_message'] = eval_map[sample_id]['error_message']
                sample['execution_time'] = eval_map[sample_id]['execution_time']
                annotated_count += 1
            else:
                # Should not happen, but handle gracefully
                sample['no_error'] = None
                sample['error_category'] = 'NOT_EVALUATED'
                sample['error_message'] = None
                sample['execution_time'] = None
            
            # Write to file
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {annotated_count} annotated samples")


def save_filtered_dataset(
    original_samples: List[Dict[str, Any]],
    results: Dict[str, Any],
    output_file: Path
):
    """
    Save filtered dataset with only samples that passed NoErr check.
    
    Args:
        original_samples: Original dataset samples
        results: Evaluation results containing sample_results
        output_file: Output JSONL file path
    """
    print(f"\nSaving filtered dataset (NoErr only) to: {output_file}")
    
    # Create a set of sample IDs that passed
    passing_ids = set()
    for sample_result in results['sample_results']:
        if sample_result['no_error']:
            passing_ids.add(sample_result['id'])
    
    # Save only passing samples
    passing_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(original_samples, 1):
            sample_id = sample.get('id', f'sample_{idx}')
            
            if sample_id in passing_ids:
                # Add NoErr flag for clarity
                sample['no_error'] = True
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                passing_count += 1
    
    print(f"Saved {passing_count} passing samples ({passing_count/len(original_samples)*100:.1f}% of total)")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AI4DB dataset generation quality using NoErr metric',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSONL file to evaluate'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('quality_report.json'),
        help='Output report JSON file (default: quality_report.json)'
    )
    
    parser.add_argument(
        '--output_annotated',
        type=Path,
        default=None,
        help='Output JSONL file with NoErr labels added to each sample (optional)'
    )
    
    parser.add_argument(
        '--output_filtered',
        type=Path,
        default=None,
        help='Output JSONL file with only passing samples (NoErr=True) (optional)'
    )
    
    parser.add_argument(
        '--stage',
        type=int,
        required=True,
        choices=[1, 2],
        help='Dataset stage (1 for rule-based, 2 for CTGAN)'
    )
    
    parser.add_argument(
        '--db_uri',
        type=str,
        default="postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated",
        help='Database URI for query execution'
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
    
    # Evaluate dataset
    results = evaluate_dataset(samples, args.db_uri, args.stage)
    
    # Print report
    print_report(results)
    
    # Save report
    save_report(results, args.output)
    
    # Save annotated dataset if requested
    if args.output_annotated:
        save_annotated_dataset(samples, results, args.output_annotated)
    
    # Save filtered dataset if requested
    if args.output_filtered:
        save_filtered_dataset(samples, results, args.output_filtered)
    
    print("\nEvaluation complete!")
    print(f"\nOutputs generated:")
    print(f"  - Quality report: {args.output}")
    if args.output_annotated:
        print(f"  - Annotated dataset: {args.output_annotated}")
    if args.output_filtered:
        print(f"  - Filtered dataset (NoErr only): {args.output_filtered}")


if __name__ == '__main__':
    main()

