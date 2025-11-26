#!/usr/bin/env python3
"""
Re-generate Ground Truth for Cleaned Benchmark
==============================================

Executes all SQL queries (with LIMIT removed) and stores actual results.
Produces a clean benchmark with only essential fields.

Usage:
    python regenerate_ground_truth.py \
        --input benchmark_taxonomy_v2.jsonl \
        --output benchmark_taxonomy_v2_gt.jsonl \
        --db_uri "postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated"
"""

import argparse
import json
import re
import time
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from uuid import UUID
from decimal import Decimal
from datetime import datetime, date


def json_serializer(obj):
    """Convert non-JSON-serializable objects to serializable types."""
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, (list, tuple)):
        return [json_serializer(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: json_serializer(v) for k, v in obj.items()}
    else:
        return obj


def remove_limit_clause(sql: str) -> str:
    """
    Remove LIMIT clause from SQL query.
    Handles various LIMIT formats:
    - LIMIT 10
    - LIMIT 10;
    - LIMIT 10 with trailing whitespace
    """
    # Remove LIMIT at the end (with optional semicolon and whitespace)
    sql = re.sub(r'\s+LIMIT\s+\d+\s*;?\s*$', '', sql, flags=re.IGNORECASE)
    
    # Remove LIMIT in the middle (followed by newline or other clause)
    sql = re.sub(r'\s+LIMIT\s+\d+\s+', ' ', sql, flags=re.IGNORECASE)
    
    # Clean up any trailing whitespace
    sql = sql.strip()
    
    # Ensure semicolon at the end if not present
    if not sql.endswith(';'):
        sql += ';'
    
    return sql


def execute_sql(sql: str, engine) -> dict:
    """Execute SQL and return results with metadata."""
    start_time = time.time()
    try:
        with engine.connect() as conn:
            conn.execute(text("SET statement_timeout = 30000;"))
            result = conn.execute(text(sql))
            rows = result.fetchall()
            duration_ms = (time.time() - start_time) * 1000
            
            # Convert rows to JSON-serializable format
            serialized_rows = []
            for row in rows:
                serialized_row = [json_serializer(val) for val in row]
                serialized_rows.append(serialized_row)
            
            return {
                'success': True,
                'result': serialized_rows,
                'rowcount': len(serialized_rows),
                'duration_ms': duration_ms,
                'error': None
            }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return {
            'success': False,
            'result': None,
            'rowcount': 0,
            'duration_ms': duration_ms,
            'error': str(e)
        }


def clean_benchmark_item(sample: dict, exec_result: dict) -> dict:
    """
    Extract only essential fields for clean benchmark.
    
    Keeps:
    - benchmark_id
    - original_id
    - task taxonomy tags (task_type, task_complexity, task_frequency, task_description)
    - domain taxonomy tags (domain_type, domain_complexity, domain_frequency, domain_description)
    - question_tone
    - question
    - sql_postgis (cleaned, without LIMIT)
    - expected_result (from execution)
    """
    cleaned = {
        'benchmark_id': sample['benchmark_id'],
        'original_id': sample['original_id'],
        
        # Question
        'question': sample['question'],
        
        # SQL (cleaned)
        'sql_postgis': sample['sql_postgis'],
        
        # Task taxonomy
        'task_type': sample['task_type'],
        'task_complexity': sample['task_complexity'],
        'task_frequency': sample['task_frequency'],
        'task_description': sample['task_description'],
        
        # Domain taxonomy
        'domain_type': sample['domain_type'],
        'domain_complexity': sample['domain_complexity'],
        'domain_frequency': sample['domain_frequency'],
        'domain_description': sample['domain_description'],
        
        # Question tone
        'question_tone': sample['question_tone'],
        
        # Spatial functions (for deep EM calculation)
        'spatial_functions': sample.get('spatial_functions', []),
        'spatial_function_count': sample.get('spatial_function_count', 0),
        
        # Ground truth result
        'expected_result': exec_result['result'],
        'expected_rowcount': exec_result['rowcount']
    }
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description='Re-generate ground truth for taxonomy benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python regenerate_ground_truth.py \\
        --input benchmark_taxonomy_v2.jsonl \\
        --output benchmark_taxonomy_v2_gt.jsonl \\
        --db_uri "postgresql://cim_wizard_user:cim_wizard_password@localhost:15432/cim_wizard_integrated"
        """
    )
    
    parser.add_argument('--input', required=True, help='Input benchmark file (JSONL)')
    parser.add_argument('--output', required=True, help='Output benchmark file (JSONL)')
    parser.add_argument('--db_uri', required=True, help='Database URI for execution')
    
    args = parser.parse_args()
    
    print("="*70)
    print("BENCHMARK GROUND TRUTH REGENERATION")
    print("="*70)
    
    # Load benchmark
    print(f"\nLoading benchmark from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(samples)} samples")
    
    # Connect to database
    print("\nConnecting to database...")
    engine = create_engine(args.db_uri, poolclass=NullPool, echo=False)
    
    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT version();"))
        print("Database connection successful")
    except Exception as e:
        print(f"Error: Cannot connect to database: {e}")
        return 1
    
    # Process samples
    print("\nProcessing samples:")
    print("  1. Removing LIMIT clauses from SQL")
    print("  2. Executing cleaned SQL")
    print("  3. Generating ground truth results")
    print("  4. Cleaning and keeping only essential fields")
    print("")
    
    cleaned_samples = []
    success_count = 0
    failed_samples = []
    
    for i, sample in enumerate(samples, 1):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(samples)}...")
        
        # Remove LIMIT clause
        original_sql = sample['sql_postgis']
        cleaned_sql = remove_limit_clause(original_sql)
        
        # Update SQL in sample
        sample['sql_postgis'] = cleaned_sql
        
        # Execute cleaned SQL
        exec_result = execute_sql(cleaned_sql, engine)
        
        if exec_result['success']:
            # Clean and keep only essential fields
            cleaned_item = clean_benchmark_item(sample, exec_result)
            cleaned_samples.append(cleaned_item)
            success_count += 1
        else:
            # Still keep the item but with error information
            cleaned_item = clean_benchmark_item(sample, {
                'result': None,
                'rowcount': 0
            })
            cleaned_item['execution_error'] = exec_result['error']
            cleaned_samples.append(cleaned_item)
            failed_samples.append((i, exec_result['error'][:100]))
    
    print(f"\n{'='*70}")
    print("EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Successfully executed: {success_count}/{len(samples)} ({success_count/len(samples)*100:.1f}%)")
    print(f"  Failed: {len(failed_samples)}")
    
    if failed_samples:
        print(f"\nFailed samples (first 5):")
        for idx, error in failed_samples[:5]:
            print(f"  Sample #{idx}: {error}...")
    
    # Statistics on results
    successful_items = [s for s in cleaned_samples if 'execution_error' not in s]
    if successful_items:
        row_counts = [s['expected_rowcount'] for s in successful_items]
        print(f"\n{'='*70}")
        print("GROUND TRUTH STATISTICS")
        print(f"{'='*70}")
        print(f"  Min rows: {min(row_counts)}")
        print(f"  Max rows: {max(row_counts)}")
        print(f"  Avg rows: {sum(row_counts)/len(row_counts):.1f}")
        print(f"  Median rows: {sorted(row_counts)[len(row_counts)//2]}")
        print(f"  Empty results (0 rows): {sum(1 for c in row_counts if c == 0)}")
        print(f"  Results with 1-10 rows: {sum(1 for c in row_counts if 1 <= c <= 10)}")
        print(f"  Results with 11-50 rows: {sum(1 for c in row_counts if 11 <= c <= 50)}")
        print(f"  Results with 50+ rows: {sum(1 for c in row_counts if c > 50)}")
    
    # Field summary
    print(f"\n{'='*70}")
    print("CLEANED BENCHMARK FIELDS")
    print(f"{'='*70}")
    if cleaned_samples:
        print("  Fields retained:")
        for key in sorted(cleaned_samples[0].keys()):
            print(f"    • {key}")
    
    # Save cleaned benchmark
    print(f"\n{'='*70}")
    print(f"Saving cleaned benchmark to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in cleaned_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(cleaned_samples)} samples")
    
    # Final verification
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    # Check for LIMIT clauses in output
    limit_count = sum(1 for s in cleaned_samples if 'LIMIT' in s['sql_postgis'].upper())
    print(f"  Queries with LIMIT clause: {limit_count}")
    if limit_count > 0:
        print(f"  WARNING: Some queries still contain LIMIT clauses!")
    
    # Check field consistency
    expected_fields = set(cleaned_samples[0].keys())
    inconsistent = []
    for i, sample in enumerate(cleaned_samples, 1):
        if set(sample.keys()) != expected_fields:
            inconsistent.append(i)
    
    print(f"  Samples with consistent fields: {len(cleaned_samples) - len(inconsistent)}/{len(cleaned_samples)}")
    if inconsistent:
        print(f"  WARNING: Inconsistent fields in samples: {inconsistent[:5]}...")
    
    print(f"\n{'='*70}")
    print("COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nClean benchmark ready for evaluation:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"\nNext step:")
    print(f"  python evaluator_v2.py \\")
    print(f"    --benchmark {args.output} \\")
    print(f"    --model <model_spec> \\")
    print(f"    --mode Q2SQL")
    
    return 0


if __name__ == '__main__':
    exit(main())
