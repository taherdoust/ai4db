#!/usr/bin/env python3
"""
Re-generate Ground Truth for Cleaned Benchmark
==============================================

Executes all SQL queries in the cleaned benchmark and stores actual results.
This ensures EX metric can properly match model outputs.

Usage:
    python regenerate_ground_truth.py \
        --input ftv2_evaluation_benchmark_100_easy_v2.jsonl \
        --output ftv2_evaluation_benchmark_100_easy_v3.jsonl \
        --db_uri "postgresql://..."
"""

import argparse
import json
import time
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


def execute_sql(sql: str, engine) -> dict:
    """Execute SQL and return results with metadata."""
    start_time = time.time()
    try:
        with engine.connect() as conn:
            conn.execute(text("SET statement_timeout = 30000;"))
            result = conn.execute(text(sql))
            rows = result.fetchall()
            duration_ms = (time.time() - start_time) * 1000
            return {
                'success': True,
                'result': [list(row) for row in rows],
                'rowcount': len(rows),
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


def main():
    parser = argparse.ArgumentParser(description='Re-generate ground truth for benchmark')
    parser.add_argument('--input', required=True, help='Input benchmark file')
    parser.add_argument('--output', required=True, help='Output benchmark file')
    parser.add_argument('--db_uri', required=True, help='Database URI')
    
    args = parser.parse_args()
    
    print(f"Loading benchmark from: {args.input}")
    with open(args.input, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(samples)} samples")
    
    print("Connecting to database...")
    engine = create_engine(args.db_uri, poolclass=NullPool, echo=False)
    
    print("\nRe-executing all SQL queries to generate ground truth...")
    success_count = 0
    failed_samples = []
    
    for i, sample in enumerate(samples, 1):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(samples)}...")
        
        sql = sample['sql_postgis']
        exec_result = execute_sql(sql, engine)
        
        if exec_result['success']:
            # Update ground truth
            sample['expected_result'] = exec_result['result']
            sample['expected_row_count'] = exec_result['rowcount']
            sample['execution_time'] = exec_result['duration_ms'] / 1000
            sample['executable'] = True
            sample['error'] = None
            success_count += 1
        else:
            sample['executable'] = False
            sample['error'] = exec_result['error']
            failed_samples.append((i, exec_result['error'][:100]))
    
    print(f"\nExecution complete:")
    print(f"  Success: {success_count}/{len(samples)} ({success_count/len(samples)*100:.1f}%)")
    print(f"  Failed: {len(failed_samples)}")
    
    if failed_samples:
        print(f"\nFailed samples:")
        for idx, error in failed_samples[:5]:
            print(f"  #{idx}: {error}")
    
    # Save updated benchmark
    print(f"\nSaving to: {args.output}")
    with open(args.output, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Done! Saved {len(samples)} samples with fresh ground truth.")
    
    # Statistics
    row_counts = [s['expected_row_count'] for s in samples if s.get('executable')]
    if row_counts:
        print(f"\nGround truth statistics:")
        print(f"  Min rows: {min(row_counts)}")
        print(f"  Max rows: {max(row_counts)}")
        print(f"  Avg rows: {sum(row_counts)/len(row_counts):.1f}")
        print(f"  Empty results: {sum(1 for c in row_counts if c == 0)}")


if __name__ == '__main__':
    main()

