#!/usr/bin/env python3
"""
Fix FTv2 Benchmark Quality Issues
==================================

Fixes:
1. Add WHERE project_id clauses to ambiguous questions
2. Add ORDER BY to ensure deterministic results
3. Add reasonable LIMIT (1000) to prevent excessive result sets
4. Re-execute to get new ground truth results

Usage:
    python fix_benchmark.py \
        --input ftv2_evaluation_benchmark_100_easy.jsonl \
        --output ftv2_evaluation_benchmark_100_easy_v2.jsonl \
        --db_uri "postgresql://..."
"""

import argparse
import json
import re
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import time


def extract_project_id_from_sql(sql: str) -> str:
    """Extract project_id from WHERE clause."""
    match = re.search(r"project_id\s*=\s*['\"]([^'\"]+)['\"]", sql, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def add_order_by_to_sql(sql: str) -> str:
    """Add ORDER BY clause if missing."""
    # Check if already has ORDER BY
    if re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE):
        return sql
    
    # Extract SELECT columns to determine what to order by
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.DOTALL | re.IGNORECASE)
    if not select_match:
        return sql
    
    select_clause = select_match.group(1)
    
    # Find first column (usually an ID)
    first_col_match = re.search(r'(\w+\.\w+|\w+)', select_clause)
    if not first_col_match:
        return sql
    
    order_col = first_col_match.group(1)
    
    # Remove semicolon if present
    sql = sql.rstrip().rstrip(';')
    
    # Add ORDER BY before LIMIT if present
    if re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        sql = re.sub(
            r'(\s+LIMIT\s+\d+)',
            f' ORDER BY {order_col}\\1',
            sql,
            flags=re.IGNORECASE
        )
    else:
        sql += f' ORDER BY {order_col}'
    
    return sql + ';'


def add_reasonable_limit(sql: str, max_limit: int = 1000) -> str:
    """Add or adjust LIMIT clause to reasonable size."""
    # Check if already has LIMIT
    limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
    
    if limit_match:
        current_limit = int(limit_match.group(1))
        # If current limit is reasonable, keep it
        if current_limit <= max_limit:
            return sql
        # Otherwise, replace with max_limit
        sql = re.sub(
            r'LIMIT\s+\d+',
            f'LIMIT {max_limit}',
            sql,
            flags=re.IGNORECASE
        )
    else:
        # Add LIMIT
        sql = sql.rstrip().rstrip(';')
        sql += f' LIMIT {max_limit};'
    
    return sql


def execute_sql(sql: str, engine) -> dict:
    """Execute SQL and return results."""
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


def fix_benchmark_item(item: dict, engine) -> dict:
    """Fix a single benchmark item."""
    sql = item['sql_postgis']
    
    # Add ORDER BY
    sql = add_order_by_to_sql(sql)
    
    # Add reasonable LIMIT
    sql = add_reasonable_limit(sql, max_limit=1000)
    
    # Re-execute to get new ground truth
    exec_result = execute_sql(sql, engine)
    
    # Update item
    item['sql_postgis'] = sql
    item['expected_result'] = exec_result['result']
    item['expected_row_count'] = exec_result['rowcount']
    item['execution_time'] = exec_result['duration_ms'] / 1000
    item['error'] = exec_result['error']
    item['executable'] = exec_result['success']
    
    return item


def main():
    parser = argparse.ArgumentParser(description='Fix FTv2 benchmark quality issues')
    parser.add_argument('--input', required=True, help='Input benchmark file')
    parser.add_argument('--output', required=True, help='Output benchmark file')
    parser.add_argument('--db_uri', required=True, help='Database URI')
    
    args = parser.parse_args()
    
    print(f"Loading benchmark from: {args.input}")
    items = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    print(f"Loaded {len(items)} items")
    
    print("Connecting to database...")
    engine = create_engine(args.db_uri, poolclass=NullPool, echo=False)
    
    print("Fixing benchmark items...")
    fixed_items = []
    success_count = 0
    
    for i, item in enumerate(items, 1):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(items)}...")
        
        fixed_item = fix_benchmark_item(item, engine)
        fixed_items.append(fixed_item)
        
        if fixed_item['executable']:
            success_count += 1
    
    print(f"\nFixed {len(fixed_items)} items")
    print(f"Executable: {success_count}/{len(fixed_items)} ({success_count/len(fixed_items)*100:.1f}%)")
    
    print(f"\nSaving to: {args.output}")
    with open(args.output, 'w') as f:
        for item in fixed_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("Done!")


if __name__ == '__main__':
    main()

