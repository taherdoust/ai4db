#!/usr/bin/env python3
"""
merge_datasets.py
Dataset Merger for Phase 4 Pipeline
====================================

Merges positive samples from Stage 1 (rule-based) and Stage 3 (augmented) 
with negative samples, ensuring standardized classification throughout the pipeline.

Note: Stage 3 output does NOT include Stage 1 samples - they are separate.
If you want both original (Stage 1) and augmented (Stage 3) samples in your
training set, use --stage1 to include Stage 1 output.

Standardized Classification Schema:
- schema_complexity: SINGLE_SCHEMA_CIM_VECTOR, MULTI_SCHEMA_WITH_CIM_VECTOR, 
                     SINGLE_SCHEMA_OTHER, MULTI_SCHEMA_WITHOUT_CIM_VECTOR
- sql_type: SIMPLE_SELECT, AGGREGATION, MULTI_JOIN, NESTED_QUERY, 
           SPATIAL_JOIN, SPATIAL_MEASUREMENT, SPATIAL_CLUSTERING, RASTER_VECTOR
- Top 15 spatial functions tracked as boolean flags
- sample_type: POSITIVE or NEGATIVE

Usage:
    # Merge Stage 3 + Negative samples
    python merge_datasets.py \
        --positive training_datasets/stage3_augmented_dataset.jsonl \
        --negative negative_samples.jsonl \
        --output training_datasets/merged_dataset_phase4.jsonl \
        --target_size 150000
    
    # Merge Stage 1 + Stage 3 + Negative samples (recommended)
    python merge_datasets.py \
        --stage1 training_datasets/stage1_v2_dataset_frequency.jsonl \
        --positive training_datasets/stage3_augmented_dataset.jsonl \
        --negative negative_samples.jsonl \
        --output training_datasets/merged_dataset_phase4.jsonl \
        --target_size 150000
    
    # Use ALL available samples (ignoring ratio)
    python merge_datasets.py \
        --stage1 training_datasets/stage1_v2_dataset_frequency.jsonl \
        --positive training_datasets/stage3_augmented_dataset.jsonl \
        --negative negative_samples.jsonl \
        --output training_datasets/merged_dataset_phase4.jsonl \
        --use_all_samples

Author: Ali Taherdoust
Date: November 2024
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime
import re

# ============================================================================
# STANDARDIZED CLASSIFICATION DEFINITIONS
# ============================================================================

# Schema complexity categories (Phase 4 standardized)
SCHEMA_COMPLEXITY_TYPES = {
    'SINGLE_SCHEMA_CIM_VECTOR': 'Single schema (cim_vector only)',
    'MULTI_SCHEMA_WITH_CIM_VECTOR': 'Multi-schema (cim_vector + other)',
    'SINGLE_SCHEMA_OTHER': 'Single schema (cim_census, cim_network, or cim_raster)',
    'MULTI_SCHEMA_WITHOUT_CIM_VECTOR': 'Multi-schema (without cim_vector)'
}

# SQL type categories (Phase 4 standardized)
SQL_TYPES = [
    'SIMPLE_SELECT',
    'AGGREGATION', 
    'MULTI_JOIN',
    'NESTED_QUERY',
    'SPATIAL_JOIN',
    'SPATIAL_MEASUREMENT',
    'SPATIAL_CLUSTERING',
    'RASTER_VECTOR'
]

# Top 15 most frequent spatial functions
TOP_15_SPATIAL_FUNCTIONS = [
    'ST_Area',
    'ST_Intersects',
    'ST_Centroid',
    'ST_Distance',
    'ST_SummaryStats',
    'ST_MakePoint',
    'ST_Year',
    'ST_DWithin',
    'ST_SetSRID',
    'ST_Within',
    'ST_Intersection',
    'ST_Clip',
    'ST_Y',
    'ST_X',
    'ST_Buffer'
]

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def extract_schemas_and_tables(sql: str) -> Tuple[set, set]:
    """Extract schema names and tables from SQL query."""
    schemas = set()
    tables = set()
    
    # Pattern to match schema.table or just table
    table_pattern = r'(?:FROM|JOIN)\s+(?:(\w+)\.)?(\w+)'
    matches = re.findall(table_pattern, sql, re.IGNORECASE)
    
    for schema, table in matches:
        if schema:
            schemas.add(schema.lower())
            tables.add(f"{schema.lower()}.{table.lower()}")
        else:
            tables.add(table.lower())
    
    return schemas, tables

def classify_schema_complexity(sql: str) -> str:
    """Classify schema complexity according to Phase 4 standards."""
    schemas, tables = extract_schemas_and_tables(sql)
    
    # Count schemas
    cim_vector_present = 'cim_vector' in schemas
    cim_census_present = 'cim_census' in schemas
    cim_network_present = 'cim_network' in schemas
    cim_raster_present = 'cim_raster' in schemas
    
    schema_count = len(schemas)
    
    if schema_count == 0:
        # No explicit schema, assume cim_vector
        return 'SINGLE_SCHEMA_CIM_VECTOR'
    elif schema_count == 1:
        if cim_vector_present:
            return 'SINGLE_SCHEMA_CIM_VECTOR'
        else:
            return 'SINGLE_SCHEMA_OTHER'
    else:  # Multiple schemas
        if cim_vector_present:
            return 'MULTI_SCHEMA_WITH_CIM_VECTOR'
        else:
            return 'MULTI_SCHEMA_WITHOUT_CIM_VECTOR'

def classify_sql_type(sql: str) -> str:
    """Classify SQL query type according to Phase 4 standards."""
    sql_upper = sql.upper()
    
    # Check for raster operations
    if 'ST_VALUE' in sql_upper or 'ST_SUMMARYSTATS' in sql_upper or 'ST_CLIP' in sql_upper:
        return 'RASTER_VECTOR'
    
    # Check for spatial clustering
    if 'ST_CLUSTERDBSCAN' in sql_upper or 'ST_CLUSTERKMEANS' in sql_upper:
        return 'SPATIAL_CLUSTERING'
    
    # Check for nested queries
    if 'WITH' in sql_upper or '(SELECT' in sql:
        return 'NESTED_QUERY'
    
    # Count joins
    join_count = sql_upper.count('JOIN')
    
    # Check for spatial joins
    spatial_predicates = ['ST_INTERSECTS', 'ST_WITHIN', 'ST_CONTAINS', 'ST_TOUCHES', 'ST_OVERLAPS', 'ST_DWITHIN']
    if join_count >= 1 and any(pred in sql_upper for pred in spatial_predicates):
        return 'SPATIAL_JOIN'
    
    # Check for multiple joins
    if join_count >= 2:
        return 'MULTI_JOIN'
    
    # Check for aggregation
    if 'GROUP BY' in sql_upper or any(agg in sql_upper for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
        return 'AGGREGATION'
    
    # Check for spatial measurement
    spatial_measurements = ['ST_AREA', 'ST_LENGTH', 'ST_DISTANCE', 'ST_PERIMETER']
    if any(func in sql_upper for func in spatial_measurements):
        return 'SPATIAL_MEASUREMENT'
    
    return 'SIMPLE_SELECT'

def extract_spatial_functions(sql: str) -> List[str]:
    """Extract all spatial functions from SQL query."""
    functions = re.findall(r'ST_\w+', sql, re.IGNORECASE)
    return list(set([f.upper() for f in functions]))

def create_spatial_function_flags(spatial_functions: List[str]) -> Dict[str, bool]:
    """Create boolean flags for top 15 spatial functions."""
    flags = {}
    normalized_funcs = [f.upper() for f in spatial_functions]
    
    for func in TOP_15_SPATIAL_FUNCTIONS:
        flags[f"has_{func.lower()}"] = func.upper() in normalized_funcs
    
    return flags

def standardize_sample(sample: Dict[str, Any], sample_type: str = 'POSITIVE') -> Dict[str, Any]:
    """
    Standardize a sample with Phase 4 classification schema.
    
    Output field order:
    1. id
    2. Classification fields (schema_complexity, sql_type, etc.)
    3. question
    4. sql_postgis
    5. result/error
    """
    sql = sample.get('sql_postgis', '') or sample.get('sql', '')
    
    # Extract schemas and tables
    #schemas, tables = extract_schemas_and_tables(sql)
    
    # Extract spatial functions
    #spatial_functions = extract_spatial_functions(sql)
    
    # Create standardized sample
    standardized = {
        # 1. ID (always first)
        'id': sample.get('id', f'sample_{random.randint(100000, 999999)}'),
        
        # 2. Classification fields
        #'sample_type': sample_type,
        #'schema_complexity': classify_schema_complexity(sql),
        #'sql_type': classify_sql_type(sql),
        #'schema_count': len(schemas),
        #'table_count': len(tables),
        #'join_count': sql.upper().count('JOIN'),
        #'function_count': len(re.findall(r'\w+\(', sql)),
        #'spatial_function_count': len(spatial_functions),
        #'spatial_functions': spatial_functions,
        
        # Add top 15 spatial function flags
        #**create_spatial_function_flags(spatial_functions),
        
        # 3. Question and instruction
        'question': sample.get('question', ''),
        #'instruction': sample.get('instruction', ''),
        
        # 4. SQL
        'sql_postgis': sql,
        
        # 5. Quality and execution info
        #'quality_score': sample.get('quality_score', 1.0),
        #'no_error': sample.get('no_error', True),
        #'error_message': sample.get('error_message', None),
        
        # 6. Metadata
        #'stage': sample.get('stage', 'unknown'),
        #'original_id': sample.get('id', ''),
        'merged_at': datetime.now().isoformat()
    }
    
    # Preserve taxonomy fields from original sample if available
    # Task taxonomy
    if 'task_type' in sample:
        standardized['task_type'] = sample['task_type']
    if 'task_complexity' in sample:
        standardized['task_complexity'] = sample['task_complexity']
    if 'task_frequency' in sample:
        standardized['task_frequency'] = sample['task_frequency']
    
    # Domain taxonomy
    if 'domain_type' in sample:
        standardized['domain_type'] = sample['domain_type']
    if 'domain_complexity' in sample:
        standardized['domain_complexity'] = sample['domain_complexity']
    if 'domain_frequency' in sample:
        standardized['domain_frequency'] = sample['domain_frequency']
    
    # Question tone - preserve from original or classify
    if 'question_tone' in sample:
        standardized['question_tone'] = sample['question_tone']
    elif sample_type == 'NEGATIVE':
        # Negative samples should have AMBIGUOUS or OUT_OF_SCOPE tone
        # Check sample_dirtiness to infer if not set
        if sample.get('sample_dirtiness') == 'OUT_OF_SCOPE':
            standardized['question_tone'] = 'OUT_OF_SCOPE'
        else:
            standardized['question_tone'] = 'AMBIGUOUS'  # Default for negative
    else:
        # Try to classify question tone for positive samples
        question_lower = standardized['question'].lower()
        if any(q in question_lower for q in ['what', 'which', 'where', 'when', 'how']):
            standardized['question_tone'] = 'INTERROGATIVE'
        elif any(q in question_lower for q in ['find', 'get', 'show', 'list']):
            standardized['question_tone'] = 'DIRECT'
        else:
            standardized['question_tone'] = 'DESCRIPTIVE'
    
    # CRITICAL: Preserve sample_dirtiness from original sample
    if 'sample_dirtiness' in sample:
        standardized['sample_dirtiness'] = sample['sample_dirtiness']
    elif sample_type == 'NEGATIVE':
        # For negative samples, infer from question_tone if not set
        if standardized.get('question_tone') in ['AMBIGUOUS', 'OUT_OF_SCOPE']:
            standardized['sample_dirtiness'] = standardized['question_tone']
        else:
            standardized['sample_dirtiness'] = 'AMBIGUOUS'  # Default for negative
    else:
        standardized['sample_dirtiness'] = 'CLEAN'  # Default for positive
    
    return standardized

# ============================================================================
# DATASET LOADING AND MERGING
# ============================================================================

def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    print(f"Loading dataset from: {file_path}")
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
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
    
    print(f"  Loaded {len(samples):,} samples")
    return samples

def merge_datasets(
    positive_samples: List[Dict[str, Any]],
    negative_samples: List[Dict[str, Any]],
    target_size: Optional[int] = None,
    negative_ratio: float = 0.20,
    use_all_samples: bool = False,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Merge positive and negative samples, respecting negative ratio or using all samples.
    
    The function can either:
    1. Use all available samples (if use_all_samples=True)
    2. Maintain target negative ratio (if use_all_samples=False)
    
    If target_size is provided, it acts as a maximum cap.
    
    Args:
        positive_samples: Positive training samples
        negative_samples: Negative training samples
        target_size: Optional maximum total dataset size (if None, uses all available)
        negative_ratio: Target ratio of negative samples (0.20 = 20%) - ignored if use_all_samples=True
        use_all_samples: If True, use all available samples regardless of ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        Merged dataset with actual size and ratio reported
    """
    random.seed(random_seed)
    
    # Standardize all samples first
    print("\nStandardizing positive samples...")
    standardized_positive = []
    for sample in positive_samples:
        standardized = standardize_sample(sample, sample_type='POSITIVE')
        standardized_positive.append(standardized)
    
    print("\nStandardizing negative samples...")
    standardized_negative = []
    for sample in negative_samples:
        standardized = standardize_sample(sample, sample_type='NEGATIVE')
        standardized_negative.append(standardized)
    
    available_positive = len(standardized_positive)
    available_negative = len(standardized_negative)
    
    print(f"\nMerging datasets:")
    print(f"  Available positive: {available_positive:,}")
    print(f"  Available negative: {available_negative:,}")
    
    if use_all_samples:
        print(f"  Mode: Use ALL available samples (ignoring ratio)")
        actual_positive = available_positive
        actual_negative = available_negative
    else:
        print(f"  Mode: Maintain negative ratio")
        print(f"  Target negative ratio: {negative_ratio:.1%}")
        if target_size:
            print(f"  Target max size: {target_size:,}")
        
        # Calculate how many negatives we need based on available positives
        # To achieve negative_ratio: neg / (pos + neg) = ratio
        # Solving: neg = pos * ratio / (1 - ratio)
        if available_positive > 0:
            required_negative = int(available_positive * negative_ratio / (1 - negative_ratio))
        else:
            required_negative = 0
        
        # Determine actual counts based on what's available
        if available_negative >= required_negative:
            # We have enough negatives - use required amount to maintain ratio
            actual_negative = min(required_negative, available_negative)
            actual_positive = available_positive
        else:
            # Not enough negatives - use all negatives, adjust positives to maintain ratio
            actual_negative = available_negative
            if actual_negative > 0:
                # Calculate how many positives we need: pos = neg * (1 - ratio) / ratio
                required_positive = int(actual_negative * (1 - negative_ratio) / negative_ratio)
                actual_positive = min(required_positive, available_positive)
            else:
                # No negatives available - use all positives
                actual_positive = available_positive
        
        # Apply target_size cap if provided
        if target_size:
            current_total = actual_positive + actual_negative
            if current_total > target_size:
                # Scale down proportionally while maintaining ratio
                scale = target_size / current_total
                actual_positive = int(actual_positive * scale)
                actual_negative = int(actual_negative * scale)
                # Ensure we don't exceed available samples
                actual_positive = min(actual_positive, available_positive)
                actual_negative = min(actual_negative, available_negative)
    
    # Sample if we have more than needed
    if len(standardized_positive) > actual_positive:
        print(f"  Sampling {actual_positive:,} from {len(standardized_positive):,} positive samples")
        standardized_positive = random.sample(standardized_positive, actual_positive)
    else:
        print(f"  Using all {len(standardized_positive):,} positive samples")
    
    if len(standardized_negative) > actual_negative:
        print(f"  Sampling {actual_negative:,} from {len(standardized_negative):,} negative samples")
        standardized_negative = random.sample(standardized_negative, actual_negative)
    else:
        print(f"  Using all {len(standardized_negative):,} negative samples")
    
    # Combine and shuffle
    merged = standardized_positive + standardized_negative
    random.shuffle(merged)
    
    actual_ratio = len(standardized_negative) / len(merged) if merged else 0
    
    print(f"\nMerged dataset:")
    print(f"  Total samples: {len(merged):,}")
    print(f"  Positive: {len(standardized_positive):,} ({len(standardized_positive)/len(merged)*100:.1f}%)")
    print(f"  Negative: {len(standardized_negative):,} ({len(standardized_negative)/len(merged)*100:.1f}%)")
    print(f"  Actual negative ratio: {actual_ratio:.1%}")
    
    if use_all_samples:
        print(f"  Note: All available samples used (ratio not enforced)")
    else:
        if target_size and len(merged) < target_size:
            print(f"  Note: Target size {target_size:,} not reached (only {len(merged):,} samples available)")
        if abs(actual_ratio - negative_ratio) > 0.05:
            print(f"  Warning: Actual ratio ({actual_ratio:.1%}) differs from target ({negative_ratio:.1%})")
            print(f"           This is because available samples don't allow exact ratio matching.")
    
    return merged

# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def generate_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive statistics for merged dataset."""
    stats = {
        'total_samples': len(samples),
        'generation_date': datetime.now().isoformat(),
        'sample_type_distribution': Counter(),
        'schema_complexity_distribution': Counter(),
        'sql_type_distribution': Counter(),
        'question_tone_distribution': Counter(),
        'spatial_function_frequency': Counter(),
        'top_15_function_coverage': {},
        'average_metrics': {}
    }
    
    # Collect statistics
    total_schemas = 0
    total_tables = 0
    total_joins = 0
    total_functions = 0
    total_spatial_functions = 0
    
    for sample in samples:
        # Sample type
        stats['sample_type_distribution'][sample.get('sample_type', 'UNKNOWN')] += 1
        
        # Schema complexity
        stats['schema_complexity_distribution'][sample.get('schema_complexity', 'UNKNOWN')] += 1
        
        # SQL type
        stats['sql_type_distribution'][sample.get('sql_type', 'UNKNOWN')] += 1
        
        # Question tone
        stats['question_tone_distribution'][sample.get('question_tone', 'UNKNOWN')] += 1
        
        # Spatial functions
        for func in sample.get('spatial_functions', []):
            stats['spatial_function_frequency'][func] += 1
        
        # Accumulate metrics
        total_schemas += sample.get('schema_count', 0)
        total_tables += sample.get('table_count', 0)
        total_joins += sample.get('join_count', 0)
        total_functions += sample.get('function_count', 0)
        total_spatial_functions += sample.get('spatial_function_count', 0)
    
    # Calculate averages
    if len(samples) > 0:
        stats['average_metrics'] = {
            'avg_schemas_per_query': total_schemas / len(samples),
            'avg_tables_per_query': total_tables / len(samples),
            'avg_joins_per_query': total_joins / len(samples),
            'avg_functions_per_query': total_functions / len(samples),
            'avg_spatial_functions_per_query': total_spatial_functions / len(samples)
        }
    
    # Calculate top 15 function coverage
    for func in TOP_15_SPATIAL_FUNCTIONS:
        flag_name = f"has_{func.lower()}"
        count = sum(1 for s in samples if s.get(flag_name, False))
        stats['top_15_function_coverage'][func] = {
            'count': count,
            'percentage': count / len(samples) * 100 if len(samples) > 0 else 0
        }
    
    return stats

def print_statistics(stats: Dict[str, Any]):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print("MERGED DATASET STATISTICS")
    print("="*80)
    
    print(f"\nTotal samples: {stats['total_samples']:,}")
    print(f"Generated: {stats['generation_date']}")
    
    print(f"\nSample Type Distribution:")
    for sample_type, count in stats['sample_type_distribution'].most_common():
        percentage = count / stats['total_samples'] * 100
        print(f"  {sample_type:20s}: {count:7,} ({percentage:5.1f}%)")
    
    print(f"\nSchema Complexity Distribution:")
    for complexity, count in stats['schema_complexity_distribution'].most_common():
        percentage = count / stats['total_samples'] * 100
        desc = SCHEMA_COMPLEXITY_TYPES.get(complexity, complexity)
        print(f"  {complexity:35s}: {count:7,} ({percentage:5.1f}%)")
    
    print(f"\nSQL Type Distribution:")
    for sql_type, count in stats['sql_type_distribution'].most_common():
        percentage = count / stats['total_samples'] * 100
        print(f"  {sql_type:20s}: {count:7,} ({percentage:5.1f}%)")
    
    print(f"\nQuestion Tone Distribution:")
    for tone, count in stats['question_tone_distribution'].most_common():
        percentage = count / stats['total_samples'] * 100
        print(f"  {tone:20s}: {count:7,} ({percentage:5.1f}%)")
    
    print(f"\nAverage Metrics:")
    for metric, value in stats['average_metrics'].items():
        print(f"  {metric:30s}: {value:8.2f}")
    
    print(f"\nTop 15 Spatial Functions Coverage:")
    for func, data in sorted(stats['top_15_function_coverage'].items(), 
                             key=lambda x: x[1]['count'], reverse=True):
        print(f"  {func:15s}: {data['count']:7,} ({data['percentage']:5.1f}%)")
    
    print("\n" + "="*80)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Merge positive and negative datasets with standardized classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge Stage 3 + Negative samples
  python merge_datasets.py --positive stage3_output.jsonl --negative negative_samples.jsonl --output merged.jsonl

  # Merge Stage 1 + Stage 3 + Negative samples
  python merge_datasets.py --stage1 stage1_output.jsonl --positive stage3_output.jsonl --negative negative_samples.jsonl --output merged.jsonl
        """
    )
    
    parser.add_argument('--stage1', type=Path, default=None,
                       help='Path to Stage 1 output (rule-based samples). Optional - if provided, will be combined with Stage 3 as positive samples.')
    parser.add_argument('--positive', type=Path, required=True,
                       help='Path to positive samples (Stage 3 augmented output)')
    parser.add_argument('--negative', type=Path, required=True,
                       help='Path to negative samples')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for merged dataset')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Optional maximum total dataset size. If not provided, uses all available samples while respecting negative ratio.')
    parser.add_argument('--negative_ratio', type=float, default=0.20,
                       help='Target ratio of negative samples (default: 0.20). Ignored if --use_all_samples is set.')
    parser.add_argument('--use_all_samples', action='store_true',
                       help='Use ALL available samples from all sources, ignoring negative ratio. Overrides --negative_ratio.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 4 DATASET MERGER")
    print("="*80)
    if args.stage1:
        print(f"Stage 1 samples: {args.stage1}")
    print(f"Stage 3 samples: {args.positive}")
    print(f"Negative samples: {args.negative}")
    print(f"Output file: {args.output}")
    if args.target_size:
        print(f"Target max size: {args.target_size:,}")
    else:
        print(f"Target max size: None (using all available samples)")
    if args.use_all_samples:
        print(f"Mode: Use ALL available samples (ignoring ratio)")
    else:
        print(f"Negative ratio: {args.negative_ratio:.1%}")
    
    # Load datasets
    stage3_samples = load_dataset(args.positive)
    
    # Load Stage 1 if provided
    stage1_samples = []
    if args.stage1:
        stage1_samples = load_dataset(args.stage1)
        print(f"\nCombining Stage 1 + Stage 3 as positive samples...")
        print(f"  Stage 1: {len(stage1_samples):,} samples")
        print(f"  Stage 3: {len(stage3_samples):,} samples")
        positive_samples = stage1_samples + stage3_samples
        print(f"  Combined positive: {len(positive_samples):,} samples")
    else:
        positive_samples = stage3_samples
        print(f"\nUsing Stage 3 only as positive samples: {len(positive_samples):,} samples")
    
    negative_samples = load_dataset(args.negative)
    
    # Merge datasets
    merged_samples = merge_datasets(
        positive_samples,
        negative_samples,
        target_size=args.target_size,
        negative_ratio=args.negative_ratio,
        use_all_samples=args.use_all_samples,
        random_seed=args.seed
    )
    
    # Generate statistics
    stats = generate_statistics(merged_samples)
    
    # Save merged dataset
    print(f"\nSaving merged dataset to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in merged_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats_file = args.output.with_suffix('.stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to: {stats_file}")
    
    # Print statistics
    print_statistics(stats)
    
    print(f"\nMerge complete!")
    print(f"Output: {args.output}")
    print(f"Ready for curation with curate_cim_dataset_ftv2.py")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
