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
from typing import List, Dict, Any, Optional, Tuple
import math
from collections import defaultdict
import random
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import time
import re


# Importance weights for stratification
IMPORTANCE_WEIGHTS = {
    'difficulty': {
        'SIMPLE': 0.30,
        'MEDIUM': 0.40,
        'HARD': 0.20,
        'VERY_HARD': 0.10
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
    },
    
    'question_tone': {
        'INTERROGATIVE': 0.70,
        'DESCRIPTIVE': 0.15,
        'IMPERATIVE': 0.10,
        'NARRATIVE': 0.05
    }
}

COMPLEXITY_BONUS = {
    'spatial_complexity': {
        'NONE': 1.0,
        'BASIC': 1.15,
        'INTERMEDIATE': 1.35,
        'ADVANCED': 1.6
    },
    'schema_complexity': {
        'SINGLE_TABLE': 1.0,
        'SINGLE_SCHEMA': 1.2,
        'MULTI_SCHEMA': 1.4
    },
    'complexity_level': {
        'A': 1.0,
        'B': 1.2,
        'C': 1.4
    },
    'join_count': {
        '0': 1.0,
        '1': 1.2,
        '2+': 1.4
    },
    'function_count': {
        '0': 1.0,
        '1': 1.15,
        '2': 1.25,
        '3+': 1.4
    }
}

ALL_SQL_TYPES = [
    'SIMPLE_SELECT',
    'AGGREGATION',
    'SPATIAL_JOIN',
    'SPATIAL_MEASUREMENT',
    'SPATIAL_PROCESSING',
    'MULTI_JOIN',
    'NESTED_QUERY',
    'SPATIAL_CLUSTERING',
    'RASTER_VECTOR',
    'WINDOW_FUNCTION'
]

SQL_TYPE_MIN_DEFAULTS = {sql_type: 1 for sql_type in ALL_SQL_TYPES}

QUERY_COMPLEXITY_TARGETS = {
    'EASY': 0.30,
    'MEDIUM': 0.60,
    'HARD': 0.10
}

SPATIAL_COMPLEXITY_TARGETS = {
    'NONE': 0.30,
    'BASIC': 0.60,
    'INTERMEDIATE': 0.10
}

SCHEMA_COMPLEXITY_TARGETS = {
    'MULTI_SCHEMA': 0.40,
    'SINGLE_SCHEMA': 0.60
}


def build_ratio_targets(
    base_targets: Dict[str, float],
    available_values: set
) -> Dict[str, float]:
    filtered = {
        key.upper(): value
        for key, value in base_targets.items()
        if key.upper() in available_values and value > 0
    }
    total = sum(filtered.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in filtered.items()}


def calculate_sample_importance(sample: Dict[str, Any]) -> float:
    """Calculate importance score for weighted sampling."""
    difficulty = sample.get('difficulty_level', 'MEDIUM')
    sql_type = sample.get('sql_type', 'SIMPLE_SELECT')
    
    difficulty_weight = IMPORTANCE_WEIGHTS['difficulty'].get(difficulty, 0.25)
    sql_type_weight = IMPORTANCE_WEIGHTS['sql_type'].get(sql_type, 1.0)
    
    spatial = sample.get('spatial_complexity', 'NONE')
    schema = sample.get('schema_complexity', 'SINGLE_TABLE')
    level = sample.get('complexity_level', 'A')
    join_count = str(sample.get('join_count', '0'))
    function_count = str(sample.get('function_count', '0'))
    
    spatial_bonus = COMPLEXITY_BONUS['spatial_complexity'].get(spatial, 1.0)
    schema_bonus = COMPLEXITY_BONUS['schema_complexity'].get(schema, 1.0)
    level_bonus = COMPLEXITY_BONUS['complexity_level'].get(level, 1.0)
    join_bonus = COMPLEXITY_BONUS['join_count'].get(join_count, 1.0)
    function_bonus = COMPLEXITY_BONUS['function_count'].get(function_count, 1.0)
    
    importance = (
        difficulty_weight
        * sql_type_weight
        * spatial_bonus
        * schema_bonus
        * level_bonus
        * join_bonus
        * function_bonus
    )
    
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


def ensure_difficulty_metadata(samples: List[Dict[str, Any]]) -> None:
    """Ensure each sample has a top-level difficulty_level for stratification."""
    for sample in samples:
        difficulty_level = sample.get('difficulty_level')
        if not difficulty_level:
            difficulty_info = sample.get('difficulty') or {}
            difficulty_level = difficulty_info.get('overall_difficulty')
        if not difficulty_level:
            difficulty_level = sample.get('query_complexity')
        if not difficulty_level:
            difficulty_level = 'MEDIUM'
        sample['difficulty_level'] = str(difficulty_level).upper()


SINGLE_SCHEMA_SET = {'SINGLE_TABLE', 'SINGLE_SCHEMA'}
MULTI_SCHEMA_SET = {'MULTI_SCHEMA'}

COMPLEXITY_FIELDS = [
    'spatial_complexity',
    'schema_complexity',
    'complexity_level',
    'function_count',
    'join_count',
    'table_count',
    'complexity_score',
    'query_complexity'
]


def ensure_sql_complexity_metadata(samples: List[Dict[str, Any]]) -> None:
    """Ensure samples expose spatial/schema complexity derived from SQL text."""
    for sample in samples:
        difficulty_info = sample.get('difficulty') or {}
        fallback_map = {
            'spatial_complexity': difficulty_info.get('spatial_complexity'),
            'schema_complexity': difficulty_info.get('schema_complexity'),
            'complexity_level': difficulty_info.get('complexity_level'),
            'function_count': difficulty_info.get('function_count'),
            'join_count': difficulty_info.get('join_count'),
            'table_count': difficulty_info.get('table_count'),
            'complexity_score': difficulty_info.get('complexity_score'),
            'query_complexity': difficulty_info.get('query_complexity'),
            'difficulty_level': difficulty_info.get('overall_difficulty')
        }
        for field, value in fallback_map.items():
            if value and sample.get(field) in (None, '', 'UNKNOWN'):
                sample[field] = value
        
        needs_update = any(sample.get(field) in (None, '', 'UNKNOWN') for field in COMPLEXITY_FIELDS)
        if not needs_update:
            continue
        sql_query = sample.get('sql_postgis') or sample.get('sql')
        if not sql_query:
            continue
        dims = calculate_difficulty_dimensions(sql_query)
        sample.setdefault('spatial_complexity', dims['spatial_complexity'])
        sample.setdefault('schema_complexity', dims['schema_complexity'])
        sample.setdefault('complexity_level', dims['complexity_level'])
        sample.setdefault('function_count', dims['function_count'])
        sample.setdefault('join_count', dims['join_count'])
        sample.setdefault('table_count', dims['table_count'])
        sample.setdefault('complexity_score', dims['complexity_score'])
        sample.setdefault('query_complexity', dims['query_complexity'])
        sample.setdefault('difficulty_level', dims['overall_difficulty'])


def ensure_question_tone_metadata(samples: List[Dict[str, Any]]) -> None:
    """Ensure question tone metadata is present and normalized."""
    for sample in samples:
        tone = sample.get('question_tone')
        if not tone:
            tone = 'INTERROGATIVE'
        sample['question_tone'] = str(tone).upper()


def enforce_complexity_target(
    selected: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    min_average: float,
    field_targets: Optional[Dict[str, Dict[str, float]]] = None
) -> List[Dict[str, Any]]:
    """Ensure the average complexity score and distributions meet target ratios."""
    if not selected:
        return selected
    
    def comp(sample):
        return float(sample.get('complexity_score', 0) or 0)
    
    def current_average(items):
        return sum(comp(s) for s in items) / len(items) if items else 0
    
    avg_score = current_average(selected)
    selected_ids = {id(s) for s in selected}
    candidate_pool = [
        (comp(sample), sample)
        for sample in pool
        if id(sample) not in selected_ids and comp(sample) > 0
    ]
    candidate_pool.sort(key=lambda x: x[0], reverse=True)
    low_selected = sorted(selected, key=comp)
    
    while min_average and avg_score < min_average and candidate_pool and low_selected:
        candidate_score, candidate = candidate_pool.pop(0)
        to_remove = low_selected.pop(0)
        if candidate_score <= comp(to_remove):
            break
        selected.remove(to_remove)
        selected.append(candidate)
        selected_ids.remove(id(to_remove))
        selected_ids.add(id(candidate))
        low_selected.append(candidate)
        low_selected.sort(key=comp)
        avg_score = current_average(selected)
    
    if min_average and avg_score < min_average:
        print(f"Warning: Average complexity score {avg_score:.2f} below target {min_average:.2f}.")
    
    if not field_targets:
        return selected
    
    def enforce_field(field: str, targets: Dict[str, float]):
        if not targets:
            return
        total = len(selected)
        current_counts = defaultdict(int)
        for sample in selected:
            key = (sample.get(field) or 'UNKNOWN').upper()
            current_counts[key] += 1
        for category, ratio in targets.items():
            desired = math.ceil(total * ratio)
            current = current_counts.get(category.upper(), 0)
            if current >= desired:
                continue
            needed = desired - current
            additions = [
                (calculate_sample_importance(sample), sample)
                for sample in pool
                if (sample.get(field) or 'UNKNOWN').upper() == category.upper()
                and id(sample) not in selected_ids
            ]
            additions.sort(key=lambda x: x[0], reverse=True)
            if not additions:
                print(f"Warning: No candidates available for {field}={category}.")
                continue
            removals = [
                (calculate_sample_importance(sample), sample)
                for sample in selected
                if (sample.get(field) or 'UNKNOWN').upper() != category.upper()
            ]
            removals.sort(key=lambda x: x[0])
            while needed > 0 and additions and removals:
                _, candidate = additions.pop(0)
                _, to_remove = removals.pop(0)
                selected.remove(to_remove)
                selected.append(candidate)
                selected_ids.remove(id(to_remove))
                selected_ids.add(id(candidate))
                needed -= 1
            if needed > 0:
                print(f"Warning: Unable to meet target ratio for {field}={category}.")
    
    for field, targets in field_targets.items():
        enforce_field(field, targets)
    
    return selected


def calculate_distribution(samples: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts = defaultdict(int)
    for sample in samples:
        value = (sample.get(field) or 'UNKNOWN')
        if isinstance(value, str):
            value = value.upper()
        counts[value] += 1
    return counts


def validate_ratios(
    counts: Dict[str, int],
    total: int,
    targets: Dict[str, float],
    tolerance: float,
    field_name: str
) -> Tuple[bool, str]:
    if total == 0:
        return False, f"No samples to validate for {field_name}"
    for category, target_ratio in targets.items():
        actual_ratio = counts.get(category.upper(), 0) / total
        if abs(actual_ratio - target_ratio) > tolerance:
            return False, f"{field_name} ratio for {category} is {actual_ratio:.2f}, outside tolerance of target {target_ratio:.2f}"
    return True, ""


def validate_sql_type_counts(counts: Dict[str, int], targets: Dict[str, int]) -> Tuple[bool, str]:
    for sql_type, min_count in targets.items():
        if counts.get(sql_type, 0) < min_count:
            return False, f"SQL type {sql_type} has {counts.get(sql_type, 0)} samples, below target {min_count}"
    return True, ""


def enforce_tone_distribution(
    selected: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    primary_tone: str,
    min_primary_ratio: float,
    max_variants: Optional[int]
) -> List[Dict[str, Any]]:
    """Ensure the benchmark meets primary/variant tone targets."""
    if not selected:
        return selected
    
    primary = (primary_tone or 'INTERROGATIVE').upper()
    min_ratio = min(max(min_primary_ratio or 0.0, 0.0), 1.0)
    
    def tone(sample):
        return (sample.get('question_tone') or 'UNKNOWN').upper()
    
    def importance(sample):
        return calculate_sample_importance(sample)
    
    total = len(selected)
    required_primary = math.ceil(total * min_ratio)
    
    selected_ids = {id(s) for s in selected}
    
    def candidates_for_primary(exclude_ids):
        candidates = [
            (importance(sample), sample)
            for sample in pool
            if tone(sample) == primary and id(sample) not in exclude_ids
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates
    
    def replace_variants(low_variants, replacement_candidates, replacements_needed):
        replacements_done = 0
        while replacements_needed > 0 and low_variants and replacement_candidates:
            _, candidate = replacement_candidates.pop(0)
            to_remove = low_variants.pop(0)
            selected.remove(to_remove)
            selected.append(candidate)
            selected_ids.remove(id(to_remove))
            selected_ids.add(id(candidate))
            replacements_needed -= 1
            replacements_done += 1
        return replacements_needed
    
    # Ensure minimum primary ratio
    current_primary = [s for s in selected if tone(s) == primary]
    primary_shortfall = required_primary - len(current_primary)
    if primary_shortfall > 0:
        variant_samples = [s for s in selected if tone(s) != primary]
        low_variants = sorted(variant_samples, key=importance)
        replacement_candidates = candidates_for_primary(selected_ids)
        remaining = replace_variants(low_variants, replacement_candidates, primary_shortfall)
        if remaining > 0:
            print("Warning: Unable to reach desired primary tone ratio.")
    
    # Enforce maximum number of variant tones
    if max_variants is not None and max_variants >= 0:
        variant_samples = [s for s in selected if tone(s) != primary]
        excess = len(variant_samples) - max_variants
        if excess > 0:
            low_variants = sorted(variant_samples, key=importance)
            replacement_candidates = candidates_for_primary(selected_ids)
            remaining = replace_variants(low_variants, replacement_candidates, excess)
            if remaining > 0:
                print("Warning: Could not reduce variant tones to desired quota.")
    
    return selected


def enforce_schema_balance(
    selected: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    max_single_ratio: float
) -> List[Dict[str, Any]]:
    """Ensure that single schema samples do not exceed the specified ratio."""
    if not selected or max_single_ratio >= 1.0:
        return selected
    
    total = len(selected)
    max_single = math.floor(total * max_single_ratio)
    
    def schema(sample):
        return (sample.get('schema_complexity') or 'UNKNOWN').upper()
    
    def importance(sample):
        return calculate_sample_importance(sample)
    
    single_samples = [s for s in selected if schema(s) in SINGLE_SCHEMA_SET]
    multi_samples = [s for s in selected if schema(s) in MULTI_SCHEMA_SET]
    
    if len(single_samples) <= max_single:
        return selected
    
    needed_multi = len(single_samples) - max_single
    selected_ids = {id(s) for s in selected}
    
    candidates = [
        (importance(sample), sample)
        for sample in pool
        if schema(sample) in MULTI_SCHEMA_SET and id(sample) not in selected_ids
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    if not candidates:
        print("Warning: No MULTI_SCHEMA candidates available to balance schema complexity.")
        return selected
    
    removable = sorted(single_samples, key=importance)
    
    while needed_multi > 0 and candidates and removable:
        _, candidate = candidates.pop(0)
        to_remove = removable.pop(0)
        selected.remove(to_remove)
        selected.append(candidate)
        selected_ids.remove(id(to_remove))
        selected_ids.add(id(candidate))
        needed_multi -= 1
    
    if needed_multi > 0:
        print("Warning: Could not achieve desired schema distribution (insufficient MULTI_SCHEMA samples).")
    
    return selected


def enforce_sql_type_targets(
    selected: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    targets: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Ensure minimum counts for specified SQL types."""
    if not selected or not targets:
        return selected
    
    from collections import defaultdict
    
    selected_ids = {id(s) for s in selected}
    selected_by_type = defaultdict(list)
    for sample in selected:
        stype = sample.get('sql_type', 'UNKNOWN')
        selected_by_type[stype].append(sample)
    
    pool_by_type = defaultdict(list)
    for sample in pool:
        stype = sample.get('sql_type', 'UNKNOWN')
        if id(sample) in selected_ids:
            continue
        pool_by_type[stype].append((calculate_sample_importance(sample), sample))
    for stype in pool_by_type:
        pool_by_type[stype].sort(key=lambda x: x[0], reverse=True)
    
    def removable_samples(skip_type: str):
        removable = []
        for sample in selected:
            stype = sample.get('sql_type', 'UNKNOWN')
            if stype == skip_type:
                continue
            if stype in targets:
                if len(selected_by_type.get(stype, [])) <= targets[stype]:
                    continue
            removable.append((calculate_sample_importance(sample), sample))
        removable.sort(key=lambda x: x[0])
        return removable
    
    for sql_type, min_count in targets.items():
        if min_count <= 0:
            continue
        current = len(selected_by_type.get(sql_type, []))
        available = len(selected_by_type.get(sql_type, [])) + len(pool_by_type.get(sql_type, []))
        if available < min_count:
            print(f"Warning: Not enough samples of SQL type '{sql_type}' to meet target ({available} available, target {min_count}).")
            continue
        
        needed = min_count - current
        if needed <= 0:
            continue
        
        additions = pool_by_type.get(sql_type, [])
        removals = removable_samples(sql_type)
        
        while needed > 0 and additions and removals:
            _, candidate = additions.pop(0)
            _, to_remove = removals.pop(0)
            
            selected.remove(to_remove)
            selected.append(candidate)
            
            selected_ids.remove(id(to_remove))
            selected_ids.add(id(candidate))
            
            removed_type = to_remove.get('sql_type', 'UNKNOWN')
            if removed_type in selected_by_type:
                try:
                    selected_by_type[removed_type].remove(to_remove)
                except ValueError:
                    pass
            selected_by_type[sql_type].append(candidate)
            
            needed -= 1
            
            # Recompute removals when necessary
            if needed > 0 and not removals:
                removals = removable_samples(sql_type)
        
        if needed > 0:
            print(f"Warning: Could not fully meet SQL type target for '{sql_type}'. Shortfall: {needed}")
    
    return selected


def apply_sampling_pipeline(
    quality_filtered: List[Dict[str, Any]],
    args,
    target_size: int,
    seed: int,
    sql_type_targets: Dict[str, int],
    field_targets: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    tone_weights = None
    tone_default_weight = None
    if args.stratify_by == 'question_tone':
        primary_key = (args.primary_tone or 'INTERROGATIVE').upper()
        tone_weights = {primary_key: max(args.primary_tone_ratio, 0.5)}
        tone_default_weight = max(0.01, (1 - args.primary_tone_ratio) / 4)
    
    samples = weighted_stratified_sample(
        quality_filtered,
        target_size,
        seed,
        stratify_by=args.stratify_by,
        custom_weights=tone_weights,
        default_weight=tone_default_weight
    )
    
    samples = enforce_tone_distribution(
        samples,
        quality_filtered,
        args.primary_tone,
        args.primary_tone_ratio,
        args.tone_variant_quota
    )
    
    samples = enforce_complexity_target(
        samples,
        quality_filtered,
        args.min_complexity_score
    )
    
    samples = enforce_sql_type_targets(
        samples,
        quality_filtered,
        sql_type_targets
    )
    
    samples = enforce_schema_balance(
        samples,
        quality_filtered,
        args.max_single_schema_ratio
    )
    
    samples = enforce_complexity_target(
        samples,
        quality_filtered,
        args.min_complexity_score,
        field_targets
    )

    samples = enforce_sql_type_targets(
        samples,
        quality_filtered,
        sql_type_targets
    )
    
    return samples


def validate_sampling_result(
    samples: List[Dict[str, Any]],
    sql_targets: Dict[str, int],
    tolerance: float,
    field_targets: Dict[str, Dict[str, float]]
) -> Tuple[bool, str]:
    total = len(samples)
    if total == 0:
        return False, "No samples generated."
    
    sql_counts = calculate_distribution(samples, 'sql_type')
    ok, msg = validate_sql_type_counts(sql_counts, sql_targets)
    if not ok:
        return False, msg
    
    for field, targets in field_targets.items():
        if not targets:
            continue
        counts = calculate_distribution(samples, field)
        ok, msg = validate_ratios(counts, total, targets, tolerance, field)
        if not ok:
            return False, msg
    
    return True, ""


def weighted_stratified_sample(
    samples: List[Dict[str, Any]], 
    target_size: int,
    seed: int = 42,
    stratify_by: str = 'difficulty',
    custom_weights: Optional[Dict[str, float]] = None,
    default_weight: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Perform weighted stratified sampling by difficulty or SQL type."""
    print(f"\nPerforming weighted stratified sampling (target: {target_size})...")
    stratify_by = stratify_by.lower()
    if stratify_by not in {'difficulty', 'sql_type', 'question_tone'}:
        print(f"Warning: Unsupported stratify_by='{stratify_by}', defaulting to difficulty")
        stratify_by = 'difficulty'
    
    random.seed(seed)
    
    group_map = defaultdict(list)
    if stratify_by == 'difficulty':
        for sample in samples:
            difficulty = sample.get('difficulty_level', 'MEDIUM') or 'MEDIUM'
            group_map[difficulty].append(sample)
        weight_map = custom_weights or IMPORTANCE_WEIGHTS['difficulty']
        default_weight = 0.25 if default_weight is None else default_weight
    elif stratify_by == 'sql_type':
        for sample in samples:
            sql_type = sample.get('sql_type', 'UNKNOWN') or 'UNKNOWN'
            group_map[sql_type].append(sample)
        weight_map = custom_weights or IMPORTANCE_WEIGHTS['sql_type']
        default_weight = 1.0 if default_weight is None else default_weight
    elif stratify_by == 'question_tone':
        for sample in samples:
            tone = (sample.get('question_tone') or 'UNKNOWN').upper()
            group_map[tone].append(sample)
        weight_map = custom_weights or IMPORTANCE_WEIGHTS.get('question_tone', {})
        default_weight = 0.05 if default_weight is None else default_weight
    else:
        for sample in samples:
            difficulty = sample.get('difficulty_level', 'MEDIUM') or 'MEDIUM'
            group_map[difficulty].append(sample)
        weight_map = custom_weights or IMPORTANCE_WEIGHTS['difficulty']
        default_weight = 0.25 if default_weight is None else default_weight
    
    selected = []

    if not group_map:
        return selected

    available_groups = list(group_map.keys())
    base_weights = {}
    total_weight = 0.0
    for group_name in available_groups:
        weight = weight_map.get(group_name, default_weight)
        base_weights[group_name] = weight
        total_weight += weight
    
    if total_weight == 0:
        total_weight = len(available_groups)
        for group_name in available_groups:
            base_weights[group_name] = 1.0
    
    normalized_weights = {
        group_name: base_weights[group_name] / total_weight
        for group_name in available_groups
    }
    
    allocations = {}
    for group_name in available_groups:
        desired_count = max(1, int(round(target_size * normalized_weights[group_name])))
        allocations[group_name] = min(desired_count, len(group_map[group_name]))
    
    allocated_total = sum(allocations.values())
    
    if allocated_total > target_size:
        while allocated_total > target_size:
            reducible = [
                d for d in available_groups
                if allocations[d] > 1
            ]
            if not reducible:
                break
            # Reduce from the group with the largest allocation first
            target_difficulty = max(reducible, key=lambda d: allocations[d])
            allocations[target_difficulty] -= 1
            allocated_total -= 1
    elif allocated_total < target_size:
        while allocated_total < target_size:
            expandable = [
                d for d in available_groups
                if allocations[d] < len(group_map[d])
            ]
            if not expandable:
                break
            target_difficulty = max(expandable, key=lambda d: normalized_weights[d])
            allocations[target_difficulty] += 1
            allocated_total += 1
    
    selected_ids = set()
    
    for group_name in available_groups:
        group_samples = group_map[group_name]
        group_target = allocations.get(group_name, 0)
        if group_target <= 0:
            continue
        
        weighted_samples = [
            (sample, calculate_sample_importance(sample))
            for sample in group_samples
        ]
        
        weighted_samples.sort(key=lambda x: x[1], reverse=True)
        
        for sample, _ in weighted_samples[:group_target]:
            selected.append(sample)
            selected_ids.add(id(sample))
    
    if len(selected) < target_size:
        remaining_candidates = [
            (sample, calculate_sample_importance(sample))
            for sample in samples
            if id(sample) not in selected_ids
        ]
        remaining_candidates.sort(key=lambda x: x[1], reverse=True)
        for sample, _ in remaining_candidates:
            selected.append(sample)
            selected_ids.add(id(sample))
            if len(selected) == target_size:
                break

    if len(selected) > target_size:
        weighted_all = [
            (sample, calculate_sample_importance(sample))
            for sample in selected
        ]
        weighted_all.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in weighted_all[:target_size]]
        selected_ids = {id(s) for s in selected}
    
    print(f"Selected {len(selected)} samples")
    
    dist = defaultdict(int)
    for s in selected:
        if stratify_by == 'difficulty':
            key = s.get('difficulty_level', 'UNKNOWN')
        elif stratify_by == 'sql_type':
            key = s.get('sql_type', 'UNKNOWN')
        else:
            key = (s.get('question_tone') or 'UNKNOWN').upper()
        dist[key] += 1
    
    print(f"\n{stratify_by.title()} distribution:")
    for key, count in dist.items():
        pct = count / len(selected) * 100
        print(f"  {key:18s}: {count:3d} ({pct:5.1f}%)")
    
    extra_fields = [
        ("Spatial complexity", 'spatial_complexity'),
        ("Schema complexity", 'schema_complexity'),
        ("Complexity level", 'complexity_level'),
        ("Join count", 'join_count'),
        ("Function count", 'function_count'),
        ("Question tone", 'question_tone')
    ]
    
    for label, field in extra_fields:
        summary = defaultdict(int)
        for sample in selected:
            summary[str(sample.get(field, 'UNKNOWN'))] += 1
        print(f"\n{label} distribution:")
        for key, count in summary.items():
            pct = count / len(selected) * 100
            print(f"  {key:18s}: {count:3d} ({pct:5.1f}%)")
    
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


def remove_limit_clause(sql: str) -> str:
    """Remove LIMIT clause from SQL query to get full result set."""
    import re
    # Remove LIMIT clause (handles various formats)
    sql = re.sub(r'\s+LIMIT\s+\d+\s*;?\s*$', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+LIMIT\s+\d+\s+', ' ', sql, flags=re.IGNORECASE)
    return sql.strip()


def execute_query(query: str, engine, timeout: int = 30) -> Dict[str, Any]:
    """Execute SQL query and capture results."""
    start_time = time.time()
    
    # Remove LIMIT clause to get full result set for ground truth
    query_no_limit = remove_limit_clause(query)
    
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {timeout * 1000};"))
            result = conn.execute(text(query_no_limit))
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
        
        # Remove LIMIT clause from ground truth SQL
        sql_query_no_limit = remove_limit_clause(sql_query) if sql_query else ''
        
        exec_result = {
            'success': None, 
            'result': None, 
            'row_count': None,
            'execution_time': None, 
            'error': None
        }
        
        if not skip_execution and sql_query_no_limit and engine:
            exec_result = execute_query(sql_query_no_limit, engine, timeout=30)
            if exec_result['success']:
                executed += 1
            else:
                failed += 1
        
        # Calculate difficulty dimensions from SQL
        difficulty_dims = calculate_difficulty_dimensions(sql_query)
        
        difficulty_level = sample.get('difficulty_level') or difficulty_dims['overall_difficulty']
        query_complexity = sample.get('query_complexity') or difficulty_dims['query_complexity']
        spatial_complexity = sample.get('spatial_complexity') or difficulty_dims['spatial_complexity']
        schema_complexity = sample.get('schema_complexity') or difficulty_dims['schema_complexity']
        complexity_level = sample.get('complexity_level') or difficulty_dims['complexity_level']
        complexity_score = sample.get('complexity_score') or difficulty_dims['complexity_score']
        function_count = str(sample.get('function_count') or difficulty_dims['function_count'])
        join_count = str(sample.get('join_count') or difficulty_dims['join_count'])
        table_count = sample.get('table_count') or difficulty_dims['table_count']
        question_tone = (sample.get('question_tone') or 'UNKNOWN').upper()
        
        benchmark_item = {
            'benchmark_id': idx,
            'original_id': sample.get('id', f'sample_{idx}'),
            'difficulty_level': difficulty_level,
            'query_complexity': query_complexity,
            'spatial_complexity': spatial_complexity,
            'schema_complexity': schema_complexity,
            'complexity_level': complexity_level,
            'complexity_score': complexity_score,
            'sql_type': sample.get('sql_type', 'UNKNOWN'),
            'spatial_functions': difficulty_dims['spatial_functions'],
            'spatial_function_count': difficulty_dims['spatial_function_count'],
            'function_count': function_count,
            'join_count': join_count,
            'table_count': table_count,
            'question_tone': question_tone,
            'importance_score': calculate_sample_importance(sample),
            'executable': exec_result['success'],
            'execution_time': exec_result['execution_time'],
            'question': sample.get('question', ''),
            'instruction': sample.get('instruction', ''),
            'sql_postgis': sql_query_no_limit,
            'expected_result': exec_result['result'],
            'expected_row_count': exec_result['row_count'],
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


def save_benchmark(
    benchmark: List[Dict[str, Any]],
    output_file: Path,
    metadata_context: Optional[Dict[str, Any]] = None
):
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
        'question_tone_distribution': {},
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
        tone = (item.get('question_tone') or 'UNKNOWN').upper()
        
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
        metadata['question_tone_distribution'][tone] = \
            metadata['question_tone_distribution'].get(tone, 0) + 1
    
    if metadata_context:
        primary_tone = (metadata_context.get('primary_tone') or 'UNKNOWN').upper()
        primary_count = metadata['question_tone_distribution'].get(primary_tone, 0)
        variant_count = metadata['benchmark_size'] - primary_count
        metadata['question_tone_distribution']['__PRIMARY_TONE__'] = primary_tone
        metadata['question_tone_distribution']['__PRIMARY_COUNT__'] = primary_count
        metadata['question_tone_distribution']['__VARIANT_COUNT__'] = variant_count
        metadata['sampling_attempts'] = metadata_context.get('sampling_attempts')
        metadata['final_candidate_size'] = metadata_context.get('final_target_size')
    
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
    
    parser.add_argument(
        '--stratify_by',
        type=str,
        choices=['difficulty', 'sql_type', 'question_tone'],
        default='difficulty',
        help='Field to stratify by (difficulty, sql_type, or question_tone)'
    )
    
    parser.add_argument(
        '--primary_tone',
        type=str,
        default='INTERROGATIVE',
        help='Primary question tone (default: INTERROGATIVE)'
    )
    
    parser.add_argument(
        '--tone_variant_quota',
        type=int,
        default=8,
        help='Maximum number of non-primary tone samples to keep (default: 8)'
    )
    
    parser.add_argument(
        '--primary_tone_ratio',
        type=float,
        default=0.70,
        help='Minimum ratio of samples that must use the primary tone (default: 0.70)'
    )
    
    parser.add_argument(
        '--min_complexity_score',
        type=float,
        default=1.7,
        help='Minimum average complexity score for the benchmark (default: 1.7)'
    )
    
    parser.add_argument(
        '--min_avg_complexity',
        type=float,
        dest='min_complexity_score',
        help='Alias for --min_complexity_score (minimum average complexity score)'
    )
    
    parser.add_argument(
        '--min_spatial_join',
        type=int,
        default=12,
        help='Minimum number of SPATIAL_JOIN samples (default: 12)'
    )
    
    parser.add_argument(
        '--min_spatial_measurement',
        type=int,
        default=12,
        help='Minimum number of SPATIAL_MEASUREMENT samples (default: 12)'
    )
    
    parser.add_argument(
        '--max_single_schema_ratio',
        type=float,
        default=0.95,
        help='Maximum ratio of SINGLE_TABLE/SINGLE_SCHEMA samples (default: 0.95)'
    )
    
    parser.add_argument(
        '--max_sampling_attempts',
        type=int,
        default=8,
        help='Maximum sampling attempts to satisfy distribution criteria (default: 8)'
    )
    
    parser.add_argument(
        '--size_increment',
        type=int,
        default=10,
        help='Increase in sample size per retry when criteria unmet (default: 10)'
    )
    
    parser.add_argument(
        '--distribution_tolerance',
        type=float,
        default=0.08,
        help='Allowed deviation for ratio-based targets (default: 0.08)'
    )
    
    parser.add_argument(
        '--easy_mode',
        action='store_true',
        help='Generate easy benchmark (80%% SIMPLE_SELECT, low complexity, adds "_easy" suffix to output)'
    )
    
    args = parser.parse_args()
    
    # Apply easy mode defaults
    if args.easy_mode:
        print("\n" + "="*70)
        print("EASY MODE ENABLED")
        print("="*70)
        print("Adjusting parameters for easy benchmark generation:")
        print("  - Stratifying by sql_type")
        print("  - Min complexity score: 0.5 (was {})".format(args.min_complexity_score))
        print("  - Min spatial join: 3 (was {})".format(args.min_spatial_join))
        print("  - Min spatial measurement: 3 (was {})".format(args.min_spatial_measurement))
        print("  - Adding '_easy' suffix to output filename")
        
        args.stratify_by = 'sql_type'
        args.min_complexity_score = 0.5
        args.min_spatial_join = 3
        args.min_spatial_measurement = 3
        
        # Add "_easy" suffix to output filename
        if not str(args.output).endswith('_easy.jsonl'):
            output_stem = args.output.stem
            if not output_stem.endswith('_easy'):
                args.output = args.output.parent / f"{output_stem}_easy{args.output.suffix}"
    
    args.primary_tone_ratio = min(max(args.primary_tone_ratio, 0.0), 1.0)
    args.max_single_schema_ratio = min(max(args.max_single_schema_ratio, 0.0), 1.0)
    args.size_increment = max(1, args.size_increment)
    args.max_sampling_attempts = max(1, args.max_sampling_attempts)
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    samples = load_dataset(args.input)
    
    if len(samples) == 0:
        print("Error: No samples loaded")
        sys.exit(1)
    
    if args.stratify_by == 'difficulty':
        ensure_difficulty_metadata(samples)
    
    ensure_sql_complexity_metadata(samples)
    ensure_question_tone_metadata(samples)
    
    print(f"\nFiltering to high-quality samples (quality_score >= 0.70, no_error = True)...")
    quality_filtered = [
        s for s in samples 
        if s.get('quality_score', 0) >= 0.70 and s.get('no_error', True)
    ]
    
    print(f"Quality filtered: {len(quality_filtered)} samples")
    
    if len(quality_filtered) < args.size:
        print(f"Warning: Only {len(quality_filtered)} samples available, using all")
        args.size = len(quality_filtered)
    
    available_sql_types = {str(s.get('sql_type', 'UNKNOWN')).upper() for s in quality_filtered}
    
    # Override SQL type targets for easy mode
    if args.easy_mode:
        sql_type_targets = {
            'SIMPLE_SELECT': int(args.size * 0.80),  # 80% SIMPLE_SELECT
            'AGGREGATION': max(int(args.size * 0.05), 1),
            'SPATIAL_JOIN': max(args.min_spatial_join, 1),
            'SPATIAL_MEASUREMENT': max(args.min_spatial_measurement, 1),
            'MULTI_JOIN': max(int(args.size * 0.02), 1),
            'NESTED_QUERY': max(int(args.size * 0.02), 1),
        }
        # Filter to only available types
        sql_type_targets = {k: v for k, v in sql_type_targets.items() if k in available_sql_types}
        print(f"\nEasy mode SQL type targets: {sql_type_targets}")
    else:
        sql_type_targets = {
            sql_type: max(SQL_TYPE_MIN_DEFAULTS.get(sql_type, 1), 1)
            for sql_type in available_sql_types
        }
    
    if args.min_spatial_join:
        if 'SPATIAL_JOIN' in sql_type_targets:
            sql_type_targets['SPATIAL_JOIN'] = max(sql_type_targets['SPATIAL_JOIN'], args.min_spatial_join)
        else:
            print("Warning: No SPATIAL_JOIN samples available to satisfy min_spatial_join target.")
    if args.min_spatial_measurement:
        if 'SPATIAL_MEASUREMENT' in sql_type_targets:
            sql_type_targets['SPATIAL_MEASUREMENT'] = max(sql_type_targets['SPATIAL_MEASUREMENT'], args.min_spatial_measurement)
        else:
            print("Warning: No SPATIAL_MEASUREMENT samples available to satisfy min_spatial_measurement target.")
    
    available_query_complexities = {str(s.get('query_complexity', 'UNKNOWN')).upper() for s in quality_filtered}
    available_spatial_complexities = {str(s.get('spatial_complexity', 'UNKNOWN')).upper() for s in quality_filtered}
    available_schema_complexities = {str(s.get('schema_complexity', 'UNKNOWN')).upper() for s in quality_filtered}
    
    field_targets = {
        'query_complexity': build_ratio_targets(QUERY_COMPLEXITY_TARGETS, available_query_complexities),
        'spatial_complexity': build_ratio_targets(SPATIAL_COMPLEXITY_TARGETS, available_spatial_complexities),
        'schema_complexity': build_ratio_targets(SCHEMA_COMPLEXITY_TARGETS, available_schema_complexities)
    }
    
    benchmark_samples = None
    current_size = args.size
    for attempt in range(1, args.max_sampling_attempts + 1):
        target_size = min(len(quality_filtered), current_size)
        attempt_seed = args.seed + attempt - 1
        print(f"\nSampling attempt {attempt}/{args.max_sampling_attempts} (target size: {target_size}, seed: {attempt_seed})")
        candidates = apply_sampling_pipeline(
            quality_filtered,
            args,
            target_size,
            attempt_seed,
            sql_type_targets,
            field_targets
        )
        ok, reason = validate_sampling_result(candidates, sql_type_targets, args.distribution_tolerance, field_targets)
        if ok:
            benchmark_samples = candidates
            print("Sampling criteria satisfied.")
            break
        else:
            print(f"Sampling criteria not met: {reason}")
            if attempt < args.max_sampling_attempts:
                current_size = min(len(quality_filtered), current_size + args.size_increment)
                print(f"Increasing target size to {current_size} and retrying...")
            else:
                print("Maximum sampling attempts reached without satisfying criteria.")
    
    if benchmark_samples is None:
        print("Error: Unable to generate benchmark meeting distribution criteria. Consider relaxing thresholds or increasing dataset size.")
        sys.exit(1)
    
    benchmark = create_ftv2_benchmark(benchmark_samples, args.db_uri, args.skip_execution)
    
    save_benchmark(
        benchmark,
        args.output,
        metadata_context={
            'primary_tone': args.primary_tone,
            'sampling_attempts': attempt,
            'final_target_size': len(benchmark_samples)
        }
    )
    
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

