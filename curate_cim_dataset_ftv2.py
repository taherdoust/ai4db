#!/usr/bin/env python3
"""
CIM Dataset Curation Script for FTv2 (Multi-Mode Training)
===========================================================

This script curates datasets for THREE training modes:
1. Q2Inst: Question → Instruction (first stage of two-stage)
2. QInst2SQL: Question + Instruction → SQL (second stage of two-stage)
3. Q2SQL: Question → SQL (single-stage direct)

All modes use the same source data but have different field requirements.

Author: Ali Taherdoust
Date: October 2025
Version: 2.0 (FTv2 Multi-Mode)
"""

import os
import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_raw_dataset(file_path: str) -> List[Dict]:
    """Load raw dataset from JSONL file."""
    print(f"Loading dataset from: {file_path}")
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading samples"):
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    print(f"Loaded {len(samples)} samples")
    return samples


def filter_quality_samples(samples: List[Dict], 
                          quality_threshold: float = 0.75,
                          min_question_length: int = 20,
                          max_question_length: int = 500,
                          min_sql_length: int = 20,
                          min_instruction_length: int = 20) -> List[Dict]:
    """Filter samples based on quality criteria."""
    print(f"\nApplying quality filters:")
    print(f"  Quality threshold: {quality_threshold}")
    print(f"  Question length: {min_question_length}-{max_question_length} chars")
    print(f"  Min SQL length: {min_sql_length}")
    print(f"  Min instruction length: {min_instruction_length}")
    
    filtered = []
    filter_reasons = Counter()
    
    for sample in tqdm(samples, desc="Filtering samples"):
        quality = sample.get('quality_score', 0.0)
        if quality < quality_threshold:
            filter_reasons['low_quality_score'] += 1
            continue
        
        question = sample.get('question', '')
        sql = sample.get('sql_postgis', '')
        instruction = sample.get('instruction', '')
        
        if not question or not sql:
            filter_reasons['missing_required_fields'] += 1
            continue
        
        if len(question) < min_question_length or len(question) > max_question_length:
            filter_reasons['question_length'] += 1
            continue
        
        if len(sql) < min_sql_length:
            filter_reasons['sql_too_short'] += 1
            continue
        
        if instruction and len(instruction) < min_instruction_length:
            filter_reasons['instruction_too_short'] += 1
            continue
        
        sql_lower = sql.lower()
        if 'select' not in sql_lower or 'from' not in sql_lower:
            filter_reasons['invalid_sql_structure'] += 1
            continue
        
        filtered.append(sample)
    
    print(f"\nFiltering results:")
    print(f"  Original samples: {len(samples)}")
    print(f"  Filtered samples: {len(filtered)}")
    print(f"  Retention rate: {len(filtered)/len(samples)*100:.1f}%")
    
    return filtered


def create_stratified_splits(samples: List[Dict],
                            train_ratio: float = 0.70,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test splits."""
    print(f"\nCreating stratified splits:")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val: {val_ratio*100:.0f}%")
    print(f"  Test: {test_ratio*100:.0f}%")
    
    df = pd.DataFrame(samples)
    
    # Create stratification key
    df['overall_difficulty'] = df['difficulty'].apply(
        lambda x: x.get('overall_difficulty', 'MEDIUM') if isinstance(x, dict) else 'MEDIUM'
    )
    df['strat_key'] = df['overall_difficulty'] + '_' + df['sql_type'].fillna('UNKNOWN')
    
    # Check stratification key distribution
    strat_key_counts = df['strat_key'].value_counts()
    rare_keys = strat_key_counts[strat_key_counts < 2].index.tolist()
    
    if rare_keys:
        print(f"\nFound {len(rare_keys)} rare stratification groups with <2 samples")
        print(f"Falling back to simple random split without stratification")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_test_ratio,
            random_state=random_state,
            shuffle=True
        )
    else:
        # Use stratified split
        print(f"All stratification groups have sufficient samples. Using stratified split.")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=df['strat_key']
        )
        
        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=temp_df['strat_key']
        )
    
    # Remove temporary columns
    for df_split in [train_df, val_df, test_df]:
        df_split.drop(columns=['strat_key', 'overall_difficulty'], inplace=True, errors='ignore')
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_mode_datasets(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      output_dir: str,
                      mode: str):
    """
    Save datasets for specific training mode.
    
    Args:
        mode: 'q2inst', 'qinst2sql', or 'q2sql'
    """
    mode_fields = {
        'q2inst': ['id', 'question', 'instruction'],
        'qinst2sql': ['id', 'question', 'instruction', 'sql_postgis'],
        'q2sql': ['id', 'question', 'sql_postgis']
    }
    
    if mode not in mode_fields:
        raise ValueError(f"Invalid mode: {mode}. Must be q2inst, qinst2sql, or q2sql")
    
    keep_fields = mode_fields[mode]
    
    print(f"\n{'='*70}")
    print(f"Saving datasets for mode: {mode.upper()}")
    print(f"  Fields: {', '.join(keep_fields)}")
    print(f"{'='*70}")
    
    # Keep only relevant fields
    train_mode = train_df[keep_fields].copy()
    val_mode = val_df[keep_fields].copy()
    test_mode = test_df[keep_fields].copy()
    
    # Save JSONL files
    train_path = os.path.join(output_dir, f'{mode}_train.jsonl')
    val_path = os.path.join(output_dir, f'{mode}_val.jsonl')
    test_path = os.path.join(output_dir, f'{mode}_test.jsonl')
    
    train_mode.to_json(train_path, orient='records', lines=True, force_ascii=False)
    val_mode.to_json(val_path, orient='records', lines=True, force_ascii=False)
    test_mode.to_json(test_path, orient='records', lines=True, force_ascii=False)
    
    print(f"  Train: {train_path} ({len(train_mode)} samples)")
    print(f"  Val: {val_path} ({len(val_mode)} samples)")
    print(f"  Test: {test_path} ({len(test_mode)} samples)")


def generate_statistics(train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        test_df: pd.DataFrame) -> Dict:
    """Generate comprehensive statistics."""
    stats = {
        'dataset_info': {
            'total_samples': len(train_df) + len(val_df) + len(test_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'curation_date': pd.Timestamp.now().isoformat()
        }
    }
    
    # SQL type distribution
    if 'sql_type' in train_df.columns:
        stats['sql_type_distribution'] = train_df['sql_type'].value_counts().to_dict()
    
    # Difficulty distribution
    if 'difficulty' in train_df.columns:
        difficulties = train_df['difficulty'].apply(
            lambda x: x.get('overall_difficulty', 'UNKNOWN') if isinstance(x, dict) else 'UNKNOWN'
        )
        stats['difficulty_distribution'] = difficulties.value_counts().to_dict()
    
    # Quality score statistics
    if 'quality_score' in train_df.columns:
        stats['quality_score_stats'] = {
            'mean': float(train_df['quality_score'].mean()),
            'median': float(train_df['quality_score'].median()),
            'min': float(train_df['quality_score'].min()),
            'max': float(train_df['quality_score'].max())
        }
    
    return stats


def main():
    """Main function for dataset curation CLI."""
    parser = argparse.ArgumentParser(
        description="Curate CIM spatial SQL dataset for three training modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  q2inst      : Question → Instruction (first stage of two-stage)
  qinst2sql   : Question + Instruction → SQL (second stage of two-stage)
  q2sql       : Question → SQL (single-stage direct)

Example:
  python curate_cim_dataset_ftv2.py \\
    ../../ai4db/training_datasets/stage3_augmented_dataset_FINAL_checkpoint.jsonl \\
    --output_dir /media/space/castangia/Ali_workspace/curated_dataset_ftv2
"""
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to raw dataset JSONL file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/media/space/castangia/Ali_workspace/curated_dataset_ftv2',
        help='Output directory for curated datasets'
    )
    
    parser.add_argument(
        '--quality_threshold',
        type=float,
        default=0.75,
        help='Minimum quality score threshold (default: 0.75)'
    )
    
    parser.add_argument(
        '--max_question_length',
        type=int,
        default=500,
        help='Maximum question length in characters (default: 500)'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CIM DATASET CURATION (FTv2 Multi-Mode)")
    print("="*70)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Modes: Q2Inst, QInst2SQL, Q2SQL")
    print("="*70)
    
    # Load raw dataset
    raw_samples = load_raw_dataset(args.input_file)
    
    # Filter quality samples
    filtered_samples = filter_quality_samples(
        raw_samples,
        quality_threshold=args.quality_threshold,
        max_question_length=args.max_question_length
    )
    
    if len(filtered_samples) == 0:
        print("ERROR: No samples passed quality filters")
        return 1
    
    # Create stratified splits (same split for all modes)
    train_df, val_df, test_df = create_stratified_splits(
        filtered_samples,
        random_state=args.random_state
    )
    
    # Generate statistics
    stats = generate_statistics(train_df, val_df, test_df)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save datasets for all three modes
    for mode in ['q2inst', 'qinst2sql', 'q2sql']:
        save_mode_datasets(train_df, val_df, test_df, args.output_dir, mode)
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, 'curation_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics saved: {stats_path}")
    
    # Create README
    readme_path = os.path.join(args.output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# CIM Wizard Spatial SQL Curated Dataset (FTv2)\n\n")
        f.write("## Dataset Modes\n\n")
        f.write("This curation generates datasets for three training modes:\n\n")
        f.write("1. **Q2Inst** (Question → Instruction): First stage of two-stage architecture\n")
        f.write("   - Fields: id, question, instruction\n")
        f.write("   - Use for: Training instruction generator\n\n")
        f.write("2. **QInst2SQL** (Question + Instruction → SQL): Second stage of two-stage\n")
        f.write("   - Fields: id, question, instruction, sql_postgis\n")
        f.write("   - Use for: Training SQL generator with instruction context\n\n")
        f.write("3. **Q2SQL** (Question → SQL): Single-stage direct\n")
        f.write("   - Fields: id, question, sql_postgis\n")
        f.write("   - Use for: Training direct SQL generator\n\n")
        f.write("## Dataset Statistics\n\n")
        f.write(f"- Total samples: {stats['dataset_info']['total_samples']:,}\n")
        f.write(f"- Train: {stats['dataset_info']['train_samples']:,} (70%)\n")
        f.write(f"- Val: {stats['dataset_info']['val_samples']:,} (15%)\n")
        f.write(f"- Test: {stats['dataset_info']['test_samples']:,} (15%)\n")
    
    print(f"README saved: {readme_path}")
    
    print("\n" + "="*70)
    print("CURATION COMPLETE")
    print("="*70)
    print(f"\nDatasets created for all three modes:")
    print(f"  - q2inst: Question → Instruction")
    print(f"  - qinst2sql: Question + Instruction → SQL")
    print(f"  - q2sql: Question → SQL (direct)")
    print(f"\nTotal samples per mode:")
    print(f"  - Train: {stats['dataset_info']['train_samples']:,}")
    print(f"  - Val: {stats['dataset_info']['val_samples']:,}")
    print(f"  - Test: {stats['dataset_info']['test_samples']:,}")
    print(f"\nReady for training with any of the 9 training scripts!")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

