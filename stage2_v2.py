#!/usr/bin/env python3
"""
stage2_v2.py - CTGAN Synthetic Data Generation (Redesigned)
Uses task and domain taxonomy features for CTGAN training
Generates synthetic samples with id_synthesized
"""

import json
import random
import argparse
from typing import Dict, List
from datetime import datetime
from collections import Counter

# ============================================================================
# TAXONOMY DEFINITIONS (Must match stage1_cim_v2.py)
# ============================================================================

TASK_TAXONOMY = {
    "SIMPLE_SELECT": {"complexity": 1, "frequency": 1},
    "SQL_AGGREGATION": {"complexity": 1, "frequency": 1},
    "SQL_JOIN": {"complexity": 2, "frequency": 2},
    "MULTI_SQL_JOIN": {"complexity": 3, "frequency": 3},
    "NESTED_QUERY": {"complexity": 3, "frequency": 3},
    "SPATIAL_PREDICATE": {"complexity": 1, "frequency": 1},
    "SPATIAL_PREDICATE_DISTANCE": {"complexity": 2, "frequency": 2},
    "SPATIAL_MEASUREMENT": {"complexity": 1, "frequency": 1},
    "SPATIAL_PROCESSING": {"complexity": 2, "frequency": 1},
    "SPATIAL_ACCESSOR": {"complexity": 1, "frequency": 2},
    "SPATIAL_CONSTRUCTOR": {"complexity": 1, "frequency": 1},
    "SPATIAL_TRANSFORM": {"complexity": 2, "frequency": 1},
    "SPATIAL_VALIDATION": {"complexity": 2, "frequency": 1},
    "SPATIAL_JOIN": {"complexity": 2, "frequency": 1},
    "MULTI_SPATIAL_JOIN": {"complexity": 3, "frequency": 3},
    "SPATIAL_CLUSTERING": {"complexity": 3, "frequency": 3},
    "RASTER_ANALYSIS": {"complexity": 3, "frequency": 2},
    "RASTER_VECTOR": {"complexity": 3, "frequency": 3}
}

DOMAIN_TAXONOMY = {
    "SINGLE_SCHEMA_CIM_VECTOR": {"complexity": 1, "frequency": 1},
    "MULTI_SCHEMA_WITH_CIM_VECTOR": {"complexity": 2, "frequency": 2},
    "SINGLE_SCHEMA_OTHER": {"complexity": 1, "frequency": 2},
    "MULTI_SCHEMA_WITHOUT_CIM_VECTOR": {"complexity": 2, "frequency": 3},
    "MULTI_SCHEMA_COMPLEX": {"complexity": 3, "frequency": 3}
}

QUESTION_TONES = ["INTERROGATIVE", "DIRECT", "DESCRIPTIVE"]

# ============================================================================
# LOAD STAGE 1 DATA
# ============================================================================

def load_stage1_data(input_file: str) -> List[Dict]:
    """Load Stage 1 rule-based samples"""
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

# ============================================================================
# FEATURE EXTRACTION FOR CTGAN
# ============================================================================

def extract_ctgan_features(samples: List[Dict]) -> List[Dict]:
    """
    Extract features for CTGAN training
    Features based on task and domain taxonomy
    """
    features = []
    
    for sample in samples:
        feature_row = {
            # Task taxonomy features (categorical + numeric)
            "task_complexity": sample["task_complexity"],
            "task_frequency": sample["task_frequency"],
            "task_type": sample["task_type"],
            
            # Domain taxonomy features (categorical + numeric)
            "domain_complexity": sample["domain_complexity"],
            "domain_frequency": sample["domain_frequency"],
            "domain_type": sample["domain_type"],
            
            # Question classification
            "question_tone": sample["question_tone"],
            
            # Sample quality
            "sample_dirtiness": sample["sample_dirtiness"],
            
            # Text lengths (for distribution learning)
            "question_length": len(sample["question"]),
            "sql_length": len(sample["sql_postgis"]),
            
            # Original text (for reference, not for CTGAN)
            "_question": sample["question"],
            "_sql_postgis": sample["sql_postgis"],
            "_id_rule": sample["id_rule"]
        }
        features.append(feature_row)
    
    return features

# ============================================================================
# CTGAN SYNTHESIS
# ============================================================================

def train_ctgan_model(features: List[Dict], epochs: int = 300):
    """
    Train CTGAN model on extracted features
    Returns trained model
    """
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
        import pandas as pd
    except ImportError:
        print("SDV not installed. Install with: pip install sdv")
        return None
    
    # Prepare DataFrame (exclude text columns)
    df_features = []
    for f in features:
        df_features.append({
            "task_complexity": f["task_complexity"],
            "task_frequency": f["task_frequency"],
            "task_type": f["task_type"],
            "domain_complexity": f["domain_complexity"],
            "domain_frequency": f["domain_frequency"],
            "domain_type": f["domain_type"],
            "question_tone": f["question_tone"],
            "sample_dirtiness": f["sample_dirtiness"],
            "question_length": f["question_length"],
            "sql_length": f["sql_length"]
        })
    
    df = pd.DataFrame(df_features)
    
    # Define metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Override column types for proper handling
    metadata.update_column(column_name="task_complexity", sdtype="categorical")
    metadata.update_column(column_name="task_frequency", sdtype="categorical")
    metadata.update_column(column_name="task_type", sdtype="categorical")
    metadata.update_column(column_name="domain_complexity", sdtype="categorical")
    metadata.update_column(column_name="domain_frequency", sdtype="categorical")
    metadata.update_column(column_name="domain_type", sdtype="categorical")
    metadata.update_column(column_name="question_tone", sdtype="categorical")
    metadata.update_column(column_name="sample_dirtiness", sdtype="categorical")
    metadata.update_column(column_name="question_length", sdtype="numerical")
    metadata.update_column(column_name="sql_length", sdtype="numerical")
    
    print(f"\n[CTGAN] Training with {len(df)} samples, {epochs} epochs...")
    print(f"[CTGAN] Features: {list(df.columns)}")
    
    # Train CTGAN
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        verbose=True
    )
    synthesizer.fit(df)
    
    return synthesizer, df, features

def generate_synthetic_samples(
    synthesizer,
    original_features: List[Dict],
    num_samples: int
) -> List[Dict]:
    """
    Generate synthetic samples using trained CTGAN
    Maps synthetic feature combinations to real SQL templates
    """
    import pandas as pd
    
    print(f"\n[CTGAN] Generating {num_samples} synthetic samples...")
    
    # Generate synthetic feature combinations
    synthetic_df = synthesizer.sample(num_samples)
    
    # Group original samples by feature combination for template matching
    template_pool = {}
    for f in original_features:
        key = (
            f["task_type"],
            f["domain_type"],
            f["task_complexity"],
            f["domain_complexity"]
        )
        if key not in template_pool:
            template_pool[key] = []
        template_pool[key].append(f)
    
    # Generate synthetic samples
    synthetic_samples = []
    
    for idx, row in synthetic_df.iterrows():
        # Find matching template
        key = (
            row["task_type"],
            row["domain_type"],
            int(row["task_complexity"]),
            int(row["domain_complexity"])
        )
        
        # Find closest match if exact match not found
        if key not in template_pool:
            # Fallback: find any template with same task_type
            fallback_keys = [k for k in template_pool.keys() if k[0] == row["task_type"]]
            if fallback_keys:
                key = random.choice(fallback_keys)
            else:
                # Use any random template
                key = random.choice(list(template_pool.keys()))
        
        # Select random template from matching pool
        template = random.choice(template_pool[key])
        
        # Create synthetic sample
        synthetic_sample = {
            "id_synthesized": f"syn_{idx:06d}",
            
            # Use synthetic feature values
            "task_complexity": int(row["task_complexity"]),
            "task_frequency": int(row["task_frequency"]),
            "task_type": row["task_type"],
            "domain_complexity": int(row["domain_complexity"]),
            "domain_frequency": int(row["domain_frequency"]),
            "domain_type": row["domain_type"],
            
            "question_tone": row["question_tone"],
            "sample_dirtiness": row["sample_dirtiness"],
            
            # Use template's question and SQL (will be augmented in stage 3)
            "question": template["_question"],
            "sql_postgis": template["_sql_postgis"],
            
            # Reference to original
            "source_id_rule": template["_id_rule"]
        }
        
        synthetic_samples.append(synthetic_sample)
    
    return synthetic_samples

# ============================================================================
# FALLBACK SYNTHESIS (without SDV)
# ============================================================================

def generate_synthetic_samples_fallback(
    features: List[Dict],
    num_samples: int,
    random_seed: int = 42
) -> List[Dict]:
    """
    Fallback synthesis without CTGAN
    Uses weighted random sampling based on frequency
    """
    random.seed(random_seed)
    
    print(f"\n[Fallback] Generating {num_samples} synthetic samples...")
    
    # Calculate weights based on frequency
    task_type_counts = Counter(f["task_type"] for f in features)
    domain_type_counts = Counter(f["domain_type"] for f in features)
    
    synthetic_samples = []
    
    for idx in range(num_samples):
        # Weighted random selection of template
        template = random.choice(features)
        
        # Apply frequency-based variation
        task_freq = template["task_frequency"]
        domain_freq = template["domain_frequency"]
        
        # Create synthetic sample
        synthetic_sample = {
            "id_synthesized": f"syn_{idx:06d}",
            
            "task_complexity": template["task_complexity"],
            "task_frequency": task_freq,
            "task_type": template["task_type"],
            "domain_complexity": template["domain_complexity"],
            "domain_frequency": domain_freq,
            "domain_type": template["domain_type"],
            
            "question_tone": random.choice(QUESTION_TONES) if random.random() < 0.3 else template["question_tone"],
            "sample_dirtiness": template["sample_dirtiness"],
            
            "question": template["_question"],
            "sql_postgis": template["_sql_postgis"],
            
            "source_id_rule": template["_id_rule"]
        }
        
        synthetic_samples.append(synthetic_sample)
    
    return synthetic_samples

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_stage2_dataset(
    input_file: str,
    output_file: str,
    num_synthetic: int = 10000,
    ctgan_epochs: int = 300,
    use_ctgan: bool = True,
    random_seed: int = 42
):
    """
    Generate Stage 2 synthetic dataset using CTGAN
    """
    random.seed(random_seed)
    
    print("="*80)
    print("STAGE 2: CTGAN SYNTHETIC DATA GENERATION")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Synthetic samples: {num_synthetic}")
    print(f"CTGAN epochs: {ctgan_epochs}")
    print(f"Use CTGAN: {use_ctgan}")
    
    # Load Stage 1 data
    print("\n[1/4] Loading Stage 1 data...")
    stage1_samples = load_stage1_data(input_file)
    print(f"      Loaded {len(stage1_samples)} samples")
    
    # Extract features
    print("\n[2/4] Extracting CTGAN features...")
    features = extract_ctgan_features(stage1_samples)
    print(f"      Extracted features for {len(features)} samples")
    
    # Feature distribution analysis
    print("\n      Feature Distribution:")
    task_types = Counter(f["task_type"] for f in features)
    for tt, count in task_types.most_common(5):
        print(f"        {tt}: {count}")
    
    # Train and generate
    print("\n[3/4] Generating synthetic samples...")
    
    if use_ctgan:
        try:
            result = train_ctgan_model(features, epochs=ctgan_epochs)
            if result:
                synthesizer, df, features = result
                synthetic_samples = generate_synthetic_samples(
                    synthesizer, features, num_synthetic
                )
            else:
                print("      CTGAN failed, using fallback...")
                synthetic_samples = generate_synthetic_samples_fallback(
                    features, num_synthetic, random_seed
                )
        except Exception as e:
            print(f"      CTGAN error: {e}")
            print("      Using fallback synthesis...")
            synthetic_samples = generate_synthetic_samples_fallback(
                features, num_synthetic, random_seed
            )
    else:
        synthetic_samples = generate_synthetic_samples_fallback(
            features, num_synthetic, random_seed
        )
    
    print(f"      Generated {len(synthetic_samples)} synthetic samples")
    
    # Save dataset
    print("\n[4/4] Saving dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in synthetic_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      Saved to: {output_file}")
    
    # Distribution analysis
    print("\n" + "="*80)
    print("SYNTHETIC DISTRIBUTION ANALYSIS")
    print("="*80)
    
    task_types = Counter(s['task_type'] for s in synthetic_samples)
    domain_types = Counter(s['domain_type'] for s in synthetic_samples)
    task_complexities = Counter(s['task_complexity'] for s in synthetic_samples)
    
    print("\nTask Type Distribution:")
    for task_type, count in task_types.most_common():
        pct = count / len(synthetic_samples) * 100
        print(f"  {task_type:30s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nDomain Type Distribution:")
    for domain_type, count in domain_types.most_common():
        pct = count / len(synthetic_samples) * 100
        print(f"  {domain_type:35s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nTask Complexity Distribution:")
    for complexity, count in sorted(task_complexities.items()):
        pct = count / len(synthetic_samples) * 100
        label = {1: "Easy", 2: "Medium", 3: "Hard"}[complexity]
        print(f"  {complexity} ({label:6s}): {count:6,} ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print(f"Stage 2 Complete: {len(synthetic_samples):,} synthetic samples")
    print("="*80)
    
    return synthetic_samples

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: CTGAN Synthetic Generation")
    parser.add_argument("--input", type=str, default="training_datasets/stage1_cim_dataset_frequency.jsonl",
                        help="Input Stage 1 dataset")
    parser.add_argument("--output", type=str, default="training_datasets/stage2_synthetic_dataset.jsonl",
                        help="Output synthetic dataset")
    parser.add_argument("--num_synthetic", type=int, default=10000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--epochs", type=int, default=300,
                        help="CTGAN training epochs")
    parser.add_argument("--no_ctgan", action="store_true",
                        help="Use fallback synthesis instead of CTGAN")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    synthetic_samples = generate_stage2_dataset(
        input_file=args.input,
        output_file=args.output,
        num_synthetic=args.num_synthetic,
        ctgan_epochs=args.epochs,
        use_ctgan=not args.no_ctgan,
        random_seed=args.seed
    )
    
    print(f"\nStage 2 Complete!")
    print(f"Total synthetic samples: {len(synthetic_samples):,}")
    print(f"Output: {args.output}")

