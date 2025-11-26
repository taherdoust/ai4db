#!/usr/bin/env python3
"""
curator.py - Phase 4 Q2SQL Dataset Curator
==========================================

Purpose
-------
Creates curated train/validation/test splits for the Q2SQL task while
respecting CIM Wizard taxonomy annotations:

- Task complexity/frequency
- Domain complexity/frequency
- Task/domain types
- Question tone
- Sample dirtiness

Key Features
------------
1. Taxonomy-aware stratified splitting with fallback heuristics.
2. Optional field-based filtering (length constraints, allowed dirtiness).
3. Shuffle stage for all splits (training set especially) to improve
   fine-tuning batch diversity.
4. Single-mode output (Question → SQL) suitable for SQLCoder fine-tuning.

Example
-------
python curator.py \
    ../../ai4db/training_datasets/stage3_augmented_dataset.jsonl \
    --output_dir /media/space/castangia/Ali_workspace/curated_q2sql \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \
    --min_question_length 20 --min_sql_length 20
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================================
# CONSTANTS & DEFAULTS
# ============================================================================

DEFAULT_TONE = "INTERROGATIVE"
DEFAULT_DIRTINESS = "CLEAN"
DEFAULT_TASK_TYPE = "UNKNOWN_TASK_TYPE"
DEFAULT_DOMAIN_TYPE = "UNKNOWN_DOMAIN_TYPE"

RESHUFFLED_COLUMNS = [
    "id",
    "question",
    "sql_postgis",
    "task_complexity",
    "task_frequency",
    "task_type",
    "domain_complexity",
    "domain_frequency",
    "domain_type",
    "question_tone",
    "sample_dirtiness",
    "generation_strategy",
    "generation_model",
    "source_id",
]


# ============================================================================
# LOADING & NORMALIZATION
# ============================================================================

def load_samples(input_path: Path) -> List[Dict]:
    """Load JSONL dataset."""
    print(f"Loading dataset from: {input_path}")
    samples: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc="Reading samples"):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  Warning: skipping invalid JSON line -> {exc}")
    print(f"Loaded {len(samples):,} total samples")
    return samples


def normalize_taxonomy_fields(sample: Dict) -> Dict:
    """Ensure taxonomy-related fields are present with default values."""
    sample["task_complexity"] = int(sample.get("task_complexity", 0) or 0)
    sample["task_frequency"] = int(sample.get("task_frequency", 0) or 0)
    sample["domain_complexity"] = int(sample.get("domain_complexity", 0) or 0)
    sample["domain_frequency"] = int(sample.get("domain_frequency", 0) or 0)
    sample["task_type"] = str(sample.get("task_type", DEFAULT_TASK_TYPE) or DEFAULT_TASK_TYPE)
    sample["domain_type"] = str(sample.get("domain_type", DEFAULT_DOMAIN_TYPE) or DEFAULT_DOMAIN_TYPE)
    sample["question_tone"] = str(sample.get("question_tone", DEFAULT_TONE) or DEFAULT_TONE)
    sample["sample_dirtiness"] = str(sample.get("sample_dirtiness", DEFAULT_DIRTINESS) or DEFAULT_DIRTINESS)
    return sample


# ============================================================================
# FILTERING
# ============================================================================

def filter_samples(
    samples: List[Dict],
    min_question_length: int,
    max_question_length: int,
    min_sql_length: int,
    allowed_dirtiness: List[str],
) -> List[Dict]:
    """Apply length and dirtiness filters."""
    print("\nApplying quality filters:")
    print(f"  Question length range: {min_question_length}-{max_question_length} characters")
    print(f"  Min SQL length: {min_sql_length} characters")
    if allowed_dirtiness:
        print(f"  Allowed dirtiness: {', '.join(allowed_dirtiness)}")
    else:
        print("  Allowed dirtiness: ANY")

    kept: List[Dict] = []
    reasons = Counter()

    for sample in tqdm(samples, desc="Filtering samples"):
        sample = normalize_taxonomy_fields(sample)
        question = (sample.get("question") or "").strip()
        sql = (sample.get("sql_postgis") or "").strip()

        if not question or not sql:
            reasons["missing_fields"] += 1
            continue

        if len(question) < min_question_length or len(question) > max_question_length:
            reasons["question_length"] += 1
            continue

        if len(sql) < min_sql_length:
            reasons["sql_length"] += 1
            continue

        if allowed_dirtiness and sample["sample_dirtiness"] not in allowed_dirtiness:
            reasons["dirtiness_filter"] += 1
            continue

        kept.append(sample)

    print(f"\nFiltering summary:")
    print(f"  Retained samples: {len(kept):,} ({len(kept)/max(len(samples),1)*100:.1f}%)")
    if reasons:
        for reason, count in reasons.most_common():
            print(f"    - {reason}: {count:,}")

    return kept


# ============================================================================
# STRATIFICATION
# ============================================================================

def build_strat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create stratification columns with different granularity levels."""
    def ensure_str(value, default):
        return str(value) if value not in (None, "") else default

    df = df.copy()

    df["strat_lvl3"] = (
        "TC"
        + df["task_complexity"].astype(str)
        + "_TF"
        + df["task_frequency"].astype(str)
        + "_DC"
        + df["domain_complexity"].astype(str)
        + "_DF"
        + df["domain_frequency"].astype(str)
        + "_Tone"
        + df["question_tone"].apply(lambda x: ensure_str(x, DEFAULT_TONE))
        + "_Dir"
        + df["sample_dirtiness"].apply(lambda x: ensure_str(x, DEFAULT_DIRTINESS))
    )

    df["strat_lvl2"] = (
        "TC"
        + df["task_complexity"].astype(str)
        + "_DC"
        + df["domain_complexity"].astype(str)
        + "_TT"
        + df["task_type"].apply(lambda x: ensure_str(x, DEFAULT_TASK_TYPE))
        + "_DT"
        + df["domain_type"].apply(lambda x: ensure_str(x, DEFAULT_DOMAIN_TYPE))
    )

    df["strat_lvl1"] = (
        df["task_type"].apply(lambda x: ensure_str(x, DEFAULT_TASK_TYPE))
        + "_"
        + df["domain_type"].apply(lambda x: ensure_str(x, DEFAULT_DOMAIN_TYPE))
        + "_Tone"
        + df["question_tone"].apply(lambda x: ensure_str(x, DEFAULT_TONE))
    )

    return df


def pick_strat_column(
    df: pd.DataFrame,
    min_group_size: int,
    coverage_threshold: float,
) -> Tuple[str, bool]:
    """
    Choose the most granular stratification column that satisfies the coverage requirement.
    Returns (column_name, used_stratification_flag).
    """
    for column in ["strat_lvl3", "strat_lvl2", "strat_lvl1"]:
        counts = df[column].value_counts()
        covered = counts[counts >= min_group_size].sum()
        coverage = covered / len(df)
        print(f"  Checking {column}: coverage {coverage*100:.1f}% (min group size {min_group_size})")
        if coverage >= coverage_threshold:
            print(f"  -> Using {column} for stratification")
            return column, True

    print("  -> No stratification column met coverage threshold. Falling back to random split.")
    return "", False


def stratified_split(
    df: pd.DataFrame,
    strat_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits with optional stratification."""
    if strat_col:
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=df[strat_col],
        )
        val_share = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_share,
            random_state=random_state,
            stratify=temp_df[strat_col],
        )
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True,
        )
        val_share = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_share,
            random_state=random_state,
            shuffle=True,
        )

    for split_df in (train_df, val_df, test_df):
        split_df.drop(columns=["strat_lvl1", "strat_lvl2", "strat_lvl3"], inplace=True, errors="ignore")

    return train_df, val_df, test_df


# ============================================================================
# SAVING & REPORTING
# ============================================================================

def ensure_id(sample: Dict, idx: int) -> str:
    """Ensure a unique ID exists for downstream training."""
    return (
        sample.get("id")
        or sample.get("id_augmented")
        or sample.get("id_rule")
        or sample.get("id_synthesized")
        or f"sample_{idx:06d}"
    )


def dataframe_from_samples(samples: List[Dict]) -> pd.DataFrame:
    """Convert list of dict samples to DataFrame with normalized IDs."""
    normalized = []
    for idx, sample in enumerate(samples):
        sample["id"] = ensure_id(sample, idx)
        normalized.append(sample)
    df = pd.DataFrame(normalized)
    df = build_strat_columns(df)
    return df


def shuffle_dataframe(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Return a shuffled copy of dataframe."""
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def save_split(df: pd.DataFrame, output_path: Path):
    """Save dataframe to JSONL, keeping standard columns plus any extras."""
    missing_cols = [col for col in RESHUFFLED_COLUMNS if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            df[col] = None
    df[RESHUFFLED_COLUMNS].to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"  Saved {len(df):,} samples -> {output_path}")


def describe_split(name: str, df: pd.DataFrame):
    """Print taxonomy-aware distribution summary."""
    print(f"\n{name} split distribution:")
    for field in [
        "task_complexity",
        "task_frequency",
        "domain_complexity",
        "domain_frequency",
        "task_type",
        "domain_type",
        "question_tone",
        "sample_dirtiness",
    ]:
        if field in df.columns:
            counts = df[field].value_counts().head(10)
            print(f"  {field}:")
            for value, count in counts.items():
                pct = count / len(df) * 100
                print(f"    - {value}: {count:,} ({pct:.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate Q2SQL dataset with taxonomy-aware stratification and shuffling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=Path, help="Path to raw dataset JSONL file")
    parser.add_argument("--output_dir", type=Path, default=Path("./curated_q2sql"), help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--min_question_length", type=int, default=15, help="Minimum question length")
    parser.add_argument("--max_question_length", type=int, default=600, help="Maximum question length")
    parser.add_argument("--min_sql_length", type=int, default=20, help="Minimum SQL length")
    parser.add_argument(
        "--allowed_dirtiness",
        type=str,
        nargs="*",
        default=[],
        help="Allowed sample dirtiness values (leave empty to allow all)",
    )
    parser.add_argument("--min_group_size", type=int, default=25, help="Minimum samples per stratification group")
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.80,
        help="Minimum proportion of samples covered by stratified groups",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.random_seed)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Train/Val/Test ratios must sum to 1.0")

    print("=" * 80)
    print("CIM WIZARD Q2SQL DATASET CURATOR")
    print("=" * 80)
    print(f"Input file : {args.input_file}")
    print(f"Output dir : {args.output_dir}")
    print(f"Ratios     : train {args.train_ratio:.2f} / val {args.val_ratio:.2f} / test {args.test_ratio:.2f}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 80)

    samples = load_samples(args.input_file)
    if not samples:
        print("ERROR: Input dataset is empty.")
        return 1

    filtered = filter_samples(
        samples,
        min_question_length=args.min_question_length,
        max_question_length=args.max_question_length,
        min_sql_length=args.min_sql_length,
        allowed_dirtiness=[d.upper() for d in args.allowed_dirtiness],
    )

    if not filtered:
        print("ERROR: No samples passed filtering criteria.")
        return 1

    df = dataframe_from_samples(filtered)
    strat_col, use_strat = pick_strat_column(
        df,
        min_group_size=args.min_group_size,
        coverage_threshold=args.coverage_threshold,
    )

    train_df, val_df, test_df = stratified_split(
        df,
        strat_col=strat_col if use_strat else "",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed,
    )

    # Shuffle all splits (training especially) for better batch diversity
    train_df = shuffle_dataframe(train_df, args.random_seed)
    val_df = shuffle_dataframe(val_df, args.random_seed + 1)
    test_df = shuffle_dataframe(test_df, args.random_seed + 2)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "q2sql_train.jsonl"
    val_path = args.output_dir / "q2sql_val.jsonl"
    test_path = args.output_dir / "q2sql_test.jsonl"

    print("\nSaving splits...")
    save_split(train_df, train_path)
    save_split(val_df, val_path)
    save_split(test_df, test_path)

    describe_split("Train", train_df)
    describe_split("Validation", val_df)
    describe_split("Test", test_df)

    print("\n" + "=" * 80)
    print("Curation complete!")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val  : {len(val_df):,} samples")
    print(f"  Test : {len(test_df):,} samples")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

