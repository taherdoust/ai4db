#!/usr/bin/env python3
"""
generate_negative_samples.py
Generate taxonomy-aligned negative samples compatible with Stage 1 / Stage 3 outputs.

Each sample now contains ONLY the following keys so it can be merged with positive
Stage 1 datasets without additional mapping:
    - id
    - task_complexity = 2
    - task_frequency  = 2
    - task_type       = "AMBIGUOUS" or "OUT_OF_SCOPE"
    - domain_complexity = 2
    - domain_frequency  = 2
    - domain_type     = "AMBIGUOUS" or "OUT_OF_SCOPE"
    - question_tone   = "AMBIGUOUS" or "OUT_OF_SCOPE"
    - sample_dirtiness= "AMBIGUOUS" or "OUT_OF_SCOPE"
    - question
    - sql_postgis     = "SELECT 'Ambiguous Request';" or "SELECT 'OUT_OF_SCOPE';"

Usage:
    python generate_negative_samples.py --output negative_samples.jsonl --count 10000
"""

import argparse
import json
import random
from collections import Counter
from statistics import mean
from typing import Dict, List

# ============================================================================
# NEGATIVE SAMPLE DEFINITIONS
# ============================================================================

LABEL_OUT_OF_SCOPE = "OUT_OF_SCOPE"
LABEL_AMBIGUOUS = "AMBIGUOUS"


OUT_OF_SCOPE_SAMPLES = {
    "CONVERSATIONAL": [
        "hi", "hello", "how are you", "good morning", "good afternoon",
        "thanks", "thank you", "bye", "goodbye", "see you later",
        "help", "help me", "what can you do", "who are you",
        "are you a bot", "can you help me", "I need assistance",
        "hello there", "hey", "hi there", "greetings"
    ],
    
    "GENERAL_KNOWLEDGE": [
        "what is the capital of Italy",
        "how tall is the Eiffel Tower",
        "when was World War 2",
        "what is machine learning",
        "explain quantum physics",
        "who is the president",
        "what's the weather today",
        "tell me about history",
        "what is Python programming",
        "explain databases",
        "how does GPS work",
        "what is climate change",
        "who invented the computer",
        "explain photosynthesis",
        "what is artificial intelligence"
    ],
    
    "WRONG_DOMAIN_SQL": [
        "SELECT * FROM customers WHERE age > 18",
        "SELECT product_name, price FROM products ORDER BY price",
        "SELECT COUNT(*) FROM orders GROUP BY customer_id",
        "SELECT user_id, email FROM users WHERE created_at > '2024-01-01'",
        "SELECT * FROM employees JOIN departments ON dept_id",
        "SELECT SUM(sales) FROM transactions WHERE region = 'North'",
        "SELECT * FROM inventory WHERE stock < 10",
        "SELECT student_id, grade FROM students WHERE grade >= 90",
        "SELECT account_number FROM bank_accounts WHERE balance > 1000",
        "SELECT flight_number FROM flights WHERE departure_time > NOW()"
    ]
}

AMBIGUOUS_SAMPLES = {
    "MISSING_PROJECT_FILTER": [
        "What are the building IDs?",
        "Show me building areas",
        "Find building heights",
        "Get building types",
        "List all buildings",
        "Show building properties",
        "Find residential buildings",
        "Get building volumes",
        "Show building floor counts",
        "List building construction years"
    ],
    
    "MISSING_SPATIAL_CONTEXT": [
        "Which buildings overlap?",
        "Find intersecting areas",
        "Show buildings within boundaries",
        "Get buildings that intersect",
        "Find overlapping geometries",
        "Which census zones intersect?",
        "Show areas that contain buildings",
        "Find buildings inside zones",
        "Get intersecting features",
        "Show spatial relationships"
    ],
    
    "LOGICALLY_INCOMPLETE": [
        "Find buildings that intersect",
        "Show areas greater than threshold",
        "Get data for analysis",
        "Calculate distances",
        "Find nearby features",
        "Show spatial data",
        "Get geometric information",
        "Calculate areas",
        "Find measurements",
        "Show statistics"
    ],
    
    "MISSING_SCENARIO": [
        "Find census zones",
        "Show grid buses",
        "Get network lines",
        "List census data",
        "Show population data",
        "Get demographic information",
        "Find grid infrastructure",
        "Show electrical network",
        "Get raster data",
        "Show elevation data"
    ]
}

INVALID_SCHEMA_SAMPLES = {
    "WRONG_TABLE_NAMES": [
        "find buildings in the 'structures' table",
        "get data from 'building_info' table",
        "select from 'spatial_data.buildings'",
        "query the 'cim_buildings' table",
        "find data in 'property_data' table",
        "select from 'building_table'",
        "get data from 'vector.buildings'",
        "query 'gis_buildings' table",
        "select from 'urban_structures'",
        "find in 'construction_data' table"
    ],
    
    "WRONG_COLUMN_NAMES": [
        "get building 'size' column",
        "find 'property_type' in buildings",
        "select 'building_name' from buildings",
        "get 'structure_height' column",
        "find 'construction_date' in properties",
        "select 'building_age' from buildings",
        "get 'floor_count' column",
        "find 'building_category' in properties",
        "select 'geometry_data' from buildings",
        "get 'spatial_reference' column"
    ],
    
    "WRONG_SCHEMA_NAMES": [
        "select from schema 'gis_data'",
        "query 'spatial.buildings' table",
        "find in 'urban_data' schema",
        "select from 'building_schema'",
        "get data from 'vector_data' schema",
        "query 'census_data.zones'",
        "select from 'grid_schema'",
        "find in 'raster_data' schema",
        "query 'cim.buildings' table",
        "select from 'wizard.buildings'"
    ]
}

ADVERSARIAL_INSTRUCTIONS = [
    {
        "question": "Find all buildings in the project",
        "instruction": "Query the 'structures' table from the 'spatial' schema"
    },
    {
        "question": "Get building areas",
        "instruction": "Select the 'size' column from buildings table"
    },
    {
        "question": "Find buildings in census zones",
        "instruction": "JOIN buildings and census using building_id"
    },
    {
        "question": "Calculate building heights",
        "instruction": "Use the 'height_meters' column from building_info table"
    },
    {
        "question": "Find grid buses by voltage",
        "instruction": "Query network_infrastructure table and filter by voltage_level column"
    }
]


# ============================================================================
# HELPERS
# ============================================================================

def build_negative_sample(sample_id: str, question: str, label: str) -> Dict:
    """Create taxonomy-aligned negative sample."""
    sql = "SELECT 'Ambiguous Request';" if label == LABEL_AMBIGUOUS else "SELECT 'OUT_OF_SCOPE';"
    return {
        "id": sample_id,
        "task_complexity": 2,
        "task_frequency": 2,
        "task_type": label,
        "domain_complexity": 2,
        "domain_frequency": 2,
        "domain_type": label,
        "question_tone": label,
        "sample_dirtiness": label,
        "question": question,
        "sql_postgis": sql
    }

# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_out_of_scope_samples(count: int) -> List[Dict]:
    """Generate OUT_OF_SCOPE samples (conversational, general knowledge, wrong domain)."""

    samples: List[Dict] = []
    distribution = {"CONVERSATIONAL": 0.33, "GENERAL_KNOWLEDGE": 0.33, "WRONG_DOMAIN_SQL": 0.34}

    for category, ratio in distribution.items():
        category_count = int(count * ratio)
        base_samples = OUT_OF_SCOPE_SAMPLES[category]

        for i in range(category_count):
            base = random.choice(base_samples)
            question = create_variation(base, category)
            sample_id = f"out_of_scope_{category.lower()}_{i:05d}"
            samples.append(build_negative_sample(sample_id, question, LABEL_OUT_OF_SCOPE))

    return samples

def generate_ambiguous_samples(count: int) -> List[Dict]:
    """Generate AMBIGUOUS samples caused by missing context or logic."""

    samples: List[Dict] = []
    distribution = {
        "MISSING_PROJECT_FILTER": 0.40,
        "MISSING_SPATIAL_CONTEXT": 0.30,
        "LOGICALLY_INCOMPLETE": 0.20,
        "MISSING_SCENARIO": 0.10
    }
    
    for category, ratio in distribution.items():
        category_count = int(count * ratio)
        base_samples = AMBIGUOUS_SAMPLES[category]
        
        for i in range(category_count):
            base = random.choice(base_samples)
            question = create_variation(base, category)
            sample_id = f"ambiguous_{category.lower()}_{i:05d}"
            samples.append(build_negative_sample(sample_id, question, LABEL_AMBIGUOUS))

    return samples

def generate_invalid_schema_samples(count: int) -> List[Dict]:
    """Generate AMBIGUOUS samples with invalid schema/table/column references."""
    
    samples: List[Dict] = []
    distribution = {"WRONG_TABLE_NAMES": 0.40, "WRONG_COLUMN_NAMES": 0.35, "WRONG_SCHEMA_NAMES": 0.25}

    for category, ratio in distribution.items():
        category_count = int(count * ratio)
        base_samples = INVALID_SCHEMA_SAMPLES[category]
        
        for i in range(category_count):
            base = random.choice(base_samples)
            question = create_variation(base, category)
            sample_id = f"invalid_schema_{category.lower()}_{i:05d}"
            samples.append(build_negative_sample(sample_id, question, LABEL_AMBIGUOUS))

    return samples

def generate_adversarial_samples(count: int) -> List[Dict]:
    """Generate adversarial instruction samples labelled as AMBIGUOUS."""

    samples: List[Dict] = []

    for i in range(count):
        template = random.choice(ADVERSARIAL_INSTRUCTIONS)
        question = f"{template['question']} (Instruction: {template['instruction']})"
        sample_id = f"adversarial_{i:05d}"
        samples.append(build_negative_sample(sample_id, question, LABEL_AMBIGUOUS))

    return samples

def create_variation(base: str, category: str) -> str:
    """Create natural variations of base sample"""
    
    variations = [base]
    
    if category == "CONVERSATIONAL":
        prefixes = ["", "Hey, ", "Hi there, ", "Hello, ", "Excuse me, "]
        suffixes = ["", "?", " please", " thanks", "!"]
        variation = f"{random.choice(prefixes)}{base}{random.choice(suffixes)}"
        return variation
    
    elif category in ["MISSING_PROJECT_FILTER", "MISSING_SPATIAL_CONTEXT", "LOGICALLY_INCOMPLETE", "MISSING_SCENARIO"]:
        contexts = [
            base,
            f"{base} in the database",
            f"{base} for analysis",
            f"Can you {base.lower()}",
            f"I need to {base.lower()}",
            f"Please {base.lower()}"
        ]
        return random.choice(contexts)
    
    elif category in ["WRONG_TABLE_NAMES", "WRONG_COLUMN_NAMES", "WRONG_SCHEMA_NAMES"]:
        contexts = [
            base,
            f"{base} in CIM Wizard",
            f"For CIM database, {base}",
            f"{base} for project analysis",
            f"Using CIM schema, {base}"
        ]
        return random.choice(contexts)
    
    return base

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_negative_dataset(
    total_count: int = 30000,
    output_file: str = "negative_samples.jsonl",
    random_seed: int = 42
) -> List[Dict]:
    """
    Generate complete negative sample dataset
    
    Distribution:
    - OUT_OF_SCOPE: 35% (10,500)
    - AMBIGUOUS: 35% (10,500)
    - INVALID_SCHEMA: 20% (6,000)
    - ADVERSARIAL: 10% (3,000)
    """
    
    random.seed(random_seed)
    
    print("="*80)
    print("NEGATIVE SAMPLE GENERATION FOR PHASE 4")
    print("="*80)
    print(f"Target samples: {total_count:,}")
    print(f"Output file: {output_file}")
    print(f"Random seed: {random_seed}")
    
    all_samples = []
    
    # Generate each category
    print("\n[1/4] Generating OUT_OF_SCOPE samples (35%)...")
    out_of_scope = generate_out_of_scope_samples(int(total_count * 0.35))
    all_samples.extend(out_of_scope)
    print(f"      Generated {len(out_of_scope):,} samples")
    
    print("\n[2/4] Generating AMBIGUOUS samples (35%)...")
    ambiguous = generate_ambiguous_samples(int(total_count * 0.35))
    all_samples.extend(ambiguous)
    print(f"      Generated {len(ambiguous):,} samples")
    
    print("\n[3/4] Generating INVALID_SCHEMA samples (20%)...")
    invalid_schema = generate_invalid_schema_samples(int(total_count * 0.20))
    all_samples.extend(invalid_schema)
    print(f"      Generated {len(invalid_schema):,} samples")
    
    print("\n[4/4] Generating ADVERSARIAL samples (10%)...")
    adversarial = generate_adversarial_samples(int(total_count * 0.10))
    all_samples.extend(adversarial)
    print(f"      Generated {len(adversarial):,} samples")
    
    # Shuffle
    random.shuffle(all_samples)
    
    print(f"\n[COMPLETE] Total negative samples: {len(all_samples):,}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Statistics
    stats = generate_statistics(all_samples)
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_samples):,} samples")
    print(f"Statistics saved to {stats_file}")
    
    print_statistics(stats)
    
    return all_samples

def generate_statistics(samples: List[Dict]) -> Dict:
    """Generate simple statistics for taxonomy-aligned negatives."""

    label_counts = Counter(sample.get("task_type", "UNKNOWN") for sample in samples)
    question_lengths = [len((sample.get("question") or "").strip()) for sample in samples if sample.get("question")]

    stats = {
        "total_samples": len(samples),
        "label_distribution": dict(label_counts),
        "avg_question_length": mean(question_lengths) if question_lengths else 0,
        "min_question_length": min(question_lengths) if question_lengths else 0,
        "max_question_length": max(question_lengths) if question_lengths else 0,
    }
    return stats


def print_statistics(stats: Dict):
    """Print formatted statistics."""

    print("\n" + "=" * 80)
    print("NEGATIVE SAMPLE STATISTICS")
    print("=" * 80)

    total = stats["total_samples"]
    print(f"\nTotal samples: {total:,}")
    print("Label distribution:")
    for label, count in stats["label_distribution"].items():
        pct = (count / total) * 100 if total else 0
        print(f"  {label:15s}: {count:6,} ({pct:5.1f}%)")

    print("\nQuestion length stats:")
    print(f"  Average: {stats['avg_question_length']:.1f} chars")
    print(f"  Min    : {stats['min_question_length']} chars")
    print(f"  Max    : {stats['max_question_length']} chars")
    print("\n" + "=" * 80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate negative training samples for robust CIM Wizard SQL model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output', type=str, default='negative_samples.jsonl',
                       help='Output JSONL file for negative samples')
    parser.add_argument('--count', type=int, default=30000,
                       help='Total number of negative samples to generate (default: 30000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate negative samples
    samples = generate_negative_dataset(
        total_count=args.count,
        output_file=args.output,
        random_seed=args.seed
    )
    
    print(f"\nNegative sample generation complete!")
    print(f"Output: {args.output}")
    print(f"\nNext steps:")
    print(f"1. Merge with positive samples from Stage 3")
    print(f"2. Curate combined dataset (150K total)")
    print(f"3. Fine-tune model with negative sample handling")

if __name__ == '__main__':
    main()

