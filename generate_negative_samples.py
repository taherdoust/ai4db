#!/usr/bin/env python3
"""
generate_negative_samples.py
Generate negative training samples for robust CIM Wizard SQL model training

Categories:
1. OUT_OF_SCOPE: Conversational, general knowledge, wrong domain
2. AMBIGUOUS: Missing project_id, spatial context, or logically incomplete
3. INVALID_SCHEMA: Wrong table/column/schema names
4. ADVERSARIAL: Incorrect instructions to test schema trust

Usage:
    python generate_negative_samples.py --output negative_samples.jsonl --count 30000
"""

import json
import random
import argparse
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

# ============================================================================
# NEGATIVE SAMPLE DEFINITIONS
# ============================================================================

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
        "instruction": "Query the 'structures' table from the 'spatial' schema",
        "correct_sql": "SELECT building_id FROM cim_vector.cim_wizard_building b JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id WHERE bp.project_id = '<UUID>' AND bp.scenario_id = '<UUID>'",
        "learning_point": "Ignore incorrect table name, use cim_vector.cim_wizard_building"
    },
    {
        "question": "Get building areas",
        "instruction": "Select the 'size' column from buildings table",
        "correct_sql": "SELECT building_id, area FROM cim_vector.cim_wizard_building_properties WHERE project_id = '<UUID>' AND scenario_id = '<UUID>'",
        "learning_point": "Ignore incorrect column name, use 'area' not 'size'"
    },
    {
        "question": "Find buildings in census zones",
        "instruction": "JOIN buildings and census using building_id",
        "correct_sql": "SELECT b.building_id FROM cim_vector.cim_wizard_building b JOIN cim_census.censusgeo c ON public.ST_Intersects(b.building_geometry, c.geometry) WHERE project_id = '<UUID>'",
        "learning_point": "Ignore incorrect join method, use spatial join ST_Intersects"
    },
    {
        "question": "Calculate building heights",
        "instruction": "Use the 'height_meters' column from building_info table",
        "correct_sql": "SELECT building_id, height FROM cim_vector.cim_wizard_building_properties WHERE project_id = '<UUID>' AND scenario_id = '<UUID>'",
        "learning_point": "Ignore wrong table and column, use building_properties.height"
    },
    {
        "question": "Find grid buses by voltage",
        "instruction": "Query network_infrastructure table and filter by voltage_level column",
        "correct_sql": "SELECT bus_id, voltage_kv FROM cim_network.network_buses WHERE voltage_kv >= 20.0 AND in_service = true",
        "learning_point": "Ignore wrong table/column, use network_buses.voltage_kv"
    }
]

# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_out_of_scope_samples(count: int) -> List[Dict]:
    """Generate out-of-scope samples (conversational, general knowledge, wrong domain)"""
    
    samples = []
    distribution = {"CONVERSATIONAL": 0.33, "GENERAL_KNOWLEDGE": 0.33, "WRONG_DOMAIN_SQL": 0.34}
    
    for category, ratio in distribution.items():
        category_count = int(count * ratio)
        base_samples = OUT_OF_SCOPE_SAMPLES[category]
        
        for i in range(category_count):
            base = random.choice(base_samples)
            question = create_variation(base, category)
            
            sample = {
                "id": f"out_of_scope_{category.lower()}_{i:05d}",
                "question": question,
                "instruction": "OUT_OF_SCOPE",
                "sql_postgis": "OUT_OF_SCOPE",
                "expected_response": "OUT_OF_SCOPE: I can only help with CIM Wizard spatial database queries about buildings, projects, census data, or grid infrastructure.",
                "is_negative_sample": True,
                "negative_category": "OUT_OF_SCOPE",
                "negative_subcategory": category,
                "stage": "negative_samples",
                "generated_at": datetime.now().isoformat()
            }
            samples.append(sample)
    
    return samples

def generate_ambiguous_samples(count: int) -> List[Dict]:
    """Generate ambiguous CIM queries that need clarification"""
    
    samples = []
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
            
            # Generate appropriate response based on category
            if category == "MISSING_PROJECT_FILTER":
                response = "AMBIGUOUS: Please specify which project and scenario. Example: 'Find buildings in project 4be7d1ff-e8bf-4374-a13e-67e7b0d52eb1 scenario baseline'."
            elif category == "MISSING_SPATIAL_CONTEXT":
                response = "AMBIGUOUS: Please specify what the buildings should intersect with (project boundary, census zones, grid infrastructure, etc.)."
            elif category == "MISSING_SCENARIO":
                response = "AMBIGUOUS: Please specify project and scenario context for this query."
            else:
                response = "AMBIGUOUS: This query is too vague. Please provide specific entities, filtering criteria, or spatial context."
            
            sample = {
                "id": f"ambiguous_{category.lower()}_{i:05d}",
                "question": question,
                "instruction": "AMBIGUOUS",
                "sql_postgis": "AMBIGUOUS",
                "expected_response": response,
                "is_negative_sample": True,
                "negative_category": "AMBIGUOUS",
                "negative_subcategory": category,
                "stage": "negative_samples",
                "generated_at": datetime.now().isoformat()
            }
            samples.append(sample)
    
    return samples

def generate_invalid_schema_samples(count: int) -> List[Dict]:
    """Generate samples with invalid schema references"""
    
    samples = []
    distribution = {"WRONG_TABLE_NAMES": 0.40, "WRONG_COLUMN_NAMES": 0.35, "WRONG_SCHEMA_NAMES": 0.25}
    
    for category, ratio in distribution.items():
        category_count = int(count * ratio)
        base_samples = INVALID_SCHEMA_SAMPLES[category]
        
        for i in range(category_count):
            base = random.choice(base_samples)
            question = create_variation(base, category)
            
            # Generate appropriate response
            if category == "WRONG_TABLE_NAMES":
                response = "AMBIGUOUS: Invalid table name. Available tables: cim_vector.cim_wizard_building, cim_vector.cim_wizard_building_properties, cim_census.censusgeo, cim_network.network_buses, cim_network.network_lines, cim_raster.dtm, cim_raster.dsm_sansalva."
            elif category == "WRONG_COLUMN_NAMES":
                response = "AMBIGUOUS: Invalid column name. Please refer to the correct CIM Wizard schema. Common columns: building_id, area, height, type, project_id, scenario_id."
            else:
                response = "AMBIGUOUS: Invalid schema name. Available schemas: cim_vector, cim_census, cim_network, cim_raster."
            
            sample = {
                "id": f"invalid_schema_{category.lower()}_{i:05d}",
                "question": question,
                "instruction": "AMBIGUOUS",
                "sql_postgis": "AMBIGUOUS",
                "expected_response": response,
                "is_negative_sample": True,
                "negative_category": "AMBIGUOUS",
                "negative_subcategory": category,
                "stage": "negative_samples",
                "generated_at": datetime.now().isoformat()
            }
            samples.append(sample)
    
    return samples

def generate_adversarial_samples(count: int) -> List[Dict]:
    """Generate adversarial instruction samples to test schema trust"""
    
    samples = []
    
    for i in range(count):
        template = random.choice(ADVERSARIAL_INSTRUCTIONS)
        
        # Generate realistic parameters
        from stage1_cim import generate_realistic_values
        params = generate_realistic_values()
        
        # Replace placeholders
        correct_sql = template["correct_sql"].replace("<UUID>", params["project_id"])
        correct_sql = correct_sql.replace("<UUID>", params["scenario_id"])
        
        sample = {
            "id": f"adversarial_{i:05d}",
            "question": template["question"],
            "instruction": template["instruction"],  # Intentionally wrong
            "sql_postgis": correct_sql,  # Model should generate this despite wrong instruction
            "expected_behavior": template["learning_point"],
            "is_negative_sample": False,  # Actually positive, but adversarial
            "is_adversarial": True,
            "negative_category": "ADVERSARIAL",
            "stage": "adversarial_training",
            "generated_at": datetime.now().isoformat()
        }
        samples.append(sample)
    
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
    """Generate statistics for negative samples"""
    
    stats = {
        "total_samples": len(samples),
        "generation_date": datetime.now().isoformat(),
        "category_distribution": {},
        "subcategory_distribution": {},
        "response_types": {}
    }
    
    for sample in samples:
        category = sample.get('negative_category', 'UNKNOWN')
        subcategory = sample.get('negative_subcategory', 'UNKNOWN')
        
        stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
        stats['subcategory_distribution'][subcategory] = stats['subcategory_distribution'].get(subcategory, 0) + 1
        
        # Track response types
        response = sample.get('expected_response', '')
        if response.startswith('OUT_OF_SCOPE'):
            response_type = 'OUT_OF_SCOPE'
        elif response.startswith('AMBIGUOUS'):
            response_type = 'AMBIGUOUS'
        else:
            response_type = 'OTHER'
        
        stats['response_types'][response_type] = stats['response_types'].get(response_type, 0) + 1
    
    return stats

def print_statistics(stats: Dict):
    """Print formatted statistics"""
    
    print("\n" + "="*80)
    print("NEGATIVE SAMPLE STATISTICS")
    print("="*80)
    
    print(f"\nTotal samples: {stats['total_samples']:,}")
    
    print(f"\nCategory Distribution:")
    for category, count in sorted(stats['category_distribution'].items(), key=lambda x: -x[1]):
        percentage = count / stats['total_samples'] * 100
        print(f"  {category:20s}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\nSubcategory Distribution:")
    for subcategory, count in sorted(stats['subcategory_distribution'].items(), key=lambda x: -x[1])[:15]:
        percentage = count / stats['total_samples'] * 100
        print(f"  {subcategory:30s}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\nResponse Types:")
    for response_type, count in sorted(stats['response_types'].items(), key=lambda x: -x[1]):
        percentage = count / stats['total_samples'] * 100
        print(f"  {response_type:20s}: {count:6,} ({percentage:5.1f}%)")
    
    print("\n" + "="*80)

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

