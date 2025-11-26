#!/usr/bin/env python3
"""
stage3_v2.py - LLM Data Augmentation for Text-to-SQL Fine-Tuning
Generates diverse question-SQL pairs using multiple augmentation strategies
with strategy-specific model selection for optimal quality/cost balance.

Strategies:
- question_paraphrase: Different phrasings (same SQL) - GPT-4o-mini
- sql_to_question: Generate question from SQL - GPT-4o-mini
- question_to_sql: Different SQL approaches (same intent) - GPT-4.1
- sql_rewrite: Equivalent SQL variations - GPT-4.1
- parameter_variation: Different parameter values - GPT-4o-mini
"""

import json
import random
import argparse
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter

# Try to load dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# ============================================================================
# OPENROUTER API CONFIGURATION
# ============================================================================

# Load .env file from the same directory as this script
if DOTENV_AVAILABLE:
    script_dir = Path(__file__).parent.absolute()
    env_file = script_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    # Note: If .env doesn't exist, we'll fall back to system environment variables
else:
    # python-dotenv not installed - will only use system environment variables
    pass

# Get API key from environment (now includes .env if loaded)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Strategy-specific model assignment
STRATEGY_MODELS = {
    # SQL reasoning tasks - use powerful model
    "question_to_sql": "openai/gpt-4.1",
    "sql_rewrite": "openai/gpt-4.1",
    
    # Language tasks - use cost-effective model
    "question_paraphrase": "openai/gpt-4o-mini",
    "sql_to_question": "openai/gpt-4o-mini",
    "parameter_variation": "openai/gpt-4o-mini"
}

# Strategy weights (distribution of samples across strategies)
STRATEGY_WEIGHTS = {
    "question_paraphrase": 0.25,   # 25% - diverse question phrasings
    "sql_to_question": 0.20,       # 20% - generate questions from SQL
    "question_to_sql": 0.20,       # 20% - diverse SQL approaches
    "sql_rewrite": 0.15,           # 15% - SQL variations
    "parameter_variation": 0.20    # 20% - different parameter values
}

# ============================================================================
# CIM WIZARD SCHEMA CONTEXT
# ============================================================================

CIM_SCHEMA_CONTEXT = """CIM Wizard Database Schema:

SCHEMAS AND TABLES:
- cim_vector.cim_wizard_project_scenario: project_id, scenario_id, project_boundary (geometry), project_center (geometry)
- cim_vector.cim_wizard_building: building_id, lod, building_geometry (geometry)
- cim_vector.cim_wizard_building_properties: building_id, height, area, volume, type, n_people, n_family
- cim_census.censusgeo: SEZ2011, geometry, P1 (population), P2, P3, P14, P15, P60
- cim_network.network_buses: bus_id, bus_name, voltage_kv, geometry
- cim_network.network_lines: line_id, from_bus_id, to_bus_id, geometry, length_km
- cim_raster.dtm: rast (raster elevation data)
- cim_raster.dsm_sansalva: rast (raster surface model)

SPATIAL FUNCTIONS (PostGIS):
- ST_Area(geometry) - calculate area
- ST_Distance(geom1, geom2) - calculate distance
- ST_Intersects(geom1, geom2) - check intersection
- ST_Within(geom1, geom2) - check if geom1 is within geom2
- ST_Contains(geom1, geom2) - check if geom1 contains geom2
- ST_DWithin(geom1, geom2, distance) - check if within distance
- ST_Buffer(geometry, distance) - create buffer
- ST_Centroid(geometry) - get centroid
- ST_Union(geometry) - union geometries
- ST_Transform(geometry, srid) - transform coordinate system

RASTER FUNCTIONS:
- ST_Value(rast, geometry) - get raster value at point
- ST_SummaryStats(rast) - get raster statistics

IMPORTANT RULES:
- Always use schema.table notation (e.g., cim_vector.cim_wizard_building)
- Building properties are in separate table (cim_wizard_building_properties)
- Join buildings with properties on building_id
- Census zones use SEZ2011 as identifier
- Do NOT add LIMIT unless explicitly requested"""

# ============================================================================
# STRATEGY PROMPTS
# ============================================================================

def get_strategy_prompts() -> Dict[str, Dict[str, str]]:
    """Get system and user prompts for each augmentation strategy"""
    
    return {
        "question_paraphrase": {
            "system": f"""You are a CIM Wizard database expert. {CIM_SCHEMA_CONTEXT}

TASK: Rephrase the given question naturally while preserving ALL specific values, IDs, thresholds, and the exact SQL intent.

GUIDELINES:
- Use different wording and sentence structure
- Preserve all numbers, IDs, and specific values EXACTLY
- Maintain the same query intent
- Vary between interrogative (What/Which/How many), direct (Find/Show/Get), and descriptive (I need/I want to know) styles
- Do NOT change the meaning or add/remove conditions
- Return ONLY the rephrased question, nothing else""",

            "user_template": """Rephrase this database question naturally:

Original: {question}

Rephrased question:"""
        },
        
        "sql_to_question": {
            "system": f"""You are a CIM Wizard database expert. {CIM_SCHEMA_CONTEXT}

TASK: Generate a natural language question that would produce this SQL query.

GUIDELINES:
- Question should match the SQL's intent EXACTLY
- Include all specific values, IDs, and thresholds from the SQL
- Use natural, conversational language
- Vary the question style (interrogative, direct command, or descriptive request)
- Return ONLY the question, nothing else""",

            "user_template": """Generate a natural language question for this SQL query:

SQL: {sql}

Question:"""
        },
        
        "question_to_sql": {
            "system": f"""You are a CIM Wizard database expert. {CIM_SCHEMA_CONTEXT}

TASK: Generate an equivalent PostGIS SQL query that answers this question. You may use a different approach (different join order, different spatial functions, subqueries vs CTEs) while maintaining the SAME result.

GUIDELINES:
- SQL must be syntactically correct PostGIS
- Must produce EQUIVALENT results to the original intent
- Use different approaches when possible:
  * Different join order
  * ST_Within vs ST_Contains (where semantically equivalent)
  * Subquery vs CTE
  * Different aggregation approach
- Preserve all specific values and IDs EXACTLY
- Always use schema.table notation
- Return ONLY the SQL query, nothing else (no markdown, no explanation)""",

            "user_template": """Generate an equivalent SQL query for this question:

Question: {question}

Reference SQL (for understanding intent):
{sql}

Equivalent SQL:"""
        },
        
        "sql_rewrite": {
            "system": f"""You are a CIM Wizard database expert. {CIM_SCHEMA_CONTEXT}

TASK: Rewrite this SQL query using different syntax or structure while producing the SAME results.

GUIDELINES:
- Use different join syntax (explicit JOIN vs implicit WHERE)
- Reorder joins when possible
- Use equivalent spatial functions where appropriate
- Convert between subquery and CTE styles
- Preserve ALL specific values and conditions
- Must produce IDENTICAL results
- Return ONLY the rewritten SQL, nothing else (no markdown, no explanation)""",

            "user_template": """Rewrite this SQL query using different syntax while keeping identical results:

Original SQL: {sql}

Rewritten SQL:"""
        },
        
        "parameter_variation": {
            "system": f"""You are a CIM Wizard database expert. {CIM_SCHEMA_CONTEXT}

TASK: Create a new question and SQL pair with different but REALISTIC parameter values while maintaining the SAME query structure.

GUIDELINES:
- Change numeric values to different but realistic values:
  * Heights: 5-50 meters
  * Areas: 50-5000 square meters
  * Distances: 10-1000 meters
  * Population: 100-10000
- Change IDs to different valid values (project_id, scenario_id, building_id)
- Building types: residential, commercial, industrial, mixed
- Keep the EXACT same query structure and logic
- Return in this EXACT format:
QUESTION: [your new question]
SQL: [your new SQL]""",

            "user_template": """Create a variation with different parameter values:

Original Question: {question}
Original SQL: {sql}

New question and SQL (format: QUESTION: ... SQL: ...):"""
        }
    }

# ============================================================================
# OPENROUTER API CLIENT
# ============================================================================

def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 500,
    temperature: float = 0.7,
    retries: int = 3
) -> Optional[str]:
    """Call OpenRouter API with retries"""
    
    try:
        import requests
    except ImportError:
        print("ERROR: requests library not installed")
        return None
    
    if not OPENROUTER_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cim-wizard.local",
        "X-Title": "CIM Wizard Stage 3 Augmentation"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                # Clean up markdown code blocks if present
                content = re.sub(r'^```sql\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                return content.strip()
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 2 ** attempt
                print(f"      Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"      API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"      API call failed: {e}")
            if attempt < retries - 1:
                time.sleep(1)
    
    return None

# ============================================================================
# STRATEGY APPLICATION
# ============================================================================

def apply_strategy(
    sample: Dict,
    strategy: str,
    prompts: Dict,
    idx: int
) -> Optional[Dict]:
    """Apply an augmentation strategy to a sample"""
    
    if strategy not in prompts:
        print(f"      Unknown strategy: {strategy}")
        return None
    
    question = sample["question"]
    sql = sample["sql_postgis"]
    
    # Get model for this strategy
    model = STRATEGY_MODELS.get(strategy, "openai/gpt-4.1")
    
    # Get prompts
    prompt_config = prompts[strategy]
    system_prompt = prompt_config["system"]
    user_prompt = prompt_config["user_template"].format(question=question, sql=sql)
    
    # Set temperature based on strategy
    if strategy in ["question_to_sql", "sql_rewrite"]:
        temperature = 0.5  # Lower for SQL generation (more deterministic)
    else:
        temperature = 0.7  # Higher for language variation
    
    # Call API
    if not OPENROUTER_API_KEY:
        return None
    
    response = call_openrouter(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=600
    )
    
    if not response:
        return None
    
    # Parse response based on strategy
    new_question = None
    new_sql = None
    
    if strategy == "question_paraphrase":
        new_question = response
        new_sql = sql  # Keep original SQL
        
    elif strategy == "sql_to_question":
        new_question = response
        new_sql = sql  # Keep original SQL
        
    elif strategy == "question_to_sql":
        new_sql = response
        new_question = question  # Keep original question
        
    elif strategy == "sql_rewrite":
        new_sql = response
        new_question = question  # Keep original question
        
    elif strategy == "parameter_variation":
        # Parse "QUESTION: ...\nSQL: ..." format
        lines = response.split("\n")
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if line_upper.startswith("QUESTION:"):
                new_question = line.split(":", 1)[1].strip()
            elif line_upper.startswith("SQL:"):
                # SQL might span multiple lines
                sql_parts = [line.split(":", 1)[1].strip()]
                for remaining_line in lines[i+1:]:
                    if remaining_line.strip() and not remaining_line.upper().startswith("QUESTION:"):
                        sql_parts.append(remaining_line.strip())
                    else:
                        break
                new_sql = " ".join(sql_parts)
        
        # Fallback if parsing fails
        if not new_question:
            new_question = question
        if not new_sql:
            new_sql = sql
    
    # Validate we got content
    if not new_question or not new_sql:
        return None
    
    # Get source ID
    source_id = sample.get("id_rule", sample.get("id_synthesized", f"unknown_{idx}"))
    
    # Create augmented sample
    augmented = {
        "id_augmented": f"aug_{strategy[:4]}_{idx:06d}",
        
        # Copy taxonomy fields
        "task_complexity": sample["task_complexity"],
        "task_frequency": sample["task_frequency"],
        "task_type": sample["task_type"],
        "domain_complexity": sample["domain_complexity"],
        "domain_frequency": sample["domain_frequency"],
        "domain_type": sample["domain_type"],
        "question_tone": sample.get("question_tone", "INTERROGATIVE"),
        "sample_dirtiness": sample.get("sample_dirtiness", "CLEAN"),
        
        # Generated content
        "question": new_question,
        "sql_postgis": new_sql,
        
        # Metadata
        "generation_strategy": strategy,
        "generation_model": model,
        "source_id": source_id,
        "original_question": question,
        "original_sql": sql
    }
    
    return augmented

# ============================================================================
# FALLBACK AUGMENTATION (No LLM)
# ============================================================================

def apply_fallback_augmentation(sample: Dict, idx: int) -> Dict:
    """Fallback augmentation when LLM is not available"""
    
    question = sample["question"]
    
    # Simple word replacements for variation
    replacements = [
        ("Find ", random.choice(["Show ", "Get ", "List ", "Display ", "Retrieve "])),
        ("What are", random.choice(["Which are", "Show me", "List"])),
        ("buildings", random.choice(["buildings", "structures", "building records"])),
        ("in project", random.choice(["in project", "for project", "within project"])),
        ("with height", random.choice(["with height", "having height", "where height is"])),
        ("greater than", random.choice(["greater than", "more than", "exceeding", "above"])),
        ("at least", random.choice(["at least", "minimum of", "no less than"]))
    ]
    
    # Apply random replacements
    augmented_question = question
    for old, new in replacements:
        if old.lower() in augmented_question.lower() and random.random() < 0.4:
            augmented_question = augmented_question.replace(old, new, 1)
    
    source_id = sample.get("id_rule", sample.get("id_synthesized", f"unknown_{idx}"))
    
    return {
        "id_augmented": f"aug_fall_{idx:06d}",
        "task_complexity": sample["task_complexity"],
        "task_frequency": sample["task_frequency"],
        "task_type": sample["task_type"],
        "domain_complexity": sample["domain_complexity"],
        "domain_frequency": sample["domain_frequency"],
        "domain_type": sample["domain_type"],
        "question_tone": sample.get("question_tone", "INTERROGATIVE"),
        "sample_dirtiness": sample.get("sample_dirtiness", "CLEAN"),
        "question": augmented_question,
        "sql_postgis": sample["sql_postgis"],
        "generation_strategy": "fallback",
        "generation_model": "template",
        "source_id": source_id,
        "original_question": question,
        "original_sql": sample["sql_postgis"]
    }

# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_augmented_dataset(
    samples: List[Dict],
    target_size: int,
    use_llm: bool,
    batch_size: int = 10,
    delay_between_batches: float = 0.5
) -> List[Dict]:
    """Generate augmented dataset using multiple strategies"""
    
    prompts = get_strategy_prompts()
    generated_samples = []
    strategy_counts = Counter()
    strategy_success = Counter()
    
    # Calculate samples per strategy
    total_weight = sum(STRATEGY_WEIGHTS.values())
    samples_per_strategy = {}
    for strategy, weight in STRATEGY_WEIGHTS.items():
        samples_per_strategy[strategy] = int(target_size * (weight / total_weight))
    
    print(f"\n[Generation] Target: {target_size:,} samples")
    print(f"[Generation] Strategies: {len(STRATEGY_WEIGHTS)}")
    print(f"[Generation] Use LLM: {use_llm and bool(OPENROUTER_API_KEY)}")
    
    for strategy, weight in STRATEGY_WEIGHTS.items():
        target_for_strategy = samples_per_strategy[strategy]
        model = STRATEGY_MODELS.get(strategy, "openai/gpt-4.1")
        
        print(f"\n  Strategy: {strategy}")
        print(f"    Model: {model}")
        print(f"    Target: {target_for_strategy:,} samples")
        
        # Select samples to augment (cycle through if needed)
        selected_samples = []
        while len(selected_samples) < target_for_strategy:
            random.shuffle(samples)
            selected_samples.extend(samples[:target_for_strategy - len(selected_samples)])
        selected_samples = selected_samples[:target_for_strategy]
        
        for idx, sample in enumerate(selected_samples):
            strategy_counts[strategy] += 1
            
            if use_llm and OPENROUTER_API_KEY:
                augmented = apply_strategy(
                    sample=sample,
                    strategy=strategy,
                    prompts=prompts,
                    idx=len(generated_samples)
                )
                
                if augmented:
                    generated_samples.append(augmented)
                    strategy_success[strategy] += 1
                else:
                    # Fallback on LLM failure
                    fallback = apply_fallback_augmentation(sample, len(generated_samples))
                    fallback["generation_strategy"] = f"{strategy}_fallback"
                    generated_samples.append(fallback)
            else:
                # No LLM - use fallback
                fallback = apply_fallback_augmentation(sample, len(generated_samples))
                generated_samples.append(fallback)
            
            # Rate limiting for LLM calls
            if use_llm and OPENROUTER_API_KEY and (idx + 1) % batch_size == 0:
                time.sleep(delay_between_batches)
            
            # Progress
            if (idx + 1) % 100 == 0 or (idx + 1) == len(selected_samples):
                success_rate = strategy_success[strategy] / strategy_counts[strategy] * 100 if strategy_counts[strategy] > 0 else 0
                print(f"      Progress: {idx + 1}/{len(selected_samples)} (Success: {success_rate:.1f}%)")
            
            # Stop if we have enough total samples
            if len(generated_samples) >= target_size:
                break
        
        if len(generated_samples) >= target_size:
            break
    
    # Summary
    print(f"\n  Generation Summary:")
    print(f"    Total generated: {len(generated_samples):,}")
    for strategy in STRATEGY_WEIGHTS.keys():
        attempted = strategy_counts[strategy]
        succeeded = strategy_success[strategy]
        rate = succeeded / attempted * 100 if attempted > 0 else 0
        print(f"    {strategy}: {succeeded}/{attempted} ({rate:.1f}%)")
    
    return generated_samples

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_stage3_dataset(
    input_file: str,
    output_file: str,
    target_size: int = 50000,
    use_llm: bool = True,
    batch_size: int = 10,
    random_seed: int = 42
):
    """
    Generate Stage 3 augmented dataset for text-to-SQL fine-tuning
    """
    random.seed(random_seed)
    
    print("="*80)
    print("STAGE 3: LLM DATA AUGMENTATION FOR TEXT-TO-SQL FINE-TUNING")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Target size: {target_size:,}")
    print(f"API Key configured: {'Yes' if OPENROUTER_API_KEY else 'No'}")
    
    print("\nStrategy Configuration:")
    for strategy, weight in STRATEGY_WEIGHTS.items():
        model = STRATEGY_MODELS.get(strategy, "openai/gpt-4.1")
        print(f"  {strategy:25s}: {weight*100:5.1f}% -> {model}")
    
    # Load input data
    print("\n[1/3] Loading input data...")
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"      Loaded {len(samples)} samples")
    
    # Determine source stage
    if samples and "id_synthesized" in samples[0]:
        print("      Source: Stage 2 (synthetic)")
    elif samples and "id_rule" in samples[0]:
        print("      Source: Stage 1 (rule-based)")
    
    # Generate augmented samples
    print("\n[2/3] Generating augmented samples...")
    
    augmented_samples = generate_augmented_dataset(
        samples=samples,
        target_size=target_size,
        use_llm=use_llm and bool(OPENROUTER_API_KEY),
        batch_size=batch_size
    )
    
    # Save dataset
    print("\n[3/3] Saving dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      Saved to: {output_file}")
    
    # Distribution analysis
    print("\n" + "="*80)
    print("AUGMENTATION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    strategies = Counter(s['generation_strategy'] for s in augmented_samples)
    models = Counter(s.get('generation_model', 'unknown') for s in augmented_samples)
    task_types = Counter(s['task_type'] for s in augmented_samples)
    domain_types = Counter(s['domain_type'] for s in augmented_samples)
    
    print("\nStrategy Distribution:")
    for strategy, count in strategies.most_common():
        pct = count / len(augmented_samples) * 100
        print(f"  {strategy:30s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nModel Distribution:")
    for model, count in models.most_common():
        pct = count / len(augmented_samples) * 100
        print(f"  {model:30s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nTask Type Distribution:")
    for task_type, count in task_types.most_common():
        pct = count / len(augmented_samples) * 100
        print(f"  {task_type:30s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nDomain Type Distribution:")
    for domain_type, count in domain_types.most_common():
        pct = count / len(augmented_samples) * 100
        print(f"  {domain_type:35s}: {count:6,} ({pct:5.1f}%)")
    
    # Sample examples
    print("\n" + "="*80)
    print("SAMPLE AUGMENTATIONS")
    print("="*80)
    
    for strategy in STRATEGY_WEIGHTS.keys():
        strategy_samples = [s for s in augmented_samples if s['generation_strategy'] == strategy]
        if strategy_samples:
            sample = random.choice(strategy_samples)
            print(f"\n[{strategy}]")
            print(f"  Original Q: {sample['original_question'][:70]}...")
            print(f"  Augmented Q: {sample['question'][:70]}...")
            if sample['sql_postgis'] != sample['original_sql']:
                print(f"  Original SQL: {sample['original_sql'][:60]}...")
                print(f"  Augmented SQL: {sample['sql_postgis'][:60]}...")
    
    print("\n" + "="*80)
    print(f"Stage 3 Complete: {len(augmented_samples):,} augmented samples")
    print("="*80)
    
    return augmented_samples

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: LLM Data Augmentation for Text-to-SQL Fine-Tuning"
    )
    parser.add_argument(
        "--input", type=str,
        default="training_datasets/stage1_cim_dataset_frequency.jsonl",
        help="Input dataset (Stage 1 rule-based samples)"
    )
    parser.add_argument(
        "--output", type=str,
        default="training_datasets/stage3_augmented_dataset.jsonl",
        help="Output augmented dataset"
    )
    parser.add_argument(
        "--target_size", type=int, default=50000,
        help="Target number of augmented samples"
    )
    parser.add_argument(
        "--no_llm", action="store_true",
        help="Use fallback augmentation instead of LLM"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size for API rate limiting"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    augmented_samples = generate_stage3_dataset(
        input_file=args.input,
        output_file=args.output,
        target_size=args.target_size,
        use_llm=not args.no_llm,
        batch_size=args.batch_size,
        random_seed=args.seed
    )
    
    print(f"\nStage 3 Complete!")
    print(f"Total augmented samples: {len(augmented_samples):,}")
    print(f"Output: {args.output}")
