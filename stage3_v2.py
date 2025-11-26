#!/usr/bin/env python3
"""
stage3_v2.py - LLM Question Augmentation (Redesigned)
Augments questions using LLM via OpenRouter API
No instruction field - only question augmentation
Generates samples with id_augmented
"""

import json
import random
import argparse
import time
import os
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter

# ============================================================================
# OPENROUTER API CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Available models
MODELS = {
    "gpt4o-mini": "openai/gpt-4o-mini",
    "gpt4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-haiku": "anthropic/claude-3-haiku"
}

# ============================================================================
# QUESTION AUGMENTATION PROMPTS
# ============================================================================

AUGMENTATION_SYSTEM_PROMPT = """You are a CIM Wizard database query assistant. Your task is to rephrase natural language questions about buildings, census data, electrical grid infrastructure, and spatial analysis.

The CIM Wizard database contains:
- cim_vector: Building geometries and properties (building_id, height, area, type, n_people)
- cim_census: Census zones with population statistics
- cim_network: Electrical grid buses and lines
- cim_raster: DTM and DSM elevation data

Guidelines for rephrasing:
1. Preserve the EXACT meaning and all specific values (IDs, numbers, thresholds)
2. Use natural, varied language
3. Match the requested question tone
4. Keep spatial terminology accurate (within, intersects, near, distance)
5. Do NOT add LIMIT clauses unless explicitly in the original
6. Return ONLY the rephrased question, nothing else"""

def get_augmentation_prompt(question: str, question_tone: str) -> str:
    """Generate prompt for question augmentation"""
    
    tone_instructions = {
        "INTERROGATIVE": "Rephrase as a question starting with What, Which, Where, How many, etc.",
        "DIRECT": "Rephrase as a direct command starting with Find, Show, Get, List, Display, etc.",
        "DESCRIPTIVE": "Rephrase as a descriptive request starting with I need, I want to know, I would like, etc."
    }
    
    instruction = tone_instructions.get(question_tone, tone_instructions["INTERROGATIVE"])
    
    return f"""Rephrase this database query question:

Original: {question}

{instruction}

Rephrased question:"""

# ============================================================================
# OPENROUTER API CLIENT
# ============================================================================

def call_openrouter(
    prompt: str,
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 200,
    temperature: float = 0.7
) -> Optional[str]:
    """Call OpenRouter API for question augmentation"""
    
    try:
        import requests
    except ImportError:
        print("requests library not installed")
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
            {"role": "system", "content": AUGMENTATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"API call failed: {e}")
        return None

# ============================================================================
# FALLBACK AUGMENTATION (Template-based)
# ============================================================================

QUESTION_VARIATIONS = {
    "INTERROGATIVE": [
        "What are the {placeholder}",
        "Which {placeholder}",
        "Where are the {placeholder}",
        "How many {placeholder}",
        "What is the {placeholder}"
    ],
    "DIRECT": [
        "Find {placeholder}",
        "Show me {placeholder}",
        "Get {placeholder}",
        "List {placeholder}",
        "Display {placeholder}",
        "Retrieve {placeholder}"
    ],
    "DESCRIPTIVE": [
        "I need {placeholder}",
        "I want to know {placeholder}",
        "I would like to see {placeholder}",
        "Can you provide {placeholder}",
        "Please show me {placeholder}"
    ]
}

def augment_question_fallback(question: str, target_tone: str) -> str:
    """Fallback augmentation without LLM"""
    
    # Simple word replacements for variation
    replacements = {
        "Find": random.choice(["Show", "Get", "List", "Display", "Retrieve"]),
        "What are": random.choice(["Which are", "Show me", "List"]),
        "buildings": random.choice(["buildings", "structures", "building records"]),
        "in project": random.choice(["in project", "for project", "within project"]),
        "with height": random.choice(["with height", "having height", "where height is"]),
        "greater than": random.choice(["greater than", "more than", "exceeding", "above"]),
        "at least": random.choice(["at least", "minimum of", "no less than"])
    }
    
    # Apply random replacements
    augmented = question
    for old, new in replacements.items():
        if old.lower() in augmented.lower():
            # Only replace with 30% probability to maintain variation
            if random.random() < 0.3:
                augmented = augmented.replace(old, new)
    
    return augmented

# ============================================================================
# BATCH AUGMENTATION
# ============================================================================

def augment_samples(
    samples: List[Dict],
    model: str = "openai/gpt-4o-mini",
    use_llm: bool = True,
    batch_size: int = 10,
    delay_between_batches: float = 1.0
) -> List[Dict]:
    """Augment all samples with varied questions"""
    
    augmented_samples = []
    total = len(samples)
    llm_success = 0
    fallback_count = 0
    
    print(f"\n[Augmentation] Processing {total} samples...")
    
    for idx, sample in enumerate(samples):
        # Determine source ID field
        if "id_synthesized" in sample:
            source_id = sample["id_synthesized"]
        elif "id_rule" in sample:
            source_id = sample["id_rule"]
        else:
            source_id = f"unknown_{idx}"
        
        original_question = sample["question"]
        question_tone = sample.get("question_tone", "INTERROGATIVE")
        
        # Try LLM augmentation
        augmented_question = None
        
        if use_llm and OPENROUTER_API_KEY:
            prompt = get_augmentation_prompt(original_question, question_tone)
            augmented_question = call_openrouter(
                prompt=prompt,
                model=model,
                temperature=0.7
            )
            
            if augmented_question:
                llm_success += 1
            
            # Rate limiting
            if (idx + 1) % batch_size == 0:
                time.sleep(delay_between_batches)
        
        # Fallback if LLM fails
        if not augmented_question:
            augmented_question = augment_question_fallback(original_question, question_tone)
            fallback_count += 1
        
        # Create augmented sample
        augmented_sample = {
            "id_augmented": f"aug_{idx:06d}",
            
            # Copy taxonomy fields
            "task_complexity": sample["task_complexity"],
            "task_frequency": sample["task_frequency"],
            "task_type": sample["task_type"],
            "domain_complexity": sample["domain_complexity"],
            "domain_frequency": sample["domain_frequency"],
            "domain_type": sample["domain_type"],
            
            "question_tone": question_tone,
            "sample_dirtiness": sample.get("sample_dirtiness", "CLEAN"),
            
            # Augmented question
            "question": augmented_question,
            "sql_postgis": sample["sql_postgis"],
            
            # Reference to source
            "source_id": source_id,
            "original_question": original_question
        }
        
        augmented_samples.append(augmented_sample)
        
        # Progress
        if (idx + 1) % 100 == 0:
            print(f"      Progress: {idx + 1}/{total} ({(idx+1)/total*100:.1f}%)")
    
    print(f"\n      LLM augmented: {llm_success}")
    print(f"      Fallback: {fallback_count}")
    
    return augmented_samples

# ============================================================================
# LOAD INPUT DATA
# ============================================================================

def load_input_data(input_file: str) -> List[Dict]:
    """Load input samples (from Stage 1 or Stage 2)"""
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_stage3_dataset(
    input_file: str,
    output_file: str,
    model: str = "gpt4o-mini",
    use_llm: bool = True,
    batch_size: int = 10,
    random_seed: int = 42
):
    """
    Generate Stage 3 augmented dataset
    """
    random.seed(random_seed)
    
    print("="*80)
    print("STAGE 3: LLM QUESTION AUGMENTATION")
    print("="*80)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {MODELS.get(model, model)}")
    print(f"Use LLM: {use_llm}")
    print(f"API Key configured: {'Yes' if OPENROUTER_API_KEY else 'No'}")
    
    # Load input data
    print("\n[1/3] Loading input data...")
    samples = load_input_data(input_file)
    print(f"      Loaded {len(samples)} samples")
    
    # Determine source stage
    if samples and "id_synthesized" in samples[0]:
        print("      Source: Stage 2 (synthetic)")
    elif samples and "id_rule" in samples[0]:
        print("      Source: Stage 1 (rule-based)")
    
    # Augment samples
    print("\n[2/3] Augmenting questions...")
    model_name = MODELS.get(model, model)
    
    augmented_samples = augment_samples(
        samples=samples,
        model=model_name,
        use_llm=use_llm and bool(OPENROUTER_API_KEY),
        batch_size=batch_size
    )
    
    print(f"      Augmented {len(augmented_samples)} samples")
    
    # Save dataset
    print("\n[3/3] Saving dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      Saved to: {output_file}")
    
    # Distribution analysis
    print("\n" + "="*80)
    print("AUGMENTED DISTRIBUTION ANALYSIS")
    print("="*80)
    
    task_types = Counter(s['task_type'] for s in augmented_samples)
    domain_types = Counter(s['domain_type'] for s in augmented_samples)
    question_tones = Counter(s['question_tone'] for s in augmented_samples)
    
    print("\nTask Type Distribution:")
    for task_type, count in task_types.most_common():
        pct = count / len(augmented_samples) * 100
        print(f"  {task_type:30s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nDomain Type Distribution:")
    for domain_type, count in domain_types.most_common():
        pct = count / len(augmented_samples) * 100
        print(f"  {domain_type:35s}: {count:6,} ({pct:5.1f}%)")
    
    print("\nQuestion Tone Distribution:")
    for tone, count in question_tones.most_common():
        pct = count / len(augmented_samples) * 100
        print(f"  {tone:20s}: {count:6,} ({pct:5.1f}%)")
    
    # Sample examples
    print("\nSample Augmentations:")
    for i, sample in enumerate(random.sample(augmented_samples, min(3, len(augmented_samples)))):
        print(f"\n  Example {i+1}:")
        print(f"    Original: {sample['original_question'][:80]}...")
        print(f"    Augmented: {sample['question'][:80]}...")
    
    print("\n" + "="*80)
    print(f"Stage 3 Complete: {len(augmented_samples):,} augmented samples")
    print("="*80)
    
    return augmented_samples

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: LLM Question Augmentation")
    parser.add_argument("--input", type=str, default="training_datasets/stage2_synthetic_dataset.jsonl",
                        help="Input dataset (Stage 1 or Stage 2)")
    parser.add_argument("--output", type=str, default="training_datasets/stage3_augmented_dataset.jsonl",
                        help="Output augmented dataset")
    parser.add_argument("--model", type=str, default="gpt4o-mini",
                        choices=list(MODELS.keys()),
                        help="LLM model to use")
    parser.add_argument("--no_llm", action="store_true",
                        help="Use fallback augmentation instead of LLM")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for API calls")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    augmented_samples = generate_stage3_dataset(
        input_file=args.input,
        output_file=args.output,
        model=args.model,
        use_llm=not args.no_llm,
        batch_size=args.batch_size,
        random_seed=args.seed
    )
    
    print(f"\nStage 3 Complete!")
    print(f"Total augmented samples: {len(augmented_samples):,}")
    print(f"Output: {args.output}")

