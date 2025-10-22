#!/usr/bin/env python3
"""
stage3_augmentation_pipeline_eclab_openrouter_enhanced.py
Stage 3: NL Question & Instruction Augmentation - ECLAB WITH OPENROUTER API

ENHANCED VERSION: Generates BOTH natural language questions AND instructions
in the same API call for efficiency and coherence.

Machine Specs (eclab):
- CPU: Intel Core i7-4790 @ 3.6GHz (4 cores, 8 threads)
- RAM: 16GB
- GPU: Radeon HD 6450 (not suitable for ML)

Configuration:
- Uses OpenRouter API for GPT-4, Claude, etc.
- Generates questions AND instructions together (cost-efficient)
- Template-based augmentation as fallback
- Secure API key management with .env file support
- Cost: $10-30 for OpenRouter API
"""

import json
import random
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
import requests
import time
import os
import logging

# Load .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Tip: Install python-dotenv for .env file support: pip install python-dotenv")

# Setup logging
def setup_logging(log_file: str):
    """Configure logging for both file and console output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# NLP libraries
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[WARNING] sentence-transformers not installed. Run: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================================
# SECURE API KEY MANAGEMENT
# ============================================================================

def get_api_key() -> Optional[str]:
    """
    Get API key from multiple sources (in priority order):
    1. Environment variable OPENROUTER_API_KEY
    2. .env file
    3. Return None if not found
    """
    # Try environment variable first
    api_key = os.environ.get('OPENROUTER_API_KEY')
    
    if api_key:
        # Mask the key for display
        masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
        print(f"[OK] API key loaded from environment ({masked_key})")
        return api_key
    
    # If dotenv is available, it already loaded .env file
    if DOTENV_AVAILABLE:
        print("[WARNING] API key not found. Please set OPENROUTER_API_KEY")
        print("   Option 1: export OPENROUTER_API_KEY='your-key'")
        print("   Option 2: Create .env file (copy from .env.example)")
    else:
        print("[WARNING] API key not found. Please set OPENROUTER_API_KEY")
        print("   Run: export OPENROUTER_API_KEY='your-key'")
    
    return None


# ============================================================================
# OPENROUTER API CLIENT WITH INSTRUCTION GENERATION
# ============================================================================

class OpenRouterClient:
    """
    Client for OpenRouter API - provides access to GPT-4, Claude, Llama, etc.
    
    Setup:
    1. Get API key from https://openrouter.ai/
    2. Option A: export OPENROUTER_API_KEY="your-key"
    3. Option B: Create .env file with OPENROUTER_API_KEY=your-key
    4. Choose model (default: GPT-4 Turbo for best quality)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4-turbo-preview",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.api_key = api_key or get_api_key()
        self.model = model
        self.base_url = base_url
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if API key is available"""
        if not self.api_key:
            return False
        
        print(f"[OK] OpenRouter available with model: {self.model}")
        return True
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 600,  # Increased for instructions + questions
        temperature: float = 0.85
    ) -> str:
        """Generate text using OpenRouter API"""
        
        if not self.available:
            return ""
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/taherdoust/ai4db",
                "X-Title": "AI4DB Training Data Generator"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.92
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # Check if response has expected format
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    print(f"[WARNING] Unexpected API response format: {result}")
                    return ""
            else:
                print(f"[WARNING] OpenRouter API error: {response.status_code}")
                print(f"   Response: {response.text}")
                return ""
        
        except KeyError as e:
            print(f"[WARNING] API response missing key: {e}")
            print(f"   Full response: {response.json() if response.status_code == 200 else response.text}")
            return ""
        except Exception as e:
            print(f"[WARNING] Generation error: {e}")
            return ""


# ============================================================================
# STRATEGY 1: TEMPLATE-BASED GENERATION
# ============================================================================

class TemplateAugmenter:
    """Generate questions using linguistic templates - Fast and efficient"""
    
    TEMPLATES = {
        "SPATIAL_JOIN": [
            "Find all {table1} that intersect with {table2} in {project}",
            "Which {table1} are within {table2}?",
            "Show me {table1} spatially related to {table2}",
            "Get all {table1} overlapping {table2} in scenario {scenario}",
            "List {table1} that touch {table2}",
            "Identify {table1} contained by {table2} in {project}"
        ],
        "AGGREGATION": [
            "Count the number of {table} grouped by {column}",
            "Calculate total {measure} for each {group}",
            "How many {table} are there per {group}?",
            "Aggregate {measure} by {column} in {project}",
            "Sum {measure} grouped by {group}",
            "What is the average {measure} per {group}?"
        ],
        "SPATIAL_MEASUREMENT": [
            "Calculate the area of {table} in {project}",
            "What is the total {measure} of {table}?",
            "Measure {metric} for all {table} where {condition}",
            "Compute {spatial_function} for {geometry_column}",
            "Find the distance between {table1} and {table2}",
            "What is the perimeter of {table} in {project}?"
        ],
        "SPATIAL_PROCESSING": [
            "Create a buffer of {distance}m around {table}",
            "Union all {table} in {project}",
            "Find the intersection between {table1} and {table2}",
            "Calculate the difference between {table1} and {table2}",
            "Simplify geometry of {table} with tolerance {value}",
            "Extract the boundary of {table}"
        ],
        "RASTER_VECTOR": [
            "Extract raster values for {table} from {raster}",
            "Calculate statistics of {raster} within {table}",
            "Get elevation values for all {table}",
            "Compute average height from DSM for {table}",
            "Retrieve terrain data intersecting {table}"
        ],
        "SPATIAL_CLUSTERING": [
            "Cluster {table} using DBSCAN with distance {distance}",
            "Group nearby {table} into clusters",
            "Identify spatial clusters of {table} in {project}",
            "Find dense regions of {table} using KMeans with {k} clusters"
        ],
        "NESTED_QUERY": [
            "Find {table1} where {condition} using a subquery",
            "Retrieve {table} with nested spatial analysis",
            "Get {table1} that satisfy complex condition on {table2}",
            "Extract {table} using CTE for intermediate results"
        ],
        "SIMPLE_SELECT": [
            "Select all {table} in {project}",
            "List {table} where {condition}",
            "Get {table} filtered by {attribute}",
            "Show me {table} in scenario {scenario}",
            "Retrieve {table} with {condition}"
        ]
    }
    
    INSTRUCTION_TEMPLATES = [
        "Convert this natural language question to PostGIS spatial SQL query",
        "Translate the following question into a PostGIS SQL query",
        "Write a PostGIS SQL query that answers this question",
        "Generate PostGIS spatial SQL for the following request",
        "Create a PostGIS query to solve this spatial problem"
    ]
    
    def __init__(self):
        pass
    
    def extract_slots(self, sql: str, metadata: Dict) -> Dict[str, str]:
        """Extract slot fillers from SQL and metadata"""
        
        slots = {
            'project': 'milan_smart_district',
            'scenario': 'baseline',
            'distance': '100',
            'value': '10',
            'k': '5',
            'condition': 'specific criteria',
            'attribute': 'type',
            'measure': 'area',
            'metric': 'distance',
            'column': 'region',
            'group': 'zone'
        }
        
        # Extract tables
        tables = metadata.get('database_schema', {}).get('tables', [])
        if tables:
            slots['table'] = self._simplify_table_name(tables[0])
            slots['table1'] = self._simplify_table_name(tables[0])
            if len(tables) > 1:
                slots['table2'] = self._simplify_table_name(tables[1])
            else:
                slots['table2'] = 'other features'
        
        # Extract spatial functions
        functions = metadata.get('spatial_functions', [])
        if functions:
            slots['spatial_function'] = functions[0]
        
        # Extract geometry column
        geom_cols = metadata.get('database_schema', {}).get('geometry_columns', [])
        if geom_cols:
            slots['geometry_column'] = geom_cols[0]
        else:
            slots['geometry_column'] = 'geometry'
        
        # Extract from SQL
        if 'project_id' in sql:
            project_match = re.search(r"project_id\s*=\s*'([^']+)'", sql)
            if project_match:
                slots['project'] = project_match.group(1)
        
        if 'scenario_id' in sql:
            scenario_match = re.search(r"scenario_id\s*=\s*'([^']+)'", sql)
            if scenario_match:
                slots['scenario'] = scenario_match.group(1)
        
        # Extract raster table
        if 'dsm_raster' in sql.lower():
            slots['raster'] = 'DSM raster'
        elif 'dtm_raster' in sql.lower():
            slots['raster'] = 'DTM raster'
        
        return slots
    
    def _simplify_table_name(self, full_table: str) -> str:
        """Convert 'cim_vector.building' to 'buildings'"""
        table = full_table.split('.')[-1]
        # Pluralize if not already
        if not table.endswith('s'):
            table += 's'
        return table
    
    def augment(self, sql: str, metadata: Dict, num: int = 2) -> List[Tuple[str, str]]:
        """Generate template-based (question, instruction) pairs"""
        
        sql_type = metadata.get('sql_type', 'SIMPLE_SELECT')
        
        # Get templates for this SQL type
        templates = self.TEMPLATES.get(sql_type, self.TEMPLATES['SIMPLE_SELECT'])
        
        # Extract slot fillers
        slots = self.extract_slots(sql, metadata)
        
        # Generate variations
        pairs = []
        selected_templates = random.sample(templates, min(num, len(templates)))
        
        for template in selected_templates:
            try:
                question = template.format(**slots)
                instruction = random.choice(self.INSTRUCTION_TEMPLATES)
                pairs.append((question, instruction))
            except KeyError:
                # Skip if missing required slot
                continue
        
        return pairs


# ============================================================================
# STRATEGY 2: LLM-BASED GENERATION WITH INSTRUCTION
# ============================================================================

class OpenRouterAugmenter:
    """Generate questions AND instructions using OpenRouter API"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4-turbo-preview"
    ):
        self.client = OpenRouterClient(api_key=api_key, model=model)
        self.available = self.client.available
    
    def generate_question_instruction_pairs(
        self, 
        sql: str, 
        metadata: Dict, 
        num: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Generate diverse (NL question, instruction) pairs using OpenRouter
        
        Returns:
            List of (question, instruction) tuples
        """
        
        if not self.available:
            return []
        
        # Build context
        tables = ', '.join(metadata.get('database_schema', {}).get('tables', [])[:3])
        functions = ', '.join(metadata.get('spatial_functions', [])[:5])
        sql_type = metadata.get('sql_type', 'QUERY')
        difficulty = metadata.get('difficulty', {}).get('overall_difficulty', 'MEDIUM')
        
        # Truncate SQL if too long
        sql_preview = sql[:400] + "..." if len(sql) > 400 else sql
        
        system_prompt = """You are an expert in spatial databases and SQL. Your task is to generate natural language questions AND corresponding instructions for spatial SQL queries. Generate diverse, professional, and natural-sounding text."""
        
        prompt = f"""Generate {num} diverse (question, instruction) pairs for this spatial SQL query.

SQL Query:
{sql_preview}

Context:
- Query Type: {sql_type}
- Tables: {tables}
- Spatial Functions: {functions}
- Difficulty: {difficulty}

Requirements:
1. Generate exactly {num} pairs
2. Each pair consists of:
   a) A natural language QUESTION that the SQL answers
   b) An INSTRUCTION that asks the model to convert the question to SQL

Format your response EXACTLY like this:
PAIR 1:
Question: [natural language question here]
Instruction: [instruction to convert question to SQL]

PAIR 2:
Question: [natural language question here]
Instruction: [instruction to convert question to SQL]

PAIR 3:
Question: [natural language question here]
Instruction: [instruction to convert question to SQL]

Guidelines:
- Questions should be diverse (direct, interrogative, analytical tones)
- Questions should clearly express the spatial intent
- Instructions should be clear and professional
- Vary the instruction phrasing (e.g., "Convert this question...", "Translate to SQL...", "Write a query for...")
"""
        
        try:
            response = self.client.generate(
                prompt, 
                system_prompt=system_prompt,
                max_tokens=600,  # Increased for pairs
                temperature=0.85
            )
            
            # Parse question-instruction pairs
            pairs = self._parse_pairs(response)
            
            return pairs[:num]
        
        except Exception as e:
            print(f"[WARNING] OpenRouter generation error: {e}")
            return []
    
    def _parse_pairs(self, response: str) -> List[Tuple[str, str]]:
        """Extract (question, instruction) pairs from LLM response"""
        
        pairs = []
        
        # Split by PAIR markers
        pair_blocks = re.split(r'PAIR\s+\d+:', response)
        
        for block in pair_blocks[1:]:  # Skip first empty block
            # Extract question
            question_match = re.search(r'Question:\s*(.+?)(?=Instruction:|$)', block, re.DOTALL)
            # Extract instruction
            instruction_match = re.search(r'Instruction:\s*(.+?)(?=PAIR|$)', block, re.DOTALL)
            
            if question_match and instruction_match:
                question = question_match.group(1).strip()
                instruction = instruction_match.group(1).strip()
                
                # Clean up
                question = ' '.join(question.split())  # Remove extra whitespace
                instruction = ' '.join(instruction.split())
                
                # Ensure question ends with ?
                if question and '?' not in question[-5:]:
                    question += '?'
                
                if len(question) > 20 and len(instruction) > 20:
                    pairs.append((question, instruction))
        
        return pairs


# ============================================================================
# STRATEGY 3: COMPOSITIONAL AUGMENTATION
# ============================================================================

class CompositionalAugmenter:
    """Modify question structure compositionally - Fast and lightweight"""
    
    def __init__(self):
        self.formality_map = {
            "Find": ["Retrieve", "Identify", "Locate", "Discover"],
            "Show": ["Display", "Present", "Provide", "Exhibit"],
            "Get": ["Obtain", "Fetch", "Extract", "Acquire"],
            "Count": ["Enumerate", "Tally", "Calculate the number of"],
            "Calculate": ["Compute", "Determine", "Evaluate"]
        }
        
        self.temporal_additions = [
            "from the current scenario",
            "in the latest project",
            "for the baseline scenario"
        ]
        
        self.instruction_variations = [
            "Convert this natural language question to PostGIS spatial SQL query",
            "Translate the following question into a PostGIS SQL query",
            "Write a PostGIS SQL query that answers this question",
            "Generate PostGIS spatial SQL for the following request"
        ]
    
    def augment(self, question: str, instruction: str, metadata: Dict) -> List[Tuple[str, str]]:
        """Apply compositional transformations to (question, instruction) pairs"""
        
        pairs = []
        
        # Formality shift on question
        for informal, formal_list in self.formality_map.items():
            if informal in question:
                formal = random.choice(formal_list)
                new_question = question.replace(informal, formal, 1)
                if new_question != question:
                    new_instruction = random.choice(self.instruction_variations)
                    pairs.append((new_question, new_instruction))
                    break
        
        # Add temporal context to question
        if len(pairs) > 0:
            temporal = random.choice(self.temporal_additions)
            temporal_question = f"{question} {temporal}"
            new_instruction = random.choice(self.instruction_variations)
            pairs.append((temporal_question, new_instruction))
        
        return pairs[:2]


# ============================================================================
# QUALITY CONTROL
# ============================================================================

def deduplicate_semantic(pairs: List[Tuple[str, str]], threshold: float = 0.95) -> List[Tuple[str, str]]:
    """Remove semantically similar duplicate question-instruction pairs"""
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE or len(pairs) <= 1:
        # Fall back to exact duplicate removal
        return list(dict.fromkeys(pairs))
    
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # Only compare questions for similarity
        questions = [q for q, i in pairs]
        embeddings = model.encode(questions, convert_to_tensor=True)
        
        keep_indices = [0]  # Always keep first
        
        for i in range(1, len(questions)):
            should_keep = True
            
            for j in keep_indices:
                similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
                if similarity > threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep_indices.append(i)
        
        return [pairs[i] for i in keep_indices]
    except Exception as e:
        print(f"[WARNING] Semantic deduplication error: {e}")
        return list(dict.fromkeys(pairs))


def filter_quality(
    pairs: List[Tuple[str, str]], 
    metadata: Dict, 
    min_length: int = 20, 
    max_length: int = 300
) -> List[Tuple[str, str]]:
    """Filter question-instruction pairs by quality criteria"""
    
    valid_pairs = []
    
    for question, instruction in pairs:
        # Length checks
        if not (min_length <= len(question) <= max_length):
            continue
        if not (min_length <= len(instruction) <= max_length):
            continue
        
        # Question must contain spatial terminology or table reference
        spatial_keywords = ['intersect', 'within', 'contain', 'area', 'distance', 'buffer', 
                           'near', 'overlap', 'touch', 'cross', 'geometry', 'spatial']
        has_spatial = any(kw in question.lower() for kw in spatial_keywords)
        
        table_keywords = ['building', 'grid', 'bus', 'line', 'census', 'raster', 'project']
        has_table_ref = any(word in question.lower() for word in table_keywords)
        
        # Instruction must mention SQL or query
        instruction_keywords = ['sql', 'query', 'postgis', 'convert', 'translate', 'write', 'generate']
        has_sql_ref = any(kw in instruction.lower() for kw in instruction_keywords)
        
        if (has_spatial or has_table_ref) and has_sql_ref:
            valid_pairs.append((question, instruction))
    
    return valid_pairs


def classify_tone(question: str) -> str:
    """Classify question tone"""
    
    q_lower = question.lower()
    
    # Spatial-specific
    if any(kw in q_lower for kw in ['within', 'intersecting', 'near', 'overlapping']):
        return "SPATIAL_SPECIFIC"
    
    # Interrogative
    if q_lower.startswith(('what', 'which', 'where', 'how many', 'how much')):
        return "INTERROGATIVE"
    
    # Direct
    if any(q_lower.startswith(kw) for kw in ['find', 'show', 'get', 'list', 'display']):
        return "DIRECT"
    
    # Analytical
    if any(kw in q_lower for kw in ['analyze', 'calculate', 'compute', 'determine']):
        return "ANALYTICAL"
    
    # Aggregate
    if any(kw in q_lower for kw in ['count', 'sum', 'average', 'total']):
        return "AGGREGATE"
    
    return "DESCRIPTIVE"


# ============================================================================
# MAIN PIPELINE FOR ECLAB WITH OPENROUTER (ENHANCED)
# ============================================================================

def run_stage3_pipeline_eclab_openrouter_enhanced(
    stage2_file: str = "training_datasets/stage2_synthetic_dataset_eclab_ctgan.jsonl",
    output_file: str = "training_datasets/stage3_augmented_dataset_eclab_openrouter_enhanced.jsonl",
    target_multiplier: int = 8,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: str = "openai/gpt-4-turbo-preview",
    checkpoint_interval: int = 1000,
    resume_from_checkpoint: bool = True
):
    """
    Execute Stage 3 augmentation pipeline with OpenRouter on eclab
    ENHANCED: Generates BOTH questions AND instructions
    WITH CHECKPOINTS: Saves progress periodically and can resume if interrupted
    
    Args:
        stage2_file: Path to Stage 2 synthetic dataset
        output_file: Output path
        target_multiplier: Target variations per SQL (8x default for eclab)
        openrouter_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        openrouter_model: OpenRouter model to use
        checkpoint_interval: Save checkpoint every N samples (default: 1000)
        resume_from_checkpoint: If True, resume from last checkpoint if exists
    
    Returns:
        List of augmented samples
    """
    
    # Setup logging
    log_file = output_file.replace('.jsonl', '_verbose.log')
    logger = setup_logging(log_file)
    
    pipeline_start_time = time.time()
    
    logger.info("="*80)
    logger.info("STAGE 3: NL QUESTION & INSTRUCTION AUGMENTATION - ECLAB (ENHANCED)")
    logger.info("="*80)
    logger.info(f"Machine: eclab (Intel i7-4790, 16GB RAM, CPU-only)")
    logger.info(f"Configuration:")
    logger.info(f"  - Target multiplier: {target_multiplier}x")
    logger.info(f"  - OpenRouter Model: {openrouter_model}")
    logger.info(f"  - Primary: OpenRouter API (generates questions + instructions)")
    logger.info(f"  - Secondary: Template-based (fast fallback)")
    logger.info(f"  - Checkpoint interval: {checkpoint_interval:,} samples")
    logger.info(f"  - Log file: {log_file}")
    logger.info(f"  - Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize augmenters
    logger.info("")
    logger.info("[1/5] Initializing augmentation strategies...")
    init_start = time.time()
    
    template_aug = TemplateAugmenter()
    logger.info("  Template augmenter ready")
    
    openrouter_aug = OpenRouterAugmenter(api_key=openrouter_api_key, model=openrouter_model)
    if openrouter_aug.available:
        logger.info(f"  OpenRouter augmenter ready ({openrouter_model})")
    else:
        logger.warning("  OpenRouter not available, using template-only mode")
    
    comp_aug = CompositionalAugmenter()
    logger.info("  Compositional augmenter ready")
    logger.info(f"  Initialization completed in {time.time() - init_start:.2f} seconds")
    
    # Checkpoint file paths
    checkpoint_file = output_file.replace('.jsonl', '_checkpoint.jsonl')
    checkpoint_meta_file = output_file.replace('.jsonl', '_checkpoint_meta.json')
    
    # Load Stage 2 data
    logger.info("")
    logger.info(f"[2/5] Loading Stage 2 data from {stage2_file}...")
    load_start = time.time()
    stage2_samples = []
    with open(stage2_file, 'r', encoding='utf-8') as f:
        for line in f:
            stage2_samples.append(json.loads(line))
    load_time = time.time() - load_start
    logger.info(f"  Loaded {len(stage2_samples):,} Stage 2 samples")
    logger.info(f"  Load time: {load_time:.2f} seconds")
    
    # Check for existing checkpoint
    start_idx = 0
    augmented_samples = []
    
    if resume_from_checkpoint and os.path.exists(checkpoint_file) and os.path.exists(checkpoint_meta_file):
        logger.info("")
        logger.info("[CHECKPOINT] Found existing checkpoint, resuming...")
        checkpoint_load_start = time.time()
        
        # Load checkpoint metadata
        with open(checkpoint_meta_file, 'r') as f:
            checkpoint_meta = json.load(f)
            start_idx = checkpoint_meta.get('last_processed_idx', 0) + 1
            prev_timestamp = checkpoint_meta.get('timestamp', 'unknown')
            
        # Load checkpoint data
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                augmented_samples.append(json.loads(line))
        
        checkpoint_load_time = time.time() - checkpoint_load_start
        logger.info(f"  Loaded {len(augmented_samples):,} samples from checkpoint")
        logger.info(f"  Previous checkpoint saved at: {prev_timestamp}")
        logger.info(f"  Resuming from Stage 2 sample {start_idx:,} of {len(stage2_samples):,}")
        logger.info(f"  Progress: {start_idx}/{len(stage2_samples)} ({100*start_idx/len(stage2_samples):.1f}%)")
        logger.info(f"  Checkpoint load time: {checkpoint_load_time:.2f} seconds")
    else:
        logger.info("")
        logger.info("  No checkpoint found, starting from beginning")
        logger.info(f"  Checkpoints will be saved every {checkpoint_interval:,} samples")
        logger.info(f"  Checkpoint file: {checkpoint_file}")
    
    # Augment each sample
    logger.info("")
    logger.info("[3/5] Generating augmented questions and instructions...")
    logger.info(f"  Total Stage 2 samples to process: {len(stage2_samples):,}")
    logger.info(f"  Starting from sample: {start_idx:,}")
    logger.info(f"  Samples remaining: {len(stage2_samples) - start_idx:,}")
    
    generation_start_time = time.time()
    last_checkpoint_time = time.time()
    batch_start_time = time.time()
    api_call_count = 0
    api_success_count = 0
    api_fail_count = 0
    
    for i, sample in enumerate(stage2_samples):
        # Skip already processed samples
        if i < start_idx:
            continue
        
        sample_start_time = time.time()
        sql = sample['sql_postgis']
        metadata = sample
        
        all_pairs = []  # List of (question, instruction) tuples
        
        # 1. Template (2x) - Fast baseline
        template_pairs = template_aug.augment(sql, metadata, num=2)
        all_pairs.extend(template_pairs)
        
        # 2. OpenRouter API (4x) - High quality (primary method)
        if openrouter_aug.available:
            api_start = time.time()
            openrouter_pairs = openrouter_aug.generate_question_instruction_pairs(sql, metadata, num=4)
            api_time = time.time() - api_start
            api_call_count += 1
            if len(openrouter_pairs) > 0:
                api_success_count += 1
                all_pairs.extend(openrouter_pairs)
            else:
                api_fail_count += 1
            # Add small delay to respect rate limits
            time.sleep(0.1)
        
        # 3. Compositional (2x) - if we have base pairs
        if len(all_pairs) > 0:
            base_question, base_instruction = all_pairs[0]
            comp_pairs = comp_aug.augment(base_question, base_instruction, metadata)
            all_pairs.extend(comp_pairs)
        
        # Filter and deduplicate
        pairs_before_filter = len(all_pairs)
        all_pairs = filter_quality(all_pairs, metadata)
        all_pairs = deduplicate_semantic(all_pairs, threshold=0.95)
        pairs_after_filter = len(all_pairs)
        
        # Create augmented samples
        for var_idx, (question, instruction) in enumerate(all_pairs[:target_multiplier]):
            aug_sample = sample.copy()
            aug_sample['id'] = f"{sample['id']}_aug{var_idx:02d}"
            aug_sample['question'] = question
            aug_sample['instruction'] = instruction
            aug_sample['question_tone'] = classify_tone(question)
            aug_sample['augmentation_stage'] = "stage3_eclab_openrouter_enhanced"
            aug_sample['variation_index'] = var_idx
            aug_sample['has_synthetic_instruction'] = True
            
            augmented_samples.append(aug_sample)
        
        sample_time = time.time() - sample_start_time
        
        # Log every 100 samples
        if (i + 1) % 100 == 0:
            elapsed = time.time() - generation_start_time
            processed = i + 1 - start_idx
            avg_time_per_sample = elapsed / processed if processed > 0 else 0
            remaining = len(stage2_samples) - (i + 1)
            eta_seconds = remaining * avg_time_per_sample
            eta_hours = eta_seconds / 3600
            
            logger.info(f"  Sample {i+1:,}/{len(stage2_samples):,} | "
                       f"Generated: {len(augmented_samples):,} | "
                       f"Avg time: {avg_time_per_sample:.2f}s/sample | "
                       f"ETA: {eta_hours:.2f}h")
        
        # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_save_start = time.time()
            time_since_last_checkpoint = time.time() - last_checkpoint_time
            
            logger.info("")
            logger.info(f"[CHECKPOINT] Milestone reached: {i + 1:,}/{len(stage2_samples):,} Stage 2 samples processed")
            logger.info(f"  Progress: {100*(i+1)/len(stage2_samples):.1f}% complete")
            logger.info(f"  Augmented samples generated: {len(augmented_samples):,}")
            logger.info(f"  Time since last checkpoint: {time_since_last_checkpoint/60:.2f} minutes")
            logger.info(f"  API calls: {api_call_count} (Success: {api_success_count}, Failed: {api_fail_count})")
            logger.info(f"  Saving checkpoint...")
            
            # Save augmented samples to checkpoint
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                for aug_sample in augmented_samples:
                    f.write(json.dumps(aug_sample, ensure_ascii=False) + '\n')
            
            # Save checkpoint metadata
            checkpoint_meta = {
                'last_processed_idx': i,
                'total_augmented_samples': len(augmented_samples),
                'timestamp': datetime.now().isoformat(),
                'stage2_file': stage2_file,
                'target_multiplier': target_multiplier,
                'api_call_count': api_call_count,
                'api_success_count': api_success_count,
                'api_fail_count': api_fail_count,
                'elapsed_time_seconds': time.time() - generation_start_time
            }
            with open(checkpoint_meta_file, 'w') as f:
                json.dump(checkpoint_meta, f, indent=2)
            
            checkpoint_save_time = time.time() - checkpoint_save_start
            logger.info(f"  Checkpoint saved successfully")
            logger.info(f"  Checkpoint save time: {checkpoint_save_time:.2f} seconds")
            logger.info(f"  File: {checkpoint_file}")
            logger.info("")
            
            last_checkpoint_time = time.time()
            # Reset API counters for next batch
            api_call_count = 0
            api_success_count = 0
            api_fail_count = 0
    
    generation_time = time.time() - generation_start_time
    logger.info("")
    logger.info(f"  Generation complete: {len(augmented_samples):,} augmented samples")
    logger.info(f"  Total generation time: {generation_time/60:.2f} minutes ({generation_time/3600:.2f} hours)")
    logger.info(f"  Average time per Stage 2 sample: {generation_time/len(stage2_samples):.2f} seconds")
    
    # Save dataset
    logger.info("")
    logger.info(f"[4/5] Saving augmented dataset to {output_file}...")
    save_start = time.time()
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    save_time = time.time() - save_start
    logger.info(f"  Dataset saved successfully")
    logger.info(f"  Save time: {save_time:.2f} seconds")
    logger.info(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Statistics
    logger.info("")
    logger.info("[5/5] Generating statistics...")
    
    # Count unique instructions
    unique_instructions = len(set(s['instruction'] for s in augmented_samples))
    unique_questions = len(set(s['question'] for s in augmented_samples))
    
    total_pipeline_time = time.time() - pipeline_start_time
    
    stats = {
        "total_samples": len(augmented_samples),
        "stage2_input": len(stage2_samples),
        "average_multiplier": len(augmented_samples) / len(stage2_samples) if stage2_samples else 0,
        "unique_instructions": unique_instructions,
        "unique_questions": unique_questions,
        "generation_date": datetime.now().isoformat(),
        "machine": "eclab",
        "configuration": {
            "target_multiplier": target_multiplier,
            "use_openrouter": openrouter_aug.available,
            "openrouter_model": openrouter_model,
            "generates_instructions": True,
            "checkpoint_interval": checkpoint_interval
        },
        "timing": {
            "total_pipeline_time_seconds": total_pipeline_time,
            "total_pipeline_time_hours": total_pipeline_time / 3600,
            "generation_time_seconds": generation_time,
            "generation_time_hours": generation_time / 3600,
            "avg_time_per_stage2_sample": generation_time / len(stage2_samples),
            "avg_time_per_augmented_sample": generation_time / len(augmented_samples)
        }
    }
    
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"  Statistics generated and saved to {stats_file}")
    
    # Clean up checkpoint files on successful completion
    logger.info("")
    logger.info("[CLEANUP] Removing checkpoint files...")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info(f"  Removed checkpoint file: {checkpoint_file}")
    if os.path.exists(checkpoint_meta_file):
        os.remove(checkpoint_meta_file)
        logger.info(f"  Removed checkpoint metadata: {checkpoint_meta_file}")
    
    logger.info("")
    logger.info("="*80)
    logger.info("STAGE 3 COMPLETE - SUMMARY")
    logger.info("="*80)
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Total augmented samples: {len(augmented_samples):,}")
    logger.info(f"  Unique questions: {unique_questions:,}")
    logger.info(f"  Unique instructions: {unique_instructions:,}")
    logger.info(f"  Average multiplier: {stats['average_multiplier']:.2f}x")
    logger.info(f"  Total pipeline time: {total_pipeline_time/60:.2f} minutes ({total_pipeline_time/3600:.2f} hours)")
    logger.info(f"  Generation time: {generation_time/60:.2f} minutes ({generation_time/3600:.2f} hours)")
    logger.info(f"  Statistics file: {stats_file}")
    logger.info(f"  Log file: {log_file}")
    logger.info("="*80)
    logger.info(f"  NOTE: Progress was checkpointed every {checkpoint_interval:,} samples")
    logger.info(f"  If interrupted, rerun this script to resume from last checkpoint")
    logger.info("="*80)
    
    return augmented_samples, stats


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    multiplier = 8
    model = "openai/gpt-4-turbo-preview"
    
    for i, arg in enumerate(sys.argv):
        if arg == '--multiplier' and i + 1 < len(sys.argv):
            multiplier = int(sys.argv[i + 1])
        elif arg == '--model' and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
    
    print(f"\n{'='*80}")
    print(f"Stage 3 Configuration (eclab with OpenRouter - ENHANCED)")
    print(f"{'='*80}")
    print(f"  Target multiplier: {multiplier}x")
    print(f"  OpenRouter Model: {model}")
    print(f"  Cost: ~$10-30")
    print(f"  [NEW] ENHANCED: Generates both questions AND instructions")
    print(f"\n[INFO] API Key Setup:")
    print(f"   Option 1: export OPENROUTER_API_KEY='sk-or-v1-your-key'")
    print(f"   Option 2: Create .env file (copy from .env.example)")
    print(f"\n[SECURITY] .env files are in .gitignore (safe for GitHub)")
    print(f"{'='*80}\n")
    
    # Run pipeline
    samples, stats = run_stage3_pipeline_eclab_openrouter_enhanced(
        stage2_file="training_datasets/stage2_synthetic_dataset_eclab_ctgan.jsonl",
        output_file="training_datasets/stage3_augmented_dataset_eclab_openrouter_enhanced.jsonl",
        target_multiplier=multiplier,
        openrouter_model=model
    )

