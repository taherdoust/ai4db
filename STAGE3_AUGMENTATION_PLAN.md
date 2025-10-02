# STAGE 3: NL Question Augmentation Plan

## 🎯 Objective
Generate diverse, high-quality natural language questions for synthetic spatial SQL queries created in Stage 2. This stage aims to achieve 10x variation in NL expressions while maintaining semantic alignment with SQL, enabling robust fine-tuning for Text-to-Spatial-SQL models.

**Input**: Stage 2 synthetic SQL queries (50,000-100,000 samples)  
**Output**: 500,000-1,000,000 (SQL, NL) pairs with diverse question formulations

---

## 🌈 Multi-Strategy Augmentation Framework

We employ **5 complementary augmentation strategies**, each contributing unique linguistic diversity:

### 1. **Template-Based Generation** (Baseline, 2x diversity)

#### Method
Use predefined linguistic templates for each SQL type, with slot-filling based on SQL components.

#### Templates by SQL Type

**SPATIAL_JOIN:**
- `"Find all {table1} that intersect with {table2} in {project}"`
- `"Which {table1} are within {table2}?"`
- `"Show me {table1} spatially related to {table2}"`
- `"Get all {table1} overlapping {table2} in scenario {scenario}"`

**AGGREGATION:**
- `"Count the number of {table} grouped by {column}"`
- `"Calculate total {measure} for each {group}"`
- `"How many {table} are there per {group}?"`
- `"Aggregate {measure} by {column} in {project}"`

**SPATIAL_MEASUREMENT:**
- `"Calculate the area of {table} in {project}"`
- `"What is the total {measure} of {table}?"`
- `"Measure {metric} for all {table} where {condition}"`
- `"Compute {spatial_function} for {geometry_column}"`

**Implementation:**
```python
def template_augment(sql: str, metadata: Dict) -> List[str]:
    """Generate 2-3 template variations per SQL"""
    templates = get_templates(metadata['sql_type'])
    slots = extract_slots(sql, metadata)  # table names, columns, values
    
    variations = []
    for template in random.sample(templates, min(2, len(templates))):
        question = template.format(**slots)
        variations.append(question)
    
    return variations
```

#### Pros & Cons
✅ **Pros**: Fast, controllable, guaranteed grammatical correctness  
❌ **Cons**: Limited diversity, predictable patterns, may lack naturalness

---

### 2. **Paraphrasing with Sentence-BERT** (Semantic preservation, 3x diversity)

#### Method
Use pre-trained paraphrase models (`paraphrase-MiniLM-L6-v2`) to generate semantically equivalent variations.

#### Implementation
```python
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Initialize paraphraser (T5-based)
paraphraser = pipeline("text2text-generation", model="ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")

def semantic_augment(original_question: str, num_variations: int = 3) -> List[str]:
    """Generate paraphrases with semantic similarity > 0.85"""
    
    # Generate candidates
    candidates = paraphraser(
        f"paraphrase: {original_question}",
        num_return_sequences=num_variations * 2,
        num_beams=5,
        temperature=1.2,
        do_sample=True
    )
    
    # Filter by semantic similarity
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    original_emb = model.encode(original_question, convert_to_tensor=True)
    
    valid_paraphrases = []
    for candidate in candidates:
        para = candidate['generated_text']
        para_emb = model.encode(para, convert_to_tensor=True)
        similarity = util.cos_sim(original_emb, para_emb).item()
        
        if 0.85 <= similarity <= 0.95:  # High similarity, not identical
            valid_paraphrases.append(para)
    
    return valid_paraphrases[:num_variations]
```

#### Pros & Cons
✅ **Pros**: Natural variations, semantic preservation, diverse phrasing  
❌ **Cons**: Computationally expensive, may introduce subtle semantic drift

---

### 3. **Back-Translation** (Linguistic diversity, 2x diversity)

#### Method
Translate NL → [Intermediate Language] → NL to introduce linguistic variations while preserving meaning.

**Translation Chains:**
- English → French → English
- English → German → English
- English → Spanish → English

#### Implementation
```python
from transformers import MarianMTModel, MarianTokenizer

class BackTranslator:
    def __init__(self):
        self.models = {
            'en-fr': self._load_model('en', 'fr'),
            'fr-en': self._load_model('fr', 'en'),
            'en-de': self._load_model('en', 'de'),
            'de-en': self._load_model('de', 'en'),
        }
    
    def _load_model(self, src: str, tgt: str):
        model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return {'model': model, 'tokenizer': tokenizer}
    
    def back_translate(self, text: str, intermediate_lang: str = 'fr') -> str:
        """Translate en->intermediate->en"""
        
        # Forward translation
        forward_pair = f'en-{intermediate_lang}'
        inputs = self.models[forward_pair]['tokenizer'](text, return_tensors="pt", padding=True)
        translated = self.models[forward_pair]['model'].generate(**inputs)
        intermediate = self.models[forward_pair]['tokenizer'].decode(translated[0], skip_special_tokens=True)
        
        # Backward translation
        backward_pair = f'{intermediate_lang}-en'
        inputs = self.models[backward_pair]['tokenizer'](intermediate, return_tensors="pt", padding=True)
        back_translated = self.models[backward_pair]['model'].generate(**inputs)
        result = self.models[backward_pair]['tokenizer'].decode(back_translated[0], skip_special_tokens=True)
        
        return result
    
    def augment(self, text: str, num_variations: int = 2) -> List[str]:
        """Generate back-translation variations"""
        variations = []
        languages = ['fr', 'de', 'es'][:num_variations]
        
        for lang in languages:
            try:
                variation = self.back_translate(text, lang)
                if variation != text:  # Skip identical results
                    variations.append(variation)
            except:
                continue
        
        return variations
```

#### Pros & Cons
✅ **Pros**: High linguistic diversity, preserves core meaning, introduces naturalistic variations  
❌ **Cons**: Can introduce grammatical errors, may lose domain-specific terminology

---

### 4. **LLM-Based Augmentation** (High quality, 3x diversity)

#### Method
Use instruction-tuned LLMs (GPT-3.5, Llama-2, Mistral) to generate diverse question formulations.

**Prompt Template:**
```
You are generating natural language questions for a spatial database query. Given the SQL query and metadata, generate 3 diverse natural language questions that would result in this query.

SQL Query:
{sql}

Query Type: {sql_type}
Tables: {tables}
Spatial Functions: {functions}
Difficulty: {difficulty}

Generate 3 questions with different tones:
1. Direct/Imperative (e.g., "Find all buildings...")
2. Interrogative (e.g., "What are the buildings that...")
3. Analytical (e.g., "Analyze the distribution of buildings...")

Ensure questions are:
- Spatially explicit (mention spatial relationships)
- Schema-aware (use correct table/column names)
- Contextually rich (include filters, conditions)
- Varied in complexity
```

#### Implementation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMAugmenter:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_questions(self, sql: str, metadata: Dict, num_variations: int = 3) -> List[str]:
        """Generate diverse NL questions using LLM"""
        
        prompt = f"""You are generating natural language questions for a spatial database query.

SQL Query:
{sql}

Query Type: {metadata['sql_type']}
Tables: {', '.join(metadata['tables'])}
Spatial Functions: {', '.join(metadata['spatial_functions'])}
Difficulty: {metadata['difficulty']}

Generate {num_variations} natural language questions with different tones that would result in this SQL query.

Questions:
1."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse questions from response
        questions = self._parse_questions(response)
        
        return questions[:num_variations]
    
    def _parse_questions(self, response: str) -> List[str]:
        """Extract numbered questions from LLM response"""
        lines = response.split('\n')
        questions = []
        
        for line in lines:
            # Match patterns like "1. Question text" or "Question: text"
            if re.match(r'^\d+\.\s+', line) or line.startswith('Question:'):
                question = re.sub(r'^\d+\.\s+|^Question:\s*', '', line).strip()
                if len(question) > 20:  # Minimum length filter
                    questions.append(question)
        
        return questions
```

#### Pros & Cons
✅ **Pros**: Highest quality, most natural, contextually rich, diverse tones  
❌ **Cons**: Computationally expensive, requires GPU, potential hallucination, API costs (if using GPT)

---

### 5. **Compositional Augmentation** (Structural diversity, 2x diversity)

#### Method
Modify question structure by adding/removing linguistic components while preserving SQL semantics.

**Transformations:**
- **Clause Addition**: Add temporal/conditional context
  - `"Find buildings"` → `"Find buildings built after 1990"`
- **Perspective Shift**: Change question perspective
  - `"Which buildings intersect parks?"` → `"For each park, which buildings are inside?"`
- **Granularity Change**: Adjust specificity level
  - `"Count buildings"` → `"Count residential buildings in each district"`
- **Formality Shift**: Change formality level
  - `"Show me buildings"` → `"I would like to retrieve all building records"`

#### Implementation
```python
class CompositionalAugmenter:
    def __init__(self):
        self.temporal_additions = [
            "from the last 5 years",
            "built after {year}",
            "in the current scenario",
            "historically"
        ]
        
        self.formality_transforms = {
            "Find": ["Retrieve", "Identify", "List", "Locate"],
            "Show": ["Display", "Present", "Provide", "Return"],
            "Get": ["Obtain", "Fetch", "Extract", "Acquire"],
            "Count": ["Enumerate", "Tally", "Calculate the number of"]
        }
    
    def augment(self, question: str, metadata: Dict) -> List[str]:
        """Apply compositional transformations"""
        
        variations = []
        
        # Add temporal context
        if '{year}' not in question:
            year = random.choice([1990, 2000, 2010, 2020])
            temporal = random.choice(self.temporal_additions)
            variations.append(f"{question} {temporal.format(year=year)}")
        
        # Formality shift
        for informal, formal_list in self.formality_transforms.items():
            if informal in question:
                for formal in random.sample(formal_list, min(1, len(formal_list))):
                    variations.append(question.replace(informal, formal, 1))
        
        # Add specificity
        if metadata['sql_type'] == 'AGGREGATION':
            variations.append(f"{question} grouped by region")
        
        return variations[:2]  # Limit to 2 variations
```

#### Pros & Cons
✅ **Pros**: Controlled diversity, preserves SQL alignment, enhances contextual richness  
❌ **Cons**: Limited scope, requires careful rule design

---

## 📊 Combined Pipeline: 10x Diversity

**Augmentation Mix:**
- Template-Based: 2x (baseline)
- Paraphrasing: 3x
- Back-Translation: 2x
- LLM-Based: 3x
- Compositional: 2x

**Total Multiplier:** 2 + 3 + 2 + 3 + 2 = **12x diversity** (exceeds 10x target)

### Diversity Distribution Strategy
For each Stage 2 SQL query:
1. Generate 2 template variations (fast baseline)
2. Select best template → 3 paraphrases (semantic preservation)
3. Select best paraphrase → 2 back-translations (linguistic diversity)
4. Use original SQL → 3 LLM-generated questions (high quality)
5. Apply compositional transformations to 2 best results

**Quality Filtering:**
- Remove duplicates (exact + semantic similarity < 0.95)
- Filter by length (20-300 characters)
- Validate grammar (language_tool_python)
- Ensure spatial terminology present

---

## 🛠️ Implementation: `stage3_augmentation_pipeline.py`

### Main Pipeline Function
```python
def run_stage3_pipeline(
    stage2_file: str,
    output_file: str,
    target_multiplier: int = 10,
    use_llm: bool = True,
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
) -> List[Dict]:
    """
    Execute Stage 3 NL augmentation pipeline
    
    Args:
        stage2_file: Stage 2 synthetic SQL dataset
        output_file: Output JSONL file
        target_multiplier: Target diversity multiplier (10x default)
        use_llm: Whether to use LLM augmentation (GPU required)
        llm_model: LLM model for generation
    
    Returns:
        List of augmented (SQL, NL) pairs
    """
    
    # Initialize augmenters
    template_aug = TemplateAugmenter()
    paraphrase_aug = ParaphraseAugmenter()
    backtrans_aug = BackTranslator()
    llm_aug = LLMAugmenter(llm_model) if use_llm else None
    comp_aug = CompositionalAugmenter()
    
    # Load Stage 2 data
    stage2_samples = load_jsonl(stage2_file)
    
    augmented_samples = []
    
    for i, sample in enumerate(stage2_samples):
        # Stage 2 samples have placeholder questions - generate real ones
        sql = sample['sql_postgis']
        metadata = extract_metadata(sample)
        
        # Apply all augmentation strategies
        variations = []
        
        # 1. Template (2x)
        variations.extend(template_aug.augment(sql, metadata, num=2))
        
        # 2. Paraphrase best template (3x)
        if variations:
            best_template = variations[0]
            variations.extend(paraphrase_aug.augment(best_template, num=3))
        
        # 3. Back-translate (2x)
        if len(variations) >= 3:
            variations.extend(backtrans_aug.augment(variations[2], num=2))
        
        # 4. LLM generation (3x)
        if use_llm and llm_aug:
            variations.extend(llm_aug.generate_questions(sql, metadata, num=3))
        
        # 5. Compositional (2x)
        if variations:
            variations.extend(comp_aug.augment(variations[0], metadata))
        
        # Filter and deduplicate
        variations = filter_quality(variations, metadata)
        variations = deduplicate_semantic(variations, threshold=0.95)
        
        # Create augmented samples
        for var_idx, question in enumerate(variations[:target_multiplier]):
            aug_sample = sample.copy()
            aug_sample['id'] = f"{sample['id']}_aug{var_idx:02d}"
            aug_sample['question'] = question
            aug_sample['question_tone'] = classify_tone(question)
            aug_sample['augmentation_method'] = "multi_strategy"
            aug_sample['variation_index'] = var_idx
            
            augmented_samples.append(aug_sample)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(stage2_samples)} samples...")
    
    # Save
    save_jsonl(augmented_samples, output_file)
    
    return augmented_samples
```

---

## 🎯 Quality Control Mechanisms

### 1. Semantic Similarity Validation
Ensure augmented questions are semantically aligned with SQL:
```python
def validate_semantic_alignment(question: str, sql: str, metadata: Dict) -> float:
    """
    Score semantic alignment between NL and SQL (0.0-1.0)
    Check if key SQL components are mentioned in question
    """
    score = 0.0
    
    # Check table mentions
    tables = metadata['tables']
    for table in tables:
        table_name = table.split('.')[-1]
        if table_name in question.lower():
            score += 0.3 / len(tables)
    
    # Check spatial function mentions
    spatial_keywords = ['intersect', 'within', 'contain', 'near', 'distance', 'area']
    for keyword in spatial_keywords:
        if keyword in question.lower():
            score += 0.2
            break
    
    # Check condition mentions (WHERE clause)
    if 'project_id' in sql and 'project' in question.lower():
        score += 0.2
    
    # Check aggregation mentions
    if 'COUNT' in sql and ('count' in question.lower() or 'how many' in question.lower()):
        score += 0.3
    
    return min(score, 1.0)
```

### 2. Grammatical Validation
```python
import language_tool_python

def check_grammar(text: str) -> Tuple[bool, int]:
    """Check grammar and return (is_valid, error_count)"""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    
    # Allow minor errors
    is_valid = len(matches) <= 2
    return is_valid, len(matches)
```

### 3. Diversity Metrics
```python
def calculate_diversity_score(questions: List[str]) -> float:
    """
    Calculate lexical diversity using Type-Token Ratio (TTR)
    and average pairwise cosine distance
    """
    # Lexical diversity
    all_words = []
    for q in questions:
        all_words.extend(q.lower().split())
    
    ttr = len(set(all_words)) / len(all_words) if all_words else 0
    
    # Semantic diversity
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(questions)
    
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = 1 - util.cos_sim(embeddings[i], embeddings[j]).item()
            distances.append(dist)
    
    avg_distance = np.mean(distances) if distances else 0
    
    # Combined score
    diversity_score = 0.4 * ttr + 0.6 * avg_distance
    
    return diversity_score
```

---

## 📈 Expected Outcomes

**Stage 3 Output:**
- **Dataset Size**: 500,000 - 1,000,000 (SQL, NL) pairs
- **Diversity**: 10-12x multiplier per SQL query
- **Quality**: Average semantic alignment > 0.80
- **Coverage**: Diverse question tones, complexity levels, spatial terminology

**Dataset Structure (per sample):**
```json
{
  "id": "cim_stage2_000001_aug05",
  "database_id": 1,
  "database_name": "cim_wizard",
  
  "question": "Which residential buildings in milan_smart_district intersect with parks?",
  "question_tone": "INTERROGATIVE",
  "augmentation_method": "paraphrase+backtrans",
  "semantic_alignment_score": 0.87,
  "variation_index": 5,
  
  "sql_postgis": "SELECT b.building_id FROM cim_vector.building b...",
  "sql_spatialite": "...",
  
  "sql_type": "SPATIAL_JOIN",
  "difficulty": {...},
  "usage_frequency": "CRITICAL",
  
  "database_schema": {...},
  "spatial_functions": ["ST_Intersects"],
  
  "instruction": "Convert this natural language question to PostGIS spatial SQL...",
  "results": [],
  "stage": "stage3_augmented"
}
```

---

## 🚀 Next Steps After Stage 3

1. **Dataset Validation**: Sample 1000 pairs for human evaluation
2. **Execute Evaluation Subset**: Run SQL queries to fill `results` field
3. **Train-Test Split**: 90% training / 10% validation
4. **LLM Fine-tuning**: Fine-tune Code-Llama-7B or StarCoder on dataset
5. **Evaluation**: Measure Execution Accuracy (EX) on test set

---

## 🔧 Required Libraries

```bash
# Stage 3 dependencies
pip install sentence-transformers==2.2.2
pip install transformers==4.36.0
pip install torch==2.1.0
pip install sacremoses==0.0.53
pip install language-tool-python==2.7.1
pip install nltk==3.8.1

# Download NLTK data
python -m nltk.downloader punkt averaged_perceptron_tagger wordnet
```

---

## ⏱️ Estimated Execution Time

**Configuration**:
- Stage 2 input: 50,000 SQL queries
- Target multiplier: 10x
- Use LLM: Yes (with GPU)

**Time Estimates per 1000 samples:**
- Template: 2 minutes
- Paraphrasing: 15 minutes (GPU)
- Back-translation: 30 minutes
- LLM generation: 45 minutes (GPU, batch size 4)
- Compositional: 3 minutes

**Total for 50K samples**: ~40-60 hours (with parallelization: 10-15 hours)

**Optimization**: Batch processing, GPU acceleration, parallel augmentation strategies

---

This completes the Stage 3 plan. Ready for implementation!

