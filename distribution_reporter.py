#!/usr/bin/env python3
"""
Dataset Distribution Reporter
==============================

Analyzes JSONL datasets to provide distribution reports based on task and domain taxonomies.
Works with both benchmark and training datasets.

Features:
- Task type distribution
- Domain type distribution
- Complexity level distribution
- Frequency level distribution
- Question tone distribution
- Cross-tabulation analysis
- ASCII bar chart visualizations

Usage:
    # Analyze benchmark dataset
    python distribution_reporter.py --input ../ai4db/ftv2_evaluation_benchmark_100.jsonl

    # Analyze training dataset
    python distribution_reporter.py --input ../ai4db/ftv2_training_data.jsonl --sql_field sql

    # Save report to file
    python distribution_reporter.py --input dataset.jsonl --output distribution_report.txt

Author: Ali Taherdoust
Date: November 2025
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime


# ============================================================================
# TASK TAXONOMY
# Complexity: 1=Easy, 2=Medium, 3=Hard
# Frequency: 1=Very Frequent, 2=Frequent, 3=Rare
# ============================================================================

TASK_TAXONOMY = {
    # Non-spatial SQL operations
    "SIMPLE_SELECT": {"complexity": 1, "frequency": 1, "description": "Simple SELECT with WHERE, no spatial operations"},
    "SQL_AGGREGATION": {"complexity": 1, "frequency": 1, "description": "Aggregation (COUNT, SUM, AVG) with GROUP BY, no spatial"},
    "SQL_JOIN": {"complexity": 2, "frequency": 2, "description": "Standard table join without spatial predicates"},
    "MULTI_SQL_JOIN": {"complexity": 3, "frequency": 3, "description": "Multiple table joins (3+ tables)"},
    "NESTED_QUERY": {"complexity": 3, "frequency": 3, "description": "CTEs, subqueries, nested SELECT"},
    
    # Spatial predicates (relationships) - most_frequent
    "SPATIAL_PREDICATE": {"complexity": 1, "frequency": 1, "description": "Spatial predicates (ST_Intersects, ST_Contains, ST_Within, ST_Touches)"},
    "SPATIAL_PREDICATE_DISTANCE": {"complexity": 2, "frequency": 2, "description": "Distance-based predicates (ST_DWithin, ST_Overlaps, ST_Crosses, ST_Disjoint)"},
    
    # Spatial measurements - most_frequent
    "SPATIAL_MEASUREMENT": {"complexity": 1, "frequency": 1, "description": "Basic measurements (ST_Area, ST_Distance, ST_Length, ST_Perimeter)"},
    
    # Spatial processing - frequent to most_frequent
    "SPATIAL_PROCESSING": {"complexity": 2, "frequency": 1, "description": "Spatial processing (ST_Buffer, ST_Union, ST_Intersection, ST_Difference)"},
    
    # Spatial accessors - frequent
    "SPATIAL_ACCESSOR": {"complexity": 1, "frequency": 2, "description": "Coordinate extraction (ST_X, ST_Y, ST_Centroid, ST_Envelope)"},
    
    # Spatial constructors - most_frequent
    "SPATIAL_CONSTRUCTOR": {"complexity": 1, "frequency": 1, "description": "Geometry construction (ST_MakePoint, ST_GeomFromText, ST_Collect)"},
    
    # Spatial transforms - most_frequent
    "SPATIAL_TRANSFORM": {"complexity": 2, "frequency": 1, "description": "Coordinate transformation (ST_Transform, ST_SetSRID)"},
    
    # Spatial validation - most_frequent
    "SPATIAL_VALIDATION": {"complexity": 2, "frequency": 1, "description": "Geometry validation (ST_IsValid, ST_MakeValid)"},
    
    # Spatial joins
    "SPATIAL_JOIN": {"complexity": 2, "frequency": 1, "description": "Join using spatial predicates"},
    "MULTI_SPATIAL_JOIN": {"complexity": 3, "frequency": 3, "description": "Multiple spatial joins with complex predicates"},
    
    # Advanced spatial operations - low_frequent
    "SPATIAL_CLUSTERING": {"complexity": 3, "frequency": 3, "description": "Spatial clustering (ST_ClusterDBSCAN, ST_ClusterKMeans)"},
    
    # Raster operations - frequent
    "RASTER_ANALYSIS": {"complexity": 3, "frequency": 2, "description": "Raster analysis and raster_accessor functions (ST_Value, ST_SummaryStats)"},
    "RASTER_VECTOR": {"complexity": 3, "frequency": 3, "description": "Raster-vector integration (ST_Clip, ST_Intersection with raster)"}
}

# ============================================================================
# DOMAIN TAXONOMY (CIM Wizard Schema Complexity)
# ============================================================================

DOMAIN_TAXONOMY = {
    "SINGLE_SCHEMA_CIM_VECTOR": {"complexity": 1, "frequency": 1, "description": "Single schema cim_vector only"},
    "MULTI_SCHEMA_WITH_CIM_VECTOR": {"complexity": 2, "frequency": 2, "description": "cim_vector + one other schema (census/network/raster)"},
    "SINGLE_SCHEMA_OTHER": {"complexity": 1, "frequency": 2, "description": "Single non-vector schema (census/network/raster only)"},
    "MULTI_SCHEMA_WITHOUT_CIM_VECTOR": {"complexity": 2, "frequency": 3, "description": "Multiple schemas without cim_vector"},
    "MULTI_SCHEMA_COMPLEX": {"complexity": 3, "frequency": 3, "description": "Three or more schemas combined"}
}

# ============================================================================
# QUESTION TONES
# ============================================================================

QUESTION_TONES = {
    "INTERROGATIVE": "Questions starting with what, which, where, how, etc.",
    "DIRECT": "Imperative statements (find, get, list, show, etc.)",
    "DESCRIPTIVE": "Other descriptive requests"
}

# ============================================================================
# SPATIAL FUNCTION PATTERNS FOR CLASSIFICATION
# ============================================================================

SPATIAL_PATTERNS = {
    "predicates": [
        r'ST_Intersects', r'ST_Contains', r'ST_Within', r'ST_Touches',
        r'ST_Equals', r'ST_Covers', r'ST_CoveredBy'
    ],
    "predicates_distance": [
        r'ST_DWithin', r'ST_Overlaps', r'ST_Crosses', r'ST_Disjoint'
    ],
    "measurements": [
        r'ST_Area', r'ST_Distance', r'ST_Length', r'ST_Perimeter',
        r'ST_3DDistance', r'ST_MaxDistance'
    ],
    "processing": [
        r'ST_Buffer', r'ST_Union', r'ST_Intersection(?!_Raster)', r'ST_Difference',
        r'ST_SymDifference', r'ST_ConvexHull', r'ST_Simplify'
    ],
    "accessors": [
        r'ST_X', r'ST_Y', r'ST_Z', r'ST_Centroid', r'ST_Envelope',
        r'ST_StartPoint', r'ST_EndPoint', r'ST_PointN', r'ST_GeometryN',
        r'ST_NumGeometries', r'ST_NumPoints', r'ST_SRID'
    ],
    "constructors": [
        r'ST_MakePoint', r'ST_GeomFromText', r'ST_Collect', r'ST_MakeLine',
        r'ST_MakePolygon', r'ST_GeomFromGeoJSON', r'ST_Point', r'ST_Polygon'
    ],
    "transforms": [
        r'ST_Transform', r'ST_SetSRID', r'ST_FlipCoordinates'
    ],
    "validation": [
        r'ST_IsValid', r'ST_MakeValid', r'ST_IsSimple', r'ST_IsClosed'
    ],
    "clustering": [
        r'ST_ClusterDBSCAN', r'ST_ClusterKMeans', r'ST_ClusterWithin'
    ],
    "raster_analysis": [
        r'ST_Value', r'ST_SummaryStats', r'ST_Histogram', r'ST_Band',
        r'ST_BandMetaData', r'ST_RasterToWorldCoord'
    ],
    "raster_vector": [
        r'ST_Clip', r'ST_Intersection_Raster', r'ST_AsRaster', r'ST_Resample'
    ]
}

# Schema patterns
SCHEMA_PATTERNS = {
    "cim_vector": [r'cim_vector\.', r'FROM\s+cim_vector', r'JOIN\s+cim_vector'],
    "cim_census": [r'cim_census\.', r'FROM\s+cim_census', r'JOIN\s+cim_census'],
    "cim_network": [r'cim_network\.', r'FROM\s+cim_network', r'JOIN\s+cim_network'],
    "cim_raster": [r'cim_raster\.', r'FROM\s+cim_raster', r'JOIN\s+cim_raster']
}


# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_task_type(sql: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classify SQL query into task taxonomy.
    Returns (task_type, metadata) where metadata includes complexity and frequency.
    """
    sql_upper = sql.upper()
    
    # Check for raster operations first (most specific)
    raster_vector_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["raster_vector"])
    raster_analysis_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["raster_analysis"])
    
    if raster_vector_found:
        return "RASTER_VECTOR", TASK_TAXONOMY["RASTER_VECTOR"]
    if raster_analysis_found:
        return "RASTER_ANALYSIS", TASK_TAXONOMY["RASTER_ANALYSIS"]
    
    # Check for clustering
    clustering_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["clustering"])
    if clustering_found:
        return "SPATIAL_CLUSTERING", TASK_TAXONOMY["SPATIAL_CLUSTERING"]
    
    # Count spatial joins (joins with spatial predicates)
    join_count = len(re.findall(r'\bJOIN\b', sql_upper))
    spatial_predicates = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["predicates"])
    spatial_predicates_dist = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["predicates_distance"])
    
    # Check for spatial joins
    if join_count >= 2 and (spatial_predicates or spatial_predicates_dist):
        return "MULTI_SPATIAL_JOIN", TASK_TAXONOMY["MULTI_SPATIAL_JOIN"]
    if join_count >= 1 and (spatial_predicates or spatial_predicates_dist):
        return "SPATIAL_JOIN", TASK_TAXONOMY["SPATIAL_JOIN"]
    
    # Check for validation
    validation_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["validation"])
    if validation_found:
        return "SPATIAL_VALIDATION", TASK_TAXONOMY["SPATIAL_VALIDATION"]
    
    # Check for transforms
    transform_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["transforms"])
    if transform_found:
        return "SPATIAL_TRANSFORM", TASK_TAXONOMY["SPATIAL_TRANSFORM"]
    
    # Check for constructors
    constructor_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["constructors"])
    if constructor_found:
        return "SPATIAL_CONSTRUCTOR", TASK_TAXONOMY["SPATIAL_CONSTRUCTOR"]
    
    # Check for accessors
    accessor_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["accessors"])
    if accessor_found:
        return "SPATIAL_ACCESSOR", TASK_TAXONOMY["SPATIAL_ACCESSOR"]
    
    # Check for processing
    processing_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["processing"])
    if processing_found:
        return "SPATIAL_PROCESSING", TASK_TAXONOMY["SPATIAL_PROCESSING"]
    
    # Check for measurements
    measurement_found = any(re.search(p, sql, re.IGNORECASE) for p in SPATIAL_PATTERNS["measurements"])
    if measurement_found:
        return "SPATIAL_MEASUREMENT", TASK_TAXONOMY["SPATIAL_MEASUREMENT"]
    
    # Check for distance predicates
    if spatial_predicates_dist:
        return "SPATIAL_PREDICATE_DISTANCE", TASK_TAXONOMY["SPATIAL_PREDICATE_DISTANCE"]
    
    # Check for basic predicates
    if spatial_predicates:
        return "SPATIAL_PREDICATE", TASK_TAXONOMY["SPATIAL_PREDICATE"]
    
    # Non-spatial SQL operations
    has_cte = 'WITH' in sql_upper and 'AS' in sql_upper
    has_subquery = sql_upper.count('SELECT') > 1
    if has_cte or has_subquery:
        return "NESTED_QUERY", TASK_TAXONOMY["NESTED_QUERY"]
    
    if join_count >= 2:
        return "MULTI_SQL_JOIN", TASK_TAXONOMY["MULTI_SQL_JOIN"]
    
    if join_count == 1:
        return "SQL_JOIN", TASK_TAXONOMY["SQL_JOIN"]
    
    # Check for aggregation
    aggregation_pattern = r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\('
    has_aggregation = re.search(aggregation_pattern, sql_upper)
    has_group_by = 'GROUP BY' in sql_upper
    if has_aggregation or has_group_by:
        return "SQL_AGGREGATION", TASK_TAXONOMY["SQL_AGGREGATION"]
    
    # Default to simple select
    return "SIMPLE_SELECT", TASK_TAXONOMY["SIMPLE_SELECT"]


def classify_domain_type(sql: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classify SQL query into domain taxonomy based on schema usage.
    Returns (domain_type, metadata) where metadata includes complexity and frequency.
    """
    schemas_used = set()
    
    for schema_name, patterns in SCHEMA_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                schemas_used.add(schema_name)
                break
    
    num_schemas = len(schemas_used)
    has_cim_vector = "cim_vector" in schemas_used
    
    if num_schemas >= 3:
        return "MULTI_SCHEMA_COMPLEX", DOMAIN_TAXONOMY["MULTI_SCHEMA_COMPLEX"]
    
    if num_schemas == 2:
        if has_cim_vector:
            return "MULTI_SCHEMA_WITH_CIM_VECTOR", DOMAIN_TAXONOMY["MULTI_SCHEMA_WITH_CIM_VECTOR"]
        else:
            return "MULTI_SCHEMA_WITHOUT_CIM_VECTOR", DOMAIN_TAXONOMY["MULTI_SCHEMA_WITHOUT_CIM_VECTOR"]
    
    if num_schemas == 1:
        if has_cim_vector:
            return "SINGLE_SCHEMA_CIM_VECTOR", DOMAIN_TAXONOMY["SINGLE_SCHEMA_CIM_VECTOR"]
        else:
            return "SINGLE_SCHEMA_OTHER", DOMAIN_TAXONOMY["SINGLE_SCHEMA_OTHER"]
    
    # Default to single schema cim_vector if no schema detected
    return "SINGLE_SCHEMA_CIM_VECTOR", DOMAIN_TAXONOMY["SINGLE_SCHEMA_CIM_VECTOR"]


def classify_question_tone(question: str) -> str:
    """Classify question tone based on structure."""
    question_lower = question.lower().strip()
    
    # Interrogative patterns
    interrogative_starts = ['what', 'which', 'where', 'who', 'how', 'why', 'when', 'is', 'are', 'can', 'do', 'does']
    if any(question_lower.startswith(start) for start in interrogative_starts) or question_lower.endswith('?'):
        return "INTERROGATIVE"
    
    # Direct patterns (imperative)
    direct_starts = ['find', 'get', 'list', 'show', 'select', 'calculate', 'compute', 'return', 'retrieve', 'identify']
    if any(question_lower.startswith(start) for start in direct_starts):
        return "DIRECT"
    
    # Default to descriptive
    return "DESCRIPTIVE"


def get_schemas_used(sql: str) -> List[str]:
    """Extract list of schemas used in SQL query."""
    schemas_used = []
    for schema_name, patterns in SCHEMA_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                schemas_used.append(schema_name)
                break
    return schemas_used


def get_spatial_functions(sql: str) -> List[str]:
    """Extract list of spatial functions used in SQL query."""
    return re.findall(r'ST_\w+', sql, re.IGNORECASE)


def classify_sample(sql: str, question: str) -> Dict[str, Any]:
    """
    Classify a sample with full taxonomy information.
    Returns classification dictionary with task, domain, and question tone.
    """
    task_type, task_meta = classify_task_type(sql)
    domain_type, domain_meta = classify_domain_type(sql)
    question_tone = classify_question_tone(question)
    schemas = get_schemas_used(sql)
    spatial_funcs = get_spatial_functions(sql)
    
    return {
        "task_type": task_type,
        "task_complexity": task_meta["complexity"],
        "task_frequency": task_meta["frequency"],
        "task_description": task_meta["description"],
        "domain_type": domain_type,
        "domain_complexity": domain_meta["complexity"],
        "domain_frequency": domain_meta["frequency"],
        "domain_description": domain_meta["description"],
        "question_tone": question_tone,
        "schemas_used": schemas,
        "spatial_functions": spatial_funcs
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(input_path: Path, sql_field: str, question_field: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset and extract SQL and question fields."""
    print(f"Loading dataset from: {input_path}")
    samples = []
    skipped = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                
                # Try to find SQL field
                sql = None
                for field in [sql_field, 'sql_postgis', 'sql', 'query', 'target', 'output']:
                    if field in item and item[field]:
                        sql = item[field]
                        break
                
                # Try to find question field
                question = None
                for field in [question_field, 'question', 'input', 'prompt', 'text']:
                    if field in item and item[field]:
                        question = item[field]
                        break
                
                if sql:
                    samples.append({
                        'sql': sql,
                        'question': question or '',
                        'original': item
                    })
                else:
                    skipped += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                skipped += 1
    
    print(f"Loaded {len(samples)} samples")
    if skipped > 0:
        print(f"Skipped {skipped} items (missing SQL field or parse errors)")
    
    return samples


# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

class DistributionAnalyzer:
    """Analyzes and aggregates distribution statistics."""
    
    def __init__(self):
        self.task_types = Counter()
        self.domain_types = Counter()
        self.task_complexity = Counter()
        self.task_frequency = Counter()
        self.domain_complexity = Counter()
        self.domain_frequency = Counter()
        self.question_tones = Counter()
        self.schemas = Counter()
        self.spatial_functions = Counter()
        
        # Cross-tabulation
        self.task_domain_cross = defaultdict(Counter)
        self.complexity_cross = defaultdict(Counter)  # task_complexity -> domain_complexity
        
        self.total_samples = 0
        self.classifications = []
    
    def add_sample(self, classification: Dict[str, Any]):
        """Add a sample's classification to the analysis."""
        self.total_samples += 1
        self.classifications.append(classification)
        
        self.task_types[classification['task_type']] += 1
        self.domain_types[classification['domain_type']] += 1
        self.task_complexity[classification['task_complexity']] += 1
        self.task_frequency[classification['task_frequency']] += 1
        self.domain_complexity[classification['domain_complexity']] += 1
        self.domain_frequency[classification['domain_frequency']] += 1
        self.question_tones[classification['question_tone']] += 1
        
        for schema in classification['schemas_used']:
            self.schemas[schema] += 1
        
        for func in classification['spatial_functions']:
            self.spatial_functions[func.upper()] += 1
        
        # Cross-tabulation
        self.task_domain_cross[classification['task_type']][classification['domain_type']] += 1
        self.complexity_cross[classification['task_complexity']][classification['domain_complexity']] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete distribution summary."""
        return {
            'total_samples': self.total_samples,
            'task_types': dict(self.task_types),
            'domain_types': dict(self.domain_types),
            'task_complexity': dict(self.task_complexity),
            'task_frequency': dict(self.task_frequency),
            'domain_complexity': dict(self.domain_complexity),
            'domain_frequency': dict(self.domain_frequency),
            'question_tones': dict(self.question_tones),
            'schemas': dict(self.schemas),
            'spatial_functions': dict(self.spatial_functions.most_common(20)),
            'task_domain_cross': {k: dict(v) for k, v in self.task_domain_cross.items()},
            'complexity_cross': {k: dict(v) for k, v in self.complexity_cross.items()}
        }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def create_bar(count: int, total: int, width: int = 30) -> str:
    """Create ASCII bar chart element."""
    if total == 0:
        return ''
    ratio = count / total
    filled = int(ratio * width)
    return '█' * filled + '░' * (width - filled)


def format_distribution_table(title: str, counter: Counter, total: int, 
                              descriptions: Optional[Dict] = None,
                              show_bar: bool = True) -> List[str]:
    """Format a distribution as a table with optional bar chart."""
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 80)
    
    if not counter:
        lines.append("  No data available")
        return lines
    
    # Sort by count descending
    sorted_items = counter.most_common()
    
    max_label_len = max(len(str(k)) for k, _ in sorted_items)
    max_label_len = max(max_label_len, 25)
    
    if show_bar:
        lines.append(f"  {'Category':<{max_label_len}} │ {'Count':>6} │ {'%':>6} │ Distribution")
        lines.append(f"  {'-'*max_label_len}─┼{'-'*8}┼{'-'*8}┼{'-'*32}")
    else:
        lines.append(f"  {'Category':<{max_label_len}} │ {'Count':>6} │ {'%':>6}")
        lines.append(f"  {'-'*max_label_len}─┼{'-'*8}┼{'-'*8}")
    
    for key, count in sorted_items:
        pct = (count / total * 100) if total > 0 else 0
        if show_bar:
            bar = create_bar(count, total, 25)
            lines.append(f"  {str(key):<{max_label_len}} │ {count:>6} │ {pct:>5.1f}% │ {bar}")
        else:
            lines.append(f"  {str(key):<{max_label_len}} │ {count:>6} │ {pct:>5.1f}%")
    
    # Add descriptions if provided
    if descriptions:
        lines.append("")
        lines.append("  Descriptions:")
        for key, _ in sorted_items:
            if key in descriptions:
                desc = descriptions[key]
                if isinstance(desc, dict):
                    desc = desc.get('description', '')
                lines.append(f"    • {key}: {desc}")
    
    return lines


def format_complexity_labels(counter: Counter) -> Counter:
    """Convert numeric complexity/frequency levels to labeled strings."""
    complexity_labels = {1: "1_Easy", 2: "2_Medium", 3: "3_Hard"}
    return Counter({complexity_labels.get(k, str(k)): v for k, v in counter.items()})


def format_frequency_labels(counter: Counter) -> Counter:
    """Convert numeric frequency levels to labeled strings."""
    frequency_labels = {1: "1_Very_Frequent", 2: "2_Frequent", 3: "3_Rare"}
    return Counter({frequency_labels.get(k, str(k)): v for k, v in counter.items()})


def generate_report(analyzer: DistributionAnalyzer, input_path: str) -> str:
    """Generate comprehensive distribution report."""
    lines = []
    total = analyzer.total_samples
    
    # Header
    lines.append("╔" + "═" * 78 + "╗")
    lines.append("║" + " DATASET DISTRIBUTION REPORT ".center(78) + "║")
    lines.append("╚" + "═" * 78 + "╝")
    lines.append("")
    lines.append(f"  Input File: {input_path}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total Samples: {total}")
    
    # Summary statistics
    lines.append("\n" + "─" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("─" * 80)
    
    # Task type categories
    spatial_tasks = sum(1 for c in analyzer.classifications 
                       if c['task_type'].startswith('SPATIAL') or c['task_type'].startswith('RASTER'))
    non_spatial_tasks = total - spatial_tasks
    
    lines.append(f"\n  Task Categories:")
    lines.append(f"    • Spatial/Raster Tasks: {spatial_tasks} ({spatial_tasks/total*100:.1f}%)")
    lines.append(f"    • Non-Spatial SQL Tasks: {non_spatial_tasks} ({non_spatial_tasks/total*100:.1f}%)")
    
    # Schema usage
    lines.append(f"\n  Schema Usage:")
    for schema, count in analyzer.schemas.most_common():
        lines.append(f"    • {schema}: {count} samples ({count/total*100:.1f}%)")
    
    # Average complexity
    avg_task_complexity = sum(c['task_complexity'] for c in analyzer.classifications) / total if total > 0 else 0
    avg_domain_complexity = sum(c['domain_complexity'] for c in analyzer.classifications) / total if total > 0 else 0
    lines.append(f"\n  Average Complexity:")
    lines.append(f"    • Task Complexity: {avg_task_complexity:.2f} (1=Easy, 3=Hard)")
    lines.append(f"    • Domain Complexity: {avg_domain_complexity:.2f} (1=Easy, 3=Hard)")
    
    # Detailed distributions
    lines.extend(format_distribution_table(
        "TASK TYPE DISTRIBUTION",
        analyzer.task_types,
        total,
        TASK_TAXONOMY
    ))
    
    lines.extend(format_distribution_table(
        "DOMAIN TYPE DISTRIBUTION",
        analyzer.domain_types,
        total,
        DOMAIN_TAXONOMY
    ))
    
    lines.extend(format_distribution_table(
        "TASK COMPLEXITY DISTRIBUTION",
        format_complexity_labels(analyzer.task_complexity),
        total
    ))
    
    lines.extend(format_distribution_table(
        "TASK FREQUENCY DISTRIBUTION",
        format_frequency_labels(analyzer.task_frequency),
        total
    ))
    
    lines.extend(format_distribution_table(
        "DOMAIN COMPLEXITY DISTRIBUTION",
        format_complexity_labels(analyzer.domain_complexity),
        total
    ))
    
    lines.extend(format_distribution_table(
        "DOMAIN FREQUENCY DISTRIBUTION",
        format_frequency_labels(analyzer.domain_frequency),
        total
    ))
    
    lines.extend(format_distribution_table(
        "QUESTION TONE DISTRIBUTION",
        analyzer.question_tones,
        total,
        QUESTION_TONES
    ))
    
    # Top spatial functions
    if analyzer.spatial_functions:
        lines.append("\n" + "=" * 80)
        lines.append("TOP 15 SPATIAL FUNCTIONS USED")
        lines.append("=" * 80)
        max_func_count = max(analyzer.spatial_functions.values()) if analyzer.spatial_functions else 1
        for func, count in analyzer.spatial_functions.most_common(15):
            bar = create_bar(count, max_func_count, 20)
            lines.append(f"  {func:<25} │ {count:>5} │ {bar}")
    
    # Cross-tabulation: Task Type x Domain Type
    lines.append("\n" + "=" * 80)
    lines.append("CROSS-TABULATION: TASK TYPE × DOMAIN TYPE")
    lines.append("=" * 80)
    
    domain_order = list(analyzer.domain_types.keys())
    task_order = [t for t, _ in analyzer.task_types.most_common()]
    
    if domain_order and task_order:
        # Header row
        col_width = 8
        header = f"  {'Task Type':<25}"
        for domain in domain_order:
            short_domain = domain[:col_width]
            header += f" │ {short_domain:>{col_width}}"
        lines.append(header)
        lines.append("  " + "-" * 25 + ("─┼" + "-" * (col_width + 1)) * len(domain_order))
        
        # Data rows
        for task in task_order:
            row = f"  {task:<25}"
            for domain in domain_order:
                count = analyzer.task_domain_cross[task].get(domain, 0)
                row += f" │ {count:>{col_width}}"
            lines.append(row)
    
    # Complexity cross-tabulation
    lines.append("\n" + "=" * 80)
    lines.append("CROSS-TABULATION: TASK COMPLEXITY × DOMAIN COMPLEXITY")
    lines.append("=" * 80)
    
    complexity_labels = {1: "Easy", 2: "Medium", 3: "Hard"}
    header_label = "Task/Domain"
    lines.append(f"  {header_label:<15} │ {'Easy':>8} │ {'Medium':>8} │ {'Hard':>8} │ {'Total':>8}")
    lines.append("  " + "-" * 15 + "─┼" + "-" * 10 + "┼" + "-" * 10 + "┼" + "-" * 10 + "┼" + "-" * 10)
    
    for task_c in [1, 2, 3]:
        row = f"  {complexity_labels[task_c]:<15}"
        row_total = 0
        for domain_c in [1, 2, 3]:
            count = analyzer.complexity_cross[task_c].get(domain_c, 0)
            row_total += count
            row += f" │ {count:>8}"
        row += f" │ {row_total:>8}"
        lines.append(row)
    
    # Column totals
    lines.append("  " + "-" * 15 + "─┼" + "-" * 10 + "┼" + "-" * 10 + "┼" + "-" * 10 + "┼" + "-" * 10)
    totals_row = f"  {'Total':<15}"
    grand_total = 0
    for domain_c in [1, 2, 3]:
        col_total = sum(analyzer.complexity_cross[tc].get(domain_c, 0) for tc in [1, 2, 3])
        grand_total += col_total
        totals_row += f" │ {col_total:>8}"
    totals_row += f" │ {grand_total:>8}"
    lines.append(totals_row)
    
    # Coverage analysis
    lines.append("\n" + "=" * 80)
    lines.append("TAXONOMY COVERAGE ANALYSIS")
    lines.append("=" * 80)
    
    # Task types coverage
    covered_tasks = len(analyzer.task_types)
    total_tasks = len(TASK_TAXONOMY)
    missing_tasks = set(TASK_TAXONOMY.keys()) - set(analyzer.task_types.keys())
    
    lines.append(f"\n  Task Type Coverage: {covered_tasks}/{total_tasks} ({covered_tasks/total_tasks*100:.1f}%)")
    if missing_tasks:
        lines.append(f"  Missing Task Types:")
        for task in sorted(missing_tasks):
            lines.append(f"    • {task}: {TASK_TAXONOMY[task]['description']}")
    
    # Domain types coverage
    covered_domains = len(analyzer.domain_types)
    total_domains = len(DOMAIN_TAXONOMY)
    missing_domains = set(DOMAIN_TAXONOMY.keys()) - set(analyzer.domain_types.keys())
    
    lines.append(f"\n  Domain Type Coverage: {covered_domains}/{total_domains} ({covered_domains/total_domains*100:.1f}%)")
    if missing_domains:
        lines.append(f"  Missing Domain Types:")
        for domain in sorted(missing_domains):
            lines.append(f"    • {domain}: {DOMAIN_TAXONOMY[domain]['description']}")
    
    # Balance analysis
    lines.append("\n" + "=" * 80)
    lines.append("BALANCE ANALYSIS")
    lines.append("=" * 80)
    
    # Task type balance
    task_counts = list(analyzer.task_types.values())
    if task_counts:
        task_min = min(task_counts)
        task_max = max(task_counts)
        task_mean = sum(task_counts) / len(task_counts)
        task_std = (sum((x - task_mean) ** 2 for x in task_counts) / len(task_counts)) ** 0.5
        
        lines.append(f"\n  Task Type Distribution:")
        lines.append(f"    • Min samples per type: {task_min}")
        lines.append(f"    • Max samples per type: {task_max}")
        lines.append(f"    • Mean: {task_mean:.1f}")
        lines.append(f"    • Std Dev: {task_std:.1f}")
        lines.append(f"    • Imbalance Ratio (max/min): {task_max/task_min:.1f}x" if task_min > 0 else "")
    
    # Domain type balance
    domain_counts = list(analyzer.domain_types.values())
    if domain_counts:
        domain_min = min(domain_counts)
        domain_max = max(domain_counts)
        domain_mean = sum(domain_counts) / len(domain_counts)
        domain_std = (sum((x - domain_mean) ** 2 for x in domain_counts) / len(domain_counts)) ** 0.5
        
        lines.append(f"\n  Domain Type Distribution:")
        lines.append(f"    • Min samples per type: {domain_min}")
        lines.append(f"    • Max samples per type: {domain_max}")
        lines.append(f"    • Mean: {domain_mean:.1f}")
        lines.append(f"    • Std Dev: {domain_std:.1f}")
        lines.append(f"    • Imbalance Ratio (max/min): {domain_max/domain_min:.1f}x" if domain_min > 0 else "")
    
    lines.append("\n" + "═" * 80)
    lines.append("END OF REPORT")
    lines.append("═" * 80)
    
    return '\n'.join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze dataset distribution based on task and domain taxonomies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze benchmark dataset
    python distribution_reporter.py --input ../ai4db/ftv2_evaluation_benchmark_100.jsonl

    # Analyze training dataset with custom SQL field
    python distribution_reporter.py --input training_data.jsonl --sql_field sql

    # Save report to file
    python distribution_reporter.py --input dataset.jsonl --output report.txt

    # Export classifications as JSON
    python distribution_reporter.py --input dataset.jsonl --export_json classifications.json
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input JSONL dataset file')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for report (default: print to stdout)')
    parser.add_argument('--sql_field', type=str, default='sql_postgis',
                       help='Name of the SQL field in the dataset (default: sql_postgis)')
    parser.add_argument('--question_field', type=str, default='question',
                       help='Name of the question field in the dataset (default: question)')
    parser.add_argument('--export_json', type=str,
                       help='Export classifications and statistics as JSON')
    parser.add_argument('--export_classified', type=str,
                       help='Export dataset with classification labels as JSONL')
    
    args = parser.parse_args()
    
    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    samples = load_dataset(input_path, args.sql_field, args.question_field)
    
    if not samples:
        print("Error: No valid samples found in dataset")
        sys.exit(1)
    
    # Analyze distribution
    print("Analyzing dataset distribution...")
    analyzer = DistributionAnalyzer()
    
    for sample in samples:
        classification = classify_sample(sample['sql'], sample['question'])
        analyzer.add_sample(classification)
    
    # Generate report
    report = generate_report(analyzer, str(input_path))
    
    # Output report
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    else:
        print(report)
    
    # Export JSON if requested
    if args.export_json:
        json_path = Path(args.export_json)
        summary = analyzer.get_summary()
        summary['input_file'] = str(input_path)
        summary['generated_at'] = datetime.now().isoformat()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"JSON statistics exported to: {json_path}")
    
    # Export classified dataset if requested
    if args.export_classified:
        classified_path = Path(args.export_classified)
        with open(classified_path, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(samples):
                classification = analyzer.classifications[i]
                output_item = {
                    **sample['original'],
                    'classification': classification
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        print(f"Classified dataset exported to: {classified_path}")


if __name__ == '__main__':
    main()

