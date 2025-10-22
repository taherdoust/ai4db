#!/usr/bin/env python3
"""
stage2_sdv_pipeline_ipazia.py
Stage 2: SDV Synthetic SQL Generation - OPTIMIZED FOR IPAZIA MACHINE

Machine Specs (ipazia):
- CPU: 2x Intel Xeon Gold 6238R @ 2.20GHz (28 cores each = 56 cores total)
- RAM: 256GB
- GPU: Quadro RTX 6000/8000

Optimizations:
- Uses CTGAN with GPU acceleration (highest quality)
- Full 300 epochs training (2-4 hours with GPU)
- Large batch sizes leveraging 256GB RAM
- Parallel processing for SQL assembly
"""

import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# SDV imports
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    print("[WARNING] SDV not installed. Run: pip install sdv==1.9.0")
    SDV_AVAILABLE = False

# SQL parsing
try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except ImportError:
    print("[WARNING] sqlparse not installed. Run: pip install sqlparse==0.4.4")
    SQLPARSE_AVAILABLE = False

# GPU check
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA not available, will use CPU (slower)")
except ImportError:
    CUDA_AVAILABLE = False
    print("[WARNING] PyTorch not installed, GPU acceleration disabled")

# Import from stage1_cim.py (updated schema and parameters)
from stage1_cim import CIM_SCHEMAS, CIM_PARAMETERS, generate_realistic_values


# ============================================================================
# CIM WIZARD SCHEMA CONSTRAINTS (SAME AS ECLAB VERSION)
# ============================================================================

class CIMSchemaRules:
    """
    CIM Wizard schema validation rules and constraints
    Ensures synthetic SQL respects actual database schema
    """
    
    # Valid tables from CIM Wizard database (from stage1_cim.py)
    VALID_TABLES = [
        "cim_vector.cim_wizard_project_scenario",
        "cim_vector.cim_wizard_building",
        "cim_vector.cim_wizard_building_properties",
        "cim_network.network_buses",
        "cim_network.network_lines",
        "cim_network.network_scenarios",
        "cim_network.scenario_buses",
        "cim_network.scenario_lines",
        "cim_census.censusgeo",
        "cim_raster.dtm_raster",
        "cim_raster.dsm_raster",
        "cim_raster.dtm",
        "cim_raster.dsm_sansalva"
    ]
    
    # Valid join pairs (table1, table2) based on schema relationships
    VALID_JOINS = [
        ("cim_vector.cim_wizard_building", "cim_vector.cim_wizard_building_properties"),
        ("cim_vector.cim_wizard_building_properties", "cim_vector.cim_wizard_building"),
        ("cim_vector.cim_wizard_building_properties", "cim_vector.cim_wizard_project_scenario"),
        ("cim_vector.cim_wizard_project_scenario", "cim_vector.cim_wizard_building_properties"),
        ("cim_vector.cim_wizard_building", "cim_census.censusgeo"),
        ("cim_census.censusgeo", "cim_vector.cim_wizard_building"),
        ("cim_vector.cim_wizard_building", "cim_raster.dsm_raster"),
        ("cim_raster.dsm_raster", "cim_vector.cim_wizard_building"),
        ("cim_vector.cim_wizard_building", "cim_raster.dtm_raster"),
        ("cim_raster.dtm_raster", "cim_vector.cim_wizard_building"),
        ("cim_vector.cim_wizard_building", "cim_raster.dtm"),
        ("cim_raster.dtm", "cim_vector.cim_wizard_building"),
        ("cim_vector.cim_wizard_building", "cim_raster.dsm_sansalva"),
        ("cim_raster.dsm_sansalva", "cim_vector.cim_wizard_building"),
        ("cim_vector.cim_wizard_building_properties", "cim_network.network_buses"),
        ("cim_network.network_buses", "cim_vector.cim_wizard_building_properties"),
        ("cim_network.network_buses", "cim_network.network_lines"),
        ("cim_network.network_lines", "cim_network.network_buses"),
        ("cim_vector.cim_wizard_project_scenario", "cim_census.censusgeo"),
        ("cim_census.censusgeo", "cim_vector.cim_wizard_project_scenario"),
    ]
    
    # Join keys for valid joins
    JOIN_KEYS = {
        ("cim_vector.cim_wizard_building", "cim_vector.cim_wizard_building_properties"): ["building_id", "lod"],
        ("cim_vector.cim_wizard_building_properties", "cim_vector.cim_wizard_building"): ["building_id", "lod"],
        ("cim_vector.cim_wizard_building_properties", "cim_vector.cim_wizard_project_scenario"): ["project_id", "scenario_id"],
        ("cim_vector.cim_wizard_project_scenario", "cim_vector.cim_wizard_building_properties"): ["project_id", "scenario_id"],
        ("cim_network.network_buses", "cim_network.network_lines"): ["bus_id"],
        ("cim_network.network_lines", "cim_network.network_buses"): ["from_bus", "to_bus"],
    }
    
    # Geometry columns for spatial operations
    GEOMETRY_COLUMNS = {
        "cim_vector.cim_wizard_project_scenario": ["project_boundary", "project_center", "census_boundary"],
        "cim_vector.cim_wizard_building": ["building_geometry"],
        "cim_network.network_buses": ["geometry"],
        "cim_network.network_lines": ["geometry"],
        "cim_census.censusgeo": ["geometry"],
        "cim_raster.dsm_raster": ["rast"],
        "cim_raster.dtm_raster": ["rast"],
        "cim_raster.dtm": ["rast"],
        "cim_raster.dsm_sansalva": ["rast"],
    }
    
    # Spatial functions by geometry type (PostGIS in public schema)
    FUNCTION_APPLICABILITY = {
        "POLYGON": [
            "ST_Area", "ST_Intersects", "ST_Contains", "ST_Within", "ST_Touches",
            "ST_Overlaps", "ST_Centroid", "ST_Buffer", "ST_Union", "ST_Intersection",
            "ST_Difference", "ST_Boundary", "ST_ConvexHull", "ST_Simplify"
        ],
        "POINT": [
            "ST_X", "ST_Y", "ST_MakePoint", "ST_Distance", "ST_DWithin",
            "ST_Intersects", "ST_Within", "ST_Contains", "ST_Buffer",
            "ST_SetSRID", "ST_Transform", "ST_AsText"
        ],
        "LINESTRING": [
            "ST_Length", "ST_StartPoint", "ST_EndPoint", "ST_Intersects",
            "ST_Distance", "ST_Buffer", "ST_LineLocatePoint", "ST_LineInterpolatePoint",
            "ST_LineSubstring", "ST_Simplify"
        ],
        "RASTER": [
            "ST_Value", "ST_SummaryStats", "ST_Intersection", "ST_Intersects", "ST_Clip"
        ]
    }
    
    # Common spatial functions (work with most geometry types)
    COMMON_SPATIAL_FUNCTIONS = [
        "ST_Intersects", "ST_Distance", "ST_Buffer", "ST_SetSRID",
        "ST_Transform", "ST_IsValid", "ST_AsText", "ST_GeomFromText"
    ]
    
    # Table primary geometry types
    TABLE_GEOMETRY_TYPES = {
        "cim_vector.cim_wizard_building": "POLYGON",
        "cim_vector.cim_wizard_project_scenario": "POLYGON",
        "cim_network.network_buses": "POINT",
        "cim_network.network_lines": "LINESTRING",
        "cim_census.censusgeo": "POLYGON",
        "cim_raster.dsm_raster": "RASTER",
        "cim_raster.dtm_raster": "RASTER",
        "cim_raster.dtm": "RASTER",
        "cim_raster.dsm_sansalva": "RASTER",
    }


# ============================================================================
# FEATURE EXTRACTION FROM STAGE 1
# ============================================================================

def extract_features(stage1_samples: List[Dict]) -> pd.DataFrame:
    """
    Extract features from Stage 1 samples for CTGAN training
    
    Features (13 total):
    - 7 numerical: cte_count, join_count, subquery_count, spatial_function_count,
                   table_count, complexity_score, schema_count
    - 6 categorical: sql_type, difficulty_level, schema_complexity, usage_frequency,
                    question_tone, primary_function_category
    """
    
    records = []
    
    for sample in stage1_samples:
        # Extract structural features
        sql = sample['sql_postgis']
        sql_upper = sql.upper()
        
        # Numerical features
        cte_count = sql_upper.count('WITH')
        join_count = sql_upper.count('JOIN')
        subquery_count = sql.count('(SELECT')
        spatial_function_count = len(sample['spatial_functions'])
        table_count = len(sample['database_schema']['tables'])
        complexity_score = sample['difficulty']['complexity_score']
        schema_count = sample['database_schema']['schema_count']
        
        # Categorical features
        sql_type = sample['sql_type']
        difficulty_level = sample['difficulty']['overall_difficulty']
        schema_complexity = sample['difficulty']['schema_complexity']
        usage_frequency = sample['usage_frequency']
        question_tone = sample['question_tone']
        
        # Determine primary function category from spatial_function_details
        primary_category = "measurement"  # default
        if 'spatial_function_details' in sample and sample['spatial_function_details']:
            # Count categories
            category_counts = {}
            for func_detail in sample['spatial_function_details']:
                func_name = func_detail['name']
                # Categorize based on function name
                if any(x in func_name for x in ['AREA', 'LENGTH', 'DISTANCE', 'PERIMETER']):
                    cat = 'measurement'
                elif any(x in func_name for x in ['INTERSECTS', 'CONTAINS', 'WITHIN', 'TOUCHES', 'OVERLAPS', 'CROSSES', 'DISJOINT', 'DWITHIN']):
                    cat = 'predicates'
                elif any(x in func_name for x in ['BUFFER', 'UNION', 'INTERSECTION', 'DIFFERENCE', 'CONVEXHULL', 'SIMPLIFY']):
                    cat = 'processing'
                elif any(x in func_name for x in ['CENTROID', 'ENVELOPE', 'STARTPOINT', 'ENDPOINT', 'X', 'Y']):
                    cat = 'accessors'
                elif any(x in func_name for x in ['MAKEPOINT', 'GEOMFROMTEXT', 'COLLECT']):
                    cat = 'constructors'
                elif any(x in func_name for x in ['TRANSFORM', 'SETSRID']):
                    cat = 'transforms'
                elif any(x in func_name for x in ['ISVALID', 'MAKEVALID']):
                    cat = 'validation'
                elif any(x in func_name for x in ['CLUSTER']):
                    cat = 'clustering'
                elif any(x in func_name for x in ['VALUE', 'SUMMARYSTATS', 'CLIP']):
                    cat = 'raster'
                else:
                    cat = 'other'
                
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Get most common category
            if category_counts:
                primary_category = max(category_counts, key=category_counts.get)
        
        record = {
            # ID for tracking
            'sample_id': sample['id'],
            
            # Numerical features
            'cte_count': cte_count,
            'join_count': join_count,
            'subquery_count': subquery_count,
            'spatial_function_count': spatial_function_count,
            'table_count': table_count,
            'complexity_score': complexity_score,
            'schema_count': schema_count,
            
            # Categorical features
            'sql_type': sql_type,
            'difficulty_level': difficulty_level,
            'schema_complexity': schema_complexity,
            'usage_frequency': usage_frequency,
            'question_tone': question_tone,
            'primary_function_category': primary_category
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    return df


# ============================================================================
# CTGAN TRAINER (GPU-ACCELERATED FOR IPAZIA)
# ============================================================================

class CTGANTrainerIPAZIA:
    """Train CTGAN on Stage 1 features - GPU-accelerated for ipazia"""
    
    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: Whether to use GPU acceleration (default True on ipazia)
        """
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.synthesizer = None
        self.metadata = None
        
        if self.use_gpu:
            print(f"[OK] GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] GPU not available, using CPU (much slower)")
    
    def create_metadata(self, df: pd.DataFrame) -> SingleTableMetadata:
        """Create SDV metadata from DataFrame"""
        
        metadata = SingleTableMetadata()
        
        # Detect from dataframe
        metadata.detect_from_dataframe(df)
        
        # Set specific types
        numerical_cols = ['cte_count', 'join_count', 'subquery_count', 
                         'spatial_function_count', 'table_count', 
                         'complexity_score', 'schema_count']
        
        categorical_cols = ['sql_type', 'difficulty_level', 'schema_complexity',
                           'usage_frequency', 'question_tone', 'primary_function_category']
        
        for col in numerical_cols:
            metadata.update_column(col, sdtype='numerical')
        
        for col in categorical_cols:
            metadata.update_column(col, sdtype='categorical')
        
        # Set primary key
        metadata.set_primary_key('sample_id')
        
        return metadata
    
    def train(self, df: pd.DataFrame, epochs: int = 300):
        """
        Train CTGAN synthesizer on feature dataframe
        
        Args:
            df: DataFrame with extracted features
            epochs: Training epochs (300 recommended for high quality)
        """
        
        print(f"\n{'='*80}")
        print(f"Training CTGAN Synthesizer (ipazia - GPU-accelerated)")
        print(f"{'='*80}")
        print(f"Training data shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        # Create metadata
        self.metadata = self.create_metadata(df)
        
        # Initialize synthesizer
        print(f"\n[OK] Initializing CTGANSynthesizer...")
        print(f"  - Epochs: {epochs}")
        print(f"  - GPU: {self.use_gpu}")
        print(f"  - Batch size: 1000 (leveraging 256GB RAM)")
        print(f"  - Estimated training time: {'2-4 hours (GPU)' if self.use_gpu else '12-24 hours (CPU)'}")
        
        self.synthesizer = CTGANSynthesizer(
            metadata=self.metadata,
            epochs=epochs,
            batch_size=1000,  # Large batch size for ipazia's RAM
            generator_dim=(512, 512),  # Larger network for better quality
            discriminator_dim=(512, 512),
            generator_lr=2e-4,
            discriminator_lr=2e-4,
            discriminator_steps=1,
            verbose=True,
            cuda=self.use_gpu
        )
        
        # Train
        print(f"\n[OK] Training CTGAN (this will take 2-4 hours with GPU)...")
        start_time = datetime.now()
        
        self.synthesizer.fit(df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"\n[SUCCESS] Training complete!")
        print(f"   Duration: {duration:.1f} minutes ({duration/60:.1f} hours)")
        
    def generate(self, num_samples: int = 50000, batch_size: int = 10000) -> pd.DataFrame:
        """Generate synthetic samples in batches"""
        
        if self.synthesizer is None:
            raise ValueError("Synthesizer not trained! Call train() first.")
        
        print(f"\n[OK] Generating {num_samples} synthetic structures (in batches of {batch_size})...")
        
        # Generate in batches for better memory management
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            batch_num = min(batch_size, num_samples - i * batch_size)
            print(f"  Batch {i+1}/{num_batches}: Generating {batch_num} samples...")
            
            batch_df = self.synthesizer.sample(num_rows=batch_num)
            all_samples.append(batch_df)
        
        # Concatenate all batches
        synthetic_df = pd.concat(all_samples, ignore_index=True)
        
        print(f"[SUCCESS] Generated {len(synthetic_df)} synthetic structures")
        
        return synthetic_df
    
    def save(self, filepath: str):
        """Save trained model"""
        if self.synthesizer:
            self.synthesizer.save(filepath)
            print(f"[SUCCESS] Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model"""
        self.synthesizer = CTGANSynthesizer.load(filepath)
        print(f"[SUCCESS] Model loaded from {filepath}")


# ============================================================================
# SQL ASSEMBLY FROM SYNTHETIC STRUCTURES (WITH PARALLEL PROCESSING)
# ============================================================================

class SchemaAwareSQLAssembler:
    """
    Assemble valid CIM Wizard SQL from synthetic structures
    Enforces schema constraints and ensures syntactic validity
    """
    
    def __init__(self, schema_rules: CIMSchemaRules):
        self.rules = schema_rules
    
    def select_valid_tables(self, structure: Dict) -> List[str]:
        """Select valid tables based on structure requirements"""
        
        table_count = int(max(1, min(structure['table_count'], 8)))
        schema_complexity = structure['schema_complexity']
        
        # Filter tables by schema complexity
        if schema_complexity == 'SINGLE_TABLE':
            # Just one table
            return [random.choice(self.rules.VALID_TABLES)]
        
        elif schema_complexity == 'SINGLE_SCHEMA':
            # Multiple tables from same schema
            schema = random.choice(['cim_vector', 'cim_census'])
            schema_tables = [t for t in self.rules.VALID_TABLES if t.startswith(schema)]
            return random.sample(schema_tables, min(table_count, len(schema_tables)))
        
        else:  # MULTI_SCHEMA
            # Tables from multiple schemas
            return random.sample(self.rules.VALID_TABLES, min(table_count, len(self.rules.VALID_TABLES)))
    
    def find_valid_join_path(self, tables: List[str]) -> List[Tuple[str, str, List[str]]]:
        """Find valid JOIN path between tables"""
        
        if len(tables) < 2:
            return []
        
        joins = []
        
        # Try to connect tables using valid join pairs
        for i in range(len(tables) - 1):
            table1 = tables[i]
            table2 = tables[i + 1]
            
            # Check if direct join is valid
            if (table1, table2) in self.rules.VALID_JOINS:
                join_keys = self.rules.JOIN_KEYS.get((table1, table2), ['id'])
                joins.append((table1, table2, join_keys))
            elif (table2, table1) in self.rules.VALID_JOINS:
                join_keys = self.rules.JOIN_KEYS.get((table2, table1), ['id'])
                joins.append((table1, table2, join_keys))
            else:
                # Use generic spatial join
                joins.append((table1, table2, ['geometry']))
        
        return joins
    
    def select_spatial_functions(self, structure: Dict, tables: List[str]) -> List[str]:
        """Select appropriate spatial functions"""
        
        func_count = int(max(1, min(structure['spatial_function_count'], 10)))
        primary_category = structure['primary_function_category']
        
        # Get geometry types for selected tables
        geometry_types = set()
        for table in tables:
            if table in self.rules.TABLE_GEOMETRY_TYPES:
                geometry_types.add(self.rules.TABLE_GEOMETRY_TYPES[table])
        
        # Start with common functions
        available_functions = list(self.rules.COMMON_SPATIAL_FUNCTIONS)
        
        # Add functions for specific geometry types
        for geom_type in geometry_types:
            if geom_type in self.rules.FUNCTION_APPLICABILITY:
                available_functions.extend(self.rules.FUNCTION_APPLICABILITY[geom_type])
        
        # Remove duplicates
        available_functions = list(set(available_functions))
        
        # Select functions
        selected = random.sample(available_functions, min(func_count, len(available_functions)))
        
        return selected
    
    def assemble_sql(self, structure: Dict) -> str:
        """
        Assemble complete SQL query from structure
        """
        
        # Select components
        tables = self.select_valid_tables(structure)
        joins = self.find_valid_join_path(tables) if len(tables) > 1 else []
        functions = self.select_spatial_functions(structure, tables)
        
        # Get parameters
        params = generate_realistic_values()
        
        # Build SQL components
        sql_parts = []
        
        # CTEs if needed
        if structure['cte_count'] > 0:
            sql_parts.append("WITH cte AS (")
            sql_parts.append(f"  SELECT * FROM {tables[0]}")
            sql_parts.append(f"  WHERE project_id = '{params['project_id']}'")
            sql_parts.append(")")
            main_table = "cte"
        else:
            main_table = tables[0]
        
        # Main SELECT
        select_cols = []
        if tables:
            # Use first table for main columns
            table_alias = tables[0].split('.')[-1][0]
            select_cols.append(f"{table_alias}.id")
            
            # Add spatial function calls (with public schema prefix)
            for i, func in enumerate(functions[:3]):  # Limit to 3 for simplicity
                if func in ['ST_Area', 'ST_Length', 'ST_Distance']:
                    select_cols.append(f"public.{func}({table_alias}.geometry) AS {func.lower()}_{i}")
                elif func == 'ST_Centroid':
                    select_cols.append(f"public.{func}({table_alias}.geometry) AS centroid")
        
        sql_parts.append(f"SELECT {', '.join(select_cols) if select_cols else '*'}")
        sql_parts.append(f"FROM {main_table if structure['cte_count'] > 0 else tables[0]} {tables[0].split('.')[-1][0]}")
        
        # JOINs
        for i, (t1, t2, keys) in enumerate(joins):
            t2_alias = t2.split('.')[-1][0]
            t1_alias = t1.split('.')[-1][0]
            
            # Spatial join if geometry involved (with public schema prefix)
            if 'ST_Intersects' in functions or 'ST_Within' in functions:
                spatial_pred = random.choice(['ST_Intersects', 'ST_Within'])
                sql_parts.append(f"JOIN {t2} {t2_alias} ON public.{spatial_pred}({t1_alias}.geometry, {t2_alias}.geometry)")
            else:
                # Regular join
                join_cond = ' AND '.join([f"{t1_alias}.{k} = {t2_alias}.{k}" for k in keys])
                sql_parts.append(f"JOIN {t2} {t2_alias} ON {join_cond}")
        
        # WHERE clause
        where_clauses = []
        if 'project_id' in str(tables):
            where_clauses.append(f"project_id = '{params['project_id']}'")
        if 'scenario_id' in str(tables):
            where_clauses.append(f"scenario_id = '{params['scenario_id']}'")
        
        if where_clauses:
            sql_parts.append(f"WHERE {' AND '.join(where_clauses)}")
        
        # GROUP BY for aggregation
        if structure['sql_type'] == 'AGGREGATION':
            sql_parts.append(f"GROUP BY {tables[0].split('.')[-1][0]}.id")
        
        # LIMIT
        sql_parts.append(f"LIMIT {params['limit']}")
        
        return '\n'.join(sql_parts)


# ============================================================================
# QUALITY ASSESSMENT (SAME AS ECLAB VERSION)
# ============================================================================

class QualityAssessor:
    """Multi-dimensional quality assessment for synthetic SQL"""
    
    def __init__(self, schema_rules: CIMSchemaRules):
        self.rules = schema_rules
    
    def check_syntactic_validity(self, sql: str) -> float:
        """Check if SQL is syntactically valid (0.0-1.0)"""
        
        if not SQLPARSE_AVAILABLE:
            # Basic checks if sqlparse not available
            sql_upper = sql.upper()
            has_select = 'SELECT' in sql_upper
            has_from = 'FROM' in sql_upper
            return 1.0 if (has_select and has_from) else 0.0
        
        try:
            parsed = sqlparse.parse(sql)
            if not parsed or len(parsed) == 0:
                return 0.0
            
            # Check for basic SQL structure
            sql_upper = sql.upper()
            has_select = 'SELECT' in sql_upper
            has_from = 'FROM' in sql_upper
            balanced_parens = sql.count('(') == sql.count(')')
            
            score = 0.0
            if has_select: score += 0.4
            if has_from: score += 0.4
            if balanced_parens: score += 0.2
            
            return min(score, 1.0)
        
        except:
            return 0.0
    
    def check_schema_compliance(self, sql: str) -> float:
        """Check if SQL uses valid tables/columns (0.0-1.0)"""
        
        # Extract table references
        table_pattern = r'(?:FROM|JOIN)\s+(\w+\.\w+|\w+)'
        tables = re.findall(table_pattern, sql, re.IGNORECASE)
        
        if not tables:
            return 0.5  # No tables found - neutral score
        
        # Check how many tables are valid
        valid_count = sum(1 for t in tables if any(t in vt for vt in self.rules.VALID_TABLES))
        
        return valid_count / len(tables)
    
    def check_semantic_coherence(self, sql: str) -> float:
        """Check if query makes logical sense (0.0-1.0)"""
        
        sql_upper = sql.upper()
        score = 0.6  # Base score
        
        # Positive indicators
        if 'WHERE' in sql_upper:
            score += 0.1
        if any(f in sql for f in ['ST_Intersects', 'ST_Within', 'ST_Contains']):
            score += 0.1
        if 'GROUP BY' in sql_upper and ('COUNT' in sql_upper or 'SUM' in sql_upper):
            score += 0.1
        
        # Negative indicators
        if sql.count('SELECT') > 5:  # Too many SELECTs
            score -= 0.2
        if sql.count('JOIN') > 10:  # Too many JOINs
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def calculate_quality_score(self, sql: str, metadata: Dict) -> Tuple[float, Dict]:
        """
        Calculate comprehensive quality score
        
        Returns:
            (overall_score, score_breakdown)
        """
        
        scores = {
            'syntactic_validity': self.check_syntactic_validity(sql),
            'schema_compliance': self.check_schema_compliance(sql),
            'semantic_coherence': self.check_semantic_coherence(sql),
        }
        
        # Weights
        weights = {
            'syntactic_validity': 0.40,
            'schema_compliance': 0.40,
            'semantic_coherence': 0.20,
        }
        
        # Weighted average
        overall_score = sum(scores[k] * weights[k] for k in scores)
        
        return overall_score, scores


# Helper function for parallel processing
def process_structure_parallel(args):
    """Process a single structure (for parallel execution)"""
    i, row, schema_rules = args
    
    structure = row.to_dict()
    assembler = SchemaAwareSQLAssembler(schema_rules)
    quality_assessor = QualityAssessor(schema_rules)
    
    try:
        sql = assembler.assemble_sql(structure)
        quality_score, breakdown = quality_assessor.calculate_quality_score(sql, structure)
        
        sample = {
            "id": f"cim_stage2_ipazia_{i:06d}",
            "database_id": 1,
            "database_name": "cim_wizard",
            "question": f"Generated question for {structure['sql_type']} query",
            "question_tone": structure['question_tone'],
            "sql_postgis": sql,
            "sql_spatialite": sql,
            "sql_type": structure['sql_type'],
            "difficulty": {
                "query_complexity": structure['difficulty_level'],
                "spatial_complexity": "INTERMEDIATE",
                "schema_complexity": structure['schema_complexity'],
                "overall_difficulty": structure['difficulty_level'],
                "complexity_score": structure['complexity_score']
            },
            "usage_frequency": structure['usage_frequency'],
            "database_schema": {
                "schemas": [],
                "tables": [],
                "table_count": structure['table_count']
            },
            "spatial_functions": [],
            "instruction": f"Convert this natural language question to PostGIS spatial SQL...",
            "results": [],
            "has_results": False,
            "stage": "stage2_synthetic_ipazia",
            "generation_method": "ctgan_gpu",
            "quality_score": quality_score,
            "quality_breakdown": breakdown,
            "synthetic_structure": structure,
            "generated_at": datetime.now().isoformat()
        }
        
        return sample, quality_score
    
    except Exception as e:
        return None, 0.0


# ============================================================================
# STAGE 2 MAIN PIPELINE FOR IPAZIA
# ============================================================================

def run_stage2_pipeline_ipazia(
    stage1_file: str = "training_datasets/stage1_cim_dataset.jsonl",
    output_file: str = "training_datasets/stage2_synthetic_dataset_ipazia.jsonl",
    num_synthetic: int = 50000,
    quality_threshold: float = 0.70,
    use_gpu: bool = True,
    epochs: int = 300,
    use_parallel: bool = True,
    num_workers: int = 28
):
    """
    Execute Stage 2 pipeline optimized for ipazia machine
    
    Args:
        stage1_file: Path to Stage 1 output
        output_file: Path for Stage 2 output
        num_synthetic: Target number of synthetic samples
        quality_threshold: Minimum quality score (0.0-1.0)
        use_gpu: Use GPU for training (default True)
        epochs: Training epochs for CTGAN (300 recommended)
        use_parallel: Use parallel processing for SQL assembly
        num_workers: Number of parallel workers (default 28 = half of 56 cores)
    """
    
    print("="*80)
    print("STAGE 2: SDV SYNTHETIC SQL GENERATION - IPAZIA VERSION")
    print("="*80)
    print(f"Machine: ipazia (2x Xeon Gold 6238R, 256GB RAM, Quadro RTX 6000)")
    print(f"Configuration:")
    print(f"  - Model: CTGAN (GPU-accelerated)")
    print(f"  - Target samples: {num_synthetic:,}")
    print(f"  - Quality threshold: {quality_threshold}")
    print(f"  - GPU: {use_gpu}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Parallel workers: {num_workers}")
    print(f"  - Estimated time: 2-4 hours (training) + 30-60 min (assembly)")
    
    # Check dependencies
    if not SDV_AVAILABLE:
        print("\n[ERROR] Error: SDV not installed!")
        print("   Run: pip install sdv==1.9.0")
        return None
    
    # Load Stage 1 data
    print(f"\n[1/6] Loading Stage 1 data from {stage1_file}...")
    stage1_samples = []
    with open(stage1_file, 'r', encoding='utf-8') as f:
        for line in f:
            stage1_samples.append(json.loads(line))
    print(f"      [OK] Loaded {len(stage1_samples):,} Stage 1 samples")
    
    # Extract features
    print(f"\n[2/6] Extracting features for CTGAN training...")
    features_df = extract_features(stage1_samples)
    print(f"      [OK] Extracted features: {features_df.shape}")
    print(f"      Features: {list(features_df.columns)}")
    
    # Train CTGAN
    print(f"\n[3/6] Training CTGAN synthesizer...")
    trainer = CTGANTrainerIPAZIA(use_gpu=use_gpu)
    trainer.train(features_df, epochs=epochs)
    
    # Save model
    model_file = output_file.replace('.jsonl', '_model.pkl')
    trainer.save(model_file)
    
    # Generate synthetic structures (1.5x target for filtering)
    print(f"\n[4/6] Generating synthetic structures...")
    generation_target = int(num_synthetic * 1.5)
    synthetic_structures = trainer.generate(num_samples=generation_target, batch_size=10000)
    print(f"      [OK] Generated {len(synthetic_structures):,} synthetic structures")
    
    # Assemble SQL from structures
    print(f"\n[5/6] Assembling SQL queries with schema enforcement...")
    schema_rules = CIMSchemaRules()
    
    synthetic_samples = []
    quality_scores = []
    
    if use_parallel:
        print(f"      Using {num_workers} parallel workers...")
        
        # Prepare arguments for parallel processing
        args_list = [(i, row, schema_rules) for i, row in synthetic_structures.iterrows()]
        
        with Pool(num_workers) as pool:
            results = pool.map(process_structure_parallel, args_list)
        
        # Collect results
        for sample, quality_score in results:
            if sample is not None:
                synthetic_samples.append(sample)
                quality_scores.append(quality_score)
    else:
        # Sequential processing (fallback)
        assembler = SchemaAwareSQLAssembler(schema_rules)
        quality_assessor = QualityAssessor(schema_rules)
        
        for i, row in synthetic_structures.iterrows():
            structure = row.to_dict()
            
            try:
                sql = assembler.assemble_sql(structure)
                quality_score, breakdown = quality_assessor.calculate_quality_score(sql, structure)
                
                sample = {
                    "id": f"cim_stage2_ipazia_{i:06d}",
                    "database_id": 1,
                    "database_name": "cim_wizard",
                    "question": f"Generated question for {structure['sql_type']} query",
                    "question_tone": structure['question_tone'],
                    "sql_postgis": sql,
                    "sql_spatialite": sql,
                    "sql_type": structure['sql_type'],
                    "difficulty": {
                        "query_complexity": structure['difficulty_level'],
                        "spatial_complexity": "INTERMEDIATE",
                        "schema_complexity": structure['schema_complexity'],
                        "overall_difficulty": structure['difficulty_level'],
                        "complexity_score": structure['complexity_score']
                    },
                    "usage_frequency": structure['usage_frequency'],
                    "database_schema": {
                        "schemas": [],
                        "tables": [],
                        "table_count": structure['table_count']
                    },
                    "spatial_functions": [],
                    "instruction": f"Convert this natural language question to PostGIS spatial SQL...",
                    "results": [],
                    "has_results": False,
                    "stage": "stage2_synthetic_ipazia",
                    "generation_method": "ctgan_gpu",
                    "quality_score": quality_score,
                    "quality_breakdown": breakdown,
                    "synthetic_structure": structure,
                    "generated_at": datetime.now().isoformat()
                }
                
                synthetic_samples.append(sample)
                quality_scores.append(quality_score)
                
            except Exception as e:
                continue
            
            if (i + 1) % 5000 == 0:
                print(f"      Progress: {i + 1:,}/{len(synthetic_structures):,} structures processed...")
    
    print(f"      [OK] Assembled {len(synthetic_samples):,} SQL queries")
    
    # Quality filtering
    print(f"\n[6/6] Filtering by quality (threshold: {quality_threshold})...")
    high_quality = [s for s in synthetic_samples if s['quality_score'] >= quality_threshold]
    
    # Take target number
    final_samples = high_quality[:num_synthetic]
    
    print(f"      [OK] High quality samples: {len(high_quality):,}")
    print(f"      [OK] Final dataset: {len(final_samples):,} samples")
    print(f"      [OK] Average quality score: {np.mean(quality_scores):.3f}")
    
    # Save dataset
    print(f"\n[OK] Saving Stage 2 dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats = {
        "total_generated": len(synthetic_samples),
        "high_quality": len(high_quality),
        "final_dataset": len(final_samples),
        "average_quality_score": float(np.mean(quality_scores)),
        "quality_threshold": quality_threshold,
        "model_type": "CTGAN",
        "machine": "ipazia",
        "gpu_used": use_gpu,
        "epochs": epochs,
        "generation_date": datetime.now().isoformat()
    }
    
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[SUCCESS] Stage 2 Complete (ipazia)!")
    print(f"   Output: {output_file}")
    print(f"   Statistics: {stats_file}")
    print(f"   Model: {model_file}")
    
    return final_samples, stats


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    num_synthetic = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    use_gpu = True if len(sys.argv) < 3 else sys.argv[3].lower() != 'false'
    
    print(f"\nStage 2 Configuration (ipazia):")
    print(f"  Model: CTGAN (GPU-accelerated)")
    print(f"  Target samples: {num_synthetic:,}")
    print(f"  Epochs: {epochs}")
    print(f"  GPU: {use_gpu}")
    
    # Run pipeline
    samples, stats = run_stage2_pipeline_ipazia(
        stage1_file="training_datasets/stage1_cim_dataset.jsonl",
        output_file="training_datasets/stage2_synthetic_dataset_ipazia.jsonl",
        num_synthetic=num_synthetic,
        quality_threshold=0.70,
        use_gpu=use_gpu,
        epochs=epochs,
        use_parallel=True,
        num_workers=28
    )

