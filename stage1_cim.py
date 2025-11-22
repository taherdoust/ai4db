#!/usr/bin/env python3
"""
stage1_cim.py - CIM Wizard Focused Spatial SQL Generator
Consolidated rule-based generation with stratified sampling for CIM Wizard database

Integrates:
- CIM Wizard schema-specific templates
- Comprehensive spatial function classification
- Stratified evaluation sampling
- Enhanced metadata for Stage 2 & 3 pipeline
"""

import json
import random
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# ============================================================================
# CIM WIZARD DATABASE SCHEMA DEFINITION
# ============================================================================

CIM_SCHEMAS = {
    "cim_vector": {
        "cim_wizard_project_scenario": {
            "columns": ["scenario_id", "project_id", "project_name", "scenario_name", "project_boundary", "project_center", "census_boundary"],
            "geometry_columns": ["project_boundary", "project_center", "census_boundary"],
            "srid": 4326
        },
        "cim_wizard_building": {
            "columns": ["building_id", "lod", "building_geometry", "building_geometry_source", "census_id", "created_at", "updated_at", "building_surfaces_lod12"],
            "geometry_columns": ["building_geometry"],
            "srid": 4326
        },
        "cim_wizard_building_properties": {
            "columns": ["scenario_id", "building_id", "project_id", "lod", "height", "area", "volume", "number_of_floors", "const_period_census", "n_family", "n_people", "type", "const_tabula", "const_year"],
            "numeric_columns": ["height", "area", "volume", "number_of_floors", "const_year", "n_people", "n_family"],
            "categorical_columns": ["type", "const_period_census", "const_tabula"]
        }
    },
    "cim_census": {
        "censusgeo": {
            "columns": [
                "SEZ2011", "geometry", "Shape_Area",
                # Administrative hierarchy
                "CODREG", "REGIONE", "CODPRO", "PROVINCIA", "CODCOM", "COMUNE", "PROCOM", "NSEZ", "ACE", "CODLOC", "CODASC",
                # Population statistics (key indicators)
                "P1", "P2", "P3", "P14", "P15", "P16", "P27", "P28", "P29", "P47", "P60", "P61", "P62",
                # Housing statistics
                "ST1", "ST2", "ST3", "ST4", "ST5",
                # Building age distribution (construction periods - critical for energy modeling)
                "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16",
                # Building attributes (available columns)
                "A2", "A3", "A5", "A6", "A7", "A44", "A46", "A47", "A48",
                # Family statistics
                "PF1", "PF2", "PF3", "PF4", "PF5"
            ],
            "geometry_columns": ["geometry"],
            "numeric_columns": [
                "P1", "P2", "P3", "P14", "P15", "P16", "P27", "P28", "P29", "P47", "P60", "P61", "P62",
                "ST1", "ST2", "ST3", "ST4", "ST5",
                "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16",
                "A2", "A3", "A5", "A6", "A7", "A44", "A46", "A47", "A48",
                "PF1", "PF2", "PF3", "PF4", "PF5"
            ],
            "categorical_columns": ["REGIONE", "PROVINCIA", "COMUNE", "CODREG", "CODPRO", "CODCOM", "PROCOM"],
            "srid": 4326
        }
    },
    "cim_network": {
        "network_buses": {
            "columns": ["bus_id", "bus_name", "bus_type", "voltage_kv", "geometry", "zone", "in_service", "min_vm_pu", "max_vm_pu", "additional_data"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["voltage_kv", "min_vm_pu", "max_vm_pu"],
            "srid": 4326
        },
        "network_lines": {
            "columns": ["line_id", "line_name", "from_bus_id", "to_bus_id", "geometry", "length_km", "r_ohm_per_km", "x_ohm_per_km", "in_service"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["length_km", "r_ohm_per_km", "x_ohm_per_km"],
            "srid": 4326
        },
        "network_scenarios": {
            "columns": ["scenario_id", "scenario_name", "description"],
            "srid": 4326
        },
        "scenario_buses": {
            "columns": ["scenario_id", "bus_id"],
            "srid": 4326
        },
        "scenario_lines": {
            "columns": ["scenario_id", "line_id"],
            "srid": 4326
        }
    },
    "cim_raster": {
        
        "dtm": {
            "columns": ["rid", "rast"],
            "raster_columns": ["rast"],
            "numeric_columns": []
        },
        "dsm_sansalva": {
            "columns": ["rid", "rast"],
            "raster_columns": ["rast"],
            "numeric_columns": []
        }
    }
}

# ============================================================================
# SPATIAL FUNCTION CLASSIFICATION
# ============================================================================

SPATIAL_FUNCTIONS = {
    # Vector-only functions
    "vector_only": {
        # Predicates (spatial relationships)
        "ST_Intersects": {
            "category": "predicate",
            "usage_frequency": "most_frequent",  # CRITICAL
            "difficulty": "basic",
            "description": "Test if geometries intersect"
        },
        "ST_Contains": {
            "category": "predicate",
            "usage_frequency": "most_frequent",  # CRITICAL
            "difficulty": "basic",
            "description": "Test if geometry A contains geometry B"
        },
        "ST_Within": {
            "category": "predicate",
            "usage_frequency": "most_frequent",  # CRITICAL
            "difficulty": "basic",
            "description": "Test if geometry A is within geometry B"
        },
        "ST_Touches": {
            "category": "predicate",
            "usage_frequency": "most_frequent",  # VERY_HIGH
            "difficulty": "basic",
            "description": "Test if geometries touch at boundary"
        },
        "ST_DWithin": {
            "category": "predicate",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "basic",
            "description": "Test if geometries are within distance"
        },
        "ST_Overlaps": {
            "category": "predicate",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "intermediate",
            "description": "Test if geometries overlap"
        },
        "ST_Crosses": {
            "category": "predicate",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "intermediate",
            "description": "Test if geometries cross"
        },
        "ST_Disjoint": {
            "category": "predicate",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "intermediate",
            "description": "Test if geometries are disjoint"
        },
        
        # Measurements
        "ST_Area": {
            "category": "measurement",
            "usage_frequency": "most_frequent",  # CRITICAL
            "difficulty": "basic",
            "description": "Calculate area of polygon"
        },
        "ST_Distance": {
            "category": "measurement",
            "usage_frequency": "most_frequent",  # CRITICAL
            "difficulty": "basic",
            "description": "Calculate distance between geometries"
        },
        "ST_Length": {
            "category": "measurement",
            "usage_frequency": "most_frequent",  # VERY_HIGH
            "difficulty": "basic",
            "description": "Calculate length of linestring"
        },
        "ST_Perimeter": {
            "category": "measurement",
            "usage_frequency": "low_frequent",  # LOW
            "difficulty": "basic",
            "description": "Calculate perimeter of polygon"
        },
        
        # Processing/Editing
        "ST_Buffer": {
            "category": "processing",
            "usage_frequency": "most_frequent",  # VERY_HIGH
            "difficulty": "intermediate",
            "description": "Create buffer around geometry"
        },
        "ST_Union": {
            "category": "processing",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "intermediate",
            "description": "Merge geometries"
        },
        "ST_Intersection": {
            "category": "processing",
            "usage_frequency": "most_frequent",  # VERY_HIGH
            "difficulty": "intermediate",
            "description": "Calculate intersection of geometries"
        },
        "ST_Difference": {
            "category": "processing",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "intermediate",
            "description": "Calculate difference between geometries"
        },
        "ST_ConvexHull": {
            "category": "processing",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "intermediate",
            "description": "Create convex hull around geometry"
        },
        "ST_Simplify": {
            "category": "processing",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "intermediate",
            "description": "Simplify geometry"
        },
        
        # Accessors/Extractors
        "ST_X": {
            "category": "accessor",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "basic",
            "description": "Extract X coordinate from point"
        },
        "ST_Y": {
            "category": "accessor",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "basic",
            "description": "Extract Y coordinate from point"
        },
        "ST_Centroid": {
            "category": "accessor",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "basic",
            "description": "Calculate centroid of geometry"
        },
        "ST_Envelope": {
            "category": "accessor",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "basic",
            "description": "Get bounding box of geometry"
        },
        "ST_StartPoint": {
            "category": "accessor",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "basic",
            "description": "Get start point of linestring"
        },
        "ST_EndPoint": {
            "category": "accessor",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "basic",
            "description": "Get end point of linestring"
        },
        "ST_NumPoints": {
            "category": "accessor",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "basic",
            "description": "Count points in geometry"
        },
        
        # Constructors
        "ST_MakePoint": {
            "category": "constructor",
            "usage_frequency": "most_frequent",  # VERY_HIGH
            "difficulty": "basic",
            "description": "Create point from coordinates"
        },
        "ST_GeomFromText": {
            "category": "constructor",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "basic",
            "description": "Create geometry from WKT"
        },
        "ST_Collect": {
            "category": "constructor",
            "usage_frequency": "low_frequent",  # LOW
            "difficulty": "intermediate",
            "description": "Collect geometries into collection"
        },
        
        # Transforms
        "ST_Transform": {
            "category": "transform",
            "usage_frequency": "most_frequent",  # VERY_HIGH
            "difficulty": "intermediate",
            "description": "Transform geometry to different CRS"
        },
        "ST_SetSRID": {
            "category": "transform",
            "usage_frequency": "frequent",  # HIGH
            "difficulty": "basic",
            "description": "Set SRID of geometry"
        },
        
        # Validation
        "ST_IsValid": {
            "category": "validation",
            "usage_frequency": "most_frequent",  # VERY_HIGH
            "difficulty": "basic",
            "description": "Check if geometry is valid"
        },
        "ST_MakeValid": {
            "category": "validation",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "intermediate",
            "description": "Fix invalid geometry"
        },
        
        # Clustering (Advanced)
        "ST_ClusterDBSCAN": {
            "category": "clustering",
            "usage_frequency": "low_frequent",  # LOW
            "difficulty": "advanced",
            "description": "DBSCAN spatial clustering"
        },
        "ST_ClusterKMeans": {
            "category": "clustering",
            "usage_frequency": "low_frequent",  # LOW
            "difficulty": "advanced",
            "description": "K-means spatial clustering"
        }
    },
    
    # Raster-only functions
    "raster_only": {
        "ST_Value": {
            "category": "raster_accessor",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "intermediate",
            "description": "Extract raster value at point"
        },
        "ST_SummaryStats": {
            "category": "raster_analysis",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "advanced",
            "description": "Calculate raster statistics"
        }
    },
    
    # Vector-Raster functions (work with both)
    "vector_raster": {
        "ST_Clip": {
            "category": "raster_processing",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "advanced",
            "description": "Clip raster by vector mask"
        },
        "ST_Intersection_Raster": {
            "category": "raster_processing",
            "usage_frequency": "frequent",  # MEDIUM
            "difficulty": "advanced",
            "description": "Intersect raster with vector"
        }
    }
}

# ============================================================================
# SQL TAXONOMY DEFINITIONS
# ============================================================================

SQL_TYPES = {
    "SIMPLE_SELECT": "Single table selection with optional WHERE",
    "SPATIAL_JOIN": "Join with spatial predicate",
    "AGGREGATION": "GROUP BY with aggregate functions",
    "NESTED_QUERY": "Subquery or CTE",
    "SPATIAL_MEASUREMENT": "Measurement functions (ST_Area, ST_Distance, ST_Length)",
    "SPATIAL_PROCESSING": "Geometry processing (ST_Buffer, ST_Union, ST_Intersection)",
    "SPATIAL_CLUSTERING": "Spatial clustering (ST_ClusterDBSCAN, ST_ClusterKMeans)",
    "RASTER_VECTOR": "Raster-vector integration",
    "MULTI_JOIN": "Multiple table joins (2+ tables)",
    "WINDOW_FUNCTION": "Window functions (ROW_NUMBER, RANK, PARTITION BY)"
}

QUESTION_TONES = {
    "DIRECT": "Direct imperative (Show me, Find, Get, List)",
    "INTERROGATIVE": "Question form (What, Which, Where, How many)",
    "DESCRIPTIVE": "Descriptive request (I need, I want to know)",
    "ANALYTICAL": "Analytical request (Analyze, Calculate, Determine)",
    "COMPARATIVE": "Comparative request (Compare, Find difference)",
    "AGGREGATE": "Aggregation request (Count, Sum, Average)",
    "CONDITIONAL": "Conditional request (If X then Y, Where X matches Y)",
    "TEMPORAL": "Temporal request (Latest, Recent, Between dates)",
    "SPATIAL_SPECIFIC": "Spatial-specific language (within, near, intersecting)"
}

# ============================================================================
# PARAMETER POOLS FOR CIM WIZARD
# ============================================================================

CIM_PARAMETERS = {
    # Real project-scenario pairs from database
    "project_scenario_pairs": [
        {
            "project_id": "4be7d1ff-e8bf-4374-a13e-67e7b0d52eb1",
            "scenario_id": "4be7d1ff-e8bf-4374-a13e-67e7b0d52eb1",
            "project_name": "Sansalva_filter",
            "scenario_name": "baseline"
        },
        {
            "project_id": "5a00fa63-2ef4-4d38-baa0-48ae5d80d21a",
            "scenario_id": "5a00fa63-2ef4-4d38-baa0-48ae5d80d21a",
            "project_name": "Sansalva_filter",
            "scenario_name": "sansalva3.5"
        },
        {
            "project_id": "e983d9e4-70ce-43e1-b778-03c25d992433",
            "scenario_id": "e983d9e4-70ce-43e1-b778-03c25d992433",
            "project_name": "Sansalva_filter_3",
            "scenario_name": "sansalva3"
        },
        {
            "project_id": "aeba11e7-ab0e-46ca-b89f-4c856c0289bf",
            "scenario_id": "aeba11e7-ab0e-46ca-b89f-4c856c0289bf",
            "project_name": "Sansalva_filter_4",
            "scenario_name": "sansalva4"
        }
    ],
    
    # Extracted unique values for backward compatibility
    "project_ids": [
        "4be7d1ff-e8bf-4374-a13e-67e7b0d52eb1",
        "5a00fa63-2ef4-4d38-baa0-48ae5d80d21a",
        "e983d9e4-70ce-43e1-b778-03c25d992433",
        "aeba11e7-ab0e-46ca-b89f-4c856c0289bf"
    ],
    "scenario_ids": [
        "4be7d1ff-e8bf-4374-a13e-67e7b0d52eb1",
        "5a00fa63-2ef4-4d38-baa0-48ae5d80d21a",
        "e983d9e4-70ce-43e1-b778-03c25d992433",
        "aeba11e7-ab0e-46ca-b89f-4c856c0289bf"
    ],
    "project_names": ["Sansalva_filter", "Sansalva_filter_3", "Sansalva_filter_4"],
    "scenario_names": ["baseline", "sansalva3.5", "sansalva3", "sansalva4"],
    
    # Real building types from database
    "building_types": ["residential", "non-residential"],
    
    # Real voltage levels from network_buses (actual values from database)
    "voltage_levels": [0.4, 20.0, 132.0, 400.0],
    
    # Real network bus IDs (sample from actual database)
    "bus_ids": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
    
    # Real census zone IDs (actual SEZ2011 values with population > 0)
    "census_zones": [
        "10180000029", "10180000018", "10240000037", "11710000046", "11830000022",
        "10900000379", "12720001707", "12720000215", "12720001637", "12720001763",
        "12720001821", "12720001884", "12720001946", "12720002002", "12720002064"
    ],
    
    # Standard SRIDs
    "srids": [4326, 3857, 32632, 32633],
    
    # Real building IDs from database (50 actual UUIDs)
    "building_ids": [
        "0033de25-d7d1-48d5-98c3-bc02973a13c0",
        "0040217f-9865-4cab-9478-731e0c443a85",
        "0047ba60-0b2f-4b72-878e-02fe589ed37f",
        "0054c487-661b-493d-a819-e143b6c66e52",
        "00633a25-715f-4e23-9c24-c8c5e4e2678c",
        "0069f14b-4147-4254-a727-04198d14fab6",
        "007a2247-cdcd-49d0-8d30-6d249f9a3124",
        "0086830a-2510-4f23-926d-5d215729cc56",
        "00887359-a93c-4b0c-8cb8-1535ae3ad1f7",
        "008a63aa-6819-4f6b-b594-1206eac879d8",
        "0098d25a-d0b1-4081-ba67-ae9112bbbf95",
        "00a42bee-c946-48fc-9603-fdf82979a7af",
        "00b01e23-3003-4864-a2f4-28e2efaf6d4b",
        "00b21ae4-2343-42ee-8c53-e12e4a96e409",
        "00b272c0-0e93-4d53-a969-d768c3280dce",
        "00c2d7d0-83a6-4f68-bb47-a3ece88124be",
        "00e03e68-75cd-4e76-990e-a5295c721e2e",
        "00f14319-532d-4ee5-b45a-32cf19d26a73",
        "00fe1681-93ca-4309-a17b-287b3f2ca616",
        "010454b9-5b54-4208-90ce-41c2b181c852",
        "012d11ed-97f7-46a8-ad80-ed38320b3fa8",
        "012e9e8b-90af-45c6-a617-63df7bfaa464",
        "013d45a8-3563-42a0-a01c-335ea67d1070",
        "014844ab-0449-47ef-bd20-797f10d7f4d6",
        "0169cc81-9df2-4f16-9390-949e940c5b4c",
        "01764da2-a801-4dec-83bf-3a535fd083ab",
        "01808035-47bd-4a6f-81ba-07ef5e44f93d",
        "019d513c-9bea-4f34-89ac-91edd496f40c",
        "019ee682-7c8e-4e1d-9bcd-877b6f2d1bb7",
        "01a0c47e-abb0-4327-9f30-f5f845b5defa",
        "01a94fd3-07e3-4ada-8014-9d7bd68a4a97",
        "01ad56a4-5437-41a8-b09d-eff4007d794d",
        "01b71fdb-8ac8-4cdf-be29-943872fda5b3",
        "0216d44d-d352-4f12-bd16-79ba0f2dc167",
        "021d6573-e91d-409f-89f6-72a608238220",
        "022f4517-5ac0-4ec3-9169-6f34b445869a",
        "024447c1-1d36-4322-8be3-bc338c082a07",
        "02491899-ba02-4ed3-afca-43985ff180d8",
        "026e8641-2eec-4b1f-bcc8-145c6ce5cc38",
        "029b85a4-6f63-4970-b0e3-2c2875388334",
        "02b81f56-6754-4a53-9fea-e93380f1242f",
        "02cb3c45-a390-4cdb-ad56-b4d4bc31568c",
        "02f3c101-222f-4f48-8a0a-6aee8b7c50e8",
        "03121e2b-62b8-466f-9595-a5145e1ca2df",
        "03158862-3db7-4c34-a566-92e512c4b29b",
        "032a80f4-2c0b-4fc5-adb5-ef81072c4fcb",
        "0331072a-ce76-4992-a80a-fbfe2bd45ce6",
        "0346fe08-1688-474e-81bf-ad26c1b79c42",
        "036a709d-07e7-412e-9c7a-5460f6820ed9",
        "036d2174-245d-4021-90e7-58d95ee6c577"
    ],
    
    # Overlap thresholds for spatial analysis
    "overlap_thresholds": [10, 20, 30, 40, 50, 60, 70, 80]
}

# ============================================================================
# LIMIT/ORDER BY DISTRIBUTION STRATEGY (PHASE 4)
# ============================================================================

LIMIT_ORDER_DISTRIBUTION = {
    # 85% - No LIMIT, No ORDER BY (typical user behavior)
    "FULL_RESULTS": {
        "ratio": 0.85,
        "description": "User wants all results for analysis"
    },
    
    # 7% - ORDER BY only (user wants sorted results)
    "ORDERED_ONLY": {
        "ratio": 0.07,
        "description": "User wants sorted data without limiting"
    },
    
    # 8% - LIMIT + ORDER BY (user wants top N)
    "TOP_N": {
        "ratio": 0.08,
        "description": "User explicitly wants top N results"
    }
}

def determine_limit_strategy(template_id: str, variation_idx: int) -> str:
    """Determine LIMIT/ORDER BY strategy for this variation (85/7/8 distribution)"""
    
    # Deterministic selection based on template + variation
    selector = hash(f"{template_id}_{variation_idx}") % 100
    
    if selector < 85:
        return "FULL_RESULTS"
    elif selector < 92:
        return "ORDERED_ONLY"
    else:
        return "TOP_N"

def remove_limit_and_order(sql: str) -> str:
    """Remove LIMIT and ORDER BY clauses from SQL"""
    # Remove ORDER BY clause
    sql = re.sub(r'\s+ORDER BY[^;]*?(?=LIMIT|$)', '', sql, flags=re.IGNORECASE)
    # Remove LIMIT clause
    sql = re.sub(r'\s+LIMIT\s+\d+', '', sql, flags=re.IGNORECASE)
    return sql.strip().rstrip(';') + ';'

def remove_limit_only(sql: str) -> str:
    """Remove LIMIT but keep ORDER BY"""
    sql = re.sub(r'\s+LIMIT\s+\d+', '', sql, flags=re.IGNORECASE)
    return sql.strip().rstrip(';') + ';'

def generate_realistic_values() -> Dict[str, any]:
    """Generate realistic parameter values for CIM database queries using actual database data"""
    
    # Select a random project-scenario pair (ensures valid combination)
    proj_scen_pair = random.choice(CIM_PARAMETERS["project_scenario_pairs"])
    
    return {
        # Use matched project-scenario pairs from real database
        "project_id": proj_scen_pair["project_id"],
        "scenario_id": proj_scen_pair["scenario_id"],
        "project_name": proj_scen_pair["project_name"],
        "scenario_name": proj_scen_pair["scenario_name"],
        
        # Real building data
        "building_type": random.choice(CIM_PARAMETERS["building_types"]),
        "building_id": random.choice(CIM_PARAMETERS["building_ids"]),
        "census_id": random.choice(CIM_PARAMETERS["building_ids"]),  # Use building_id for census_id placeholder
        
        # Network and infrastructure
        "voltage_kv": random.choice(CIM_PARAMETERS["voltage_levels"]),
        "bus_id": random.choice(CIM_PARAMETERS["bus_ids"]),
        "census_zone": random.choice(CIM_PARAMETERS["census_zones"]),
        
        # Geographic and administrative
        "srid": random.choice(CIM_PARAMETERS["srids"]),
        
        # Spatial analysis parameters
        "overlap_threshold": random.choice(CIM_PARAMETERS["overlap_thresholds"]),
        "buffer_distance": random.choice([100, 500, 1000, 2000]),
        "max_distance": random.choice([500, 1000, 2000, 5000]),
        "cluster_distance": random.choice([1000, 2000, 5000]),
        
        # Numeric thresholds
        "min_area": random.randint(50, 500),
        "max_area": random.randint(1000, 5000),
        "min_height": random.randint(3, 10),
        "max_height": random.randint(15, 100),
        "min_people": random.randint(1, 5),
        "max_people": random.randint(6, 20),
        "year": random.randint(1950, 2024),
        "min_population": random.choice([100, 500, 1000]),
        "min_buildings": random.choice([5, 10, 20]),
        "min_areas": random.choice([3, 5, 10]),
        
        # Clustering parameters
        "cluster_count": random.choice([3, 5, 8, 10]),
        "min_cluster_size": random.choice([3, 5, 10]),
        "min_points": random.choice([3, 5, 8]),
        
        # Coordinates (approximate bounds for Italy)
        "lon": round(random.uniform(7.0, 18.0), 6),
        "lat": round(random.uniform(36.0, 47.0), 6),
        
        # Query limits (only used for TOP_N strategy - 8% of samples)
        "limit": random.choice([10, 25, 50, 100])
    }

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SqlPair:
    """Represents a SQL query pair with enhanced metadata"""
    template_id: str
    complexity: str  # A, B, C
    postgis_sql: str
    spatialite_sql: str
    natural_language_desc: str
    tags: Set[str]
    evidence: Dict[str, any]

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def extract_spatial_functions(sql: str) -> List[str]:
    """Extract all spatial functions from SQL query"""
    functions = re.findall(r'ST_\w+', sql, re.IGNORECASE)
    return list(set([f.upper() for f in functions]))

def classify_function_data_type(func: str) -> str:
    """Classify function by data type: vector_only, raster_only, vector_raster"""
    for data_type, funcs in SPATIAL_FUNCTIONS.items():
        if func in funcs:
            return data_type
    return "vector_only"  # default

def classify_function_usage(func: str) -> str:
    """Classify function usage frequency"""
    for data_type, funcs in SPATIAL_FUNCTIONS.items():
        if func in funcs:
            return funcs[func].get("usage_frequency", "frequent")
    return "frequent"

def classify_function_difficulty(func: str) -> str:
    """Classify function difficulty"""
    for data_type, funcs in SPATIAL_FUNCTIONS.items():
        if func in funcs:
            return funcs[func].get("difficulty", "intermediate")
    return "intermediate"

def classify_sql_type(sql: str, metadata: Dict) -> str:
    """Classify SQL query type"""
    sql_upper = sql.upper()
    
    # Raster operations
    if 'ST_VALUE' in sql_upper or 'ST_SUMMARYSTATS' in sql_upper:
        return "RASTER_VECTOR"
    
    # Spatial clustering
    if 'ST_CLUSTERDBSCAN' in sql_upper or 'ST_CLUSTERKMEANS' in sql_upper:
        return "SPATIAL_CLUSTERING"
    
    # Window functions
    if 'ROW_NUMBER' in sql_upper or 'RANK(' in sql_upper or 'PARTITION BY' in sql_upper:
        return "WINDOW_FUNCTION"
    
    # Nested queries (CTEs)
    if 'WITH' in sql_upper and 'AS (' in sql_upper:
        return "NESTED_QUERY"
    
    # Multiple joins
    join_count = sql_upper.count('JOIN')
    if join_count >= 2:
        return "MULTI_JOIN"
    
    # Spatial joins
    spatial_predicates = ['ST_INTERSECTS', 'ST_WITHIN', 'ST_CONTAINS', 'ST_TOUCHES', 'ST_OVERLAPS', 'ST_DWITHIN']
    if join_count >= 1 and any(pred in sql_upper for pred in spatial_predicates):
        return "SPATIAL_JOIN"
    
    # Aggregation queries
    if 'GROUP BY' in sql_upper:
        return "AGGREGATION"
    
    # Spatial processing
    spatial_processing = ['ST_BUFFER', 'ST_UNION', 'ST_INTERSECTION', 'ST_DIFFERENCE']
    if any(func in sql_upper for func in spatial_processing):
        return "SPATIAL_PROCESSING"
    
    # Spatial measurement
    spatial_measurement = ['ST_AREA', 'ST_LENGTH', 'ST_DISTANCE', 'ST_PERIMETER']
    if any(func in sql_upper for func in spatial_measurement):
        return "SPATIAL_MEASUREMENT"
    
    return "SIMPLE_SELECT"

def classify_question_tone(natural_language: str) -> str:
    """Classify question tone"""
    nl_lower = natural_language.lower()
    
    # Spatial-specific
    spatial_keywords = ['within', 'near', 'intersecting', 'overlapping', 'adjacent', 'touching']
    if any(kw in nl_lower for kw in spatial_keywords):
        return "SPATIAL_SPECIFIC"
    
    # Temporal
    temporal_keywords = ['latest', 'recent', 'historical', 'between', 'before', 'after']
    if any(kw in nl_lower for kw in temporal_keywords):
        return "TEMPORAL"
    
    # Comparative
    comparative_keywords = ['compare', 'difference between', 'versus', 'more than', 'less than']
    if any(kw in nl_lower for kw in comparative_keywords):
        return "COMPARATIVE"
    
    # Analytical
    analytical_keywords = ['analyze', 'examine', 'evaluate', 'calculate', 'compute', 'determine']
    if any(kw in nl_lower for kw in analytical_keywords):
        return "ANALYTICAL"
    
    # Aggregation
    aggregate_keywords = ['count', 'sum', 'average', 'total', 'how many', 'how much']
    if any(kw in nl_lower for kw in aggregate_keywords):
        return "AGGREGATE"
    
    # Conditional
    conditional_keywords = ['if', 'when', 'where', 'that match']
    if any(kw in nl_lower for kw in conditional_keywords):
        return "CONDITIONAL"
    
    # Interrogative
    interrogative_starters = ['what', 'which', 'where', 'who', 'how']
    if any(nl_lower.startswith(starter) for starter in interrogative_starters):
        return "INTERROGATIVE"
    
    # Direct
    direct_keywords = ['show', 'find', 'get', 'list', 'display', 'return']
    if any(nl_lower.startswith(kw) for kw in direct_keywords):
        return "DIRECT"
    
    # Descriptive
    descriptive_phrases = ['i need', 'i want', 'i would like']
    if any(phrase in nl_lower for phrase in descriptive_phrases):
        return "DESCRIPTIVE"
    
    return "INTERROGATIVE"

def calculate_difficulty_dimensions(sql: str, metadata: Dict) -> Dict[str, str]:
    """Calculate difficulty across all dimensions"""
    sql_upper = sql.upper()
    
    # Extract structural components
    cte_count = sql_upper.count('WITH')
    join_count = sql_upper.count('JOIN')
    subquery_count = sql.count('(SELECT')
    spatial_func_count = len(metadata.get('spatial_functions', []))
    table_count = len(metadata.get('tables', []))
    
    # Query complexity (no EXPERT level)
    complexity_score = 0
    if cte_count >= 2:
        complexity_score += 2
    elif cte_count == 1:
        complexity_score += 1
    
    if join_count >= 2:
        complexity_score += 2
    elif join_count == 1:
        complexity_score += 1
    
    if subquery_count >= 2:
        complexity_score += 2
    elif subquery_count == 1:
        complexity_score += 1
    
    if 'PARTITION BY' in sql_upper or 'ROW_NUMBER' in sql_upper:
        complexity_score += 2
    
    # Map to EASY, MEDIUM, HARD (no EXPERT)
    if complexity_score >= 5:
        query_complexity = "HARD"
    elif complexity_score >= 3:
        query_complexity = "MEDIUM"
    else:
        query_complexity = "EASY"
    
    # Spatial complexity
    advanced_spatial = ['ST_CLUSTER', 'ST_SUMMARYSTATS', 'ST_VALUE']
    intermediate_spatial = ['ST_BUFFER', 'ST_TRANSFORM', 'ST_UNION', 'ST_INTERSECTION']
    
    if any(func in sql_upper for func in advanced_spatial):
        spatial_complexity = "ADVANCED"
    elif any(func in sql_upper for func in intermediate_spatial) or spatial_func_count >= 2:
        spatial_complexity = "INTERMEDIATE"
    else:
        spatial_complexity = "BASIC"
    
    # Schema complexity
    schema_count = len(set(table.split('.')[0] for table in metadata.get('tables', []) if '.' in table))
    if schema_count >= 2:
        schema_complexity = "MULTI_SCHEMA"
    elif table_count >= 2:
        schema_complexity = "SINGLE_SCHEMA"
    else:
        schema_complexity = "SINGLE_TABLE"
    
    # Function count: 1, 2, 3+
    if spatial_func_count >= 3:
        function_count = "3+"
    elif spatial_func_count == 2:
        function_count = "2"
    else:
        function_count = "1"
    
    # Join count: 0, 1, 2+
    if join_count >= 2:
        join_count_cat = "2+"
    elif join_count == 1:
        join_count_cat = "1"
    else:
        join_count_cat = "0"
    
    # Derive overall complexity level (A, B, C)
    # A: Easy queries, basic spatial, single table/schema
    # B: Medium queries, intermediate spatial, joins
    # C: Hard queries, advanced spatial, multi-schema
    if query_complexity == "HARD" or spatial_complexity == "ADVANCED" or schema_complexity == "MULTI_SCHEMA":
        overall_complexity = "C"
    elif query_complexity == "MEDIUM" or spatial_complexity == "INTERMEDIATE" or join_count >= 1:
        overall_complexity = "B"
    else:
        overall_complexity = "A"
    
    return {
        "query_complexity": query_complexity,
        "spatial_complexity": spatial_complexity,
        "schema_complexity": schema_complexity,
        "function_count": function_count,
        "join_count": join_count_cat,
        "overall_difficulty": query_complexity,
        "complexity_level": overall_complexity,
        "complexity_score": complexity_score
    }

def extract_evidence(sql: str, template_id: str, tags: Set[str]) -> Dict[str, any]:
    """Extract evidence of schemas, tables, columns, functions"""
    evidence = {
        "database": "cim_wizard",
        "schemas": set(),
        "tables": set(), 
        "columns": set(),
        "functions": set(),
        "template_source": "cim_wizard"
    }
    
    # Extract schemas (schema.table patterns)
    schema_matches = re.findall(r'(\w+)\.(\w+)', sql)
    for schema, table in schema_matches:
        evidence["schemas"].add(schema)
        evidence["tables"].add(f"{schema}.{table}")
    
    # Extract spatial functions
    func_matches = re.findall(r'ST_\w+', sql)
    evidence["functions"].update(func_matches)
    
    # Convert sets to lists for JSON serialization
    return {k: list(v) if isinstance(v, set) else v for k, v in evidence.items()}

# ============================================================================
# CIM WIZARD TEMPLATES
# ============================================================================

def infer_template_priority(template_id: str, tags: Set[str]) -> int:
    """
    Infer priority from template_id and tags.
    Priority 1: Inner cim_vector (A1, A2, A5-A7, A10-A20, B1, B5, B7, C1, C2)
    Priority 2: Cross-schema with cim_vector (A8-A9, B2-B3, B8-B9, C3-C4, C6, C8-C9)
    Priority 3: Inner census/network/raster (A3-A4, B4, B6, B9, C5, C7)
    """
    template_num = template_id.split('_')[1] if '_' in template_id else ""
    
    # Priority 1: A1, A2, A5-A7, A10-A20, B1, B5, B7, C1, C2
    priority_1_templates = ['A1', 'A2', 'A5', 'A6', 'A7', 'A10', 'A11', 'A12', 'A13', 'A14', 
                           'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'B1', 'B5', 'B7', 'C1', 'C2']
    
    # Priority 3: A3, A4, B4, B6, C5, C7
    priority_3_templates = ['A3', 'A4', 'B4', 'B6', 'C5', 'C7']
    
    if template_num in priority_1_templates:
        return 1
    elif template_num in priority_3_templates:
        return 3
    else:
        return 2  # Everything else is Priority 2

def generate_cim_templates() -> List[Tuple[str, str, str, Set[str]]]:
    """
    Generate CIM Wizard templates with priority-based organization
    
    Template Priority System:
    - Priority 1: Inner cim_vector schema (buildings, properties, projects)
    - Priority 2: Cross-schema (cim_vector + cim_census/cim_network/cim_raster)
    - Priority 3: Inner cim_census/cim_raster/cim_network
    
    Returns: List of (template_id, sql, natural_language, tags)
    """
    templates = []
    
    # ==========================================================================
    # PRIORITY 1: INNER CIM_VECTOR SCHEMA (HIGHEST PRIORITY)
    # Focus on core building, properties, and project queries
    # ==========================================================================
    # ========== COMPLEXITY A: BASIC OPERATIONS ==========
    
    # A1: Simple building selection by type and area
    templates.append((
        "CIM_A1_building_by_type",
        """
SELECT b.building_id, b.lod, public.ST_Area(b.building_geometry) as area_sqm
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND bp.type = '{building_type}'
  AND public.ST_Area(b.building_geometry) > {min_area}
LIMIT {limit};
        """.strip(),
        "Find {building_type} buildings with area greater than {min_area} square meters in project {project_id} scenario {scenario_id}, limit to {limit} results",
        {"building", "area_filter", "type_filter", "basic"}
    ))
    
    # A2: Project at location
    templates.append((
        "CIM_A2_project_at_location",
        """
SELECT ps.project_name, ps.scenario_name, public.ST_Area(ps.project_boundary) as project_area_sqm
FROM cim_vector.cim_wizard_project_scenario ps
WHERE public.ST_Intersects(ps.project_boundary, public.ST_SetSRID(public.ST_MakePoint({lon}, {lat}), {srid}))
LIMIT {limit};
        """.strip(),
        "Find project scenarios that contain the geographic point at longitude {lon} latitude {lat} with SRID {srid}, limit to {limit} results",
        {"project", "point_in_polygon", "basic"}
    ))
    
    # ==========================================================================
    # PRIORITY 3: INNER CIM_CENSUS/CIM_NETWORK/CIM_RASTER
    # Queries within non-vector schemas
    # ==========================================================================
    
    # A3: Grid buses by voltage
    templates.append((
        "CIM_A3_grid_buses_by_voltage",
        """
SELECT gb.bus_id, gb.bus_name, gb.voltage_kv, public.ST_X(gb.geometry) as lon, public.ST_Y(gb.geometry) as lat
FROM cim_network.network_buses gb
WHERE gb.voltage_kv >= {voltage_kv}
  AND gb.in_service = true
LIMIT {limit};
        """.strip(),
        "Find active grid buses with voltage at or above {voltage_kv} kV, limit to {limit} results",
        {"grid", "voltage_filter", "basic"}
    ))
    
    # A4: Census population by zone
    templates.append((
        "CIM_A4_census_population",
        """
SELECT c.sez2011, c.p1 as total_population,
       c.p2 as male_population,
       c.p3 as female_population
FROM cim_census.censusgeo c
WHERE c.p1 >= {min_population}
ORDER BY c.p1 DESC
LIMIT {limit};
        """.strip(),
        "Analyze population distribution by gender in census areas with minimum population of {min_population}, ordered by total population descending, limit to {limit} results",
        {"census", "demographics", "basic"}
    ))
    
    # ==========================================================================
    # BACK TO PRIORITY 1: INNER CIM_VECTOR SCHEMA
    # ==========================================================================
    
    # A5: Building height from properties
    templates.append((
        "CIM_A5_building_height",
        """
SELECT bp.building_id, bp.height, bp.type, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND bp.height >= {min_height}
ORDER BY bp.height DESC
LIMIT {limit};
        """.strip(),
        "Retrieve building heights of at least {min_height} meters from properties for project {project_id} scenario {scenario_id}, ordered by height descending, limit to {limit} results",
        {"building", "height", "properties", "basic"}
    ))
    
    # A6: Building distance calculation
    templates.append((
        "CIM_A6_building_distance",
        """
SELECT b.building_id, 
       public.ST_Distance(b.building_geometry, public.ST_SetSRID(public.ST_MakePoint({lon}, {lat}), 4326)) as distance_m
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
ORDER BY distance_m ASC
LIMIT {limit};
        """.strip(),
        "Calculate distance from buildings in project {project_id} scenario {scenario_id} to the point at longitude {lon} latitude {lat} in SRID 4326, ordered by distance ascending, limit to {limit} results",
        {"building", "distance", "measurement", "basic"}
    ))
    
    # A7: Buildings inside project boundary
    templates.append((
        "CIM_A7_buildings_in_project",
        """
SELECT b.building_id, b.lod, bp.type, bp.height, public.ST_Area(b.building_geometry) as area_sqm
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_vector.cim_wizard_project_scenario ps ON bp.project_id = ps.project_id AND bp.scenario_id = ps.scenario_id
WHERE ps.project_id = '{project_id}'
  AND ps.scenario_id = '{scenario_id}'
  AND public.ST_Intersects(b.building_geometry, ps.project_boundary)
LIMIT {limit};
        """.strip(),
        "Find buildings that intersect with the boundary of project {project_id} scenario {scenario_id}, limit to {limit} results",
        {"building", "project", "spatial_predicate", "basic"}
    ))
    
    # NEW PRIORITY 1 TEMPLATES - Simple cim_vector queries
    
    # A10: Count buildings by type
    templates.append((
        "CIM_A10_count_buildings_by_type",
        """
SELECT bp.type, COUNT(*) as building_count
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.type
ORDER BY building_count DESC;
        """.strip(),
        "Count buildings by type in project {project_id} scenario {scenario_id}, ordered by count descending",
        {"building", "aggregation", "type", "basic"}
    ))
    
    # A11: Buildings by area range
    templates.append((
        "CIM_A11_buildings_by_area_range",
        """
SELECT bp.building_id, bp.type, bp.area, bp.height
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.area BETWEEN {min_area} AND {max_area}
ORDER BY bp.area DESC
LIMIT {limit};
        """.strip(),
        "Find buildings with area between {min_area} and {max_area} square meters in project {project_id} scenario {scenario_id}, ordered by area descending, limit to {limit} results",
        {"building", "area_filter", "range", "basic"}
    ))
    
    # A12: Average building metrics
    templates.append((
        "CIM_A12_average_building_metrics",
        """
SELECT AVG(bp.height) as avg_height,
       AVG(bp.area) as avg_area,
       AVG(bp.volume) as avg_volume,
       AVG(bp.number_of_floors) as avg_floors,
       COUNT(*) as total_buildings
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}';
        """.strip(),
        "Calculate average building metrics (height, area, volume, floors) for project {project_id} scenario {scenario_id}",
        {"building", "aggregation", "statistics", "basic"}
    ))
    
    # A13: Buildings by construction year
    templates.append((
        "CIM_A13_buildings_by_year",
        """
SELECT bp.building_id, bp.type, bp.const_year, bp.height, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.const_year >= {year}
ORDER BY bp.const_year DESC
LIMIT {limit};
        """.strip(),
        "Find buildings constructed since year {year} in project {project_id} scenario {scenario_id}, ordered by construction year descending, limit to {limit} results",
        {"building", "year_filter", "temporal", "basic"}
    ))
    
    # A14: Tall buildings
    templates.append((
        "CIM_A14_tall_buildings",
        """
SELECT bp.building_id, bp.type, bp.height, bp.number_of_floors, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.height >= {min_height}
ORDER BY bp.height DESC
LIMIT {limit};
        """.strip(),
        "Find tall buildings (height >= {min_height} meters) in project {project_id} scenario {scenario_id}, ordered by height descending, limit to {limit} results",
        {"building", "height_filter", "basic"}
    ))
    
    # A15: Buildings by number of floors
    templates.append((
        "CIM_A15_buildings_by_floors",
        """
SELECT bp.building_id, bp.type, bp.number_of_floors, bp.height, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.number_of_floors >= {min_people}
ORDER BY bp.number_of_floors DESC
LIMIT {limit};
        """.strip(),
        "Find buildings with at least {min_people} floors in project {project_id} scenario {scenario_id}, ordered by floor count descending, limit to {limit} results",
        {"building", "floors_filter", "basic"}
    ))
    
    # A16: Total building area by type
    templates.append((
        "CIM_A16_total_area_by_type",
        """
SELECT bp.type, 
       COUNT(*) as building_count,
       SUM(bp.area) as total_area,
       AVG(bp.area) as avg_area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.type
ORDER BY total_area DESC;
        """.strip(),
        "Calculate total and average building area by type for project {project_id} scenario {scenario_id}, ordered by total area descending",
        {"building", "aggregation", "area", "basic"}
    ))
    
    # A17: Buildings with population data
    templates.append((
        "CIM_A17_buildings_with_population",
        """
SELECT bp.building_id, bp.type, bp.n_people, bp.n_family, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.n_people > 0
ORDER BY bp.n_people DESC
LIMIT {limit};
        """.strip(),
        "Find buildings with population data in project {project_id} scenario {scenario_id}, ordered by number of people descending, limit to {limit} results",
        {"building", "population", "filter", "basic"}
    ))
    
    # A18: Project scenario summary
    templates.append((
        "CIM_A18_project_summary",
        """
SELECT ps.project_id, ps.project_name, ps.scenario_name,
       public.ST_Area(ps.project_boundary) as boundary_area_sqm,
       (SELECT COUNT(*) FROM cim_vector.cim_wizard_building_properties bp 
        WHERE bp.project_id = ps.project_id AND bp.scenario_id = ps.scenario_id) as building_count
FROM cim_vector.cim_wizard_project_scenario ps
WHERE ps.project_id = '{project_id}'
  AND ps.scenario_id = '{scenario_id}';
        """.strip(),
        "Get project scenario summary with boundary area and building count for project {project_id} scenario {scenario_id}",
        {"project", "summary", "aggregation", "basic"}
    ))
    
    # A19: Building density by area
    templates.append((
        "CIM_A19_building_density",
        """
SELECT bp.type,
       COUNT(*) as building_count,
       SUM(bp.area) as total_footprint_area,
       AVG(bp.height) as avg_height
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.area > {min_area}
GROUP BY bp.type
HAVING COUNT(*) >= {min_buildings}
ORDER BY building_count DESC;
        """.strip(),
        "Analyze building density and metrics by type for buildings with area > {min_area} sqm in project {project_id} scenario {scenario_id}, showing types with at least {min_buildings} buildings, ordered by count descending",
        {"building", "aggregation", "density", "basic"}
    ))
    
    # A20: Simple building list
    templates.append((
        "CIM_A20_simple_building_list",
        """
SELECT b.building_id, bp.type, bp.height, bp.area
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
LIMIT {limit};
        """.strip(),
        "List buildings with basic properties in project {project_id} scenario {scenario_id}, limit to {limit} results",
        {"building", "list", "basic"}
    ))
    
    # ==========================================================================
    # PRIORITY 2: CROSS-SCHEMA WITH CIM_VECTOR
    # Queries combining cim_vector with census/network/raster
    # ==========================================================================
    
    # A8: Buildings inside census zone
    templates.append((
        "CIM_A8_buildings_in_census",
        """
SELECT b.building_id, bp.type, bp.height, 
       c.sez2011,
       public.ST_Area(b.building_geometry) as building_area_sqm
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_census.censusgeo c ON public.ST_Within(public.ST_Centroid(b.building_geometry), c.geometry)
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
LIMIT {limit};
        """.strip(),
        "Find buildings within census zones for project {project_id} scenario {scenario_id}, limit to {limit} results",
        {"building", "census", "spatial_predicate", "basic"}
    ))
    
    # A9: Census zones intersecting project boundary
    templates.append((
        "CIM_A9_census_in_project",
        """
SELECT c.sez2011,
       c.p1 as total_population,
       public.ST_Area(c.geometry) as census_area_sqm
FROM cim_census.censusgeo c
JOIN cim_vector.cim_wizard_project_scenario ps ON public.ST_Intersects(c.geometry, ps.project_boundary)
WHERE ps.project_id = '{project_id}'
  AND ps.scenario_id = '{scenario_id}'
ORDER BY c.p1 DESC
LIMIT {limit};
        """.strip(),
        "Find census zones that intersect with project {project_id} scenario {scenario_id} boundary, ordered by total population descending, limit to {limit} results",
        {"census", "project", "spatial_predicate", "basic"}
    ))
    
    # ========== COMPLEXITY B: INTERMEDIATE OPERATIONS ==========
    
    # B1: Building statistics by type
    templates.append((
        "CIM_B1_building_stats_by_type",
        """
SELECT bp.type as building_type,
       COUNT(*) as building_count,
       AVG(bp.height) as avg_height,
       AVG(bp.area) as avg_area,
       SUM(bp.n_people) as total_population
FROM cim_vector.cim_wizard_building_properties bp
JOIN cim_vector.cim_wizard_building b ON bp.building_id = b.building_id AND bp.lod = b.lod
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.type
ORDER BY building_count DESC;
        """.strip(),
        "Calculate building statistics (count, average height, average area, total population) grouped by type for project {project_id} scenario {scenario_id}, ordered by building count descending",
        {"building", "aggregation", "statistics", "grouping"}
    ))
    
    # B2: Buildings near grid infrastructure
    templates.append((
        "CIM_B2_buildings_near_grid",
        """
SELECT b.building_id, 
       bp.type,
       bp.height,
       public.ST_Distance(b.building_geometry, gb.geometry) as distance_to_grid_m,
       gb.voltage_kv
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
CROSS JOIN cim_network.network_buses gb
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND gb.voltage_kv >= {voltage_kv}
  AND gb.in_service = true
  AND public.ST_DWithin(b.building_geometry, gb.geometry, {max_distance})
ORDER BY distance_to_grid_m ASC
LIMIT {limit};
        """.strip(),
        "Find buildings in project {project_id} scenario {scenario_id} within {max_distance} meters of active grid buses with voltage at or above {voltage_kv} kV, ordered by distance ascending, limit to {limit} results",
        {"building", "grid", "distance", "proximity", "spatial_join"}
    ))
    
    # B3: Building-census aggregation
    templates.append((
        "CIM_B3_building_census_aggregation",
        """
SELECT cg.comune as municipality,
       COUNT(b.building_id) as buildings_count,
       SUM(bp.n_people) as total_population,
       AVG(bp.area) as avg_building_area
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_census.censusgeo cg ON public.ST_Within(public.ST_Centroid(b.building_geometry), cg.geometry)
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
GROUP BY cg.comune
ORDER BY total_population DESC
LIMIT {limit};
        """.strip(),
        "Aggregate building data (count, total population, average area) by census municipality for project {project_id} scenario {scenario_id}, ordered by total population descending, limit to {limit} results",
        {"building", "census", "aggregation", "spatial_join"}
    ))
    
    # B4: Grid line connectivity
    templates.append((
        "CIM_B4_grid_line_connectivity",
        """
SELECT gl.line_id, gl.line_name, gl.length_km,
       gb1.bus_name as from_bus_name,
       gb2.bus_name as to_bus_name,
       gb1.voltage_kv
FROM cim_network.network_lines gl
JOIN cim_network.network_buses gb1 ON gl.from_bus_id = gb1.bus_id
JOIN cim_network.network_buses gb2 ON gl.to_bus_id = gb2.bus_id
WHERE gl.in_service = true
  AND gb1.in_service = true
  AND gb2.in_service = true
ORDER BY gl.length_km DESC
LIMIT {limit};
        """.strip(),
        "Analyze electrical grid line connectivity between active bus stations, ordered by line length descending, limit to {limit} results",
        {"grid", "network", "connectivity", "multi_join"}
    ))
    
    # B5: Building buffering analysis
    templates.append((
        "CIM_B5_building_buffer_analysis",
        """
WITH buffered_buildings AS (
  SELECT b.building_id, 
         bp.type,
         public.ST_Buffer(b.building_geometry, {buffer_distance}) as buffer_geom
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
)
SELECT building_id, type, public.ST_Area(buffer_geom) as buffer_area_sqm
FROM buffered_buildings
ORDER BY buffer_area_sqm DESC
LIMIT {limit};
        """.strip(),
        "Create {buffer_distance} meter buffers around buildings in project {project_id} scenario {scenario_id} and calculate buffer areas, ordered by area descending, limit to {limit} results",
        {"building", "buffer", "processing", "cte"}
    ))
    
    # B6: Census employment analysis
    templates.append((
        "CIM_B6_census_employment",
        """
SELECT COUNT(*) as census_areas,
       AVG((c.p62::float / NULLIF(c.p60, 0)) * 100) as avg_unemployment_rate,
       SUM(c.p61) as total_employed
FROM cim_census.censusgeo c
WHERE c.p60 > 0;
        """.strip(),
        "Analyze employment and unemployment rates across all census zones with working age population greater than zero",
        {"census", "employment", "aggregation", "statistics"}
    ))
    
    # B7: Nearest buildings to a specific building
    templates.append((
        "CIM_B7_nearest_buildings",
        """
SELECT b1.building_id,
       bp1.type,
       bp1.height,
       public.ST_Distance(public.ST_Centroid(b1.building_geometry), public.ST_Centroid(b2.building_geometry)) as distance_m
FROM cim_vector.cim_wizard_building b1
JOIN cim_vector.cim_wizard_building_properties bp1 ON b1.building_id = bp1.building_id AND b1.lod = bp1.lod
CROSS JOIN cim_vector.cim_wizard_building b2
WHERE bp1.project_id = '{project_id}'
  AND bp1.scenario_id = '{scenario_id}'
  AND b2.building_id = '{census_id}'
  AND b1.building_id != b2.building_id
  AND public.ST_Distance(public.ST_Centroid(b1.building_geometry), public.ST_Centroid(b2.building_geometry)) < {max_distance}
ORDER BY distance_m ASC
LIMIT 10;
        """.strip(),
        "Find 10 nearest buildings to building {census_id} within {max_distance} meters in project {project_id} scenario {scenario_id}, using centroid distance and ordered by distance ascending",
        {"building", "distance", "nearest_neighbor", "proximity"}
    ))
    
    # B8: Closest grid bus to a building
    templates.append((
        "CIM_B8_closest_grid_to_building",
        """
SELECT gb.bus_id,
       gb.name,
       gb.voltage_kv,
       public.ST_Distance(public.ST_Centroid(b.building_geometry), gb.geometry) as distance_m
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
CROSS JOIN cim_network.network_buses gb
WHERE b.building_id = '{census_id}'
  AND bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND gb.in_service = true
ORDER BY distance_m ASC
LIMIT 1;
        """.strip(),
        "Find the closest active grid bus to building {census_id} in project {project_id} scenario {scenario_id} by centroid distance",
        {"building", "grid", "nearest_neighbor", "distance"}
    ))
    
    # B9: Average raster elevation analysis
    templates.append((
        "CIM_B9_raster_average_elevation",
        """
SELECT 
    'DTM' as raster_type,
    AVG((public.ST_SummaryStats(rast)).mean) as avg_elevation,
    COUNT(*) as tile_count
FROM cim_raster.dtm
UNION ALL
SELECT 
    'DSM' as raster_type,
    AVG((public.ST_SummaryStats(rast)).mean) as avg_elevation,
    COUNT(*) as tile_count
FROM cim_raster.dsm_sansalva;
        """.strip(),
        "Calculate average elevation values and tile counts from DTM and DSM rasters",
        {"raster", "elevation", "statistics", "analysis"}
    ))
    
    # ========== COMPLEXITY C: ADVANCED OPERATIONS ==========
    
    # C1: Building type and area analysis
    templates.append((
        "CIM_C1_building_type_area_analysis",
        """
WITH building_metrics AS (
  SELECT b.building_id,
         bp.type,
         bp.height as declared_height,
         public.ST_Area(b.building_geometry) as footprint_area,
         bp.n_people,
         bp.area as building_area,
         CASE 
           WHEN bp.height > 20 THEN 'high_rise'
           WHEN bp.height > 10 THEN 'mid_rise'
           ELSE 'low_rise'
         END as height_category
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
),
type_analysis AS (
  SELECT type,
         height_category,
         COUNT(*) as building_count,
         AVG(declared_height) as avg_height,
         AVG(footprint_area) as avg_footprint,
         SUM(n_people) as total_residents
  FROM building_metrics
  GROUP BY type, height_category
)
SELECT type, height_category, building_count, 
       ROUND((avg_height)::numeric, 2) as avg_height_m,
       ROUND((avg_footprint)::numeric, 2) as avg_footprint_sqm,
       total_residents
FROM type_analysis
ORDER BY building_count DESC;
        """.strip(),
        "Analyze building type distribution and height categories (high_rise > 20m, mid_rise > 10m, low_rise <= 10m) for project {project_id} scenario {scenario_id}, showing count, average height, average footprint, and total residents, ordered by building count descending",
        {"building", "type_analysis", "height_analysis", "advanced", "cte"}
    ))
    
    # C2: Spatial clustering of buildings
    templates.append((
        "CIM_C2_building_clustering",
        """
WITH spatial_clusters AS (
  SELECT b.building_id, bp.type, bp.n_people,
         public.ST_ClusterDBSCAN(public.ST_Centroid(b.building_geometry), eps := {cluster_distance}, minpoints := {min_points}) 
         OVER (PARTITION BY bp.type) AS cluster_id
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
),
cluster_stats AS (
  SELECT cluster_id, type,
         COUNT(*) AS building_count,
         SUM(n_people) AS total_residents
  FROM spatial_clusters
  WHERE cluster_id IS NOT NULL
  GROUP BY cluster_id, type
  HAVING COUNT(*) >= {min_cluster_size}
)
SELECT cluster_id, type, building_count, total_residents
FROM cluster_stats
ORDER BY total_residents DESC
LIMIT {limit};
        """.strip(),
        "Perform DBSCAN spatial clustering on buildings by type in project {project_id} scenario {scenario_id} with epsilon {cluster_distance} meters and minimum {min_points} points, showing clusters with at least {min_cluster_size} buildings, ordered by total residents descending, limit to {limit} results",
        {"building", "clustering", "advanced", "cte", "window_function"}
    ))
    
    # C3: Multi-schema integration analysis
    templates.append((
        "CIM_C3_multi_schema_integration",
        """
WITH building_census_overlay AS (
  SELECT b.building_id, bp.type, bp.height, bp.area, bp.n_people,
         c.sez2011, c.p1 as census_population,
         public.ST_Area(public.ST_Intersection(b.building_geometry, c.geometry)) / public.ST_Area(b.building_geometry) as overlap_ratio
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  JOIN cim_census.censusgeo c ON public.ST_Intersects(b.building_geometry, c.geometry)
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
    AND public.ST_Area(public.ST_Intersection(b.building_geometry, c.geometry)) / public.ST_Area(b.building_geometry) > 0.5
),
grid_proximity AS (
  SELECT bco.building_id, bco.type, bco.height,
         MIN(public.ST_Distance(b.building_geometry, gb.geometry)) as min_grid_distance
  FROM building_census_overlay bco
  JOIN cim_vector.cim_wizard_building b ON bco.building_id = b.building_id
  CROSS JOIN cim_network.network_buses gb
  WHERE gb.in_service = true
  GROUP BY bco.building_id, bco.type, bco.height
)
SELECT type,
       COUNT(*) as building_count,
       AVG(height) as avg_height,
       AVG(min_grid_distance) as avg_grid_distance
FROM grid_proximity
GROUP BY type
HAVING COUNT(*) >= {min_buildings}
ORDER BY building_count DESC;
        """.strip(),
        "Comprehensive multi-schema analysis integrating buildings from project {project_id} scenario {scenario_id} with census data (requiring > 50% overlap) and grid infrastructure, grouped by building type, showing only groups with at least {min_buildings} buildings, ordered by building count descending",
        {"building", "census", "grid", "multi_schema", "advanced", "cte", "spatial_join"}
    ))
    
    # C4: Raster value extraction at building centroids
    templates.append((
        "CIM_C4_raster_value_extraction",
        """
SELECT b.building_id,
       bp.type,
       bp.height as declared_height,
       public.ST_Value(dtm.rast, public.ST_Centroid(b.building_geometry)) as ground_elevation,
       public.ST_Value(dsm.rast, public.ST_Centroid(b.building_geometry)) as surface_elevation,
       public.ST_Area(b.building_geometry) as footprint_area
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_raster.dtm dtm ON public.ST_Intersects(dtm.rast, b.building_geometry)
JOIN cim_raster.dsm_sansalva dsm ON public.ST_Intersects(dsm.rast, b.building_geometry)
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND bp.type = '{building_type}'
LIMIT {limit};
        """.strip(),
        "Extract DTM and DSM raster elevation values at centroids of {building_type} buildings in project {project_id} scenario {scenario_id}, limit to {limit} results",
        {"building", "raster", "raster_vector", "advanced", "multi_join"}
    ))
    
    # C5: Census demographic transition analysis
    templates.append((
        "CIM_C5_census_demographic_transition",
        """
WITH demographic_indicators AS (
  SELECT c.sez2011,
         c.p1 as total_population,
         (c.p14 + c.p15 + c.p16) as youth_0_14,
         (c.p27 + c.p28 + c.p29) as elderly_65_plus,
         c.pf3 as single_households,
         c.p47 as university_graduates,
         ROUND((((c.p27 + c.p28 + c.p29)::float / NULLIF((c.p14 + c.p15 + c.p16))::numeric, 0)), 2) as aging_ratio,
         ROUND(((c.p47::float / NULLIF(c.p1)::numeric, 0)) * 100, 1) as education_modernization
  FROM cim_census.censusgeo c
  WHERE c.p1 >= {min_population}
),
transition_classification AS (
  SELECT "SEZ2011",
         aging_ratio, education_modernization,
         CASE 
           WHEN aging_ratio > 1.5 AND education_modernization > 10 THEN 'POST_TRANSITION_ADVANCED'
           WHEN aging_ratio > 1.0 THEN 'LATE_TRANSITION'
           ELSE 'MID_TRANSITION'
         END as demographic_stage
  FROM demographic_indicators
)
SELECT demographic_stage,
       COUNT(*) as areas_count,
       AVG(aging_ratio) as avg_aging_ratio,
       AVG(education_modernization) as avg_education_mod
FROM transition_classification
GROUP BY demographic_stage
HAVING COUNT(*) >= {min_areas}
ORDER BY avg_aging_ratio DESC;
        """.strip(),
        "Comprehensive demographic transition analysis with minimum population {min_population}, classifying areas by aging ratio (elderly/youth) and education modernization (university graduates percentage), grouped by demographic stage, showing only groups with at least {min_areas} areas, ordered by aging ratio descending",
        {"census", "demographics", "advanced", "cte", "statistical_analysis"}
    ))
    
    # C6: Merge census zones for project boundary
    templates.append((
        "CIM_C6_merge_census_zones",
        """
WITH project_census AS (
  SELECT ps.project_id, ps.scenario_id, ps.project_name,
         c.sez2011, c.geometry
  FROM cim_vector.cim_wizard_project_scenario ps
  JOIN cim_census.censusgeo c ON public.ST_Intersects(ps.project_boundary, c.geometry)
  WHERE ps.project_id = '{project_id}'
    AND ps.scenario_id = '{scenario_id}'
)
SELECT project_id, scenario_id, project_name,
       COUNT(DISTINCT "SEZ2011") as census_zones_count,
       public.ST_Union(geometry) as merged_census_boundary,
       public.ST_Area(public.ST_Union(geometry)) as total_area_sqm
FROM project_census
GROUP BY project_id, scenario_id, project_name;
        """.strip(),
        "Merge all census zones intersecting with project {project_id} scenario {scenario_id} boundary to create unified census boundary, showing count of zones and total area",
        {"census", "project", "union", "aggregation", "advanced", "cte"}
    ))
    
    # C7: Projects with overlapping land coverage
    templates.append((
        "CIM_C7_overlapping_projects",
        """
WITH project_overlaps AS (
  SELECT p1.project_id as project1_id,
         p1.project_name as project1_name,
         p2.project_id as project2_id,
         p2.project_name as project2_name,
         public.ST_Area(p1.project_boundary) as project1_area,
         public.ST_Area(public.ST_Intersection(p1.project_boundary, p2.project_boundary)) as overlap_area
  FROM cim_vector.cim_wizard_project_scenario p1
  CROSS JOIN cim_vector.cim_wizard_project_scenario p2
  WHERE p1.project_id < p2.project_id
    AND public.ST_Intersects(p1.project_boundary, p2.project_boundary)
),
overlap_percentages AS (
  SELECT project1_id, project1_name,
         project2_id, project2_name,
         overlap_area,
         ROUND(((overlap_area / NULLIF(project1_area)::numeric, 0)) * 100, 2) as overlap_percentage
  FROM project_overlaps
)
SELECT project1_id, project1_name,
       project2_id, project2_name,
       overlap_area as overlap_sqm,
       overlap_percentage
FROM overlap_percentages
WHERE overlap_percentage >= {overlap_threshold}
ORDER BY overlap_percentage DESC;
        """.strip(),
        "Find project pairs with land coverage overlap of at least {overlap_threshold} percent, showing overlap area and percentage, ordered by overlap percentage descending",
        {"project", "overlap", "intersection", "percentage", "advanced", "cte"}
    ))
    
    # C8: Clip raster by building footprint
    templates.append((
        "CIM_C8_clip_raster_by_building",
        """
WITH building_geom AS (
  SELECT b.building_id, bp.type, b.building_geometry
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE b.building_id = '{census_id}'
    AND bp.project_id = '{project_id}'
    AND bp.scenario_id = '{scenario_id}'
)
SELECT bg.building_id,
       bg.type,
       public.ST_Clip(dtm.rast, bg.building_geometry, true) as clipped_dtm_raster,
       (public.ST_SummaryStats(public.ST_Clip(dtm.rast, bg.building_geometry, true))).mean as avg_ground_elevation
FROM building_geom bg
JOIN cim_raster.dtm dtm ON public.ST_Intersects(dtm.rast, bg.building_geometry)
LIMIT 1;
        """.strip(),
        "Clip DTM raster by footprint of building {census_id} in project {project_id} scenario {scenario_id} and extract average ground elevation statistics",
        {"building", "raster", "clip", "processing", "advanced", "cte"}
    ))
    
    # C9: Calculate building height from DSM and DTM difference
    templates.append((
        "CIM_C9_building_height_from_rasters",
        """
WITH building_geom AS (
  SELECT b.building_id, bp.type, bp.height as declared_height, b.building_geometry
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE b.building_id = '{census_id}'
    AND bp.project_id = '{project_id}'
    AND bp.scenario_id = '{scenario_id}'
),
raster_values AS (
  SELECT bg.building_id,
         bg.type,
         bg.declared_height,
         (public.ST_SummaryStats(public.ST_Clip(dsm.rast, bg.building_geometry, true))).mean as avg_surface_elevation,
         (public.ST_SummaryStats(public.ST_Clip(dtm.rast, bg.building_geometry, true))).mean as avg_ground_elevation
  FROM building_geom bg
  JOIN cim_raster.dsm_sansalva dsm ON public.ST_Intersects(dsm.rast, bg.building_geometry)
  JOIN cim_raster.dtm dtm ON public.ST_Intersects(dtm.rast, bg.building_geometry)
)
SELECT building_id,
       type,
       declared_height,
       ROUND((avg_surface_elevation)::numeric, 2) as avg_surface_elevation_m,
       ROUND((avg_ground_elevation)::numeric, 2) as avg_ground_elevation_m,
       ROUND((avg_surface_elevation - avg_ground_elevation)::numeric, 2) as calculated_height_m,
       ROUND((ABS(declared_height - (avg_surface_elevation - avg_ground_elevation)))::numeric, 2) as height_difference_m
FROM raster_values;
        """.strip(),
        "Calculate building height from DSM and DTM raster difference for building {census_id} in project {project_id} scenario {scenario_id}, comparing calculated height (DSM - DTM) with declared height and showing the difference",
        {"building", "raster", "clip", "height_calculation", "advanced", "cte", "multi_raster"}
    ))
    
    return templates

# ============================================================================
# SAMPLE CREATION
# ============================================================================

def create_comprehensive_sample(
    sample_id: str,
    sql_pair: SqlPair,
    values: Dict,
    database_id: int = 1,
    include_results: bool = False
) -> Dict:
    """Create comprehensive training sample with all metadata"""
    
    # Extract spatial functions
    spatial_functions = extract_spatial_functions(sql_pair.postgis_sql)
    
    # Get table and schema information
    tables = list(sql_pair.evidence.get('tables', []))
    schemas = list(sql_pair.evidence.get('schemas', []))
    
    # Create metadata for classification
    metadata = {
        'spatial_functions': spatial_functions,
        'tables': tables,
        'schemas': schemas
    }
    
    # Perform classifications
    sql_type = classify_sql_type(sql_pair.postgis_sql, metadata)
    question_tone = classify_question_tone(sql_pair.natural_language_desc)
    difficulty = calculate_difficulty_dimensions(sql_pair.postgis_sql, metadata)
    
    # Classify functions
    function_info = []
    for func in spatial_functions:
        function_info.append({
            "name": func,
            "data_type": classify_function_data_type(func),
            "usage_frequency": classify_function_usage(func),
            "difficulty": classify_function_difficulty(func)
        })
    
    # Determine overall usage frequency for the query
    usage_frequencies = [f["usage_frequency"] for f in function_info]
    if "most_frequent" in usage_frequencies:
        query_usage_frequency = "most_frequent"
    elif "frequent" in usage_frequencies:
        query_usage_frequency = "frequent"
    else:
        query_usage_frequency = "low_frequent"
    
    # Determine priority
    template_priority = getattr(sql_pair, 'priority', infer_template_priority(sql_pair.template_id, sql_pair.tags))
    
    # Create comprehensive sample
    comprehensive_sample = {
        # Core Identifiers
        "id": sample_id,
        "database_id": database_id,
        "database_name": "cim_wizard",
        
        # Natural Language Question
        "question": sql_pair.natural_language_desc,
        "question_tone": question_tone,
        
        # Priority
        "priority": template_priority,
        
        # SQL Queries
        "sql_postgis": sql_pair.postgis_sql,
        "sql_spatialite": sql_pair.spatialite_sql,
        
        # SQL Classification & Taxonomy
        "sql_type": sql_type,
        "sql_taxonomy": {
            "operation_type": sql_type,
            "has_cte": "WITH" in sql_pair.postgis_sql.upper(),
            "has_subquery": "(SELECT" in sql_pair.postgis_sql,
            "has_aggregation": "GROUP BY" in sql_pair.postgis_sql.upper(),
            "has_window_function": "PARTITION BY" in sql_pair.postgis_sql.upper(),
            "join_type": "spatial" if sql_type == "SPATIAL_JOIN" else "standard" if "JOIN" in sql_pair.postgis_sql.upper() else "none"
        },
        
        # Difficulty Dimensions
        "difficulty": difficulty,
        "difficulty_level": difficulty['overall_difficulty'],
        
        # Usage Frequency
        "usage_frequency": query_usage_frequency,
        
        # Database Schema Information
        "database_schema": {
            "schemas": schemas,
            "tables": tables,
            "primary_schema": schemas[0] if schemas else None,
            "table_count": len(tables),
            "schema_count": len(set(schemas))
        },
        
        # Spatial Functions with Classifications
        "spatial_functions": spatial_functions,
        "spatial_function_count": len(spatial_functions),
        "spatial_function_details": function_info,
        
        # Evidence
        "evidence": sql_pair.evidence,
        
        # Instruction for LLM Training
        "instruction": f"Convert this natural language question to PostGIS spatial SQL for the CIM Wizard database: {sql_pair.natural_language_desc}",
        
        # Results for Evaluation
        "results": None if include_results else [],
        "has_results": include_results,
        
        # Pipeline Metadata
        "stage": "stage1_cim",
        "generation_method": "rule_based_cim_wizard",
        "template_id": sql_pair.template_id,
        "complexity_level": sql_pair.complexity,
        "tags": list(sql_pair.tags),
        "generation_params": values,
        "generated_at": datetime.now().isoformat()
    }
    
    return comprehensive_sample

# ============================================================================
# STRATIFIED SAMPLING FOR EVALUATION SET
# ============================================================================

def stratified_evaluation_sampling(
    enhanced_samples: List[Dict],
    evaluation_sample_size: int = 100,
    random_seed: int = 42
) -> List[int]:
    """
    Perform stratified sampling for representative evaluation set
    
    Stratification Dimensions:
    1. SQL Type
    2. Difficulty Level (query_complexity)
    3. Usage Frequency
    4. Complexity Level (A, B, C)
    """
    
    random.seed(random_seed)
    
    print(f"\n[STRATIFIED SAMPLING] Creating representative evaluation set")
    print(f"Target size: {evaluation_sample_size} samples")
    
    # Group samples by stratification key
    strata = defaultdict(list)
    
    for idx, sample in enumerate(enhanced_samples):
        key = (
            sample['sql_type'],
            sample['difficulty']['query_complexity'],
            sample['usage_frequency'],
            sample['complexity_level']
        )
        strata[key].append(idx)
    
    print(f"  Found {len(strata)} unique strata combinations")
    
    # Calculate proportional allocation
    total_samples = len(enhanced_samples)
    selected_indices = []
    
    # First pass: Allocate proportionally
    allocation = {}
    for stratum_key, stratum_indices in strata.items():
        proportion = len(stratum_indices) / total_samples
        allocated_count = max(1, int(proportion * evaluation_sample_size))
        allocation[stratum_key] = allocated_count
    
    # Adjust if over-allocated
    total_allocated = sum(allocation.values())
    if total_allocated > evaluation_sample_size:
        sorted_strata = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
        excess = total_allocated - evaluation_sample_size
        for stratum_key, count in sorted_strata:
            if excess == 0:
                break
            reduction = min(count - 1, excess)
            allocation[stratum_key] -= reduction
            excess -= reduction
    
    # Under-allocated: add to largest strata
    total_allocated = sum(allocation.values())
    if total_allocated < evaluation_sample_size:
        sorted_strata = sorted(allocation.items(), key=lambda x: len(strata[x[0]]), reverse=True)
        deficit = evaluation_sample_size - total_allocated
        for stratum_key, count in sorted_strata:
            if deficit == 0:
                break
            available = len(strata[stratum_key]) - count
            addition = min(available, deficit)
            allocation[stratum_key] += addition
            deficit -= addition
    
    # Sample from each stratum
    print(f"\n  Stratification Summary:")
    print(f"  {'SQL Type':<25} {'Complexity':<12} {'Frequency':<15} {'Level':<6} {'Total':<8} {'Selected':<10}")
    print(f"  {'-'*85}")
    
    for stratum_key, stratum_indices in sorted(strata.items()):
        allocated_count = allocation[stratum_key]
        sampled = random.sample(stratum_indices, min(allocated_count, len(stratum_indices)))
        selected_indices.extend(sampled)
        
        sql_type, difficulty, freq, complexity = stratum_key
        print(f"  {sql_type:<25} {difficulty:<12} {freq:<15} {complexity:<6} {len(stratum_indices):<8} {len(sampled):<10}")
    
    print(f"  {'-'*85}")
    print(f"  {'TOTAL':<60} {total_samples:<8} {len(selected_indices):<10}")
    
    return selected_indices

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_stage1_cim_dataset(
    num_variations: int = 200,
    output_file: str = "training_datasets/stage1_cim_dataset.jsonl",
    evaluation_sample_size: int = 100,
    random_seed: int = 42
):
    """
    Generate Stage 1 CIM Wizard dataset with comprehensive metadata
    """
    
    random.seed(random_seed)
    
    print("="*80)
    print("STAGE 1: CIM WIZARD SPATIAL SQL DATASET GENERATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Variations per template: {num_variations}")
    print(f"  - Evaluation samples: {evaluation_sample_size}")
    print(f"  - Random seed: {random_seed}")
    print(f"  - Output file: {output_file}")
    
    # Generate base templates
    print("\n[1/5] Generating CIM Wizard templates...")
    templates = generate_cim_templates()
    print(f"      Total templates: {len(templates)}")
    
    # Create variations with LIMIT/ORDER BY strategy
    print(f"\n[2/5] Creating template variations with LIMIT/ORDER BY distribution (85/7/8)...")
    dataset = []
    limit_strategy_counts = {"FULL_RESULTS": 0, "ORDERED_ONLY": 0, "TOP_N": 0}
    
    for template_id, sql_template, nl_desc, tags in templates:
        # Determine complexity from template_id
        complexity = template_id.split('_')[1][0]  # Extract A, B, or C
        
        for i in range(num_variations):
            values = generate_realistic_values()
            
            try:
                # Apply parameter substitution to SQL
                postgis_sql = sql_template.format(**values)
                spatialite_sql = postgis_sql  # For now, same as PostGIS
                
                # Apply parameter substitution to natural language description
                enhanced_desc = nl_desc.format(**values)
                
                # PHASE 4: Apply LIMIT/ORDER BY strategy (85% no LIMIT, 7% ORDER only, 8% LIMIT+ORDER)
                limit_strategy = determine_limit_strategy(template_id, i)
                limit_strategy_counts[limit_strategy] += 1
                
                if limit_strategy == "FULL_RESULTS":
                    # Remove LIMIT and ORDER BY
                    postgis_sql = remove_limit_and_order(postgis_sql)
                    enhanced_desc = enhanced_desc.replace(", limit to {limit} results", "")
                    enhanced_desc = enhanced_desc.replace("limit to {limit} results", "")
                    enhanced_desc = enhanced_desc.replace(", ordered by", ", showing")
                    enhanced_desc = enhanced_desc.replace("ordered by", "showing")
                
                elif limit_strategy == "ORDERED_ONLY":
                    # Keep ORDER BY, remove LIMIT
                    postgis_sql = remove_limit_only(postgis_sql)
                    enhanced_desc = enhanced_desc.replace(", limit to {limit} results", "")
                    enhanced_desc = enhanced_desc.replace("limit to {limit} results", "")
                
                # TOP_N keeps both LIMIT and ORDER BY as-is
                
                # Extract evidence
                evidence = extract_evidence(postgis_sql, f"{template_id}_var_{i+1}", tags)
                
                # Create SqlPair with priority
                pair = SqlPair(
                    template_id=f"{template_id}_var_{i+1}",
                    complexity=complexity,
                    postgis_sql=postgis_sql,
                    spatialite_sql=spatialite_sql,
                    natural_language_desc=enhanced_desc,
                    tags=tags,
                    evidence=evidence
                )
                
                # Add priority metadata to pair
                pair.priority = infer_template_priority(template_id, tags)
                
                dataset.append(pair)
                
            except KeyError as e:
                print(f"      Warning: Template {template_id} missing parameter {e}, skipping variation {i+1}")
                continue
    
    print(f"      Generated {len(dataset)} SQL pairs")
    print(f"      LIMIT/ORDER BY distribution:")
    for strategy, count in limit_strategy_counts.items():
        percentage = count / len(dataset) * 100 if dataset else 0
        print(f"        {strategy:15s}: {count:5,} ({percentage:5.1f}%)")
    
    # Create enhanced samples with comprehensive metadata
    print(f"\n[3/5] Creating enhanced samples with comprehensive metadata...")
    enhanced_samples = []
    
    for i, pair in enumerate(dataset):
        values = generate_realistic_values()
        sample_id = f"cim_{i:06d}"
        
        enhanced_sample = create_comprehensive_sample(
            sample_id=sample_id,
            sql_pair=pair,
            values=values,
            database_id=1,
            include_results=False
        )
        
        # Add LIMIT/ORDER BY metadata (PHASE 4)
        enhanced_sample['has_limit'] = 'LIMIT' in pair.postgis_sql.upper()
        enhanced_sample['has_order_by'] = 'ORDER BY' in pair.postgis_sql.upper()
        enhanced_sample['limit_strategy'] = determine_limit_strategy(pair.template_id, i)
        
        enhanced_samples.append(enhanced_sample)
        
        if (i + 1) % 500 == 0:
            print(f"      Progress: {i + 1}/{len(dataset)} samples processed...")
    
    print(f"      Created {len(enhanced_samples)} enhanced samples")
    
    # Select evaluation samples
    print(f"\n[4/5] Selecting evaluation samples using stratified sampling...")
    eval_size = min(evaluation_sample_size, len(enhanced_samples))
    eval_indices = set(stratified_evaluation_sampling(
        enhanced_samples, 
        eval_size, 
        random_seed
    ))
    
    # Update evaluation flags
    for idx in eval_indices:
        enhanced_samples[idx]['has_results'] = True
        enhanced_samples[idx]['results'] = None
    
    # Save datasets
    print(f"\n[5/5] Saving datasets...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in enhanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      Main dataset: {output_file}")
    
    # Save evaluation subset
    eval_samples = [s for s in enhanced_samples if s['has_results']]
    eval_file = output_file.replace('.jsonl', '_eval.jsonl')
    with open(eval_file, 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      Evaluation subset: {eval_file} ({len(eval_samples)} samples)")
    
    # Generate statistics
    stats = generate_comprehensive_statistics(enhanced_samples)
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"      Statistics: {stats_file}")
    
    # Print summary
    print_summary_statistics(stats)
    
    return enhanced_samples, stats

def generate_comprehensive_statistics(samples: List[Dict]) -> Dict:
    """Generate comprehensive statistics"""
    
    stats = {
        "dataset_info": {
            "total_samples": len(samples),
            "evaluation_samples": len([s for s in samples if s['has_results']]),
            "training_samples": len([s for s in samples if not s['has_results']]),
            "generation_date": datetime.now().isoformat()
        },
        "priority_distribution": {},
        "sql_types": {},
        "question_tones": {},
        "difficulty_levels": {},
        "complexity_levels": {},
        "usage_frequency": {},
        "spatial_functions": {},
        "function_usage_frequency": {},
        "function_data_types": {},
        "schema_complexity": {}
    }
    
    # Collect statistics
    for sample in samples:
        # Priority distribution
        priority = sample.get('priority', 0)
        stats['priority_distribution'][f'Priority_{priority}'] = stats['priority_distribution'].get(f'Priority_{priority}', 0) + 1
        
        # SQL types
        sql_type = sample['sql_type']
        stats['sql_types'][sql_type] = stats['sql_types'].get(sql_type, 0) + 1
        
        # Question tones
        tone = sample['question_tone']
        stats['question_tones'][tone] = stats['question_tones'].get(tone, 0) + 1
        
        # Difficulty levels
        difficulty = sample['difficulty']['query_complexity']
        stats['difficulty_levels'][difficulty] = stats['difficulty_levels'].get(difficulty, 0) + 1
        
        # Complexity levels (A, B, C)
        complexity = sample['complexity_level']
        stats['complexity_levels'][complexity] = stats['complexity_levels'].get(complexity, 0) + 1
        
        # Usage frequency
        freq = sample['usage_frequency']
        stats['usage_frequency'][freq] = stats['usage_frequency'].get(freq, 0) + 1
        
        # Spatial functions
        for func in sample['spatial_functions']:
            stats['spatial_functions'][func] = stats['spatial_functions'].get(func, 0) + 1
        
        # Function details
        for func_detail in sample.get('spatial_function_details', []):
            func = func_detail['name']
            usage_freq = func_detail['usage_frequency']
            data_type = func_detail['data_type']
            
            stats['function_usage_frequency'][usage_freq] = stats['function_usage_frequency'].get(usage_freq, 0) + 1
            stats['function_data_types'][data_type] = stats['function_data_types'].get(data_type, 0) + 1
        
        # Schema complexity
        schema_complexity = sample['difficulty']['schema_complexity']
        stats['schema_complexity'][schema_complexity] = stats['schema_complexity'].get(schema_complexity, 0) + 1
    
    # Sort by frequency
    for key in ['sql_types', 'question_tones', 'difficulty_levels', 'complexity_levels', 
                'usage_frequency', 'spatial_functions', 'schema_complexity']:
        stats[key] = dict(sorted(stats[key].items(), key=lambda x: -x[1]))
    
    return stats

def print_summary_statistics(stats: Dict):
    """Print formatted summary statistics"""
    
    print("\n" + "="*80)
    print("STAGE 1 CIM WIZARD DATASET - SUMMARY STATISTICS")
    print("="*80)
    
    info = stats['dataset_info']
    print(f"\nDataset Overview:")
    print(f"   Total samples: {info['total_samples']:,}")
    print(f"   Training samples: {info['training_samples']:,}")
    print(f"   Evaluation samples: {info['evaluation_samples']:,}")
    
    print(f"\nPriority Distribution:")
    for priority in ['Priority_1', 'Priority_2', 'Priority_3']:
        count = stats['priority_distribution'].get(priority, 0)
        percentage = (count / info['total_samples']) * 100 if count > 0 else 0
        priority_desc = {
            'Priority_1': 'Inner cim_vector (highest)',
            'Priority_2': 'Cross-schema with cim_vector',
            'Priority_3': 'Inner census/network/raster'
        }.get(priority, '')
        print(f"   {priority} ({priority_desc}): {count:,} ({percentage:.1f}%)")
    
    print(f"\nSQL Type Distribution:")
    for sql_type, count in list(stats['sql_types'].items())[:10]:
        percentage = (count / info['total_samples']) * 100
        print(f"   {sql_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nComplexity Level Distribution:")
    for complexity, count in stats['complexity_levels'].items():
        percentage = (count / info['total_samples']) * 100
        print(f"   Level {complexity}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nDifficulty Distribution:")
    for difficulty in ['EASY', 'MEDIUM', 'HARD']:
        count = stats['difficulty_levels'].get(difficulty, 0)
        percentage = (count / info['total_samples']) * 100 if count > 0 else 0
        print(f"   {difficulty}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nUsage Frequency Distribution:")
    for freq in ['most_frequent', 'frequent', 'low_frequent']:
        count = stats['usage_frequency'].get(freq, 0)
        percentage = (count / info['total_samples']) * 100 if count > 0 else 0
        print(f"   {freq}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nFunction Data Types:")
    for data_type, count in stats['function_data_types'].items():
        print(f"   {data_type}: {count:,}")
    
    print(f"\nTop 15 Spatial Functions:")
    for i, (func, count) in enumerate(list(stats['spatial_functions'].items())[:15], 1):
        print(f"   {i}. {func}: {count:,}")
    
    print(f"\nSchema Complexity:")
    for complexity, count in stats['schema_complexity'].items():
        percentage = (count / info['total_samples']) * 100
        print(f"   {complexity}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nDataset Generation Complete!")
    print(f"   Ready for Stage 2 (SDV Synthetic Generation)")
    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_variations = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    eval_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Generate CIM Wizard dataset
    samples, stats = generate_stage1_cim_dataset(
        num_variations=num_variations,
        output_file="training_datasets/stage1_cim_dataset.jsonl",
        evaluation_sample_size=eval_size,
        random_seed=42
    )
    
    print(f"\nStage 1 CIM Wizard Dataset Successfully Created!")
    print(f"   Total samples: {len(samples):,}")
    print(f"   Output: training_datasets/stage1_cim_dataset.jsonl")
    print(f"\n  Next step: Run Stage 2 (SDV Synthetic Generation)")

