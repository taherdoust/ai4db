#!/usr/bin/env python3
"""
stage1_cim_v2.py - CIM Wizard Spatial SQL Generator (Redesigned)
Clear separation: TASK_TAXONOMY (SQL operations) vs DOMAIN_TAXONOMY (CIM schema)
All templates from stage1_cim.py converted to new taxonomy format
"""

from contextlib import nullcontext
import json
import random
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

# ============================================================================
# CIM WIZARD DATABASE SCHEMA DEFINITION
# Note: census metadata is not fully included yet
# ============================================================================

CIM_SCHEMAS = {
    "cim_vector": {
        "cim_wizard_project_scenario": {
            "columns": ["scenario_id", "project_id", "project_name", "scenario_name", "project_boundary", "project_center", "census_boundary"],
            "geometry_columns": ["project_boundary", "project_center", "census_boundary"]
        },
        "cim_wizard_building": {
            "columns": ["building_id", "lod", "building_geometry", "building_geometry_source", "census_id"],
            "geometry_columns": ["building_geometry"]
        },
        "cim_wizard_building_properties": {
            "columns": ["scenario_id", "building_id", "project_id", "lod", "height", "area", "volume", "number_of_floors", "n_family", "n_people", "type", "const_year", "const_period_census"],
            "numeric_columns": ["height", "area", "volume", "number_of_floors", "const_year", "n_people", "n_family"]
        }
    },
    "cim_census": {
        "censusgeo": {
            "columns": ["SEZ2011", "geometry", "P1", "P2", "P3", "p14", "p15", "p16", "p27", "p28", "p29", "p47", "p60", "p61", "p62", "pf3", "ST1", "ST2", "E8", "E9", "PF1", "PF2", "comune"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["P1", "P2", "P3", "ST1", "ST2", "E8", "E9", "PF1", "PF2"]
        }
    },
    "cim_network": {
        "network_buses": {
            "columns": ["bus_id", "bus_name", "name", "voltage_kv", "geometry", "in_service"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["voltage_kv"]
        },
        "network_lines": {
            "columns": ["line_id", "line_name", "from_bus_id", "to_bus_id", "geometry", "length_km", "in_service"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["length_km"]
        }
    },
    "cim_raster": {
        "dtm": {"columns": ["rid", "rast"], "raster_columns": ["rast"]},
        "dsm_sansalva": {"columns": ["rid", "rast"], "raster_columns": ["rast"]}
    }
}

# ============================================================================
# TASK TAXONOMY (SQL and Spatial SQL Operations)
# Based on SPATIAL_FUNCTIONS from stage1_cim.py
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
    "RASTER_VECTOR": {"complexity": 3, "frequency": 3, "description": "Raster-vector integration (ST_Clip, ST_Intersection with raster) and raster_processing functions like ST_Intersection_Raster"}
}

# ============================================================================
# DOMAIN TAXONOMY (CIM Wizard Schema Complexity)
# Complexity: 1=Easy, 2=Medium, 3=Hard
# Frequency: 1=Very Frequent, 2=Frequent, 3=Rare
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
    "INTERROGATIVE": 0.70,
    "DIRECT": 0.20,
    "DESCRIPTIVE": 0.10
}

# ============================================================================
# CIM PARAMETERS POOL
# ============================================================================

CIM_PARAMETERS = {
    "project_scenario_pairs": [
        {"project_id": "4be7d1ff-e8bf-4374-a13e-67e7b0d52eb1", "scenario_id": "4be7d1ff-e8bf-4374-a13e-67e7b0d52eb1", "project_name": "Sansalva_filter", "scenario_name": "baseline"},
        {"project_id": "5a00fa63-2ef4-4d38-baa0-48ae5d80d21a", "scenario_id": "5a00fa63-2ef4-4d38-baa0-48ae5d80d21a", "project_name": "Sansalva_filter", "scenario_name": "sansalva3.5"},
        {"project_id": "e983d9e4-70ce-43e1-b778-03c25d992433", "scenario_id": "e983d9e4-70ce-43e1-b778-03c25d992433", "project_name": "Sansalva_filter_3", "scenario_name": "sansalva3"},
        {"project_id": "aeba11e7-ab0e-46ca-b89f-4c856c0289bf", "scenario_id": "aeba11e7-ab0e-46ca-b89f-4c856c0289bf", "project_name": "Sansalva_filter_4", "scenario_name": "sansalva4"}
    ],
    "building_types": ["residential", "non-residential"],
    "voltage_levels": [0.4, 20.0, 132.0, 400.0],
    "building_ids": [
        "0033de25-d7d1-48d5-98c3-bc02973a13c0", "0040217f-9865-4cab-9478-731e0c443a85",
        "0047ba60-0b2f-4b72-878e-02fe589ed37f", "0054c487-661b-493d-a819-e143b6c66e52"
    ],
    "overlap_thresholds": [10, 20, 30, 40, 50]
}

def generate_realistic_values() -> Dict:
    """Generate realistic parameter values for CIM database queries"""
    proj_scen_pair = random.choice(CIM_PARAMETERS["project_scenario_pairs"])
    
    return {
        "project_id": proj_scen_pair["project_id"],
        "scenario_id": proj_scen_pair["scenario_id"],
        "project_name": proj_scen_pair["project_name"],
        "scenario_name": proj_scen_pair["scenario_name"],
        "building_type": random.choice(CIM_PARAMETERS["building_types"]),
        "building_id": random.choice(CIM_PARAMETERS["building_ids"]),
        "voltage_kv": random.choice(CIM_PARAMETERS["voltage_levels"]),
        "min_area": random.randint(50, 500),
        "max_area": random.randint(1000, 5000),
        "min_height": random.randint(3, 10),
        "max_height": random.randint(15, 100),
        "min_people": random.randint(1, 5),
        "year": random.randint(1950, 2024),
        "buffer_distance": random.choice([100, 500, 1000]),
        "max_distance": random.choice([500, 1000, 2000]),
        "cluster_distance": random.choice([1000, 2000, 5000]),
        "min_points": random.choice([3, 5, 8]),
        "min_cluster_size": random.choice([3, 5, 10]),
        "min_buildings": random.choice([5, 10, 20]),
        "min_areas": random.choice([3, 5, 10]),
        "min_population": random.choice([100, 500, 1000]),
        "overlap_threshold": random.choice(CIM_PARAMETERS["overlap_thresholds"]),
        "lon": round(random.uniform(7.0, 18.0), 6),
        "lat": round(random.uniform(36.0, 47.0), 6),
        "srid": 4326
    }

# ============================================================================
# SQL PAIR DATA CLASS
# ============================================================================

@dataclass
class SqlPair:
    """SQL query pair with task and domain taxonomy tags"""
    template_id: str
    #sample_id_rule: str
    #sample_id_syntetized: str
    #sample_id_augmented: str
    task_complexity: int      # 1=Easy, 2=Medium, 3=Hard
    task_frequency: int       # 1=Very Frequent, 2=Frequent, 3=Rare
    domain_complexity: int    # 1=Easy, 2=Medium, 3=Hard
    domain_frequency: int     # 1=Very Frequent, 2=Frequent, 3=Rare
    task_type: str
    domain_type: str
    question_tone: str
    dirtiness: str
    natural_language_question: str
    postgis_sql: str

# ============================================================================
# TEMPLATE POOL - All templates from stage1_cim.py converted to new format
# ============================================================================

def generate_template_pool() -> List[SqlPair]:
    """Generate comprehensive template pool with task and domain taxonomy"""
    templates = []
    
    # ==========================================================================
    # PRIORITY 1: INNER CIM_VECTOR SCHEMA - SIMPLE_SELECT (task_complexity=1)
    # ==========================================================================
    
    # A1: Simple building selection by type and area (uses ST_Area measurement)
    templates.append(SqlPair(
        template_id="A1", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SPATIAL_MEASUREMENT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find {building_type} buildings with area greater than {min_area} square meters in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT b.building_id, b.lod, ST_Area(b.building_geometry) as area_sqm
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND bp.type = '{building_type}'
  AND ST_Area(b.building_geometry) > {min_area};
        """.strip()
    ))
    
    # A2: Project at location (uses ST_Intersects predicate, ST_MakePoint constructor)
    templates.append(SqlPair(
        template_id="A2", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SPATIAL_PREDICATE", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find project scenarios that contain the geographic point at longitude {lon} latitude {lat} with SRID {srid}",
        postgis_sql="""
SELECT ps.project_name, ps.scenario_name, ST_Area(ps.project_boundary) as project_area_sqm
FROM cim_vector.cim_wizard_project_scenario ps
WHERE ST_Intersects(ps.project_boundary, ST_SetSRID(ST_MakePoint({lon}, {lat}), {srid}));
        """.strip()
    ))
    
    # A3: Grid buses by voltage (uses ST_X, ST_Y accessors)
    templates.append(SqlPair(
        template_id="A3", task_complexity=1, task_frequency=2,
        domain_complexity=1, domain_frequency=2,
        task_type="SPATIAL_ACCESSOR", domain_type="SINGLE_SCHEMA_OTHER",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find active grid buses with voltage at or above {voltage_kv} kV",
        postgis_sql="""
SELECT gb.bus_id, gb.bus_name, gb.voltage_kv, ST_X(gb.geometry) as lon, ST_Y(gb.geometry) as lat
FROM cim_network.network_buses gb
WHERE gb.voltage_kv >= {voltage_kv}
  AND gb.in_service = true;
        """.strip()
    ))
    
    # A4: Census population by zone (Single schema other - census)
    templates.append(SqlPair(
        template_id="A4", task_complexity=1, task_frequency=2,
        domain_complexity=1, domain_frequency=2,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_OTHER",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the population distribution by gender in census areas with minimum population of {min_population}",
        postgis_sql="""
SELECT c.sez2011, c.p1 as total_population, c.p2 as male_population, c.p3 as female_population
FROM cim_census.censusgeo c
WHERE c.p1 >= {min_population}
ORDER BY c.p1 DESC;
        """.strip()
    ))
    
    # A5: Building height from properties
    templates.append(SqlPair(
        template_id="A5", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Retrieve building heights of at least {min_height} meters from properties for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.height, bp.type, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND bp.height >= {min_height}
ORDER BY bp.height DESC;
        """.strip()
    ))
    
    # A6: Building distance calculation
    templates.append(SqlPair(
        template_id="A6", task_complexity=2, task_frequency=2,
        domain_complexity=1, domain_frequency=1,
        task_type="SPATIAL_MEASUREMENT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the distance from buildings in project {project_id} scenario {scenario_id} to the point at longitude {lon} latitude {lat}",
        postgis_sql="""
SELECT b.building_id, 
       ST_Distance(b.building_geometry, ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)) as distance_m
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
ORDER BY distance_m ASC;
        """.strip()
    ))
    
    # A7: Buildings inside project boundary (spatial join with ST_Intersects)
    templates.append(SqlPair(
        template_id="A7", task_complexity=2, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SPATIAL_JOIN", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings that intersect with the boundary of project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT b.building_id, b.lod, bp.type, bp.height, ST_Area(b.building_geometry) as area_sqm
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_vector.cim_wizard_project_scenario ps ON bp.project_id = ps.project_id AND bp.scenario_id = ps.scenario_id
WHERE ps.project_id = '{project_id}'
  AND ps.scenario_id = '{scenario_id}'
  AND ST_Intersects(b.building_geometry, ps.project_boundary);
        """.strip()
    ))
    
    # A8: Buildings inside census zone (Multi-schema)
    templates.append(SqlPair(
        template_id="A8", task_complexity=2, task_frequency=1,
        domain_complexity=2, domain_frequency=2,
        task_type="SPATIAL_JOIN", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings within census zones for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT b.building_id, bp.type, bp.height, c.sez2011, ST_Area(b.building_geometry) as building_area_sqm
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_census.censusgeo c ON ST_Within(ST_Centroid(b.building_geometry), c.geometry)
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}';
        """.strip()
    ))
    
    # A9: Census zones intersecting project boundary (Multi-schema)
    templates.append(SqlPair(
        template_id="A9", task_complexity=2, task_frequency=1,
        domain_complexity=2, domain_frequency=2,
        task_type="SPATIAL_JOIN", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find census zones that intersect with project {project_id} scenario {scenario_id} boundary",
        postgis_sql="""
SELECT c.sez2011, c.p1 as total_population, ST_Area(c.geometry) as census_area_sqm
FROM cim_census.censusgeo c
JOIN cim_vector.cim_wizard_project_scenario ps ON ST_Intersects(c.geometry, ps.project_boundary)
WHERE ps.project_id = '{project_id}'
  AND ps.scenario_id = '{scenario_id}'
ORDER BY c.p1 DESC;
        """.strip()
    ))
    
    # A10: Count buildings by type
    templates.append(SqlPair(
        template_id="A10", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="How many buildings of each type are in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.type, COUNT(*) as building_count
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.type
ORDER BY building_count DESC;
        """.strip()
    ))
    
    # A11: Buildings by area range
    templates.append(SqlPair(
        template_id="A11", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings with area between {min_area} and {max_area} square meters in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.area, bp.height
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.area BETWEEN {min_area} AND {max_area}
ORDER BY bp.area DESC;
        """.strip()
    ))
    
    # A12: Average building metrics
    templates.append(SqlPair(
        template_id="A12", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the average height, area, volume and floors of buildings in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT AVG(bp.height) as avg_height,
       AVG(bp.area) as avg_area,
       AVG(bp.volume) as avg_volume,
       AVG(bp.number_of_floors) as avg_floors,
       COUNT(*) as total_buildings
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}';
        """.strip()
    ))
    
    # A13: Buildings by construction year
    templates.append(SqlPair(
        template_id="A13", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings constructed since year {year} in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.const_year, bp.height, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.const_year >= {year}
ORDER BY bp.const_year DESC;
        """.strip()
    ))
    
    # A14: Tall buildings
    templates.append(SqlPair(
        template_id="A14", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find tall buildings with height at least {min_height} meters in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.height, bp.number_of_floors, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.height >= {min_height}
ORDER BY bp.height DESC;
        """.strip()
    ))
    
    # A15: Buildings by number of floors
    templates.append(SqlPair(
        template_id="A15", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings with at least {min_people} floors in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.number_of_floors, bp.height, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.number_of_floors >= {min_people}
ORDER BY bp.number_of_floors DESC;
        """.strip()
    ))
    
    # A16: Total building area by type
    templates.append(SqlPair(
        template_id="A16", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the total and average building area by type for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.type, COUNT(*) as building_count, SUM(bp.area) as total_area, AVG(bp.area) as avg_area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.type
ORDER BY total_area DESC;
        """.strip()
    ))
    
    # A17: Buildings with population data
    templates.append(SqlPair(
        template_id="A17", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings with population data in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.n_people, bp.n_family, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.n_people > 0
ORDER BY bp.n_people DESC;
        """.strip()
    ))
    
    # A18: Project scenario summary (uses subquery)
    templates.append(SqlPair(
        template_id="A18", task_complexity=2, task_frequency=2,
        domain_complexity=1, domain_frequency=1,
        task_type="NESTED_QUERY", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Get project scenario summary with boundary area and building count for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT ps.project_id, ps.project_name, ps.scenario_name,
       ST_Area(ps.project_boundary) as boundary_area_sqm,
       (SELECT COUNT(*) FROM cim_vector.cim_wizard_building_properties bp 
        WHERE bp.project_id = ps.project_id AND bp.scenario_id = ps.scenario_id) as building_count
FROM cim_vector.cim_wizard_project_scenario ps
WHERE ps.project_id = '{project_id}'
  AND ps.scenario_id = '{scenario_id}';
        """.strip()
    ))
    
    # A19: Building density by area
    templates.append(SqlPair(
        template_id="A19", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the building density by type for buildings with area greater than {min_area} sqm in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.type, COUNT(*) as building_count, SUM(bp.area) as total_footprint_area, AVG(bp.height) as avg_height
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.area > {min_area}
GROUP BY bp.type
HAVING COUNT(*) >= {min_buildings}
ORDER BY building_count DESC;
        """.strip()
    ))
    
    # A20: Simple building list
    templates.append(SqlPair(
        template_id="A20", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_JOIN", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="List buildings with basic properties in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT b.building_id, bp.type, bp.height, bp.area
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}';
        """.strip()
    ))
    
    # A21: Buildings with specific height range
    templates.append(SqlPair(
        template_id="A21", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings with height between {min_height} and {max_height} meters in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.height, bp.area, bp.n_people
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.height BETWEEN {min_height} AND {max_height};
        """.strip()
    ))
    
    # A22: Count residential buildings
    templates.append(SqlPair(
        template_id="A22", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="How many residential buildings are in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT COUNT(*) as residential_count
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.type = 'residential';
        """.strip()
    ))
    
    # A23: Maximum building height
    templates.append(SqlPair(
        template_id="A23", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the maximum, minimum and average building height for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT MAX(bp.height) as max_height, MIN(bp.height) as min_height, AVG(bp.height) as avg_height
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}';
        """.strip()
    ))
    
    # A24: Buildings with families
    templates.append(SqlPair(
        template_id="A24", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings with at least {min_people} families in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.n_family, bp.n_people, bp.area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.n_family >= {min_people};
        """.strip()
    ))
    
    # A25: Total population in project
    templates.append(SqlPair(
        template_id="A25", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the total population and average people per building for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT SUM(bp.n_people) as total_population, COUNT(*) as building_count, AVG(bp.n_people) as avg_people_per_building
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}';
        """.strip()
    ))
    
    # A26: Buildings by construction period
    templates.append(SqlPair(
        template_id="A26", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="How are buildings grouped by construction period for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.const_period_census, COUNT(*) as building_count, AVG(bp.area) as avg_area
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.const_period_census
ORDER BY building_count DESC;
        """.strip()
    ))
    
    # A27: Large buildings
    templates.append(SqlPair(
        template_id="A27", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find large buildings with area at least {max_area} sqm in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.area, bp.volume, bp.height
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.area >= {max_area}
ORDER BY bp.area DESC;
        """.strip()
    ))
    
    # A28: Buildings with many floors
    templates.append(SqlPair(
        template_id="A28", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SIMPLE_SELECT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings with {min_people} or more floors in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.building_id, bp.type, bp.number_of_floors, bp.height
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.number_of_floors >= {min_people};
        """.strip()
    ))
    
    # A29: Average building metrics by type
    templates.append(SqlPair(
        template_id="A29", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the average height, area and volume by building type for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.type, AVG(bp.height) as avg_height, AVG(bp.area) as avg_area, AVG(bp.volume) as avg_volume
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.type;
        """.strip()
    ))
    
    # A30: Building volume statistics
    templates.append(SqlPair(
        template_id="A30", task_complexity=1, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What are the volume statistics for buildings in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT COUNT(*) as building_count, SUM(bp.volume) as total_volume, AVG(bp.volume) as avg_volume, MAX(bp.volume) as max_volume
FROM cim_vector.cim_wizard_building_properties bp
WHERE bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND bp.volume > 0;
        """.strip()
    ))
    
    # ==========================================================================
    # COMPLEXITY B: INTERMEDIATE OPERATIONS
    # ==========================================================================
    
    # B1: Building statistics by type
    templates.append(SqlPair(
        template_id="B1", task_complexity=2, task_frequency=2,
        domain_complexity=1, domain_frequency=1,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What are the building statistics by type for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT bp.type as building_type, COUNT(*) as building_count, AVG(bp.height) as avg_height, AVG(bp.area) as avg_area, SUM(bp.n_people) as total_population
FROM cim_vector.cim_wizard_building_properties bp
JOIN cim_vector.cim_wizard_building b ON bp.building_id = b.building_id AND bp.lod = b.lod
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
GROUP BY bp.type
ORDER BY building_count DESC;
        """.strip()
    ))
    
    # B2: Buildings near grid infrastructure (Multi-schema)
    templates.append(SqlPair(
        template_id="B2", task_complexity=2, task_frequency=2,
        domain_complexity=2, domain_frequency=2,
        task_type="SPATIAL_JOIN", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find buildings in project {project_id} scenario {scenario_id} within {max_distance} meters of active grid buses with voltage at or above {voltage_kv} kV",
        postgis_sql="""
SELECT b.building_id, bp.type, bp.height, ST_Distance(b.building_geometry, gb.geometry) as distance_to_grid_m, gb.voltage_kv
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
CROSS JOIN cim_network.network_buses gb
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND gb.voltage_kv >= {voltage_kv}
  AND gb.in_service = true
  AND ST_DWithin(b.building_geometry, gb.geometry, {max_distance})
ORDER BY distance_to_grid_m ASC;
        """.strip()
    ))
    
    # B3: Building-census aggregation (Multi-schema)
    templates.append(SqlPair(
        template_id="B3", task_complexity=2, task_frequency=2,
        domain_complexity=2, domain_frequency=2,
        task_type="SPATIAL_JOIN", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the building count and population by municipality for project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT cg.comune as municipality, COUNT(b.building_id) as buildings_count, SUM(bp.n_people) as total_population, AVG(bp.area) as avg_building_area
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_census.censusgeo cg ON ST_Within(ST_Centroid(b.building_geometry), cg.geometry)
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
GROUP BY cg.comune
ORDER BY total_population DESC;
        """.strip()
    ))
    
    # B4: Grid line connectivity (Single schema other - network)
    templates.append(SqlPair(
        template_id="B4", task_complexity=2, task_frequency=3,
        domain_complexity=1, domain_frequency=2,
        task_type="MULTI_SQL_JOIN", domain_type="SINGLE_SCHEMA_OTHER",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the electrical grid line connectivity between active bus stations",
        postgis_sql="""
SELECT gl.line_id, gl.line_name, gl.length_km, gb1.bus_name as from_bus_name, gb2.bus_name as to_bus_name, gb1.voltage_kv
FROM cim_network.network_lines gl
JOIN cim_network.network_buses gb1 ON gl.from_bus_id = gb1.bus_id
JOIN cim_network.network_buses gb2 ON gl.to_bus_id = gb2.bus_id
WHERE gl.in_service = true
  AND gb1.in_service = true
  AND gb2.in_service = true
ORDER BY gl.length_km DESC;
        """.strip()
    ))
    
    # B5: Building buffering analysis (uses ST_Buffer spatial processing)
    templates.append(SqlPair(
        template_id="B5", task_complexity=2, task_frequency=1,
        domain_complexity=1, domain_frequency=1,
        task_type="SPATIAL_PROCESSING", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DESCRIPTIVE", dirtiness="CLEAN",
        natural_language_question="I need to create {buffer_distance} meter buffers around buildings in project {project_id} scenario {scenario_id} and calculate buffer areas",
        postgis_sql="""
WITH buffered_buildings AS (
  SELECT b.building_id, bp.type, ST_Buffer(b.building_geometry, {buffer_distance}) as buffer_geom
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
)
SELECT building_id, type, ST_Area(buffer_geom) as buffer_area_sqm
FROM buffered_buildings
ORDER BY buffer_area_sqm DESC;
        """.strip()
    ))
    
    # B6: Census employment analysis (Single schema other - census)
    templates.append(SqlPair(
        template_id="B6", task_complexity=2, task_frequency=3,
        domain_complexity=1, domain_frequency=2,
        task_type="SQL_AGGREGATION", domain_type="SINGLE_SCHEMA_OTHER",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the employment and unemployment rate across census zones",
        postgis_sql="""
SELECT COUNT(*) as census_areas, AVG((c.p62::float / NULLIF(c.p60, 0)) * 100) as avg_unemployment_rate, SUM(c.p61) as total_employed
FROM cim_census.censusgeo c
WHERE c.p60 > 0;
        """.strip()
    ))
    
    # B7: Nearest buildings to a specific building
    templates.append(SqlPair(
        template_id="B7", task_complexity=2, task_frequency=2,
        domain_complexity=1, domain_frequency=1,
        task_type="SPATIAL_MEASUREMENT", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find 10 nearest buildings to building {building_id} within {max_distance} meters in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT b1.building_id, bp1.type, bp1.height, ST_Distance(ST_Centroid(b1.building_geometry), ST_Centroid(b2.building_geometry)) as distance_m
FROM cim_vector.cim_wizard_building b1
JOIN cim_vector.cim_wizard_building_properties bp1 ON b1.building_id = bp1.building_id AND b1.lod = bp1.lod
CROSS JOIN cim_vector.cim_wizard_building b2
WHERE bp1.project_id = '{project_id}'
  AND bp1.scenario_id = '{scenario_id}'
  AND b2.building_id = '{building_id}'
  AND b1.building_id != b2.building_id
  AND ST_Distance(ST_Centroid(b1.building_geometry), ST_Centroid(b2.building_geometry)) < {max_distance}
ORDER BY distance_m ASC
LIMIT 10;
        """.strip()
    ))
    
    # B8: Closest grid bus to a building (Multi-schema)
    templates.append(SqlPair(
        template_id="B8", task_complexity=2, task_frequency=2,
        domain_complexity=2, domain_frequency=2,
        task_type="SPATIAL_MEASUREMENT", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find the closest active grid bus to building {building_id} in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT gb.bus_id, gb.bus_name, gb.voltage_kv, ST_Distance(ST_Centroid(b.building_geometry), gb.geometry) as distance_m
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
CROSS JOIN cim_network.network_buses gb
WHERE b.building_id = '{building_id}'
  AND bp.project_id = '{project_id}'
  AND bp.scenario_id = '{scenario_id}'
  AND gb.in_service = true
ORDER BY distance_m ASC
LIMIT 1;
        """.strip()
    ))
    
    # B9: Average raster elevation analysis (uses ST_SummaryStats raster analysis)
    templates.append(SqlPair(
        template_id="B9", task_complexity=3, task_frequency=2,
        domain_complexity=1, domain_frequency=3,
        task_type="RASTER_ANALYSIS", domain_type="SINGLE_SCHEMA_OTHER",
        question_tone="INTERROGATIVE", dirtiness="CLEAN",
        natural_language_question="What is the average elevation from DTM and DSM rasters",
        postgis_sql="""
SELECT 'DTM' as raster_type, AVG((ST_SummaryStats(rast)).mean) as avg_elevation, COUNT(*) as tile_count
FROM cim_raster.dtm
UNION ALL
SELECT 'DSM' as raster_type, AVG((ST_SummaryStats(rast)).mean) as avg_elevation, COUNT(*) as tile_count
FROM cim_raster.dsm_sansalva;
        """.strip()
    ))
    
    # ==========================================================================
    # COMPLEXITY C: ADVANCED OPERATIONS
    # ==========================================================================
    
    # C1: Building type and area analysis (Complex CTE)
    templates.append(SqlPair(
        template_id="C1", task_complexity=3, task_frequency=3,
        domain_complexity=1, domain_frequency=1,
        task_type="NESTED_QUERY", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DESCRIPTIVE", dirtiness="CLEAN",
        natural_language_question="I want to analyze building type distribution and height categories for project {project_id} scenario {scenario_id}",
        postgis_sql="""
WITH building_metrics AS (
  SELECT b.building_id, bp.type, bp.height as declared_height, ST_Area(b.building_geometry) as footprint_area, bp.n_people, bp.area as building_area,
         CASE WHEN bp.height > 20 THEN 'high_rise' WHEN bp.height > 10 THEN 'mid_rise' ELSE 'low_rise' END as height_category
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
),
type_analysis AS (
  SELECT type, height_category, COUNT(*) as building_count, AVG(declared_height) as avg_height, AVG(footprint_area) as avg_footprint, SUM(n_people) as total_residents
  FROM building_metrics
  GROUP BY type, height_category
)
SELECT type, height_category, building_count, ROUND((avg_height)::numeric, 2) as avg_height_m, ROUND((avg_footprint)::numeric, 2) as avg_footprint_sqm, total_residents
FROM type_analysis
ORDER BY building_count DESC;
        """.strip()
    ))
    
    # C2: Spatial clustering of buildings
    templates.append(SqlPair(
        template_id="C2", task_complexity=3, task_frequency=3,
        domain_complexity=1, domain_frequency=1,
        task_type="SPATIAL_CLUSTERING", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DESCRIPTIVE", dirtiness="CLEAN",
        natural_language_question="I want to perform DBSCAN spatial clustering on buildings in project {project_id} scenario {scenario_id} with {cluster_distance} meter epsilon",
        postgis_sql="""
WITH spatial_clusters AS (
  SELECT b.building_id, bp.type, bp.n_people,
         ST_ClusterDBSCAN(ST_Centroid(b.building_geometry), eps := {cluster_distance}, minpoints := {min_points}) OVER (PARTITION BY bp.type) AS cluster_id
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
),
cluster_stats AS (
  SELECT cluster_id, type, COUNT(*) AS building_count, SUM(n_people) AS total_residents
  FROM spatial_clusters
  WHERE cluster_id IS NOT NULL
  GROUP BY cluster_id, type
  HAVING COUNT(*) >= {min_cluster_size}
)
SELECT cluster_id, type, building_count, total_residents
FROM cluster_stats
ORDER BY total_residents DESC;
        """.strip()
    ))
    
    # C3: Multi-schema integration analysis (uses ST_Intersection processing, multiple schemas)
    templates.append(SqlPair(
        template_id="C3", task_complexity=3, task_frequency=3,
        domain_complexity=3, domain_frequency=3,
        task_type="MULTI_SPATIAL_JOIN", domain_type="MULTI_SCHEMA_COMPLEX",
        question_tone="DESCRIPTIVE", dirtiness="CLEAN",
        natural_language_question="I need comprehensive multi-schema analysis integrating buildings with census and grid infrastructure for project {project_id} scenario {scenario_id}",
        postgis_sql="""
WITH building_census_overlay AS (
  SELECT b.building_id, bp.type, bp.height, bp.area, bp.n_people, c.sez2011, c.p1 as census_population,
         ST_Area(ST_Intersection(b.building_geometry, c.geometry)) / ST_Area(b.building_geometry) as overlap_ratio
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  JOIN cim_census.censusgeo c ON ST_Intersects(b.building_geometry, c.geometry)
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
    AND ST_Area(ST_Intersection(b.building_geometry, c.geometry)) / ST_Area(b.building_geometry) > 0.5
),
grid_proximity AS (
  SELECT bco.building_id, bco.type, bco.height, MIN(ST_Distance(b.building_geometry, gb.geometry)) as min_grid_distance
  FROM building_census_overlay bco
  JOIN cim_vector.cim_wizard_building b ON bco.building_id = b.building_id
  CROSS JOIN cim_network.network_buses gb
  WHERE gb.in_service = true
  GROUP BY bco.building_id, bco.type, bco.height
)
SELECT type, COUNT(*) as building_count, AVG(height) as avg_height, AVG(min_grid_distance) as avg_grid_distance
FROM grid_proximity
GROUP BY type
HAVING COUNT(*) >= {min_buildings}
ORDER BY building_count DESC;
        """.strip()
    ))
    
    # C4: Raster value extraction at building centroids (Raster-vector)
    templates.append(SqlPair(
        template_id="C4", task_complexity=3, task_frequency=3,
        domain_complexity=2, domain_frequency=3,
        task_type="RASTER_VECTOR", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Extract DTM and DSM raster elevation values at centroids of {building_type} buildings in project {project_id} scenario {scenario_id}",
        postgis_sql="""
SELECT b.building_id, bp.type, bp.height as declared_height,
       ST_Value(dtm.rast, ST_Centroid(b.building_geometry)) as ground_elevation,
       ST_Value(dsm.rast, ST_Centroid(b.building_geometry)) as surface_elevation,
       ST_Area(b.building_geometry) as footprint_area
FROM cim_vector.cim_wizard_building b
JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
JOIN cim_raster.dtm dtm ON ST_Intersects(dtm.rast, b.building_geometry)
JOIN cim_raster.dsm_sansalva dsm ON ST_Intersects(dsm.rast, b.building_geometry)
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND bp.type = '{building_type}';
        """.strip()
    ))
    
    # C5: Census demographic transition analysis (Single schema other - census)
    templates.append(SqlPair(
        template_id="C5", task_complexity=3, task_frequency=3,
        domain_complexity=1, domain_frequency=2,
        task_type="NESTED_QUERY", domain_type="SINGLE_SCHEMA_OTHER",
        question_tone="DESCRIPTIVE", dirtiness="CLEAN",
        natural_language_question="I want comprehensive demographic transition analysis with minimum population {min_population}",
        postgis_sql="""
WITH demographic_indicators AS (
  SELECT c.sez2011, c.p1 as total_population, (c.p14 + c.p15 + c.p16) as youth_0_14, (c.p27 + c.p28 + c.p29) as elderly_65_plus,
         c.pf3 as single_households, c.p47 as university_graduates,
         ROUND((((c.p27 + c.p28 + c.p29)::float / NULLIF((c.p14 + c.p15 + c.p16), 0))::numeric), 2) as aging_ratio,
         ROUND(((c.p47::float / NULLIF(c.p1, 0))::numeric) * 100, 1) as education_modernization
  FROM cim_census.censusgeo c
  WHERE c.p1 >= {min_population}
),
transition_classification AS (
  SELECT sez2011, aging_ratio, education_modernization,
         CASE WHEN aging_ratio > 1.5 AND education_modernization > 10 THEN 'POST_TRANSITION_ADVANCED'
              WHEN aging_ratio > 1.0 THEN 'LATE_TRANSITION'
              ELSE 'MID_TRANSITION' END as demographic_stage
  FROM demographic_indicators
)
SELECT demographic_stage, COUNT(*) as areas_count, AVG(aging_ratio) as avg_aging_ratio, AVG(education_modernization) as avg_education_mod
FROM transition_classification
GROUP BY demographic_stage
HAVING COUNT(*) >= {min_areas}
ORDER BY avg_aging_ratio DESC;
        """.strip()
    ))
    
    # C6: Merge census zones for project boundary (Multi-schema with CTE)
    templates.append(SqlPair(
        template_id="C6", task_complexity=3, task_frequency=3,
        domain_complexity=2, domain_frequency=2,
        task_type="NESTED_QUERY", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Merge all census zones intersecting with project {project_id} scenario {scenario_id} boundary",
        postgis_sql="""
WITH project_census AS (
  SELECT ps.project_id, ps.scenario_id, ps.project_name, c.sez2011, c.geometry
  FROM cim_vector.cim_wizard_project_scenario ps
  JOIN cim_census.censusgeo c ON ST_Intersects(ps.project_boundary, c.geometry)
  WHERE ps.project_id = '{project_id}'
    AND ps.scenario_id = '{scenario_id}'
)
SELECT project_id, scenario_id, project_name, COUNT(DISTINCT sez2011) as census_zones_count,
       ST_Union(geometry) as merged_census_boundary, ST_Area(ST_Union(geometry)) as total_area_sqm
FROM project_census
GROUP BY project_id, scenario_id, project_name;
        """.strip()
    ))
    
    # C7: Projects with overlapping land coverage (CTE with self-join)
    templates.append(SqlPair(
        template_id="C7", task_complexity=3, task_frequency=3,
        domain_complexity=1, domain_frequency=1,
        task_type="NESTED_QUERY", domain_type="SINGLE_SCHEMA_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Find project pairs with land coverage overlap of at least {overlap_threshold} percent",
        postgis_sql="""
WITH project_overlaps AS (
  SELECT p1.project_id as project1_id, p1.project_name as project1_name,
         p2.project_id as project2_id, p2.project_name as project2_name,
         ST_Area(p1.project_boundary) as project1_area,
         ST_Area(ST_Intersection(p1.project_boundary, p2.project_boundary)) as overlap_area
  FROM cim_vector.cim_wizard_project_scenario p1
  CROSS JOIN cim_vector.cim_wizard_project_scenario p2
  WHERE p1.project_id < p2.project_id
    AND ST_Intersects(p1.project_boundary, p2.project_boundary)
),
overlap_percentages AS (
  SELECT project1_id, project1_name, project2_id, project2_name, overlap_area,
         ROUND(((overlap_area / NULLIF(project1_area, 0))::numeric) * 100, 2) as overlap_percentage
  FROM project_overlaps
)
SELECT project1_id, project1_name, project2_id, project2_name, overlap_area as overlap_sqm, overlap_percentage
FROM overlap_percentages
WHERE overlap_percentage >= {overlap_threshold}
ORDER BY overlap_percentage DESC;
        """.strip()
    ))
    
    # C8: Clip raster by building footprint (Raster-vector)
    templates.append(SqlPair(
        template_id="C8", task_complexity=3, task_frequency=3,
        domain_complexity=2, domain_frequency=3,
        task_type="RASTER_VECTOR", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Clip DTM raster by footprint of building {building_id} in project {project_id} scenario {scenario_id}",
        postgis_sql="""
WITH building_geom AS (
  SELECT b.building_id, bp.type, b.building_geometry
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE b.building_id = '{building_id}'
    AND bp.project_id = '{project_id}'
    AND bp.scenario_id = '{scenario_id}'
)
SELECT bg.building_id, bg.type,
       ST_Clip(dtm.rast, bg.building_geometry, true) as clipped_dtm_raster,
       (ST_SummaryStats(ST_Clip(dtm.rast, bg.building_geometry, true))).mean as avg_ground_elevation
FROM building_geom bg
JOIN cim_raster.dtm dtm ON ST_Intersects(dtm.rast, bg.building_geometry)
LIMIT 1;
        """.strip()
    ))
    
    # C9: Calculate building height from DSM and DTM difference (Multi-raster)
    templates.append(SqlPair(
        template_id="C9", task_complexity=3, task_frequency=3,
        domain_complexity=2, domain_frequency=3,
        task_type="RASTER_VECTOR", domain_type="MULTI_SCHEMA_WITH_CIM_VECTOR",
        question_tone="DIRECT", dirtiness="CLEAN",
        natural_language_question="Calculate building height from DSM and DTM raster difference for building {building_id} in project {project_id} scenario {scenario_id}",
        postgis_sql="""
WITH building_geom AS (
  SELECT b.building_id, bp.type, bp.height as declared_height, b.building_geometry
  FROM cim_vector.cim_wizard_building b
  JOIN cim_vector.cim_wizard_building_properties bp ON b.building_id = bp.building_id AND b.lod = bp.lod
  WHERE b.building_id = '{building_id}'
    AND bp.project_id = '{project_id}'
    AND bp.scenario_id = '{scenario_id}'
),
raster_values AS (
  SELECT bg.building_id, bg.type, bg.declared_height,
         (ST_SummaryStats(ST_Clip(dsm.rast, bg.building_geometry, true))).mean as avg_surface_elevation,
         (ST_SummaryStats(ST_Clip(dtm.rast, bg.building_geometry, true))).mean as avg_ground_elevation
  FROM building_geom bg
  JOIN cim_raster.dsm_sansalva dsm ON ST_Intersects(dsm.rast, bg.building_geometry)
  JOIN cim_raster.dtm dtm ON ST_Intersects(dtm.rast, bg.building_geometry)
)
SELECT building_id, type, declared_height,
       ROUND((avg_surface_elevation)::numeric, 2) as avg_surface_elevation_m,
       ROUND((avg_ground_elevation)::numeric, 2) as avg_ground_elevation_m,
       ROUND((avg_surface_elevation - avg_ground_elevation)::numeric, 2) as calculated_height_m,
       ROUND((ABS(declared_height - (avg_surface_elevation - avg_ground_elevation)))::numeric, 2) as height_difference_m
FROM raster_values;
        """.strip()
    ))
    
    return templates

# ============================================================================
# DATASET GENERATOR
# ============================================================================

def generate_stage1_dataset(
    strategy: str = "frequency",
    output_file: str = "stage1_v2_dataset.jsonl",
    random_seed: int = 42
):
    """Generate Stage 1 CIM Wizard dataset with three strategies"""
    
    random.seed(random_seed)
    
    print("="*80)
    print("STAGE 1: CIM WIZARD DATASET GENERATION")
    print("="*80)
    print(f"Strategy: {strategy.upper()}")
    print(f"Output file: {output_file}")
    
    # Generate template pool
    print("\n[1/3] Generating template pool...")
    templates = generate_template_pool()
    print(f"      Total templates: {len(templates)}")
    
    # Determine multiplication factors based on strategy
    print(f"\n[2/3] Applying {strategy} strategy...")
    
    if strategy == "frequency":
        def get_multiplier(template):
            freq_multiplier = {1: 200, 2: 50, 3: 10}
            return freq_multiplier[template.task_frequency] * freq_multiplier[template.domain_frequency] // 100
    elif strategy == "complexity":
        def get_multiplier(template):
            comp_multiplier = {1: 50, 2: 100, 3: 200}
            return comp_multiplier[template.task_complexity] * comp_multiplier[template.domain_complexity] // 100
    else:  # balance
        def get_multiplier(template):
            return 100
    
    # Create variations
    dataset = []
    for template in templates:
        multiplier = max(1, get_multiplier(template))
        
        for i in range(multiplier):
            values = generate_realistic_values()
            
            try:
                question = template.natural_language_question.format(**values)
                sql = template.postgis_sql.format(**values)
                
                
                
                
                sample = {
                    "id_rule": f"cim_{len(dataset):06d}",
                    
                    "task_complexity": template.task_complexity,
                    "task_frequency": template.task_frequency,
                    "task_type": template.task_type,
                    "domain_complexity": template.domain_complexity,
                    "domain_frequency": template.domain_frequency,
                    "domain_type": template.domain_type,
                    
                    "question_tone": template.question_tone,
                    "sample_dirtiness": template.dirtiness,
                    
                    "question": question,
                    "sql_postgis": sql,
                    
                }
                
                dataset.append(sample)
                
            except KeyError as e:
                print(f"      Warning: Template {template.template_id} missing parameter {e}")
                continue
    
    print(f"      Generated {len(dataset)} samples")
    
    # Save dataset
    print(f"\n[3/3] Saving dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"      Saved to: {output_file}")
    
    # Print distribution analysis
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)
    
    task_types = Counter(s['task_type'] for s in dataset)
    domain_types = Counter(s['domain_type'] for s in dataset)
    task_complexities = Counter(s['task_complexity'] for s in dataset)
    task_frequencies = Counter(s['task_frequency'] for s in dataset)
    domain_complexities = Counter(s['domain_complexity'] for s in dataset)
    domain_frequencies = Counter(s['domain_frequency'] for s in dataset)
    
    print("\nTask Type Distribution:")
    for task_type, count in task_types.most_common():
        percentage = count / len(dataset) * 100
        print(f"  {task_type:25s}: {count:6,} ({percentage:5.1f}%)")
    
    print("\nDomain Type Distribution:")
    for domain_type, count in domain_types.most_common():
        percentage = count / len(dataset) * 100
        print(f"  {domain_type:35s}: {count:6,} ({percentage:5.1f}%)")
    
    print("\nTask Complexity Distribution:")
    for complexity, count in sorted(task_complexities.items()):
        percentage = count / len(dataset) * 100
        label = {1: "Easy", 2: "Medium", 3: "Hard"}[complexity]
        print(f"  {complexity} ({label:6s}): {count:6,} ({percentage:5.1f}%)")
    
    print("\nTask Frequency Distribution:")
    for frequency, count in sorted(task_frequencies.items()):
        percentage = count / len(dataset) * 100
        label = {1: "Very Frequent", 2: "Frequent", 3: "Rare"}[frequency]
        print(f"  {frequency} ({label:12s}): {count:6,} ({percentage:5.1f}%)")
    
    print("\nDomain Complexity Distribution:")
    for complexity, count in sorted(domain_complexities.items()):
        percentage = count / len(dataset) * 100
        label = {1: "Easy", 2: "Medium", 3: "Hard"}[complexity]
        print(f"  {complexity} ({label:6s}): {count:6,} ({percentage:5.1f}%)")
    
    print("\nDomain Frequency Distribution:")
    for frequency, count in sorted(domain_frequencies.items()):
        percentage = count / len(dataset) * 100
        label = {1: "Very Frequent", 2: "Frequent", 3: "Rare"}[frequency]
        print(f"  {frequency} ({label:12s}): {count:6,} ({percentage:5.1f}%)")
    
    print("\n" + "="*80)
    print(f"Dataset generation complete: {len(dataset):,} samples")
    print(f"Strategy: {strategy}")
    print("="*80)
    
    return dataset

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    strategy = sys.argv[1] if len(sys.argv) > 1 else "frequency"
    
    if strategy not in ["frequency", "complexity", "balance"]:
        print(f"Invalid strategy: {strategy}")
        print("Valid options: frequency, complexity, balance")
        sys.exit(1)
    
    dataset = generate_stage1_dataset(
        strategy=strategy,
        output_file=f"stage1_v2_dataset_{strategy}.jsonl",
        random_seed=42
    )
    
    print(f"\nStage 1 Complete!")
    print(f"Total samples: {len(dataset):,}")
    print(f"Output: stage1_v2_dataset_{strategy}.jsonl")
