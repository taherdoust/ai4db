#!/usr/bin/env python3
"""
Enhanced Spatial SQL Generator for CIM Wizard Database Schema
Integrates with realistic database schema for comprehensive training data generation
"""

import random
import json
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from rule_based_ssql_generator import (
    Template, SqlPair, GEOM_TYPES, FUNCTION_FREQUENCY, 
    adapt, get_frequency_tags, export_training_dataset, generate_statistics,
    determine_usage_index, extract_evidence
)

# CIM Wizard Database Schema Configuration
CIM_SCHEMAS = {
    "cim_vector": {
        "project_scenario": {
            "columns": ["project_id", "scenario_id", "project_name", "scenario_name", "project_boundary", "project_center", "census_boundary"],
            "geometry_columns": ["project_boundary", "project_center", "census_boundary"],
            "srid": 4326
        },
        "building": {
            "columns": ["id", "building_id", "lod", "building_geometry", "building_geometry_source", "census_id"],
            "geometry_columns": ["building_geometry"],
            "srid": 4326
        },
        "building_properties": {
            "columns": ["id", "building_id", "project_id", "scenario_id", "height", "area", "volume", "number_of_floors", "type", "const_year", "n_people", "n_family", "gross_floor_area", "heating", "cooling", "hvac_type"],
            "numeric_columns": ["height", "area", "volume", "number_of_floors", "const_year", "n_people", "n_family", "gross_floor_area"],
            "categorical_columns": ["type", "hvac_type", "building_geometry_source"]
        },
        "grid_bus": {
            "columns": ["id", "network_id", "bus_id", "project_id", "scenario_id", "geometry", "name", "voltage_kv", "zone", "in_service"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["voltage_kv"],
            "srid": 4326
        },
        "grid_line": {
            "columns": ["id", "network_id", "line_id", "project_id", "scenario_id", "geometry", "name", "from_bus", "to_bus", "length_km", "max_loading_percent"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["length_km", "max_loading_percent"],
            "srid": 4326
        }
    },
    "cim_census": {
        "census_geo": {
            "columns": ["id", "SEZ2011", "geometry", "CODREG", "REGIONE", "PROVINCIA", "COMUNE", "P1", "P2", "P3", "ST1", "ST2", "A2", "A3", "PF1", "PF2", "E1", "E2"],
            "geometry_columns": ["geometry"],
            "numeric_columns": ["P1", "P2", "P3", "ST1", "ST2", "A2", "A3", "PF1", "PF2", "E1", "E2"],
            "categorical_columns": ["REGIONE", "PROVINCIA", "COMUNE"],
            "srid": 4326
        }
    },
    "cim_raster": {
        "building_height_cache": {
            "columns": ["id", "building_id", "project_id", "scenario_id", "dtm_avg_height", "dsm_avg_height", "building_height", "coverage_percentage", "confidence_score"],
            "numeric_columns": ["dtm_avg_height", "dsm_avg_height", "building_height", "coverage_percentage", "confidence_score"]
        },
        "dtm_raster": {
            "columns": ["id", "rast", "filename", "srid", "min_elevation", "max_elevation"],
            "raster_columns": ["rast"],
            "numeric_columns": ["min_elevation", "max_elevation"]
        },
        "dsm_raster": {
            "columns": ["id", "rast", "filename", "srid", "min_elevation", "max_elevation"],
            "raster_columns": ["rast"],
            "numeric_columns": ["min_elevation", "max_elevation"]
        }
    }
}

# Realistic parameter values for the CIM database
CIM_PARAMETER_POOLS = {
    "project_ids": ["milan_smart_district", "bologna_energy_hub", "rome_green_quarter", "turin_innovation_zone", "florence_heritage_area"],
    "scenario_ids": ["baseline", "renewable_2030", "efficiency_max", "grid_modernization", "zero_emission"],
    "building_types": ["residential", "commercial", "industrial", "mixed_use", "public"],
    "hvac_types": ["heat_pump", "gas_boiler", "district_heating", "electric", "hybrid"],
    "regions": ["Lombardia", "Emilia-Romagna", "Lazio", "Piemonte", "Toscana"],
    "provinces": ["Milano", "Bologna", "Roma", "Torino", "Firenze"],
    "voltage_levels": [0.4, 10, 20, 132, 220, 400],
    "srids": [4326, 3857, 32632, 32633],  # Common CRS for Italy
}

def generate_realistic_values() -> Dict[str, any]:
    """Generate realistic parameter values for CIM database queries"""
    return {
        "project_id": random.choice(CIM_PARAMETER_POOLS["project_ids"]),
        "scenario_id": random.choice(CIM_PARAMETER_POOLS["scenario_ids"]),
        "building_type": random.choice(CIM_PARAMETER_POOLS["building_types"]),
        "hvac_type": random.choice(CIM_PARAMETER_POOLS["hvac_types"]),
        "region": random.choice(CIM_PARAMETER_POOLS["regions"]),
        "province": random.choice(CIM_PARAMETER_POOLS["provinces"]),
        "voltage_kv": random.choice(CIM_PARAMETER_POOLS["voltage_levels"]),
        "srid": random.choice(CIM_PARAMETER_POOLS["srids"]),
        "buffer_distance": random.choice([100, 500, 1000, 2000]),
        "min_area": random.randint(50, 500),
        "max_area": random.randint(1000, 5000),
        "min_height": random.randint(3, 10),
        "max_height": random.randint(15, 100),
        "min_people": random.randint(1, 5),
        "max_people": random.randint(6, 20),
        "year": random.randint(1950, 2024),
        "census_id": random.randint(1000000, 9999999),
        "lon": round(random.uniform(7.0, 18.0), 6),  # Italy longitude range
        "lat": round(random.uniform(36.0, 47.0), 6),  # Italy latitude range
        "limit": random.choice([10, 25, 50, 100]),
        # New parameters for enhanced templates
        "cluster_count": random.choice([3, 5, 8, 10]),
        "min_cluster_size": random.choice([3, 5, 10]),
        "max_distance": random.choice([500, 1000, 2000, 5000]),
        "min_buildings": random.choice([5, 10, 20]),
        "min_points": random.choice([3, 5, 8]),
        # Census-specific parameters
        "min_areas": random.choice([3, 5, 10]),
        "cluster_distance": random.choice([1000, 2000, 5000]),
        "min_population": random.choice([100, 500, 1000])
    }

def cim_wizard_templates(dialect: str) -> List[Template]:
    """Generate CIM Wizard specific templates using realistic schema"""
    t: List[Template] = []

    # A-level: Basic CIM operations
    sql_a1 = adapt(dialect, """
SELECT b.building_id, b.lod, ST_Area(b.building_geometry) as area_sqm
FROM cim_vector.building b
JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND bp.type = '{building_type}'
  AND ST_Area(b.building_geometry) > {min_area};
""".strip())
    t.append(Template(
        id="CIM_A1_buildings_by_type_area",
        complexity="A", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Find buildings of specific type with area above threshold in a project scenario",
        tags={"cim_building", "area_filter", "type_filter", "basic"} | get_frequency_tags(sql_a1),
        sql=sql_a1))

    sql_a2 = adapt(dialect, """
SELECT ps.project_name, ps.scenario_name, ST_Area(ps.project_boundary) as project_area_sqm
FROM cim_vector.project_scenario ps
WHERE ST_Intersects(ps.project_boundary, ST_SetSRID(ST_MakePoint({lon}, {lat}), {srid}));
""".strip())
    t.append(Template(
        id="CIM_A2_project_at_location",
        complexity="A", dialect=dialect,
        geom_applicability=["POINT"],
        natural_language_desc="Find project scenarios that contain a specific geographic point",
        tags={"cim_project", "point_in_polygon", "basic"} | get_frequency_tags(sql_a2),
        sql=sql_a2))

    sql_a3 = adapt(dialect, """
SELECT gb.bus_id, gb.name, gb.voltage_kv, ST_AsText(gb.geometry) as location
FROM cim_vector.grid_bus gb
WHERE gb.project_id = '{project_id}' 
  AND gb.scenario_id = '{scenario_id}'
  AND gb.voltage_kv >= {voltage_kv}
  AND gb.in_service = true;
""".strip())
    t.append(Template(
        id="CIM_A3_grid_buses_by_voltage",
        complexity="A", dialect=dialect,
        geom_applicability=["POINT"],
        natural_language_desc="Find active grid buses above certain voltage level in a project scenario",
        tags={"cim_grid", "voltage_filter", "basic"} | get_frequency_tags(sql_a3),
        sql=sql_a3))

    # B-level: Intermediate CIM operations
    sql_b1 = adapt(dialect, """
SELECT bp.type as building_type,
       COUNT(*) as building_count,
       AVG(bp.height) as avg_height,
       AVG(bp.area) as avg_area,
       SUM(bp.n_people) as total_population
FROM cim_vector.building_properties bp
JOIN cim_vector.building b ON bp.building_id = b.building_id
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND b.building_geometry IS NOT NULL
GROUP BY bp.type
ORDER BY building_count DESC;
""".strip())
    t.append(Template(
        id="CIM_B1_building_stats_by_type",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Calculate building statistics grouped by type for a project scenario",
        tags={"cim_building", "aggregation", "statistics", "grouping"} | get_frequency_tags(sql_b1),
        sql=sql_b1))

    sql_b2 = adapt(dialect, """
SELECT b.building_id, 
       bp.height,
       ST_Distance(b.building_geometry, gb.geometry) as distance_to_grid_m
FROM cim_vector.building b
JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
JOIN cim_vector.grid_bus gb ON bp.project_id = gb.project_id AND bp.scenario_id = gb.scenario_id
WHERE bp.project_id = '{project_id}' 
  AND bp.scenario_id = '{scenario_id}'
  AND gb.voltage_kv >= {voltage_kv}
ORDER BY distance_to_grid_m ASC
LIMIT {limit};
""".strip())
    t.append(Template(
        id="CIM_B2_buildings_near_grid",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Find buildings closest to high-voltage grid infrastructure",
        tags={"cim_building", "cim_grid", "distance", "proximity"} | get_frequency_tags(sql_b2),
        sql=sql_b2))

    sql_b3 = adapt(dialect, """
WITH building_census AS (
  SELECT b.building_id, 
         b.census_id,
         bp.n_people,
         bp.area,
         ST_Centroid(b.building_geometry) as building_center
  FROM cim_vector.building b
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
)
SELECT cg.COMUNE as municipality,
       COUNT(bc.building_id) as buildings_count,
       SUM(bc.n_people) as total_population,
       AVG(bc.area) as avg_building_area
FROM building_census bc
JOIN cim_census.census_geo cg ON bc.census_id = cg.SEZ2011
GROUP BY cg.COMUNE
ORDER BY total_population DESC;
""".strip())
    t.append(Template(
        id="CIM_B3_building_census_aggregation",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Aggregate building data by census municipality boundaries",
        tags={"cim_building", "cim_census", "aggregation", "demographic"} | get_frequency_tags(sql_b3),
        sql=sql_b3))

    # C-level: Advanced CIM operations
    sql_c1 = adapt(dialect, """
WITH building_elevation AS (
  SELECT b.building_id,
         bp.height as declared_height,
         bhc.building_height as raster_height,
         ST_Area(b.building_geometry) as footprint_area,
         bp.volume,
         bp.n_people
  FROM cim_vector.building b
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  JOIN cim_raster.building_height_cache bhc ON b.building_id = bhc.building_id
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
),
height_analysis AS (
  SELECT building_id,
         declared_height,
         raster_height,
         ABS(declared_height - raster_height) as height_difference,
         footprint_area,
         volume,
         n_people,
         CASE 
           WHEN ABS(declared_height - raster_height) > 5 THEN 'significant_difference'
           WHEN ABS(declared_height - raster_height) > 2 THEN 'moderate_difference'
           ELSE 'consistent'
         END as height_consistency
  FROM building_elevation
  WHERE raster_height IS NOT NULL
)
SELECT height_consistency,
       COUNT(*) as building_count,
       AVG(height_difference) as avg_height_diff,
       AVG(footprint_area) as avg_footprint_area,
       SUM(n_people) as total_population
FROM height_analysis
GROUP BY height_consistency
ORDER BY building_count DESC;
""".strip())
    t.append(Template(
        id="CIM_C1_building_height_validation",
        complexity="C", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Validate building heights using raster elevation data and analyze discrepancies",
        tags={"cim_building", "cim_raster", "validation", "height_analysis", "advanced"} | get_frequency_tags(sql_c1),
        sql=sql_c1))

    sql_c2 = adapt(dialect, """
WITH grid_network AS (
  SELECT gl.line_id, gl.from_bus, gl.to_bus, gl.length_km, gl.max_loading_percent,
         ST_StartPoint(gl.geometry) as start_point,
         ST_EndPoint(gl.geometry) as end_point,
         gl.geometry as line_geom
  FROM cim_vector.grid_line gl
  WHERE gl.project_id = '{project_id}' AND gl.scenario_id = '{scenario_id}'
),
bus_connectivity AS (
  SELECT gb.bus_id, gb.voltage_kv, gb.geometry as bus_geom,
         COUNT(gn.line_id) as connected_lines
  FROM cim_vector.grid_bus gb
  LEFT JOIN grid_network gn ON gb.bus_id = gn.from_bus OR gb.bus_id = gn.to_bus
  WHERE gb.project_id = '{project_id}' AND gb.scenario_id = '{scenario_id}'
  GROUP BY gb.bus_id, gb.voltage_kv, gb.geometry
),
building_grid_proximity AS (
  SELECT b.building_id,
         bp.type,
         bp.area,
         bp.n_people,
         MIN(ST_Distance(b.building_geometry, bc.bus_geom)) as min_distance_to_bus,
         (SELECT voltage_kv FROM bus_connectivity 
          WHERE ST_DWithin(bus_geom, b.building_geometry, 1000) 
          ORDER BY ST_Distance(bus_geom, b.building_geometry) 
          LIMIT 1) as nearest_voltage_level
  FROM cim_vector.building b
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  CROSS JOIN bus_connectivity bc
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
  GROUP BY b.building_id, bp.type, bp.area, bp.n_people, b.building_geometry
)
SELECT type as building_type,
       nearest_voltage_level,
       COUNT(*) as building_count,
       AVG(area) as avg_area,
       SUM(n_people) as total_population,
       AVG(min_distance_to_bus) as avg_distance_to_grid
FROM building_grid_proximity
WHERE nearest_voltage_level IS NOT NULL
GROUP BY type, nearest_voltage_level
ORDER BY nearest_voltage_level DESC, building_count DESC;
""".strip())
    t.append(Template(
        id="CIM_C2_building_grid_proximity_analysis",
        complexity="C", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze building proximity to electrical grid infrastructure by voltage level",
        tags={"cim_building", "cim_grid", "network_analysis", "proximity", "advanced"} | get_frequency_tags(sql_c2),
        sql=sql_c2))

    # PostGIS specific - 3D analysis with rasters
    if dialect == "postgis":
        sql_c3 = adapt(dialect, """
WITH building_raster_stats AS (
  SELECT b.building_id,
         bp.type,
         bp.height as declared_height,
         ST_Value(dtm.rast, ST_Centroid(b.building_geometry)) as ground_elevation,
         ST_Value(dsm.rast, ST_Centroid(b.building_geometry)) as surface_elevation,
         ST_Area(b.building_geometry) as footprint_area
  FROM cim_vector.building b
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  JOIN cim_raster.dtm_raster dtm ON ST_Intersects(dtm.rast, b.building_geometry)
  JOIN cim_raster.dsm_raster dsm ON ST_Intersects(dsm.rast, b.building_geometry)
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
)
SELECT type,
       COUNT(*) as building_count,
       AVG(declared_height) as avg_declared_height,
       AVG(surface_elevation - ground_elevation) as avg_raster_height,
       AVG(ground_elevation) as avg_ground_elevation,
       SUM(footprint_area) as total_footprint_area,
       AVG(ABS(declared_height - (surface_elevation - ground_elevation))) as avg_height_discrepancy
FROM building_raster_stats
WHERE ground_elevation IS NOT NULL AND surface_elevation IS NOT NULL
GROUP BY type
ORDER BY avg_height_discrepancy DESC;
""".strip())
        t.append(Template(
            id="CIM_C3_3d_raster_building_analysis",
            complexity="C", dialect=dialect,
            geom_applicability=["POLYGON", "MULTIPOLYGON"],
            natural_language_desc="3D analysis of buildings using DTM and DSM raster data",
            tags={"cim_building", "cim_raster", "3d_analysis", "elevation", "postgis_only"} | get_frequency_tags(sql_c3),
            sql=sql_c3))

    # Enhanced cross-table CIM templates for complex realistic operations
    if dialect == "postgis":
        # Building height calculation with DSM/DTM raster clipping
        sql_c4 = """
WITH building_raster_intersections AS (
  SELECT b.building_id, bp.type, bp.height as declared_height,
         ST_Intersection(b.building_geometry, dsm.rast) as dsm_clip,
         ST_Intersection(b.building_geometry, dtm.rast) as dtm_clip,
         ST_Area(b.building_geometry) as building_area
  FROM cim_vector.building b
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  JOIN cim_raster.dsm_raster dsm ON ST_Intersects(b.building_geometry, dsm.rast)
  JOIN cim_raster.dtm_raster dtm ON ST_Intersects(b.building_geometry, dtm.rast)
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
    AND bp.type = '{building_type}'
),
height_calculations AS (
  SELECT building_id, type, declared_height,
         (ST_SummaryStats(dsm_clip)).mean as avg_dsm_elevation,
         (ST_SummaryStats(dtm_clip)).mean as avg_dtm_elevation,
         building_area,
         (ST_SummaryStats(dsm_clip)).count as dsm_pixel_count,
         (ST_SummaryStats(dtm_clip)).count as dtm_pixel_count
  FROM building_raster_intersections
  WHERE dsm_clip IS NOT NULL AND dtm_clip IS NOT NULL
)
SELECT building_id, type, declared_height, building_area,
       ROUND(avg_dsm_elevation, 2) as surface_elevation,
       ROUND(avg_dtm_elevation, 2) as ground_elevation,
       ROUND((avg_dsm_elevation - avg_dtm_elevation), 2) as calculated_height,
       ROUND(ABS(declared_height - (avg_dsm_elevation - avg_dtm_elevation)), 2) as height_difference,
       dsm_pixel_count, dtm_pixel_count
FROM height_calculations
WHERE avg_dsm_elevation IS NOT NULL AND avg_dtm_elevation IS NOT NULL
  AND (avg_dsm_elevation - avg_dtm_elevation) > {min_height}
ORDER BY height_difference DESC;
""".strip()

        t.append(Template(
            id="CIM_C4_precise_building_height_raster",
            complexity="C", dialect=dialect,
            geom_applicability=["POLYGON", "MULTIPOLYGON"],
            natural_language_desc="Calculate precise building heights by clipping DSM and DTM rasters with building footprints and computing elevation differences",
            tags={"cim_building", "cim_raster", "raster_vector", "cross_table", "height_analysis", "dsm", "dtm", "postgis_only"} | get_frequency_tags(sql_c4),
            sql=sql_c4))

        # Comprehensive census-building-grid integration analysis
        sql_c5 = """
WITH building_census_overlay AS (
  SELECT b.building_id, bp.type, bp.height, bp.area, bp.n_people,
         c.SEZ2011, c.P1 as total_population, c.REGIONE, c.PROVINCIA, c.COMUNE,
         ST_Area(ST_Intersection(b.building_geometry, c.geometry)) / ST_Area(b.building_geometry) as coverage_ratio
  FROM cim_vector.building b
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  JOIN cim_census.census_geo c ON ST_Intersects(b.building_geometry, c.geometry)
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
    AND ST_Area(ST_Intersection(b.building_geometry, c.geometry)) / ST_Area(b.building_geometry) > 0.7
),
grid_proximity AS (
  SELECT bco.building_id, bco.type, bco.height, bco.area, bco.n_people,
         bco.REGIONE, bco.PROVINCIA, bco.total_population,
         gb.bus_id, gb.voltage_kv, gb.name as substation_name,
         ST_Distance(b.building_geometry, gb.geometry) as grid_distance,
         ROW_NUMBER() OVER (PARTITION BY bco.building_id ORDER BY ST_Distance(b.building_geometry, gb.geometry)) as proximity_rank
  FROM building_census_overlay bco
  JOIN cim_vector.building b ON bco.building_id = b.building_id
  JOIN cim_vector.grid_bus gb ON gb.project_id = '{project_id}' AND gb.scenario_id = '{scenario_id}'
  WHERE gb.in_service = true
    AND ST_DWithin(b.building_geometry, gb.geometry, {max_distance})
),
energy_analysis AS (
  SELECT gp.building_id, gp.type, gp.height, gp.area, gp.n_people,
         gp.REGIONE, gp.PROVINCIA, gp.total_population,
         gp.grid_distance, gp.voltage_kv, gp.substation_name,
         CASE 
           WHEN gp.type = 'industrial' THEN gp.area * 0.05  -- 50 W/sqm
           WHEN gp.type = 'commercial' THEN gp.area * 0.03  -- 30 W/sqm  
           WHEN gp.type = 'residential' THEN gp.n_people * 1.5  -- 1.5 kW per person
           ELSE gp.area * 0.02
         END as estimated_demand_kw,
         CASE
           WHEN gp.voltage_kv >= 10 THEN 'high_voltage'
           WHEN gp.voltage_kv >= 1 THEN 'medium_voltage'
           ELSE 'low_voltage'
         END as grid_level
  FROM grid_proximity gp
  WHERE gp.proximity_rank = 1
)
SELECT REGIONE, PROVINCIA, type, grid_level,
       COUNT(*) as building_count,
       ROUND(AVG(height), 1) as avg_height,
       ROUND(SUM(area), 0) as total_area,
       ROUND(SUM(estimated_demand_kw), 1) as total_demand_kw,
       ROUND(AVG(grid_distance), 0) as avg_grid_distance,
       ROUND(AVG(total_population), 0) as avg_census_population,
       ROUND(SUM(estimated_demand_kw) / COUNT(*), 2) as demand_per_building
FROM energy_analysis
GROUP BY REGIONE, PROVINCIA, type, grid_level
HAVING COUNT(*) >= {min_buildings}
ORDER BY total_demand_kw DESC;
""".strip()

        t.append(Template(
            id="CIM_C5_integrated_census_grid_analysis", 
            complexity="C", dialect=dialect,
            geom_applicability=["POLYGON", "POINT"],
            natural_language_desc="Comprehensive analysis integrating building properties, census demographics, and electrical grid infrastructure for energy demand assessment",
            tags={"cim_building", "cim_census", "cim_grid", "cross_table", "multi_schema", "energy_analysis", "infrastructure", "postgis_only"} | get_frequency_tags(sql_c5),
            sql=sql_c5))

    # Advanced cross-schema analysis for both dialects (simplified for SpatiaLite)
    sql_c6 = adapt(dialect, """
WITH project_buildings AS (
  SELECT b.building_id, bp.type, bp.height, bp.area, bp.n_people,
         ST_Centroid(b.building_geometry) as building_center,
         CASE 
           WHEN bp.type = 'residential' THEN 'housing'
           WHEN bp.type IN ('commercial', 'industrial') THEN 'economic'
           ELSE 'other'
         END as functional_category
  FROM cim_vector.building b
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
),
spatial_clustering AS (
  SELECT building_id, type, height, area, n_people, functional_category,
         ST_ClusterKMeans(building_center, {cluster_count}) OVER () as cluster_id
  FROM project_buildings
),
cluster_analysis AS (
  SELECT cluster_id, functional_category,
         COUNT(*) as building_count,
         ROUND(AVG(height), 1) as avg_height,
         ROUND(SUM(area), 0) as total_area,
         SUM(n_people) as total_residents,
         ROUND(AVG(area), 0) as avg_building_size
  FROM spatial_clustering
  GROUP BY cluster_id, functional_category
  HAVING COUNT(*) >= {min_cluster_size}
)
SELECT cluster_id, functional_category, building_count, avg_height,
       total_area, total_residents, avg_building_size,
       ROUND(total_residents::float / NULLIF(building_count, 0), 1) as people_per_building,
       ROUND(total_area::float / NULLIF(building_count, 0), 0) as area_per_building
FROM cluster_analysis
ORDER BY cluster_id, functional_category;
""".strip())

    t.append(Template(
        id="CIM_C6_multi_schema_clustering",
        complexity="C", dialect=dialect,
        geom_applicability=["POLYGON", "POINT"],
        natural_language_desc="Advanced spatial clustering analysis across building properties with functional categorization and demographic insights",
        tags={"cim_building", "cross_table", "clustering", "multi_schema", "functional_analysis", "demographics"} | get_frequency_tags(sql_c6),
        sql=sql_c6))

    # Enhanced Census-focused templates with comprehensive demographic analysis
    
    # A-level: Basic census operations
    sql_census_a1 = adapt(dialect, """
SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE,
       c.P1 as total_population,
       c.P2 as male_population,
       c.P3 as female_population,
       ROUND((c.P2::float / NULLIF(c.P1, 0)) * 100, 1) as male_percentage
FROM cim_census.census_geo c
WHERE c.REGIONE = '{region}'
  AND c.P1 >= {min_population}
ORDER BY c.P1 DESC
LIMIT {limit};
""".strip())
    t.append(Template(
        id="CIM_CENSUS_A1_population_by_gender",
        complexity="A", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze population distribution by gender in census areas for a specific region",
        tags={"cim_census", "demographics", "gender_analysis", "basic_stats"} | get_frequency_tags(sql_census_a1),
        sql=sql_census_a1))

    sql_census_a2 = adapt(dialect, """
SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA,
       c.P14 as children_under_5,
       c.P27 + c.P28 + c.P29 as elderly_65_plus,
       c.P1 as total_population,
       ROUND(((c.P14 + c.P27 + c.P28 + c.P29)::float / NULLIF(c.P1, 0)) * 100, 1) as dependency_ratio
FROM cim_census.census_geo c
WHERE c.PROVINCIA = '{province}'
  AND c.P1 > 0
ORDER BY dependency_ratio DESC
LIMIT {limit};
""".strip())
    t.append(Template(
        id="CIM_CENSUS_A2_age_dependency_ratio",
        complexity="A", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Calculate age dependency ratio (children under 5 + elderly 65+) for census areas in a province",
        tags={"cim_census", "demographics", "age_analysis", "dependency"} | get_frequency_tags(sql_census_a2),
        sql=sql_census_a2))

    sql_census_a3 = adapt(dialect, """
SELECT c.SEZ2011, c.COMUNE,
       c.P47 as university_graduates,
       c.P48 as high_school_graduates,
       c.P50 as elementary_only,
       c.P52 as illiterate,
       c.P46 as population_6_plus,
       ROUND((c.P47::float / NULLIF(c.P46, 0)) * 100, 1) as university_rate
FROM cim_census.census_geo c
WHERE c.REGIONE = '{region}'
  AND c.P46 >= {min_population}
ORDER BY university_rate DESC
LIMIT {limit};
""".strip())
    t.append(Template(
        id="CIM_CENSUS_A3_education_levels",
        complexity="A", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze education levels and university graduation rates in census areas",
        tags={"cim_census", "education", "literacy", "qualification"} | get_frequency_tags(sql_census_a3),
        sql=sql_census_a3))

    # B-level: Intermediate census analysis
    sql_census_b1 = adapt(dialect, """
WITH age_groups AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE,
         c.P14 + c.P15 + c.P16 as youth_0_14,
         c.P17 + c.P18 + c.P19 + c.P20 + c.P21 + c.P22 + c.P23 + c.P24 + c.P25 + c.P26 as adults_15_64,
         c.P27 + c.P28 + c.P29 as elderly_65_plus,
         c.P1 as total_population
  FROM cim_census.census_geo c
  WHERE c.REGIONE = '{region}' AND c.P1 > 0
),
demographic_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         youth_0_14, adults_15_64, elderly_65_plus, total_population,
         ROUND((youth_0_14::float / NULLIF(total_population, 0)) * 100, 1) as youth_percentage,
         ROUND((elderly_65_plus::float / NULLIF(total_population, 0)) * 100, 1) as aging_index,
         ROUND(((youth_0_14 + elderly_65_plus)::float / NULLIF(adults_15_64, 0)) * 100, 1) as dependency_burden
  FROM age_groups
)
SELECT PROVINCIA, 
       COUNT(*) as census_sections,
       ROUND(AVG(youth_percentage), 1) as avg_youth_pct,
       ROUND(AVG(aging_index), 1) as avg_aging_index,
       ROUND(AVG(dependency_burden), 1) as avg_dependency_burden,
       SUM(total_population) as total_provincial_population
FROM demographic_indicators
GROUP BY PROVINCIA
ORDER BY avg_aging_index DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_B1_demographic_pyramid_analysis",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze demographic pyramid and aging patterns across provinces with dependency burden calculations",
        tags={"cim_census", "demographics", "aging", "statistical_analysis", "aggregation"} | get_frequency_tags(sql_census_b1),
        sql=sql_census_b1))

    sql_census_b2 = adapt(dialect, """
WITH employment_stats AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE,
         c.P60 as labor_force_total,
         c.P61 as employed_total,
         c.P62 as unemployed_total,
         c.P128 as not_in_labor_force,
         c.P130 as housewives,
         c.P131 as students,
         c.P139 as income_earners,
         c.P1 as total_population
  FROM cim_census.census_geo c
  WHERE c.PROVINCIA = '{province}' AND c.P60 > 0
),
employment_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         ROUND((unemployed_total::float / NULLIF(labor_force_total, 0)) * 100, 1) as unemployment_rate,
         ROUND((employed_total::float / NULLIF(total_population, 0)) * 100, 1) as employment_ratio,
         ROUND((labor_force_total::float / NULLIF(total_population, 0)) * 100, 1) as participation_rate,
         ROUND((income_earners::float / NULLIF(total_population, 0)) * 100, 1) as income_earner_rate
  FROM employment_stats
  WHERE labor_force_total > 0
)
SELECT COMUNE,
       COUNT(*) as census_areas,
       ROUND(AVG(unemployment_rate), 1) as avg_unemployment_rate,
       ROUND(AVG(employment_ratio), 1) as avg_employment_ratio,
       ROUND(AVG(participation_rate), 1) as avg_participation_rate,
       ROUND(AVG(income_earner_rate), 1) as avg_income_earner_rate
FROM employment_indicators
GROUP BY COMUNE
HAVING COUNT(*) >= {min_areas}
ORDER BY avg_unemployment_rate DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_B2_employment_labor_analysis",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Comprehensive employment and labor force analysis with unemployment rates and participation indicators",
        tags={"cim_census", "employment", "labor_force", "economic_indicators"} | get_frequency_tags(sql_census_b2),
        sql=sql_census_b2))

    sql_census_b3 = adapt(dialect, """
WITH housing_analysis AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE,
         c.A2 as occupied_homes,
         c.A6 as empty_homes,
         c.A46 as rented_families,
         c.A47 as owned_families,
         c.A44 as total_housing_surface,
         c.PF1 as total_families,
         c.PF2 as total_family_members
  FROM cim_census.census_geo c
  WHERE c.REGIONE = '{region}' AND c.A2 > 0 AND c.PF1 > 0
),
housing_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         ROUND((empty_homes::float / NULLIF((occupied_homes + empty_homes), 0)) * 100, 1) as vacancy_rate,
         ROUND((rented_families::float / NULLIF(total_families, 0)) * 100, 1) as rental_rate,
         ROUND((total_housing_surface::float / NULLIF(occupied_homes, 0)), 1) as avg_home_size_sqm,
         ROUND((total_family_members::float / NULLIF(total_families, 0)), 1) as avg_family_size,
         ROUND((total_housing_surface::float / NULLIF(total_family_members, 0)), 1) as sqm_per_person
  FROM housing_analysis
)
SELECT PROVINCIA,
       COUNT(*) as census_sections,
       ROUND(AVG(vacancy_rate), 1) as avg_vacancy_rate,
       ROUND(AVG(rental_rate), 1) as avg_rental_rate,
       ROUND(AVG(avg_home_size_sqm), 0) as avg_home_size,
       ROUND(AVG(avg_family_size), 1) as avg_family_size,
       ROUND(AVG(sqm_per_person), 1) as avg_sqm_per_person
FROM housing_indicators
GROUP BY PROVINCIA
ORDER BY avg_vacancy_rate DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_B3_housing_characteristics",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze housing characteristics including vacancy rates, rental patterns, and living space per person",
        tags={"cim_census", "housing", "vacancy", "rental_market", "living_space"} | get_frequency_tags(sql_census_b3),
        sql=sql_census_b3))

    # C-level: Advanced census analysis with spatial integration
    sql_census_c1 = adapt(dialect, """
WITH census_demographics AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE, c.geometry,
         c.P1 as total_population,
         c.ST1 as foreign_population,
         c.ST9 as european_foreigners,
         c.ST10 as african_foreigners,
         c.ST11 as american_foreigners,
         c.ST12 as asian_foreigners,
         ROUND((c.ST1::float / NULLIF(c.P1, 0)) * 100, 1) as foreign_percentage,
         c.E1 as total_buildings,
         c.E3 as residential_buildings,
         c.E8 + c.E9 as pre_1945_buildings,
         c.E15 + c.E16 as post_2000_buildings
  FROM cim_census.census_geo c
  WHERE c.REGIONE = '{region}' AND c.P1 >= {min_population}
),
spatial_clustering AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         total_population, foreign_population, foreign_percentage,
         total_buildings, residential_buildings,
         ST_ClusterDBSCAN(ST_Centroid(geometry), eps := {cluster_distance}, minpoints := {min_points}) 
         OVER (PARTITION BY PROVINCIA) as cluster_id
  FROM census_demographics
),
cluster_analysis AS (
  SELECT cluster_id, PROVINCIA,
         COUNT(*) as census_areas_in_cluster,
         SUM(total_population) as cluster_population,
         ROUND(AVG(foreign_percentage), 1) as avg_foreign_pct,
         SUM(total_buildings) as total_buildings_cluster,
         ROUND(AVG(foreign_percentage)) as diversity_index,
         ST_ConvexHull(ST_Collect(ST_Centroid(cd.geometry))) as cluster_boundary
  FROM spatial_clustering sc
  JOIN census_demographics cd ON sc.SEZ2011 = cd.SEZ2011
  WHERE cluster_id IS NOT NULL
  GROUP BY cluster_id, PROVINCIA
  HAVING COUNT(*) >= {min_cluster_size}
)
SELECT cluster_id, PROVINCIA, census_areas_in_cluster, cluster_population,
       avg_foreign_pct, total_buildings_cluster,
       ST_Area(cluster_boundary) as cluster_area_sqm,
       ROUND((cluster_population::float / (ST_Area(cluster_boundary) / 10000)), 1) as population_density_per_hectare
FROM cluster_analysis
ORDER BY population_density_per_hectare DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_C1_spatial_diversity_clustering",
        complexity="C", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Advanced spatial clustering analysis of demographic diversity and foreign population distribution patterns",
        tags={"cim_census", "clustering", "diversity", "foreign_population", "spatial_analysis", "density"} | get_frequency_tags(sql_census_c1),
        sql=sql_census_c1))

    sql_census_c2 = adapt(dialect, """
WITH building_age_analysis AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE, c.geometry,
         c.E1 as total_buildings,
         c.E8 as pre_1919_buildings,
         c.E9 as buildings_1919_1945,
         c.E10 as buildings_1946_1960,
         c.E11 as buildings_1961_1970,
         c.E12 as buildings_1971_1980,
         c.E13 as buildings_1981_1990,
         c.E14 as buildings_1991_2000,
         c.E15 as buildings_2001_2005,
         c.E16 as post_2005_buildings,
         c.E28 as excellent_condition,
         c.E29 as good_condition,
         c.E30 as mediocre_condition,
         c.E31 as poor_condition,
         c.P1 as total_population
  FROM cim_census.census_geo c
  WHERE c.PROVINCIA = '{province}' AND c.E1 > 0
),
age_quality_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         total_buildings, total_population,
         ROUND(((pre_1919_buildings + buildings_1919_1945)::float / NULLIF(total_buildings, 0)) * 100, 1) as historical_building_pct,
         ROUND(((buildings_2001_2005 + post_2005_buildings)::float / NULLIF(total_buildings, 0)) * 100, 1) as modern_building_pct,
         ROUND(((excellent_condition + good_condition)::float / NULLIF(total_buildings, 0)) * 100, 1) as good_quality_pct,
         ROUND((poor_condition::float / NULLIF(total_buildings, 0)) * 100, 1) as deteriorated_pct,
         ROUND((total_population::float / NULLIF(total_buildings, 0)), 1) as people_per_building
  FROM building_age_analysis
  WHERE total_buildings > 0
),
renovation_priority AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         historical_building_pct, modern_building_pct, good_quality_pct, deteriorated_pct,
         people_per_building,
         CASE 
           WHEN deteriorated_pct > 20 AND historical_building_pct > 30 THEN 'URGENT_HERITAGE_RENOVATION'
           WHEN deteriorated_pct > 15 THEN 'HIGH_PRIORITY_RENOVATION'
           WHEN good_quality_pct > 80 AND modern_building_pct > 50 THEN 'WELL_MAINTAINED_MODERN'
           WHEN historical_building_pct > 50 THEN 'HERITAGE_PRESERVATION'
           ELSE 'STANDARD_MAINTENANCE'
         END as renovation_category
  FROM age_quality_indicators
)
SELECT renovation_category,
       COUNT(*) as areas_count,
       ROUND(AVG(historical_building_pct), 1) as avg_historical_pct,
       ROUND(AVG(modern_building_pct), 1) as avg_modern_pct,
       ROUND(AVG(good_quality_pct), 1) as avg_quality_pct,
       ROUND(AVG(deteriorated_pct), 1) as avg_deterioration_pct,
       ROUND(AVG(people_per_building), 1) as avg_occupancy_density
FROM renovation_priority
GROUP BY renovation_category
ORDER BY areas_count DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_C2_building_heritage_renovation_analysis",
        complexity="C", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Comprehensive building age and condition analysis for heritage preservation and renovation priority assessment",
        tags={"cim_census", "building_analysis", "heritage", "renovation", "urban_planning", "condition_assessment"} | get_frequency_tags(sql_census_c2),
        sql=sql_census_c2))

    # Cross-schema census-building integration templates
    if dialect == "postgis":
        sql_census_c3 = """
WITH census_building_overlay AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE,
         c.P1 as census_population,
         c.P47 as university_graduates,
         c.P61 as employed_residents,
         c.A47 as owned_families,
         c.PF1 as total_families,
         c.E3 as residential_buildings_census,
         b.building_id, bp.type, bp.height, bp.area, bp.n_people,
         ST_Area(ST_Intersection(b.building_geometry, c.geometry)) / ST_Area(b.building_geometry) as overlap_ratio
  FROM cim_census.census_geo c
  JOIN cim_vector.building b ON ST_Intersects(b.building_geometry, c.geometry)
  JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
    AND c.REGIONE = '{region}'
    AND ST_Area(ST_Intersection(b.building_geometry, c.geometry)) / ST_Area(b.building_geometry) > 0.6
),
socioeconomic_building_profile AS (
  SELECT cbo.SEZ2011, cbo.REGIONE, cbo.PROVINCIA, cbo.COMUNE,
         cbo.building_id, cbo.type, cbo.height, cbo.area, cbo.n_people,
         cbo.census_population, cbo.university_graduates, cbo.employed_residents,
         ROUND((cbo.university_graduates::float / NULLIF(cbo.census_population, 0)) * 100, 1) as education_index,
         ROUND((cbo.employed_residents::float / NULLIF(cbo.census_population, 0)) * 100, 1) as employment_index,
         ROUND((cbo.owned_families::float / NULLIF(cbo.total_families, 0)) * 100, 1) as ownership_rate,
         CASE 
           WHEN cbo.type = 'residential' AND (cbo.university_graduates::float / NULLIF(cbo.census_population, 0)) > 0.25 THEN 'HIGH_EDUCATION_RESIDENTIAL'
           WHEN cbo.type = 'commercial' AND (cbo.employed_residents::float / NULLIF(cbo.census_population, 0)) > 0.70 THEN 'ACTIVE_COMMERCIAL_ZONE'
           WHEN cbo.type = 'industrial' AND (cbo.employed_residents::float / NULLIF(cbo.census_population, 0)) > 0.65 THEN 'INDUSTRIAL_EMPLOYMENT_HUB'
           ELSE 'STANDARD_MIXED_USE'
         END as socioeconomic_profile
  FROM census_building_overlay cbo
)
SELECT socioeconomic_profile, PROVINCIA,
       COUNT(DISTINCT building_id) as buildings_count,
       COUNT(DISTINCT SEZ2011) as census_areas_count,
       ROUND(AVG(education_index), 1) as avg_education_index,
       ROUND(AVG(employment_index), 1) as avg_employment_index,
       ROUND(AVG(ownership_rate), 1) as avg_ownership_rate,
       ROUND(AVG(height), 1) as avg_building_height,
       ROUND(SUM(area), 0) as total_building_area,
       SUM(n_people) as total_building_residents
FROM socioeconomic_building_profile
GROUP BY socioeconomic_profile, PROVINCIA
HAVING COUNT(DISTINCT building_id) >= {min_buildings}
ORDER BY avg_education_index DESC;
""".strip()

        t.append(Template(
            id="CIM_CENSUS_C3_socioeconomic_building_integration",
            complexity="C", dialect=dialect,
            geom_applicability=["POLYGON", "POINT"],
            natural_language_desc="Integrate census socioeconomic data with building properties for comprehensive urban profiling and development analysis",
            tags={"cim_census", "cim_building", "cross_schema", "socioeconomic", "urban_profiling", "education", "employment", "postgis_only"} | get_frequency_tags(sql_census_c3),
            sql=sql_census_c3))

        sql_census_c4 = """
WITH demographic_density_analysis AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE, c.geometry,
         c.P1 as total_population,
         c.P14 + c.P15 + c.P16 as children_0_14,
         c.P27 + c.P28 + c.P29 as elderly_65_plus,
         c.PF3 as single_person_families,
         c.PF8 as large_families_6plus,
         c.A2 as occupied_homes,
         ST_Area(c.geometry) as census_area_sqm
  FROM cim_census.census_geo c
  WHERE c.REGIONE = '{region}' AND c.P1 >= {min_population}
),
building_density_overlay AS (
  SELECT dda.SEZ2011, dda.REGIONE, dda.PROVINCIA, dda.COMUNE,
         dda.total_population, dda.children_0_14, dda.elderly_65_plus,
         dda.single_person_families, dda.large_families_6plus, dda.occupied_homes,
         dda.census_area_sqm,
         COUNT(b.building_id) as buildings_in_area,
         SUM(bp.area) as total_building_footprint,
         AVG(bp.height) as avg_building_height,
         SUM(bp.n_people) as building_residents
  FROM demographic_density_analysis dda
  LEFT JOIN cim_vector.building b ON ST_Within(ST_Centroid(b.building_geometry), dda.geometry)
  LEFT JOIN cim_vector.building_properties bp ON b.building_id = bp.building_id
  WHERE bp.project_id = '{project_id}' AND bp.scenario_id = '{scenario_id}'
  GROUP BY dda.SEZ2011, dda.REGIONE, dda.PROVINCIA, dda.COMUNE,
           dda.total_population, dda.children_0_14, dda.elderly_65_plus,
           dda.single_person_families, dda.large_families_6plus, dda.occupied_homes,
           dda.census_area_sqm
),
density_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         total_population, buildings_in_area, avg_building_height,
         ROUND((total_population::float / (census_area_sqm / 10000)), 1) as population_density_per_hectare,
         ROUND((total_building_footprint::float / census_area_sqm) * 100, 1) as building_coverage_pct,
         ROUND((children_0_14::float / NULLIF(total_population, 0)) * 100, 1) as child_ratio,
         ROUND((elderly_65_plus::float / NULLIF(total_population, 0)) * 100, 1) as elderly_ratio,
         ROUND((single_person_families::float / NULLIF(occupied_homes, 0)) * 100, 1) as single_household_pct,
         CASE 
           WHEN buildings_in_area > 0 THEN ROUND((building_residents::float / NULLIF(buildings_in_area, 0)), 1)
           ELSE 0
         END as avg_residents_per_building
  FROM building_density_overlay
  WHERE census_area_sqm > 0
),
urban_morphology_classification AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         population_density_per_hectare, building_coverage_pct, child_ratio, elderly_ratio,
         single_household_pct, avg_residents_per_building,
         CASE 
           WHEN population_density_per_hectare > 150 AND building_coverage_pct > 30 THEN 'DENSE_URBAN_CORE'
           WHEN population_density_per_hectare > 100 AND elderly_ratio > 25 THEN 'AGING_DENSE_NEIGHBORHOOD'
           WHEN population_density_per_hectare > 80 AND child_ratio > 20 THEN 'FAMILY_ORIENTED_DISTRICT'
           WHEN population_density_per_hectare < 50 AND single_household_pct > 40 THEN 'SUBURBAN_SINGLES'
           WHEN building_coverage_pct < 15 AND population_density_per_hectare < 30 THEN 'RURAL_SPARSE'
           ELSE 'MIXED_RESIDENTIAL'
         END as urban_morphology_type
  FROM density_indicators
)
SELECT urban_morphology_type, PROVINCIA,
       COUNT(*) as areas_count,
       ROUND(AVG(population_density_per_hectare), 1) as avg_pop_density,
       ROUND(AVG(building_coverage_pct), 1) as avg_building_coverage,
       ROUND(AVG(child_ratio), 1) as avg_child_ratio,
       ROUND(AVG(elderly_ratio), 1) as avg_elderly_ratio,
       ROUND(AVG(single_household_pct), 1) as avg_single_households,
       ROUND(AVG(avg_residents_per_building), 1) as avg_residents_per_bldg
FROM urban_morphology_classification
GROUP BY urban_morphology_type, PROVINCIA
HAVING COUNT(*) >= {min_areas}
ORDER BY avg_pop_density DESC;
""".strip()

        t.append(Template(
            id="CIM_CENSUS_C4_urban_morphology_classification",
            complexity="C", dialect=dialect,
            geom_applicability=["POLYGON", "POINT"],
            natural_language_desc="Advanced urban morphology classification combining census demographics with building density for urban planning insights",
            tags={"cim_census", "cim_building", "urban_morphology", "density_analysis", "family_structure", "aging", "cross_schema", "postgis_only"} | get_frequency_tags(sql_census_c4),
            sql=sql_census_c4))

    # Additional comprehensive census templates for full coverage
    
    # Marital status and family structure analysis
    sql_census_a4 = adapt(dialect, """
SELECT c.SEZ2011, c.COMUNE,
       c.P4 as singles,
       c.P5 as married_defacto,
       c.P6 as legally_separated,
       c.P7 as widowed,
       c.P8 as divorced,
       c.P1 as total_population,
       ROUND((c.P4::float / NULLIF(c.P1, 0)) * 100, 1) as singles_percentage,
       ROUND(((c.P6 + c.P8)::float / NULLIF(c.P1, 0)) * 100, 1) as dissolved_marriages_pct
FROM cim_census.census_geo c
WHERE c.PROVINCIA = '{province}'
  AND c.P1 >= {min_population}
ORDER BY singles_percentage DESC
LIMIT {limit};
""".strip())
    t.append(Template(
        id="CIM_CENSUS_A4_marital_status_analysis",
        complexity="A", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze marital status distribution and family dissolution rates in census areas",
        tags={"cim_census", "marital_status", "family_structure", "social_demographics"} | get_frequency_tags(sql_census_a4),
        sql=sql_census_a4))

    # Family size composition analysis
    sql_census_a5 = adapt(dialect, """
SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA,
       c.PF1 as total_families,
       c.PF3 as single_person_families,
       c.PF4 as two_person_families,
       c.PF5 as three_person_families,
       c.PF6 as four_person_families,
       c.PF8 as large_families_6plus,
       ROUND((c.PF3::float / NULLIF(c.PF1, 0)) * 100, 1) as single_household_pct,
       ROUND((c.PF8::float / NULLIF(c.PF1, 0)) * 100, 1) as large_family_pct
FROM cim_census.census_geo c
WHERE c.REGIONE = '{region}'
  AND c.PF1 > 0
ORDER BY large_family_pct DESC
LIMIT {limit};
""".strip())
    t.append(Template(
        id="CIM_CENSUS_A5_family_composition",
        complexity="A", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze family size composition from single households to large families",
        tags={"cim_census", "family_size", "household_composition", "demographics"} | get_frequency_tags(sql_census_a5),
        sql=sql_census_a5))

    # Foreign population diversity analysis
    sql_census_b4 = adapt(dialect, """
WITH foreign_demographics AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE,
         c.P1 as total_population,
         c.ST1 as total_foreigners,
         c.ST9 as european_foreigners,
         c.ST10 as african_foreigners,
         c.ST11 as american_foreigners,
         c.ST12 as asian_foreigners,
         c.ST13 as oceania_foreigners,
         c.ST14 as stateless_persons
  FROM cim_census.census_geo c
  WHERE c.PROVINCIA = '{province}' AND c.ST1 > 0
),
diversity_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         total_population, total_foreigners,
         ROUND((total_foreigners::float / NULLIF(total_population, 0)) * 100, 1) as foreign_percentage,
         ROUND((european_foreigners::float / NULLIF(total_foreigners, 0)) * 100, 1) as european_pct,
         ROUND((african_foreigners::float / NULLIF(total_foreigners, 0)) * 100, 1) as african_pct,
         ROUND((asian_foreigners::float / NULLIF(total_foreigners, 0)) * 100, 1) as asian_pct,
         CASE 
           WHEN european_foreigners + african_foreigners + american_foreigners + asian_foreigners > 0 THEN
             ROUND(1.0 - (POWER(european_foreigners::float / NULLIF(total_foreigners, 0), 2) + 
                         POWER(african_foreigners::float / NULLIF(total_foreigners, 0), 2) + 
                         POWER(american_foreigners::float / NULLIF(total_foreigners, 0), 2) + 
                         POWER(asian_foreigners::float / NULLIF(total_foreigners, 0), 2)), 3)
           ELSE 0
         END as diversity_index
  FROM foreign_demographics
)
SELECT COMUNE,
       COUNT(*) as census_areas,
       ROUND(AVG(foreign_percentage), 1) as avg_foreign_pct,
       ROUND(AVG(diversity_index), 3) as avg_diversity_index,
       ROUND(AVG(european_pct), 1) as avg_european_pct,
       ROUND(AVG(african_pct), 1) as avg_african_pct,
       ROUND(AVG(asian_pct), 1) as avg_asian_pct,
       SUM(total_foreigners) as total_foreign_population
FROM diversity_indicators
GROUP BY COMUNE
HAVING COUNT(*) >= {min_areas}
ORDER BY avg_diversity_index DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_B4_foreign_population_diversity",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Comprehensive analysis of foreign population diversity with continental origin distribution and diversity indices",
        tags={"cim_census", "foreign_population", "diversity", "multiculturalism", "continental_analysis"} | get_frequency_tags(sql_census_b4),
        sql=sql_census_b4))

    # Education-Employment correlation analysis
    sql_census_b5 = adapt(dialect, """
WITH education_employment AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE,
         c.P46 as population_6_plus,
         c.P47 as university_graduates,
         c.P48 as high_school_graduates,
         c.P49 as middle_school,
         c.P50 as elementary_only,
         c.P52 as illiterate,
         c.P60 as labor_force,
         c.P61 as employed,
         c.P62 as unemployed,
         c.P139 as income_earners
  FROM cim_census.census_geo c
  WHERE c.REGIONE = '{region}' AND c.P46 > 0 AND c.P60 > 0
),
education_employment_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         ROUND((university_graduates::float / NULLIF(population_6_plus, 0)) * 100, 1) as university_rate,
         ROUND((high_school_graduates::float / NULLIF(population_6_plus, 0)) * 100, 1) as high_school_rate,
         ROUND((illiterate::float / NULLIF(population_6_plus, 0)) * 100, 1) as illiteracy_rate,
         ROUND((unemployed::float / NULLIF(labor_force, 0)) * 100, 1) as unemployment_rate,
         ROUND((income_earners::float / NULLIF(employed, 0)) * 100, 1) as income_earner_ratio,
         university_graduates, employed, income_earners
  FROM education_employment
),
correlation_analysis AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         university_rate, unemployment_rate, income_earner_ratio,
         CASE 
           WHEN university_rate > 20 AND unemployment_rate < 5 THEN 'HIGH_EDUCATION_LOW_UNEMPLOYMENT'
           WHEN university_rate > 15 AND income_earner_ratio > 90 THEN 'EDUCATED_HIGH_INCOME'
           WHEN university_rate < 5 AND unemployment_rate > 15 THEN 'LOW_EDUCATION_HIGH_UNEMPLOYMENT'
           WHEN unemployment_rate > 20 THEN 'ECONOMIC_DISTRESS'
           ELSE 'AVERAGE_PROFILE'
         END as socioeconomic_profile
  FROM education_employment_indicators
)
SELECT socioeconomic_profile,
       COUNT(*) as areas_count,
       ROUND(AVG(university_rate), 1) as avg_university_rate,
       ROUND(AVG(unemployment_rate), 1) as avg_unemployment_rate,
       ROUND(AVG(income_earner_ratio), 1) as avg_income_ratio
FROM correlation_analysis
GROUP BY socioeconomic_profile
ORDER BY avg_university_rate DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_B5_education_employment_correlation",
        complexity="B", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Correlation analysis between education levels and employment outcomes for socioeconomic profiling",
        tags={"cim_census", "education", "employment", "correlation", "socioeconomic_profiling"} | get_frequency_tags(sql_census_b5),
        sql=sql_census_b5))

    # Building floor and interior distribution analysis
    sql_census_a6 = adapt(dialect, """
SELECT c.SEZ2011, c.PROVINCIA, c.COMUNE,
       c.E17 as single_floor_buildings,
       c.E18 as two_floor_buildings,
       c.E19 as three_floor_buildings,
       c.E20 as four_plus_floor_buildings,
       c.E21 as single_interior_buildings,
       c.E22 as two_interior_buildings,
       c.E26 as buildings_16plus_interiors,
       c.E1 as total_buildings,
       ROUND((c.E20::float / NULLIF(c.E1, 0)) * 100, 1) as high_rise_percentage,
       ROUND((c.E26::float / NULLIF(c.E1, 0)) * 100, 1) as large_complex_percentage
FROM cim_census.census_geo c
WHERE c.PROVINCIA = '{province}'
  AND c.E1 >= {min_buildings}
ORDER BY high_rise_percentage DESC
LIMIT {limit};
""".strip())
    t.append(Template(
        id="CIM_CENSUS_A6_building_structure_analysis",
        complexity="A", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Analyze building height distribution and interior complexity in census areas",
        tags={"cim_census", "building_structure", "floor_distribution", "interior_analysis"} | get_frequency_tags(sql_census_a6),
        sql=sql_census_a6))

    # Comprehensive demographic transition analysis
    sql_census_c5 = adapt(dialect, """
WITH demographic_transition AS (
  SELECT c.SEZ2011, c.REGIONE, c.PROVINCIA, c.COMUNE, c.geometry,
         -- Age structure indicators
         c.P14 + c.P15 + c.P16 as youth_0_14,
         c.P17 + c.P18 + c.P19 + c.P20 + c.P21 + c.P22 + c.P23 + c.P24 + c.P25 + c.P26 as working_age_15_64,
         c.P27 + c.P28 + c.P29 as elderly_65_plus,
         -- Family and social structure
         c.PF3 as single_households,
         c.PF8 as large_families,
         c.P4 as singles,
         c.P5 as married,
         -- Economic indicators
         c.P61 as employed,
         c.P62 as unemployed,
         c.P130 as housewives,
         c.P131 as students,
         -- Education
         c.P47 as university_graduates,
         c.P52 as illiterate,
         -- Foreign population
         c.ST1 as foreigners,
         c.P1 as total_population
  FROM cim_census.census_geo c
  WHERE c.REGIONE = '{region}' AND c.P1 >= {min_population}
),
transition_indicators AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         -- Demographic transition indicators
         ROUND((elderly_65_plus::float / NULLIF(youth_0_14, 0)), 2) as aging_ratio,
         ROUND((youth_0_14::float / NULLIF(working_age_15_64, 0)) * 100, 1) as youth_dependency,
         ROUND((elderly_65_plus::float / NULLIF(working_age_15_64, 0)) * 100, 1) as old_age_dependency,
         -- Social modernization indicators  
         ROUND((single_households::float / NULLIF(total_population, 0)) * 100, 1) as individualization_index,
         ROUND((university_graduates::float / NULLIF(total_population, 0)) * 100, 1) as education_modernization,
         ROUND((foreigners::float / NULLIF(total_population, 0)) * 100, 1) as cultural_diversity,
         -- Economic transition
         ROUND((unemployed::float / NULLIF(employed + unemployed, 0)) * 100, 1) as unemployment_rate,
         ROUND((housewives::float / NULLIF(total_population, 0)) * 100, 1) as traditional_gender_roles,
         total_population
  FROM demographic_transition
),
transition_classification AS (
  SELECT SEZ2011, REGIONE, PROVINCIA, COMUNE,
         aging_ratio, youth_dependency, old_age_dependency,
         individualization_index, education_modernization, cultural_diversity,
         unemployment_rate, traditional_gender_roles,
         CASE 
           WHEN aging_ratio > 1.5 AND individualization_index > 15 AND education_modernization > 10 THEN 'POST_TRANSITION_ADVANCED'
           WHEN aging_ratio > 1.0 AND education_modernization > 5 THEN 'LATE_TRANSITION'
           WHEN youth_dependency > 25 AND traditional_gender_roles > 10 THEN 'PRE_TRANSITION_TRADITIONAL'
           WHEN unemployment_rate > 15 AND cultural_diversity > 5 THEN 'TRANSITION_WITH_CHALLENGES'
           ELSE 'MID_TRANSITION'
         END as demographic_transition_stage
  FROM transition_indicators
)
SELECT demographic_transition_stage, PROVINCIA,
       COUNT(*) as areas_count,
       ROUND(AVG(aging_ratio), 2) as avg_aging_ratio,
       ROUND(AVG(individualization_index), 1) as avg_individualization,
       ROUND(AVG(education_modernization), 1) as avg_education_mod,
       ROUND(AVG(cultural_diversity), 1) as avg_diversity,
       ROUND(AVG(unemployment_rate), 1) as avg_unemployment,
       ROUND(AVG(traditional_gender_roles), 1) as avg_traditional_roles
FROM transition_classification
GROUP BY demographic_transition_stage, PROVINCIA
HAVING COUNT(*) >= {min_areas}
ORDER BY avg_aging_ratio DESC;
""".strip())
    t.append(Template(
        id="CIM_CENSUS_C5_demographic_transition_analysis",
        complexity="C", dialect=dialect,
        geom_applicability=["POLYGON", "MULTIPOLYGON"],
        natural_language_desc="Comprehensive demographic transition analysis combining aging, modernization, and social change indicators",
        tags={"cim_census", "demographic_transition", "modernization", "aging", "social_change", "economic_transition"} | get_frequency_tags(sql_census_c5),
        sql=sql_census_c5))

    return t

def generate_cim_wizard_pairs(num_variations: int = 100) -> List[SqlPair]:
    """Generate CIM Wizard SQL pairs with parameter variations"""
    base_templates_postgis = cim_wizard_templates("postgis")
    base_templates_spatialite = cim_wizard_templates("spatialite")
    
    pairs = []
    
    # Create pairs for common templates
    common_template_ids = set(t.id for t in base_templates_postgis) & set(t.id for t in base_templates_spatialite)
    
    for template_id in common_template_ids:
        postgis_template = next(t for t in base_templates_postgis if t.id == template_id)
        spatialite_template = next(t for t in base_templates_spatialite if t.id == template_id)
        
        # Generate multiple variations
        for i in range(num_variations):
            values = generate_realistic_values()
            
            # Apply parameter substitution
            postgis_sql = postgis_template.sql.format(**values)
            spatialite_sql = spatialite_template.sql.format(**values)
            
            # Create natural language with specific values
            enhanced_desc = f"{postgis_template.natural_language_desc} (Project: {values['project_id']}, Scenario: {values['scenario_id']})"
            
            merged_tags = postgis_template.tags | {"cim_wizard", "realistic_params"} | get_frequency_tags(postgis_sql)
            pair = SqlPair(
                template_id=f"{template_id}_var_{i+1}",
                complexity=postgis_template.complexity,
                geom_types=list(postgis_template.geom_applicability),
                postgis_sql=postgis_sql,
                spatialite_sql=spatialite_sql,
                natural_language_desc=enhanced_desc,
                tags=merged_tags,
                usage_index=determine_usage_index(postgis_sql, merged_tags),
                evidence=extract_evidence(postgis_sql, f"{template_id}_var_{i+1}", merged_tags)
            )
            pairs.append(pair)
    
    # Add PostGIS-only templates
    postgis_only = set(t.id for t in base_templates_postgis) - common_template_ids
    for template_id in postgis_only:
        template = next(t for t in base_templates_postgis if t.id == template_id)
        
        for i in range(num_variations):
            values = generate_realistic_values()
            postgis_sql = template.sql.format(**values)
            enhanced_desc = f"{template.natural_language_desc} (Project: {values['project_id']}, Scenario: {values['scenario_id']})"
            
            merged_tags = template.tags | {"cim_wizard", "realistic_params", "postgis_only"} | get_frequency_tags(postgis_sql)
            pair = SqlPair(
                template_id=f"{template_id}_var_{i+1}",
                complexity=template.complexity,
                geom_types=list(template.geom_applicability),
                postgis_sql=postgis_sql,
                spatialite_sql="-- Not available in SpatiaLite (raster operations)",
                natural_language_desc=enhanced_desc,
                tags=merged_tags,
                usage_index=determine_usage_index(postgis_sql, merged_tags),
                evidence=extract_evidence(postgis_sql, f"{template_id}_var_{i+1}", merged_tags)
            )
            pairs.append(pair)
    
    return pairs

def generate_comprehensive_cim_dataset(
    base_variations: int = 100,
    include_original_templates: bool = True
) -> List[SqlPair]:
    """Generate comprehensive dataset combining original and CIM Wizard templates"""
    
    pairs = []
    
    # Add original rule-based templates if requested
    if include_original_templates:
        from rule_based_ssql_generator import generate_sql_pairs
        original_pairs = generate_sql_pairs()
        pairs.extend(original_pairs)
    
    # Add CIM Wizard specific templates
    cim_pairs = generate_cim_wizard_pairs(base_variations)
    pairs.extend(cim_pairs)
    
    return pairs

if __name__ == "__main__":
    print("="*80)
    print("CIM Wizard Enhanced Spatial SQL Generator")
    print("="*80)
    
    # Generate comprehensive dataset
    print("Generating comprehensive dataset with CIM Wizard integration...")
    
    # Generate with different variation counts for demonstration
    small_dataset = generate_comprehensive_cim_dataset(base_variations=10)
    medium_dataset = generate_comprehensive_cim_dataset(base_variations=50)
    large_dataset = generate_comprehensive_cim_dataset(base_variations=200)
    
    print(f"\nDataset sizes:")
    print(f"  Small (10 variations):  {len(small_dataset):,} samples")
    print(f"  Medium (50 variations): {len(medium_dataset):,} samples")
    print(f"  Large (200 variations): {len(large_dataset):,} samples")
    
    # Show statistics for medium dataset
    print(f"\n" + "="*80)
    print("SAMPLE ANALYSIS (Medium Dataset)")
    print("="*80)
    
    stats = generate_statistics(medium_dataset)
    print(f"Total SQL pairs: {stats['total_pairs']:,}")
    print(f"  - Complexity A: {stats['by_complexity'].get('A', 0):,}")
    print(f"  - Complexity B: {stats['by_complexity'].get('B', 0):,}")
    print(f"  - Complexity C: {stats['by_complexity'].get('C', 0):,}")
    print(f"  - Both dialects: {stats['dialect_specific']['both_dialects']:,}")
    print(f"  - PostGIS only: {stats['dialect_specific']['postgis_only']:,}")
    print(f"  - SpatiaLite only: {stats['dialect_specific']['spatialite_only']:,}")
    
    # Count CIM-specific vs original templates
    cim_samples = len([p for p in medium_dataset if "cim_wizard" in p.tags])
    original_samples = len(medium_dataset) - cim_samples
    
    print(f"\nTemplate distribution:")
    print(f"  - Original spatial templates: {original_samples:,}")
    print(f"  - CIM Wizard specific: {cim_samples:,}")
    
    # Show sample CIM Wizard SQL
    print(f"\n" + "="*80)
    print("SAMPLE CIM WIZARD SQL PAIR")
    print("="*80)
    
    cim_sample = next((p for p in medium_dataset if "cim_wizard" in p.tags), None)
    if cim_sample:
        print(f"Template ID: {cim_sample.template_id}")
        print(f"Complexity: {cim_sample.complexity}")
        print(f"Description: {cim_sample.natural_language_desc}")
        print(f"Tags: {', '.join(sorted(cim_sample.tags))}")
        print(f"\nPostGIS SQL:")
        print(cim_sample.postgis_sql)
        print(f"\nSpatiaLite SQL:")
        print(cim_sample.spatialite_sql[:200] + "..." if len(cim_sample.spatialite_sql) > 200 else cim_sample.spatialite_sql)
    
    # Export and save training datasets
    print(f"\n" + "="*80)
    print("SAVING CIM WIZARD TRAINING DATASETS")
    print("="*80)
    
    from rule_based_ssql_generator import save_training_dataset
    import os
    
    # Save all generated datasets
    datasets_to_save = [
        (small_dataset, "cim_wizard_small"),
        (medium_dataset, "cim_wizard_medium"), 
        (large_dataset, "cim_wizard_large")
    ]
    
    for dataset, prefix in datasets_to_save:
        print(f"\nSaving {prefix} dataset...")
        saved_files = save_training_dataset(dataset, prefix)
        
        print(f"  {prefix.upper()} dataset files:")
        for format_type, filepath in saved_files.items():
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024  # Size in KB
                print(f"    {format_type.upper()}: {filepath} ({file_size:.1f} KB)")
    
    # Show sample from medium dataset JSONL
    print(f"\n" + "="*80)
    print("SAMPLE JSONL CONTENT (Medium Dataset)")
    print("="*80)
    
    import glob
    from datetime import datetime
    
    medium_jsonl_file = f"training_datasets/cim_wizard_medium_{datetime.now().strftime('%Y%m%d')}_*.jsonl"
    
    # Find the most recent medium JSONL file
    pattern = "training_datasets/cim_wizard_medium_*.jsonl"
    jsonl_files = glob.glob(pattern)
    if jsonl_files:
        latest_jsonl = max(jsonl_files, key=os.path.getctime)
        
        with open(latest_jsonl, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"JSONL file: {latest_jsonl}")
        print(f"Total training examples: {len(lines)}")
        
        # Show first CIM Wizard example
        import json as json_module
        for line in lines:
            item = json_module.loads(line.strip())
            if "cim_wizard" in item.get('evidence', {}).get('template_source', ''):
                print(f"\nSample CIM Wizard Training Example:")
                print(f"  ID: {item['id']}")
                print(f"  Instruction: {item['instruction'][:80]}...")
                print(f"  PostGIS Output: {item['output_postgis'][:100]}...")
                print(f"  SpatiaLite Output: {item['output_spatialite'][:100]}...")
                print(f"  Complexity: {item['complexity']}")
                print(f"  Usage Index: {item['usage_index']}")
                print(f"  Evidence Schemas: {', '.join(item['evidence'].get('schemas', []))}")
                print(f"  Evidence Tables: {', '.join(item['evidence'].get('tables', [])[:3])}...")
                break
    
    print(f"\n[SUCCESS] READY FOR LLM FINE-TUNING!")
    print(f"[SUCCESS] JSONL files saved in training_datasets/ directory")
    print(f"[SUCCESS] Use medium dataset (~624 samples) for proof-of-concept")
    print(f"[SUCCESS] Use large dataset (~2,424 samples) for 7B model training")
    print(f"[SUCCESS] Scale to 10,000+ samples with: generate_comprehensive_cim_dataset(base_variations=1000)")
