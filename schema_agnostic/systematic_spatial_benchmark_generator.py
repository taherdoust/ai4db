"""
Systematic Spatial SQL Benchmark Generator
Generates a balanced dataset with all 4 constraints:
1. Schema-agnostic design
2. Enhanced natural language diversity 
3. Non-spatial attributes (3-type paradigm)
4. Balanced dataset composition (40/30/20/10 split)
"""

import itertools
import random
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Schema-agnostic table definitions with 3-type attributes
BENCHMARK_SCHEMA = {
    "table_point": {
        "geometry_type": "Point",
        "valid_functions": ["ST_Buffer", "ST_Centroid", "ST_Distance", "ST_DWithin", "ST_Within", "ST_Contains", "ST_Intersects", "ST_Touches", "ST_Disjoint", "ST_Equals", "ST_Covers", "ST_CoveredBy"]
    },
    "table_line": {
        "geometry_type": "LineString", 
        "valid_functions": ["ST_Length", "ST_Buffer", "ST_Centroid", "ST_Distance", "ST_DWithin", "ST_Within", "ST_Contains", "ST_Intersects", "ST_Touches", "ST_Disjoint", "ST_Equals", "ST_Covers", "ST_CoveredBy"]
    },
    "table_polygon": {
        "geometry_type": "Polygon",
        "valid_functions": ["ST_Area", "ST_Perimeter", "ST_Buffer", "ST_Centroid", "ST_Distance", "ST_DWithin", "ST_Within", "ST_Contains", "ST_Intersects", "ST_Touches", "ST_Disjoint", "ST_Equals", "ST_Covers", "ST_CoveredBy"]
    },
    "table_multipoint": {
        "geometry_type": "MultiPoint",
        "valid_functions": ["ST_Buffer", "ST_Centroid", "ST_NPoints", "ST_Distance", "ST_DWithin", "ST_Within", "ST_Contains", "ST_Intersects", "ST_Touches", "ST_Disjoint", "ST_Equals", "ST_Covers", "ST_CoveredBy"]
    },
    "table_multiline": {
        "geometry_type": "MultiLineString",
        "valid_functions": ["ST_Length", "ST_Buffer", "ST_Centroid", "ST_NPoints", "ST_Distance", "ST_DWithin", "ST_Within", "ST_Contains", "ST_Intersects", "ST_Touches", "ST_Disjoint", "ST_Equals", "ST_Covers", "ST_CoveredBy"]
    },
    "table_multipolygon": {
        "geometry_type": "MultiPolygon", 
        "valid_functions": ["ST_Area", "ST_Perimeter", "ST_Buffer", "ST_Centroid", "ST_NPoints", "ST_Distance", "ST_DWithin", "ST_Within", "ST_Contains", "ST_Intersects", "ST_Touches", "ST_Disjoint", "ST_Equals", "ST_Covers", "ST_CoveredBy"]
    }
}

# Three-type attribute system
ATTRIBUTE_SYSTEM = {
    "identification": {
        "id": {"type": "UUID", "description": "Primary key identifier"}
    },
    "spatial": {
        "geom": {"type": "GEOMETRY", "description": "Spatial geometry column"}
    },
    "non_spatial": {
        "attr_text": {
            "type": "VARCHAR(255)",
            "values": ["'type_a'", "'type_b'", "'type_c'", "'category_1'", "'category_2'", "'status_active'", "'status_inactive'"],
            "operators": ["=", "!=", "IN", "NOT IN"]
        },
        "attr_numeric": {
            "type": "FLOAT", 
            "values": [10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            "operators": ["=", "!=", ">", "<", ">=", "<=", "BETWEEN"]
        },
        "attr_boolean": {
            "type": "BOOLEAN",
            "values": [True, False],
            "operators": ["=", "!="]
        },
        "attr_date": {
            "type": "DATE",
            "values": ["'2020-01-01'", "'2021-01-01'", "'2022-01-01'", "'2023-01-01'"],
            "operators": ["=", "!=", ">", "<", ">=", "<="]
        }
    }
}

# Comprehensive natural language templates
NATURAL_LANGUAGE_TEMPLATES = {
    # Level 1: Simple spatial functions (40% of dataset)
    "level1_simple": {
        "unary_measurement": {
            "ST_Area": [
                "What is the area of {entity}?",
                "Calculate the area of {entity}",
                "How big is {entity}?",
                "Find the area value for {entity}",
                "Measure the area of {entity}",
                "What's the size of {entity}?",
                "Determine the area covered by {entity}",
                "Show the area measurement for {entity}"
            ],
            "ST_Length": [
                "What is the length of {entity}?",
                "Calculate the length of {entity}",
                "How long is {entity}?",
                "Find the length value for {entity}",
                "Measure the length of {entity}",
                "What's the distance along {entity}?",
                "Determine the total length of {entity}",
                "Show the length measurement for {entity}"
            ],
            "ST_Perimeter": [
                "What is the perimeter of {entity}?",
                "Calculate the perimeter of {entity}",
                "How long is the boundary of {entity}?",
                "Find the perimeter value for {entity}",
                "Measure the perimeter of {entity}",
                "What's the edge length of {entity}?",
                "Determine the boundary length of {entity}"
            ],
            "ST_NPoints": [
                "How many points are in {entity}?",
                "Count the points in {entity}",
                "What is the point count for {entity}?",
                "Find the number of points in {entity}",
                "Determine how many points {entity} contains"
            ]
        },
        "unary_transformation": {
            "ST_Centroid": [
                "Where is the center of {entity}?",
                "Find the centroid of {entity}",
                "What is the center point of {entity}?",
                "Locate the geometric center of {entity}",
                "Show the centroid location for {entity}",
                "Determine the central point of {entity}"
            ],
            "ST_Buffer": [
                "Create a {distance}m buffer around {entity}",
                "Generate a {distance} meter buffer zone for {entity}",
                "What is the {distance}m service area around {entity}?",
                "Find the {distance} meter catchment area for {entity}",
                "Show the {distance}m radius around {entity}",
                "Build a {distance} meter buffer surrounding {entity}"
            ]
        },
        "binary_relationships": {
            "ST_Within": [
                "Is {entity1} within {entity2}?",
                "Does {entity2} contain {entity1}?",
                "Is {entity1} inside {entity2}?",
                "Does {entity1} fall within {entity2}?",
                "Is {entity1} contained in {entity2}?",
                "Does {entity1} lie within {entity2}?"
            ],
            "ST_Contains": [
                "Does {entity1} contain {entity2}?",
                "Is {entity2} within {entity1}?",
                "Does {entity1} encompass {entity2}?",
                "Is {entity2} inside {entity1}?",
                "Does {entity1} include {entity2}?",
                "Is {entity2} contained in {entity1}?"
            ],
            "ST_Intersects": [
                "Do {entity1} and {entity2} intersect?",
                "Does {entity1} overlap with {entity2}?",
                "Is there overlap between {entity1} and {entity2}?",
                "Do {entity1} and {entity2} cross each other?",
                "Does {entity1} intersect {entity2}?",
                "Are {entity1} and {entity2} overlapping?"
            ],
            "ST_Distance": [
                "What is the distance between {entity1} and {entity2}?",
                "How far is {entity1} from {entity2}?",
                "Calculate the distance from {entity1} to {entity2}",
                "Find the separation between {entity1} and {entity2}",
                "Measure the distance between {entity1} and {entity2}",
                "What's the gap between {entity1} and {entity2}?"
            ],
            "ST_DWithin": [
                "Is {entity1} within {distance}m of {entity2}?",
                "Are {entity1} and {entity2} within {distance} meters?",
                "Is {entity1} close to {entity2} (within {distance}m)?",
                "Does {entity1} fall within {distance}m of {entity2}?",
                "Are {entity1} and {entity2} less than {distance}m apart?",
                "Is the distance between {entity1} and {entity2} under {distance}m?"
            ]
        }
    },
    
    # Level 2: Spatial + attributes (30% of dataset)
    "level2_attributes": {
        "attribute_filtering": [
            "Find {entities} where {condition}",
            "Show {entities} with {condition}",
            "List {entities} that have {condition}",
            "Which {entities} meet the condition {condition}?",
            "Display {entities} where {condition}",
            "Select {entities} that satisfy {condition}"
        ],
        "spatial_with_conditions": [
            "Find {entities1} where {condition} within {distance}m of {entity2}",
            "Show {entities1} with {condition} near {entity2} (within {distance}m)",
            "List {entities1} where {condition} and close to {entity2}",
            "Which {entities1} have {condition} and are within {distance}m of {entity2}?",
            "Display {entities1} that meet {condition} and are near {entity2}"
        ],
        "relationship_with_conditions": [
            "Find {entities1} where {condition} that intersect {entity2}",
            "Show {entities1} with {condition} contained in {entity2}", 
            "List {entities1} where {condition} that overlap {entity2}",
            "Which {entities1} have {condition} and are within {entity2}?"
        ]
    },
    
    # Level 3: Aggregations (20% of dataset)
    "level3_aggregation": {
        "count_aggregation": [
            "How many {entities1} are within each {entity2_type}?",
            "Count {entities1} inside each {entity2_type}",
            "Find the number of {entities1} contained in each {entity2_type}",
            "Calculate {entity1_type} density for each {entity2_type}",
            "Show {entity1_type} count per {entity2_type}"
        ],
        "conditional_aggregation": [
            "Count {entities1} where {condition} within each {entity2_type}",
            "How many {entities1} with {condition} are in each {entity2_type}?",
            "Find the number of {entities1} that have {condition} inside each {entity2_type}",
            "Calculate density of {entities1} with {condition} per {entity2_type}"
        ],
        "measurement_aggregation": [
            "What is the total area of {entities} in each {grouping_entity}?",
            "Calculate the average length of {entities} per {grouping_entity}",
            "Find the sum of {measurement} for {entities} grouped by {grouping_entity}",
            "Show the total {measurement} of {entities} within each {grouping_entity}"
        ]
    },
    
    # Level 4: Complex reasoning (10% of dataset)
    "level4_complex": {
        "negation": [
            "Find {entities1} that are NOT within {distance}m of any {entity2_type}",
            "Show {entities1} that don't have any {entity2_type} within {distance} meters",
            "List {entities1} with no {entity2_type}s nearby (within {distance}m)",
            "Which {entities1} are isolated from {entity2_type}s by more than {distance}m?",
            "Display {entities1} that are far from all {entity2_type}s"
        ],
        "nearest_neighbor": [
            "Find the closest {entity2_type} to each {entity1_type}",
            "What is the nearest {entity2_type} for each {entity1_type}?",
            "Show the closest {entity2_type} to every {entity1_type}",
            "List the nearest {entity2_type} for each {entity1_type}",
            "Identify the most proximate {entity2_type} to each {entity1_type}"
        ],
        "conditional_complex": [
            "Find {entities1} where {condition1} that are NOT near any {entity2_type} with {condition2}",
            "Show {entities1} with {condition1} that don't intersect {entities2} where {condition2}",
            "List {entities1} where {condition1} and no {entity2_type} with {condition2} within {distance}m"
        ]
    }
}

# SQL generation templates
SQL_GENERATION_TEMPLATES = {
    "level1_simple": {
        "unary_measurement": {
            "ST_Area": "SELECT id, ST_Area(geom::geography) AS area_m2 FROM {table} WHERE id = '{placeholder}';",
            "ST_Length": "SELECT id, ST_Length(geom::geography) AS length_m FROM {table} WHERE id = '{placeholder}';",
            "ST_Perimeter": "SELECT id, ST_Perimeter(geom::geography) AS perimeter_m FROM {table} WHERE id = '{placeholder}';",
            "ST_NPoints": "SELECT id, ST_NPoints(geom) AS point_count FROM {table} WHERE id = '{placeholder}';"
        },
        "unary_transformation": {
            "ST_Centroid": "SELECT id, ST_Centroid(geom) AS centroid FROM {table} WHERE id = '{placeholder}';",
            "ST_Buffer": "SELECT id, ST_Buffer(geom::geography, {distance})::geometry AS buffer_geom FROM {table} WHERE id = '{placeholder}';"
        },
        "binary_relationships": {
            "ST_Within": "SELECT t1.id FROM {table1} t1, {table2} t2 WHERE ST_Within(t1.geom, t2.geom) AND t1.id = '{placeholder1}' AND t2.id = '{placeholder2}';",
            "ST_Contains": "SELECT t1.id FROM {table1} t1, {table2} t2 WHERE ST_Contains(t1.geom, t2.geom) AND t1.id = '{placeholder1}' AND t2.id = '{placeholder2}';",
            "ST_Intersects": "SELECT t1.id FROM {table1} t1, {table2} t2 WHERE ST_Intersects(t1.geom, t2.geom) AND t1.id = '{placeholder1}' AND t2.id = '{placeholder2}';",
            "ST_Distance": "SELECT t1.id, t2.id, ST_Distance(t1.geom::geography, t2.geom::geography) AS distance_m FROM {table1} t1, {table2} t2 WHERE t1.id = '{placeholder1}' AND t2.id = '{placeholder2}';",
            "ST_DWithin": "SELECT t1.id FROM {table1} t1, {table2} t2 WHERE ST_DWithin(t1.geom::geography, t2.geom::geography, {distance}) AND t1.id = '{placeholder1}' AND t2.id = '{placeholder2}';"
        }
    },
    "level2_attributes": {
        "attribute_filtering": "SELECT id FROM {table} WHERE {condition};",
        "spatial_with_conditions": "SELECT t1.id FROM {table1} t1, {table2} t2 WHERE {condition} AND ST_DWithin(t1.geom::geography, t2.geom::geography, {distance}) AND t2.id = '{placeholder}';",
        "relationship_with_conditions": "SELECT t1.id FROM {table1} t1, {table2} t2 WHERE {condition} AND ST_{relationship}(t1.geom, t2.geom) AND t2.id = '{placeholder}';"
    },
    "level3_aggregation": {
        "count_aggregation": "SELECT t2.id, COUNT(t1.id) AS {table1}_count FROM {table2} t2 LEFT JOIN {table1} t1 ON ST_Within(t1.geom, t2.geom) GROUP BY t2.id;",
        "conditional_aggregation": "SELECT t2.id, COUNT(t1.id) AS filtered_count FROM {table2} t2 LEFT JOIN {table1} t1 ON ST_Within(t1.geom, t2.geom) AND {condition} GROUP BY t2.id;",
        "measurement_aggregation": "SELECT t2.id, SUM(ST_{measurement}(t1.geom::geography)) AS total_{measurement} FROM {table2} t2 LEFT JOIN {table1} t1 ON ST_Within(t1.geom, t2.geom) GROUP BY t2.id;"
    },
    "level4_complex": {
        "negation": "SELECT id FROM {table1} WHERE NOT EXISTS (SELECT 1 FROM {table2} WHERE ST_DWithin({table1}.geom::geography, {table2}.geom::geography, {distance}));",
        "nearest_neighbor": "SELECT DISTINCT ON (t1.id) t1.id, t2.id AS nearest_{table2}_id, ST_Distance(t1.geom::geography, t2.geom::geography) AS distance_m FROM {table1} t1 CROSS JOIN {table2} t2 ORDER BY t1.id, ST_Distance(t1.geom::geography, t2.geom::geography);",
        "conditional_complex": "SELECT t1.id FROM {table1} t1 WHERE {condition1} AND NOT EXISTS (SELECT 1 FROM {table2} t2 WHERE {condition2} AND ST_DWithin(t1.geom::geography, t2.geom::geography, {distance}));"
    }
}

class SystematicBenchmarkGenerator:
    def __init__(self, target_size: int = 5000):
        self.target_size = target_size
        self.distances = [50, 100, 200, 500, 1000, 2000]
        self.samples = []
        
        # Target distribution (balanced composition)
        self.target_distribution = {
            "level1_simple": int(target_size * 0.40),     # 40%
            "level2_attributes": int(target_size * 0.30), # 30%
            "level3_aggregation": int(target_size * 0.20), # 20%
            "level4_complex": int(target_size * 0.10)     # 10%
        }
        
        print(f"Target dataset size: {target_size}")
        print(f"Distribution: {self.target_distribution}")
    
    def generate_entity_reference(self, table: str, with_id: bool = True) -> str:
        """Generate natural entity references"""
        entity_type = table.replace("table_", "")
        if with_id:
            return f"{entity_type} '<id_{random.randint(1000, 9999)}>'"
        return entity_type
    
    def generate_condition(self, attr_name: str) -> Tuple[str, str]:
        """Generate attribute conditions"""
        attr_info = ATTRIBUTE_SYSTEM["non_spatial"][attr_name]
        operator = random.choice(attr_info["operators"])
        value = random.choice(attr_info["values"])
        
        if operator == "BETWEEN" and attr_name == "attr_numeric":
            val1, val2 = sorted(random.sample(attr_info["values"], 2))
            condition = f"{attr_name} BETWEEN {val1} AND {val2}"
            description = f"{attr_name} between {val1} and {val2}"
        elif operator in ["IN", "NOT IN"] and attr_name == "attr_text":
            values = random.sample(attr_info["values"], min(3, len(attr_info["values"])))
            condition = f"{attr_name} {operator} ({', '.join(values)})"
            description = f"{attr_name} {operator.lower()} {', '.join(values)}"
        else:
            condition = f"{attr_name} {operator} {value}"
            description = f"{attr_name} {operator} {value}"
        
        return condition, description
    
    def generate_level1_simple(self) -> List[Dict]:
        """Generate Level 1: Simple spatial functions (40%)"""
        samples = []
        target = self.target_distribution["level1_simple"]
        
        # Distribute across subcategories
        unary_measurement_target = int(target * 0.25)
        unary_transformation_target = int(target * 0.15)
        binary_relationships_target = target - unary_measurement_target - unary_transformation_target
        
        # Unary measurement functions
        for table, schema_info in BENCHMARK_SCHEMA.items():
            for function in ["ST_Area", "ST_Length", "ST_Perimeter", "ST_NPoints"]:
                if function in schema_info["valid_functions"]:
                    templates = NATURAL_LANGUAGE_TEMPLATES["level1_simple"]["unary_measurement"][function]
                    for _ in range(max(1, unary_measurement_target // (len(BENCHMARK_SCHEMA) * 4))):
                        if len(samples) >= unary_measurement_target:
                            break
                        
                        entity = self.generate_entity_reference(table)
                        template = random.choice(templates)
                        input_text = template.format(entity=entity)
                        
                        placeholder = entity.split("'")[1]
                        sql = SQL_GENERATION_TEMPLATES["level1_simple"]["unary_measurement"][function].format(
                            table=table, placeholder=placeholder
                        )
                        
                        samples.append({
                            "complexity_level": "level1_simple",
                            "subcategory": "unary_measurement",
                            "function": function,
                            "table": table,
                            "geometry_type": schema_info["geometry_type"],
                            "attribute_types_used": ["identification", "spatial"],
                            "input_text": input_text,
                            "spatial_sql": sql
                        })
        
        # Unary transformation functions  
        for table, schema_info in BENCHMARK_SCHEMA.items():
            for function in ["ST_Centroid", "ST_Buffer"]:
                if function in schema_info["valid_functions"]:
                    templates = NATURAL_LANGUAGE_TEMPLATES["level1_simple"]["unary_transformation"][function]
                    iterations = max(1, unary_transformation_target // (len(BENCHMARK_SCHEMA) * 2))
                    
                    for _ in range(iterations):
                        if len([s for s in samples if s["subcategory"] == "unary_transformation"]) >= unary_transformation_target:
                            break
                            
                        entity = self.generate_entity_reference(table)
                        template = random.choice(templates)
                        
                        if function == "ST_Buffer":
                            distance = random.choice(self.distances)
                            input_text = template.format(entity=entity, distance=distance)
                            placeholder = entity.split("'")[1]
                            sql = SQL_GENERATION_TEMPLATES["level1_simple"]["unary_transformation"][function].format(
                                table=table, placeholder=placeholder, distance=distance
                            )
                        else:
                            input_text = template.format(entity=entity)
                            placeholder = entity.split("'")[1]
                            sql = SQL_GENERATION_TEMPLATES["level1_simple"]["unary_transformation"][function].format(
                                table=table, placeholder=placeholder
                            )
                        
                        samples.append({
                            "complexity_level": "level1_simple",
                            "subcategory": "unary_transformation",
                            "function": function,
                            "table": table,
                            "geometry_type": schema_info["geometry_type"],
                            "attribute_types_used": ["identification", "spatial"],
                            "input_text": input_text,
                            "spatial_sql": sql
                        })
        
        # Binary relationship functions
        table_pairs = list(itertools.combinations(BENCHMARK_SCHEMA.keys(), 2))
        table_pairs.extend([(t, t) for t in BENCHMARK_SCHEMA.keys()])  # Same table pairs
        
        functions = ["ST_Within", "ST_Contains", "ST_Intersects", "ST_Distance", "ST_DWithin"]
        binary_samples = []
        
        for table1, table2 in table_pairs:
            for function in functions:
                if (function in BENCHMARK_SCHEMA[table1]["valid_functions"] and 
                    function in BENCHMARK_SCHEMA[table2]["valid_functions"]):
                    
                    templates = NATURAL_LANGUAGE_TEMPLATES["level1_simple"]["binary_relationships"][function]
                    iterations = max(1, binary_relationships_target // (len(table_pairs) * len(functions)))
                    
                    for _ in range(iterations):
                        if len(binary_samples) >= binary_relationships_target:
                            break
                        
                        entity1 = self.generate_entity_reference(table1)
                        entity2 = self.generate_entity_reference(table2)
                        template = random.choice(templates)
                        
                        if function == "ST_DWithin":
                            distance = random.choice(self.distances)
                            input_text = template.format(entity1=entity1, entity2=entity2, distance=distance)
                            placeholder1, placeholder2 = entity1.split("'")[1], entity2.split("'")[1]
                            sql = SQL_GENERATION_TEMPLATES["level1_simple"]["binary_relationships"][function].format(
                                table1=table1, table2=table2, placeholder1=placeholder1, 
                                placeholder2=placeholder2, distance=distance
                            )
                        else:
                            input_text = template.format(entity1=entity1, entity2=entity2)
                            placeholder1, placeholder2 = entity1.split("'")[1], entity2.split("'")[1]
                            sql = SQL_GENERATION_TEMPLATES["level1_simple"]["binary_relationships"][function].format(
                                table1=table1, table2=table2, placeholder1=placeholder1, placeholder2=placeholder2
                            )
                        
                        binary_samples.append({
                            "complexity_level": "level1_simple",
                            "subcategory": "binary_relationships",
                            "function": function,
                            "tables": [table1, table2],
                            "geometry_types": [BENCHMARK_SCHEMA[table1]["geometry_type"], BENCHMARK_SCHEMA[table2]["geometry_type"]],
                            "attribute_types_used": ["identification", "spatial"],
                            "input_text": input_text,
                            "spatial_sql": sql
                        })
        
        samples.extend(binary_samples)
        return samples[:target]
    
    def generate_level2_attributes(self) -> List[Dict]:
        """Generate Level 2: Spatial + attributes (30%)"""
        samples = []
        target = self.target_distribution["level2_attributes"]
        
        # Distribute across subcategories
        attribute_filtering_target = int(target * 0.33)
        spatial_conditions_target = int(target * 0.33)
        relationship_conditions_target = target - attribute_filtering_target - spatial_conditions_target
        
        # Attribute filtering only
        for table in BENCHMARK_SCHEMA.keys():
            for attr_name in ATTRIBUTE_SYSTEM["non_spatial"].keys():
                iterations = max(1, attribute_filtering_target // (len(BENCHMARK_SCHEMA) * len(ATTRIBUTE_SYSTEM["non_spatial"])))
                
                for _ in range(iterations):
                    if len([s for s in samples if s["subcategory"] == "attribute_filtering"]) >= attribute_filtering_target:
                        break
                    
                    condition_sql, condition_desc = self.generate_condition(attr_name)
                    entities = table.replace("table_", "") + "s"
                    template = random.choice(NATURAL_LANGUAGE_TEMPLATES["level2_attributes"]["attribute_filtering"])
                    
                    input_text = template.format(entities=entities, condition=condition_desc)
                    sql = SQL_GENERATION_TEMPLATES["level2_attributes"]["attribute_filtering"].format(
                        table=table, condition=condition_sql
                    )
                    
                    samples.append({
                        "complexity_level": "level2_attributes",
                        "subcategory": "attribute_filtering",
                        "function": "attribute_filter",
                        "table": table,
                        "attribute_types_used": ["identification", attr_name.split("_")[1]],
                        "input_text": input_text,
                        "spatial_sql": sql
                    })
        
        # Spatial with conditions
        table_pairs = list(itertools.combinations(BENCHMARK_SCHEMA.keys(), 2))
        
        for table1, table2 in table_pairs:
            for attr_name in ATTRIBUTE_SYSTEM["non_spatial"].keys():
                iterations = max(1, spatial_conditions_target // (len(table_pairs) * len(ATTRIBUTE_SYSTEM["non_spatial"])))
                
                for _ in range(iterations):
                    if len([s for s in samples if s["subcategory"] == "spatial_with_conditions"]) >= spatial_conditions_target:
                        break
                    
                    condition_sql, condition_desc = self.generate_condition(attr_name)
                    entities1 = table1.replace("table_", "") + "s"
                    entity2 = self.generate_entity_reference(table2)
                    distance = random.choice(self.distances)
                    
                    template = random.choice(NATURAL_LANGUAGE_TEMPLATES["level2_attributes"]["spatial_with_conditions"])
                    input_text = template.format(
                        entities1=entities1, condition=condition_desc, 
                        distance=distance, entity2=entity2
                    )
                    
                    placeholder = entity2.split("'")[1]
                    sql = SQL_GENERATION_TEMPLATES["level2_attributes"]["spatial_with_conditions"].format(
                        table1=table1, table2=table2, condition=condition_sql,
                        distance=distance, placeholder=placeholder
                    )
                    
                    samples.append({
                        "complexity_level": "level2_attributes",
                        "subcategory": "spatial_with_conditions",
                        "function": "spatial_filter_with_attributes",
                        "tables": [table1, table2],
                        "attribute_types_used": ["identification", "spatial", attr_name.split("_")[1]],
                        "input_text": input_text,
                        "spatial_sql": sql
                    })
        
        return samples[:target]
    
    def generate_level3_aggregation(self) -> List[Dict]:
        """Generate Level 3: Aggregations (20%)"""
        samples = []
        target = self.target_distribution["level3_aggregation"]
        
        # Focus on meaningful spatial aggregations
        meaningful_pairs = [
            ("table_point", "table_polygon"),
            ("table_point", "table_multipolygon"),
            ("table_multipoint", "table_polygon"),
            ("table_line", "table_polygon"),
            ("table_multiline", "table_multipolygon")
        ]
        
        # Count aggregations
        for table1, table2 in meaningful_pairs:
            iterations = max(1, target // (len(meaningful_pairs) * 3))
            
            for _ in range(iterations):
                if len(samples) >= target:
                    break
                
                entities1 = table1.replace("table_", "") + "s"
                entity2_type = table2.replace("table_", "")
                
                template = random.choice(NATURAL_LANGUAGE_TEMPLATES["level3_aggregation"]["count_aggregation"])
                input_text = template.format(entities1=entities1, entity2_type=entity2_type, 
                                           entity1_type=table1.replace("table_", ""))
                
                sql = SQL_GENERATION_TEMPLATES["level3_aggregation"]["count_aggregation"].format(
                    table1=table1, table2=table2
                )
                
                samples.append({
                    "complexity_level": "level3_aggregation",
                    "subcategory": "count_aggregation",
                    "function": "spatial_count",
                    "tables": [table1, table2],
                    "attribute_types_used": ["identification", "spatial"],
                    "input_text": input_text,
                    "spatial_sql": sql
                })
        
        # Conditional aggregations
        for table1, table2 in meaningful_pairs:
            for attr_name in ["attr_text", "attr_numeric", "attr_boolean"]:
                if len(samples) >= target:
                    break
                
                condition_sql, condition_desc = self.generate_condition(attr_name)
                entities1 = table1.replace("table_", "") + "s"
                entity2_type = table2.replace("table_", "")
                
                template = random.choice(NATURAL_LANGUAGE_TEMPLATES["level3_aggregation"]["conditional_aggregation"])
                input_text = template.format(
                    entities1=entities1, condition=condition_desc, entity2_type=entity2_type
                )
                
                sql = SQL_GENERATION_TEMPLATES["level3_aggregation"]["conditional_aggregation"].format(
                    table1=table1, table2=table2, condition=condition_sql
                )
                
                samples.append({
                    "complexity_level": "level3_aggregation",
                    "subcategory": "conditional_aggregation", 
                    "function": "spatial_conditional_count",
                    "tables": [table1, table2],
                    "attribute_types_used": ["identification", "spatial", attr_name.split("_")[1]],
                    "input_text": input_text,
                    "spatial_sql": sql
                })
        
        return samples[:target]
    
    def generate_level4_complex(self) -> List[Dict]:
        """Generate Level 4: Complex reasoning (10%)"""
        samples = []
        target = self.target_distribution["level4_complex"]
        
        table_pairs = [
            ("table_point", "table_line"),
            ("table_point", "table_polygon"),
            ("table_line", "table_polygon"),
            ("table_multipoint", "table_multipolygon")
        ]
        
        # Negation queries
        for table1, table2 in table_pairs:
            iterations = max(1, target // (len(table_pairs) * 2))
            
            for _ in range(iterations):
                if len([s for s in samples if s["subcategory"] == "negation"]) >= target // 2:
                    break
                
                entities1 = table1.replace("table_", "") + "s"
                entity2_type = table2.replace("table_", "")
                distance = random.choice([500, 1000, 2000])
                
                template = random.choice(NATURAL_LANGUAGE_TEMPLATES["level4_complex"]["negation"])
                input_text = template.format(
                    entities1=entities1, entity2_type=entity2_type, distance=distance
                )
                
                sql = SQL_GENERATION_TEMPLATES["level4_complex"]["negation"].format(
                    table1=table1, table2=table2, distance=distance
                )
                
                samples.append({
                    "complexity_level": "level4_complex",
                    "subcategory": "negation",
                    "function": "spatial_negation",
                    "tables": [table1, table2],
                    "attribute_types_used": ["identification", "spatial"],
                    "input_text": input_text,
                    "spatial_sql": sql
                })
        
        # Nearest neighbor queries
        for table1, table2 in table_pairs:
            if len(samples) >= target:
                break
            
            entity1_type = table1.replace("table_", "")
            entity2_type = table2.replace("table_", "")
            
            template = random.choice(NATURAL_LANGUAGE_TEMPLATES["level4_complex"]["nearest_neighbor"])
            input_text = template.format(entity1_type=entity1_type, entity2_type=entity2_type)
            
            sql = SQL_GENERATION_TEMPLATES["level4_complex"]["nearest_neighbor"].format(
                table1=table1, table2=table2
            )
            
            samples.append({
                "complexity_level": "level4_complex",
                "subcategory": "nearest_neighbor",
                "function": "nearest_neighbor",
                "tables": [table1, table2],
                "attribute_types_used": ["identification", "spatial"],
                "input_text": input_text,
                "spatial_sql": sql
            })
        
        return samples[:target]
    
    def generate_systematic_dataset(self) -> List[Dict]:
        """Generate the complete systematic dataset"""
        print("\nGenerating Systematic Spatial SQL Benchmark Dataset")
        print("=" * 60)
        
        # Generate each complexity level
        print("Level 1: Simple spatial functions (40%)...")
        level1_samples = self.generate_level1_simple()
        print(f"   Generated: {len(level1_samples)} samples")
        
        print("Level 2: Spatial + attributes (30%)...")
        level2_samples = self.generate_level2_attributes()
        print(f"   Generated: {len(level2_samples)} samples")
        
        print("Level 3: Aggregations (20%)...")
        level3_samples = self.generate_level3_aggregation()
        print(f"   Generated: {len(level3_samples)} samples")
        
        print("Level 4: Complex reasoning (10%)...")
        level4_samples = self.generate_level4_complex()
        print(f"   Generated: {len(level4_samples)} samples")
        
        # Combine all samples
        all_samples = level1_samples + level2_samples + level3_samples + level4_samples
        
        # Shuffle for training diversity
        random.shuffle(all_samples)
        
        # Add sample IDs
        for i, sample in enumerate(all_samples):
            sample["sample_id"] = f"spatial_benchmark_{i+1:05d}"
        
        print("\nDataset Statistics:")
        print(f"   Total samples: {len(all_samples)}")
        print(f"   Level 1 (Simple): {len(level1_samples)} ({len(level1_samples)/len(all_samples)*100:.1f}%)")
        print(f"   Level 2 (Attributes): {len(level2_samples)} ({len(level2_samples)/len(all_samples)*100:.1f}%)")
        print(f"   Level 3 (Aggregation): {len(level3_samples)} ({len(level3_samples)/len(all_samples)*100:.1f}%)")
        print(f"   Level 4 (Complex): {len(level4_samples)} ({len(level4_samples)/len(all_samples)*100:.1f}%)")
        
        return all_samples

if __name__ == "__main__":
    # Generate the systematic benchmark dataset
    generator = SystematicBenchmarkGenerator(target_size=5000)
    dataset = generator.generate_systematic_dataset()
    
    # Save to file
    output_file = "systematic_spatial_benchmark.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")
    
    print(f"\nSystematic benchmark saved to: {output_file}")
    
    # Show sample outputs from each level
    print("\nSample Outputs by Complexity Level:")
    print("=" * 60)
    
    for level in ["level1_simple", "level2_attributes", "level3_aggregation", "level4_complex"]:
        level_samples = [s for s in dataset if s["complexity_level"] == level]
        if level_samples:
            sample = level_samples[0]
            print(f"\n{level.upper().replace('_', ' ')}:")
            print(f"   Input: {sample['input_text']}")
            print(f"   SQL: {sample['spatial_sql']}")
            print(f"   Attributes: {sample['attribute_types_used']}") 