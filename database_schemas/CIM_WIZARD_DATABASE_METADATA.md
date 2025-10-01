# CIM Wizard Integrated Database - Comprehensive Metadata Documentation

## Executive Summary

CIM Wizard Integrated is a comprehensive **City Information Modeling** platform designed for urban planning, smart city development, and building energy analysis. The system combines geospatial data processing, demographic analysis, and energy modeling to provide intelligent insights for urban development projects.

**CIM** stands for **City Information Modeling** - analogous to Building Information Modeling (BIM) but applied at the urban scale.

## Business Context & Purpose

### Core Mission
The CIM Wizard database enables urban planners, architects, and city administrators to:
- **Analyze building energy performance** at neighborhood and city scales
- **Integrate census demographics** with building characteristics
- **Calculate building properties** using multiple data sources and fallback methods
- **Model urban scenarios** for sustainable development planning
- **Generate insights** for smart city initiatives

### Key Business Applications
1. **Urban Energy Planning**: Calculate building energy demand and efficiency
2. **Demographic Analysis**: Map population distribution to building characteristics
3. **Smart City Development**: Support data-driven urban planning decisions
4. **Building Performance Assessment**: Evaluate construction periods, types, and energy characteristics
5. **Scenario Modeling**: Compare different urban development scenarios

## Database Architecture

The CIM Wizard database uses **PostgreSQL 12+** with **PostGIS 3.0+** extensions, organized into three specialized schemas:

```
CIM_WIZARD_DATABASE
├── cim_vector   (Building & Project Data)
├── cim_census   (Italian Demographic Data)
└── cim_raster   (Digital Terrain Models)
```

---

## 1. CIM_VECTOR Schema - Building & Project Management

### Business Purpose
Manages **building geometries**, **project scenarios**, **calculated properties**, and **electrical grid data** for urban modeling projects.

### Core Tables

#### 1.1 PROJECT_SCENARIO
**Business Purpose**: Central management of urban modeling projects and scenarios
**Usage**: Each project can have multiple scenarios for comparison (baseline, proposed, optimized)

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `project_id` | VARCHAR(100) | **Primary Key** - Unique project identifier | "milan_porta_garibaldi_2024" |
| `scenario_id` | VARCHAR(100) | **Primary Key** - Scenario within project | "baseline", "optimized", "proposed" |
| `project_name` | VARCHAR(255) | Human-readable project name | "Porta Garibaldi District Renovation" |
| `scenario_name` | VARCHAR(255) | Human-readable scenario name | "Energy Efficiency Upgrade Scenario" |
| `project_boundary` | GEOMETRY(POLYGON, 4326) | **Spatial boundary** of the project area | WGS84 polygon coordinates |
| `project_center` | GEOMETRY(POINT, 4326) | **Map center point** for visualization | WGS84 lat/lng coordinates |
| `project_zoom` | INTEGER | **Default zoom level** for maps | 15 (typical urban scale) |
| `project_crs` | INTEGER | **Coordinate Reference System** | 4326 (WGS84) |
| `census_boundary` | GEOMETRY(MULTIPOLYGON, 4326) | **Combined census zones** covering project | Union of all relevant SEZ2011 zones |
| `cosimulator_config_mongo_path` | VARCHAR | **Path to energy simulation config** in MongoDB | "/configs/simulation_setup.json" |
| `network_mongo_path` | JSON | **Electrical grid model** references | JSON with network topology data |
| `results_mongo_path` | VARCHAR | **Path to simulation results** | "/results/2024_scenario_1.json" |
| `created_at` | TIMESTAMP WITH TIME ZONE | **Creation timestamp** | Auto-generated |
| `updated_at` | TIMESTAMP WITH TIME ZONE | **Last modification** | Auto-updated |

#### 1.2 BUILDING
**Business Purpose**: Stores building geometries and basic identification
**Usage**: Master table for all buildings in the system, referenced by properties

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** - Internal unique ID | 1, 2, 3... |
| `building_id` | VARCHAR(100) | **Business Key** - External building identifier | "OSM_12345", "CATASTO_AB123" |
| `lod` | INTEGER | **Level of Detail** (0=footprint, 1=with height, 2=detailed) | 0, 1, 2 |
| `building_geometry` | GEOMETRY(GEOMETRY, 4326) | **Building footprint** geometry | Polygon, MultiPolygon |
| `building_geometry_source` | VARCHAR(50) | **Data source** for geometry | "osm", "catasto", "lidar" |
| `census_id` | BIGINT | **Link to census zone** (SEZ2011) | 59006001001 |
| `building_surfaces_lod12` | JSON | **3D surfaces** for detailed models | JSON with wall/roof geometries |
| `created_at` | TIMESTAMP WITH TIME ZONE | **Creation timestamp** | Auto-generated |
| `updated_at` | TIMESTAMP WITH TIME ZONE | **Last modification** | Auto-updated |

#### 1.3 BUILDING_PROPERTIES
**Business Purpose**: Stores calculated and derived building characteristics
**Usage**: Contains all computed properties like energy performance, demographics, construction details

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `building_id` | VARCHAR(100) | **Foreign Key** to building | "OSM_12345" |
| `project_id` | VARCHAR(100) | **Foreign Key** to project | "milan_porta_garibaldi_2024" |
| `scenario_id` | VARCHAR(100) | **Foreign Key** to scenario | "baseline" |
| `lod` | INTEGER | **Level of Detail** for this calculation | 0, 1, 2 |
| `building_fk` | INTEGER | **Foreign Key** to building table | 1, 2, 3... |
| **Physical Properties** |
| `height` | FLOAT | **Building height** in meters (DSM-DTM or estimated) | 15.5, 23.2, 45.8 |
| `area` | FLOAT | **Footprint area** in square meters | 125.3, 890.7 |
| `volume` | FLOAT | **Building volume** (height × area) in cubic meters | 1940.5, 20571.6 |
| `number_of_floors` | FLOAT | **Estimated floors** (height ÷ floor_height) | 3.5, 7.2, 15.0 |
| **Building Characteristics** |
| `type` | VARCHAR(50) | **Building use type** | "residential", "commercial", "industrial" |
| `const_period_census` | VARCHAR(10) | **ISTAT construction period** | "E8", "E12", "E16" |
| `const_year` | INTEGER | **Estimated construction year** | 1975, 1998, 2010 |
| `const_TABULA` | VARCHAR(15) | **TABULA energy efficiency period** | "TABULA_4", "TABULA_6" |
| **Demographics** |
| `n_people` | INTEGER | **Estimated residents/occupants** | 3, 8, 45 |
| `n_family` | INTEGER | **Estimated family units** | 1, 2, 15 |
| **Additional Properties** |
| `gross_floor_area` | FLOAT | **Total floor area** (area × floors) | 438.55, 6245.04 |
| `net_leased_area` | FLOAT | **Usable floor area** | 394.7, 5620.5 |
| `neighbours_surfaces` | JSON | **Adjacent building surfaces** for energy calc | JSON array of shared walls |
| `neighbours_ids` | JSON | **IDs of adjacent buildings** | ["OSM_12346", "OSM_12344"] |
| **Energy Properties** |
| `heating` | BOOLEAN | **Has heating system** | true, false |
| `cooling` | BOOLEAN | **Has cooling system** | true, false |
| `w2w` | FLOAT | **Window-to-wall ratio** | 0.15, 0.30, 0.45 |
| `hvac_type` | VARCHAR(50) | **HVAC system type** | "heat_pump", "gas_boiler" |
| `extra_prop` | JSON | **Additional calculated properties** | JSON object with custom fields |
| `created_at` | TIMESTAMP WITH TIME ZONE | **Creation timestamp** | Auto-generated |
| `updated_at` | TIMESTAMP WITH TIME ZONE | **Last modification** | Auto-updated |

#### 1.4 GRID_BUS
**Business Purpose**: Electrical grid bus/node management for energy distribution modeling
**Usage**: Represents electrical connection points in urban power grid analysis

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `network_id` | VARCHAR(100) | **Electrical network identifier** | "milan_grid_lv_01" |
| `bus_id` | INTEGER | **Bus number** within network | 1, 15, 847 |
| `project_id` | VARCHAR(100) | **Foreign Key** to project | "milan_porta_garibaldi_2024" |
| `scenario_id` | VARCHAR(100) | **Foreign Key** to scenario | "baseline" |
| `geometry` | GEOMETRY(POINT, 4326) | **Geographic location** of bus | WGS84 point coordinates |
| `name` | VARCHAR(255) | **Human-readable bus name** | "Substation A", "Distribution Point 15" |
| `voltage_kv` | FLOAT | **Operating voltage** in kilovolts | 0.4, 15.0, 132.0 |
| `zone` | VARCHAR(50) | **Grid zone** classification | "LV", "MV", "HV" |
| `in_service` | BOOLEAN | **Operational status** | true, false |
| `created_at` | TIMESTAMP WITH TIME ZONE | **Creation timestamp** | Auto-generated |
| `updated_at` | TIMESTAMP WITH TIME ZONE | **Last modification** | Auto-updated |

#### 1.5 GRID_LINE
**Business Purpose**: Electrical transmission lines connecting grid buses
**Usage**: Models electrical connections and power flow capacity in urban grids

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `network_id` | VARCHAR(100) | **Electrical network identifier** | "milan_grid_lv_01" |
| `line_id` | INTEGER | **Line number** within network | 1, 23, 156 |
| `project_id` | VARCHAR(100) | **Foreign Key** to project | "milan_porta_garibaldi_2024" |
| `scenario_id` | VARCHAR(100) | **Foreign Key** to scenario | "baseline" |
| `geometry` | GEOMETRY(LINESTRING, 4326) | **Geographic path** of line | WGS84 linestring coordinates |
| `name` | VARCHAR(255) | **Human-readable line name** | "Feeder A-1", "Distribution Line 23" |
| `from_bus` | INTEGER | **Source bus ID** | 1, 15, 847 |
| `to_bus` | INTEGER | **Destination bus ID** | 2, 16, 848 |
| `length_km` | FLOAT | **Line length** in kilometers | 0.145, 2.34, 15.8 |
| `max_loading_percent` | FLOAT | **Maximum capacity utilization** | 80.0, 95.5 |
| `created_at` | TIMESTAMP WITH TIME ZONE | **Creation timestamp** | Auto-generated |
| `updated_at` | TIMESTAMP WITH TIME ZONE | **Last modification** | Auto-updated |

---

## 2. CIM_CENSUS Schema - Italian Demographic Data

### Business Purpose
Stores **Italian National Census (ISTAT)** data for demographic analysis and building occupancy estimation. Based on **Sezioni di Censimento 2011** (Census Sections).

### Understanding Italian Census Variables

The Italian census system organizes data by **SEZ2011** codes (Sezione di Censimento 2011 - Census Section 2011), which are the smallest territorial units for statistical purposes.

#### 2.1 CENSUS_GEO
**Business Purpose**: Complete Italian census data with spatial boundaries
**Usage**: Provides demographic context for building occupancy and energy demand calculations

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `SEZ2011` | BIGINT | **ISTAT Census Section ID** (unique) | 59006001001, 15063002015 |
| `geometry` | GEOMETRY(MULTIPOLYGON, 4326) | **Census zone boundary** | WGS84 multipolygon |
| `crs` | VARCHAR(100) | **Coordinate Reference System** | "urn:ogc:def:crs:OGC:1.3:CRS84" |
| **Administrative Hierarchy** |
| `Shape_Area` | FLOAT | **Zone area** in square meters | 125847.3, 89234.7 |
| `CODREG` | VARCHAR(10) | **Region code** (Italy has 20 regions) | "03" (Lombardia), "12" (Lazio) |
| `REGIONE` | VARCHAR(50) | **Region name** | "Lombardia", "Lazio", "Piemonte" |
| `CODPRO` | VARCHAR(10) | **Province code** | "015" (Milano), "058" (Roma) |
| `PROVINCIA` | VARCHAR(50) | **Province name** | "Milano", "Roma", "Torino" |
| `CODCOM` | VARCHAR(10) | **Municipality code** | "015146" (Milano) |
| `COMUNE` | VARCHAR(50) | **Municipality name** | "Milano", "Roma", "Napoli" |
| `PROCOM` | VARCHAR(10) | **Province+Municipality code** | "015146" |
| `NSEZ` | VARCHAR(10) | **Section number** within municipality | "0001", "0045" |
| `ACE` | VARCHAR(10) | **Census enumeration area** | "001" |
| `CODLOC` | VARCHAR(10) | **Locality code** | "0001" |
| `CODASC` | VARCHAR(10) | **Sub-locality code** | "001" |
| **Population Statistics (P1-P140 Variables)** |
| `P1` | INTEGER | **Total Population** | 1247, 3892 |
| `P2` | INTEGER | **Males** | 598, 1876 |
| `P3` | INTEGER | **Females** | 649, 2016 |
| `P4`-`P66` | INTEGER | **Age groups, education, employment** | Various demographic breakdowns |
| `P128`-`P140` | INTEGER | **Extended population attributes** | Additional demographic details |
| **Housing Statistics (ST1-ST15 Variables)** |
| `ST1` | INTEGER | **Total Housing Units** | 534, 1678 |
| `ST2` | INTEGER | **Occupied Housing Units** | 489, 1456 |
| `ST3` | INTEGER | **Vacant Housing Units** | 45, 222 |
| `ST4`-`ST15` | INTEGER | **Housing characteristics** | Room counts, ownership types |
| **Building Age Distribution (E1-E31 Variables)** |
| `E8` | INTEGER | **Buildings before 1918** | 12, 89 |
| `E9` | INTEGER | **Buildings 1919-1945** | 23, 145 |
| `E10` | INTEGER | **Buildings 1946-1960** | 67, 234 |
| `E11` | INTEGER | **Buildings 1961-1970** | 89, 178 |
| `E12` | INTEGER | **Buildings 1971-1980** | 123, 267 |
| `E13` | INTEGER | **Buildings 1981-1990** | 87, 198 |
| `E14` | INTEGER | **Buildings 1991-2000** | 45, 134 |
| `E15` | INTEGER | **Buildings 2001-2005** | 34, 98 |
| `E16` | INTEGER | **Buildings after 2005** | 28, 76 |
| **Building Attributes (A2-A48 Variables)** |
| `A2`-`A48` | INTEGER | **Building materials, condition** | Construction details |
| **Family Statistics (PF1-PF9 Variables)** |
| `PF1` | INTEGER | **Total Families** | 456, 1423 |
| `PF2`-`PF9` | INTEGER | **Family size distribution** | 1-person, 2-person families, etc. |

---

## 3. CIM_RASTER Schema - Digital Terrain Models

### Business Purpose
Manages **Digital Terrain Models (DTM)** and **Digital Surface Models (DSM)** for accurate building height calculation and 3D urban modeling.

#### 3.1 RASTER_MODEL
**Business Purpose**: Generic raster data storage with metadata
**Usage**: Base table for all raster datasets in the system

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `rast` | BYTEA | **Raster data** in binary format | Binary GeoTIFF data |
| `filename` | VARCHAR(255) | **Original filename** | "milano_dtm_2023.tif" |
| `raster_type` | VARCHAR(50) | **Type of raster** | "DTM", "DSM", "CHM" |
| `srid` | INTEGER | **Spatial Reference ID** | 4326, 32632 |
| `width` | INTEGER | **Raster width** in pixels | 1024, 2048, 4096 |
| `height` | INTEGER | **Raster height** in pixels | 1024, 2048, 4096 |
| `scale_x` | FLOAT | **Pixel size** in X direction | 0.5, 1.0, 2.0 |
| `scale_y` | FLOAT | **Pixel size** in Y direction | 0.5, 1.0, 2.0 |
| `upperleft_x` | FLOAT | **Upper left X coordinate** | 1514000.0 |
| `upperleft_y` | FLOAT | **Upper left Y coordinate** | 5034000.0 |
| `upload_date` | TIMESTAMP WITH TIME ZONE | **Upload timestamp** | Auto-generated |
| `updated_at` | TIMESTAMP WITH TIME ZONE | **Last modification** | Auto-updated |

#### 3.2 DTM_RASTER
**Business Purpose**: Digital Terrain Model - ground elevation without buildings
**Usage**: Provides bare earth elevation for building height calculations

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `rast` | BYTEA | **DTM raster data** | Binary elevation data |
| `filename` | VARCHAR(255) | **DTM filename** | "lombardia_dtm_2023.tif" |
| `upload_date` | TIMESTAMP WITH TIME ZONE | **Upload timestamp** | Auto-generated |
| `srid` | INTEGER | **Spatial Reference ID** | 4326, 32632 |
| `min_elevation` | FLOAT | **Minimum ground elevation** in meters | 95.2, 145.8 |
| `max_elevation` | FLOAT | **Maximum ground elevation** in meters | 387.9, 892.3 |

#### 3.3 DSM_RASTER
**Business Purpose**: Digital Surface Model - elevation including buildings and vegetation
**Usage**: Provides top-of-surface elevation for building height calculations

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `rast` | BYTEA | **DSM raster data** | Binary elevation data |
| `filename` | VARCHAR(255) | **DSM filename** | "lombardia_dsm_2023.tif" |
| `upload_date` | TIMESTAMP WITH TIME ZONE | **Upload timestamp** | Auto-generated |
| `srid` | INTEGER | **Spatial Reference ID** | 4326, 32632 |
| `min_elevation` | FLOAT | **Minimum surface elevation** in meters | 95.2, 145.8 |
| `max_elevation` | FLOAT | **Maximum surface elevation** in meters | 445.7, 987.1 |

#### 3.4 BUILDING_HEIGHT_CACHE
**Business Purpose**: Cached building height calculations for performance optimization
**Usage**: Stores pre-calculated height values to avoid repeated raster processing

| Column | Data Type | Business Meaning | Example Values |
|--------|-----------|------------------|----------------|
| `id` | SERIAL | **Primary Key** | 1, 2, 3... |
| `building_id` | VARCHAR(100) | **Reference to building** | "OSM_12345" |
| `project_id` | VARCHAR(100) | **Reference to project** | "milan_porta_garibaldi_2024" |
| `scenario_id` | VARCHAR(100) | **Reference to scenario** | "baseline" |
| `dtm_avg_height` | FLOAT | **Average ground elevation** in meters | 125.3, 89.7 |
| `dsm_avg_height` | FLOAT | **Average surface elevation** in meters | 140.8, 114.2 |
| `building_height` | FLOAT | **Calculated height** (DSM - DTM) in meters | 15.5, 24.5 |
| `calculation_date` | TIMESTAMP WITH TIME ZONE | **Calculation timestamp** | Auto-generated |
| `calculation_method` | VARCHAR(50) | **Method used** | "raster_intersection" |
| `coverage_percentage` | FLOAT | **Raster coverage** of building footprint | 0.85, 0.92 |
| `confidence_score` | FLOAT | **Confidence in calculation** | 0.89, 0.95 |

---

## Key Business Concepts & Calculations

### TABULA Classification System
**TABULA** is a European building typology system for energy efficiency analysis:
- **TABULA_1**: Before 1900 - Very poor insulation
- **TABULA_2**: 1901-1920 - Poor insulation
- **TABULA_3**: 1921-1945 - Basic construction
- **TABULA_4**: 1946-1960 - Post-war reconstruction
- **TABULA_5**: 1961-1975 - Economic boom period
- **TABULA_6**: 1976-1990 - First energy standards
- **TABULA_7**: 1991-2005 - Modern energy standards

### Building Height Calculation Methods
The system uses multiple fallback methods:
1. **Raster Method** (Priority 1): DSM - DTM calculation
2. **OSM Height** (Priority 2): Height tags from OpenStreetMap
3. **Default Estimate** (Priority 3): Type-based estimates

### Demographic Distribution Algorithm
Uses Italian census data (E8-E16 variables) to:
- Assign construction periods to buildings
- Estimate occupancy based on building type and census population
- Calculate energy demand based on demographics

---

## Data Quality & Validation

### Spatial Data Quality
- **SRID 4326** (WGS84) for all geometries
- **PostGIS validation** ensures geometric integrity
- **Spatial indexes** on all geometry columns for performance

### Data Lineage
- All calculated values include **source tracking**
- **Confidence scores** for estimation accuracy
- **Update timestamps** for data freshness monitoring

### Business Rules
- Building heights must be between 0-500 meters
- Population values must be non-negative
- Census zones must have valid SEZ2011 codes
- Project boundaries must be valid polygons

---

## Technical Implementation

### Database Features
- **PostgreSQL 12+** with **PostGIS 3.0+**
- **Multi-schema architecture** for data separation
- **Spatial indexing** with GiST indexes
- **ACID compliance** for data integrity
- **Foreign key constraints** for referential integrity

### Performance Optimizations
- **Spatial indexes** on all geometry columns
- **Composite indexes** on frequently joined columns
- **Raster caching** for expensive height calculations
- **Connection pooling** for concurrent access

This comprehensive metadata documentation provides the complete business and technical context for the CIM Wizard Integrated database system.
