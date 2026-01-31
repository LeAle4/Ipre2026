# Geoglyph Dataset Processing Pipeline

A Python-based toolkit for extracting, processing, and managing geoglyph polygon data from geo-referenced orthomosaic imagery across multiple archaeological study areas in northern Chile.

## Overview

This project provides a complete pipeline for working with geoglyph datasets from three study areas:
- **Cerro Unita**
- **ChugChug**
- **Lluta**

The toolkit enables extraction of image crops from orthomosaic TIFs based on labeled polygons (geoglyphs, ground, and roads), providing both the original geo-referenced images and visualization overlays.

## Project Structure

```
.
├── data/                      # Raw and processed data directory
│   ├── unita_raw/            # Raw data for Cerro Unita area
│   ├── chugchug_raw/         # Raw data for ChugChug area
│   ├── lluta_raw/            # Raw data for Lluta area
│   ├── unita_polygons/       # Processed polygon extracts (Unita)
│   ├── chugchug_polygons/    # Processed polygon extracts (ChugChug)
│   └── lluta_polygons/       # Processed polygon extracts (Lluta)
│
├── dataset/                   # Dataset processing tools
│   ├── extract.py            # Main extraction script (see below)
│   ├── crop.py               # (Placeholder for cropping utilities)
│   └── resize.py             # (Placeholder for resizing utilities)
│
├── visualization/             # Visualization and analysis tools
│   ├── viewer.py             # (Placeholder for data viewer)
│   └── data_properties.py    # Commented analysis utilities
│
├── tests/                     # Unit tests directory
│
├── utils.py                   # Core utility library (see below)
└── README.md                  # This file
```

## Core Components

### `utils.py` - Data Management Utility Library

**Purpose**: Provides a unified, object-oriented interface for accessing and managing polygon data across all study areas.

**Key Features**:
- **Path Management**: Centralized configuration for raw data, processed polygons, and summary files for each study area
- **Class Definitions**: Standardized polygon classification (`geo`=1, `ground`=2, `road`=3)
- **Polygon Class**: Rich object representation including:
  - Spatial properties (coordinates, dimensions in pixels and meters)
  - File paths (JPEG, TIF, overlay images)
  - Complete metadata access
  - Shapely geometry objects for GIS operations
- **Data Retrieval Functions**:
  - `get_polygons(area_name, classes)`: Fetch polygons with optional class filtering
  - `get_geos(area)`: Quick access to geoglyph-only data

**Usage Example**:
```python
from utils import get_polygons, get_geos, CLASSES

# Get all geoglyphs from Unita
geos = get_geos("unita")

# Get ground and road polygons from ChugChug
non_geos = get_polygons("chugchug", classes=(CLASSES["ground"], CLASSES["road"]))

# Access polygon properties
for poly in geos:
    print(f"Polygon {poly.id}: {poly.size_m[0]:.2f}m x {poly.size_m[1]:.2f}m")
    print(f"  JPEG: {poly.jpeg_path}")
```

**Why it's useful**: Eliminates boilerplate code for data loading, provides consistent interface across different study areas, and ensures type safety with well-defined data structures.

---

### `dataset/extract.py` - Polygon Image Extraction Tool

**Purpose**: Extract geo-referenced image crops from orthomosaic TIFs for each polygon in a geopackage file.

**Key Features**:
- **Multi-format Output**:
  - Original geo-referenced TIF (preserves coordinate system and transform)
  - Standard JPEG (for easy viewing/sharing)
  - Overlay JPEG with polygon boundaries highlighted in yellow
- **Accurate Measurements**: Calculates bounding box dimensions in meters using geodetic calculations (WGS84)
- **Flexible Filtering**:
  - Process specific polygon classes (geoglyphs, ground, or roads)
  - Limit processing to first N polygons for testing
- **CRS Handling**: Automatically reprojects polygons to match orthomosaic coordinate reference system
- **Comprehensive Metadata**: Generates JSON metadata for each polygon including:
  - Spatial dimensions (pixels and meters)
  - Coordinate bounds
  - CRS information
  - Polygon vertex coordinates
  - File references

**Command-line Interface**:
```bash
# Extract all polygons from a geopackage
python dataset/extract.py \
  --layers data/unita_raw/ML_labeling_UNITA.gpkg \
  --ortho data/unita_raw/CerroUnita_ortomosaico.tif \
  --output data/unita_polygons

# Extract only first 10 geoglyphs (class 1)
python dataset/extract.py \
  --layers data/unita_raw/ML_labeling_UNITA.gpkg \
  --ortho data/unita_raw/CerroUnita_ortomosaico.tif \
  --output data/unita_polygons \
  --limit 10 \
  --class-filter 1
```

**Output Structure**:
```
output_directory/
├── geoglif_0000_ortho.tif       # Geo-referenced TIF
├── geoglif_0000_ortho.jpg       # Standard JPEG
├── geoglif_0000_overlay.jpg     # JPEG with polygon overlay
├── geoglif_0000_metadata.json   # Polygon metadata
├── ...
└── summary.json                 # Dataset summary
```

**Why it's useful**: Automates the tedious process of extracting training data from large orthomosaic files, ensures consistency across extractions, and preserves all necessary metadata for machine learning workflows or GIS analysis.

---

### `visualization/data_properties.py`

**Purpose**: Utility functions for analyzing spatial resolution of geo-referenced TIF files.

Contains commented-out function `obtener_resolucion_cm()` that calculates pixel resolution in centimeters, handling both projected (meters/feet) and geographic (degrees) coordinate systems.

---

## Data Organization

### Raw Data Structure

Each study area's raw data directory should contain:
- **Orthomosaic TIF**: High-resolution georeferenced imagery
- **Geopackage (.gpkg)**: Vector polygons with `class` attribute
- **Auxiliary files**: `.tif.aux.xml` metadata files

Example for Unita:
```
data/unita_raw/
├── CerroUnita_ortomosaico.tif
├── CerroUnita_ortomosaico.tif.aux.xml
├── ML_labeling_UNITA.gpkg
├── CerroUnita_DEM.tif
└── Unita.tif
```

### Processed Data Structure

After running extraction, each polygon directory contains:
```
data/{area}_polygons/
├── geoglif_0000_ortho.tif
├── geoglif_0000_ortho.jpg
├── geoglif_0000_overlay.jpg
├── geoglif_0000_metadata.json
├── ...
└── {area}_summary.json
```

## Polygon Classification

The project uses a three-class system:

| Class ID | Name    | Description                          |
|----------|---------|--------------------------------------|
| 1        | geo     | Geoglyphs (archaeological features)  |
| 2        | ground  | Ground/background areas              |
| 3        | road    | Roads and paths                      |

## Dependencies

Core Python packages required:
- `geopandas` - GeoPackage reading and spatial operations
- `rasterio` - Raster data I/O and processing
- `shapely` - Geometric operations
- `matplotlib` - Overlay image generation
- `numpy` - Array operations
- `pyproj` - Coordinate system transformations

## Workflow

### 1. Data Extraction
Extract polygon images from raw data:
```bash
python dataset/extract.py \
  --layers data/unita_raw/ML_labeling_UNITA.gpkg \
  --ortho data/unita_raw/CerroUnita_ortomosaico.tif \
  --output data/unita_polygons
```

### 2. Data Access
Load and work with extracted polygons:
```python
from utils import get_polygons

# Load all polygons from Unita
polygons = get_polygons("unita")

# Iterate through geoglyphs
for poly in polygons:
    if poly.class_id == 1:  # Geoglyphs only
        # Access images
        img_path = poly.jpeg_path
        # Access spatial info
        width_m, height_m = poly.size_m
        # Access geometry
        area = poly.polygon.area
```

## Future Development

The following components are placeholders for future functionality:
- `dataset/crop.py` - Advanced cropping utilities
- `dataset/resize.py` - Batch resizing operations
- `visualization/viewer.py` - Interactive data viewer
- Expand `visualization/data_properties.py` - Dataset statistics and quality checks

## Notes

- All coordinate operations are performed using geodetic calculations for accuracy
- The extraction script preserves the original CRS and transformation matrix in TIF outputs
- Overlay images include 5% padding around polygon bounds for context
- Summary JSON files are compatible with the `utils.py` Polygon class structure

## Author & Context

Developed for archaeological geoglyph research in northern Chile (IPRES 2026).
Study areas: Cerro Unita, ChugChug, and Lluta.
