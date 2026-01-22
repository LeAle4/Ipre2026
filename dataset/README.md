# Geoglif Image Extraction and Analysis

This directory contains tools for extracting geo-referenced images from orthomosaic data and analyzing their spatial dimensions. The project processes polygons from geopackage files and generates image crops with metadata.

## Project Structure

```
project/
├── data/
│   ├── unita_raw/               # Raw extracted images for UNITA
│   ├── unita_geos/              # Class 1 (geo) images only - UNITA
│   ├── unita_polygons/          # UNITA polygon metadata
│   ├── chugchug_raw/            # Raw extracted images for CHUG
│   ├── chug_geos/               # Class 1 (geo) images only - CHUG
│   ├── chug_polygons/           # CHUG polygon metadata
│   ├── lluta_raw/               # Raw extracted images for LLUTA
│   ├── lluta_geos/              # Class 1 (geo) images only - LLUTA
│   ├── lluta_polygons/          # LLUTA polygon metadata
│   ├── salvador_raw/            # Raw extracted images for SALVADOR
│   ├── granllama_raw/           # Raw extracted images for GRANLLAMA
│   └── [other area directories]
└── dataset/
    ├── handle.py                # Centralized file handling and paths
    ├── extract_polygon_images.py
    ├── analysis.py
    └── README.md
```

## Overview

### handle.py
**New centralized module for file handling and path management.** Provides:
- **Common directory paths** - All data directories for raw, geos, and polygon data
- **Dataset configuration** - DATASETS dictionary with flexible configuration for all areas
- **File I/O utilities** - `load_json()`, `save_json()` functions for consistent data handling
- **Dataset helpers** - `get_dataset_info()`, `get_all_datasets()` for easy dataset access

This module eliminates code duplication and provides a single source of truth for all paths and common functions used across the project.

### extract_polygon_images.py
Extracts images from orthomosaic TIF files for each polygon in a geopackage. For each polygon, it generates:
- **Original ortho image crop** (GeoTIFF + JPEG)
- **Ortho image with polygon overlay** (JPEG with polygon boundaries)
- **Metadata JSON** containing bounding box coordinates, CRS, image shape, and dimensions in meters

### analysis.py
Analyzes the spatial dimensions of extracted geo polygons across multiple datasets. Now uses `handle.py` for path and file management. It:
- Loads metadata from previously extracted images using centralized paths
- Calculates statistics for polygon dimensions (width, height, area)
- Removes outliers using the IQR (Interquartile Range) method
- Estimates pixel scale and suggests sliding window sizes
- Generates comparative visualizations across datasets
- Optionally copies geo-class images to organized output directories

## Installation

1. Install required dependencies:
```bash
pip install -r ../requirements.txt
```

Required packages:
- `geopandas` - Geospatial data handling
- `rasterio` - Raster data I/O
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `shapely` - Geometric operations
- `pyproj` - Coordinate transformations

## Usage

### 1. Extract Polygon Images

#### Basic Usage
Extract all polygons from a geopackage and orthomosaic:

```bash
cd dataset
python extract_polygon_images.py \
  --layers ../data/unita_polygons/ML_labeling_UNITA.gpkg \
  --ortho ../data/unita_raw/CerroUnita_ortomosaico.tif \
  --output ../data/unita_raw
```

#### With Limit
Process only the first N polygons (useful for testing):

```bash
python extract_polygon_images.py --limit 10
```

#### Custom Output Directory
```bash
python extract_polygon_images.py --output ../data/custom_output
```

#### Full Example with All Parameters
```bash
python extract_polygon_images.py \
  --layers ../data/lluta_polygons/ML_labeling_LLUTA.gpkg \
  --ortho ../data/lluta_raw/CerroLluta_ortomosaico.tif \
  --output ../data/lluta_raw \
  --limit 100
```

#### Command-line Arguments
- `--layers` (Path): Geopackage file containing polygons. *Default:* `./data/ML_labeling_UNITA.gpkg`
- `--ortho` (Path): Orthomosaic TIF file. *Default:* `./data/CerroUnita_ortomosaico.tif`
- `--output` (Path): Output directory for extracted images. *Default:* `extracted_geoglifs/`
- `--limit` (int): Limit processing to first N polygons. *Default:* Process all

#### Output Files
For each processed polygon, the script generates:
- `geoglif_XXXX_ortho.tif` - Original ortho crop (GeoTIFF format with spatial reference)
- `geoglif_XXXX_ortho.jpg` - Original ortho crop (JPEG)
- `geoglif_XXXX_overlay.jpg` - Ortho with polygon boundary overlay (JPEG, 5% padding)
- `geoglif_XXXX_metadata.json` - Polygon metadata (coordinates, CRS, dimensions)
- `summary.json` - Summary of all processed polygons

### 2. Analyze Geo Dimensions

#### Basic Usage
Analyze dimensions of geo-class polygons across datasets:

```bash
cd dataset
python analysis.py
```

The script will:
1. Load summary data from all three datasets (UNITA, CHUG, LLUTA)
2. Extract class 1 (geo) polygons from each
3. Calculate statistics (width, height, area)
4. Remove outliers using IQR method
5. Estimate pixel scale and sliding window sizes
6. Display comparative visualizations

#### Output
The script prints:
```
==========================================================================================
GEO DIMENSIONS STATISTICS (for sliding window determination)
==========================================================================================

UNITA:
  Count:           45 (outliers removed from mean calculation)
  Width:  avg=12.3m  min=8.5m  max=18.2m
  Height: avg=11.8m  min=7.9m  max=16.5m
  Area:   avg=145.3m²  min=67.2m²  max=300.1m²
  Pixel scale: 0.0254 m/pixel
  Suggested sliding window: 13m x 13m (512px x 512px)

CHUG:
  ...

LLUTA:
  ...
```

And generates three visualization figures:
1. **Statistics Summary** - Textual statistics for each dataset
2. **Individual Distribution Histograms** - Distribution plots for each dataset
3. **Comparative Box Plots** - Side-by-side comparison across datasets

Press `Ctrl+C` or close the matplotlib windows to exit after viewing plots.

## Customization

### Using handle.py for Custom Paths

The `handle.py` module provides centralized path management. To use it in your own scripts:

```python
from handle import (
    UNITA_GEOS_DIR,     # Path to UNITA geos directory
    CHUG_SUMMARY_PATH,  # Path to CHUG summary.json
    load_json,          # Load JSON files
    save_json,          # Save JSON files
    get_dataset_info,   # Get dataset configuration
    get_all_datasets    # List all available datasets
)

# Load a summary file
summary = load_json(UNITA_SUMMARY_PATH)

# Get information for a specific dataset
dataset = get_dataset_info('unita')
print(dataset['geos_dir'])  # Prints path to unita_geos directory

# List all datasets with complete directory structure
datasets = get_all_datasets()  # Returns ['unita', 'chug', 'lluta']

# Save data with automatic directory creation
save_json(my_data, output_path / "results.json")
```

#### Adding New Datasets

To add a new dataset area, edit the `DATASETS` dictionary in `handle.py`:

```python
DATASETS = {
    'mynewarea': {
        'name': 'MYNEWAREA',
        'raw_dir': DATA_DIR / "mynewarea_raw",
        'geos_dir': DATA_DIR / "mynewarea_geos",
        'polygons_dir': DATA_DIR / "mynewarea_polygons",
        'summary_path': DATA_DIR / "mynewarea_raw" / "summary.json"
    }
}
```

### Modifying extract_polygon_images.py

#### Change Output Format
Modify the `save_image_no_axes()` function to use different raster drivers:
```python
# Line 88: Change the driver parameter
with rasterio.open(
    str(output_path),
    'w',
    driver='GTiff',  # Change to 'PNG' or other GDAL drivers
    ...
)
```

#### Adjust Polygon Overlay Padding
Modify the padding percentage in `extract_polygon_images()`:
```python
# Line 163: Current padding is 5%
padding_x = width * 0.05  # Change 0.05 to desired padding fraction
padding_y = height * 0.05
```

#### Change Image DPI/Quality
Modify the `dpi` parameter in save functions:
```python
# Line 95, 123, 156: Default is 150 DPI
save_jpeg_no_axes(ortho_rgb, jpeg_path, dpi=300)  # Higher quality
```

#### Process Specific Polygon Classes
Modify the polygon selection in the main loop:
```python
# Around line 220: Filter by class
if row['class'] == 1:  # Only process class 1 (geos)
    metadata = extract_polygon_images(...)
```

### Modifying analysis.py

#### Using Centralized Paths

The script now imports paths from `handle.py`:
```python
from handle import (
    UNITA_SUMMARY_PATH,
    CHUG_SUMMARY_PATH,
    LLUTA_SUMMARY_PATH,
    load_json
)

# Load summary files using centralized paths
summary = load_json(UNITA_SUMMARY_PATH)
```

#### Change Outlier Detection
Modify the IQR multiplier:
```python
# Line 72: Current multiplier is 1.5 (standard)
def _remove_outliers_iqr(data, multiplier=1.5):
    # multiplier=1.0 is more aggressive, 3.0 is more lenient
```

#### Add New Dataset to Analysis

To analyze an additional dataset, use the centralized configuration:
```python
from handle import get_dataset_info, load_json

# Get dataset configuration
new_dataset = get_dataset_info('salvador')

# Load its summary
summary = load_json(new_dataset['summary_path'])

# Add to analysis
summaries.append(summary)
dataset_names.append(new_dataset['name'])
```

#### Adjust Histogram Bins
Change bin count for distribution plots:
```python
# In plot functions: bins parameter
plot_histograms(widths, heights, areas, bins=30)  # Default is 20
```

#### Change Visualization Colors
Modify the colors list:
```python
# Line 166
colors = ['skyblue', 'lightgreen', 'lightcoral']
# Change to your preferred colors
```

#### Copy Geos Files by Area
Call the `copy_geos_files_by_area()` function to organize extracted images:
```python
copy_geos_files_by_area(
    summary_json_path="../data/unita_raw/summary.json",
    source_dir="../data/unita_raw",
    output_base_dir="../data/organized_geos",
    area_name="UNITA"
)
```

## Workflow Example

### Step 1: Extract Images for UNITA
```bash
cd dataset
python extract_polygon_images.py \
  --layers ../data/unita_polygons/ML_labeling_UNITA.gpkg \
  --ortho ../data/unita_raw/CerroUnita_ortomosaico.tif \
  --output ../data/unita_raw \
  --limit 50  # Start with first 50 for testing
```

Expected output:
```
Loading polygons from ../data/unita_polygons/ML_labeling_UNITA.gpkg...
Loaded 127 polygons
Original CRS: EPSG:32719
Limiting to first 50 polygons
Ortho CRS: EPSG:32719

Processing polygon 1/50 (FID: 0, class: 1 - geo)...
  Bounding box: 12.34m x 11.89m (area: 146.66 m²)

Processing polygon 2/50 (FID: 1, class: 2 - ground)...
  Bounding box: 15.67m x 14.23m (area: 222.86 m²)
...

Done! Processed 50 polygons.
Output directory: ../data/unita_raw
Summary saved to: ../data/unita_raw/summary.json
```

### Step 2: Repeat for Other Areas
```bash
# CHUG
python extract_polygon_images.py \
  --layers ../data/chug_polygons/ML_labeling_CHUG.gpkg \
  --ortho ../data/chugchug_raw/CerroChug_ortomosaico.tif \
  --output ../data/chugchug_raw

# LLUTA
python extract_polygon_images.py \
  --layers ../data/lluta_polygons/ML_labeling_LLUTA.gpkg \
  --ortho ../data/lluta_raw/CerroLluta_ortomosaico.tif \
  --output ../data/lluta_raw
```

### Step 3: Analyze All Datasets
```bash
python analysis.py
```

This will:
- Compare dimensions across UNITA, CHUG, and LLUTA
- Suggest optimal sliding window sizes for each area
- Display statistical summaries and visualizations

## Output Metadata Format

Each polygon generates a metadata JSON file with the following structure:

```json
{
  "polygon_index": 5,
  "class": 1,
  "bounds": {
    "minx": 535643.2,
    "miny": 6234567.8,
    "maxx": 535655.5,
    "maxy": 6234579.1
  },
  "bbox_size_meters": {
    "width_m": 12.3,
    "height_m": 11.4,
    "area_m2": 140.22
  },
  "crs": "EPSG:32719",
  "image_shape": {
    "height": 450,
    "width": 485,
    "channels": 4
  },
  "files": {
    "ortho_tif": "geoglif_0005_ortho.tif",
    "ortho_jpeg": "geoglif_0005_ortho.jpg",
    "overlay_jpeg": "geoglif_0005_overlay.jpg"
  }
}
```

## Troubleshooting

**Issue: CRS mismatch between geopackage and orthomosaic**
- Solution: The script automatically reprojects the geodataframe to match the orthomosaic CRS. Check console output for "Converting from X to Y" message.

**Issue: Some polygons show as processed but files don't exist**
- Solution: Check the error messages in console. Common causes:
  - Polygon bounding box extends outside orthomosaic bounds
  - CRS transformation errors
  - Insufficient disk space

**Issue: Analysis script shows "No module named 'module_name'"**
- Solution: Ensure all packages in requirements.txt are installed:
  ```bash
  pip install -r ../requirements.txt
  ```

**Issue: Memory error when processing many polygons**
- Solution: Use the `--limit` parameter to process in batches:
  ```bash
  python extract_polygon_images.py --limit 100
  ```

## Notes

- The `extract_polygon_images.py` script handles both simple Polygon and MultiPolygon geometries
- Polygon overlay images include 5% padding around the bounding box for context
- All coordinates are preserved in their original CRS and stored in metadata
- The analysis script uses outlier removal (IQR method) to avoid skewing statistics with anomalies
- Pixel scale is automatically calculated from image dimensions and real-world coordinates
