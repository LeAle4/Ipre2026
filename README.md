# Geoglyph Dataset Processing Pipeline

A Python-based toolkit for extracting, processing, and managing geoglyph polygon data from geo-referenced orthomosaic imagery across multiple archaeological study areas in northern Chile.

## Overview

Pipeline for three study areas: **Cerro Unita**, **ChugChug**, **Lluta**. It extracts polygon imagery from orthomosaics, resizes, and generates crops with metadata.

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
│   ├── extract.py            # Polygon extraction from raw data
│   ├── resize.py             # Batch polygon resizing tool
│   ├── crop.py               # Sliding-window crops from resized polygons
│   ├── negatives.py          # Negative sample generation
│   └── pipeline.py           # Complete processing pipeline (runs all steps)
│
├── visualization/             # Visualization and analysis tools
│   ├── viewer.py             # (Placeholder for data viewer)
│   └── data_properties.py    # Commented analysis utilities
│
├── tests/                     # Unit tests directory
│
├── handle.py                  # Project configuration + shared helpers
├── text.py                    # Small CLI formatting helpers
├── utils.py                   # Polygon data model + coordinate helpers
└── README.md                  # This file
```

## Core Components

- **handle.py**: Project configuration - paths, constants, scales, and shared helpers
- **utils.py**: Polygon data model and coordinate transformation helpers
- **text.py**: CLI formatting utilities (`title`, `tabbed`)

## Dataset Processing Tools

The `dataset/` folder contains five processing scripts that can be used individually or together via the pipeline.

### Quick Start: pipeline.py (Recommended)

Run the complete processing pipeline for one or more study areas:

```bash
# Process all steps (extract → resize → crop → negatives) for one area
python dataset/pipeline.py --area unita

# Process multiple areas at once
python dataset/pipeline.py --area unita chugchug lluta

# Run only specific steps
python dataset/pipeline.py --area chugchug --steps extract resize
python dataset/pipeline.py --area lluta --steps crop negatives
```

**Available steps**: `extract`, `resize`, `crop`, `negatives`

---

### Individual Tools

#### 1. extract.py - Polygon Extraction

Extracts polygon crops from orthomosaics based on labeled geopackage files. Outputs geo-referenced TIF, JPEG, overlay images, and metadata.

```bash
# Extract all polygons for one or more areas
python dataset/extract.py --area unita
python dataset/extract.py --area unita chugchug lluta

# Extract only first 10 geoglyphs (class 1)
python dataset/extract.py --area chugchug --limit 10 --class-filter 1
```

**Output** → `data/{area}_polygons/`
- `geoglif_0000_ortho.tif` - Geo-referenced TIF
- `geoglif_0000_ortho.jpg` - Standard JPEG
- `geoglif_0000_overlay.jpg` - JPEG with polygon overlay
---

#### 2. resize.py - Polygon Resizing

Resizes extracted polygons to standardized dimensions using Lanczos interpolation. Updates metadata to reflect new dimensions.

```bash
# Resize polygons for single or multiple areas
python dataset/resize.py --area unita
python dataset/resize.py --area unita chugchug lluta
```

**Output** → `data/{area}_resized/`
- `{area}_class1_{id}_resized.png` - Resized polygon images

---

#### 3. crop.py - Crop Generation

Generates fixed-size sliding-window crops from resized polygons. Filters crops by geoglyph content coverage using the threshold defined in `handle.py`.

```bash
# Generate crops for one or more areas
python dataset/crop.py --area unita
python dataset/crop.py --area unita chugchug lluta
```

**Output** → `data/{area}_crops/geo_{id}/`
- `{area}_class1_{id}_crop{n}.png` - Individual crop images

---

#### 4. negatives.py - Negative Sample Generation

Extracts negative samples (areas without geoglyphs) from the orthomosaics. Ensures no overlap with existing geoglyph polygons.

```bash
# Generate negative samples for one or more areas
python dataset/negatives.py --area unita
python dataset/negatives.py --area unita chugchug lluta
```

**Output** → `data/{area}_negatives/`
- `{area}_class2_crop{n}_0.png` - Negative sample images

## Configuration Guide: handle.py

All dataset processing behavior is controlled through [handle.py](handle.py). This centralized configuration makes it easy to adapt the pipeline to your needs.

### Directory Structure Configuration

Update these paths if your project structure changes:

```python
# Main project paths
PROJECT_PATH = Path(__file__).resolve().parent
DATA_DIR = PROJECT_PATH / "data"

# Add new study areas by creating path dictionaries
NEW_AREA_PATHS = {
    "raw": DATA_DIR / "newarea_raw",
    "polygons": DATA_DIR / "newarea_polygons",
    "summary": DATA_DIR / "newarea_polygons" / "summary.json",
    "resized": DATA_DIR / "newarea_resized",
    "crops": DATA_DIR / "newarea_crops",
    "negatives": DATA_DIR / "newarea_negatives",
}

# Register the new area in PATHS dictionary
PATHS = {
    "unita": UNITA_PATHS,
    "chugchug": CHUGCHUG_PATHS,
    "lluta": LLUTA_PATHS,
    "newarea": NEW_AREA_PATHS,  # Add your new area here
}
```

### Image Analysis Parameters

Adjust these constants to change how images are processed:

```python
# Image scale factors (meters per pixel) for each study area
# Used to calculate real-world dimensions
SCALES = {
    'unita': 0.886,      # Adjust based on your orthomosaic resolution
    'lluta': 0.218,
    'chugchug': 0.18
}

# Crop window parameters
WINDOW_SIZE = 224              # Size of each crop in pixels (change for different input sizes)
STRIDE = int(WINDOW_SIZE / 2)  # Step size for sliding window (reduce for more overlap)

# Quality filtering
THRESHOLD_CROP_CONTENT = 0.8   # Min fraction of geoglyph pixels in crop (0.0-1.0)
                               # Higher = stricter filtering, fewer but better crops
                               # Lower = more crops but may include partial geoglyphs

# Negative sampling
NEGATIVES_RATIO = 3            # Number of negative samples per positive crop
                               # Increase for more balanced datasets
```

### Class Labels

```python
# Polygon class mappings
CLASSES = {
    "geo": 1,      # Geoglyphs (positive samples)
    "ground": 2,   # Ground/negative samples
    "road": 3      # Roads (if applicable)
}
```

### Quick Configuration Examples

**Example 1: Generate smaller crops with more overlap**
```python
WINDOW_SIZE = 128
STRIDE = 64  # 50% overlap
```

**Example 2: More lenient filtering for sparse geoglyphs**
```python
THRESHOLD_CROP_CONTENT = 0.5  # Accept crops with 50%+ coverage
```

**Example 3: Balance dataset with more negatives**
```python
NEGATIVES_RATIO = 5  # 5 negative samples per positive
```

## Notes

- Summary JSON files are compatible with the Polygon model in [utils.py](utils.py)
