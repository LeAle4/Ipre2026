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
│   ├── crop.py               # Sliding-window crops from resized polygons
│   └── resize.py             # Batch polygon resizing tool
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

- handle.py: paths, constants, shared helpers
- utils.py: Polygon model + coordinate helpers
- text.py: CLI formatting (`title`, `tabbed`)

### dataset/extract.py

Extracts polygon crops from orthomosaics, writes TIF/JPG/overlay plus metadata.

**Command-line Interface**:
```bash
# Extract all polygons for one or more areas
python dataset/extract.py --area unita
python dataset/extract.py --area unita chugchug lluta

# Extract only first 10 geoglyphs (class 1)
python dataset/extract.py --area chugchug --limit 10 --class-filter 1
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

---

### dataset/resize.py

Resizes extracted polygons using LCI interpolation and updates metadata.

**Command-line Interface**:
```bash
# Resize polygons for a single area
python dataset/resize.py --area unita

# Resize polygons for multiple areas
python dataset/resize.py --area unita chugchug lluta
```

**Output Structure**:
```
data/{area}_resized/
├── unita_class1_0_resized.png
├── unita_class1_1_resized.png
├── unita_class1_2_resized.png
└── ...
```

---

### dataset/crop.py

Generates fixed-size crops from resized polygons and filters by polygon coverage.

**Command-line Interface**:
```bash
python dataset/crop.py --area unita
python dataset/crop.py --area unita chugchug lluta
```

---

---

### 1. Data Extraction
Extract polygon images from raw data:
```bash
python dataset/extract.py --area unita
```

### 2. Data Resizing
Standardize polygon image dimensions using high-quality interpolation:
```bash
python dataset/resize.py --area unita chugchug lluta
```

### 3. Crop Generation
```bash
python dataset/crop.py --area unita
```

## Adapting to Your File Structure

If you move raw or processed folders, update the path configuration in handle.py.

Key items to adjust in [handle.py](handle.py):
- `PROJECT_PATH` and `DATA_DIR` if the project or data root changes.
- Area mappings in `UNITA_PATHS`, `CHUGCHUG_PATHS`, `LLUTA_PATHS` if folder names change.
- `PATHS` is the single source of truth for `raw`, `polygons`, `resized`, and `crops` locations.

Once handle.py reflects your structure, the dataset scripts will automatically read from the correct raw data folders and write outputs to the correct processed folders.

## Notes

- Summary JSON files are compatible with the Polygon model in [utils.py](utils.py)
