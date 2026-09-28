# Geoglyph Dataset Processing Pipeline

A Python-based toolkit for extracting, standardizing, and generating machine learning ready datasets from geo-referenced orthomosaic imagery and polygon vector labels of archaeological geoglyphs across study areas in northern Chile (e.g., **Cerro Unita**, **ChugChug**, **Lluta**).

## Overview

The processing pipeline ingests raw orthomosaic TIF files and GeoPackage vector label files (`.gpkg`), standardizes spatial resolution using Lagrange-Chebyshev Interpolation (LCI), extracts sliding-window positive crops over geoglyph polygons, samples non-overlapping negative background patches, and exports ML-ready image files along with rich site-level and crop-level metadata.

---

## Project Structure

```
.
├── ML/                         # Generated ML-ready output dataset directory
│   ├── sites_metadata.csv      # Global metadata summary across all study sites
│   ├── Unita/                  # ML dataset assets for site Unita
│   │   ├── image_metadata.csv  # Crop-level metadata for site Unita
│   │   ├── shape_geometries.gpkg # Spatial geometries for generated crops
│   │   └── Unita_*.png         # Extracted positive & negative PNG crops (224x224)
│   ├── ChugChug/               # ML dataset assets for site ChugChug
│   └── Lluta/                  # ML dataset assets for site Lluta
│
├── data/                       # Raw input data directory (grouped by site folder)
│   ├── Unita/                  # Unita.tif, Unita_label.gpkg, Unita_DEM.tif
│   ├── ChugChug/               # ChugChug.tif, ChugChug_label.gpkg, ChugChug_DEM.tif
│   └── Lluta/                  # Lluta.tif, Lluta_label.gpkg
│
├── pipeline.py                 # Pipeline execution entrypoint script
├── pipeline_config.json        # Centralized JSON configuration for pipeline runs
├── datamanager.py              # Data structures (SiteData, DataPoint, DataWriter) & IO
├── extract.py                  # Extractor for polygon region bounding boxes
├── resize.py                   # Lagrange-Chebyshev Interpolation (LCI) image resizer
├── crop.py                     # Sliding-window crop generator with spatial filtering
├── negatives.py                # Random spatial negative sample sampler
├── parameters.py               # Global class constants and fallback defaults
│
├── training/                   # Model training and evaluation scripts
│   └── train.py                # Model training script
└── visualization/              # Dataset exploration and plotting tools
    └── data_properties.py      # Statistical and visual analysis tools
```

---

## Quick Start: Running `pipeline.py`

The primary execution point for the dataset generation pipeline is [`pipeline.py`](pipeline.py).

### Running the Pipeline

Execute the pipeline from the project root directory using Python:

```bash
python pipeline.py
```

### Execution Flow of `pipeline.py`
When executed, [`pipeline.py`](pipeline.py):
1. **Loads Configuration**: Reads run parameters from [`pipeline_config.json`](pipeline_config.json) (data directory path, scale, crop window size, stride, content threshold, and negatives ratio).
2. **Site Discovery**: Scans `data_dir` and automatically instantiates [`SiteData`](datamanager.py#L66) objects for all subdirectories containing orthomosaic TIFs and GeoPackage labels.
3. **Site Iteration**: Loops through each site, initializes a [`DataWriter`](datamanager.py#L177) session, extracts geoglyph polygons, resizes them, generates sliding-window positive crops, and samples negative ground crops.
4. **Export**: Saves all processed PNG crop images into `ML/<SiteName>/`, exports per-site `image_metadata.csv` and `shape_geometries.gpkg`, and writes the aggregate global `ML/sites_metadata.csv`.

---

## Configuration Guide: `pipeline_config.json`

All pipeline execution parameters are centralized in [`pipeline_config.json`](pipeline_config.json).

### File Format

```json
{
    "data_dir": "data",
    "target_scale": 0.05,
    "window_size": 224,
    "stride": 112,
    "threshold_crop_content": 0.25,
    "negatives_ratio": 3
}
```

### Parameter Reference

| Parameter Key | Data Type | Description | Default / Typical Value |
| :--- | :--- | :--- | :--- |
| `data_dir` | `string` | Relative path from project root to the input data directory containing study site folders. | `"data"` |
| `target_scale` | `float` | Target spatial resolution in meters per pixel (m/px). Standardizes orthomosaic patches across sites of varying original resolutions. | `0.05` |
| `window_size` | `integer` | Pixel width and height ($W \times H$) for generated square crops (e.g. `224` for $224 \times 224$ neural network input). | `224` |
| `stride` | `integer` | Step size in pixels for sliding window during crop generation (e.g., `112` yields 50% overlap). | `112` |
| `threshold_crop_content` | `float` | Minimum ratio ($0.0 \text{--} 1.0$) of geoglyph polygon spatial coverage within a crop for it to be accepted as a positive sample. | `0.25` (25% coverage) |
| `negatives_ratio` | `integer` / `float` | Multiplier determining the number of negative background crop samples generated per positive sample ($N_{\text{neg}} = N_{\text{pos}} \times \text{negatives\_ratio}$). | `3` |

### Modifying Configuration

- **Higher Overlap Crops**: Reduce `stride` (e.g., `"stride": 56` for 75% window overlap).
- **Stricter Geoglyph Filtering**: Increase `"threshold_crop_content": 0.50` so crops must contain at least 50% geoglyph area.
- **Larger Model Input Sizes**: Update `"window_size": 512` and `"stride": 256` for larger spatial contextual inputs.

---

## End-to-End Data Processing Pipeline Overview

Data flows sequentially through six main stages:

```
[Raw Site Data (TIF + GPKG)]
           │
           ▼
1. Site Discovery & Spatial Metadata Parsing (datamanager.py)
           │
           ▼
2. Polygon Extent Target Expansion (extract.py)
           │
           ▼
3. Lagrange-Chebyshev Resizing & Grid Snapping (resize.py)
           │
           ▼
4. Sliding Window Crop Generation & Coverage Thresholding (crop.py)
           │
           ▼
5. Non-Overlapping Negative Sample Mining (negatives.py)
           │
           ▼
6. ML-Ready Dataset & Metadata Export (datamanager.py) ──► [ML Directory]
```

### Stage Details

1. **Site Discovery & Spatial Metadata Parsing ([`datamanager.py`](datamanager.py))**:
   - `get_sites()` inspects `data_dir` for site directories.
   - Parses spatial metadata from orthomosaic headers (`CRS`, bounding box, Affine transform, channel count, nodata value).
   - Determines native spatial resolution (`m_px`) and calculates ground dimensions in meters.

2. **Polygon Extent Extraction ([`extract.py`](extract.py))**:
   - Filters GeoPackage vector labels for positive geoglyph polygons (`GEO_CLASS = 1`).
   - For each polygon, `scale_bounds()` calculates expanded real-world ground boundaries in meters centered around the geoglyph so that after scaling to `target_scale`, the pixel dimensions perfectly fit `window_size + k * stride`.
   - Extracts raw unscaled source pixels directly from the TIF raster.

3. **LCI Image Resizing ([`resize.py`](resize.py))**:
   - Resizes source patch imagery to `target_scale` (m/px) using **Lagrange-Chebyshev Interpolation (LCI)** (`lci()`), avoiding traditional bilinear blurring artifacts.
   - `snap_to_crop_grid()` snaps the target pixel dimensions to exact grid multiples compatible with sliding-window sliding views.

4. **Sliding-Window Crop Generation ([`crop.py`](crop.py))**:
   - Applies row-major 2D sliding window (`view_as_windows`) of size `window_size` $\times$ `window_size` with step size `stride`.
   - Computes spatial intersection between each crop bounding box and the labeled geoglyph polygon geometry.
   - Yields crops whose intersection ratio ($\frac{\text{Area}(\text{Polygon} \cap \text{Crop})}{\text{Area}(\text{Crop})}$) meets or exceeds `threshold_crop_content`.
   - Guarantees at least one crop is retained per geoglyph (yielding the crop with maximum intersection ratio if none meet the threshold).

5. **Negative Sample Mining ([`negatives.py`](negatives.py))**:
   - Calculates target number of negative background samples ($N_{\text{neg}} = N_{\text{pos}} \times \text{negatives\_ratio}$).
   - Randomly samples raw spatial candidate windows across the site orthomosaic.
   - Validates candidate windows using two criteria:
     1. `in_actual_data()`: Ensures $\ge 99\%$ of window pixels are valid non-nodata data.
     2. `not_in_positive_polygons()`: Verifies zero spatial intersection with all positive geoglyph geometries.
   - Resizes valid background patches to `target_scale` and outputs negative `DataPoint` crops (`DATA_LABEL = 2`).

6. **ML Asset & Metadata Export ([`datamanager.py`](datamanager.py))**:
   - Saves crop images as PNG files to `ML/<SiteName>/<ID>.png`.
   - Exports per-site crop metadata to `ML/<SiteName>/image_metadata.csv` and vector geometries to `ML/<SiteName>/shape_geometries.gpkg`.
   - Compiles global site spatial and channel statistics into `ML/sites_metadata.csv`.

---

## ML-Ready Output Folder Structure

After running `pipeline.py`, the dataset is organized in the `ML/` directory as follows:

```
ML/
├── sites_metadata.csv                  # Aggregate site-level metadata CSV
│
├── Unita/                              # Study Area 1
│   ├── image_metadata.csv              # Crop-level metadata CSV for Unita
│   ├── shape_geometries.gpkg           # Spatial GeoPackage containing crop boundaries
│   ├── Unita_00000_1.png               # Positive crop (Site_GeoID_CropID)
│   ├── Unita_00000_2.png               # Positive crop 2 from same polygon
│   ├── ...
│   ├── Unita_00045_1.png               # Negative background crop sample
│   └── ...
│
├── ChugChug/                           # Study Area 2
│   ├── image_metadata.csv
│   ├── shape_geometries.gpkg
│   └── ChugChug_*.png
│
└── Lluta/                              # Study Area 3
    ├── image_metadata.csv
    ├── shape_geometries.gpkg
    └── Lluta_*.png
```

### Image Naming Convention

Each image saved in `ML/<SiteName>/` follows the standardized format:
`{SITE_NAME}_{GEO_ID}_{CROP_ID}.png`

- **`SITE_NAME`**: Name of the site (e.g. `Unita`).
- **`GEO_ID`**: 5-digit zero-padded index of the parent polygon / sampled datapoint (e.g. `00000`).
- **`CROP_ID`**: 5-digit zero-padded crop index within the parent datapoint (e.g. `00001`).

---

## Metadata Reference & Summaries

The pipeline produces two CSV metadata tables: global [`sites_metadata.csv`](datamanager.py#L179) and per-site [`image_metadata.csv`](datamanager.py#L207).

### 1. `sites_metadata.csv` (Global Summary)

Located at `ML/sites_metadata.csv`. Contains one record per study area summarizing spatial dimensions, class counts, sample totals, and multi-channel pixel statistics.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `SITE_NAME` | `string` | Name of the study area (e.g., `Unita`, `ChugChug`, `Lluta`). |
| `CRS` | `string` | Coordinate Reference System string (e.g., `EPSG:32719`). |
| `QUAD_AREA` | `float` | Total spatial area of the orthomosaic quadrant (in square meters or CRS units). |
| `QUAD_WIDTH_PX` | `float` | Width of original orthomosaic raster in pixels. |
| `QUAD_HEIGHT_PX` | `float` | Height of original orthomosaic raster in pixels. |
| `QUAD_WIDTH_M` | `float` | Width of study area in ground meters. |
| `QUAD_HEIGHT_M` | `float` | Height of study area in ground meters. |
| `SCALE_MPX` | `float` | Native spatial resolution of the original TIF (meters/pixel). |
| `NUMBER_OF_LABELS` | `integer` | Total vector label geometries present in site GeoPackage. |
| `NUMBER_OF_GEOGLYPHS` | `integer` | Count of positive geoglyph polygons (`class = 1`). |
| `NUMBER_OF_GROUND` | `integer` | Count of labeled ground polygons (`class = 2`). |
| `NUMBER_OF_IMAGES` | `integer` | Total ML crop images generated for this site (Positives + Negatives). |
| `NUMBER_OF_POSITIVES` | `integer` | Total positive geoglyph crop images generated. |
| `NUMBER_OF_NEGATIVES` | `integer` | Total negative background crop images sampled. |
| `CHANNELS` | `integer` | Number of image channels (e.g. `3` for RGB, `4` for RGBA/multispectral). |
| `CHANNEL_n_MIN` | `numeric` | Minimum pixel intensity value for channel $n$ across the orthomosaic. |
| `CHANNEL_n_MAX` | `numeric` | Maximum pixel intensity value for channel $n$ across the orthomosaic. |
| `CHANNEL_n_AVG` | `float` | Mean pixel intensity value for channel $n$ across the orthomosaic. |

---

### 2. `image_metadata.csv` (Crop-Level Summary)

Located at `ML/<SiteName>/image_metadata.csv`. Contains one record per generated PNG crop image.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `SITE_NAME` | `string` | Name of the study area to which the crop belongs. |
| `ID` | `string` | Unique full composite ID (`{SITE_NAME}_{GEO_ID}_{CROP_ID}`). Matches image filename `{ID}.png`. |
| `GEO_ID` | `string` | Parent polygon/datapoint ID (5-digit padded integer). |
| `CROP_ID` | `string` | Sequential crop index within parent datapoint (5-digit padded integer). |
| `DATA_LABEL` | `integer` | Numerical class label: `1` = Geoglyph (Positive), `2` = Ground (Negative). |
| `SCALE_MPX` | `float` | Standardized spatial scale of crop image in meters per pixel (e.g., `0.05`). |
| `CHANNELS` | `integer` | Number of color/spectral channels in crop image. |
| `CRS` | `string` | Coordinate Reference System of the crop spatial bounds. |
| `THRESHOLD_CLEAR` | `float` | Fraction ($0.0 \text{--} 1.0$) of geoglyph polygon spatial coverage inside the crop window. Set to `1.0` for negative ground samples. |

---

## Technical Core Components

- **[`datamanager.py`](datamanager.py)**: Central data abstractions (`SiteData`, `LabelTable`, `DataPoint`, `Crop`, `DataWriter`).
- **[`extract.py`](extract.py)**: Bounding extent calculation and TIF spatial window extraction.
- **[`resize.py`](resize.py)**: Image scaling using Lagrange-Chebyshev Interpolation (LCI) and crop grid snapping.
- **[`crop.py`](crop.py)**: 2D sliding-window generation and Shapely polygon intersection analysis.
- **[`negatives.py`](negatives.py)**: Random spatial background sampling with nodata and positive polygon non-overlap validation.
- **[`parameters.py`](parameters.py)**: Default parameters and class mapping constants (`GEO_CLASS = 1`, `GROUND_CLASS = 2`).

