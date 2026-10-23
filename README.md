# Geoglyph Dataset Processing Pipeline

A Python-based toolkit for extracting, standardizing, and generating machine learning ready datasets from geo-referenced orthomosaic imagery and polygon vector labels of archaeological geoglyphs across study areas in northern Chile (e.g., **Cerro Unita**, **ChugChug**, **Lluta**).

## Overview

The processing pipeline ingests raw orthomosaic TIF files and GeoPackage vector label files (`.gpkg`), standardizes spatial resolution to a target metric scale (m/px) using area-weighted downsampling or Lagrange-Chebyshev Interpolation (LCI), extracts integer grid-aligned sliding-window positive crops over geoglyph polygons, samples non-overlapping negative background patches, and exports ML-ready image files along with rich site-level and crop-level metadata and vector geometries.

---

## Project Structure

```
.
├── ML/                         # Generated ML-ready output dataset directory
│   ├── sites_metadata.csv      # Global metadata summary across all study sites
│   ├── Unita/                  # ML dataset assets for site Unita
│   │   ├── image_metadata.csv  # Crop-level metadata for site Unita
│   │   ├── shape_geometries.gpkg   # Spatial geometries for generated individual crops
│   │   ├── polygon_geometries.gpkg # Spatial geometries for parent geoglyph polygons
│   │   └── Unita_*.png         # Extracted positive & negative PNG crops (e.g. 224x224)
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
├── extract.py                  # Extractor for polygon region bounding boxes and grid snapping
├── resize.py                   # Image resizer (Area-Weighted Averaging & LCI)
├── crop.py                     # Sliding-window crop generator with spatial intersection filtering
├── negatives.py                # Random spatial negative background sampler
├── parameters.py               # Global class constants and fallback defaults
└── README.md                   # Project documentation
```

---

## Quick Start: Running `pipeline.py`

The primary execution entrypoint for the dataset generation pipeline is [`pipeline.py`](pipeline.py).

### Running the Pipeline

Execute the pipeline from the project root directory using Python:

```bash
python pipeline.py
```

### Execution Flow of `pipeline.py`

When executed, [`pipeline.py`](pipeline.py):
1. **Loads Configuration**: Reads run parameters from [`pipeline_config.json`](pipeline_config.json) (`data_dir`, `target_scale`, `window_size`, `stride`, `threshold_crop_content`, and `negatives_ratio`).
2. **Site Discovery & CRS Check**: Scans `data_dir` and instantiates [`SiteData`](datamanager.py#L128) objects for each site subdirectory. Validates the Coordinate Reference System (CRS) against the expected metric projection (`EPSG:32719`, UTM Zone 19S) and extracts per-axis pixel scale (`PixelScale`).
3. **Site Iteration**:
   - Initializes a [`DataWriter`](datamanager.py#L290) session and runs memory-efficient block-wise channel statistics across raster blocks (`block_windows`) without loading entire multi-gigabyte rasters into memory.
   - Extracts geoglyph polygons via [`get_polygon_imgs`](extract.py#L111), snapping bounds to the integer crop grid (`window_size + k * stride`) and extracting raw unscaled source pixels.
   - Adds parent geoglyphs to `polygon_geometries.gpkg`.
   - Resizes extracted patches to `target_scale` using [`resize_datapoint_image`](resize.py#L170) (defaulting to area-weighted averaging).
   - Generates sliding-window crops with [`generate_crops`](crop.py#L38), filtering by spatial coverage threshold ($\ge \text{threshold\_crop\_content}$) while guaranteeing at least one crop per geoglyph.
   - Mines negative background crops via [`generate_negative_samples`](negatives.py#L85), validating against nodata areas and ensuring zero spatial overlap with positive geoglyphs.
4. **Export**: Saves all processed PNG crop images into `ML/<SiteName>/`, writes per-site `image_metadata.csv`, `shape_geometries.gpkg`, and `polygon_geometries.gpkg`, and compiles the global `ML/sites_metadata.csv`.

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

Data flows sequentially through six primary stages:

```
[Raw Site Data (TIF + GPKG)]
           │
           ▼
1. Site Discovery, CRS Validation & Blockwise Statistics (datamanager.py)
           │
           ▼
2. Integer Grid Snapping & Raw Pixel Extraction (extract.py)
           │
           ▼
3. Image Resizing (Area-Weighted Downsampling / LCI) (resize.py)
           │
           ▼
4. Sliding Window Crop Generation & Coverage Thresholding (crop.py)
           │
           ▼
5. Non-Overlapping Negative Sample Mining (negatives.py)
           │
           ▼
6. ML-Ready Dataset & Multi-Table Export (datamanager.py) ──► [ML Directory]
```

### Stage Details

1. **Site Discovery & CRS Validation ([`datamanager.py`](datamanager.py))**:
   - `get_sites()` inspects `data_dir` for site directories.
   - `validate_crs()` checks that site orthomosaics match the expected metric projection (`TARGET_UTM_EPSG = 32719`, WGS 84 / UTM Zone 19S) and warns if geographic CRS or alternate datums are present.
   - `get_pixel_scale()` preserves independent X and Y pixel dimensions via `PixelScale(x, y)`, computing latitude-corrected ground dimensions if geographic.
   - Reprojects vector label GeoDataFrames to the raster CRS on load if necessary.
   - Calculates per-channel minimum, maximum, and average using block-by-block streaming (`tif_area.block_windows()`) to prevent out-of-memory errors on massive orthomosaics.

2. **Polygon Extent Extraction & Integer Grid Snapping ([`extract.py`](extract.py))**:
   - Filters GeoPackage vector labels for positive geoglyph polygons (`GEO_CLASS = 1`).
   - `snap_bounds_integer()` converts ground coordinates to target pixel coordinates (`coord / desired_scale`), applies `math.floor` to minimums and `math.ceil` to maximums, and snaps pixel dimensions to `window_size + k * stride` ($k \ge 0$).
   - Symmetrically distributes residual padding so the geoglyph remains centered within the snapped bounding box.
   - Extracts raw unscaled source pixels directly from the TIF raster using `rasterio.windows.from_bounds` without truncating pixel dimensions.
   - Transposes image channels to standard HWC (`Height, Width, Channels`) numpy arrays.

3. **Image Resizing & Precondition Validation ([`resize.py`](resize.py))**:
   - `resize_datapoint_image()` computes target pixel dimensions directly from `datapoint.image_box.bounds` and strictly verifies that dimensions are already aligned with the crop grid (`(dim - window_size) % stride == 0`).
   - **Area-Weighted Downsampling (`area_weighted_resize`)**: The default resampling method. Each output pixel represents the area-weighted average of source pixels it overlaps, preserving radiometric fidelity and avoiding Gibbs ringing artifacts on orthomosaic imagery.
   - **Lagrange-Chebyshev Interpolation (`lci`)**: An alternative high-order polynomial image interpolation method utilizing Inverse Discrete Cosine Transforms (IDCT) for sharp resampling.
   - Updates `datapoint.modify_image()` and symmetrically updates `datapoint.image_box` via `update_image_box()` keeping the spatial centroid fixed.

4. **Sliding-Window Crop Generation ([`crop.py`](crop.py))**:
   - Generates geographic crop boundaries in row-major order (top-to-bottom, left-to-right) via `calculate_crop_boundaries()`.
   - Applies row-major 2D sliding window (`view_as_windows`) of size `window_size` $\times$ `window_size` with step size `stride`.
   - Computes spatial intersection between each crop bounding box and the labeled geoglyph polygon geometry.
   - Yields crops whose intersection ratio ($\frac{\text{Area}(\text{Polygon} \cap \text{Crop})}{\text{Area}(\text{Crop})}$) meets or exceeds `threshold_crop_content`.
   - Guarantees at least one crop is retained per geoglyph (yielding the crop with maximum intersection ratio if none meet the threshold).

5. **Negative Sample Mining ([`negatives.py`](negatives.py))**:
   - Calculates target number of negative background samples ($N_{\text{neg}} = N_{\text{pos}} \times \text{negatives\_ratio}$).
   - Randomly samples raw spatial candidate windows across the site orthomosaic in source pixel space.
   - Validates candidate windows using two criteria:
     1. `in_actual_data()`: Ensures $\ge 99\%$ of window pixels are valid non-nodata data (evaluating dataset `nodata` or uint8 extremes).
     2. `not_in_positive_polygons()`: Verifies zero spatial intersection with all positive geoglyph geometries.
   - Resizes valid background patches to `target_scale` and outputs negative `DataPoint` crops (`DATA_LABEL = 2`).

6. **ML Asset & Multi-Table Export ([`datamanager.py`](datamanager.py))**:
   - Saves crop images as PNG files to `ML/<SiteName>/<ID>.png`.
   - Exports per-site crop metadata to `ML/<SiteName>/image_metadata.csv`.
   - Exports per-site crop vector geometries to `ML/<SiteName>/shape_geometries.gpkg`.
   - Exports per-site parent geoglyph vector geometries to `ML/<SiteName>/polygon_geometries.gpkg`.
   - Compiles global site spatial and multi-channel statistics into `ML/sites_metadata.csv`.

---

## ML-Ready Output Folder Structure

After running `pipeline.py`, the dataset is organized in the `ML/` directory as follows:

```
ML/
├── sites_metadata.csv                  # Aggregate site-level metadata CSV
│
├── Unita/                              # Study Area 1
│   ├── image_metadata.csv              # Crop-level metadata CSV for Unita
│   ├── shape_geometries.gpkg           # Spatial GeoPackage of individual crop bounding boxes
│   ├── polygon_geometries.gpkg         # Spatial GeoPackage of parent geoglyph polygons
│   ├── Unita_02413_0.png               # Positive crop (Site_GeoID_CropID)
│   ├── Unita_02413_1.png               # Positive crop from same parent polygon
│   ├── ...
│   ├── Unita_02420_1.png               # Negative background crop sample
│   └── ...
│
├── ChugChug/                           # Study Area 2
│   ├── image_metadata.csv
│   ├── shape_geometries.gpkg
│   ├── polygon_geometries.gpkg
│   └── ChugChug_*.png
│
└── Lluta/                              # Study Area 3
    ├── image_metadata.csv
    ├── shape_geometries.gpkg
    ├── polygon_geometries.gpkg
    └── Lluta_*.png
```

### Image Naming Convention

Each image saved in `ML/<SiteName>/` follows the standardized format:
`{SITE_NAME}_{GEO_ID}_{CROP_ID}.png`

- **`SITE_NAME`**: Name of the site (e.g. `Unita`).
- **`GEO_ID`**: 5-digit zero-padded index of the parent polygon / sampled datapoint (e.g. `02413`).
- **`CROP_ID`**: Crop index within the parent datapoint (e.g. `0`, `1`, `2`...).

---

## Metadata Reference & Summaries

The pipeline produces tabular metadata and vector spatial layers:

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
| `CROP_ID` | `string` | Sequential crop index within parent datapoint (5-digit padded integer, e.g. `00000`). |
| `DATA_LABEL` | `integer` | Numerical class label: `1` = Geoglyph (Positive), `2` = Ground (Negative). |
| `SCALE_MPX` | `float` | Standardized spatial scale of crop image in meters per pixel (e.g., `0.05`). |
| `CENTER_X` | `float` | X coordinate of the crop center in CRS units. |
| `CENTER_Y` | `float` | Y coordinate of the crop center in CRS units. |
| `AREA` | `float` | Spatial ground area of the crop in square meters. |
| `CHANNELS` | `integer` | Number of color/spectral channels in crop image. |
| `CRS` | `string` | Coordinate Reference System of the crop spatial bounds. |
| `THRESHOLD_CLEAR` | `float` | Fraction ($0.0 \text{--} 1.0$) of geoglyph polygon spatial coverage inside the crop window. Set to `1.0` for negative ground samples. |

---

### 3. Vector Geometries: `shape_geometries.gpkg` and `polygon_geometries.gpkg`

Located in each site folder `ML/<SiteName>/`:
- **`shape_geometries.gpkg`**: Contains the bounding box polygon geometries for each generated individual crop, along with crop IDs, centroid coordinates, and intersection proportions.
- **`polygon_geometries.gpkg`**: Contains the original vector polygon geometries of each labeled geoglyph (`GEO_ID`), preserving the ground-truth archaeological footprint for spatial analysis and GIS mapping.

---

## Technical Core Components

- **[`datamanager.py`](datamanager.py)**: Central data abstractions (`SiteData`, `LabelTable`, `DataPoint`, `Crop`, `DataWriter`), metric CRS validation, anisotropic pixel scale tracking, and windowed channel statistics.
- **[`extract.py`](extract.py)**: Bounding extent calculation with integer grid expansion (`snap_bounds_integer`) and raw TIF raster extraction.
- **[`resize.py`](resize.py)**: Image scaling via Area-Weighted Downsampling (`area_weighted_resize`) and Lagrange-Chebyshev Interpolation (`lci`), with crop-grid alignment assertion.
- **[`crop.py`](crop.py)**: Row-major 2D sliding-window generation and Shapely polygon intersection analysis.
- **[`negatives.py`](negatives.py)**: Random spatial background sampling with nodata validation and positive polygon non-overlap checks.
- **[`parameters.py`](parameters.py)**: Default parameters and class mapping constants (`GEO_CLASS = 1`, `GROUND_CLASS = 2`, `ROAD_CLASS = 3`).
