# Geoglif Image Extraction and Analysis

Tools for extracting geo-referenced crops from orthomosaics, organizing outputs, and analyzing polygon dimensions across areas.

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
    ├── format.py                # Cropping utilities for fixed/random/polygon crops
    ├── extract_polygon_images.py
    ├── analysis.py
    ├── geoglyph_viewer.ipynb    # Interactive notebook to browse metadata and crops
    ├── analysis.ipynb           # Notebook version of the analysis workflow
    ├── extract.ipynb            # Notebook version of the extraction workflow
    └── README.md
```

## Overview

### handle.py
Centralized paths and dataset config used by every script. Exposes common directories, the `DATASETS` dictionary, and helpers: `load_json`, `save_json`, `get_dataset_info`, `get_all_datasets`.

### format.py
Cropping/padding utilities for training windows:
- `make_random_crops`, `make_fixed_crops`, `make_polygon_thresholds_crops` generate crops.
- `crop_image` returns crop + mask; `fill_with_noise` fills padded regions (Gaussian/uniform/blurred noise).

Quick example:
```python
from format import make_fixed_crops, fill_with_noise
crops = make_fixed_crops(img, window_size=512, n_crops=16, stride=256)
crops = [(fill_with_noise(c, m, noise_level=0.2), m) for c, m in crops]
```

### extract_polygon_images.py
CLI to extract per-polygon ortho crops from a geopackage + orthomosaic. Produces GeoTIFF + JPEG crops, overlay JPEG, per-polygon metadata JSONs, and a `summary.json` in the output folder.

### data_create.py
CLI to generate random, fixed-grid, or polygon-threshold crops using a single script and CLI arguments.

### analysis.py
Loads summaries (UNITA/CHUG/LLUTA) via `handle.py`, filters class 1 (geo) polygons, removes outliers, reports width/height/area stats, estimates pixel scale, and plots histograms/box plots. Can also copy geo-class images into organized folders.

### geoglyph_viewer.ipynb
Interactive ipywidgets notebook to browse metadata JSONs, view overlay/ortho images, and visualize random/fixed/polygon-threshold crops using `format.py`. Steps: open notebook (bootstraps ipywidgets/Pillow/numpy), set a metadata path (e.g., `test_geos/unita_geoglif_0000_metadata.json`), choose image type + crop method, click **Load Geoglyph**.

### analysis.ipynb
Notebook companion to `analysis.py` for interactive exploration of dimensions and plots.

### extract.ipynb
Notebook companion to `extract_polygon_images.py` for running extractions interactively and inspecting outputs.

## Installation

```
pip install -r ../requirements.txt
```

Key packages: geopandas, rasterio, matplotlib, numpy, shapely, pyproj, ipywidgets (viewer).

## Usage (quick start)

### Extract polygon images
```
cd dataset
python extract_polygon_images.py \
  --layers ../data/unita_polygons/ML_labeling_UNITA.gpkg \
  --ortho ../data/unita_raw/CerroUnita_ortomosaico.tif \
  --output ../data/unita_raw \
  --limit 50   # optional for a quick test
```
Key flags: `--layers` (gpkg), `--ortho` (tif), `--output` (folder), `--limit` (subset). Outputs crops + overlays + metadata JSONs + `summary.json`.

### Analyze geo dimensions
```
cd dataset
python analysis.py
```
Reads summaries, filters class 1, removes outliers, prints width/height/area stats, estimates pixel scale, and shows histograms/box plots.

### Generate crops (single script)
Use `data_create.py` instead of the old test scripts.

Random crops:
```
cd dataset
python data_create.py --area lluta --strategy random --n-crops 100 --window-size 256 --output ../data/crops_output
```

Fixed-grid crops:
```
cd dataset
python data_create.py --area chugchug --strategy fixed --n-crops 50 --window-size 256 --stride 128 --output ../data/crops_output
```

Polygon-threshold crops:
```
cd dataset
python data_create.py --area unita --strategy polygon --n-crops 30 --window-size 256 --threshold 0.3 --output ../data/crops_output
```

Batch mode (multiple window sizes / noise levels):
```
cd dataset
python data_create.py --area lluta --strategy fixed --n-crops 50 --batch \
  --batch-window-sizes 256 512 \
  --batch-noise-levels 0.0 0.5 1.0 \
  --batch-noise-types gaussian uniform \
  --batch-strides auto 128 256 \
  --output ../data/crops_output
```

### View geoglyphs and crops
Open `geoglyph_viewer.ipynb`, set a metadata JSON path, pick image type and crop method, then click **Load Geoglyph** to compare original and generated crop grids.

## Customization (quick pointers)

- `handle.py`: add new areas by extending `DATASETS`; reuse `load_json`/`save_json` in your own scripts.
- `extract_polygon_images.py`: adjust padding, output driver, or class filter in the main loop; use `--limit` for batches.
- `analysis.py`: tweak IQR multiplier or plotting colors/bins in the helpers.

## Minimal workflow

1) Extract: run `extract_polygon_images.py` with `--layers`, `--ortho`, `--output` (add `--limit` to sample).
2) View: open `geoglyph_viewer.ipynb`, point to a metadata JSON, explore crops.
3) Analyze: run `analysis.py` for stats/plots and optional organized geo copies.

## Metadata JSON (shape)

Each polygon gets a metadata JSON with bounds, bbox size (m), CRS, image shape, and file names for ortho/overlay.

## Notes / tips

- Install deps once: `pip install -r ../requirements.txt`.
- Use `--limit` on extraction for quick trials.
- If CRS differs, the extractor reprojects automatically.
