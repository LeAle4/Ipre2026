# Raw Input Data Directory (`data/`)

This directory serves as the root container for raw, high-resolution geospatial inputs (orthomosaics, vector annotations, and digital elevation models) used by the geoglyph dataset processing pipeline.

> [!NOTE]
> This `README.md` file acts as a placeholder so the `data/` folder is tracked by version control. Large geospatial files (`.tif`, `.gpkg`) stored here are excluded from Git via `.gitignore` due to their multi-gigabyte sizes.

---

## Directory Organization

Raw inputs are organized strictly into **site-specific subdirectories**. Each site folder must be named with the exact identifier of the archaeological study area (e.g., `Unita`, `ChugChug`, `Lluta`).

```
data/
├── README.md                   # This placeholder and guide
├── Unita/                      # Study Site 1
│   ├── Unita.tif               # Raw orthomosaic raster (Required)
│   ├── Unita_label.gpkg        # Vector polygon labels (Required)
│   └── Unita_DEM.tif           # Digital elevation model raster (Optional)
├── ChugChug/                   # Study Site 2
│   ├── ChugChug.tif
│   ├── ChugChug_label.gpkg
│   └── ChugChug_DEM.tif
└── Lluta/                      # Study Site 3
    ├── Lluta.tif
    └── Lluta_label.gpkg
```

---

## File Requirements & Naming Conventions

The pipeline (`datamanager.py`) automatically discovers site directories and looks for specific filename patterns. For a site folder named `<SiteName>`:

### 1. Orthomosaic Raster (`<SiteName>.tif` or `.tiff`) — **Required**
- **Filename**: Must match the site folder name exactly: `<SiteName>.tif` (e.g. `Unita/Unita.tif`).
- **Format**: GeoTIFF (`.tif` or `.tiff`).
- **Coordinate Reference System (CRS)**: Must be a projected metric coordinate system, ideally **WGS 84 / UTM Zone 19S** (`EPSG:32719`).
- **Channels**: Multi-band imagery (typically 3-channel RGB or 4-channel RGBA / multispectral).
- **Data Integrity**: Should have valid spatial extent and resolution (`PixelScale`), with nodata values defined in raster metadata where applicable.

### 2. Vector Labels (`<SiteName>_label.gpkg`) — **Required**
- **Filename**: Must follow the format: `<SiteName>_label.gpkg` (e.g. `Unita/Unita_label.gpkg`).
- **Format**: OGC GeoPackage (`.gpkg`).
- **Geometry Type**: `Polygon` or `MultiPolygon`.
- **Attributes / Fields**:
  - `class`: Integer indicating the feature category:
    - `1`: **Geoglyph** (positive archaeological feature).
    - `2`: **Ground** (negative background terrain).
    - `3`: **Road** (roads / infrastructure).
  - Geometric properties: `Shape_Area`, `Shape_Leng`, and `geometry`.
- **CRS**: Automatically reprojected to match the raster CRS if different, but matching `EPSG:32719` is strongly recommended.

### 3. Digital Elevation Model (`<SiteName>_DEM.tif` or `.tiff`) — **Optional**
- **Filename**: Must follow the format: `<SiteName>_DEM.tif` (e.g. `Unita/Unita_DEM.tif`).
- **Format**: Single-band or multi-band GeoTIFF.
- **CRS**: Must cover the same spatial extent and match the projection of the orthomosaic.

---

## How to Add New Site Data

To add a new study area to the pipeline:

1. **Create the Site Directory**:
   ```bash
   mkdir data/<NewSiteName>
   ```
   *Example*: `mkdir data/Atacama`

2. **Add the Orthomosaic**:
   Copy your raw GeoTIFF into `data/<NewSiteName>/<NewSiteName>.tif`. Ensure the filename stem matches the folder name exactly.

3. **Add Vector Labels**:
   Export your polygon annotations from QGIS/ArcGIS as a GeoPackage named `<NewSiteName>_label.gpkg` containing the `class` column (`1` = geoglyph).

4. **Add Optional DEM**:
   If available, place the elevation raster as `<NewSiteName>_DEM.tif`.

5. **Verify and Run**:
   Verify configuration parameters in `pipeline_config.json`, then launch the extraction pipeline from the root directory:
   ```bash
   python pipeline.py
   ```
