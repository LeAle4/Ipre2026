# Format Tools

## resize_dataset.py

Resizes all geoglif orthomosaic images using the LCI (Lagrange-Chebyshev Interpolation) method while preserving georeferencing information.

### Quick Start

**Basic usage (uses default paths):**
```bash
python resize_dataset.py
```

This will:
- Process all `*_geos` directories in `../data/`
- Output resized images to `../data/resized_geos/`
- Apply area-specific scale factors automatically

### Scale Factors

The script uses predefined scale factors for each area:
- **unita**: 0.886
- **lluta**: 0.218
- **chugchug**: 0.18

### Custom Paths

**Specify custom data directory:**
```bash
python resize_dataset.py --data-dir /path/to/data
```

**Specify custom output directory:**
```bash
python resize_dataset.py --output-dir /path/to/output
```

**Both custom paths:**
```bash
python resize_dataset.py --data-dir /path/to/data --output-dir /path/to/output
```

### Output

For each area, the script creates:
- `*_ortho.tif` - Georeferenced GeoTIFF with updated transform
- `*_resized.png` - Visual reference image
- `*_metadata.json` - Updated metadata with new dimensions

### Requirements

```bash
pip install numpy pillow rasterio scipy
```
