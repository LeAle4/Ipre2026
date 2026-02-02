#!/usr/bin/env python3
"""
Extract images from geo-referenced data for each polygon in the geopackage.

For each polygon, creates:
- Original ortho image crop (TIF + JPEG)
- Ortho image with polygon overlay (JPEG)
- Bounding box size in meters
"""
import sys
import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import transform
from pyproj import Geod, Transformer
from PIL import Image, ImageDraw

# Add parent directory to path to import project helpers
UTLS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTLS_PATH))
from handle import CLASSES, CLASS_IDS, POLYGON_DATA_DIR, PATHS
from text import title, tabbed
from utils import Polygon as PolygonData

# Create reverse mapping for class names
CLASS_NAMES = {v: k for k, v in CLASSES.items()}

def calculate_bbox_size_meters(bounds, crs):
    """Calculate the size of a bounding box in meters."""
    minx, miny, maxx, maxy = bounds

    # Convert to WGS84 if needed
    if crs and crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        p1_wgs84 = transform(transformer.transform, Point(minx, miny))
        p2_wgs84 = transform(transformer.transform, Point(maxx, miny))
        p3_wgs84 = transform(transformer.transform, Point(minx, maxy))
        
        minx_wgs, miny_wgs = p1_wgs84.x, p1_wgs84.y
        maxx_wgs = p2_wgs84.x
        maxy_wgs = p3_wgs84.y
    else:
        minx_wgs, miny_wgs = minx, miny
        maxx_wgs, maxy_wgs = maxx, maxy

    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(minx_wgs, miny_wgs, maxx_wgs, miny_wgs)
    _, _, height_m = geod.inv(minx_wgs, miny_wgs, minx_wgs, maxy_wgs)

    return (abs(width_m), abs(height_m))


def save_tif(array, output_path: Path, transform, crs):
    """Save numpy array as georeferenced TIF."""
    if len(array.shape) == 2:
        array = array[np.newaxis, ...]

    count, height, width = array.shape
    with rasterio.open(
        str(output_path), 'w',
        driver='GTiff', height=height, width=width,
        count=count, dtype=array.dtype,
        crs=crs, transform=transform
    ) as dst:
        dst.write(array)


def save_jpeg(array, output_path: Path):
    """Save numpy array as JPEG using PIL."""
    if array.dtype != np.uint8:
        arr_min, arr_max = array.min(), array.max()
        array = (array - arr_min) / (arr_max - arr_min) if arr_max > arr_min else array
        array = (array * 255).astype(np.uint8)

    mode = 'RGB' if len(array.shape) == 3 else 'L'
    img = Image.fromarray(array, mode=mode)
    img.save(str(output_path), 'JPEG', quality=95, optimize=True)


def save_overlay_jpeg(array, polygons, transform, output_path: Path):
    """Save array with polygon overlay as JPEG using PIL."""
    # Convert to (H, W, C) and normalize
    if len(array.shape) == 3 and array.shape[0] in [3, 4]:
        array = array.transpose(1, 2, 0)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    
    if array.dtype != np.uint8:
        arr_min, arr_max = array.min(), array.max()
        array = (array - arr_min) / (arr_max - arr_min) if arr_max > arr_min else array
        array = (array * 255).astype(np.uint8)

    h, w = array.shape[:2]
    img = Image.fromarray(array, mode='RGB')
    draw = ImageDraw.Draw(img)

    # Calculate coordinate transformation
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    scale_x = w / (x1 - x0)
    scale_y = h / (y1 - y0)

    # Draw polygons
    for poly in polygons:
        if hasattr(poly, 'exterior'):
            pixel_coords = [
                ((coord[0] - x0) * scale_x, (coord[1] - y0) * scale_y)
                for coord in poly.exterior.coords
            ]
            draw.line(pixel_coords, fill='yellow', width=3)
    
    img.save(str(output_path), 'JPEG', quality=95, optimize=True)


def poly_to_coords(poly: Polygon):
    """Convert polygon to 2D coordinate lists, dropping Z values."""
    def to_xy(seq):
        return [(float(coord[0]), float(coord[1])) for coord in seq]
    
    exterior = to_xy(poly.exterior.coords) if poly.exterior else []
    interiors = [to_xy(interior.coords) for interior in poly.interiors]
    return {'exterior': exterior, 'interiors': interiors}


def read_ortho_window(ortho_path: Path, bounds, pad_factor=0.0):
    """Read a window from orthomosaic based on bounds."""
    minx, miny, maxx, maxy = bounds
    
    with rasterio.open(str(ortho_path)) as ortho:
        if pad_factor > 0:
            pad = max(maxx - minx, maxy - miny) * pad_factor
            minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
        
        row_min, col_min = ortho.index(minx, maxy)
        row_max, col_max = ortho.index(maxx, miny)
        window = Window.from_slices((row_min, row_max), (col_min, col_max))
        
        chunk = ortho.read(window=window)
        transform = ortho.window_transform(window)
        crs = ortho.crs
    
    return chunk, transform, crs


def create_polygon_metadata(polygon_idx, geometry, polygon_class, ortho_chunk, bbox_size, output_dir, area):
    """Create and save polygon metadata."""
    minx, miny, maxx, maxy = geometry.bounds
    
    # Handle MultiPolygon
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    polygon_points = poly_to_coords(polygons[0])['exterior']
    
    # Create output paths
    base_name = f"geoglif_{polygon_idx:04d}"
    tif_path = output_dir / f"{base_name}_ortho.tif"
    jpeg_path = output_dir / f"{base_name}_ortho.jpg"
    overlay_path = output_dir / f"{base_name}_overlay.jpg"
    
    # Create metadata object
    poly_obj = PolygonData()
    poly_obj.id = polygon_idx
    poly_obj.class_id = int(polygon_class)
    poly_obj.area = area
    poly_obj.polygon_points = polygon_points
    poly_obj.shape = (ortho_chunk.shape[2], ortho_chunk.shape[1], ortho_chunk.shape[0])
    poly_obj.size_m = bbox_size
    poly_obj.coords = {'top': miny, 'left': minx, 'bottom': maxy, 'right': maxx}
    poly_obj.polygon = polygons[0]
    poly_obj.jpeg_path = jpeg_path
    poly_obj.tif_path = tif_path
    poly_obj.overlay_path = overlay_path
    poly_obj.resized_path = Path()
    poly_obj.crop_paths = []
    poly_obj.augmented_paths = []
    
    # Save metadata
    POLYGON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    poly_obj.save_metadata(POLYGON_DATA_DIR / f"{area}_class{poly_obj.class_id}_{poly_obj.id}_metadata.json")
    
    return tif_path, jpeg_path, overlay_path, polygons


def extract_polygon_images(polygon_idx, geometry, gdf_crs, polygon_class, ortho_path: Path, output_dir: Path, area: str):
    """Extract and save images for a single polygon."""
    # Read ortho data for exact bounds
    ortho_chunk, ortho_transform, ortho_crs = read_ortho_window(ortho_path, geometry.bounds)
    
    # Read ortho data with padding for overlay
    overlay_chunk, overlay_transform, _ = read_ortho_window(ortho_path, geometry.bounds, pad_factor=0.05)
    
    # Calculate size in meters
    bbox_size = calculate_bbox_size_meters(geometry.bounds, gdf_crs)
    
    # Create metadata and get paths
    tif_path, jpeg_path, overlay_path, polygons = create_polygon_metadata(
        polygon_idx, geometry, polygon_class, ortho_chunk, bbox_size, output_dir, area
    )
    
    # Save images
    save_tif(ortho_chunk, tif_path, ortho_transform, ortho_crs)
    save_jpeg(ortho_chunk[:3].transpose(1, 2, 0), jpeg_path)
    save_overlay_jpeg(overlay_chunk, polygons, overlay_transform, overlay_path)


def load_area_data(area):
    """Load geopackage and orthomosaic paths for an area."""
    raw_dir = PATHS[area]["raw"]
    output_dir = PATHS[area]["polygons"]
    
    gpkg_files = list(raw_dir.glob("*.gpkg"))
    tif_files = list(raw_dir.glob("*ortomosaico.tif")) or list(raw_dir.glob("*.tif"))
    
    if not gpkg_files:
        raise FileNotFoundError(f"No .gpkg files found in {raw_dir}")
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {raw_dir}")
    
    return gpkg_files[0], tif_files[0], output_dir


def load_geodataframe(gpkg_path, ortho_path, limit=None):
    """Load and prepare geodataframe from geopackage."""
    gdf = gpd.read_file(str(gpkg_path))
    
    if limit is not None:
        gdf = gdf.head(limit)
    
    # Convert to ortho CRS if needed
    with rasterio.open(str(ortho_path)) as ortho:
        ortho_crs = ortho.crs
        if gdf.crs != ortho_crs:
            gdf = gdf.to_crs(ortho_crs)
    
    return gdf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract images from geo-referenced data for each polygon in the geopackage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all polygons in unita area
  python extract.py --area unita

  # Process only geoglyphs from chugchug
  python extract.py --area chugchug --class-filter 1

  # Process first 10 polygons from lluta
  python extract.py --area lluta --limit 10
        """
    )

    parser.add_argument(
        '--area',
        type=str,
        required=True,
        nargs="+",
        choices=['unita', 'chugchug', 'lluta'],
        help='Study area name (unita, chugchug, or lluta)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit processing to the first N polygons (default: process all)'
    )

    parser.add_argument(
        '--class-filter',
        type=int,
        default=None,
        choices=CLASS_IDS,
        help=f'Filter to process only polygons of a specific class. Default: process all classes'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Process each area
    for area in args.area:
        process_area(area, args)

def process_area(area, args):
    """Process extraction for a single area."""
    print(title(f"Extracting polygons from area: {area}"))
    
    # Load data
    gpkg_path, ortho_path, output_dir = load_area_data(area)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gdf = load_geodataframe(gpkg_path, ortho_path, args.limit)
    print(f"Loaded {len(gdf)} polygons")
    
    if args.class_filter is not None:
        print(f"Filtering to class {args.class_filter} ({CLASS_NAMES.get(args.class_filter, 'unknown')})\n")
    
    # Process each polygon
    processed_count = 0
    for idx, row in gdf.iterrows():
        polygon_class = row['class']
        
        # Apply class filter
        if args.class_filter is not None and polygon_class != args.class_filter:
            continue
        
        class_name = CLASS_NAMES.get(polygon_class, 'unknown')
        print(f"Processing polygon {idx} (class: {polygon_class} - {class_name})...")
        
        extract_polygon_images(
            idx, row.geometry, gdf.crs, polygon_class,
            ortho_path, output_dir, area
        )
        processed_count += 1
    
    print(f"\nDone! Processed {processed_count} polygons for {area}")


if __name__ == "__main__":
    main()