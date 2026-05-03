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
import math
from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np

import shapely

from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import transform
from PIL import Image, ImageDraw

# Add parent directory to path to import project helpers
UTLS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTLS_PATH))
from handle import ROOT, CLASSES, CLASS_IDS, PATHS, SCALE_FACTORS, WINDOW_SIZE, get_area_tif, get_area_labels, PolygonData, make_jpeg_path, make_overlay_path, make_tif_path
from text import title
from utils import Polygon, calculate_bbox_size_meters

# Create reverse mapping for class names
IDS_TO_NAMES = {v: k for k, v in CLASSES.items()}

def save_tif(array, output_path: Path, transform, crs):
    """Save numpy array as georeferenced TIF."""
    if len(array.shape) == 2:
        array = array[np.newaxis, ...]

    count, height, width = array.shape
    output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), 'JPEG', quality=95, optimize=True)


def poly_to_coords(poly: Polygon):
    """Convert polygon to 2D coordinate lists, dropping Z values."""
    def to_xy(seq):
        return [(float(coord[0]), float(coord[1])) for coord in seq]
    
    exterior = to_xy(poly.exterior.coords) if poly.exterior else []
    interiors = [to_xy(interior.coords) for interior in poly.interiors]
    return {'exterior': exterior, 'interiors': interiors}

def round_up_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

def read_ortho_window(area: str, ortho_path: Path, bounds):
    """Read a window from orthomosaic based on bounds."""
    minx, miny, maxx, maxy = bounds
    
    with rasterio.open(str(ortho_path)) as ortho:

        row_min, col_min = ortho.index(minx, maxy)
        row_max, col_max = ortho.index(maxx, miny)

        #We do the following procedure to ensure that when we scale later, we have enough pixels for the view window, so as to avoid noise padding
        width = col_max - col_min
        height = row_max - row_min

        #We compare our current size with the target size, and if it's smaller than window size, we expand the window while keeping the center fixed
        target_width = int(math.ceil(round_up_to_multiple(width * SCALE_FACTORS[area], WINDOW_SIZE) / SCALE_FACTORS[area]))
        target_height = int(math.ceil(round_up_to_multiple(height * SCALE_FACTORS[area], WINDOW_SIZE) / SCALE_FACTORS[area]))

        dw = int((target_width - width) / 2)
        dh = int((target_height - height) / 2)

        col_min = max(col_min - dw, 0)
        col_max = min(col_max + dw, ortho.width)
        row_min = max(row_min - dh, 0)
        row_max = min(row_max + dh, ortho.height)

        if col_max - col_min < target_width:
            missing = target_width - (col_max - col_min)
            col_min = max(col_min - missing, 0)
            col_max = min(col_min + target_width, ortho.width)

        if row_max - row_min < target_height:
            missing = target_height - (row_max - row_min)
            row_min = max(row_min - missing, 0)
            row_max = min(row_min + target_height, ortho.height)

        window = Window.from_slices((row_min, row_max), (col_min, col_max))
        
        chunk = ortho.read(window=window)
        transform = ortho.window_transform(window)
        crs = ortho.crs
    
    return chunk, transform, crs


def create_polygon_metadata(polygon_idx, geometry, polygon_class, ortho_chunk, bbox_size, area):
    """Create and save polygon metadata."""
    minx, miny, maxx, maxy = geometry.bounds
    
    # Handle MultiPolygon
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    polygon_points = poly_to_coords(polygons[0])['exterior']
    
    # Create metadata object
    poly_obj = Polygon()
    poly_obj.id = polygon_idx
    poly_obj.class_id = int(polygon_class)
    poly_obj.area = area
    poly_obj.polygon_points = polygon_points
    poly_obj.shape = (ortho_chunk.shape[1], ortho_chunk.shape[2])
    poly_obj.size_m = bbox_size
    poly_obj.coords = {'top': maxy, 'left': minx, 'bottom': miny, 'right': maxx}
    poly_obj.polygon = polygons[0]
    poly_obj.jpeg_path = Path()
    poly_obj.tif_path = Path()
    poly_obj.overlay_path = Path()
    poly_obj.resized_path = Path()
    poly_obj.crop_paths = []
    poly_obj.augmented_paths = []
    
    # Save metadata
    PolygonData.save_polygons([poly_obj])
    return poly_obj

def extract_polygon_geometries(polygon_idx, geometry, gdf_crs, polygon_class, ortho_path: Path, area: str):
    """Extract and save images for a single polygon."""
    # Read ortho data for exact bounds
    ortho_chunk, ortho_transform, ortho_crs = read_ortho_window(area, ortho_path, geometry.bounds)
    
    # Calculate size in meters
    bbox_size = calculate_bbox_size_meters(geometry.bounds, gdf_crs)
    
    # Create metadata and get paths
    polygon_obj = create_polygon_metadata(
        polygon_idx, geometry, polygon_class, ortho_chunk, bbox_size, area
    )

    return polygon_obj, ortho_chunk, ortho_transform, ortho_crs

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
        polygons, geometries = extract_area(area, limit=args.limit, class_filter=args.class_filter)
        save_data(area, polygons, geometries)

def save_data(area, polygons: tuple[Polygon], geometries: dict[str, shapely.geometry.base.BaseGeometry]):
    output_dir = PATHS[area]["polygons"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(title(f"Saving extracted data for area: {area}"))
    for poly, geom in zip(polygons, geometries):
        print(f"Saving polygon ID {poly.id}...")

        tif_path = make_tif_path(area, poly.id, poly.class_id)
        jpeg_path = make_jpeg_path(area, poly.id, poly.class_id)
        overlay_path = make_overlay_path(area, poly.id, poly.class_id)

        poly.tif_path = tif_path
        poly.jpeg_path = jpeg_path
        poly.overlay_path = overlay_path
        
        # Save images
        save_tif(geom["chunk"], tif_path, geom["transform"], geom["crs"])
        save_jpeg(geom["chunk"][:3].transpose(1, 2, 0), jpeg_path)
        save_overlay_jpeg(geom["chunk"], [poly.polygon], geom["transform"], overlay_path)

def extract_area(area: str, limit: int = None, class_filter: int = CLASSES["geo"]) -> None:
    """Extract images from geo-referenced data for a single area.
    
    Args:
        area: Study area to process (e.g., 'unita', 'chugchug', 'lluta').
        limit: Maximum number of polygons to process. None to process all.
        class_filter: Process only polygons of a specific class. None to process all classes.
    """
    print(title(f"Extracting polygons from area: {area}"))
    
    # Load data
    gpkg_path = get_area_labels(area)
    ortho_path = get_area_tif(area)
    gdf = load_geodataframe(gpkg_path, ortho_path, limit)
    
    print(f"Loaded {len(gdf)} polygons")
    
    if class_filter is not None:
        print(f"Filtering to class {class_filter} ({IDS_TO_NAMES.get(class_filter, 'unknown')})\n")
    
    # Process each polygon
    processed_count = 0
    polygons = []
    geometries = []
    for idx, row in gdf.iterrows():
        polygon_class = row['class']
        
        # Apply class filter
        if class_filter is not None and polygon_class != class_filter:
            continue
        
        class_name = IDS_TO_NAMES.get(polygon_class, 'unknown')
        print(f"Processing polygon {idx} (class: {polygon_class} - {class_name})...")
        
        polygon_obj, ortho_chunk, ortho_transform, ortho_crs = extract_polygon_geometries(
            idx, row.geometry, gdf.crs, polygon_class,
            ortho_path, area
        )
        processed_count += 1
        polygons.append(polygon_obj)
        geometries.append({ "chunk": ortho_chunk, "transform": ortho_transform, "crs": ortho_crs })

    print(f"\nDone! Processed {processed_count} polygons for {area}")
    return polygons, geometries

if __name__ == "__main__":
    main()