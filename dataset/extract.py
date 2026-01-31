#!/usr/bin/env python3
"""
Extract images from geo-referenced data for each polygon in the geopackage.

For each polygon, creates:
- Original ortho image crop (TIF + JPEG)
- Ortho image with polygon overlay (JPEG)
- Bounding box size in meters
"""

import argparse
from pathlib import Path
import json
import sys
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from pyproj import Geod

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import CLASSES, CLASS_IDS

# Create reverse mapping for class names
CLASS_NAMES = {v: k for k, v in CLASSES.items()}

def calculate_bbox_size_meters(bounds, crs):
    """
    Calculate the size of a bounding box in meters.

    Args:
        bounds: (minx, miny, maxx, maxy) in the CRS coordinates
        crs: The coordinate reference system

    Returns:
        dict with width_m, height_m, area_m2
    """
    from shapely.geometry import Point
    
    minx, miny, maxx, maxy = bounds

    # Convert corner points to WGS84 if needed
    if crs and crs.to_epsg() != 4326:
        # CRS is not WGS84, need to convert
        proj_crs = crs
        wgs84_crs = "EPSG:4326"
        
        # Create points in original CRS
        p1 = Point(minx, miny)
        p2 = Point(maxx, miny)
        p3 = Point(minx, maxy)
        
        # Project to WGS84
        from shapely.ops import transform
        from pyproj import Transformer
        
        transformer = Transformer.from_crs(proj_crs, wgs84_crs, always_xy=True)
        
        p1_wgs84 = transform(transformer.transform, p1)
        p2_wgs84 = transform(transformer.transform, p2)
        p3_wgs84 = transform(transformer.transform, p3)
        
        minx_wgs, miny_wgs = p1_wgs84.x, p1_wgs84.y
        maxx_wgs, miny_wgs2 = p2_wgs84.x, p2_wgs84.y
        minx_wgs2, maxy_wgs = p3_wgs84.x, p3_wgs84.y
    else:
        # Already in WGS84
        minx_wgs, miny_wgs = minx, miny
        maxx_wgs, maxy_wgs = maxx, maxy

    # Create a geoid for accurate distance calculation
    geod = Geod(ellps="WGS84")

    # Calculate width (distance along bottom edge)
    _, _, width_m = geod.inv(minx_wgs, miny_wgs, maxx_wgs, miny_wgs)

    # Calculate height (distance along left edge)
    _, _, height_m = geod.inv(minx_wgs, miny_wgs, minx_wgs, maxy_wgs)

    return {
        'width_m': abs(width_m),
        'height_m': abs(height_m),
        'area_m2': abs(width_m * height_m)
    }


def save_image_no_axes(array, output_path: Path, transform=None, crs=None, is_rgb=False):
    """
    Save image without axes as TIF.

    Args:
        array: numpy array (C, H, W) or (H, W)
        output_path: Path to save TIF file
        transform: rasterio transform
        crs: coordinate reference system
        is_rgb: whether the image is RGB
    """
    if len(array.shape) == 2:
        array = array[np.newaxis, ...]  # Add channel dimension

    count, height, width = array.shape

    with rasterio.open(
        str(output_path),
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array)


def save_jpeg_no_axes(array, output_path: Path, dpi=150):
    """
    Save array as JPEG without axes or borders.

    Args:
        array: numpy array, either (H, W, C) for RGB or (H, W) for grayscale
        output_path: Path to save JPEG file
        dpi: resolution for the output
    """
    # Normalize array to 0-255 range if needed
    if array.dtype != np.uint8:
        # Normalize to 0-1 range first
        arr_min, arr_max = array.min(), array.max()
        if arr_max > arr_min:
            array_norm = (array - arr_min) / (arr_max - arr_min)
        else:
            array_norm = array
        array = (array_norm * 255).astype(np.uint8)

    # Create figure with exact size
    h, w = array.shape[:2]
    fig_width = w / dpi
    fig_height = h / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))  # (left, bottom, width, height)
    ax.axis('off')

    if len(array.shape) == 3:
        ax.imshow(array)
    else:
        ax.imshow(array, cmap='gray')

    plt.savefig(str(output_path), dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_overlay_jpeg(array, polygons, transform, output_path: Path, dpi=150):
    """
    Save array with polygon overlay as JPEG without axes.

    Args:
        array: numpy array (C, H, W) for RGB
        polygons: list of shapely polygons
        transform: rasterio transform for coordinate mapping
        output_path: Path to save JPEG file
        dpi: resolution for the output
    """
    # Convert array to (H, W, C) format
    if len(array.shape) == 3 and array.shape[0] in [3, 4]:
        array = array.transpose(1, 2, 0)

    # Normalize to 0-255 if needed
    if array.dtype != np.uint8:
        arr_min, arr_max = array.min(), array.max()
        if arr_max > arr_min:
            array_norm = (array - arr_min) / (arr_max - arr_min)
        else:
            array_norm = array
        array = (array_norm * 255).astype(np.uint8)

    # Take only RGB channels if there are more
    if array.shape[2] > 3:
        array = array[:, :, :3]

    h, w = array.shape[:2]
    fig_width = w / dpi
    fig_height = h / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis('off')

    # Calculate extent from transform
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    extent = (x0, x1, y1, y0)

    ax.imshow(array, extent=extent, interpolation='nearest')

    # Draw polygon boundaries
    for poly in polygons:
        if hasattr(poly, 'exterior'):
            x, y = poly.exterior.xy
            ax.plot(x, y, color='yellow', linewidth=2)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.savefig(str(output_path), dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def extract_polygon_images(polygon_idx, geometry, gdf_crs, polygon_class, ortho_path: Path, output_dir: Path):
    """
    Extract images for a single polygon.

    Args:
        polygon_idx: index of the polygon
        geometry: shapely geometry object
        gdf_crs: CRS of the geodataframe
        polygon_class: class of the polygon (1=geo, 2=ground, 3=road)
        ortho_path: Path to the orthomosaic TIF
        output_dir: Path to directory to save outputs
    """
    # Get polygon bounding box
    minx, miny, maxx, maxy = geometry.bounds

    with rasterio.open(str(ortho_path)) as ortho:
        # Convert map coords to raster pixel coords (no padding for ortho)
        row_min, col_min = ortho.index(minx, maxy)  # top-left corner
        row_max, col_max = ortho.index(maxx, miny)  # bottom-right corner

        # Create raster window for ortho image
        ortho_win = Window.from_slices(
            (row_min, row_max),
            (col_min, col_max)
        )

        # Read ortho data
        ortho_chunk = ortho.read(window=ortho_win)
        ortho_transform = ortho.window_transform(ortho_win)
        ortho_crs = ortho.crs

        # Create padded window for overlay image (5% padding in each direction)
        width = maxx - minx
        height = maxy - miny
        padding_x = width * 0.05
        padding_y = height * 0.05

        minx_padded = minx - padding_x
        maxx_padded = maxx + padding_x
        miny_padded = miny - padding_y
        maxy_padded = maxy + padding_y

        # Convert padded map coords to raster pixel coords
        row_min_padded, col_min_padded = ortho.index(minx_padded, maxy_padded)
        row_max_padded, col_max_padded = ortho.index(maxx_padded, miny_padded)

        # Create padded window for overlay
        overlay_win = Window.from_slices(
            (row_min_padded, row_max_padded),
            (col_min_padded, col_max_padded)
        )

        # Read overlay data
        overlay_chunk = ortho.read(window=overlay_win)
        overlay_transform = ortho.window_transform(overlay_win)

    def _poly_to_coords(poly: Polygon):
        """Convert polygon to 2D coordinate lists, dropping any Z values."""
        def _to_xy(seq):
            xy = []
            for coord in seq:
                # coord can be (x, y) or (x, y, z, ...); keep first two
                xy.append((float(coord[0]), float(coord[1])))
            return xy

        exterior = _to_xy(poly.exterior.coords) if poly.exterior else []
        interiors = [_to_xy(interior.coords) for interior in poly.interiors]
        return {
            'exterior': exterior,
            'interiors': interiors
        }

    # Handle MultiPolygon and keep coordinate lists for metadata
    if isinstance(geometry, MultiPolygon):
        polygons = [p for p in geometry.geoms]
        polygon_points = [_poly_to_coords(p) for p in polygons]
    else:
        polygons = [geometry]
        polygon_points = [_poly_to_coords(geometry)]

    # Calculate bounding box size in meters
    bbox_size = calculate_bbox_size_meters(geometry.bounds, gdf_crs)

    # Create output filenames
    base_name = f"geoglif_{polygon_idx:04d}"
    tif_path = output_dir / f"{base_name}_ortho.tif"
    jpeg_path = output_dir / f"{base_name}_ortho.jpg"
    overlay_path = output_dir / f"{base_name}_overlay.jpg"
    metadata_path = output_dir / f"{base_name}_metadata.json"

    # Save original ortho image as TIF (original data type)
    save_image_no_axes(ortho_chunk, tif_path, ortho_transform, ortho_crs, is_rgb=True)

    # Save original ortho image as JPEG
    ortho_rgb = ortho_chunk[:3].transpose(1, 2, 0)  # Convert to (H, W, C)
    save_jpeg_no_axes(ortho_rgb, jpeg_path)

    # Save overlay image as JPEG (using padded data)
    save_overlay_jpeg(overlay_chunk, polygons, overlay_transform, overlay_path)

    # Save metadata
    metadata = {
        'polygon_index': polygon_idx,
        'class': int(polygon_class),
        'bounds': {
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy
        },
        'bbox_size_meters': bbox_size,
        'crs': str(gdf_crs),
        'polygon_points': polygon_points,
        'image_shape': {
            'height': ortho_chunk.shape[1],
            'width': ortho_chunk.shape[2],
            'channels': ortho_chunk.shape[0]
        },
        'files': {
            'ortho_tif': tif_path.name,
            'ortho_jpeg': jpeg_path.name,
            'overlay_jpeg': overlay_path.name
        }
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract images from geo-referenced data for each polygon in the geopackage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all polygons
  python extract_polygon_images.py

  # Process only the first 10 polygons
  python extract_polygon_images.py --limit 10

  # Process first 20 polygons with custom output directory
  python extract_polygon_images.py --limit 20 --output custom_output

  # Use custom input files
  python extract_polygon_images.py --layers my_layers.gpkg --ortho my_ortho.tif
        """
    )

    parser.add_argument(
        '--layers',
        type=Path,
        required=True,
        help='Path to the geopackage file containing polygons'
    )

    parser.add_argument(
        '--ortho',
        type=Path,
        required=True,
        help='Path to the orthomosaic TIF file'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for extracted images'
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
        help=f'Filter to process only polygons of a specific class ({", ".join(f"{v}={k}" for k, v in CLASSES.items())}). Default: process all classes'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load polygons
    print(f"Loading polygons from {args.layers}...")
    gdf = gpd.read_file(str(args.layers))
    print(f"Loaded {len(gdf)} polygons")
    print(f"Original CRS: {gdf.crs}")

    # Apply limit if specified
    if args.limit is not None:
        print(f"Limiting to first {args.limit} polygons")
        gdf = gdf.head(args.limit)

    # Load ortho to get CRS
    with rasterio.open(str(args.ortho)) as ortho:
        ortho_crs = ortho.crs
        print(f"Ortho CRS: {ortho_crs}")

    # Convert GDF to ortho CRS if needed
    if gdf.crs != ortho_crs:
        print(f"Converting from {gdf.crs} to {ortho_crs}")
        gdf = gdf.to_crs(ortho_crs)

    # Process each polygon
    all_metadata = []
    skipped_count = 0
    
    if args.class_filter is not None:
        print(f"\nFiltering to class {args.class_filter} ({CLASS_NAMES.get(args.class_filter, 'unknown')}) only\n")
    
    for idx, row in gdf.iterrows():
        polygon_class = row['class']
        class_name = CLASS_NAMES.get(polygon_class, 'unknown')
        
        # Skip if class filter is set and polygon doesn't match
        if args.class_filter is not None and polygon_class != args.class_filter:
            skipped_count += 1
            continue
        
        fid = idx  # FID from GeoPackage
        print(f"\nProcessing polygon {idx + 1}/{len(gdf)} (FID: {fid}, class: {polygon_class} - {class_name})...")
        try:
            metadata = extract_polygon_images(
                idx,
                row.geometry,
                gdf.crs,
                polygon_class,
                args.ortho,
                args.output
            )
            all_metadata.append(metadata)
            print(f"  Bounding box: {metadata['bbox_size_meters']['width_m']:.2f}m x "
                  f"{metadata['bbox_size_meters']['height_m']:.2f}m "
                  f"(area: {metadata['bbox_size_meters']['area_m2']:.2f} m²)")
        except Exception as e:
            print(f"  Error processing polygon {idx}: {e}")

    # Save summary metadata
    summary_path = args.output / "summary.json"
    summary_data = {
        'total_polygons': len(gdf),
        'processed_polygons': len(all_metadata),
        'output_directory': str(args.output),
        'source_files': {
            'layers': str(args.layers),
            'ortho': str(args.ortho)
        },
        'polygons': all_metadata
    }
    summary_path.write_text(json.dumps(summary_data, indent=2))

    print(f"\n\nDone! Processed {len(all_metadata)} polygons.")
    if args.class_filter is not None:
        print(f"Skipped {skipped_count} polygons (class filter: {args.class_filter})")
    print(f"Output directory: {args.output}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()