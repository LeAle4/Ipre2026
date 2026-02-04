import sys
import argparse
import random

from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd

from rasterio.windows import Window
from shapely.geometry import box, Polygon as ShapelyPolygon
from PIL import Image

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import NEGATIVES_RATIO, crops_in_area, AREA_NAMES, CLASSES, WINDOW_SIZE, POLYGON_DATA_DIR, SCALES, get_area_tif, get_area_labels, make_negative_path
from utils import Polygon, calculate_bbox_size_meters
from text import title, tabbed
from resize import lci

def _negatives_per_area(area:str) -> int:
    """Calculate the number of negative samples to extract for a given area.
    
    Args:
        area: Study area name (e.g., 'unita', 'chugchug', 'lluta').
    """
    num_crops = crops_in_area(area)
    return int(num_crops * NEGATIVES_RATIO)

def load_tif(tif_path:Path):
    return rasterio.open(tif_path)

def get_all_polygon_boundaries(boundary_path: Path, target_crs) -> list[ShapelyPolygon]:    
    gdf = gpd.read_file(boundary_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return list(gdf.geometry)

def in_actual_data(tif_area, view_window:Window, threshold:float = 1) -> bool:
    """Check if the specified window contains valid data above the threshold.
    
    Args:
        tif_area: Rasterio dataset object of the study area.
        view_window: Window object defining the area to check.
        threshold: Minimum fraction of valid data required (0-1).
    """
    data = tif_area.read(1, window=view_window)
    no_data_value = tif_area.nodata
    
    # If nodata is defined, use it
    if no_data_value is not None:
        valid_pixels = np.sum(data != no_data_value)
    else:
        # If nodata is None, assume common no-data values (white=255 for uint8)
        # Count pixels that are not pure white (255) and not pure black (0)
        valid_pixels = np.sum((data != 255) & (data != 0))
    
    total_pixels = data.size
    fraction_valid = valid_pixels / total_pixels if total_pixels > 0 else 0
    
    return fraction_valid >= threshold

def not_overlapping(negative_boundary:ShapelyPolygon, boundaries:list[ShapelyPolygon]) -> bool:
    for boundary in boundaries:
        if negative_boundary.intersects(boundary) or boundary.contains(negative_boundary):
            return False
    return True

def sample_boundary(tif_area, area:str, window_size:int = WINDOW_SIZE) -> tuple[Window, ShapelyPolygon]:
    """Sample a random boundary box within the tif area.
    
    Args:
        tif_area: Path to the tif file of the study area.
        window_size: Size of the square boundary to sample in pixels.
    """
    window_size = int(window_size // SCALES[area])
    x,y = random.randint(0, tif_area.width - window_size), random.randint(0, tif_area.height - window_size)
    window = Window(x, y, window_size, window_size)
    bounds = tif_area.window_bounds(window)
    return window, box(*bounds)
    
def save_boundary_as_geo(boundary:ShapelyPolygon, area:str, negative_id:int, img_save_path:Path, img_shape:tuple[int,int, int], crs, view_window:Window = None) -> None:
    """Generate and save a Polygon file for the negative sample boundary."""
    minx, miny, maxx, maxy = boundary.bounds
    
    # Calculate size in meters using boundary (which is created from view_window)
    size_m = calculate_bbox_size_meters(boundary.bounds, crs)
    
    polygon = Polygon(
        id = negative_id,
        class_id = CLASSES["ground"],
        area = area,
        polygon_points = list(boundary.exterior.coords),
        shape = img_shape,  # Use actual image shape (height, width, channels)
        size_m = size_m,
        coords = {'top': maxy, 'left': minx, 'bottom': miny, 'right': maxx},
        jpeg_path = Path(),
        tif_path = Path(),
        overlay_path = Path(),
        resized_path = Path(),
        crop_paths = [img_save_path],
        augmented_paths = [],
        polygon = boundary
    )
    
    # Save metadata
    POLYGON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    polygon.save_metadata(POLYGON_DATA_DIR)

def save_boundary_image(src, boundary:Window, save_path:Path) -> tuple[int, int, int]:
    """
    Save the RGB image corresponding to the boundary window to the specified path.

    Args:
        boundary: Window object defining the boundary to save.
        save_path: Path to save the boundary image.
    """
    img = src.read(window=boundary)
    shape = img.shape
    # Transpose from (bands, height, width) to (height, width, bands)
    img = img.transpose(1, 2, 0)
    resized = lci(img, WINDOW_SIZE, WINDOW_SIZE)
    img = Image.fromarray(resized)
    img.save(save_path)

    return shape

def parse_args():
    parser = argparse.ArgumentParser(description="Extract negative samples from polygon dataset.")
    parser.add_argument(
        "--area",
        type=str,
        nargs="+",
        choices=AREA_NAMES,
        required=True,
        help="Study area to process.",
    )

    return parser.parse_args()

def extract_negatives_area(area: str) -> None:
    """Extract negative samples from polygon dataset for a single area.
    
    Args:
        area: Study area to process (e.g., 'unita', 'chugchug', 'lluta').
    """
    sampled = 0
    limit = _negatives_per_area(area)
    print(title(f"Extracting negative samples for area: {area}, amount to sample: {limit}"))

    # Load the area tif
    tif_path = get_area_tif(area)
    labels_path = get_area_labels(area)

    tif_img = load_tif(tif_path)
    boundaries = get_all_polygon_boundaries(labels_path, tif_img.crs)

    while sampled < limit:
        view_window, boundary = sample_boundary(tif_img, area)
        # Check both polygon intersection and valid data presence
        if not_overlapping(boundary, boundaries) and in_actual_data(tif_img, view_window):
            sampled += 1
            print(tabbed(f"Sampled negative {sampled}/{limit} for area {area}"))
            # Since areas are so big, there is a low probability of sampling a crop from the same area, speeds up code
            # boundaries.append(boundary)
            save_path = make_negative_path(area, sampled)
            img_shape = save_boundary_image(tif_img, view_window, save_path)
            save_boundary_as_geo(boundary, area, sampled, save_path, img_shape, tif_img.crs, view_window)


if __name__ == "__main__":
    args = parse_args()
    areas = args.area

    for area in areas:
        extract_negatives_area(area)