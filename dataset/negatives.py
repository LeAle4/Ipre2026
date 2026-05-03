import sys
import argparse
import random

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import rasterio
import geopandas as gpd

from rasterio.windows import Window
from shapely.geometry import box, Polygon as ShapelyPolygon
from shapely.prepared import prep
from PIL import Image

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import ROOT, PATHS, NEGATIVES_RATIO, AREA_NAMES, CLASSES, SCALE_FACTORS, WINDOW_SIZE, PolygonData, get_area_tif, get_area_labels, make_negative_path
from utils import Polygon, calculate_bbox_size_meters
from text import title, tabbed
from resize import lci

def _negatives_per_area(area:str) -> int:
    """Calculate the number of negative samples to extract for a given area.
    
    Args:
        area: Study area name (e.g., 'unita', 'chugchug', 'lluta').
    """
    num_crops = PolygonData.crops_in_area(area)
    return int(num_crops * NEGATIVES_RATIO)

def load_tif(tif_path:Path):
    return rasterio.open(tif_path)

def get_all_polygon_boundaries(boundary_path: Path, target_crs) -> tuple[ShapelyPolygon]:    
    gdf = gpd.read_file(boundary_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return tuple(gdf.geometry)

def in_actual_data(tif_area, view_window:Window, no_data_value, threshold:float = 1) -> bool:
    """Check if the specified window contains valid data above the threshold.
    
    Args:
        tif_area: Rasterio dataset object of the study area.
        view_window: Window object defining the area to check.
        no_data_value: Cached nodata value from tif_area.nodata.
        threshold: Minimum fraction of valid data required (0-1).
    """
    data = tif_area.read(1, window=view_window)
    
    # If nodata is defined, use it
    if no_data_value is not None:
        valid_pixels = np.count_nonzero(data != no_data_value)
    else:
        # If nodata is None, assume common no-data values (white=255 for uint8)
        valid_pixels = np.count_nonzero((data != 255) & (data != 0))
    
    total_pixels = data.size
    fraction_valid = valid_pixels / total_pixels if total_pixels > 0 else 0
    
    return fraction_valid >= threshold

def not_overlapping(negative_boundary:ShapelyPolygon, prepared_boundaries:list) -> bool:
    """Check if negative boundary doesn't overlap with any existing boundaries (uses prepared geometry).
    
    Args:
        negative_boundary: Candidate boundary to check.
        prepared_boundaries: List of prepared geometry objects for faster intersection checks.
    """
    for prep_boundary in prepared_boundaries:
        if prep_boundary.intersects(negative_boundary):
            return False
    return True

def sample_boundary(tif_area, area:str, window_size:int = WINDOW_SIZE) -> tuple[Window, ShapelyPolygon]:
    """Sample a random boundary box within the tif area.
    
    Args:
        tif_area: Path to the tif file of the study area.
        window_size: Size of the square boundary to sample in pixels.
    """
    window_size = int(window_size // SCALE_FACTORS[area])
    x,y = random.randint(0, tif_area.width - window_size), random.randint(0, tif_area.height - window_size)
    window = Window(x, y, window_size, window_size)
    bounds = tif_area.window_bounds(window)
    return window, box(*bounds)
    
def _create_negative_polygon(boundary:ShapelyPolygon, area:str, negative_id:int, img_save_path:Path, img_shape:tuple[int,int, int], crs, view_window:Window = None) -> Polygon:
    """Create a Polygon object for the negative sample boundary (without saving).
    
    Returns:
        Polygon object ready to be saved in batch.
    """
    minx, miny, maxx, maxy = boundary.bounds
    
    # Calculate size in meters using boundary (which is created from view_window)
    size_m = calculate_bbox_size_meters(boundary.bounds, crs)
    
    return Polygon(
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
    output_path = ROOT / save_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

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

def sampler(area: str, start_id: int, num_to_sample: int):
    tif_path = get_area_tif(area)
    tif_img = load_tif(tif_path)
    boundaries = get_all_polygon_boundaries(get_area_labels(area), tif_img.crs)
    prepared_boundaries = [prep(boundary) for boundary in boundaries]
    nodata_value = tif_img.nodata
    sampled_boundaries = []
    polygons_to_save = []  # Batch metadata saves
    while (sampled := len(sampled_boundaries)) < num_to_sample:
        print(tabbed(f"Sampling negative {start_id + sampled + 1}/{start_id + num_to_sample} for {area}"))
        view_window, boundary = sample_boundary(tif_img, area)
        if not_overlapping(boundary, prepared_boundaries) and in_actual_data(tif_img, view_window, nodata_value):
            sampled_boundaries.append(boundary)
            # Create Polygon object and add to batch save list
            save_path = make_negative_path(area, sampled + start_id)
            img_shape = save_boundary_image(tif_img, view_window, save_path)
            polygon = _create_negative_polygon(boundary, area, len(sampled_boundaries), save_path, img_shape, tif_img.crs, view_window)
            polygons_to_save.append(polygon)
    
    return sampled_boundaries, polygons_to_save

def extract_negatives_area_parallel(area:str) -> tuple[list[ShapelyPolygon], list[Polygon]]:
    num_to_sample = _negatives_per_area(area)
    num_processes = min(4, num_to_sample)  # Limit to 4 processes or number of samples
    samples_per_process = num_to_sample // num_processes
    extra_samples = num_to_sample % num_processes

    args_list = []
    start_id = 0
    for i in range(num_processes):
        count = samples_per_process + (1 if i < extra_samples else 0)  # Distribute extra samples
        args_list.append((area, start_id, count))
        start_id += count

    with Pool(processes=num_processes) as pool:
        try:
            results = pool.starmap(sampler, args_list)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

    all_boundaries = [boundary for result in results for boundary in result[0]]
    all_polygons = [polygon for result in results for polygon in result[1]]

    PolygonData.save_polygons(all_polygons)

    tif_img = load_tif(get_area_tif(area))

    return all_boundaries, tif_img

def save_negatives_boundaries(boundaries:list[ShapelyPolygon], area:str, area_crs:str) -> None:
    """Save the negative sample boundaries as a GeoJSON file."""
    gdf = gpd.GeoDataFrame(geometry=boundaries, crs=area_crs)
    save_path = PATHS[area]["negatives"] / f"{area}_negatives.geojson"
    gdf.to_file(save_path, driver="GeoJSON")
    print(tabbed(f"Saved negative boundaries for area {area} to {save_path}"))

if __name__ == "__main__":
    args = parse_args()
    areas = args.area

    for area in areas:
        save_boundaries, area_tif = extract_negatives_area_parallel(area)
        save_negatives_boundaries(save_boundaries, area, area_tif.crs)