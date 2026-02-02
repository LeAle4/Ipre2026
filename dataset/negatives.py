import sys
import argparse

from pathlib import Path

import rasterio
import geopandas as gpd
from shapely.geometry import box, Polygon as ShapelyPolygon

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import NEGATIVES_RATIO, PATHS, AREA_NAMES, CLASS_IDS, CLASS_NAMES, WINDOW_SIZE, POLYGON_DATA_DIR, get_area_tif, get_area_labels
from utils import Polygon

def _negatives_per_area(area:str) -> int:
    """Count the number of crop files in the specified study area.
    
    Args:
        area: Study area name (e.g., 'unita', 'chugchug', 'lluta').
    """
    crop_dir = PATHS[area]["crops"]
    crop_files = list(crop_dir.rglob("*.png"))
    return len(crop_files) * NEGATIVES_RATIO

def load_tif(tif_path:Path):
    raise NotImplementedError("This function is not yet implemented.")

def get_all_polygon_boundaries(boundary_path: Path)-> list[ShapelyPolygon]:
    raise NotImplementedError("This function is not yet implemented.")

def check_negative_validity(negative_boundary:tuple[float,float], boundaries:tuple[tuple[float,float],...], min_distance:float = 0.0001) -> bool:
    raise NotImplementedError("This function is not yet implemented.")

def sample_boundary(tif_area, window_size:int = WINDOW_SIZE) -> tuple[tuple[float,float],...]:
    raise NotImplementedError("This function is not yet implemented.")

def save_boundary_as_geo() -> None:
    raise NotImplementedError("This function is not yet implemented.")

def save_boundary_image(boundary:tuple[float,float], save_path:Path) -> None:
    raise NotImplementedError("This function is not yet implemented.")

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

if __name__ == "__main__":
    args = parse_args()
    areas = args.area

    for area in areas:
        sampled = 0
        limit = _negatives_per_area(area)
        print(f"Extracting negative samples for area: {area}, amount to sample: {limit}")

        #load the area tif
        tif_path = get_area_tif(area)
        labels_path = get_area_labels(area)

        boundaries = get_all_polygon_boundaries(labels_path)
        tif_img = load_tif(tif_path)

        while sampled < limit:
            boundary = sample_boundary(tif_path)
            if check_negative_validity(boundary, boundaries):
                sampled += 1
                print(f"Sampled negative {sampled}/{limit} for area {area} at boundary {boundary}")
                boundaries.append(boundary)
                save_path = make_negative_path(area, sampled)
                save_boundary_image(boundary, save_path)
                save_boundary_as_geo(boundary, area, sampled)

        
