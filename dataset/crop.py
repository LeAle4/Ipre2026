
import argparse
import sys

import numpy as np
import geopandas as gpd
import rasterio

from pathlib import Path
from typing import Generator
from skimage.util import view_as_windows
from PIL import Image
from shapely.geometry import box

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import ROOT, PATHS, CLASSES, PolygonData, WINDOW_SIZE, STRIDE, THRESHOLD_CROP_CONTENT, make_crop_path, load_img_array_from_path, get_area_tif
from text import title, tabbed
from utils import Polygon

def calculate_crop_proportion(geo:Polygon, crop_borders:tuple[float,float,float,float]) -> float:
    """Calculate the proportion of polygon area inside the crop.
    Args:
        geo: Polygon object containing the mask path.
        crop_borders: Tuple of (minx, miny, maxx, maxy) borders of the crop in geographic coordinates.
    Returns:
        Proportion of crop area covered by polygon (0.0 to 1.0).
    """
    crop_polygon = box(*crop_borders)
    intersection = crop_polygon.intersection(geo.polygon)
    crop_area = crop_polygon.area
    if crop_area <= 0 or intersection.is_empty:
        return 0.0
    return intersection.area / crop_area

def _calculate_crop_borders(i:int, j:int, geo:Polygon, window_size:int, stride:int) -> tuple[float,float,float,float]:
    """Calculate crop borders in geographic coordinates from pixel window placement.
    
    Args:
        i: Row index of the crop.
        j: Column index of the crop.
        geo: Polygon object containing bounds and image shape.
        window_size: Crop size in pixels.
        stride: Stride used for cropping.
    """
    img_height, img_width = geo.shape[0], geo.shape[1]

    x0_px = j * stride
    y0_px = i * stride
    x1_px = min(x0_px + window_size, img_width)
    y1_px = min(y0_px + window_size, img_height)

    x_left = geo.coords["left"]
    x_right = geo.coords["right"]
    y_top = geo.coords["top"]
    y_bottom = geo.coords["bottom"]

    x_min = min(x_left, x_right)
    x_max = max(x_left, x_right)
    y_min = min(y_top, y_bottom)
    y_max = max(y_top, y_bottom)

    crop_minx = x_min + (x0_px / img_width) * (x_max - x_min)
    crop_maxx = x_min + (x1_px / img_width) * (x_max - x_min)
    crop_maxy = y_max - (y0_px / img_height) * (y_max - y_min)
    crop_miny = y_max - (y1_px / img_height) * (y_max - y_min)

    return (min(crop_minx, crop_maxx), min(crop_miny, crop_maxy),
            max(crop_minx, crop_maxx), max(crop_miny, crop_maxy))

def save_crop_boundaries_geojson(geo: Polygon, boundaries: list, rows: list[int], cols: list[int], proportions: list[float]) -> None:
    """Persist evaluated crop boundaries to a GeoJSON file for analysis."""
    output_dir = PATHS[geo.area]["crops"] / f"geo_{geo.id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{geo.area}_class{geo.class_id}_{geo.id}_crop_boundaries.geojson"
    area_tif = get_area_tif(geo.area)
    with rasterio.open(area_tif) as src:
        area_crs = src.crs

    gdf = gpd.GeoDataFrame(
        {
            "polygon_id": int(geo.id),
            "area": geo.area,
            "class_id": int(geo.class_id),
            "row": rows,
            "col": cols,
            "proportion": proportions,
        },
        geometry=boundaries,
        crs=area_crs,
    )
    gdf.to_file(output_path, driver="GeoJSON")

def make_crops(geo:Polygon, img_array:np.ndarray, crop_size:int, stride:int) -> Generator[np.ndarray, None, None]:
    """Generate crops from the input image array. Guarantees at least one crop per polygon.
    
    Args:
        img_array: Input image as a NumPy array.
        crop_size: Size of each square crop.
        stride: Stride for moving the crop window.
    """
    print(tabbed(f"Image shape: {img_array.shape}, crop size: {crop_size}, stride: {stride}"))
    view = view_as_windows(img_array, (crop_size, crop_size, img_array.shape[2]), step=stride)

    crop_count = 0
    best_crop = None
    best_proportion = 0.0
    boundaries = []
    rows = []
    cols = []
    proportions = []
    
    for i in range(view.shape[0]):
        for j in range(view.shape[1]):
            crop_borders = _calculate_crop_borders(i, j, geo, crop_size, stride)
            proportion = calculate_crop_proportion(geo, crop_borders)

            minx, miny, maxx, maxy = crop_borders

            boundaries.append(box(minx, miny, maxx, maxy))
            rows.append(int(i))
            cols.append(int(j))
            proportions.append(float(proportion))
            
            if proportion > best_proportion:
                best_proportion = proportion
                best_crop = view[i, j, 0].copy()
            
            if proportion >= THRESHOLD_CROP_CONTENT:
                yield view[i, j, 0]
                crop_count += 1
    
    # If no crops were generated, yield the best one we found
    if crop_count == 0 and best_crop is not None and best_proportion > 0.0:
        yield best_crop

    save_crop_boundaries_geojson(geo, boundaries, rows, cols, proportions)

def get_polygon_crops(polygon:Polygon, crop_size:int=WINDOW_SIZE, stride:int=STRIDE) -> Generator[np.ndarray, None, None]:
    """Generate crops from the polygon's image.
    
    Args:
        polygon: Polygon object containing the image path.
        crop_size: Size of each square crop.
        stride: Stride for moving the crop window.
    """
    img_array = load_img_array_from_path(polygon.resized_path)
    return make_crops(polygon, img_array, crop_size, stride)

def save_polygon_crop(geo, crop_array:np.ndarray, save_path:Path) -> None:
    """Save a crop array as an image.
    
    Args:
        geo: Polygon object to update with the crop path.
        crop_array: Crop image as a NumPy array.
        save_path: Path to save the crop image.
    """
    img = Image.fromarray(crop_array)
    output_path = ROOT / save_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    geo.crop_paths.append(save_path)
    return geo

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate crops from resized polygon images.")
    parser.add_argument(
        "--area",
        type=str,
        nargs = "+",
        choices=["unita", "chugchug", "lluta"],
        required=True,
        help="Study area to process.",
    )
    return parser.parse_args()

def crop_area(area: str) -> None:
    """Generate crops from resized polygon images for a single area.
    
    Args:
        area: Study area to process (e.g., 'unita', 'chugchug', 'lluta').
    """
    print(title(f"Generating crops for polygons in area: {area}"))
    for geo in PolygonData.polygons(area_filter = (area,), classes_filter=(CLASSES["geo"],)):
        print(f"Generating crops for polygon ID {geo.id}...")
        for id, geo_crop in enumerate(get_polygon_crops(geo)):
            print(tabbed(f"Saving crop ID {id}..."))
            crop_path = make_crop_path(geo, area, id)
            updated_geo = save_polygon_crop(geo, geo_crop, crop_path)
        
        PolygonData.save_polygons([updated_geo])

if __name__ == "__main__":
    args = parse_arguments()
    areas = args.area

    for area in areas:
        crop_area(area)