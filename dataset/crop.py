
import argparse
import sys

import numpy as np
import geopandas as gpd
import rasterio

from pathlib import Path
from typing import Generator
from skimage.util import view_as_windows
from PIL import Image
from shapely.geometry import box, Polygon as ShapelyPolygon

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import ROOT, PATHS, CLASSES, PolygonData, WINDOW_SIZE, STRIDE, THRESHOLD_CROP_CONTENT, make_crop_path, load_img_array_from_path, get_area_tif, SCALE_FACTORS
from text import title, tabbed
from utils import Polygon, save_georeferenced_tif

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

def save_crops_boundaries(geo_ids: list[int], crop_ids: list[int], geometries: list, area: str, crs = None) -> None:
    """Save the boundaries of all geo crops as a GeoJSON file."""
    save_path = PATHS[area]["crops"] / f"{area}_geocrops.geojson"
    
    gdf = gpd.GeoDataFrame({
        "geo_id": geo_ids,
        "crop_id": crop_ids,
        "geometry": geometries
    }, crs=crs)

    gdf.to_file(save_path, driver="GeoJSON")
    print(tabbed(f"Saved all geo crops for area {area} to {save_path}"))

def make_crops(geo:Polygon, img_array:np.ndarray, crop_size:int, stride:int) -> Generator[tuple[np.ndarray, int, int], None, None]:
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
            crop_geometry = box(minx, miny, maxx, maxy)

            boundaries.append(crop_geometry)
            rows.append(int(i))
            cols.append(int(j))
            proportions.append(float(proportion))
            
            if proportion > best_proportion:
                best_proportion = proportion
                best_crop = view[i, j, 0].copy()
            
            if proportion >= THRESHOLD_CROP_CONTENT:
                yield view[i, j, 0], crop_geometry, i, j
                crop_count += 1
    
    # If no crops were generated, yield the best one we found
    if crop_count == 0 and best_crop is not None and best_proportion > 0.0:
        yield best_crop, crop_geometry, -1, -1

def get_polygon_crops(polygon:Polygon, crop_size:int=WINDOW_SIZE, stride:int=STRIDE) -> Generator[tuple[np.ndarray, int, int], None, None]:
    """Generate crops from the polygon's image.
    
    Args:
        polygon: Polygon object containing the image path.
        crop_size: Size of each square crop.
        stride: Stride for moving the crop window.
    """
    img_array = load_img_array_from_path(polygon.resized_path)
    return make_crops(polygon, img_array, crop_size, stride)

from rasterio.transform import Affine

def save_polygon_crop(geo, crop_array:np.ndarray, save_path:Path, row:int, col:int, stride:int, original_transform=None, crs=None) -> Polygon:
    """Save a crop array as a georeferenced TIFF image.
    
    Args:
        geo: Polygon object to update with the crop path.
        crop_array: Crop image as a NumPy array.
        save_path: Path to save the crop image.
        row: Row index of the crop window.
        col: Column index of the crop window.
        stride: Stride of the crop window tracking.
        original_transform: Optional manually provided transform.
        crs: Optional manually provided CRS.
    """
    output_path = ROOT / save_path

    # Adjust path extension to .tif
    if output_path.suffix != '.tif':
        output_path = output_path.with_suffix('.tif')
        save_path = save_path.with_suffix('.tif')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if original_transform is None or crs is None:
        if str(geo.resized_path) != '.' and (ROOT / geo.resized_path).is_file():
            with rasterio.open(ROOT / geo.resized_path) as src:
                crs = src.crs
                original_transform = src.transform
        else:
            with rasterio.open(ROOT / geo.tif_path) as src:
                crs = src.crs
                base_transform = src.transform
            scale = SCALE_FACTORS[geo.area]
            original_transform = base_transform * Affine.scale(1 / scale, 1 / scale)
        
    x_offset = col * stride
    y_offset = row * stride
    
    new_transform = original_transform * Affine.translation(x_offset, y_offset)

    save_georeferenced_tif(crop_array, output_path, new_transform, crs)

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

def crop_area(area: str, geos=None, resized_arrays=None, transforms=None, crss=None) -> None:
    """Generate crops from resized polygon images for a single area.
    
    Args:
        area: Study area to process (e.g., 'unita', 'chugchug', 'lluta').
        geos: Optional list of Polygon objects.
        resized_arrays: Optional list of resized image arrays, matching `geos`.
        transforms: Optional list of resized transforms.
        crss: Optional list of CRSs.
    """
    print(title(f"Generating crops for polygons in area: {area}"))
    
    if geos is None:
        geos = list(PolygonData.polygons(area_filter=(area,), classes_filter=(CLASSES["geo"],)))

    geo_ids = []
    crop_ids = []
    geometries = []
    for i, geo in enumerate(geos):
        print(f"Generating crops for polygon ID {geo.id}...")
        
        orig_transform = transforms[i] if transforms else None
        crs = crss[i] if crss else None

        if resized_arrays is not None:
            img_array = resized_arrays[i]
            crop_generator = make_crops(geo, img_array, crop_size=WINDOW_SIZE, stride=STRIDE)
        else:
            crop_generator = get_polygon_crops(geo)
        
        for id, (geo_crop, crop_geometry, row, col) in enumerate(crop_generator):
            print(tabbed(f"Saving crop ID {id}..."))
            crop_path = make_crop_path(geo, area, id)
            updated_geo = save_polygon_crop(geo, geo_crop, crop_path, row, col, STRIDE, orig_transform, crs)
            
            print(tabbed(f"Adding crop shapely boundary to list for batch save..."))
            geo_ids.append(geo.id)
            crop_ids.append(id)
            geometries.append(crop_geometry)

        PolygonData.save_polygons([updated_geo])
    
    print(tabbed(f"Saving all crop boundaries for area {area} to GeoJSON..."))
    save_crops_boundaries(geo_ids, crop_ids, geometries, area, crs = crs)

    
if __name__ == "__main__":
    args = parse_arguments()
    areas = args.area

    for area in areas:
        crop_area(area)