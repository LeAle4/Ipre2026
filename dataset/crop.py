
import argparse
import sys

import numpy as np

from pathlib import Path
from typing import Generator
from skimage.util import view_as_windows
from PIL import Image
from shapely.geometry import box

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import POLYGON_DATA_DIR, WINDOW_SIZE, STRIDE, THRESHOLD_CROP_CONTENT, geos_from_polygon_data, make_crop_path, load_img_array_from_path
from text import title, tabbed
from utils import Polygon, pixels_to_coordinates

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
    return intersection.area / crop_area

def valid_crop(geo:Polygon, crop_borders:tuple[float,float,float,float], threshold:float=THRESHOLD_CROP_CONTENT) -> bool:
    """Determine if a crop contains sufficient geoglyph content to be considered valid.
    Args:
        geo: Polygon object containing the mask path.
        crop_borders: Tuple of (minx, miny, maxx, maxy) borders of the crop in geographic coordinates.
        threshold: Minimum fraction of geoglyph pixels in the crop to be considered valid.
    """
    return calculate_crop_proportion(geo, crop_borders) >= threshold

def _calculate_crop_borders(i,j, geo:Polygon, window_size:int, stride: int) -> tuple[float,float,float,float]:
    """Calculate the borders of a crop in the original image coordinates.
    
    Args:
        i: Row index of the crop.
        j: Column index of the crop.
        geo: Polygon object containing the image shape and geotransform.
        stride: Stride used for cropping."""
    
    top_left = pixels_to_coordinates(geo, (j * stride, i * stride))
    bottom_right = pixels_to_coordinates(geo, (j * stride + window_size, i * stride + window_size))

    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])  # (minx, miny, maxx, maxy)


def pad_to_window_size(img_array:np.ndarray, window_size:int) -> np.ndarray:
    """Pad the input image array to ensure both dimensions are at least window_size.
    
    Args:
        img_array: Input image as a NumPy array.
        window_size: Minimum size for each dimension.
        
    Returns:
        Padded image as a NumPy array with dimensions at least (window_size, window_size, C).
    """
    H, W, C = img_array.shape
    
    # Calculate target dimensions (at least window_size, but keep larger dimensions)
    target_H = max(H, window_size)
    target_W = max(W, window_size)

    # Create new array filled with random noise
    padded_array = np.random.randint(0, 256, (target_H, target_W, C), dtype=img_array.dtype)

    # Copy original image into the center
    top = (target_H - H) // 2
    left = (target_W - W) // 2
    padded_array[top:top + H, left:left + W] = img_array

    return padded_array

def make_crops(geo:Polygon, img_array:np.ndarray, crop_size:int, stride:int) -> Generator[np.ndarray, None, None]:
    """Generate crops from the input image array. Guarantees at least one crop per polygon.
    
    Args:
        img_array: Input image as a NumPy array.
        crop_size: Size of each square crop.
        stride: Stride for moving the crop window.
    """
    small = False
    # If the image is smaller than the crop size, we add random noise to the borders so that the image can be passed to the network without resizing
    if img_array.shape[0] < crop_size or img_array.shape[1] < crop_size:
        padded_array = pad_to_window_size(img_array, crop_size)
        #Crop is small and could bypass threshold, we mark it as such
        small = True
    else:
        padded_array = img_array

    view = view_as_windows(padded_array, (crop_size, crop_size, padded_array.shape[2]), step=stride)

    crop_count = 0
    best_crop = None
    best_proportion = 0.0
    
    for i in range(view.shape[0]):
        for j in range(view.shape[1]):
            crop_borders = _calculate_crop_borders(i, j, geo, crop_size, stride)
            proportion = calculate_crop_proportion(geo, crop_borders)
            
            if proportion > best_proportion:
                best_proportion = proportion
                best_crop = view[i, j, 0].copy()
            
            if proportion >= THRESHOLD_CROP_CONTENT or small:
                yield view[i, j, 0]
                crop_count += 1
    
    # If no crops were generated, yield the best one we found
    if crop_count == 0 and best_crop is not None:
        yield best_crop

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
    img.save(save_path)
    geo.crop_paths.append(save_path)

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
    for geo in geos_from_polygon_data(area):
        print(f"Generating crops for polygon ID {geo.id}...")
        for id, geo_crop in enumerate(get_polygon_crops(geo)):
            print(tabbed(f"Saving crop ID {id}..."))
            crop_path = make_crop_path(geo, area, id)
            save_polygon_crop(geo, geo_crop, crop_path)
        
        geo.save_metadata(POLYGON_DATA_DIR)


if __name__ == "__main__":
    args = parse_arguments()
    areas = args.area

    for area in areas:
        crop_area(area)