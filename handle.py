# Core imports
from utils import get_file_resolution
import json
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image

from utils import Polygon

# Project directory structure
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
POLYGON_DATA_FILE = DATA_DIR / "polygon_data.json"

# Path mappings for Unita study area
UNITA_PATHS = {
    "raw": DATA_DIR / "unita_raw",
    "polygons": DATA_DIR / "unita_polygons",
    "summary": DATA_DIR / "unita_polygons" / "summary.json",
    "resized": DATA_DIR / "unita_resized",
    "crops": DATA_DIR / "unita_crops",
    "negatives": DATA_DIR / "unita_negatives",
}

# Path mappings for ChugChug study area
CHUGCHUG_PATHS = {
    "raw": DATA_DIR / "chugchug_raw",
    "polygons": DATA_DIR / "chugchug_polygons",
    "summary": DATA_DIR / "chugchug_polygons" / "summary.json",
    "resized": DATA_DIR / "chugchug_resized",
    "crops": DATA_DIR / "chugchug_crops",
    "negatives": DATA_DIR / "chugchug_negatives",
}
# Path mappings for Lluta study area
LLUTA_PATHS = {
    "raw": DATA_DIR / "lluta_raw",
    "polygons": DATA_DIR / "lluta_polygons",
    "summary": DATA_DIR / "lluta_polygons" / "summary.json",
    "resized": DATA_DIR / "lluta_resized",
    "crops": DATA_DIR / "lluta_crops",
    "negatives": DATA_DIR / "lluta_negatives",
}

PATHS = {
    "unita": UNITA_PATHS,
    "chugchug": CHUGCHUG_PATHS,
    "lluta": LLUTA_PATHS,
}

# Polygon class mappings: geoglyphs, ground, and road
CLASSES = {
    "geo":1,
    "ground":2,
    "road":3
}
CLASS_IDS = tuple(CLASSES.values())  # (1, 2, 3)
CLASS_NAMES = tuple(CLASSES.keys())  # ('geo', 'ground', 'road')
AREA_NAMES = tuple(PATHS.keys())  # ('unita', 'chugchug', 'lluta')

#Change to be calculated
TARGET_SCALE = 0.05  # Desired scale in meters per pixel for the resized images
WINDOW_SIZE = 224
STRIDE = int(WINDOW_SIZE / 2)
THRESHOLD_CROP_CONTENT = 0.4  # Minimum fraction of geoglyph pixels in a crop to be considered valid
NEGATIVES_RATIO = 3 # Number of negative samples per positive sample

class PolygonData:
    dir = DATA_DIR
    polygon_data_file = POLYGON_DATA_FILE

    if not polygon_data_file.exists():
        polygon_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(polygon_data_file, "w") as f:
            json.dump({}, f)

    @classmethod
    def polygons(cls, area_filter: tuple[str, ...], classes_filter: tuple[int, ...]) -> Generator[Polygon, None, None]:
        """Generator yielding Polygon objects from polygon data directory.
        """
        with open(cls.polygon_data_file, "r") as f:
            polygon_data_dict = json.load(f)

            for area in area_filter:
                if area not in polygon_data_dict:
                    raise ValueError(f"Area '{area}' not found in polygon data.")
                for class_id in classes_filter:
                    class_key = str(class_id)
                    if class_key not in polygon_data_dict[area]:
                        raise ValueError(f"Class ID '{class_id}' not found in area '{area}' polygon data.")
                    for polygon_info in polygon_data_dict[area][class_key].values():
                        yield Polygon().load_from_metadata(polygon_info)

    @classmethod
    def crops_in_area(cls, area:str) -> int:
        """Count the number of crop files in the specified study area.
        
        Args:
            area: Study area name (e.g., 'unita', 'chugchug', 'lluta').
        """
        return sum(polygon.num_crops() for polygon in cls.polygons(area_filter=(area,), classes_filter=(CLASSES["geo"],)))
        
    @classmethod
    def positive_count(cls, areas:tuple[str, ...]) -> int:
        """Count the total number of positive samples (geoglyphs) across specified areas.
        
        Args:
            areas: Tuple of study area names to include in the count.
        """
        return sum(1 for _ in cls.polygons(area_filter=areas, classes_filter=(CLASSES["geo"],)))

    @classmethod
    def negative_count(cls, areas:tuple[str, ...]) -> int:
        """Calculate the total number of negative samples.
        
        Args:
            areas: Tuple of study area names to include in the count.
        """
        return sum(1 for _ in cls.polygons(area_filter=areas, classes_filter=(CLASSES["ground"],)))

    @classmethod
    def save_polygon_metadata(cls, polygon_dict_list: list[dict]) -> None:
        """Save a list of polygon metadata dictionaries to the polygon data JSON file.
        
        Args:
            polygon_dict_list: List of dictionaries containing polygon metadata.
        """
        # Load existing data
        if cls.polygon_data_file.exists():
            with open(cls.polygon_data_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        # Update with new data
        for polygon_metadata in polygon_dict_list:
            area = polygon_metadata["area"]
            id = polygon_metadata["id"]
            class_id = polygon_metadata["class_id"]
            class_key = str(class_id)
            if area not in existing_data:
                existing_data[area] = {str(id): {} for id in CLASS_IDS}
            if class_key not in existing_data[area]:
                existing_data[area][class_key] = {}
            existing_data[area][class_key][id]=polygon_metadata

        # Save updated data
        with open(cls.polygon_data_file, "w") as f:
            json.dump(existing_data, f, indent=4)

    @classmethod
    def save_polygons(cls, polygons: list[Polygon]) -> None:
        """Save a list of Polygon objects to the polygon data JSON file.
        
        Args:
            polygons: List of Polygon objects to save.
        """
        polygon_dict_list = [polygon.get_metadata() for polygon in polygons]
        cls.save_polygon_metadata(polygon_dict_list)

def calculate_area_scale(area:str) -> float:
    """Calculate the scale factor for a given study area based on the average size of geoglyphs.
    
    Args:
        area: Name of the study area ('unita', 'chugchug', or 'lluta').
    """
    area_polygons = tuple(PolygonData.polygons(area_filter=(area,), classes_filter=(CLASSES["geo"],)))
    scales = []
    for geo in area_polygons:
        pix_x, pix_y = geo.shape
        size_x_m, size_y_m = geo.size_m
        scale = (pix_x / size_x_m + pix_y / size_y_m)/2
        scales.append(scale)
    average_scale = sum(scales) / len(scales)
    return average_scale

def get_area_tif(area:str) -> Path:
    """Get the path to the orthomosaic GeoTIFF for the specified study area.
    
    Args:
        area: Name of the study area ('unita', 'chugchug', or 'lluta')."""
    raw_path = PATHS[area]["raw"]
    tif_file = raw_path.glob("*ortomosaico.tif")
    return next(tif_file)

def get_area_labels(area:str) -> Path:
    """Get the path to the labels GeoJSON for the specified study area.
    
    Args:
        area: Name of the study area ('unita', 'chugchug', or 'lluta')."""
    raw_path = PATHS[area]["raw"]
    geojson_file = raw_path.glob("*.gpkg")
    return next(geojson_file)

def get_area_DEM(area:str) -> Path:
    """Get the path to the DEM GeoTIFF for the specified study area.
    
    Args:
        area: Name of the study area ('unita', 'chugchug', or 'lluta')."""
    raw_path = PATHS[area]["raw"]
    dem_file = raw_path.glob("*DEM.tif")
    return next(dem_file)

def load_img_array_from_path(path:Path) -> np.ndarray:
    """Load a GeoTIFF image from the given path and return as a NumPy array.
    
    Args:
        tif_path: Path to the GeoTIFF file.
    """
    candidate_path = Path(path)
    resolved_path = ROOT / candidate_path
    return np.array(Image.open(resolved_path))

def make_jpeg_path(area, geo_id, class_id) -> Path:
    """Construct the path for the JPEG version of a polygon image.
    
    Args:
        area: Study area name.
        geo_id: Unique identifier for the geoglyph.
        class_id: Class ID of the polygon (e.g., 1 for geoglyph).
    """
    jpeg_dir = PATHS[area]["polygons"]
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    relative_jpeg_dir = jpeg_dir.relative_to(ROOT)
    return relative_jpeg_dir / f"{area}_class{class_id}_{geo_id}_ortho.jpg"

def make_tif_path(area, geo_id, class_id) -> Path:
    """Construct the path for the GeoTIFF version of a polygon image.
    
    Args:
        area: Study area name.
        geo_id: Unique identifier for the geoglyph.
        class_id: Class ID of the polygon (e.g., 1 for geoglyph).
    """
    tif_dir = PATHS[area]["polygons"]
    tif_dir.mkdir(parents=True, exist_ok=True)
    relative_tif_dir = tif_dir.relative_to(ROOT)
    return relative_tif_dir / f"{area}_class{class_id}_{geo_id}_ortho.tif"

def make_overlay_path(area, geo_id, class_id) -> Path:
    """Construct the path for the overlay image of a polygon.
    
    Args:
        area: Study area name.
        geo_id: Unique identifier for the geoglyph.
        class_id: Class ID of the polygon (e.g., 1 for geoglyph).
    """
    overlay_dir = PATHS[area]["polygons"]
    overlay_dir.mkdir(parents=True, exist_ok=True)
    relative_overlay_dir = overlay_dir.relative_to(ROOT)
    return relative_overlay_dir / f"{area}_class{class_id}_{geo_id}_overlay.jpg"

def make_resized_path(geo:Polygon, area:str) -> Path:
    """Construct the path for the resized polygon image.
    
    Args:
        geo: Polygon object.
    """
    resized_dir = PATHS[area]["resized"]
    resized_dir.mkdir(parents=True, exist_ok=True)
    relative_resized_dir = resized_dir.relative_to(ROOT)
    return relative_resized_dir / f"{geo.area}_class{geo.class_id}_{geo.id}_resized.tif"

def make_crop_path(geo:Polygon, area:str, crop_id:int) -> Path:
    """Construct the path for a specific crop of the polygon image.
    
    Args:
        geo: Polygon object.
        crop_id: Identifier for the crop.
    """
    crop_dir = PATHS[area]["crops"] / f"geo_{geo.id}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    relative_crop_dir = crop_dir.relative_to(ROOT)
    return relative_crop_dir / f"{geo.area}_class{geo.class_id}_{geo.id}_crop{crop_id}.png"

def make_negative_path(area:str, negative_id:int) -> Path:
    """Construct the path for a negative sample image.
    
    Args:
        area: Study area name.
        negative_id: Identifier for the negative sample.
    """
    negative_dir = PATHS[area]["negatives"]
    negative_dir.mkdir(parents=True, exist_ok=True)
    relative_negative_dir = negative_dir.relative_to(ROOT)
    return relative_negative_dir / f"{area}_class{CLASSES['ground']}_crop{negative_id}.png"

def count_crops(area:str):
    """
    Counts the number of crops in a given area.

    Args:
        area (str): The name of the area to count crops in.

    Returns:
        int: The number of crops in the specified area.
    """
    crop_path = PATHS[area]["crops"]
    sum = 0
    for element in crop_path.iterdir():
        if element.is_dir():
            sum += len(tuple(crop_path.joinpath(element).glob("*.tif")))
    
    return sum

def count_negatives(area):
    """
    Counts the number of negative samples in a given area.

    Args:
        area (str): The name of the area to count negative samples in.

    Returns:
        int: The number of negative samples in the specified area.
    """
    negative_path = PATHS[area]["negatives"]
    return len(tuple(negative_path.glob("*.png")))

SCALE_FACTORS = {area_name: get_file_resolution(get_area_tif(area_name)) / TARGET_SCALE for area_name in AREA_NAMES}