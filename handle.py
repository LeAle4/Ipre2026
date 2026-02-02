# Core imports
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image

from utils import Polygon


# Project directory structure
PROJECT_PATH = Path(__file__).resolve().parent
DATA_DIR = PROJECT_PATH / "data"
POLYGON_DATA_DIR = DATA_DIR / "polygon_data"

# Path mappings for Unita study area
UNITA_PATHS = {
    "raw": DATA_DIR / "unita_raw",
    "polygons": DATA_DIR / "unita_polygons",
    "summary": DATA_DIR / "unita_polygons" / "summary.json",
    "resized": DATA_DIR / "unita_resized",
    "crops": DATA_DIR / "unita_crops",
}

# Path mappings for ChugChug study area
CHUGCHUG_PATHS = {
    "raw": DATA_DIR / "chugchug_raw",
    "polygons": DATA_DIR / "chugchug_polygons",
    "summary": DATA_DIR / "chugchug_polygons" / "summary.json",
    "resized": DATA_DIR / "chugchug_resized",
    "crops": DATA_DIR / "chugchug_crops",
}
# Path mappings for Lluta study area
LLUTA_PATHS = {
    "raw": DATA_DIR / "lluta_raw",
    "polygons": DATA_DIR / "lluta_polygons",
    "summary": DATA_DIR / "lluta_polygons" / "summary.json",
    "resized": DATA_DIR / "lluta_resized",
    "crops": DATA_DIR / "lluta_crops",
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

SCALES = {'unita': 0.886, 'lluta': 0.218, 'chugchug': 0.18}
WINDOW_SIZE = 224
STRIDE = int(WINDOW_SIZE / 2)
THRESHOLD_CROP_CONTENT = 0.8  # Minimum fraction of geoglyph pixels in a crop to be considered valid
NEGATIVES_RATIO = 3 # Number of negative samples per positive sample

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

def geos_from_polygon_data(area) -> Generator[Polygon, None, None]:
    """Generator yielding Polygon objects from polygon data directory, filtered by area and class.
    Args:
        area_filter: Tuple of area names to include (e.g., ('unita', 'chugchug')).
        class_filter: Tuple of class IDs to include (default: (1,) for geoglyphs).
    """
    for metadata_file in POLYGON_DATA_DIR.glob("*_metadata.json"):
        polygon = Polygon().load_from_metadata(metadata_file)
        if polygon.area == area and polygon.class_id == CLASSES["geo"]:
            yield polygon

def load_img_array_from_path(path:Path) -> np.ndarray:
    """Load a GeoTIFF image from the given path and return as a NumPy array.
    
    Args:
        tif_path: Path to the GeoTIFF file.
    """
    return np.array(Image.open(path))

def make_resized_path(geo:Polygon, area:str) -> Path:
    """Construct the path for the resized polygon image.
    
    Args:
        geo: Polygon object.
    """
    resized_dir = PATHS[area]["resized"]
    resized_dir.mkdir(parents=True, exist_ok=True)
    return resized_dir / f"{geo.area}_class{geo.class_id}_{geo.id}_resized.png"

def make_crop_path(geo:Polygon, area:str, crop_id:int) -> Path:
    """Construct the path for a specific crop of the polygon image.
    
    Args:
        geo: Polygon object.
        crop_id: Identifier for the crop.
    """
    crop_dir = PATHS[area]["crops"] / f"geo_{geo.id}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    return crop_dir / f"{geo.area}_class{geo.class_id}_{geo.id}_crop{crop_id}.png"
