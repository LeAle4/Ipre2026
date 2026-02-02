"""Utility functions for managing geoglyph polygon data across different study areas.

Provides a unified interface for accessing polygon data from Unita, ChugChug, and Lluta sites.
Includes path management, polygon metadata parsing, and filtering by class.
"""
import json
import shapely
import numpy as np

from PIL import Image
from typing import Generator
from pathlib import Path


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

class Polygon:
    """Represents a single polygon (geoglyph, ground, or road) from the dataset.
    
    Encapsulates all metadata, geometry, and file paths for a polygon.
    """ 

    def load_from_extract_metadata(self, metadata:dict, area:str)-> "Polygon":
        # Basic properties
        self.id = metadata["polygon_index"]
        self.class_id = metadata["class"]
        self.area = area

        #Image spatial properties
        self.polygon_points = metadata["polygon_points"][0]["exterior"]
        self.shape = (metadata["image_shape"]["width"], metadata["image_shape"]["height"], metadata["image_shape"]["channels"])
        self.size_m = (metadata["bbox_size_meters"]["width_m"], metadata["bbox_size_meters"]["height_m"])
        self.coords = {"top": metadata["bounds"]["miny"], "left": metadata["bounds"]["minx"],
                       "bottom": metadata["bounds"]["maxy"], "right": metadata["bounds"]["maxx"]}
        self.polygon = shapely.geometry.Polygon(self.polygon_points)
        # File paths: JPEG, TIF, and overlay images
        paths = _get_path_from_area(area)
        polygons_path = paths["polygons"]

        self.jpeg_path = polygons_path / metadata["files"]["ortho_jpeg"]
        self.tif_path = polygons_path / metadata["files"]["ortho_tif"]
        self.overlay_path = polygons_path / metadata["files"]["overlay_jpeg"]
        self.resized_path = Path("") 
        self.crop_paths = []
        self.augmented_paths = []

        return self

    def load_from_metadata(self, path:Path) -> "Polygon":
        """Load polygon metadata from a JSON file.
        
        Args:
            path: Path to the metadata JSON file.
        """
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        self.id = metadata["id"]
        self.class_id = metadata["class_id"]
        self.area = metadata["area"]
        self.polygon_points = metadata["polygon_points"]
        self.shape = tuple(metadata["shape"])
        self.size_m = tuple(metadata["size_m"])
        self.coords = metadata["coords"]
        self.tif_path = Path(metadata["tif_path"])
        self.overlay_path = Path(metadata["overlay_path"])
        self.resized_path = Path(metadata["resized_path"])
        self.crop_paths = [Path(p) for p in metadata["crop_paths"]]
        self.augmented_paths = [Path(p) for p in metadata["augmented_paths"]]
        self.polygon = shapely.geometry.Polygon(self.polygon_points)

        return self

    def save_metadata(self, save_path:Path) -> None:
        """Save the polygon's metadata to a JSON file.
        
        Args:
            save_path: Path to save the metadata JSON file.
        """
        metadata = {
            "id": self.id,
            "class_id": self.class_id,
            "area": self.area,
            "polygon_points": self.polygon_points,
            "shape": self.shape,
            "size_m": self.size_m,
            "coords": self.coords,
            "tif_path": str(self.tif_path),
            "overlay_path": str(self.overlay_path),
            "resized_path": str(self.resized_path),
            "crop_paths": [str(p) for p in self.crop_paths],
            "augmented_paths": [str(p) for p in self.augmented_paths],
        }

        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=4)

def _get_path_from_area(area_name:str) -> dict:
    """Get path dictionary for a specific study area.
    
    Args:
        area_name: Name of the area ('unita', 'chugchug', or 'lluta')
        
    Returns:
        Dictionary with 'raw', 'polygons', and 'summary' path keys
    """
    if area_name == "unita":
        return UNITA_PATHS
    elif area_name == "chugchug":
        return CHUGCHUG_PATHS
    elif area_name == "lluta":
        return LLUTA_PATHS
    else:
        raise ValueError(f"Unknown area name: {area_name}")

def _parse_polygons_from_summary(area:str) -> Generator[Polygon, None, None]:
    """Parse all polygons from an area's summary JSON file.
    
    Args:
        area: Name of the area to parse
        
    Yields:
        Polygon objects for each entry in the summary file
    """
    summary_path = _get_path_from_area(area)["summary"]
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    for polygon in summary_data["polygons"]:
        poly_obj = Polygon().load_from_extract_metadata(polygon, area)
        yield poly_obj


def get_polygons_from_summary(area_name:str, classes: tuple = CLASS_IDS) -> tuple[Polygon, ...]:
    """Get polygons from an area, optionally filtered by class.
    
    Args:
        area_name: Name of the area ('unita', 'chugchug', or 'lluta')
        classes: Tuple of class IDs to include (default: all classes)
        
    Returns:
        Tuple of Polygon objects matching the specified classes
    """
    if len(classes)<=3:
        return tuple(filter(lambda p: p.class_id in classes, _parse_polygons_from_summary(area_name)))
    else:
        raise ValueError(f"Classes must be a subset of {CLASS_IDS}, got {classes}")


def get_geos_from_summary(area:str):
    """Get only geoglyph polygons (class 1) from a specific area.
    
    Args:
        area: Name of the area ('unita', 'chugchug', or 'lluta')
        
    Returns:
        Tuple of Polygon objects with class_id=1 (geoglyphs)
    """
    return get_polygons_from_summary(area, classes=(CLASSES["geo"],))

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

def pixels_to_coordinates(polygon:Polygon, pixel_coords:tuple[int, int]) -> tuple[float, float]:
    """Convert pixel coordinates within the polygon image to geographic coordinates.
    
    Args:
        polygon: Polygon object with georeferencing info.
        pixel_coords: Tuple of (x_pixel, y_pixel) coordinates.
        
    Returns:
        Tuple of (longitude, latitude) geographic coordinates.
    """
    x_pixel, y_pixel = pixel_coords
    x_min, y_min, x_max, y_max = polygon.coords.values()
    img_width, img_height = polygon.shape[1], polygon.shape[0]

    lon = x_min + (x_pixel / img_width) * (x_max - x_min)
    lat = y_max - (y_pixel / img_height) * (y_max - y_min)  # Invert y-axis for latitude

    return (lon, lat)

def title(text:str) -> str:
    """Convert a string to title case, replacing underscores with spaces.
    
    Args:
        text: Input string.
        
    Returns:
        Title-cased string.
    """
    txt = "-"*40 + "\n"
    txt += text + "\n"
    txt += "-"*40
    return txt

def tabbed(text:str, n_tabs:int=1) -> str:
    """Indent each line of the input text by a specified number of tabs.
    
    Args:
        text: Input string.
        n_tabs: Number of tabs to indent.
        
    Returns:
        Indented string.
    """
    tab_str = "\t" * n_tabs
    return tab_str + text

if __name__ == "__main__":
    pass