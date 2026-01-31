"""Utility functions for managing geoglyph polygon data across different study areas.

Provides a unified interface for accessing polygon data from Unita, ChugChug, and Lluta sites.
Includes path management, polygon metadata parsing, and filtering by class.
"""
import json
import shapely
from typing import Generator
from pathlib import Path

# Project directory structure
PROJECT_PATH = Path(__file__).resolve().parent
DATA_DIR = PROJECT_PATH / "data"

# Path mappings for Unita study area
UNITA_PATHS = {
    "raw": DATA_DIR / "unita_raw",
    "polygons": DATA_DIR / "unita_polygons",
    "summary": DATA_DIR / "unita_polygons" / "unita_summary.json",
}

# Path mappings for ChugChug study area
CHUGCHUG_PATHS = {
    "raw": DATA_DIR / "chugchug_raw",
    "polygons": DATA_DIR / "chugchug_polygons",
    "summary": DATA_DIR / "chugchug_polygons" / "chugchug_summary.json",
}

# Path mappings for Lluta study area
LLUTA_PATHS = {
    "raw": DATA_DIR / "lluta_raw",
    "polygons": DATA_DIR / "lluta_polygons",
    "summary": DATA_DIR / "lluta_polygons" / "lluta_summary.json",
}

# Polygon class mappings: geoglyphs, ground, and road
CLASSES = {
    "geo":1,
    "ground":2,
    "road":3
}
CLASS_IDS = tuple(CLASSES.values())  # (1, 2, 3)

class Polygon:
    """Represents a single polygon (geoglyph, ground, or road) from the dataset.
    
    Encapsulates all metadata, geometry, and file paths for a polygon.
    """
    def __init__(self, metadata:dict, area:str):
        # Basic properties
        self.id = metadata["polygon_index"]
        self.class_id = metadata["class"]
        self.area = area

        #Image spatial properties
        self.coordinates = metadata["polygon_points"]["exterior"]
        self.size_px = (metadata["image_shape"]["width"], metadata["image_shape"]["height"])
        self.size_m = (metadata["bbox_size_meters"]["width_m"], metadata["bbox_size_meters"]["height_m"])
        self.coords = metadata["bounds"]
        self.polygon = shapely.geometry.Polygon(self.coordinates)

        # File paths: JPEG, TIF, and overlay images
        paths = _get_path_from_area(area)
        polygons_path = paths["polygons"]

        self.jpeg_path = polygons_path / metadata["files"]["ortho_jpeg"]
        self.tif_path = polygons_path / metadata["files"]["ortho_tif"]
        self.overlay_path = polygons_path / metadata["files"]["overlay_jpeg"]

        # Store complete metadata for additional access
        self.metadata = metadata


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

def _parse_polygons_from_area(area:str) -> Generator[Polygon, None, None]:
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
        yield Polygon(polygon, area)

def get_polygons(area_name:str, classes: tuple = CLASS_IDS) -> tuple:
    """Get polygons from an area, optionally filtered by class.
    
    Args:
        area_name: Name of the area ('unita', 'chugchug', or 'lluta')
        classes: Tuple of class IDs to include (default: all classes)
        
    Returns:
        Tuple of Polygon objects matching the specified classes
    """
    if len(classes) < 3:
        filtered = []
        for polygon in _parse_polygons_from_area(area_name):
            if polygon.class_id in classes:
                filtered.append(polygon)
        return tuple(filtered)
    elif len(classes) == 3:
        return tuple(_parse_polygons_from_area(area_name))
    else:
        raise ValueError(f"Classes must be a subset of {CLASS_IDS}")

def get_geos(area:str):
    """Get only geoglyph polygons (class 1) from a specific area.
    
    Args:
        area: Name of the area ('unita', 'chugchug', or 'lluta')
        
    Returns:
        Tuple of Polygon objects with class_id=1 (geoglyphs)
    """
    return get_polygons(area, classes=(CLASSES["geo"],))

if __name__ == "__main__":
    pass