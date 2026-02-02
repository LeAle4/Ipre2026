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

class Polygon:
    """Represents a single polygon (geoglyph, ground, or road) from the dataset.
    
    Encapsulates all metadata, geometry, and file paths for a polygon.
    """ 

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
        self.jpeg_path = Path(metadata["jpeg_path"])
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
            "jpeg_path": str(self.jpeg_path),
            "tif_path": str(self.tif_path),
            "overlay_path": str(self.overlay_path),
            "resized_path": str(self.resized_path),
            "crop_paths": [str(p) for p in self.crop_paths],
            "augmented_paths": [str(p) for p in self.augmented_paths],
        }

        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=4)

def pixels_to_coordinates(polygon:Polygon, pixel_coords:tuple[int, int]) -> tuple[float, float]:
    """Convert pixel coordinates within the polygon image to geographic coordinates.
    
    Args:
        polygon: Polygon object with georeferencing info.
        pixel_coords: Tuple of (x_pixel, y_pixel) coordinates.
        
    Returns:
        Tuple of (longitude, latitude) geographic coordinates.
    """
    x_pixel, y_pixel = pixel_coords
    x_min = polygon.coords['left']
    x_max = polygon.coords['right']
    y_min = polygon.coords['bottom']
    y_max = polygon.coords['top']
    img_width, img_height = polygon.shape[1], polygon.shape[0]

    lon = x_min + (x_pixel / img_width) * (x_max - x_min)
    lat = y_max - (y_pixel / img_height) * (y_max - y_min)  # Invert y-axis for latitude

    return (lon, lat)

if __name__ == "__main__":
    pass