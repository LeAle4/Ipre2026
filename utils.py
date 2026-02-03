"""Utility functions for managing geoglyph polygon data across different study areas.

Provides a unified interface for accessing polygon data from Unita, ChugChug, and Lluta sites.
Includes path management, polygon metadata parsing, and filtering by class.
"""
import json
import shapely

from typing import Optional
from pathlib import Path
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Geod, Transformer

class Polygon:
    """Represents a single polygon (geoglyph, ground, or road) from the dataset.
    
    Encapsulates all metadata, geometry, and file paths for a polygon.
    """ 
    def __init__(self, id:int = 0, class_id:int = 0, area:str = "", polygon_points:list = [], shape = (0,0), size_m = (0.0,0.0),
                 coords = {}, jpeg_path = Path(), tif_path = Path(), overlay_path = Path(),
                 resized_path = Path(), crop_paths = [], augmented_paths = [], polygon:Optional[shapely.geometry.Polygon] = None):
        self.id = id
        self.class_id = class_id
        self.area = area
        self.polygon_points = polygon_points
        self.shape = shape
        self.size_m = size_m
        self.coords = coords
        self.jpeg_path = jpeg_path
        self.tif_path = tif_path
        self.overlay_path = overlay_path
        self.resized_path = resized_path
        self.crop_paths = crop_paths
        self.augmented_paths = augmented_paths
        self.polygon = polygon

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

def calculate_bbox_size_meters(bounds, crs):
    """Calculate the size of a bounding box in meters."""
    minx, miny, maxx, maxy = bounds

    # Convert to WGS84 if needed
    if crs and crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        p1_wgs84 = transform(transformer.transform, Point(minx, miny))
        p2_wgs84 = transform(transformer.transform, Point(maxx, miny))
        p3_wgs84 = transform(transformer.transform, Point(minx, maxy))
        
        minx_wgs, miny_wgs = p1_wgs84.x, p1_wgs84.y
        maxx_wgs = p2_wgs84.x
        maxy_wgs = p3_wgs84.y
    else:
        minx_wgs, miny_wgs = minx, miny
        maxx_wgs, maxy_wgs = maxx, maxy

    geod = Geod(ellps="WGS84")
    _, _, width_m = geod.inv(minx_wgs, miny_wgs, maxx_wgs, miny_wgs)
    _, _, height_m = geod.inv(minx_wgs, miny_wgs, minx_wgs, maxy_wgs)

    return (abs(width_m), abs(height_m))

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