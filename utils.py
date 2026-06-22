"""Utility functions for managing geoglyph polygon data across different study areas.

Provides a unified interface for accessing polygon data from Unita, ChugChug, and Lluta sites.
Includes path management, polygon metadata parsing, and filtering by class.
"""
import shapely

from typing import Optional
from pathlib import Path
from shapely.geometry import Point
from shapely.ops import transform
import rasterio
from pyproj import Geod, Transformer
import numpy as np

def save_georeferenced_tif(array: np.ndarray, output_path: Path, transform, crs) -> None:
    """Save a numpy array as a georeferenced TIF.
    
    Args:
        array: Numpy array (H, W) or (H, W, C) or (C, H, W).
        output_path: Path object to save the TIF file (absolute or relative to ROOT).
        transform: Georeferencing transform (e.g. from rasterio / Affine).
        crs: Coordinate Reference System.
    """
    # Ensure it's absolute
    if not output_path.is_absolute():
        from handle import ROOT
        output_path = ROOT / output_path
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle array shape (ensure C, H, W for rasterio)
    if len(array.shape) == 2:
        count = 1
        raster_data = array[np.newaxis, ...]
        height, width = array.shape
    else:
        # If the trailing dimension is small, assume it's (H, W, C)
        if array.shape[2] <= 4:
            count = array.shape[2]
            raster_data = array.transpose(2, 0, 1)
            height, width = array.shape[:2]
        else:
            # Assume it's already (C, H, W)
            count = array.shape[0]
            raster_data = array
            height, width = array.shape[1:3]

    with rasterio.open(
        str(output_path),
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(raster_data)

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

    def load_from_metadata(self, metadata:dict) -> "Polygon":
        """Load polygon metadata from a JSON file.
        
        Args:
            metadata: Dictionary containing the polygon metadata.
        """
        self.id = int(metadata["id"])
        self.class_id = int(metadata["class_id"])
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

    def get_metadata(self) -> dict:
        """Get the polygon's metadata as a dictionary.
        
        Returns:
            Dictionary containing the polygon metadata.
        """
        metadata = {
            "id": int(self.id),
            "class_id": int(self.class_id),
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

        return metadata

    def num_crops(self) -> int:
        """Return the number of crop paths associated with this polygon."""
        return len(self.crop_paths)

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

def get_file_resolution(file_path:Path) -> float:
    """Get the scale factor for a given study area.
    
    Attempts to extract pixel size from GeoTIFF transform. Detects unit (meters or km)
    and converts to meters if needed.
    
    Args:
        file_path: Path to the file.
    """
    with rasterio.open(file_path) as dataset:

        if dataset.crs.is_geographic:
            # If the CRS is geographic, we need to calculate the pixel size in meters
            # using the latitude of the area (assuming it's near the equator for simplicity)
            lat = dataset.bounds.top  # Use the top latitude of the dataset
            pixel_size_x = abs(dataset.transform.a) * (111320 * np.cos(np.radians(lat)))  # Convert degrees to meters
            pixel_size_y = abs(dataset.transform.e) * 111320  # Convert degrees to meters
            scale = (pixel_size_x + pixel_size_y) / 2
        else:
            pixel_size_x = abs(dataset.transform.a)
            pixel_size_y = abs(dataset.transform.e)
            scale = (pixel_size_x + pixel_size_y) / 2

    return scale