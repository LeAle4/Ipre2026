"""
Common utility functions for loading and processing geospatial data.
"""

import json
from pathlib import Path
import numpy as np
import rasterio
from shapely.geometry import Polygon, box


def load_tif_image(tif_path):
    """
    Load a TIF image and return as numpy array (H, W, C).
    
    Parameters
    ----------
    tif_path : str or Path
        Path to the TIF file
    
    Returns
    -------
    np.ndarray
        Image array of shape (H, W, 3) with dtype uint8 (RGB)
    """
    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
        else:
            # Grayscale - convert to RGB
            img = src.read(1)
            img = np.stack([img, img, img], axis=-1)
    return img.astype(np.uint8)


def load_polygon_from_metadata(json_path):
    """
    Load polygon vertices from metadata JSON file.
    
    Tries multiple common keys for polygon data:
    - polygon_points (Shapely format with exterior/interiors)
    - polygon
    - geometry (GeoJSON format)
    - coordinates
    
    Parameters
    ----------
    json_path : str or Path
        Path to metadata JSON file
    
    Returns
    -------
    list or None
        List of (x, y) tuples in geographic coordinates, or None if not found
    """
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Try different possible keys for polygon data
        if 'polygon_points' in metadata:
            polygon_points = metadata['polygon_points']
            if polygon_points and isinstance(polygon_points, list) and 'exterior' in polygon_points[0]:
                coords = polygon_points[0]['exterior']
            else:
                coords = polygon_points
        elif 'polygon' in metadata:
            coords = metadata['polygon']
        elif 'geometry' in metadata:
            geom = metadata['geometry']
            if isinstance(geom, dict) and 'coordinates' in geom:
                coords = geom['coordinates']
                if coords and isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (list, tuple)):
                    coords = coords[0]
            else:
                coords = geom
        elif 'coordinates' in metadata:
            coords = metadata['coordinates']
        else:
            return None
        
        # Convert to list of tuples if needed
        if isinstance(coords, list):
            polygon_vertices = [(float(x), float(y)) for x, y in coords]
        else:
            polygon_vertices = coords
        
        return polygon_vertices
    
    except Exception:
        return None


def load_metadata(json_path):
    """
    Load metadata from JSON file.
    
    Parameters
    ----------
    json_path : str or Path
        Path to metadata JSON file
    
    Returns
    -------
    dict or None
        Metadata dictionary, or None if load fails
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def geo_to_pixel(lon, lat, bounds, img_shape):
    """
    Convert geographic coordinates to pixel coordinates.
    
    Parameters
    ----------
    lon, lat : float
        Geographic coordinates (longitude, latitude)
    bounds : dict
        Dict with 'minx', 'miny', 'maxx', 'maxy' keys defining geographic extent
    img_shape : dict
        Dict with 'width' and 'height' keys defining image dimensions
    
    Returns
    -------
    tuple
        (pixel_x, pixel_y) coordinates
    """
    minx = bounds['minx']
    miny = bounds['miny']
    maxx = bounds['maxx']
    maxy = bounds['maxy']
    
    width = img_shape['width']
    height = img_shape['height']
    
    # Avoid division by zero
    dx = maxx - minx or 1.0
    dy = maxy - miny or 1.0
    
    pixel_x = int((lon - minx) / dx * width)
    pixel_y = int((maxy - lat) / dy * height)
    
    return (pixel_x, pixel_y)


def convert_polygon_geo_to_pixel(polygon_vertices, bounds, img_shape):
    """
    Convert polygon from geographic to pixel coordinates.
    
    Parameters
    ----------
    polygon_vertices : list
        List of (lon, lat) tuples in geographic coordinates
    bounds : dict
        Dict with 'minx', 'miny', 'maxx', 'maxy' keys defining geographic extent
    img_shape : dict
        Dict with 'width' and 'height' keys defining image dimensions
    
    Returns
    -------
    list
        List of (pixel_x, pixel_y) tuples in pixel coordinate space
    """
    return [geo_to_pixel(lon, lat, bounds, img_shape) for lon, lat in polygon_vertices]


def calculate_polygon_overlap(polygon_pixel_coords, img_shape):
    """
    Calculate the area and bounding box of a polygon in pixel space.
    
    Parameters
    ----------
    polygon_pixel_coords : list
        List of (x, y) tuples defining polygon in pixel coordinates
    img_shape : tuple
        (height, width) of image
    
    Returns
    -------
    dict
        Contains 'area', 'bounds', 'valid', and percentage coverage of image
    """
    try:
        polygon = Polygon(polygon_pixel_coords)
        if not polygon.is_valid:
            return {'valid': False, 'area': 0}
        
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        area = polygon.area
        img_area = img_shape[0] * img_shape[1]
        coverage = (area / img_area) * 100 if img_area > 0 else 0
        
        return {
            'valid': True,
            'area': area,
            'bounds': bounds,
            'coverage_percent': coverage
        }
    except Exception:
        return {'valid': False, 'area': 0}


def calculate_patch_overlap(patch_window, polygon_pixel_coords, threshold):
    """
    Check if patch window overlaps polygon by at least threshold percentage.
    
    Parameters
    ----------
    patch_window : tuple
        (x_min, y_min, x_max, y_max) in pixel coordinates
    polygon_pixel_coords : list
        List of (x, y) tuples defining polygon in pixel coordinates
    threshold : float
        Overlap threshold (0.0 to 1.0) as fraction of patch area
    
    Returns
    -------
    bool
        True if overlap meets or exceeds threshold, False otherwise
    """
    try:
        patch_polygon = box(*patch_window)
        geoglif_polygon = Polygon(polygon_pixel_coords)
        
        if not patch_polygon.is_valid or not geoglif_polygon.is_valid:
            return False
        
        intersection_area = patch_polygon.intersection(geoglif_polygon).area
        patch_area = patch_polygon.area
        
        if patch_area == 0:
            return False
        
        overlap_ratio = intersection_area / patch_area
        return overlap_ratio >= threshold
    
    except Exception:
        return False
