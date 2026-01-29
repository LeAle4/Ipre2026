"""
Image formatting and cropping module for geoglyph dataset processing.

This module provides utilities for cropping images with various strategies
(random, fixed grid, polygon-based) and filling out-of-bounds regions with noise.
Key functions support generating training data from geoglyph imagery.
"""

import shapely
from pathlib import Path
import numpy as np
import cv2
import json

from handle import (
    BASE_DIR,
    PROJECT_DIR,
    DATA_DIR,
    LLUTA_GEOS_DIR,
    UNITA_GEOS_DIR,
    CHUG_GEOS_DIR
)

TEST_DIR = BASE_DIR / "test_geos"
OUTPUT_DIR = BASE_DIR / "crops_output"
RAND_CROPS_DIR = OUTPUT_DIR / "random_crops"
FIXED_CROPS_DIR = OUTPUT_DIR / "fixed_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Utility Functions
# ============================================================================

def extract_area_from_filename(filename: str) -> str:
    """
    Extract area name from a geoglyph filename.
    
    Recognizes standard naming conventions:
    - Files starting with 'unita_' -> 'unita'
    - Files starting with 'lluta_' -> 'lluta'
    - Files starting with 'chugchug_' or 'chug_' -> 'chugchug'
    - Files starting with 'granllama_' -> 'granllama'
    - Files starting with 'salvador_' -> 'salvador'
    
    Args:
        filename: Name of the file (not full path)
    
    Returns:
        Area name ('unita', 'lluta', 'chugchug', 'granllama', 'salvador') or 'unknown' if not recognized
    """
    name = filename.lower()
    if name.startswith('unita_'):
        return 'unita'
    elif name.startswith('lluta_'):
        return 'lluta'
    elif name.startswith('chugchug_') or name.startswith('chug_'):
        return 'chugchug'
    elif name.startswith('granllama_'):
        return 'granllama'
    elif name.startswith('salvador_'):
        return 'salvador'
    return 'unknown'


def load_polygon_from_metadata(json_path):
    """
    Load polygon vertices from metadata JSON file.
    
    Expects JSON format with 'polygon_points', 'polygon' or 'geometry' field containing coordinates.
    
    Args:
        json_path: Path to the metadata JSON file
    
    Returns:
        List of (x, y) tuples representing polygon vertices, or None if not found
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
                # Handle nested coordinate arrays (e.g., from GeoJSON)
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
    
    except Exception as e:
        return None


def find_metadata_file(image_path):
    """
    Find the corresponding metadata JSON file for an image.
    
    Args:
        image_path: Path to the image file (.jpg or .tif)
    
    Returns:
        Path to metadata file or None if not found
    """
    # Extract the stem and look for metadata file
    # E.g., unita_geoglif_0000_ortho.jpg -> unita_geoglif_0000_metadata.json
    stem = image_path.stem.replace('_ortho', '')
    json_path = image_path.parent / f"{stem}_metadata.json"
    
    return json_path if json_path.exists() else None


def geo_to_pixel(lon, lat, bounds, img_shape):
    """
    Convert geographic coordinates to pixel coordinates.
    
    Args:
        lon: Longitude (x) coordinate
        lat: Latitude (y) coordinate
        bounds: Dict with 'minx', 'miny', 'maxx', 'maxy' keys defining geographic bounds
        img_shape: Dict with 'width' and 'height' keys defining image dimensions
    
    Returns:
        Tuple of (x_pixel, y_pixel) coordinates
    """
    geo_minx, geo_miny = bounds['minx'], bounds['miny']
    geo_maxx, geo_maxy = bounds['maxx'], bounds['maxy']
    img_width, img_height = img_shape['width'], img_shape['height']
    
    # Normalize to [0, 1]
    x_norm = (lon - geo_minx) / (geo_maxx - geo_minx)
    y_norm = (geo_maxy - lat) / (geo_maxy - geo_miny)  # Invert Y axis
    
    # Convert to pixel coordinates
    x_pixel = x_norm * img_width
    y_pixel = y_norm * img_height
    
    return (x_pixel, y_pixel)


def convert_polygon_geo_to_pixel(polygon_vertices, bounds, img_shape):
    """
    Convert a list of geographic polygon vertices to pixel coordinates.
    
    Args:
        polygon_vertices: List of (lon, lat) tuples in geographic coordinates
        bounds: Dict with 'minx', 'miny', 'maxx', 'maxy' keys defining geographic bounds
        img_shape: Dict with 'width' and 'height' keys defining image dimensions
    
    Returns:
        List of (x, y) tuples in pixel coordinates
    """
    return [geo_to_pixel(lon, lat, bounds, img_shape) for lon, lat in polygon_vertices]


# ============================================================================
# Noise Filling
# ============================================================================

def fill_with_noise(image, mask, noise_level=0.1, noise_type='gaussian', seed=None):
    """
    Fill out-of-bounds regions in a cropped image with noise.
    
    Args:
        image: Input image (numpy array, H x W or H x W x C)
        mask: Binary mask where 1 indicates valid pixels, 0 indicates out-of-bounds (H x W)
        noise_level: Controls randomness (0-1). 0 = average color, 1 = pure random noise
        noise_type: Type of noise - 'gaussian', 'uniform', or 'perlin'
        seed: Random seed for reproducibility
    
    Returns:
        Image with out-of-bounds regions filled with noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    result = image.copy().astype(np.float32)
    
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    invalid_mask = (1 - mask).astype(bool)
    
    if not np.any(invalid_mask):
        return result.astype(image.dtype)
    
    # Calculate average color of valid pixels
    if len(result.shape) == 3:
        avg_color = np.zeros(result.shape[2])
        for c in range(result.shape[2]):
            channel_values = result[:, :, c][~invalid_mask]
            if len(channel_values) > 0:
                avg_color[c] = np.mean(channel_values)
            else:
                avg_color[c] = 128
    else:
        valid_values = result[~invalid_mask]
        avg_color = np.mean(valid_values) if len(valid_values) > 0 else 128
    
    # Generate noise based on type
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 50, result.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-255, 255, result.shape)
    elif noise_type == 'perlin':
        # Simple pseudo-Perlin-like noise using gaussian blur
        noise = np.random.normal(0, 100, result.shape)
        if len(noise.shape) == 3:
            for c in range(noise.shape[2]):
                noise[:, :, c] = cv2.GaussianBlur(noise[:, :, c], (5, 5), 0)
        else:
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Blend noise: 0 = average color, 1 = full random noise
    blended_noise = avg_color * (1 - noise_level) + noise * noise_level
    
    # Fill invalid regions
    result[invalid_mask] = blended_noise[invalid_mask]
    
    # Clip values to valid range
    result = np.clip(result, 0, 255)
    
    return result.astype(image.dtype)


# ============================================================================
# Image Cropping
# ============================================================================

def crop_image(image, center, window_size):
    """
    Crop an image around a specified center with given window size.
    
    Args:
        image: Input image (numpy array, H x W or H x W x C)
        center: Tuple (x, y) for the center of the crop
        window_size: Size of the crop (int)
    
    Returns:
        Cropped image and binary mask indicating valid pixels
    """
    h, w = image.shape[:2]
    half_size = window_size // 2
    x_center, y_center = center
    
    x_start = x_center - half_size
    y_start = y_center - half_size
    x_end = x_start + window_size
    y_end = y_start + window_size
    
    crop = np.zeros((window_size, window_size) + image.shape[2:], dtype=image.dtype)
    mask = np.zeros((window_size, window_size), dtype=np.uint8)
    
    x_start_img = max(0, x_start)
    y_start_img = max(0, y_start)
    x_end_img = min(w, x_end)
    y_end_img = min(h, y_end)
    
    crop_x_start = x_start_img - x_start
    crop_y_start = y_start_img - y_start
    crop_x_end = crop_x_start + (x_end_img - x_start_img)
    crop_y_end = crop_y_start + (y_end_img - y_start_img)
    
    crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = image[y_start_img:y_end_img, x_start_img:x_end_img]
    mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = 1
    
    return crop, mask


# ============================================================================
# Random Crops
# ============================================================================

def make_random_crops(image, window_size, n_crops, seed=None):
    """
    Generate random crops from an image.
    
    Args:
        image: Input image (numpy array, H x W or H x W x C)
        window_size: Size of each crop (int)
        n_crops: Number of random crops to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of tuples (crop, mask) where crop is the cropped image and mask indicates valid pixels
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = image.shape[:2]
    crops = []
    
    for _ in range(n_crops):
        # Generate random center coordinates
        x_center = np.random.randint(0, w)
        y_center = np.random.randint(0, h)
        
        crop, mask = crop_image(image, (x_center, y_center), window_size)
        crops.append((crop, mask))
    
    return crops


# ============================================================================
# Fixed Grid Crops
# ============================================================================

def make_fixed_crops(image, window_size, n_crops, stride=None):
    """
    Generate deterministic crops by sliding from the top-left with a fixed offset.

    Args:
        image: Input image (numpy array, H x W or H x W x C)
        window_size: Size of each crop (int)
        n_crops: Number of crops to return
        stride: Optional step between crop centers. Can be an int or (x, y) tuple.
                If not provided, the stride is computed to evenly spread the crops
                across the image based on its size and the requested crop count.

    Returns:
        List of tuples (crop, mask) where crop is the cropped image and mask indicates valid pixels
    """
    if n_crops <= 0:
        return []

    h, w = image.shape[:2]
    half_size = window_size // 2

    def _parse_stride(step):
        if isinstance(step, (list, tuple, np.ndarray)):
            if len(step) != 2:
                raise ValueError("stride tuple must have length 2 (x, y)")
            return float(step[0]), float(step[1])
        return float(step), float(step)

    # Compute grid of centers
    if stride is None:
        aspect = w / h if h != 0 else 1.0
        cols = max(1, int(np.ceil(np.sqrt(n_crops * aspect))))
        rows = max(1, int(np.ceil(n_crops / cols)))

        x_start, x_end = half_size, max(half_size, w - half_size)
        y_start, y_end = half_size, max(half_size, h - half_size)

        xs = np.linspace(x_start, x_end, num=cols)
        ys = np.linspace(y_start, y_end, num=rows)
    else:
        stride_x, stride_y = _parse_stride(stride)
        if stride_x <= 0 or stride_y <= 0:
            raise ValueError("stride must be positive")

        xs = np.arange(half_size, max(w, half_size) + stride_x, stride_x)
        ys = np.arange(half_size, max(h, half_size) + stride_y, stride_y)

    centers = []
    for y in ys:
        for x in xs:
            centers.append((int(round(x)), int(round(y))))
            if len(centers) >= n_crops:
                break
        if len(centers) >= n_crops:
            break

    crops = []
    for center in centers:
        crop, mask = crop_image(image, center, window_size)
        crops.append((crop, mask))

    return crops


# ============================================================================
# Polygon-based Crops
# ============================================================================

def make_polygon_thresholds_crops(image, polygon_vertices, window_size, n_crops, stride, threshold=0.1):
    """
    Generate crops uniformly distributed across a polygon with minimum overlap threshold.
    
    Works like make_fixed_crops but only includes crops where the intersection area
    between the crop window and the polygon is at least `threshold` percent of the crop area.
    Crops are uniformly distributed across the entire polygon area.
    
    Args:
        image: Input image (numpy array, H x W or H x W x C)
        polygon_vertices: Vertices of the polygon (list/array of (x, y) tuples)
        window_size: Size of each crop (int)
        n_crops: Number of crops to return (or maximum if fewer valid crops exist)
        stride: Step between crop centers. Can be an int or (x, y) tuple.
                If not provided, the stride is computed to evenly spread the crops
                across the image based on its size and the requested crop count.
        threshold: Minimum overlap ratio (0-1). Default is 0.1 (10%)
    
    Returns:
        List of tuples (crop, mask) where crop is the cropped image and mask indicates valid pixels
    """
    if n_crops <= 0:
        return []
    
    # Create polygon from vertices
    polygon = shapely.Polygon(polygon_vertices)
    crop_area = window_size * window_size
    
    h, w = image.shape[:2]
    half_size = window_size // 2
    
    def _parse_stride(step):
        if isinstance(step, (list, tuple, np.ndarray)):
            if len(step) != 2:
                raise ValueError("stride tuple must have length 2 (x, y)")
            return float(step[0]), float(step[1])
        return float(step), float(step)
    
    # Compute grid of centers
    if stride is None:
        aspect = w / h if h != 0 else 1.0
        cols = max(1, int(np.ceil(np.sqrt(n_crops * aspect))))
        rows = max(1, int(np.ceil(n_crops / cols)))

        x_start, x_end = half_size, max(half_size, w - half_size)
        y_start, y_end = half_size, max(half_size, h - half_size)

        xs = np.linspace(x_start, x_end, num=cols)
        ys = np.linspace(y_start, y_end, num=rows)
    else:
        stride_x, stride_y = _parse_stride(stride)
        if stride_x <= 0 or stride_y <= 0:
            raise ValueError("stride must be positive")

        xs = np.arange(half_size, max(w, half_size) + stride_x, stride_x)
        ys = np.arange(half_size, max(h, half_size) + stride_y, stride_y)
    
    # Collect all valid crops (centers and their indices)
    valid_crops_data = []
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            center = (int(round(x)), int(round(y)))
            
            # Create crop square as shapely box
            crop_box = shapely.box(
                center[0] - half_size,
                center[1] - half_size,
                center[0] + half_size,
                center[1] + half_size
            )
            
            # Check if overlap threshold is met
            # For threshold close to 1.0, use strict containment check
            if threshold >= 0.9999:
                # Crop must be entirely within polygon
                if polygon.contains(crop_box):
                    valid_crops_data.append((y_idx, x_idx, center))
            else:
                # Calculate intersection area for partial overlap
                intersection = polygon.intersection(crop_box)
                intersection_area = intersection.area
                
                # Use strict > comparison to avoid floating point issues
                if intersection_area / crop_area > threshold - 1e-9:
                    valid_crops_data.append((y_idx, x_idx, center))
    
    # If we have fewer valid crops than requested, return all of them
    if len(valid_crops_data) <= n_crops:
        crops = []
        for y_idx, x_idx, center in valid_crops_data:
            crop, mask = crop_image(image, center, window_size)
            crops.append((crop, mask))
        return crops
    
    # Uniformly sample n_crops from valid crops
    # Calculate step size to uniformly distribute selections
    step = len(valid_crops_data) / n_crops
    selected_indices = [int(i * step) for i in range(n_crops)]
    
    crops = []
    for idx in selected_indices:
        y_idx, x_idx, center = valid_crops_data[idx]
        crop, mask = crop_image(image, center, window_size)
        crops.append((crop, mask))
    
    return crops


# ============================================================================
# Negative Sample Generation
# ============================================================================

def create_negative_samples(area, n_samples, window_size, positive_polygon_coords, seed=None):
    pass


if __name__ == "__main__":
    pass