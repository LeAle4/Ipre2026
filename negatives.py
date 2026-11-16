from __future__ import annotations
from typing import Generator

import math
import random
import rasterio
import shapely
import numpy as np
from resize import resize_datapoint_image
from datamanager import SiteData, DataPoint
from parameters import DEFAULT_NEGATIVES_RATIO, DEFAULT_TARGET_SCALE, DEFAULT_WINDOW_SIZE, GROUND_CLASS

def in_actual_data(tif_area, view_window:rasterio.windows.Window, no_data_value, threshold:float = 1) -> bool:
    """Check if the specified window contains valid data above the threshold.
    
    Args:
        tif_area: Rasterio dataset object of the study area.
        view_window: Window object defining the area to check.
        no_data_value: Cached nodata value from tif_area.nodata.
        threshold: Minimum fraction of valid data required (0-1).
    """
    data = tif_area.read(1, window=view_window)
    
    # If nodata is defined, use it
    if no_data_value is not None:
        valid_pixels = np.count_nonzero(data != no_data_value)
    else:
        # If nodata is None, assume common no-data values (white=255 for uint8)
        valid_pixels = np.count_nonzero((data != 255) & (data != 0))
    
    total_pixels = data.size
    fraction_valid = valid_pixels / total_pixels if total_pixels > 0 else 0
    
    return fraction_valid >= threshold

def calculate_target_extent(
    current_meters: float,
    desired_window_px: int,
    stride_px: int,
    desired_scale: float,
    eps: float = 1e-6
) -> float:
    """Calculates ground distance in meters needed so that after scaling to
    desired_scale (m/px), the pixel dimensions fit desired_window_px + k * stride_px.
    """
    current_px = current_meters / desired_scale
    if current_px <= desired_window_px + eps:
        target_px = desired_window_px
    else:
        k = math.ceil((current_px - desired_window_px - eps) / stride_px)
        target_px = desired_window_px + k * stride_px
    
    return target_px * desired_scale


def scale_bounds(
    bounds: tuple[float, float, float, float],
    desired_window_size: int,
    stride: int,
    desired_metric_scale: float
) -> tuple[float, float, float, float]:
    """Expands bounding box (minx, miny, maxx, maxy) in ground meters based on desired_scale."""
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    new_width = calculate_target_extent(width, desired_window_size, stride, desired_metric_scale)
    new_height = calculate_target_extent(height, desired_window_size, stride, desired_metric_scale)

    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0

    return (
        center_x - new_width / 2.0,
        center_y - new_height / 2.0,
        center_x + new_width / 2.0,
        center_y + new_height / 2.0
    )

def not_in_positive_polygons(window:rasterio.windows.Window, transform, positive_polygon_boundaries: tuple[shapely.geometry.Polygon]) -> bool:
    """Check if the specified window does not intersect with any positive polygons.
    
    Args:
        window: Window object defining the area to check.
        transform: Affine transform of the raster dataset.
        positive_polygon_boundaries: Tuple of shapely Polygon objects representing positive areas.
    """
    # Convert window to bounds in spatial coordinates
    bounds = rasterio.windows.bounds(window, transform)
    window_polygon = shapely.geometry.box(*bounds)
    
    # Check for intersection with any positive polygon
    return all(not window_polygon.intersects(poly) for poly in positive_polygon_boundaries)

def generate_negative_img(
    tif_area: rasterio.DatasetReader,
    src_scale: float,
    window_size: int,
    desired_scale: float,
    positive_polygon_boundaries: tuple[shapely.geometry.Polygon],
    max_attempts: int = 1000
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Sample a random raw window in source pixel space that will scale down to window_size later."""
    # Compute size in ORIGINAL TIF pixels needed to cover target ground extent
    src_win_px = int(round((window_size * desired_scale) / src_scale))

    max_x = tif_area.width - src_win_px
    max_y = tif_area.height - src_win_px

    for _ in range(max_attempts):
        # Sample top-left corner in source pixel space
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        window = rasterio.windows.Window(x, y, src_win_px, src_win_px)

        if in_actual_data(tif_area, window, tif_area.nodata, threshold=0.99) and not_in_positive_polygons(window, tif_area.transform, positive_polygon_boundaries):
            # Extract raw unscaled pixels (e.g., 1120x1120) for resize2.py to process later
            img = tif_area.read(window=window, out_shape=(tif_area.count, int(window.height), int(window.width)))
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC format
            win_bounds = tif_area.window_bounds(window)
            cx = (win_bounds[0] + win_bounds[2]) / 2.0
            cy = (win_bounds[1] + win_bounds[3]) / 2.0
            half_dim = (window_size * desired_scale) / 2.0
            bounds = (cx - half_dim, cy - half_dim, cx + half_dim, cy + half_dim)
            return img, bounds

    raise RuntimeError(f"Could not find a valid data window after {max_attempts} attempts.")

def generate_negative_samples(site_data:SiteData, num_positive_samples: int, positive_polygon_boundaries: tuple[shapely.geometry.Polygon],window_size: int = DEFAULT_WINDOW_SIZE, desired_scale:float = DEFAULT_TARGET_SCALE, negative_ratio: float = DEFAULT_NEGATIVES_RATIO) -> Generator[DataPoint, None, None]:
    """Calculate the number of negative samples to generate based on the number of positive samples."""
    num_to_generate = int(num_positive_samples * negative_ratio)

    with site_data.access_tif() as tif_area:
        for n in range(num_to_generate):
            img, bounds = generate_negative_img(tif_area, site_data.m_px, window_size, desired_scale, positive_polygon_boundaries)
            datapoint = DataPoint(
                site_data.name,
                GROUND_CLASS,
                shapely.geometry.box(*bounds),
                site_data.crs,
                img,
                bounds,
                m_px=site_data.pixel_scale,
            )
            resize_datapoint_image(datapoint, desired_scale, window_size=window_size)
            yield datapoint