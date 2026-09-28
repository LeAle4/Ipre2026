from typing import Generator

import random
import rasterio
import shapely
import numpy as np
from resize2 import resize_datapoint_image
from extract2 import scale_bounds
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
    data = tif_area.read(1, window=view_window, out_shape=(tif_area.count, int(view_window.height), int(view_window.width)))
    
    # If nodata is defined, use it
    if no_data_value is not None:
        valid_pixels = np.count_nonzero(data != no_data_value)
    else:
        # If nodata is None, assume common no-data values (white=255 for uint8)
        valid_pixels = np.count_nonzero((data != 255) & (data != 0))
    
    total_pixels = data.size
    fraction_valid = valid_pixels / total_pixels if total_pixels > 0 else 0
    
    return fraction_valid >= threshold


def generate_negative_img(
    tif_area: rasterio.DatasetReader,
    src_scale: float,
    window_size: int,
    desired_scale: float,
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

        if in_actual_data(tif_area, window, tif_area.nodata, threshold=0.99):
            # Extract raw unscaled pixels (e.g., 1120x1120) for resize2.py to process later
            img = tif_area.read(window=window, out_shape=(tif_area.count, int(window.height), int(window.width)))
            bounds = tif_area.window_bounds(window)
            return img, bounds

    raise RuntimeError(f"Could not find a valid data window after {max_attempts} attempts.")

def generate_negative_samples(site_data:SiteData, num_positive_samples: int, window_size: int = DEFAULT_WINDOW_SIZE, desired_scale:float = DEFAULT_TARGET_SCALE, negative_ratio: float = DEFAULT_NEGATIVES_RATIO) -> Generator[DataPoint, None, None]:
    """Calculate the number of negative samples to generate based on the number of positive samples."""
    num_to_generate = int(num_positive_samples * negative_ratio)

    with site_data.access_tif() as tif_area:
        for n in range(num_to_generate):
           img, bounds = generate_negative_img(tif_area, site_data.m_px, window_size, desired_scale)
           datapoint = DataPoint(site_data.name, GROUND_CLASS, shapely.geometry.box(*bounds), site_data.crs, img, bounds, site_data.m_px)
           resize_datapoint_image(datapoint, desired_scale)
           yield datapoint