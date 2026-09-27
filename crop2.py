from typing import Generator

import numpy as np
import shapely
from skimage.util import view_as_windows

from datamanager import DataPoint
from parameters import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_THRESHOLD_CROP_CONTENT

def calculate_crop_boundaries(data_point: DataPoint, window_size: int, stride: int) -> tuple[shapely.geometry.Polygon, ...]:
    """Calculate the geographic boundaries of crops based on the original image bounds, window size, and stride.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy) representing the original image bounds in geographic coordinates.
        window_size: The size of the crop window in pixels.
        stride: The stride used for cropping in pixels.
    Returns:
        A list of tuples representing the geographic boundaries of each crop in the format (minx, miny, maxx, maxy).
    """
    minx, miny, maxx, maxy = data_point.image_box.bounds
    window_size_meters_source = window_size * data_point.origin_scale
    stride_meters_source = stride * data_point.origin_scale
    crop_boundaries = []
    for x in np.arange(minx, maxx - window_size_meters_source + 1e-6, stride_meters_source):
        for y in np.arange(miny, maxy - window_size_meters_source + 1e-6, stride_meters_source):
            crop_boundaries.append(shapely.box(x, y, x + window_size_meters_source, y + window_size_meters_source))
    return tuple(crop_boundaries)

def generate_crops(self, data_point: DataPoint, window_size: int = DEFAULT_WINDOW_SIZE, stride: int = DEFAULT_STRIDE, threshold: float = DEFAULT_THRESHOLD_CROP_CONTENT) -> Generator[tuple[np.ndarray, shapely.geometry.Polygon], None, None]:
        """Generate crops from the data point's image.

        Args:
            data_point: The DataPoint object containing the image and bounds.
            crop_size: The size of the crops to generate (in pixels).
            overlap: The fraction of overlap between adjacent crops (0.0 to 1.0).

        Yields:
            Crops of the image as numpy arrays.
        """
        possible_img_crops = view_as_windows(data_point.image, (window_size, window_size, data_point.image.shape[2]), step=stride)
        crop_boundaries = calculate_crop_boundaries(data_point, window_size, stride)
        for img_crop, bounds in zip(possible_img_crops.reshape(-1, window_size, window_size, data_point.image.shape[2]), crop_boundaries):
            if data_point.polygon_bounds.intersects(bounds):
                intersection_area = data_point.polygon_bounds.intersection(bounds).area
                crop_area = bounds.area
                if crop_area > 0 and (intersection_area / crop_area) >= threshold:
                    yield img_crop, bounds