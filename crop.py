from typing import Generator
import numpy as np
import shapely
from skimage.util import view_as_windows

from datamanager import DataPoint, Crop
from parameters import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_THRESHOLD_CROP_CONTENT


def calculate_crop_boundaries(
    bounds: tuple[float, float, float, float],
    window_size: int,
    stride: int,
    scale: float
) -> tuple[shapely.geometry.Polygon, ...]:
    """Calculate geographic crop boundaries in row-major raster order (Top->Bottom, Left->Right)."""
    minx, miny, maxx, maxy = bounds
    win_m = window_size * scale
    stride_m = stride * scale

    num_rows = int(round((maxy - miny - win_m) / stride_m)) + 1
    num_cols = int(round((maxx - minx - win_m) / stride_m)) + 1

    crop_boundaries = []
    # Row loop (Top to Bottom: maxy down to miny)
    for r in range(num_rows):
        top_y = maxy - r * stride_m
        bottom_y = top_y - win_m
        # Col loop (Left to Right: minx up to maxx)
        for c in range(num_cols):
            left_x = minx + c * stride_m
            right_x = left_x + win_m
            crop_boundaries.append(shapely.box(left_x, bottom_y, right_x, top_y))

    return tuple(crop_boundaries)


def generate_crops(
    data_point: DataPoint,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    threshold: float = DEFAULT_THRESHOLD_CROP_CONTENT
) -> Generator[Crop, None, None]:
    """Generate crops from the data point's image along with matching geographic bounding boxes."""
    image = data_point.image

    height, width, channels = image.shape

    # 1. Slide window over image array (Row-Major: Y then X)
    print(f"Generating crops for DataPoint {data_point.id} with image shape {image.shape}, window_size={window_size}, stride={stride}")
    possible_img_crops = view_as_windows(
        image,
        window_shape=(window_size, window_size, channels),
        step=stride
    )
    # Reshape to 4D array: (N_crops, window_size, window_size, channels)
    img_crops_flat = possible_img_crops.reshape(-1, window_size, window_size, channels)

    # 2. Generate matching spatial bounding boxes (Row-Major using data_point.scale)
    crop_boundaries = calculate_crop_boundaries(
        bounds=data_point.image_box.bounds,
        window_size=window_size,
        stride=stride,
        scale=data_point.m_px  # Use scale matching the image array (m/px)
    )

    # 3. Yield matching pairs

    #We guarantee that at least 1 crop will be yielded, even if it doesn't meet the threshold
    yielded_at_least_one = False
    best_threshold = 0
    best_crop, best_bounds = None, None
    for img_crop, bounds in zip(img_crops_flat, crop_boundaries):
        if data_point.polygon_bounds.intersects(bounds):
            intersection_area = data_point.polygon_bounds.intersection(bounds).area
            crop_area = bounds.area
            print(f"Crop bounds: {bounds.bounds}, Intersection area: {intersection_area}, Crop area: {crop_area}, Threshold: {threshold}, Intersection ratio: {intersection_area / crop_area}")
            if crop_area > 0 and (intersection_area / crop_area) >= threshold:
                yielded_at_least_one = True
                yield Crop(data_point, img_crop, bounds, intersection_area / crop_area)
            elif crop_area > 0 and (intersection_area / crop_area) > best_threshold:
                best_threshold = intersection_area / crop_area
                best_crop, best_bounds = img_crop, bounds
    if not yielded_at_least_one and best_crop is not None:
        # If no crops met the threshold, yield the first intersecting crop as a backup
        yield Crop(data_point, best_crop, best_bounds)