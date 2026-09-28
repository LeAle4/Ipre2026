import math
from typing import Generator
import rasterio
import numpy as np

from parameters import GEO_CLASS
from datamanager import Label, SiteData, DataPoint


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


def extract_datapoint(
    site_data: SiteData,
    orto_view: rasterio.DatasetReader,
    label: Label,
    window_size: int,
    stride: int,
    desired_scale: float
) -> DataPoint:
    geometry_bounds = label.geometry.bounds
    desired_bounds = scale_bounds(geometry_bounds, window_size, stride, desired_scale)

    # Convert ground bounds (meters) to window slice in source raster coordinates
    window_view = rasterio.windows.from_bounds(*desired_bounds, transform=orto_view.transform)

    # Extract RAW source pixels (e.g., 303x303) without out_shape resampling
    window_img = orto_view.read(window=window_view, out_shape=(orto_view.count, int(window_view.height), int(window_view.width)))

    return DataPoint(
        site_data.name,
        label.data_label,
        label.geometry,
        site_data.crs,
        window_img,
        desired_bounds,
        site_data.m_px  # Pass raw TIF resolution so resize2.py knows original scale
    )

def get_polygon_imgs(site: SiteData,window_size: int, stride: int, desired_scale: float) -> Generator[DataPoint, None, None]:
    labels = site.label_data()
    geo_labels = labels.filter_class(GEO_CLASS)
    with site.access_tif() as tif:
        for label in geo_labels:
            yield extract_datapoint(site, tif, label, window_size, stride, desired_scale)