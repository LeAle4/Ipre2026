import math
from typing import Generator

import rasterio

from parameters import DATA_DIR, GEO_CLASS
from datamanager import Label, SiteData, DataPoint, get_sites

def calculate_target_extent(
    current_meters: float,
    desired_window_px: int,
    stride_px: int,
    desired_scale: float,
    eps: float = 1e-6
) -> float:
    """Calculates required ground extent in meters to fit integer tile steps at desired_scale."""
    current_px = current_meters / desired_scale
    if current_px <= desired_window_px + eps:
        target_px = desired_window_px
    else:
        # Number of strides k needed, using epsilon tolerance to avoid float triggers
        k = math.ceil((current_px - desired_window_px - eps) / stride_px)
        target_px = desired_window_px + k * stride_px
    
    return target_px * desired_scale

def scale_bounds(
    bounds: tuple[float, float, float, float],
    desired_window_size: int,
    stride: int,
    desired_metric_scale: float
) -> tuple[float, float, float, float]:
    """Expands bounding box (minx, miny, maxx, maxy) in meters to tile cleanly
    at desired_metric_scale (m/px) with given window_size (px) and stride (px).
    """
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    new_width = calculate_target_extent(width, desired_window_size, stride, desired_metric_scale)
    new_height = calculate_target_extent(height, desired_window_size, stride, desired_metric_scale)

    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0

    new_minx = center_x - new_width / 2.0
    new_maxx = center_x + new_width / 2.0
    new_miny = center_y - new_height / 2.0
    new_maxy = center_y + new_height / 2.0

    return new_minx, new_miny, new_maxx, new_maxy

def extract_datapoint(site_data: SiteData, orto_view: rasterio.DatasetReader, label: Label, window_size:int, stride: int, desired_scale: float) -> DataPoint:
    geometry_bounds = label.geometry.bounds
    desired_bounds = scale_bounds(geometry_bounds, window_size, stride, desired_scale)
    print(desired_bounds)
    window_view = rasterio.windows.from_bounds(*desired_bounds, transform=orto_view.transform)
    window_img = orto_view.read(window=window_view)
    print(window_img.shape)
    return DataPoint(site_data.name, label.data_label, label.geometry, site_data.crs, window_img, desired_bounds, site_data.m_px)

def get_polygon_imgs(site: SiteData,window_size: int, stride: int, desired_scale: float, verbose: bool) -> Generator[DataPoint, None, None]:
    if verbose:
        print(f"Processing site: {site.name}")
    labels = site.label_data()
    geo_labels = labels.filter_class(GEO_CLASS)
    if verbose:
        print(f"Found {len(geo_labels)} geoglyph labels in site {site.name}")
    with site.access_tif() as tif:
        for label in geo_labels:
            if verbose:
                print(f"Processing label {label.data_label} in site {site.name}")
            yield extract_datapoint(site, tif, label, window_size, stride, desired_scale)