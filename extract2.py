from typing import Generator

import rasterio

from parameters import DATA_DIR, GEO_CLASS
from datamanager import Label, SiteData, DataPoint, get_sites

def size_for_multiplicity(current_size: float, desired_size: float, stride: float) -> float:
    if current_size <= desired_size:
        return desired_size
    else:
        candidate =  (current_size - desired_size) // stride * stride + desired_size
        if candidate < current_size:
            candidate += stride
        return candidate

def scale_bounds(bounds: tuple[float, float, float, float], desired_window_size: int, stride: int, metric_scale: float, desired_metric_scale: float) -> tuple[float, float, float, float]:
    """Scale the bounds to ensure that the img we are extracting is a multiple of the desired window size.
    
    bounds: A tuple representing the bounding box (minx, miny, maxx, maxy).
    desired_window_size: The desired window size to which the bounds should be scaled.
    stride: The stride to be used for the window.
    metric_scale: The scale of the bounds in meters.
    desired_metric_scale: The desired scale of the image in meters.
    """
    minx, miny, maxx, maxy = bounds
    #What is the actual size of the window in the bounds scale
    window_size_metric = desired_window_size * metric_scale
    #How bigger or smaller should the image be so that later in the rescaling the image has the desired size.
    scaling_factor = metric_scale / desired_metric_scale
    #The window in meters that we want to extract from the
    actual_windows_size_metric = window_size_metric * scaling_factor

    stride_scaled = stride * metric_scale * scaling_factor

    width = maxx - minx
    height = maxy - miny

    # Calculate the scaling factor needed to make the width and height multiples of the desired window size
    scale_factor_x = size_for_multiplicity(width, actual_windows_size_metric, stride_scaled) / width
    scale_factor_y = size_for_multiplicity(height, actual_windows_size_metric, stride_scaled) / height

    # Apply the scaling factor to the bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    new_width = width * scale_factor_x
    new_height = height * scale_factor_y

    new_minx = center_x - new_width / 2
    new_maxx = center_x + new_width / 2
    new_miny = center_y - new_height / 2
    new_maxy = center_y + new_height / 2

    return new_minx, new_miny, new_maxx, new_maxy

def extract_datapoint(site_data: SiteData, orto_view: rasterio.DatasetReader, label: Label, window_size:int, stride: int, desired_scale: float) -> DataPoint:
    geometry_bounds = label.geometry.bounds
    desired_bounds = scale_bounds(geometry_bounds, window_size, stride, site_data.m_px, desired_scale)
    window_view = rasterio.windows.from_bounds(*desired_bounds, transform=orto_view.transform)
    window_img = orto_view.read(window=window_view)
    return DataPoint(site_data.name, label, window_img, desired_bounds, site_data.m_px)

def get_polygon_imgs(window_size: int, stride: int, desired_scale: float) -> Generator:
    sites = get_sites(DATA_DIR)

    for site in sites.values():
        labels = site.label_data()
        geo_labels = labels.filter_class(GEO_CLASS)
        with site.access_tif() as tif:
            for label in geo_labels:
                yield extract_datapoint(site, tif, label, window_size, stride, desired_scale)

def main():
    from parameters import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_TARGET_SCALE
    for datapoint in get_polygon_imgs(DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_TARGET_SCALE):
        print(datapoint)
        
if __name__ == "__main__":
    main()