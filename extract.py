import math
from typing import Generator
import rasterio
import numpy as np

from parameters import GEO_CLASS
from datamanager import Label, SiteData, DataPoint

def snap_bounds_integer(
    bounds: tuple[float, float, float, float],
    desired_scale: float,
    window_size: int,
    stride: int,
) -> tuple[float, float, float, float]:
    """Expand bounding box to exact pixel grid using floor/ceil.

    The output bounds guarantee that:
      - pixel dimensions = window_size + k * stride  (for k >= 0)
      - min coordinates are floored, max coordinates are ceiled
        on the pixel grid defined by *desired_scale*.

    Parameters
    ----------
    bounds : (minx, miny, maxx, maxy)
        Geographic bounding box in metres.
    desired_scale : float
        Target resolution in m/px.
    window_size, stride : int
        Sliding-window crop parameters in pixels.

    Returns
    -------
    (minx, miny, maxx, maxy) : tuple of float
        Expanded bounding box aligned to the pixel grid.
    """
    minx, miny, maxx, maxy = bounds

    # Convert to pixel coordinates on the target grid
    px_minx = minx / desired_scale
    px_miny = miny / desired_scale
    px_maxx = maxx / desired_scale
    px_maxy = maxy / desired_scale

    # Floor min, ceil max (expand outward)
    px_minx = math.floor(px_minx)
    px_miny = math.floor(px_miny)
    px_maxx = math.ceil(px_maxx)
    px_maxy = math.ceil(px_maxy)

    # Snap dimensions to window_size + k * stride
    width_px = px_maxx - px_minx
    height_px = px_maxy - px_miny

    def _snap_dim(dim_px: int) -> int:
        if dim_px <= window_size:
            return window_size
        k = math.ceil((dim_px - window_size) / stride)
        return window_size + k * stride

    snapped_w = _snap_dim(width_px)
    snapped_h = _snap_dim(height_px)

    # Distribute the extra padding symmetrically
    pad_w = snapped_w - width_px
    pad_h = snapped_h - height_px

    px_minx -= pad_w // 2
    px_maxx += pad_w - pad_w // 2
    px_miny -= pad_h // 2
    px_maxy += pad_h - pad_h // 2

    return (
        px_minx * desired_scale,
        px_miny * desired_scale,
        px_maxx * desired_scale,
        px_maxy * desired_scale,
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
    # Factor 3: integer floor/ceil bounding-box expansion
    desired_bounds = snap_bounds_integer(geometry_bounds, desired_scale, window_size, stride)

    # Convert ground bounds (meters) to window slice in source raster coordinates
    window_view = rasterio.windows.from_bounds(*desired_bounds, transform=orto_view.transform)

    # Extract RAW source pixels without truncation
    src_h = int(round(window_view.height))
    src_w = int(round(window_view.width))
    window_img = orto_view.read(window=window_view, out_shape=(orto_view.count, src_h, src_w))
    window_img = np.transpose(window_img, (1, 2, 0))  # Convert to HWC format

    return DataPoint(
        site_data.name,
        label.data_label,
        label.geometry,
        site_data.crs,
        window_img,
        desired_bounds,
        m_px=site_data.pixel_scale,
    )

def get_polygon_imgs(site: SiteData,window_size: int, stride: int, desired_scale: float) -> Generator[DataPoint, None, None]:
    labels = site.label_data()
    geo_labels = labels.filter_class(GEO_CLASS)
    with site.access_tif() as tif:
        for label in geo_labels:
            yield extract_datapoint(site, tif, label, window_size, stride, desired_scale)