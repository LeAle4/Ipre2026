import geopandas as gpd
import shapely
import numpy as np
from pathlib import Path
from PIL import Image

from extract2 import get_polygon_imgs
from resize2 import resize_datapoint_image
from crop2 import generate_crops
from negatives2 import generate_negative_samples

from datamanager import get_sites, DataPoint
from parameters import DATA_DIR, DEFAULT_TARGET_SCALE, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_THRESHOLD_CROP_CONTENT, GEO_CLASS

def save_crops_boundaries(data_point: DataPoint, bounds: tuple[shapely.geometry.Polygon]) -> None:
    """Save the boundaries of all geo crops as a GeoJSON file."""
    
    gdf = gpd.GeoDataFrame({
        "geo_id": data_point.id,
        "geometry": bounds
    }, crs=data_point.crs)
    gdf.to_file("test.gpkg", driver="GeoJSON")

def save_crops(crops: tuple[np.ndarray]) -> None:
    img_path = Path("crops")
    img_path.mkdir(parents=True, exist_ok=True)
    """Save the crops as individual image files."""
    for i, crop in enumerate(crops):
        crop_img = Image.fromarray(crop)
        crop_img.save(img_path / f"crop_{i:03d}.png")

def main():
    sites = get_sites(DATA_DIR)
    lluta = sites["Lluta"]
    negatives = tuple(generate_negative_samples(lluta, 10, DEFAULT_WINDOW_SIZE, DEFAULT_TARGET_SCALE))
    save_crops(tuple(map(lambda x: x.image, negatives)))
    # for site in sites.values():
    #     for geo_datapoint in get_polygon_imgs(site, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_TARGET_SCALE, verbose=True):
    #         resize_datapoint_image(geo_datapoint, DEFAULT_TARGET_SCALE)
    #         crops = tuple(generate_crops(geo_datapoint, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_THRESHOLD_CROP_CONTENT))


if __name__ == "__main__":
    main()