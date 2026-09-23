from __future__ import annotations

import numpy as np
from pathlib import Path

import rasterio
import geopandas as gpd
from shapely.geometry.multipolygon import MultiPolygon

class Label:
    def __init__(self, fid:int, class_name:str, area:float, length:float, geometry: MultiPolygon):
        self.fid = fid
        self.class_name = class_name
        self.area = area
        self.length = length
        self.geometry = geometry

class LabelTable:
    FID = "fid"
    CLASS = "class"
    AREA = "shape_area"
    LENGTH = "shape_length"
    TYPE = "type"
    GEOMETRY = "geometry"
    HEADERS = (FID, CLASS, AREA, LENGTH, TYPE, GEOMETRY)

    def __init__(self, geo_table: gpd.GeoDataFrame, crs: rasterio.crs.CRS | None = None):
        if crs is not None:
            geo_table = geo_table.to_crs(crs)        
        self._gdf = geo_table

    def filter_class(self, class_name: int) -> LabelTable:
        """Filter the GeoDataFrame to include only polygons of the specified class.

        Args:
            class_name: The name of the class to filter by (e.g., 'geo', 'ground', 'road').

        Returns:
            A LabelTable object containing only the polygons of the specified class.
        """
        return LabelTable(self._gdf[self._gdf[self.CLASS] == class_name])

    def __iter__(self):
        yield from map(lambda datatuple: Label(*datatuple), self._gdf.itertuples(index=False, name=None))

    def __str__(self):
        return str(self._gdf)

    def __len__(self):
        return len(self._gdf)

class SiteData:
    TIF_EXTENSIONS = (".tif", ".tiff")
    LABEL_EXTENSIONS = (".gpkg")
    DEM_EXTENSIONS = (".tif", ".tiff")

    def __init__(self, site_name:str, site_path:Path):
        self.site_name = site_name
        self.site_path = site_path
        self.tiff_path, self.label_path, self.dem_path = self._get_files_paths()
        self.crs, self.bounds, self.transform, self.is_geographic, self.m_px = self._get_tif_data()

    def access_tif(self) -> rasterio.DatasetReader:
        return rasterio.open(self.tiff_path)

    def label_data(self) -> LabelTable:
        """Load the label data (geoglyphs, ground, and road polygons) from the GeoPackage file.

        Returns:
            A LabelTable object containing the label data.
        """
        return LabelTable(gpd.read_file(self.label_path), crs=self.crs) if self.label_path is not None else LabelTable(gpd.GeoDataFrame())

    def access_dem(self) -> rasterio.DatasetReader:
        return rasterio.open(self.dem_path) if self.dem_path is not None else None

    def _get_files_paths(self) -> tuple[Path | None, Path | None, Path | None]:
        tiff, labels, dem = None, None, None
        for file in self.site_path.iterdir():
            if file.stem == self.site_name and file.suffix in self.TIF_EXTENSIONS:
                tiff = file
            if file.stem == f"{self.site_name}_label" and file.suffix in self.LABEL_EXTENSIONS:
                labels = file
            if file.stem == f"{self.site_name}_DEM" and file.suffix in self.DEM_EXTENSIONS:
                dem = file
        return tiff, labels, dem

    def _get_tif_data(self) -> tuple[rasterio.crs.CRS | None, rasterio.coords.BoundingBox | None, rasterio.transform.Affine | None, bool | None, float | None]:
        crs, bounds, transform, is_geographic, m_px = None, None, None, None, None
        if self.tiff_path is not None:
            with rasterio.open(self.tiff_path) as dataset:
                crs = dataset.crs
                bounds = dataset.bounds
                transform = dataset.transform
                is_geographic = dataset.crs.is_geographic
                if is_geographic:
                    # If the CRS is geographic, we need to calculate the pixel size in meters
                    # using the latitude of the area (assuming it's near the equator for simplicity)
                    lat = dataset.bounds.top  # Use the top latitude of the dataset
                    pixel_size_x = abs(dataset.transform.a) * (111320 * np.cos(np.radians(lat)))  # Convert degrees to meters
                    pixel_size_y = abs(dataset.transform.e) * 111320  # Convert degrees to meters
                    m_px = (pixel_size_x + pixel_size_y) / 2
                else:
                    pixel_size_x = abs(dataset.transform.a)
                    pixel_size_y = abs(dataset.transform.e)
                    m_px = (pixel_size_x + pixel_size_y) / 2
        return crs, bounds, transform, is_geographic, m_px

    def __str__(self):
        return f"SiteData(site_name={self.site_name}, site_path={self.site_path}, tiff_path={self.tiff_path}, label_path={self.label_path}, dem_path={self.dem_path})"

def get_sites(data_path:Path) -> dict[str, SiteData]:
    """Get a list of SiteData objects for each site in the data path.

    Args:
        data_path: Path to the directory containing site folders.

    Returns:
        A dictionary mapping site names to SiteData objects.
    """
    sites = {}
    for site_folder in data_path.iterdir():
        if site_folder.is_dir():
            site_name = site_folder.name
            site_data = SiteData(site_name, site_folder)
            sites[site_name] = site_data
    return sites

if __name__ == "__main__":
    from parameters import DATA_DIR
    sites = get_sites(DATA_DIR)

    unita = sites["Unita"]
    unita_labels = unita.label_data()
    class1 = unita_labels.filter_class(1)
    for label in class1:
        print(label)