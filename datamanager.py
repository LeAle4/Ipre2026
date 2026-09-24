from __future__ import annotations
from operator import index

import numpy as np
from pathlib import Path

import rasterio
import geopandas as gpd
import shapely
from shapely.geometry.multipolygon import MultiPolygon

class Label:
    def __init__(self, fid:int, data_label:int, area:float, length:float, geometry: MultiPolygon, is_metric:bool):
        self.fid = fid
        self.data_label = data_label
        self.area = area
        self.length = length
        self.geometry = geometry
        self.is_metric = is_metric

    @classmethod
    def from_row(cls, row, is_metric:bool) -> Label:
        return cls(
            fid=row.Index, 
            data_label=row._4, 
            area=row.Shape_Area, 
            length=row.Shape_Leng, 
            geometry=row.geometry, 
            is_metric=is_metric)

    def __str__(self):
        return f"Label(fid={self.fid}, data_label={self.data_label}, area={self.area}, length={self.length}, geometry={self.geometry.bounds}, is_metric={self.is_metric})"

class LabelTable:
    CLASS = "class"

    def __init__(self, geo_table: gpd.GeoDataFrame, is_metric: bool = True, convert_to_crs: rasterio.crs.CRS | None = None):
        if convert_to_crs is not None:
            geo_table = geo_table.to_crs(convert_to_crs)
        self._gdf = geo_table
        self.is_metric = is_metric

    def filter_class(self, class_name: int) -> LabelTable:
        """Filter the GeoDataFrame to include only polygons of the specified class.

        Args:
            class_name: The name of the class to filter by (e.g., 'geo', 'ground', 'road').

        Returns:
            A LabelTable object containing only the polygons of the specified class.
        """
        return LabelTable(self._gdf[self._gdf[self.CLASS] == class_name], is_metric=self.is_metric)

    def __iter__(self):
        for row in self._gdf.itertuples():
            yield Label.from_row(row, is_metric=self.is_metric)

    def __str__(self):
        return str(self._gdf)

    def __len__(self):
        return len(self._gdf)

class SiteData:
    TIF_EXTENSIONS = (".tif", ".tiff")
    LABEL_EXTENSIONS = (".gpkg")
    DEM_EXTENSIONS = (".tif", ".tiff")

    def __init__(self, site_name:str, site_path:Path):
        self.name = site_name
        self.path = site_path
        self.tiff_path, self.label_path, self.dem_path = self._get_files_paths()
        self.crs, self.bounds, self.transform, self.is_geographic, self.m_px = self._get_tif_data()

        self.is_metric = not self.is_geographic

    def access_tif(self) -> rasterio.DatasetReader:
        return rasterio.open(self.tiff_path)

    def label_data(self) -> LabelTable:
        """Load the label data (geoglyphs, ground, and road polygons) from the GeoPackage file.

        Returns:
            A LabelTable object containing the label data.
        """
        return LabelTable(gpd.read_file(self.label_path), is_metric=self.is_metric, convert_to_crs=self.crs) if self.label_path is not None else LabelTable(gpd.GeoDataFrame())

    def access_dem(self) -> rasterio.DatasetReader:
        return rasterio.open(self.dem_path) if self.dem_path is not None else None

    def _get_files_paths(self) -> tuple[Path | None, Path | None, Path | None]:
        tiff, labels, dem = None, None, None
        for file in self.path.iterdir():
            if file.stem == self.name and file.suffix in self.TIF_EXTENSIONS:
                tiff = file
            if file.stem == f"{self.name}_label" and file.suffix in self.LABEL_EXTENSIONS:
                labels = file
            if file.stem == f"{self.name}_DEM" and file.suffix in self.DEM_EXTENSIONS:
                dem = file
        if None in (tiff, labels):
            raise FileNotFoundError(f"Missing required files for site '{self.name}' in path '{self.path}'. Found: tiff={bool(tiff)}, labels={bool(labels)}, dem={bool(dem)}")
        return tiff, labels, dem

    def _get_tif_data(self) -> tuple[rasterio.crs.CRS, rasterio.coords.BoundingBox, rasterio.transform.Affine, bool, float]:
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
        return f"SiteData(site_name={self.name}, site_path={self.path}, tiff_path={self.tiff_path}, label_path={self.label_path}, dem_path={self.dem_path})"

class DataPoint:
    def __init__(self, site_name:str, label: Label, image:np.ndarray, image_bounds: tuple[float, float, float, float], m_px: float):
        self.name = site_name
        self.label = label
        self.image = image
        self.bounds = shapely.box(*image_bounds)
        self.m_px = m_px

    def __str__(self):
        return f"DataPoint(site_name={self.name}, label={self.label}, image_shape={self.image.shape}, bounds={self.bounds.bounds}, m_px={self.m_px})"

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