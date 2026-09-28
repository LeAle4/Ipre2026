from __future__ import annotations
from operator import index

import numpy as np
import pandas as pd
import csv
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
        self.crs, self.bounds, self.transform, self.is_geographic, self.m_px, self.nodata_value = self._get_tif_data()

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

    def _get_tif_data(self) -> tuple[rasterio.crs.CRS, rasterio.coords.BoundingBox, rasterio.transform.Affine, bool, float, float]:
        if self.tiff_path is not None:
            with rasterio.open(self.tiff_path) as dataset:
                crs = dataset.crs
                bounds = dataset.bounds
                transform = dataset.transform
                is_geographic = dataset.crs.is_geographic
                nodata = dataset.nodata
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
        return crs, bounds, transform, is_geographic, m_px, nodata

    def __str__(self):
        return f"SiteData(site_name={self.name}, site_path={self.path}, tiff_path={self.tiff_path}, label_path={self.label_path}, dem_path={self.dem_path})"

class DataPoint:

    last_id = 0

    def __init__(self, site_name:str, data_label: int, polygon_bounds: MultiPolygon, crs:str, image:np.ndarray, image_bounds: tuple[float, float, float, float], m_px: float):
        self.id = f"{DataPoint.last_id:05d}"
        DataPoint.last_id += 1
        self.name = site_name
        self.data_label = data_label
        self.polygon_bounds = polygon_bounds
        self.origin_scale = m_px
        self.crs = crs
        self.image = image
        #image shape given is (channels, height, width) but we want to store it as (height, width, channels)
        if len(self.image.shape) == 3:
            self.image = np.transpose(self.image, (1, 2, 0))

        self.image_box = shapely.box(*image_bounds)
        self.m_px = m_px
        self.image_resized = False

    def modify_image(self, new_image: np.ndarray, new_scale: float) -> None:
        """Modify the image of the DataPoint and update its bounds and metric scale.

        Args:
            new_image: The new image to replace the current one.
            new_bounds: The new bounds of the image in the format (minx, miny, maxx, maxy).
            new_m_px: The new metric scale in meters per pixel.
        """
        self.image = new_image
        self.m_px = new_scale
        self.image_resized = True

    def __str__(self):
        return f"DataPoint(site_name={self.name}, label={self.data_label}, polygon_bounds={self.polygon_bounds}, image_shape={self.image.shape}, bounds={self.image_box.bounds}, m_px={self.m_px})"

class Crop:

    last_id = 0

    def __init__(self, data_point: DataPoint, crop_image: np.ndarray, crop_bounds: shapely.geometry.Polygon, intersection_proportion: float = 0.0):
        self.id = f"{Crop.last_id:05d}"
        Crop.last_id += 1
        self.data_point = data_point
        self.crop_image = crop_image
        self.crop_bounds = crop_bounds
        self.intersection_proportion = intersection_proportion

class DataWriter:
    BASE = Path("ML")
    WRITE_BUFFER_SIZE = 100  # Number of entries to buffer before writing to CSV

    def __init__(self, sites_data: dict[str, SiteData]):
        self.site_paths = {site_name: self.BASE / site_name for site_name in sites_data.keys()}
        self.sites_metadata_path = self.BASE / "sites_metadata.csv"
        self.image_metadata_path = self.BASE / "image_metadata.csv"
        self.sites_metadata, self.image_metadata, self.shape_geometries = self._initialize_metadata()
        self.write_buffer = 0

        self._ensure_sites_dir()

    def _initialize_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:

        sites_metadata = pd.DataFrame(
            columns=["SITE_NAME", 
                        "CRS", 
                        "QUAD_AREA", 
                        "QUAD_CENTER_LATITUDE", 
                        "QUAD_CENTER_LONGITUDE", 
                        "QUAD_WIDTH_PX", 
                        "QUAD_HEIGHT_PX", 
                        "QUAD_WIDTH_M", 
                        "QUAD_HEIGHT_M",
                        "SCALE_MPX", 
                        "NUMBER_OF_LABELS",
                        "NUMBER_OF_GEOGLYPHS",
                        "NUMBER_OF_GROUND",
                        "CHANNELS",
                        "CHANNEL_1_MIN",
                        "CHANNEL_1_MAX",
                        "CHANNEL_1_AVG",
                        "CHANNEL_2_MIN",
                        "CHANNEL_2_MAX",
                        "CHANNEL_2_AVG",
                        "CHANNEL_3_MIN",
                    "CHANNEL_3_MAX",
                    "CHANNEL_3_AVG",
                    "CHANNEL_4_MIN",
                    "CHANNEL_4_MAX",
                    "CHANNEL_4_AVG",
                    ]
                        )

        image_metadata = pd.DataFrame(
            columns=["SITE_NAME",
                        "ID",
                        "GEO_ID",
                        "CROP_ID",
                        "DATA_LABEL",
                        "TOTAL_CROP_COUNT",
                        "SCALE_MPX",
                        "CHANNELS",
                        "CRS",
                        "THRESHOLD_CLEAR"
                    ]
            )

        shape_geometries = gpd.GeoDataFrame(
            columns=["SITE_NAME", "ID", "GEO_ID", "DATA_LABEL", "GEOMETRY"], geometry="GEOMETRY")

        return sites_metadata, image_metadata, shape_geometries

    def _construct_full_id(self, site_name:str, data_point: DataPoint, crop: Crop | None) -> str:
        """Construct a full ID string for the data point or crop.

        Args:
            site_name: The name of the site.
            data_point: The DataPoint object.
            crop: The Crop object (optional).
        """
        return f"{site_name}_{data_point.id}" + (f"_{crop.id}" if crop else "_0")

    def _flush_images(self) -> None:
        """Write the metadata DataFrames to CSV files."""
        self.sites_metadata.to_csv(self.sites_metadata_path, index=False)
        self.image_metadata.to_csv(self.image_metadata_path, index=False)
        self.shape_geometries.to_file(self.BASE / "shape_geometries.gpkg", driver="GPKG")
        self.write_buffer = 0

    def close(self) -> None:
        pass

    def _ensure_sites_dir(self) -> None:
        """Ensure that the directories for all sites exist."""
        for site_path in self.site_paths.values():
            site_path.mkdir(parents=True, exist_ok=True)

    def open_data_point(self, site_name:str, data_point: DataPoint) -> None:
        """Open a new data point for writing crops and metadata.

        Args:
            site_name: The name of the site.
            data_point: The DataPoint object.
        """
        pass

    def close_data_point(self, site_name:str, data_point: DataPoint) -> None:
        """Close the current data point and flush metadata if necessary.

        Args:
            site_name: The name of the site.
            data_point: The DataPoint object.
        """
        pass

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