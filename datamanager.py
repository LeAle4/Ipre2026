from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from parameters import GEO_CLASS, GROUND_CLASS
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
        self.crs, self.bounds, self.transform, self.is_geographic, self.m_px, self.nodata_value, self.channels = self._get_tif_data()
        self.width = self.bounds.right - self.bounds.left
        self.height = self.bounds.top - self.bounds.bottom
        self.width_px, self.height_px = self.width / self.m_px, self.height / self.m_px
        self.area = self.width * self.height

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

    def _get_tif_data(self) \
        -> tuple[rasterio.crs.CRS, rasterio.coords.BoundingBox, rasterio.transform.Affine, bool, float, float, int]:
        
        if self.tiff_path is not None:
            with rasterio.open(self.tiff_path) as dataset:
                crs = dataset.crs
                bounds = dataset.bounds
                transform = dataset.transform
                is_geographic = dataset.crs.is_geographic
                nodata = dataset.nodata
                channels = dataset.count
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
        return crs, bounds, transform, is_geographic, m_px, nodata, channels

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
        self.area = polygon_bounds.area
        self.origin_scale = m_px
        self.crs = crs
        self.image = image
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

    def __init__(self, data_point: DataPoint, crop_image: np.ndarray, crop_bounds: shapely.geometry.Polygon, intersection_proportion: float = 0.0):
        self.data_point = data_point
        self.crop_image = crop_image
        self.crop_bounds = crop_bounds
        self.intersection_proportion = intersection_proportion

class DataWriter:
    BASE = Path("ML")
    SITE_COLUMNS = ("SITE_NAME", 
                        "CRS", 
                        "QUAD_AREA",
                        "QUAD_WIDTH_PX", 
                        "QUAD_HEIGHT_PX", 
                        "QUAD_WIDTH_M", 
                        "QUAD_HEIGHT_M",
                        "SCALE_MPX", 
                        "NUMBER_OF_LABELS",
                        "NUMBER_OF_GEOGLYPHS",
                        "NUMBER_OF_GROUND",
                        "NUMBER_OF_IMAGES",
                        "NUMBER_OF_POSITIVES",
                        "NUMBER_OF_NEGATIVES",
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
                    )
    IMG_COLUMNS = ("SITE_NAME",
                        "ID",
                        "GEO_ID",
                        "CROP_ID",
                        "DATA_LABEL",
                        "SCALE_MPX",
                        "CENTER_X",
                        "CENTER_Y",
                        "AREA",
                        "CHANNELS",
                        "CRS",
                        "THRESHOLD_CLEAR"
                    )
    SHAPE_GEOMETRIES_COLUMNS = ("SITE_NAME", "ID", "GEO_ID", "AREA", "CENTER_X", "CENTER_Y", "DATA_LABEL", "GEOMETRY")
    GEOMETRY_COLUMN = "GEOMETRY"

    def __init__(self, sites_data: dict[str, SiteData]):
        self.site_paths = {site_name: self.BASE / site_name for site_name in sites_data.keys()}
        self.sites_metadata = self._init_sites_metadata()
        self.image_metadata = self._init_image_metadata()
        self.working_site = None
        self.working_datapoint = None
        self.shape_records = []
        self.polygon_records = []

        self._ensure_sites_dir()

    @property
    def positive_count(self) -> int:
        """Return the number of positive samples in the image metadata."""
        return len(self.image_metadata[self.image_metadata["DATA_LABEL"] == GEO_CLASS])

    @property
    def negative_count(self) -> int:
        """Return the number of negative samples in the image metadata."""
        return len(self.image_metadata[self.image_metadata["DATA_LABEL"] == GROUND_CLASS])

    def _init_sites_metadata(self) -> pd.DataFrame:
        """Initialize the sites metadata DataFrame with the appropriate columns."""
        return pd.DataFrame(columns=DataWriter.SITE_COLUMNS)

    def _init_image_metadata(self) -> pd.DataFrame:
        """Initialize the image metadata DataFrame with the appropriate columns."""
        return pd.DataFrame(columns=DataWriter.IMG_COLUMNS)

    def _construct_full_id(self, site_name:str, data_point: DataPoint, crop_number:int) -> str:
        """Construct a full ID string for the data point or crop.

        Args:
            site_name: The name of the site.
            data_point: The DataPoint object.
            crop_number: The number of the crop.
        """
        return f"{site_name}_{data_point.id}" + (f"_{crop_number}" )

    def _ensure_sites_dir(self) -> None:
        """Ensure that the directories for all sites exist."""
        for site_path in self.site_paths.values():
            site_path.mkdir(parents=True, exist_ok=True)

    def _process_channel_statistics(self, site_data: SiteData) -> dict[str, float]:
        """Calculate min, max, and average for each channel in the site's TIFF image.

        Args:
            site_data: The SiteData object for the site.

        Returns:
            A dictionary containing the min, max, and average values for each channel.
        """
        channel_stats = {}
        with site_data.access_tif() as tif_area:
            print(f"Calculating channel statistics for site: {site_data.name}")

            for channel in range(1, site_data.channels + 1):
                print(
                    f"Processing channel {channel}/{site_data.channels} "
                    f"for site: {site_data.name}"
                )

                minimum = np.inf
                maximum = -np.inf
                total = 0.0
                count = 0

                for _, window in tif_area.block_windows(channel):
                    data = tif_area.read(channel, window=window, masked=True)

                    if data.count() == 0:
                        continue

                    minimum = min(minimum, float(data.min()))
                    maximum = max(maximum, float(data.max()))
                    total += float(data.sum())
                    count += data.count()

                channel_stats[f"CHANNEL_{channel}_MIN"] = minimum
                channel_stats[f"CHANNEL_{channel}_MAX"] = maximum
                channel_stats[f"CHANNEL_{channel}_AVG"] = total / count

        return channel_stats

    def start_site(self, site_data: SiteData) -> None:
        """Start processing a new site.

        Args:
            site_data: The SiteData object for the site.
        """
        self.working_site = {}
        self.working_site["SITE_NAME"] = site_data.name
        self.working_site["CRS"] = site_data.crs.to_string()
        self.working_site["QUAD_AREA"] = site_data.area
        self.working_site["QUAD_WIDTH_PX"] = site_data.width_px
        self.working_site["QUAD_HEIGHT_PX"] = site_data.height_px
        self.working_site["QUAD_WIDTH_M"] = site_data.width
        self.working_site["QUAD_HEIGHT_M"] = site_data.height
        self.working_site["SCALE_MPX"] = site_data.m_px
        self.working_site["NUMBER_OF_IMAGES"] = 0
        self.working_site["NUMBER_OF_POSITIVES"] = 0
        self.working_site["NUMBER_OF_NEGATIVES"] = 0
        labels = site_data.label_data()
        self.working_site["NUMBER_OF_LABELS"] = len(labels)
        self.working_site["NUMBER_OF_GEOGLYPHS"] = len(labels.filter_class(GEO_CLASS))
        self.working_site["NUMBER_OF_GROUND"] = len(labels.filter_class(GROUND_CLASS))
        self.working_site["CHANNELS"] = site_data.channels
        self.working_site.update(self._process_channel_statistics(site_data))

    def add_crop(self, datapoint:DataPoint, crop: Crop | None = None, crop_n:int = 1) -> None:
        """Add a new image datapoint to the metadata.

        Args:
            datapoint: The DataPoint object.
            crop: The Crop object (optional).
            threshold_clear: The threshold clear value for the image.
        """
        if self.working_site is None:
            raise RuntimeError("No site is currently being processed. Call start_site() before adding images.")
        
        full_id = self._construct_full_id(self.working_site["SITE_NAME"], datapoint, crop_n)
        
        if crop is None:
            threshold_clear = 1.0
            image = datapoint.image
        else:
            threshold_clear = crop.intersection_proportion
            image = crop.crop_image

        self.image_metadata.loc[len(self.image_metadata)] = {
            "SITE_NAME": self.working_site["SITE_NAME"],
            "ID": full_id,
            "GEO_ID": datapoint.id,
            "CROP_ID": f"{crop_n:05d}",
            "DATA_LABEL": datapoint.data_label,
            "SCALE_MPX": datapoint.m_px,
            "CENTER_X": datapoint.image_box.centroid.x,
            "CENTER_Y": datapoint.image_box.centroid.y,
            "CHANNELS": datapoint.image.shape[2],
            "CRS": datapoint.crs,
            "THRESHOLD_CLEAR": threshold_clear
        }
        self.working_site["NUMBER_OF_IMAGES"] += 1
        if datapoint.data_label == GEO_CLASS:
            self.working_site["NUMBER_OF_POSITIVES"] += 1
        elif datapoint.data_label == GROUND_CLASS:
            self.working_site["NUMBER_OF_NEGATIVES"] += 1

        if crop is None:
            geometry = datapoint.polygon_bounds
        else:
            geometry = crop.crop_bounds
        
        self.shape_records.append({
                    "SITE_NAME": self.working_site["SITE_NAME"],
                    "ID": full_id,
                    "GEO_ID": datapoint.id,
                    "DATA_LABEL": datapoint.data_label,
                    "AREA": datapoint.area,
                    "CENTER_X": datapoint.polygon_bounds.centroid.x,
                    "CENTER_Y": datapoint.polygon_bounds.centroid.y,
                    "THRESHOLD_CLEAR": threshold_clear,
                    "geometry": geometry,
                })

        img = Image.fromarray(image)
        img.save(self.BASE / self.working_site["SITE_NAME"] / f"{full_id}.png")

    def add_polygon_datapoint(self, datapoint: DataPoint) -> None:
        """Add a polygon datapoint to the polygon records for the current site.

        Args:
            datapoint: The DataPoint object representing the polygon.
        """
        if self.working_site is None:
            raise RuntimeError("No site is currently being processed. Call start_site() before adding polygon datapoints.")
        
        self.polygon_records.append({
            "SITE_NAME": self.working_site["SITE_NAME"],
            "GEO_ID": datapoint.id,
            "geometry": datapoint.polygon_bounds
        })

    def close_site(self) -> None:
        """Finalize the current site and write its metadata to the sites metadata DataFrame."""
        if self.working_site is None:
            raise RuntimeError("No site is currently being processed. Call start_site() before closing a site.")
        
        self.sites_metadata.loc[len(self.sites_metadata)] = self.working_site
        self.image_metadata.to_csv(self.BASE / self.working_site["SITE_NAME"] / "image_metadata.csv", index=False)
        shape_gdf = gpd.GeoDataFrame(
            self.shape_records,
            geometry="geometry",
            crs=self.working_site["CRS"]
        )
        shape_gdf.to_file(self.BASE / self.working_site["SITE_NAME"] / "shape_geometries.gpkg", driver="GPKG")
        
        polygon_gdf = gpd.GeoDataFrame(
            self.polygon_records,
            geometry="geometry",
            crs=self.working_site["CRS"]
        )
        polygon_gdf.to_file(self.BASE / self.working_site["SITE_NAME"] / "polygon_geometries.gpkg", driver="GPKG")

        self.working_site = None
        self.working_datapoint = None
        self.shape_records = []
        self.image_metadata = self._init_image_metadata()  # Reset image metadata for the next site
 
    def save_sites_metadata(self) -> None:
        """Save the sites metadata DataFrame to a CSV file."""
        self.sites_metadata.to_csv(self.BASE / "sites_metadata.csv", index=False)

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