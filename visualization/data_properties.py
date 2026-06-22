import sys

import rasterio
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import CLASSES, PolygonData, get_area_tif, get_area_DEM, count_crops, count_negatives
from utils import get_file_resolution

def calculate_size(tif_path):
    """Calculate the size of a given study area.
    Args:
        tif_path: Path to the study area's TIF file.
    Returns:
        Tuple containing the size of the study area.
    """
    with rasterio.open(tif_path) as src:
        pixel_width, pixel_height = src.shape
        #Gb
        file_size = tif_path.stat().st_size / 1e9
        return (pixel_width, pixel_height, file_size)

def calculate_site_properties(area:str) -> dict:
    """Calculate the properties of a given study area.
    Args:
        area: Study area name (e.g., 'unita', 'chugchug', 'lluta').
    Returns:
        Dictionary containing the properties of the study area.
    """
    tif_file = get_area_tif(area)
    px_w, px_h, file_size = calculate_size(tif_file)

    num_positive_samples = count_crops(area)
    num_negative_samples = count_negatives(area)
    num_total_samples = num_positive_samples + num_negative_samples
    num_polygons = PolygonData.positive_count((area,))
    avg_crops_per_polygon = round(num_positive_samples / num_polygons)
    resolution =  round(get_file_resolution(tif_file),4)

    width = round(px_w * resolution /1000,4)
    height = round(px_h * resolution /1000,4)

    return {
        "Ventanas de geoglifos": [num_positive_samples],
        "Ventanas negativas": [num_negative_samples],
        "Total de ventanas": [num_total_samples],
        "Número de polígonos": [num_polygons],
        "Ventanas de geoglifos por polígono": [avg_crops_per_polygon],
        "Resolución (m/px)": [resolution],
        "Tamaño del cuadrante (px)": [str(px_w) + "x" + str(px_h)],
        "Tamaño del cuadrante (km)": [str(width) + "x" + str(height)],
        "Tamaño de archivo (Gb)": [file_size],
    }

def calculate_geoglyph_crop_distribution(area):
    crop_amounts = []
    for polygon in PolygonData.polygons(area_filter=(area,), classes_filter=(CLASSES["geo"],)):
        crop_amounts.append(polygon.num_crops())
    
    return crop_amounts

def plot_geoglyph_crop_distribution(areas):
    for area in areas:
        crop_amounts = calculate_geoglyph_crop_distribution(area)
        plt.hist(crop_amounts, bins=range(min(crop_amounts), max(crop_amounts) + 2), align='left')
        plt.xlabel("Número de ventanas por polígono")
        plt.ylabel("Número de polígonos")
        plt.title(f"Distribución de ventanas de geoglifos en {area}")
        plt.show()
        
def calculate_DEM_properties(area:str) -> dict:
    """Calculate the properties of a given DEM file.
    Args:
        area: Name of the study area ('unita', 'chugchug', or 'lluta').
    Returns:
        Dictionary containing the properties of the DEM file."""
    DEM_path = get_area_DEM(area)
    px_w, px_h,file_size = calculate_size(DEM_path)
    
    resolution = round(get_file_resolution(DEM_path), 4)

    width = round(px_w * resolution /1000,4)
    height = round(px_h * resolution /1000,4)

    return {
        "Tamaño del cuadrante (px)": [str(px_w) + "x" + str(px_h)],
        "Tamaño del cuadrante (km)": [str(width) + "x" + str(height)],
        "Resolución (m/px)": [resolution],
        "Tamaño del archivo (Gb)": [file_size],
    }

def plot_site_properties(areas:tuple[str, ...]) -> None:
    """Plot the properties of a given study area.
    Args:
        area: Study area name (e.g., 'unita', 'chugchug', 'lluta').
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    # Create one DataFrame per area
    dataframes = []
    for area in areas:
        df = pd.DataFrame(calculate_site_properties(area))
        df.index = [area]
        dataframes.append(df)
    
    # Concatenate DataFrames
    properties = pd.concat(dataframes)
    properties = properties.T
    print(properties)
    
    table = ax.table(
        cellText=properties.values, 
        colLabels=properties.columns, 
        rowLabels=properties.index, 
        cellLoc="center",
        colLoc="center",
        loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)
    plt.show()

def plot_DEM_properties(areas:tuple[str, ...]) -> None:
    """Plot the properties of a given DEM file.
    Args:
        area: Name of the study area ('unita', 'chugchug', or 'lluta').
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    # Create one DataFrame per area
    dataframes = []
    for area in areas:
        df = pd.DataFrame(calculate_DEM_properties(area))
        df.index = [area]
        dataframes.append(df)
    
    # Concatenate DataFrames
    properties = pd.concat(dataframes)
    properties = properties.T
    print(properties)
    
    table = ax.table(
        cellText=properties.values, 
        colLabels=properties.columns, 
        rowLabels=properties.index, 
        cellLoc="center",
        colLoc="center",
        loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)
    plt.show()

if __name__ == "__main__":
    #plot_site_properties(("unita", "chugchug", "lluta"))
    #plot_DEM_properties(("unita", "chugchug"))
    plot_geoglyph_crop_distribution(("unita", "chugchug", "lluta"))