import argparse
import sys
from pathlib import Path

# Add parent directory to path
UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import PolygonData, AREA_NAMES, WINDOW_SIZE, STRIDE
from text import title

# Import processing functions from each module
import extract
import crop
import resize

from negatives import extract_negatives_area_parallel, save_negatives_boundaries


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the full geoglyph processing pipeline.")
    parser.add_argument(
        "--area",
        type=str,
        nargs="+",
        choices=AREA_NAMES,
        required=True,
        help="Study area(s) to process.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["extract", "resize", "crop", "negatives"],
        default=["extract", "resize", "crop", "negatives"],
        help="Processing steps to run (default: all).",
    )
    parser.add_argument(
        "--savestop",
        type =str,
        nargs="+",
        choices=["extract", "resize"],
        default = [],
        help="Step at which to save intermediate results (extraction and resizing steps only).",
    )

    return parser.parse_args()

def run_pipeline(area: str, steps: list, savestop: list) -> None:
    """Run the processing pipeline for a given area.
    
    Args:
        area: Study area to process.
        steps: List of processing steps to run.
        savestop: List of steps at which to save intermediate results.
    """
    print(title(f"Starting pipeline for area: {area}"))
    
    geos = None
    img_arrays = None
    resized_arrays = None
    
    if "extract" in steps:
        print(title("STEP 1: Extracting polygons"))
        geos, geometries = extract.extract_area(area)
        # Convert (C, H, W) to (H, W, C) for LCI and upcoming steps
        img_arrays = [geom["chunk"].transpose(1, 2, 0) for geom in geometries]
        if "extract" in savestop:
            extract.save_data(area, geos, geometries)
    
    if "resize" in steps:
        print(title("STEP 2: Resizing polygons"))
        resized_arrays = []
        resized_geos = []
        for geo, img_array in zip(geos, img_arrays):
            print(f"Resizing polygon ID {geo.id}...")
            resized_array = resize.resize_polygon(geo, img_array, scale=resize.SCALE_FACTORS[area])
            resized_arrays.append(resized_array)
            resized_geos.append(geo)
        if "resize" in savestop:
            resize.save_data(area, resized_geos, resized_arrays)
        
        geos = resized_geos
        img_arrays = resized_arrays
    
    if "crop" in steps:
        print(title("STEP 3: Generating crops"))
        updated_geos = []
        for geos, img_array in zip(geos, img_arrays):
            print(f"Generating crops for polygon ID {geos.id}...")
            for id, geo_crop in enumerate(crop.make_crops(geos, img_array, WINDOW_SIZE, STRIDE)):
                print(f"Saving crop ID {id}...")
                crop_path = crop.make_crop_path(geos, area, id)
                updated_geo = crop.save_polygon_crop(geos, geo_crop, crop_path)
                updated_geos.append(updated_geo)
        
        PolygonData.save_polygons(updated_geos)  # Save all updated metadata with crop paths
    
    if "negatives" in steps:
        print(title("STEP 4: Extracting negatives"))
        save_boundaries, area_tif = extract_negatives_area_parallel(area)
        save_negatives_boundaries(save_boundaries, area, area_tif.crs)
    
    print(title(f"Pipeline complete for area: {area}"))

if __name__ == "__main__":
    args = parse_args()
    
    for area in args.area:
        run_pipeline(area, args.steps, args.savestop)