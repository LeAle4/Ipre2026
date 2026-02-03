import argparse
import sys
from pathlib import Path

# Add parent directory to path
UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from handle import AREA_NAMES, POLYGON_DATA_DIR
from text import title

# Import processing functions from each module
from extract import extract_area
from crop import crop_area
from resize import resize_area
from negatives import extract_negatives_area


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
    return parser.parse_args()


def run_pipeline(area: str, steps: list) -> None:
    """Run the processing pipeline for a given area.
    
    Args:
        area: Study area to process.
        steps: List of processing steps to run.
    """
    print(title(f"Starting pipeline for area: {area}"))
    
    # Ensure output directory exists
    POLYGON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if "extract" in steps:
        print(title("STEP 1: Extracting polygons"))
        extract_area(area)
    
    if "resize" in steps:
        print(title("STEP 2: Resizing polygons"))
        resize_area(area)
    
    if "crop" in steps:
        print(title("STEP 3: Generating crops"))
        crop_area(area)
    
    if "negatives" in steps:
        print(title("STEP 4: Extracting negatives"))
        extract_negatives_area(area)
    
    print(title(f"Pipeline complete for area: {area}"))


if __name__ == "__main__":
    args = parse_args()
    
    for area in args.area:
        run_pipeline(area, args.steps)