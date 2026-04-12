import sys
import random
import shutil
import argparse
from pathlib import Path

UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from utils import Polygon
from handle import CLASSES, NEGATIVES_RATIO, PolygonData

def sample_n_crops_from_area(geo_list: list[Polygon], n: int) -> list[Path]:
    """Randomly sample n crop paths from a list of Polygon objects.
    
    Args:
        geo_list: List of Polygon objects to sample from.
        n: Number of crop paths to sample.
    Returns:
        List of sampled crop paths.
    """
    crop_list = []
    while len(crop_list) < n and geo_list:
        geo = random.choice(geo_list)
        if geo.crop_paths:
            crop_path = random.choice(geo.crop_paths)
            if crop_path not in crop_list:
                crop_list.append(crop_path)
        else:
            geo_list.remove(geo)  # Remove the polygon to avoid resampling
    
    return crop_list

def sample_n_negatives_from_area(geo_list: list[Polygon], n: int) -> list[Path]:
    """Randomly sample n crop paths from a list of Polygon objects.
    
    Args:
        geo_list: List of Polygon objects to sample from.
        n: Number of crop paths to sample.
    Returns:
        List of sampled crop paths.
    """
    negative_list = []
    while len(negative_list) < n and geo_list:
        geo = random.choice(geo_list)
        if geo.crop_paths:
            crop_path = random.choice(geo.crop_paths)
            if crop_path not in negative_list:
                negative_list.append(crop_path)
        else:
            geo_list.remove(geo)  # Remove the polygon to avoid resampling
    
    return negative_list

def create_test_batch(areas: list[str], area_weights: list[float], positive_size: int, negatives_ratio: float = NEGATIVES_RATIO) -> dict[str, dict[str, list[Path]]]:
    """Create a test batch of geoglyph crops from the specified areas.
    
    Args:
        areas: List of area names to include in the batch (e.g., ['unita', 'chugchug']).
        area_weights: List of weights corresponding to each area for sampling (e.g., [0.5, 0.5]).
        batch_size: Total number of crops to include in the batch.
        negatives_ratio: Ratio of negative samples to positive samples in the batch.
    Returns:
        List of polygons selected for the test batch.
    """
    #Load all of the polygons from each area
    loaded_polygons = {}
    for area in areas:
        print(f"Loading polygons from area: {area}")
        loaded_polygons[area] = {"positives": [], "negatives": []}
        for geo in PolygonData.polygons(area_filter=(area,), classes_filter=(CLASSES["geo"],)):
            loaded_polygons[area]["positives"].append(geo)
            print(f"Loaded {len(loaded_polygons[area]['positives'])} positive polygons from area: {area}")
        for geo in PolygonData.polygons(area_filter=(area,), classes_filter=(CLASSES["ground"],)):
            loaded_polygons[area]["negatives"].append(geo)
            print(f"Loaded {len(loaded_polygons[area]['negatives'])} negative polygons from area: {area}")

    #Calculate the number of samples to draw from each area based on the weights
    negative_size = int(positive_size * negatives_ratio)
    total_weight = sum(area_weights)

    selected_crops = {}
    for area, weight in zip(areas, area_weights):
        selected_crops[area] = {"positives": [], "negatives": []}
        num_positives = int(weight * positive_size / total_weight)
        num_negatives = int(weight * negative_size / total_weight)
        selected_crops[area]["positives"] = sample_n_crops_from_area(loaded_polygons[area]["positives"], num_positives)
        selected_crops[area]["negatives"] = sample_n_negatives_from_area(loaded_polygons[area]["negatives"], num_negatives)

    return selected_crops

def save_test_batch(selected_polygons: dict[str, dict[str, list[Path]]], save_path:Path) -> None:
    """Save the list of crop paths to a text file.
    
    Args:
        selected_polygons: List of selected crop paths.
        save_path: Path to save the text file containing the crop paths.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    for area, classes in selected_polygons.items():
        for class_name, crop_paths in classes.items():
            class_path = save_path / area / class_name
            class_path.mkdir(parents=True, exist_ok=True)
            for crop_path in crop_paths:
                destination = class_path / crop_path.name
                shutil.copy(crop_path, destination)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates a batch of sample images from the dataset.")
    parser.add_argument(
        "--areas",
        type=str,
        nargs = "+",
        choices=["unita", "chugchug", "lluta"],
        required=True,
        help="Study area to process.",
    )
    parser.add_argument(
        "--area_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each area when sampling. If omitted, equal weights are used.",
    )
    parser.add_argument(
        "--negatives_ratio",
        type=float,
        default=NEGATIVES_RATIO,
        help="Ratio of negative samples to positive samples in the batch (default: 1.0).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Total number of crops to include in the batch (default: 100).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="test_batch",
        help="Directory where sampled crops are copied (default: 'test_batch').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for reproducible sampling (default: random behavior).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using global random seed: {args.seed}")

    if args.area_weights is None:
        area_weights = [1.0] * len(args.areas)
    else:
        area_weights = args.area_weights

    if len(area_weights) != len(args.areas):
        raise ValueError(
            f"Expected {len(args.areas)} area weights for areas {args.areas}, got {len(area_weights)}."
        )

    output_dir = Path(args.save_path).resolve()
    print(f"Saving test batch to: {output_dir}")

    selected_crops = create_test_batch(
        areas=args.areas,
        area_weights=area_weights,
        positive_size=args.batch_size,
        negatives_ratio=args.negatives_ratio
    )
    save_test_batch(selected_crops, output_dir)