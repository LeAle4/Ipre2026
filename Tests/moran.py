import numpy as np
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from libpysal.weights import lat2W
from esda.moran import Moran
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TARGET_SIZE = (100, 100)  # width, height in pixels
RESULTS_CSV = BASE_DIR / "Tests" / "moran_results.csv"
RESULTS_BOX_PNG = BASE_DIR / "Tests" / "moran_box.png"
RESULTS_HIST_PNG = BASE_DIR / "Tests" / "moran_hist.png"

def load_json_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_images(summary):
    for polygon in summary['polygons']:
        img_id = polygon['polygon_index']
        class_ = polygon['class']
        img_name = polygon["files"]['ortho_jpeg']
        img_sizes = polygon["bbox_size_meters"]
        yield {"id": img_id, "class": class_, "img_name": img_name, "img_sizes": img_sizes}


def load_image(path):
    return Image.open(path)

def morans_i(image_df, adjacency: str = "rook", wrap: bool = False, nan_policy: str = "omit") -> float:
    """
    Compute Moran's I for a 2D image represented as a pandas DataFrame using libpysal/esda.

    Args:
        image_df: pandas DataFrame (H x W) where each cell is a pixel value.
        adjacency: 'rook' (4-neighbors) or 'queen' (8-neighbors) contiguity.
        wrap: whether to wrap the lattice at boundaries (toroidal). Default False.
        nan_policy: 'omit' to drop NaN pixels from the analysis; 'propagate' to keep
                    NaNs (will likely result in NaN I).

    Returns:
        Moran's I statistic (float).
    """

    # Ensure float array
    arr = np.array(image_df, dtype=float)
    nrows, ncols = arr.shape

    # Build lattice contiguity weights
    rook = True if adjacency.lower() == "rook" else False
    w = lat2W(nrows, ncols, rook=rook, id_type="int")
    w.transform = "r"  # row-standardize

    y = arr.ravel()

    if nan_policy == "omit":
        if np.isnan(y).any():
            # Older libpysal lacks W.subgraph; impute NaNs with the mean of valid pixels
            mean_val = np.nanmean(y)
            y = np.where(np.isnan(y), mean_val, y)
    elif nan_policy == "propagate":
        pass  # allow NaNs to propagate into Moran
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    mi = Moran(y, w)
    return float(mi.I)


if __name__ == "__main__":
    unita_summary = load_json_summary(DATA_DIR / "lluta_polygons/summary.json")
    results = []
    for image in parse_images(unita_summary):
        img_path = DATA_DIR / "lluta_polygons" / image["img_name"]
        img = load_image(img_path)
        img_gray = img.convert("L").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        img_array = np.array(img_gray)
        img_df = pd.DataFrame(img_array)
        I = morans_i(img_df, adjacency="rook", wrap=False, nan_policy="omit")
        results.append({"id": image['id'], "class": image['class'], "moran": I})
        print(f"Image ID: {image['id']}, Class: {image['class']}, Moran's I: {I:.4f}")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"Saved Moran's I results to {RESULTS_CSV}")

    # Box plot by class
    classes = sorted(df_results['class'].unique())

    # Box plot by class
    plt.figure(figsize=(8, 5))
    data = [df_results[df_results['class'] == cls]['moran'] for cls in classes]
    plt.boxplot(data, tick_labels=[f"class {cls}" for cls in classes], showfliers=True)
    plt.ylabel("Moran's I")
    plt.title("Moran's I distribution by class")
    plt.tight_layout()
    plt.savefig(RESULTS_BOX_PNG, dpi=200)
    print(f"Saved box plot to {RESULTS_BOX_PNG}")

    # Histogram by class
    plt.figure(figsize=(8, 5))
    for cls in classes:
        subset = df_results[df_results['class'] == cls]['moran']
        plt.hist(subset, bins=20, alpha=0.5, label=f"class {cls}")
    plt.xlabel("Moran's I")
    plt.ylabel("Frequency")
    plt.title("Moran's I histogram by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_HIST_PNG, dpi=200)
    print(f"Saved histogram to {RESULTS_HIST_PNG}")
    plt.show()