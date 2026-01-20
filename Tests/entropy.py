import numpy as np
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from libpysal.weights import lat2W
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TARGET_SIZE = (100, 100)  # width, height in pixels
RESULTS_CSV = BASE_DIR / "Tests" / "entropy_results.csv"
RESULTS_BOX_PNG = BASE_DIR / "Tests" / "entropy_box.png"
RESULTS_HIST_PNG = BASE_DIR / "Tests" / "entropy_hist.png"

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

def spatial_entropy(
    image_df,
    adjacency: str = "rook",
    wrap: bool = False,
    nan_policy: str = "omit",
    bins: int = 256,
) -> float:
    """
    Compute 2D spatial entropy from neighbor pairs using libpysal lattice weights.

    Forms neighbor pairs from a contiguity lattice (rook or queen), builds a joint
    histogram of paired pixel intensities (binned), and computes Shannon entropy.

    Args:
        image_df: pandas DataFrame (H x W) where each cell is a pixel value.
        adjacency: 'rook' (4-neighbors) or 'queen' (8-neighbors) contiguity.
        wrap: kept for API symmetry; lat2W here does not torus-wrap.
        nan_policy: 'omit' to drop any pair containing NaN; 'propagate' to allow NaN
                    (returns NaN if any are present).
        bins: number of bins for the joint histogram (default 256 for 8-bit grayscale).

    Returns:
        Shannon entropy (bits) of the joint neighbor distribution.
    """

    # Ensure float array
    arr = np.array(image_df, dtype=float)
    nrows, ncols = arr.shape

    # Build lattice contiguity weights
    rook = True if adjacency.lower() == "rook" else False
    w = lat2W(nrows, ncols, rook=rook, id_type="int")

    y = arr.ravel()

    if nan_policy == "propagate" and np.isnan(y).any():
        return float("nan")
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    # Collect neighbor pairs (upper triangle to avoid duplicates)
    pairs_a = []
    pairs_b = []
    for i, neighs in w.neighbors.items():
        for j in neighs:
            if j > i:  # avoid double counting pairs
                a, b = y[i], y[j]
                if nan_policy == "omit" and (np.isnan(a) or np.isnan(b)):
                    continue
                pairs_a.append(a)
                pairs_b.append(b)

    if len(pairs_a) == 0:
        return float("nan")

    pairs_a = np.array(pairs_a)
    pairs_b = np.array(pairs_b)

    vmin = min(pairs_a.min(), pairs_b.min())
    vmax = max(pairs_a.max(), pairs_b.max())
    if vmax == vmin:
        return 0.0  # no variability => zero entropy

    # Build 2D histogram of neighbor pairs
    hist, _, _ = np.histogram2d(
        pairs_a, pairs_b, bins=bins, range=[[vmin, vmax], [vmin, vmax]]
    )
    p = hist / hist.sum()
    p = p[p > 0]  # only non-zero probabilities
    entropy_bits = -np.sum(p * np.log2(p))
    return float(entropy_bits)


if __name__ == "__main__":
    unita_summary = load_json_summary(DATA_DIR / "lluta_polygons/summary.json")
    results = []
    for image in parse_images(unita_summary):
        img_path = DATA_DIR / "lluta_polygons" / image["img_name"]
        img = load_image(img_path)
        img_gray = img.convert("L").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        img_array = np.array(img_gray)
        img_df = pd.DataFrame(img_array)
        ent = spatial_entropy(img_df, adjacency="rook", wrap=False, nan_policy="omit")
        results.append({"id": image['id'], "class": image['class'], "entropy": ent})
        print(f"Image ID: {image['id']}, Class: {image['class']}, Entropy: {ent:.4f}")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"Saved entropy results to {RESULTS_CSV}")

    # Box plot by class
    classes = sorted(df_results['class'].unique())

    # Box plot by class
    plt.figure(figsize=(8, 5))
    data = [df_results[df_results['class'] == cls]['entropy'] for cls in classes]
    plt.boxplot(data, tick_labels=[f"class {cls}" for cls in classes], showfliers=True)
    plt.ylabel("Entropy (bits)")
    plt.title("Entropy distribution by class")
    plt.tight_layout()
    plt.savefig(RESULTS_BOX_PNG, dpi=200)
    print(f"Saved box plot to {RESULTS_BOX_PNG}")

    # Histogram by class
    plt.figure(figsize=(8, 5))
    for cls in classes:
        subset = df_results[df_results['class'] == cls]['entropy']
        plt.hist(subset, bins=20, alpha=0.5, label=f"class {cls}")
    plt.xlabel("Entropy (bits)")
    plt.ylabel("Frequency")
    plt.title("Entropy histogram by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_HIST_PNG, dpi=200)
    print(f"Saved histogram to {RESULTS_HIST_PNG}")
    plt.show()
