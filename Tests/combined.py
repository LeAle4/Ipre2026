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
AREA_SUBDIR = "unita_polygons"  # change this to switch dataset folder
TARGET_SIZE = (300, 300)  # width, height in pixels
RESULTS_CSV = BASE_DIR / "Tests" / "combined_results.csv"
SCATTER_PNG = BASE_DIR / "Tests" / "unita" / "unita_combined300x300.png"


def load_json_summary(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_images(summary):
    for polygon in summary["polygons"]:
        img_id = polygon["polygon_index"]
        class_ = polygon["class"]
        img_name = polygon["files"]["ortho_jpeg"]
        img_sizes = polygon["bbox_size_meters"]
        yield {"id": img_id, "class": class_, "img_name": img_name, "img_sizes": img_sizes}


def load_image(path):
    return Image.open(path)


def morans_i(image_df, adjacency: str = "rook", wrap: bool = False, nan_policy: str = "omit") -> float:
    """Compute Moran's I for a 2D image using libpysal/esda."""
    arr = np.array(image_df, dtype=float)
    nrows, ncols = arr.shape

    rook = True if adjacency.lower() == "rook" else False
    w = lat2W(nrows, ncols, rook=rook, id_type="int")
    w.transform = "r"

    y = arr.ravel()

    if nan_policy == "omit":
        if np.isnan(y).any():
            mean_val = np.nanmean(y)
            y = np.where(np.isnan(y), mean_val, y)
    elif nan_policy == "propagate":
        pass
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    mi = Moran(y, w)
    return float(mi.I)


def spatial_entropy(
    image_df,
    adjacency: str = "rook",
    wrap: bool = False,
    nan_policy: str = "omit",
    bins: int = 256,
) -> float:
    """Compute 2D spatial entropy from neighbor pairs using libpysal lattice weights."""
    arr = np.array(image_df, dtype=float)
    nrows, ncols = arr.shape

    rook = True if adjacency.lower() == "rook" else False
    w = lat2W(nrows, ncols, rook=rook, id_type="int")

    y = arr.ravel()

    if nan_policy == "propagate" and np.isnan(y).any():
        return float("nan")
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    pairs_a = []
    pairs_b = []
    for i, neighs in w.neighbors.items():
        for j in neighs:
            if j > i:
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
        return 0.0

    hist, _, _ = np.histogram2d(
        pairs_a, pairs_b, bins=bins, range=[[vmin, vmax], [vmin, vmax]]
    )
    p = hist / hist.sum()
    p = p[p > 0]
    entropy_bits = -np.sum(p * np.log2(p))
    return float(entropy_bits)


if __name__ == "__main__":
    unita_summary = load_json_summary(DATA_DIR / f"{AREA_SUBDIR}/summary.json")
    results = []
    
    for image in parse_images(unita_summary):
        img_path = DATA_DIR / AREA_SUBDIR / image["img_name"]
        img = load_image(img_path)
        img_gray = img.convert("L").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
        img_array = np.array(img_gray)
        img_df = pd.DataFrame(img_array)
        
        # Calculate both metrics
        moran = morans_i(img_df, adjacency="rook", wrap=False, nan_policy="omit")
        entropy = spatial_entropy(img_df, adjacency="rook", wrap=False, nan_policy="omit")
        
        results.append({
            "id": image["id"],
            "class": image["class"],
            "moran": moran,
            "entropy": entropy
        })
        print(f"Image ID: {image['id']}, Class: {image['class']}, Moran's I: {moran:.4f}, Entropy: {entropy:.4f}")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"Saved combined results to {RESULTS_CSV}")

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Separate by class
    class1 = df_results[df_results["class"] == 1]
    other_classes = df_results[df_results["class"] != 1]
    
    # Plot class 1 as green
    plt.scatter(class1["entropy"], class1["moran"], c="green", label="Class 1", alpha=0.7, s=50)
    
    # Plot other classes as black
    plt.scatter(other_classes["entropy"], other_classes["moran"], c="black", label="Other Classes", alpha=0.7, s=50)
    
    plt.xlabel("Entropy (bits)")
    plt.ylabel("Moran's I")
    plt.title("Spatial Entropy vs Moran's I by Class")
    # Display the resized target size on the figure
    plt.figtext(0.99, 0.01, f"Resized to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
                ha="right", va="bottom", fontsize=9, color="gray")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(SCATTER_PNG, dpi=200)
    print(f"Saved scatter plot to {SCATTER_PNG}")
    
    # Show plot
    plt.show()
