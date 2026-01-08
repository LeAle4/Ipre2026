"""
Image entropy helper utilities.

This module loads an image, converts it to a grayscale matrix (0..1), and
reuses the entropy routines defined in main.py to compute all supported
measures.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt

from main import (
    compute_all_entropies,
    load_image_as_grayscale_matrix,
    save_matrix_as_image,
    visualize_matrix,
    create_comparison_plot,
    plot_matrices_with_entropies,
    plot_entropy_comparison,
)


def analyze_image_entropies(
    image_path: str,
    *,
    size: int = 400,
    output_dir: str = "output",
    visualize: bool = False,
    save_image: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    """
    Load an image (grayscale 0..1) and run every entropy measure defined in main.py.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    size : int, optional
        Target square size to resize the image before binarization.
    output_dir : str, optional
        Directory where the grayscale image is saved when save_image is True.
    visualize : bool, optional
        Whether to show the grayscale image.
    save_image : bool, optional
        Whether to save the grayscale image PNG in output_dir.

    Returns
    -------
    (results, matrix)
        results: dict with entropy values (Sample, Approximate, Fuzzy).
        matrix: numpy.ndarray containing the grayscale image (0..1).
    """

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    matrix = load_image_as_grayscale_matrix(str(img_path), size=size)

    if save_image:
        # Save a reusable copy of the grayscale image for inspection.
        save_matrix_as_image(matrix, img_path.stem, output_dir)

    if visualize:
        visualize_matrix(matrix, title=f"{img_path.name} (grayscale)", show=True)

    results = compute_all_entropies(matrix, matrix_name=img_path.name)
    return results, matrix


__all__ = ["analyze_image_entropies"]


def visualize_entropy_results(results: Dict[str, Any], title: str = "Image Entropy") -> None:
    """Plot a bar chart of the entropy values contained in ``results``.

    Expected keys: SampleEntropy, ApproximateEntropy, FuzzyEntropy.
    Missing/None values are shown as 0.
    """

    entropy_keys = [
        "SampleEntropy",
        "ApproximateEntropy",
        "FuzzyEntropy",
    ]

    values = [results.get(k) if results.get(k) is not None else 0 for k in entropy_keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(entropy_keys, values, color="steelblue", alpha=0.8)
    ax.set_ylabel("Entropy Value")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


__all__.append("visualize_entropy_results")


def visualize_image_and_entropies(
    matrix: Any,
    results: Dict[str, Any],
    title: str = "Image and Entropy Measures",
) -> None:
    """Show the grayscale image alongside a bar chart of its entropy measures."""

    entropy_keys = [
        "SampleEntropy",
        "ApproximateEntropy",
        "FuzzyEntropy",
    ]
    values = [results.get(k) if results.get(k) is not None else 0 for k in entropy_keys]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    ax_img.imshow(matrix, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax_img.set_title("Grayscale Image")
    ax_img.axis("off")

    bars = ax_bar.bar(entropy_keys, values, color="steelblue", alpha=0.85)
    ax_bar.set_ylabel("Entropy Value")
    ax_bar.set_title("Entropy Measures")
    ax_bar.tick_params(axis="x", rotation=25)
    ax_bar.grid(axis="y", alpha=0.3)

    # Add value labels above bars for quick reading.
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


__all__.append("visualize_image_and_entropies")


def analyze_directory_entropies(
    folder_path: str,
    *,
    size: int = 400,
    output_dir: str = "output",
    visualize: bool = True,
    save_images: bool = False,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
) -> Tuple[list, dict]:
    """
    Compute entropies for all images in a directory and generate comparison plots.

    Parameters
    ----------
    folder_path : str
        Directory containing images.
    size : int
        Target side length for resizing images prior to analysis.
    output_dir : str
        Folder for saving any generated images/plots.
    visualize : bool
        Whether to produce summary comparison plots.
    save_images : bool
        Save the normalized grayscale images to `output_dir`.
    exts : tuple[str, ...]
        Allowed file extensions to consider as images.

    Returns
    -------
    (results_list, matrices_dict)
        results_list: list of entropy dictionaries (Sample/Approx/Fuzzy).
        matrices_dict: mapping of image name -> grayscale matrix (0..1).
    """

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    matrices = {}
    results = []

    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if not files:
        print(f"No image files found in {folder} with extensions {exts}.")
        return results, matrices

    for img_path in files:
        try:
            matrix = load_image_as_grayscale_matrix(str(img_path), size=size)
            key = img_path.name
            matrices[key] = matrix

            if save_images:
                save_matrix_as_image(matrix, img_path.stem, output_dir)

            res = compute_all_entropies(matrix, matrix_name=key)
            results.append(res)
        except Exception as e:
            print(f"Warning: Skipping {img_path}: {e}")

    if visualize:
        # Grid of images
        create_comparison_plot(matrices, save_path=str(Path(output_dir) / "dir_comparison.png"), show=False)
        # Per-image panel with values
        plot_matrices_with_entropies(
            matrices,
            results,
            save_path=str(Path(output_dir) / "dir_matrices_with_entropies.png"),
            show=False,
        )
        # Bar charts by entropy type
        plot_entropy_comparison(
            results,
            save_path=str(Path(output_dir) / "dir_entropy_comparison.png"),
            show=False,
        )

    return results, matrices


__all__.append("analyze_directory_entropies")

if __name__ == "__main__":
    # Example: analyze a single image
    # entropies, grayscale_matrix = analyze_image_entropies(
    #     "image.png",
    #     size=400,
    #     output_dir="output",
    #     visualize=True,
    #     save_image=True,
    # )
    # visualize_image_and_entropies(grayscale_matrix, entropies, title="Entropy Measures for Sample Image")

    # Example: analyze all images in a folder
    results, matrices = analyze_directory_entropies(
        folder_path="./pics",
        size=128,
        output_dir="output",
        visualize=True,
        save_images=False,
    )
    print(f"Analyzed {len(results)} images from folder.")
