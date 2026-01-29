"""
Compare original, LCI-resized, and bicubic-resized geoglif images.

This script creates a 2x3 comparison visualization:
- Row 1: Full images (Original | LCI | Bicubic)
- Row 2: Zoomed regions (Original | LCI | Bicubic)

Usage:
    python comparison.py <original_tif_path> [--zoom-size 200] [--output output.png]

Example:
    python comparison.py ../data/unita_geos/geoglif_0000_ortho.tif --output comparison.png
"""

import argparse
import numpy as np
import rasterio
from scipy.ndimage import zoom as scipy_zoom
from skimage.transform import resize as skimage_resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def load_image_from_tif(tif_path):
    """Load RGB image from GeoTIFF file and extract georeferencing info."""
    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            # Read RGB bands (1, 2, 3)
            img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
        else:
            # Grayscale
            img = src.read(1)
            img = np.stack([img, img, img], axis=-1)  # Convert to RGB
        
        # Extract georeferencing info
        bounds = src.bounds
        transform = src.transform
        
        # Calculate pixel size in ground units
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        
        # Calculate geographic area (width and height in ground units)
        geo_width = bounds.right - bounds.left
        geo_height = bounds.top - bounds.bottom
        
        # Calculate area in square meters
        area_m2 = geo_width * geo_height
        
        geo_info = {
            'pixel_width': pixel_width,
            'pixel_height': pixel_height,
            'geo_width': geo_width,
            'geo_height': geo_height,
            'area_m2': area_m2
        }
    
    return img.astype(np.uint8), geo_info


def find_resized_image(original_path):
    """
    Find the corresponding resized LCI image.
    
    Looks in the resized_geos folder for the same filename.
    The area name is determined from the parent directory of the original image.
    """
    original_path = Path(original_path)
    
    # Get area name from parent directory (e.g., 'unita_geos' -> 'unita')
    parent_dir = original_path.parent.name  # e.g., 'unita_geos'
    
    # Construct path to resized image
    dataset_dir = Path(__file__).resolve().parent
    resized_dir = dataset_dir / "resized_geos" / parent_dir
    resized_path = resized_dir / original_path.name
    
    if resized_path.exists():
        return resized_path
    else:
        raise FileNotFoundError(f"Resized image not found: {resized_path}")


def resize_with_bicubic(image, target_height, target_width):
    """
    Resize image using bicubic interpolation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C) or (H, W)
    target_height : int
        Target height
    target_width : int
        Target width
    
    Returns
    -------
    np.ndarray
        Resized image
    """
    if image.ndim == 3:
        # RGB image - resize each channel
        resized = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        for c in range(image.shape[2]):
            resized[:, :, c] = skimage_resize(
                image[:, :, c],
                (target_height, target_width),
                order=3,  # Bicubic
                preserve_range=True
            ).astype(image.dtype)
        return resized
    else:
        # Grayscale
        return skimage_resize(
            image,
            (target_height, target_width),
            order=3,  # Bicubic
            preserve_range=True
        ).astype(image.dtype)


def extract_zoom_region(image, center_ratio=0.5, zoom_size=200):
    """
    Extract a zoomed region from the center of the image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) or (H, W, C)
    center_ratio : float
        Position of zoom center as ratio (0.5 = center)
    zoom_size : int
        Size of the zoom region (zoom_size x zoom_size)
    
    Returns
    -------
    tuple
        (zoomed_region, (y_start, x_start)) - region and its position
    """
    h, w = image.shape[:2]
    
    # Calculate center
    center_y = int(h * center_ratio)
    center_x = int(w * center_ratio)
    
    # Calculate region bounds
    y_start = max(0, center_y - zoom_size // 2)
    y_end = min(h, y_start + zoom_size)
    x_start = max(0, center_x - zoom_size // 2)
    x_end = min(w, x_start + zoom_size)
    
    # Adjust if region is at edge
    if y_end - y_start < zoom_size:
        y_start = max(0, y_end - zoom_size)
    if x_end - x_start < zoom_size:
        x_start = max(0, x_end - zoom_size)
    
    y_end = y_start + zoom_size
    x_end = x_start + zoom_size
    
    zoomed = image[y_start:y_end, x_start:x_end]
    return zoomed, (y_start, x_start, y_end, x_end)


def create_comparison_figure(original_img, lci_img, bicubic_img, geo_info_orig, zoom_size=200, output_path=None):
    """
    Create a 2x3 comparison figure.
    
    Parameters
    ----------
    original_img : np.ndarray
        Original image
    lci_img : np.ndarray
        LCI-resized image
    bicubic_img : np.ndarray
        Bicubic-resized image
    geo_info_orig : dict
        Geographic information from original image
    zoom_size : int
        Size of zoom region
    output_path : str or Path, optional
        Path to save the figure. If None, displays the figure.
    """
    # Extract zoom regions (from original image to identify area)
    zoom_orig, zoom_coords = extract_zoom_region(original_img, zoom_size=zoom_size)
    y_start, x_start, y_end, x_end = zoom_coords
    
    # Calculate scale factors to get corresponding regions in resized images
    scale_y = lci_img.shape[0] / original_img.shape[0]
    scale_x = lci_img.shape[1] / original_img.shape[1]
    
    # Get corresponding zoom regions from resized images
    y_start_lci = int(y_start * scale_y)
    x_start_lci = int(x_start * scale_x)
    y_end_lci = int(y_end * scale_y)
    x_end_lci = int(x_end * scale_x)
    zoom_lci = lci_img[y_start_lci:y_end_lci, x_start_lci:x_end_lci]
    
    # Bicubic has same dimensions as LCI
    y_start_bic = int(y_start * scale_y)
    x_start_bic = int(x_start * scale_x)
    y_end_bic = int(y_end * scale_y)
    x_end_bic = int(x_end * scale_x)
    zoom_bicubic = bicubic_img[y_start_bic:y_end_bic, x_start_bic:x_end_bic]
    
    # Format area information
    area_km2 = geo_info_orig['area_m2'] / 1e6
    geo_width_m = geo_info_orig['geo_width']
    geo_height_m = geo_info_orig['geo_height']
    
    orig_title = (f"Original\n{original_img.shape[0]}×{original_img.shape[1]} px\n"
                  f"{geo_width_m:.1f}×{geo_height_m:.1f} m\n"
                  f"Area: {area_km2:.3f} km²")
    
    # Calculate area for resized images (geographic area stays the same)
    lci_title = (f"LCI Resized\n{lci_img.shape[0]}×{lci_img.shape[1]} px\n"
                 f"{geo_width_m:.1f}×{geo_height_m:.1f} m\n"
                 f"Area: {area_km2:.3f} km²")
    
    bicubic_title = (f"Bicubic Resized\n{bicubic_img.shape[0]}×{bicubic_img.shape[1]} px\n"
                     f"{geo_width_m:.1f}×{geo_height_m:.1f} m\n"
                     f"Area: {area_km2:.3f} km²")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    fig.suptitle('Geoglif Image Comparison: Original vs LCI vs Bicubic', fontsize=16, fontweight='bold')
    
    # Top row - full images
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(orig_title, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(lci_img)
    axes[0, 1].set_title(lci_title, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(bicubic_img)
    axes[0, 2].set_title(bicubic_title, fontweight='bold')
    axes[0, 2].axis('off')
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'Original\n{original_img.shape[0]}x{original_img.shape[1]}', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(lci_img)
    axes[0, 1].set_title(f'LCI Resized\n{lci_img.shape[0]}x{lci_img.shape[1]}', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(bicubic_img)
    axes[0, 2].set_title(f'Bicubic Resized\n{bicubic_img.shape[0]}x{bicubic_img.shape[1]}', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Draw zoom rectangle on full images
    rect_orig = patches.Rectangle(
        (x_start, y_start), x_end - x_start, y_end - y_start,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    axes[0, 0].add_patch(rect_orig)
    
    rect_lci = patches.Rectangle(
        (x_start_lci, y_start_lci), x_end_lci - x_start_lci, y_end_lci - y_start_lci,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    axes[0, 1].add_patch(rect_lci)
    
    rect_bic = patches.Rectangle(
        (x_start_bic, y_start_bic), x_end_bic - x_start_bic, y_end_bic - y_start_bic,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    axes[0, 2].add_patch(rect_bic)
    
    # Bottom row - zoomed regions
    axes[1, 0].imshow(zoom_orig)
    axes[1, 0].set_title('Zoomed Original', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(zoom_lci)
    axes[1, 1].set_title('Zoomed LCI', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(zoom_bicubic)
    axes[1, 2].set_title('Zoomed Bicubic', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout(pad=3.0, h_pad=2.5)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {output_path}")
    else:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare original, LCI-resized, and bicubic-resized geoglif images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comparison.py ../data/unita_geos/geoglif_0000_ortho.tif
  python comparison.py ../data/unita_geos/geoglif_0000_ortho.tif --output comparison.png
  python comparison.py ../data/lluta_geos/geoglif_0005_ortho.tif --zoom-size 300
        """
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the original geoglif orthomosaic .tif file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the comparison figure. If not specified, displays the figure."
    )
    
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=200,
        help="Size of the zoom region in pixels (default: 200)"
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    print("=" * 70)
    print("GEOGLIF IMAGE COMPARISON")
    print("=" * 70)
    print(f"\nLoading original image: {image_path}")
    
    # Load images
    original_img, geo_info = load_image_from_tif(image_path)
    print(f"  Shape: {original_img.shape}")
    print(f"  Geographic area: {geo_info['area_m2']/1e6:.3f} km²")
    print(f"  Dimensions: {geo_info['geo_width']:.1f} × {geo_info['geo_height']:.1f} m")
    
    # Find and load LCI-resized image
    print("\nFinding LCI-resized image...")
    lci_path = find_resized_image(image_path)
    print(f"  Found: {lci_path}")
    lci_img, _ = load_image_from_tif(lci_path)
    print(f"  Shape: {lci_img.shape}")
    
    # Create bicubic-resized image (same size as LCI)
    print("\nGenerating bicubic-resized image...")
    target_height, target_width = lci_img.shape[:2]
    bicubic_img = resize_with_bicubic(original_img, target_height, target_width)
    print(f"  Shape: {bicubic_img.shape}")
    
    # Create comparison figure
    print("\nCreating comparison figure...")
    create_comparison_figure(
        original_img,
        lci_img,
        bicubic_img,
        geo_info,
        zoom_size=args.zoom_size,
        output_path=args.output
    )
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
