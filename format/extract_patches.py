from skimage.util import view_as_windows
from pathlib import Path
import numpy as np
import json
from PIL import Image

from utils import (
    load_tif_image,
    load_polygon_from_metadata,
    load_metadata,
    convert_polygon_geo_to_pixel,
    calculate_patch_overlap
)


WINDOW_SIZE = 244
STRIDE = 122
CHANNELS = 3
THRESHOLD = 0.9  # 90% overlap threshold

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "resized_geos"
OUTPUT_DIR = BASE_DIR / "data" / "patches"

def extract_patches(img):
    """
    Extract patches from an image using sliding window.
    Returns None if image is too small for patch extraction.
    """
    h, w = img.shape[:2]
    if h < WINDOW_SIZE or w < WINDOW_SIZE:
        return None
    
    patches = view_as_windows(img, (WINDOW_SIZE, WINDOW_SIZE, CHANNELS), step=STRIDE)
    return patches


def save_patches(patches, polygon_pixel_coords, output_dir, base_name, threshold):
    """
    Save patches as PNG images, filtering by polygon overlap threshold.
    
    Parameters
    ----------
    patches : np.ndarray
        Array of shape (rows, cols, 1, H, W, C) from view_as_windows
    polygon_pixel_coords : list
        List of (x, y) tuples defining polygon in pixel coordinates
    output_dir : Path
        Directory to save patches (specific to this geoglif)
    base_name : str
        Base name for patch files (e.g., 'geoglif_0000')
    threshold : float
        Overlap threshold (0.0 to 1.0) as fraction of patch area
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = output_dir / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    
    rows, cols = patches.shape[0], patches.shape[1]
    patch_count = 0
    skipped_count = 0
    
    for i in range(rows):
        for j in range(cols):
            # Define patch window in pixel coordinates
            patch_x_min = j * STRIDE
            patch_y_min = i * STRIDE
            patch_x_max = patch_x_min + WINDOW_SIZE
            patch_y_max = patch_y_min + WINDOW_SIZE
            
            patch = patches[i, j, 0]  # Extract (H, W, C) from (1, H, W, C)
            patch_name = f"patch_{i:03d}_{j:03d}.png"
            
            # Check if patch overlaps polygon enough
            if calculate_patch_overlap(
                (patch_x_min, patch_y_min, patch_x_max, patch_y_max),
                polygon_pixel_coords,
                threshold
            ):
                # Save valid patch
                patch_path = output_dir / patch_name
                Image.fromarray(patch.astype(np.uint8)).save(patch_path)
                patch_count += 1
            else:
                # Save rejected patch
                rejected_path = rejected_dir / patch_name
                Image.fromarray(patch.astype(np.uint8)).save(rejected_path)
                skipped_count += 1
    
    return patch_count, skipped_count


def save_original_image(img, output_dir, base_name):
    """
    Save original image when it's too small for patch extraction.
    
    Parameters
    ----------
    img : np.ndarray
        Original image array
    output_dir : Path
        Directory to save the image
    base_name : str
        Base name for the file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = output_dir / f"{base_name}_original.png"
    Image.fromarray(img.astype(np.uint8)).save(image_path)
    
    return 1  # Return 1 to indicate one image saved


def process_all_geos():
    """Process all resized geoglif images and extract patches with polygon filtering."""
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please run resize_dataset.py first!")
        return
    
    # Find all *_geos directories
    geos_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith('_geos')])
    
    if not geos_dirs:
        print("No *_geos directories found in resized_geos folder!")
        return
    
    print("=" * 70)
    print("EXTRACTING PATCHES FROM RESIZED GEOGLIF IMAGES")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(geos_dirs)} area(s) to process\n")
    print(f"Patch settings: {WINDOW_SIZE}x{WINDOW_SIZE}, stride={STRIDE}")
    print(f"Polygon overlap threshold: {THRESHOLD*100:.0f}%\n")
    
    total_images = 0
    total_patches = 0
    total_originals = 0
    total_skipped_patches = 0
    
    for geos_dir in geos_dirs:
        area_name = geos_dir.name.replace('_geos', '')
        
        # Find all .tif files
        tif_files = sorted(geos_dir.glob("*_ortho.tif"))
        
        if not tif_files:
            print(f"[{area_name.upper()}] No .tif files found, skipping...")
            continue
        
        print(f"[{area_name.upper()}] Processing {len(tif_files)} images")
        print("-" * 60)
        
        # Create output directory for this area
        area_output_dir = OUTPUT_DIR / area_name
        
        for idx, tif_path in enumerate(tif_files, 1):
            base_name = tif_path.stem.replace('_ortho', '')
            
            # Create separate folder for this geoglif's patches
            geo_output_dir = area_output_dir / base_name
            
            # Find corresponding metadata JSON
            metadata_name = tif_path.stem.replace('_ortho', '_metadata.json')
            metadata_path = geos_dir / metadata_name
            
            try:
                print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - loading...", end=" ")
                
                # Load image
                img = load_tif_image(tif_path)
                h, w = img.shape[:2]
                
                print(f"({h}x{w}) ", end="")
                
                # Load polygon from metadata
                polygon_geo_coords = None
                if metadata_path.exists():
                    try:
                        metadata = load_metadata(metadata_path)
                        polygon_geo_coords = load_polygon_from_metadata(metadata_path)
                        
                        # Convert geographic coords to pixel coords
                        if polygon_geo_coords and 'bounds' in metadata and 'image_shape' in metadata:
                            polygon_pixel_coords = convert_polygon_geo_to_pixel(
                                polygon_geo_coords,
                                metadata['bounds'],
                                metadata['image_shape']
                            )
                        else:
                            polygon_pixel_coords = None
                    except Exception as e:
                        print(f"(metadata load error: {e}) ", end="")
                        polygon_pixel_coords = None
                else:
                    polygon_pixel_coords = None
                
                # Always save the original image first
                save_original_image(img, geo_output_dir, base_name)
                
                # Extract patches
                patches = extract_patches(img)
                
                if patches is None:
                    # Image too small, only original saved
                    print("too small, saved original only...", end=" ")
                    print(f"✓")
                    total_originals += 1
                else:
                    # Extract and save patches
                    print("extracting...", end=" ")
                    
                    if polygon_pixel_coords is None:
                        print("(no polygon, saving all patches) ", end="")
                        # Save all patches without filtering
                        rows, cols = patches.shape[0], patches.shape[1]
                        patch_count = 0
                        for i in range(rows):
                            for j in range(cols):
                                patch = patches[i, j, 0]
                                patch_name = f"patch_{i:03d}_{j:03d}.png"
                                patch_path = geo_output_dir / patch_name
                                patch_path.parent.mkdir(parents=True, exist_ok=True)
                                Image.fromarray(patch.astype(np.uint8)).save(patch_path)
                                patch_count += 1
                        skipped = 0
                    else:
                        # Save patches with polygon overlap filtering
                        patch_count, skipped = save_patches(
                            patches,
                            polygon_pixel_coords,
                            geo_output_dir,
                            base_name,
                            THRESHOLD
                        )
                        total_skipped_patches += skipped
                    
                    print(f"✓ ({patch_count} patches + original", end="")
                    if polygon_pixel_coords is not None:
                        print(f", {skipped} skipped)", end="")
                    print(")")
                    
                    total_patches += patch_count
                
                total_images += 1
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {total_images}")
    print(f"Total patches extracted: {total_patches}")
    print(f"Total patches skipped: {total_skipped_patches}")
    print(f"Total originals saved: {total_originals}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    process_all_geos()
