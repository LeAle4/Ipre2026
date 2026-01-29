"""
Resize all geoglif images using LCI method with georeferencing preserved.

This script processes all geoglif orthomosaic images from the data folders,
resizes them using the Lagrange-Chebyshev Interpolation method, and saves
them with updated geospatial metadata in the dataset/resized_geos folder.
Also copies and updates the corresponding metadata JSON files.
"""

from pathlib import Path
from lci import lci_georeferenced
import json
import sys
import numpy as np
from PIL import Image
import rasterio

# Scale factors for each area (based on analysis)
AREA_SCALES = {
    'unita': 0.886,
    'lluta': 0.218,
    'chugchug': 0.18
}

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = DATASET_DIR / "resized_geos"


def get_geos_directories():
    """Find all *_geos directories in data folder."""
    return sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith('_geos')])


def extract_area_name(dir_name):
    """Extract area name from directory name (e.g., 'unita_geos' -> 'unita')."""
    return dir_name.replace('_geos', '')


def update_metadata_json(metadata_path, output_path, scale_factor):
    """
    Copy and update metadata JSON file with new image dimensions.
    
    Parameters
    ----------
    metadata_path : Path
        Path to original metadata JSON file
    output_path : Path
        Path where updated metadata JSON will be saved
    scale_factor : float
        Scale factor used for resizing
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Update image_shape if present
    if 'image_shape' in metadata:
        old_width = metadata['image_shape']['width']
        old_height = metadata['image_shape']['height']
        
        metadata['image_shape']['width'] = int(round(old_width * scale_factor))
        metadata['image_shape']['height'] = int(round(old_height * scale_factor))
    
    # Update file size info if present
    if 'file_size_mb' in metadata:
        # Approximate new file size (scales roughly with pixel count)
        metadata['file_size_mb'] = metadata['file_size_mb'] * (scale_factor ** 2)
    
    # Add resize info
    metadata['resized'] = True
    metadata['scale_factor'] = scale_factor
    metadata['original_dimensions'] = {
        'width': old_width if 'image_shape' in metadata else None,
        'height': old_height if 'image_shape' in metadata else None
    }
    
    # Bounds remain the same (same geographic area)
    # CRS remains the same
    # Polygon coordinates remain the same (geographic coordinates)
    
    # Save updated metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def resize_all_geos():
    """Process all geoglif images and resize with georeferencing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    geos_dirs = get_geos_directories()
    
    if not geos_dirs:
        print("No *_geos directories found in data folder!")
        return
    
    print("=" * 70)
    print("RESIZING ALL GEOGLIF IMAGES WITH LCI + GEOREFERENCING")
    print("=" * 70)
    print(f"\nFound {len(geos_dirs)} area(s) to process")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    total_processed = 0
    total_skipped = 0
    
    for geos_dir in geos_dirs:
        area_name = extract_area_name(geos_dir.name)
        
        # Get scale factor for this area
        scale_factor = AREA_SCALES.get(area_name)
        if scale_factor is None:
            print(f"\n[{area_name.upper()}] No scale factor defined, skipping...")
            continue
        
        # Find all .tif files (orthomosaic images)
        tif_files = sorted(geos_dir.glob("*_ortho.tif"))
        
        if not tif_files:
            print(f"\n[{area_name.upper()}] No orthomosaic .tif files found, skipping...")
            continue
        
        print(f"\n[{area_name.upper()}] Processing {len(tif_files)} images (scale: {scale_factor})")
        print("-" * 60)
        
        # Create output subdirectory for this area
        area_output_dir = OUTPUT_DIR / f"{area_name}_geos"
        area_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, tif_path in enumerate(tif_files, 1):
            output_path = area_output_dir / tif_path.name
            png_path = area_output_dir / tif_path.stem.replace('_ortho', '_resized.png')
            
            # Find corresponding metadata JSON
            metadata_name = tif_path.stem.replace('_ortho', '_metadata.json')
            metadata_path = geos_dir / metadata_name
            output_metadata_path = area_output_dir / metadata_name
            
            # Skip if already processed
            if output_path.exists():
                print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - already exists, skipping")
                total_skipped += 1
                continue
            
            try:
                print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - resizing...", end=" ")
                
                # Read source image for PNG saving
                with rasterio.open(str(tif_path)) as src:
                    if src.count >= 3:
                        img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
                    else:
                        img = src.read(1)
                
                # Apply LCI resizing
                from lci import lci
                img_resized = lci(img, scale_factor)
                
                # Save as GeoTIFF with georeferencing
                lci_georeferenced(str(tif_path), str(output_path), scale_factor)
                
                # Save as PNG (visual record)
                Image.fromarray(img_resized).save(str(png_path))
                
                print("✓", end=" ")
                
                # Copy and update metadata if it exists
                if metadata_path.exists():
                    print("+ metadata...", end=" ")
                    update_metadata_json(metadata_path, output_metadata_path, scale_factor)
                    print("✓")
                else:
                    print("(no metadata)")
                
                total_processed += 1
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {total_processed}")
    print(f"Total images skipped: {total_skipped}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    resize_all_geos()