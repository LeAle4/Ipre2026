"""
Test script for generating crops from geoglifs in test_geos directory.
Tests different window sizes and noise filling methods.
Organizes output by area, method, and window size.
"""

import cv2
import numpy as np
from pathlib import Path
from format import make_random_crops, fill_with_noise, TEST_DIR

# Configuration
WINDOW_SIZES = [122, 244, 488]
N_CROPS_PER_IMAGE = 10
NOISE_LEVELS = [0.0, 0.5, 1.0]  # 0=avg color, 0.5=blend, 1.0=full noise
NOISE_TYPES = ['gaussian', 'uniform']
OUTPUT_BASE = Path(__file__).resolve().parent / "crops_output"

def extract_area_from_filename(filename):
    """Extract area name (unita, lluta, chug) from filename."""
    name = filename.lower()
    if name.startswith('unita_'):
        return 'unita'
    elif name.startswith('lluta_'):
        return 'lluta'
    elif name.startswith('chug_'):
        return 'chug'
    return 'unknown'

def process_geoglif(image_path, area, output_dir):
    """Process a single geoglif image with multiple crop configurations."""
    print(f"\nProcessing: {image_path.name}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Error: Could not load {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"  Image size: {w}x{h}")
    
    geoglif_id = image_path.stem.replace('_ortho', '')
    
    # Test different window sizes
    for window_size in WINDOW_SIZES:
        print(f"  Testing window size: {window_size}px")
        
        # Skip if window is larger than image
        if window_size > min(h, w):
            print(f"    Skipped (image too small)")
            continue
        
        # Generate random crops
        crops = make_random_crops(image, window_size, N_CROPS_PER_IMAGE, seed=42)
        
        # Test different noise filling methods
        for noise_type in NOISE_TYPES:
            for noise_level in NOISE_LEVELS:
                method_name = f"{noise_type}_noise{int(noise_level*100)}"
                
                # Create output directory structure
                crop_output_dir = output_dir / area / f"window_{window_size}" / method_name
                crop_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process and save each crop
                for idx, (crop, mask) in enumerate(crops):
                    # Fill out-of-bounds regions with noise
                    filled_crop = fill_with_noise(crop, mask, noise_level, noise_type, seed=42+idx)
                    
                    # Save the crop
                    output_filename = f"{geoglif_id}_crop_{idx:02d}.jpg"
                    output_path = crop_output_dir / output_filename
                    cv2.imwrite(str(output_path), filled_crop)
        
        print(f"    Generated {N_CROPS_PER_IMAGE} crops with {len(NOISE_TYPES) * len(NOISE_LEVELS)} methods")

def main():
    """Main test execution."""
    print("="*80)
    print("CROPS GENERATION TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Window sizes: {WINDOW_SIZES}")
    print(f"  Crops per image: {N_CROPS_PER_IMAGE}")
    print(f"  Noise types: {NOISE_TYPES}")
    print(f"  Noise levels: {NOISE_LEVELS}")
    print(f"  Input directory: {TEST_DIR}")
    print(f"  Output directory: {OUTPUT_BASE}")
    
    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Find all ortho JPEG images in test_geos
    ortho_images = sorted(TEST_DIR.glob("*_ortho.jpg"))
    
    if not ortho_images:
        print("\nError: No ortho images found in test_geos directory!")
        return
    
    print(f"\nFound {len(ortho_images)} images to process")
    
    # Group images by area
    images_by_area = {'unita': [], 'lluta': [], 'chug': []}
    for img_path in ortho_images:
        area = extract_area_from_filename(img_path.name)
        if area != 'unknown':
            images_by_area[area].append(img_path)
    
    print(f"\nImages by area:")
    for area, imgs in images_by_area.items():
        print(f"  {area.upper()}: {len(imgs)} images")
    
    # Process each image
    total_images = len(ortho_images)
    for idx, image_path in enumerate(ortho_images, 1):
        area = extract_area_from_filename(image_path.name)
        print(f"\n[{idx}/{total_images}] Processing {area.upper()} image")
        process_geoglif(image_path, area, OUTPUT_BASE)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_crops = 0
    for area in ['unita', 'lluta', 'chug']:
        area_dir = OUTPUT_BASE / area
        if area_dir.exists():
            crop_count = len(list(area_dir.rglob("*.jpg")))
            print(f"{area.upper()}: {crop_count} crops generated")
            total_crops += crop_count
    
    print(f"\nTotal crops generated: {total_crops}")
    print(f"Output directory: {OUTPUT_BASE}")
    print("\nDirectory structure:")
    print("  crops_output/")
    print("    ├── unita/")
    print("    │   ├── window_128/")
    print("    │   │   ├── gaussian_noise0/")
    print("    │   │   ├── gaussian_noise50/")
    print("    │   │   └── ...")
    print("    │   ├── window_256/")
    print("    │   └── window_512/")
    print("    ├── lluta/")
    print("    └── chug/")
    print("\nTest completed!")

if __name__ == "__main__":
    main()
