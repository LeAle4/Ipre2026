"""Augment patches using PyTorch transforms."""

from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "patches"
OUTPUT_DIR = BASE_DIR / "data" / "augmented_patches"

# PyTorch augmentation pipeline
augment = v2.Compose([
    v2.RandomRotation(degrees=15),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomCrop(size=224, pad_if_needed=True, padding_mode='reflect'),
])

def augment_patches():
    """Augment all patches with same folder structure."""
    assert INPUT_DIR.exists(), f"Input directory not found: {INPUT_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AUGMENTING PATCHES WITH PYTORCH TRANSFORMS")
    print("=" * 70)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    total_augmented = total_skipped = 0
    
    # Iterate through area folders
    for area_dir in sorted(INPUT_DIR.iterdir()):
        if not area_dir.is_dir():
            continue
        
        area_name = area_dir.name
        area_output_dir = OUTPUT_DIR / area_name
        area_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{area_name.upper()}] Processing geoglyphs...")
        
        # Iterate through geoglif folders
        for geo_dir in sorted(area_dir.iterdir()):
            if not geo_dir.is_dir():
                continue
            
            geo_name = geo_dir.name
            geo_output_dir = area_output_dir / geo_name
            geo_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy original image
            original = geo_dir / f"{geo_name}_original.png"
            if original.exists():
                Image.open(original).save(geo_output_dir / f"{geo_name}_original.png")
            
            # Process patches
            patch_files = sorted(geo_dir.glob("patch_*.png"))
            if not patch_files:
                print(f"  {geo_name}: No patches found")
                continue
            
            print(f"  {geo_name}: {len(patch_files)} patches", end="")
            augmented = 0
            
            for patch_path in patch_files:
                patch_name = patch_path.stem
                
                try:
                    img = Image.open(patch_path).convert('RGB')
                    
                    # Generate augmentations (apply multiple times for variety)
                    for aug_idx in range(1, 4):  # 3 augmented versions per patch
                        aug_img = augment(img)
                        aug_path = geo_output_dir / f"{patch_name}_aug{aug_idx}.png"
                        aug_img.save(aug_path)
                        augmented += 1
                
                except Exception as e:
                    total_skipped += 1
            
            print(f" -> {augmented} augmented")
            total_augmented += augmented
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total patches augmented: {total_augmented}")
    print(f"Total patches skipped: {total_skipped}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    augment_patches()
