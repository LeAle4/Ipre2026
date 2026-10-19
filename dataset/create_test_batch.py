import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AREAS = ["unita", "chugchug", "lluta"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def collect_images_from_dir(dir_path: Path) -> list[Path]:
    """Recursively collect all image files from a directory."""
    if not dir_path.exists():
        print(f"Warning: Directory {dir_path} does not exist.")
        return []
    
    images = [
        f for f in dir_path.rglob("*") 
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images

def organize_dataset(output_dir: Path = Path("dataset")):
    """Collect all positive and negative crops from all areas and structure as:
    
    data/
    ├── Area 1/
    │   ├── positives/
    │   └── negatives/
    ├── Area 2/
    │   ├── positives/
    │   └── negatives/
    ├── Area 3/
    │   ├── positives/
    │   └── negatives/
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for area in AREAS:
        crops_dir = DATA_DIR / f"{area}_crops"
        negatives_dir = DATA_DIR / f"{area}_negatives"
        
        pos_target = output_dir / area / "positives"
        neg_target = output_dir / area / "negatives"
        
        pos_target.mkdir(parents=True, exist_ok=True)
        neg_target.mkdir(parents=True, exist_ok=True)
        
        pos_images = collect_images_from_dir(crops_dir)
        neg_images = collect_images_from_dir(negatives_dir)
        
        print(f"Area '{area}': Found {len(pos_images)} positive images, {len(neg_images)} negative images.")
        
        # Copy positives
        print(f"  Copying positive images to {pos_target}...")
        for img in pos_images:
            shutil.copy(img, pos_target / img.name)
            
        # Copy negatives
        print(f"  Copying negative images to {neg_target}...")
        for img in neg_images:
            shutil.copy(img, neg_target / img.name)

if __name__ == "__main__":
    print("Organizing full dataset by area into positives and negatives...")
    organize_dataset(DATA_DIR)
    print("Dataset organization complete!")