"""
Geoglyph dataset creation and image cropping orchestration.

This script orchestrates the generation of cropped image samples from geoglyph
imagery using various cropping strategies (random, fixed grid, polygon-based),
and saving the results to a user-specified output directory.

The code is designed to be easily extensible: simply add new cropping strategies
to the CROP_STRATEGY_REGISTRY and they will automatically be available.

Usage:
    python data_create.py --area lluta --strategy random --n-crops 100 --output ./data
    python data_create.py --area chugchug --strategy fixed --n-crops 50 --window-size 256 --output ./data
    python data_create.py --area unita --strategy polygon --n-crops 30 --output ./data
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import cv2

# Import our custom modules
from format import (
    crop_image,
    make_random_crops,
    make_fixed_crops,
    make_polygon_thresholds_crops,
    fill_with_noise,
    extract_area_from_filename
)
from handle import (
    BASE_DIR,
    DATA_DIR,
    LLUTA_GEOS_DIR,
    UNITA_GEOS_DIR,
    CHUG_GEOS_DIR
)

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Directory naming patterns
DIR_PATTERNS = {
    'geos': '{area}_geos',
    'polygons': '{area}_polygons',
    'raw': '{area}_raw',
}

# File naming patterns
FILE_PATTERNS = {
    'crop_id': '{area}_{image_id:04d}_{crop_idx:03d}',
    'metadata_summary': 'summary.json',
    'crop_results_csv': '{area}_{strategy}_crops.csv',
    'batch_summary_csv': '{area}_{strategy}_batch_summary.csv',
    'crop_file': '{crop_id}.png',
}

# Default parameters
DEFAULTS = {
    'window_size': 256,
    'noise_level': 0.1,
    'noise_type': 'gaussian',
    'threshold': 0.1,
    'stride': None,
    'seed': None,
    'organize_by_stride': False,  # Include stride in output directory structure
}

# Batch mode defaults
BATCH_DEFAULTS = {
    'window_sizes': [256],
    'noise_types': ['gaussian', 'uniform'],
    'noise_levels': [0.0, 0.5, 1.0],
    'strides': [None],  # Can test multiple stride values
}

# Output formatting
OUTPUT_FORMAT = {
    'separator': '='*70,
    'separator_short': '-'*60,
}

# Available areas
AVAILABLE_AREAS = ['lluta', 'chugchug', 'unita', 'granllama', 'salvador']


# ============================================================================
# Utility Functions
# ============================================================================

def stride_label(stride) -> str:
    """
    Generate a short label for stride directory naming.
    
    Args:
        stride: Stride value (None, int, or tuple)
    
    Returns:
        String label for directory naming
    """
    if stride is None:
        return "auto"
    if isinstance(stride, (list, tuple, np.ndarray)):
        if len(stride) == 2:
            return f"{stride[0]}x{stride[1]}"
        return "invalid"
    return str(stride)


# ============================================================================
# CROP STRATEGY REGISTRY: Central configuration for all available strategies
# ============================================================================
# Add new cropping strategies here without modifying other parts of the code
CROP_STRATEGY_REGISTRY = {
    'random': {
        'name': "Random Crops",
        'description': "Randomly sampled crops from image",
        'params': ['n_crops', 'window_size', 'seed'],
        'defaults': {'seed': None},
        'crop_func': lambda img, **kwargs: make_random_crops(
            img,
            window_size=kwargs['window_size'],
            n_crops=kwargs['n_crops'],
            seed=kwargs.get('seed')
        ),
    },
    'fixed': {
        'name': "Fixed Grid Crops",
        'description': "Deterministic crops in fixed grid pattern",
        'params': ['n_crops', 'window_size', 'stride'],
        'defaults': {'stride': None},
        'crop_func': lambda img, **kwargs: make_fixed_crops(
            img,
            window_size=kwargs['window_size'],
            n_crops=kwargs['n_crops'],
            stride=kwargs.get('stride')
        ),
    },
    'polygon': {
        'name': "Polygon-based Crops",
        'description': "Crops distributed within polygon boundaries with overlap threshold",
        'params': ['n_crops', 'window_size', 'stride', 'threshold'],
        'defaults': {'stride': None, 'threshold': 0.1},
        'crop_func': None,  # Special handling in GeoglyphDataCreator
    },
}


class GeoglyphDataCreator:
    """Orchestrates dataset creation by cropping geoglyph images."""
    
    def __init__(
        self,
        area_name: str,
        strategy: str,
        window_size: int = 256,
        output_dir: str = "./data",
        organize_by_stride: bool = False
    ):
        """
        Initialize the data creator.
        
        Args:
            area_name: Name of the area (e.g., 'lluta', 'chugchug', 'unita', 'granllama', 'salvador').
            strategy: Cropping strategy (key in CROP_STRATEGY_REGISTRY).
            window_size: Size of crop windows.
            output_dir: Directory to save cropped images.
            organize_by_stride: Include stride in output directory structure.
        """
        self.area_name = area_name
        self.strategy = strategy
        self.window_size = window_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.organize_by_stride = organize_by_stride
        self.current_stride = None  # Track current stride for directory naming
        
        # Setup data paths using patterns
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.area_subdir = DIR_PATTERNS['geos'].format(area=area_name)
        self.geos_dir = self.data_dir / self.area_subdir
        self.metadata_summary_path = (self.data_dir / DIR_PATTERNS['polygons'].format(area=area_name) 
                                      / FILE_PATTERNS['metadata_summary'])
        
        # Validate
        if not self.metadata_summary_path.exists():
            raise FileNotFoundError(f"Metadata summary not found: {self.metadata_summary_path}")
        if strategy not in CROP_STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.results = []
        self.crop_count = 0
    
    def load_metadata_summary(self) -> dict:
        """Load the area metadata summary JSON."""
        with open(self.metadata_summary_path, 'r') as f:
            return json.load(f)
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image and convert to uint8 numpy array (grayscale or color)."""
        img = Image.open(image_path)
        return np.array(img)
    
    def parse_images(self, summary: dict):
        """Yield image metadata from summary."""
        for entry in summary.get('metadata', []):
            yield {
                "id": entry.get('geoglyph_id', entry.get('polygon_index', 'unknown')),
                "class": entry.get('class', 'unknown'),
                "img_name": entry.get('filename'),
                "polygon_coords": entry.get('polygon_coords'),
            }
    
    def save_crop(self, crop_array: np.ndarray, crop_id: str, metadata: Optional[dict] = None) -> str:
        """Save a single crop image to disk."""
        # Create subdirectories
        crop_filename = FILE_PATTERNS['crop_file'].format(crop_id=crop_id)
        
        # Build path with optional stride directory
        if self.organize_by_stride and self.current_stride is not None:
            stride_dir = f"stride_{stride_label(self.current_stride)}"
            crop_path = self.output_dir / self.strategy / self.area_name / stride_dir / crop_filename
        else:
            crop_path = self.output_dir / self.strategy / self.area_name / crop_filename
        
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        Image.fromarray(crop_array.astype(np.uint8)).save(crop_path)
        return str(crop_path.relative_to(self.output_dir))
    
    def generate_crops(
        self,
        image_array: np.ndarray,
        n_crops: int,
        stride: Optional[int] = None,
        threshold: float = 0.1,
        polygon_coords: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate crops using the configured strategy.
        
        Args:
            image_array: Input image as numpy array.
            n_crops: Number of crops to generate.
            stride: Step size for fixed/polygon strategies (optional).
            threshold: Overlap threshold for polygon strategy.
            polygon_coords: Polygon vertices for polygon strategy.
            seed: Random seed for reproducible results.
        
        Returns:
            List of (crop, mask) tuples.
        """
        crop_params = {
            'random': {'image': image_array, 'window_size': self.window_size, 'n_crops': n_crops, 'seed': seed},
            'fixed': {'image': image_array, 'window_size': self.window_size, 'n_crops': n_crops, 'stride': stride},
            'polygon': {'image': image_array, 'polygon_vertices': polygon_coords, 'window_size': self.window_size, 
                       'n_crops': n_crops, 'stride': stride, 'threshold': threshold}
        }
        
        crop_funcs = {
            'random': make_random_crops,
            'fixed': make_fixed_crops,
            'polygon': make_polygon_thresholds_crops
        }
        
        return crop_funcs[self.strategy](**crop_params[self.strategy])
    
    def run_data_creation(
        self,
        n_crops: int,
        stride: Optional[int] = None,
        threshold: Optional[float] = None,
        seed: Optional[int] = None,
        fill_noise: bool = False,
        noise_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run data creation on all images in the area.
        
        Args:
            n_crops: Number of crops per image.
            stride: Step size for fixed/polygon strategies.
            threshold: Overlap threshold for polygon strategy.
            seed: Random seed for reproducibility.
            fill_noise: Whether to fill out-of-bounds regions with noise.
            noise_type: Type of noise ('gaussian', 'uniform', 'perlin').
        
        Returns:
            DataFrame with crop metadata.
        """
        # Apply defaults
        threshold = threshold if threshold is not None else DEFAULTS['threshold']
        noise_type = noise_type if noise_type is not None else DEFAULTS['noise_type']
        
        # Track stride for directory organization
        self.current_stride = stride
        
        print(f"Processing dataset: {self.area_subdir}")
        print(f"Strategy: {self.strategy} ({CROP_STRATEGY_REGISTRY[self.strategy]['name']})")
        print(f"Window size: {self.window_size}x{self.window_size}")
        print(f"Crops per image: {n_crops}")
        print(f"Fill noise: {fill_noise} ({noise_type})" if fill_noise else "Fill noise: False")
        print(OUTPUT_FORMAT['separator_short'])
        
        metadata_summary = self.load_metadata_summary()
        results = []
        
        for image_meta in self.parse_images(metadata_summary):
            image_id = image_meta.get('id', 'unknown')
            image_filename = image_meta.get('img_name')
            image_class = image_meta.get('class', 'unknown')
            polygon_coords = image_meta.get('polygon_coords')
            
            if not image_filename:
                print(f"Warning: No filename for image {image_id}, skipping.")
                continue
            
            img_path = self.geos_dir / image_filename
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Load image
            img_array = self.load_image(img_path)
            if img_array is None:
                print(f"Warning: Failed to load image {image_id}, skipping.")
                continue
            
            # Convert to grayscale if needed (for consistent processing)
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            # Generate crops
            crops = self.generate_crops(
                img_gray,
                n_crops=n_crops,
                stride=stride,
                threshold=threshold,
                polygon_coords=polygon_coords,
                seed=seed
            )
            
            if not crops:
                print(f"Warning: No crops generated for image {image_id}")
                continue
            
            # Process and save crops
            for crop_idx, (crop, mask) in enumerate(crops):
                if crop is None or mask is None:
                    continue
                
                # Apply noise filling if requested
                if fill_noise:
                    crop = fill_with_noise(crop, mask, noise_level=DEFAULTS['noise_level'], noise_type=noise_type)
                
                # Save crop
                crop_id = FILE_PATTERNS['crop_id'].format(
                    area=self.area_name, 
                    image_id=image_id, 
                    crop_idx=crop_idx
                )
                crop_path = self.save_crop(crop, crop_id, image_meta)
                
                # Record metadata
                result = {
                    "crop_id": crop_id,
                    "source_image_id": image_id,
                    "source_class": image_class,
                    "crop_index": crop_idx,
                    "crop_path": crop_path,
                    "window_size": self.window_size,
                    "mask_coverage": float(np.sum(mask) / mask.size)
                }
                results.append(result)
                self.crop_count += 1
            
            print(f"Image {image_id}: Generated {len(crops)} crops")
        
        df_results = pd.DataFrame(results)
        return df_results
    
    def save_results(self, df_results: pd.DataFrame) -> None:
        """Save crop metadata to CSV."""
        csv_filename = FILE_PATTERNS['crop_results_csv'].format(area=self.area_name, strategy=self.strategy)
        csv_path = self.output_dir / csv_filename
        df_results.to_csv(csv_path, index=False)
        print(f"\nSaved crop metadata to {csv_path}")
    
    def print_summary(self, df_results: pd.DataFrame) -> None:
        """Print summary statistics."""
        sep = OUTPUT_FORMAT['separator']
        print(f"\n{sep}\nData Creation Summary\n{sep}")
        print(f"Area: {self.area_name}")
        print(f"Strategy: {self.strategy}")
        print(f"Total crops created: {self.crop_count}")
        print(f"Window size: {self.window_size}x{self.window_size}")
        
        if len(df_results) > 0:
            print(f"\nCrop statistics:")
            print(f"  Total records: {len(df_results)}")
            print(f"  Classes represented: {df_results['source_class'].nunique()}")
            print(f"  Avg mask coverage: {df_results['mask_coverage'].mean():.2%}")
            print(f"  Min mask coverage: {df_results['mask_coverage'].min():.2%}")
            print(f"  Max mask coverage: {df_results['mask_coverage'].max():.2%}")
            print(f"\nClass distribution:")
            print(df_results['source_class'].value_counts())
        
        print(f"\nResults saved to: {self.output_dir}\n{sep}\n")
    
    def run_full_pipeline(self, n_crops: int, stride: Optional[int] = None, threshold: Optional[float] = None, 
                          seed: Optional[int] = None, fill_noise: bool = False, noise_type: Optional[str] = None) -> None:
        """Execute full data creation pipeline."""
        df_results = self.run_data_creation(n_crops, stride, threshold, seed, fill_noise, noise_type)
        self.save_results(df_results)
        self.print_summary(df_results)
    
    def run_batch_pipeline(
        self,
        n_crops: int,
        window_sizes: Optional[List[int]] = None,
        noise_types: Optional[List[str]] = None,
        noise_levels: Optional[List[float]] = None,
        strides: Optional[List] = None,
        threshold: Optional[float] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Execute batch data creation with multiple configurations.
        
        This method tests multiple window sizes, stride values, and noise configurations,
        organizing output by configuration. Useful for testing and comparison.
        
        Args:
            n_crops: Number of crops per image per configuration.
            window_sizes: List of window sizes to test (default: from BATCH_DEFAULTS).
            noise_types: List of noise types to test (default: from BATCH_DEFAULTS).
            noise_levels: List of noise levels to test (default: from BATCH_DEFAULTS).
            strides: List of stride values to test (default: from BATCH_DEFAULTS).
            threshold: Overlap threshold for polygon strategy.
            seed: Random seed for reproducibility.
        """
        # Apply defaults
        window_sizes = window_sizes if window_sizes is not None else BATCH_DEFAULTS['window_sizes']
        noise_types = noise_types if noise_types is not None else BATCH_DEFAULTS['noise_types']
        noise_levels = noise_levels if noise_levels is not None else BATCH_DEFAULTS['noise_levels']
        strides = strides if strides is not None else BATCH_DEFAULTS['strides']
        threshold = threshold if threshold is not None else DEFAULTS['threshold']
        
        sep = OUTPUT_FORMAT['separator']
        print(f"\n{sep}\nBATCH DATA CREATION - {self.area_name.upper()}\n{sep}")
        print(f"Strategy: {self.strategy}")
        print(f"Window sizes: {window_sizes}")
        print(f"Strides: {[stride_label(s) for s in strides]}")
        print(f"Noise types: {noise_types}")
        print(f"Noise levels: {noise_levels}")
        print(f"Crops per image per config: {n_crops}")
        print(f"{sep}\n")
        
        all_results = []
        config_count = len(window_sizes) * len(strides) * len(noise_types) * len(noise_levels)
        current_config = 0
        original_window_size, original_output_dir = self.window_size, self.output_dir
        
        for window_size in window_sizes:
            self.window_size = window_size
            for stride in strides:
                for noise_type in noise_types:
                    for noise_level in noise_levels:
                        current_config += 1
                        config_label = f"w{window_size}_s{stride_label(stride)}_n{noise_type[0]}{int(noise_level*100)}"
                        print(f"[{current_config}/{config_count}] {config_label}")
                        
                        # Set output dir - include stride if organize_by_stride is enabled
                        if self.organize_by_stride:
                            self.output_dir = original_output_dir / self.strategy / self.area_name / f"window_{window_size}" / f"stride_{stride_label(stride)}" / f"{noise_type}_noise{int(noise_level*100)}"
                        else:
                            self.output_dir = original_output_dir / self.strategy / self.area_name / f"window_{window_size}" / f"{noise_type}_noise{int(noise_level*100)}"
                        self.output_dir.mkdir(parents=True, exist_ok=True)
                        
                        df_results = self.run_data_creation(n_crops, stride, threshold, seed, noise_level > 0, noise_type)
                        df_results['window_size'] = window_size
                        df_results['stride'] = stride_label(stride)
                        df_results['noise_type'] = noise_type
                        df_results['noise_level'] = noise_level
                        all_results.append(df_results)
                        self.save_results(df_results)
                        print(f"  ✓ Generated {len(df_results)} crops\n")
        
        # Restore state
        self.window_size, self.output_dir = original_window_size, original_output_dir
        
        # Save combined results
        if all_results:
            df_combined = pd.concat(all_results, ignore_index=True)
            combined_csv_filename = FILE_PATTERNS['batch_summary_csv'].format(area=self.area_name, strategy=self.strategy)
            combined_csv = self.output_dir / combined_csv_filename
            df_combined.to_csv(combined_csv, index=False)
            print(f"\nBatch summary saved to: {combined_csv}")
            sep = OUTPUT_FORMAT['separator']
            print(f"\n{sep}\nBATCH SUMMARY\n{sep}")
            print(f"Total configurations tested: {config_count}")
            print(f"Total crops generated: {len(df_combined)}")
            print(f"Output directory: {self.output_dir}\n{sep}\n")
        else:
            print("\nNo crops generated in batch.")


def main():
    """Parse arguments and run data creation."""
    parser = argparse.ArgumentParser(
        description="Geoglyph dataset creation and image cropping orchestration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_create.py --area lluta --strategy random --n-crops 100
  python data_create.py --area chugchug --strategy fixed --n-crops 50 --window-size 256
  python data_create.py --area unita --strategy polygon --n-crops 30 --threshold 0.5
  python data_create.py --area lluta --strategy random --n-crops 100 --fill-noise --noise-type gaussian
        """
    )
    
    parser.add_argument(
        "--area",
        type=str,
        required=True,
        choices=AVAILABLE_AREAS,
        help=f"Area name ({', '.join(AVAILABLE_AREAS)})"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=list(CROP_STRATEGY_REGISTRY.keys()),
        help="Cropping strategy to use"
    )
    
    parser.add_argument(
        "--n-crops",
        type=int,
        required=True,
        help="Number of crops per image"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULTS['window_size'],
        help=f"Size of crop windows (default: {DEFAULTS['window_size']})"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULTS['stride'],
        help="Step size for fixed/polygon strategies (optional)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULTS['threshold'],
        help=f"Overlap threshold for polygon strategy (default: {DEFAULTS['threshold']})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory for cropped images (default: ./data)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    
    parser.add_argument(
        "--fill-noise",
        action="store_true",
        help="Fill out-of-bounds regions with noise"
    )
    
    parser.add_argument(
        "--organize-by-stride",
        action="store_true",
        help="Include stride in output directory structure"
    )
    
    parser.add_argument(
        "--noise-type",
        type=str,
        choices=['gaussian', 'uniform', 'perlin'],
        default=DEFAULTS['noise_type'],
        help=f"Type of noise for filling out-of-bounds regions (default: {DEFAULTS['noise_type']})"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch mode with multiple window sizes and noise configurations"
    )
    
    parser.add_argument(
        "--batch-window-sizes",
        type=int,
        nargs='+',
        default=BATCH_DEFAULTS['window_sizes'],
        help=f"Window sizes for batch mode (default: {BATCH_DEFAULTS['window_sizes']})"
    )
    
    parser.add_argument(
        "--batch-noise-types",
        type=str,
        nargs='+',
        choices=['gaussian', 'uniform', 'perlin'],
        default=BATCH_DEFAULTS['noise_types'],
        help=f"Noise types for batch mode (default: {' '.join(BATCH_DEFAULTS['noise_types'])})"
    )
    
    parser.add_argument(
        "--batch-noise-levels",
        type=float,
        nargs='+',
        default=BATCH_DEFAULTS['noise_levels'],
        help=f"Noise levels for batch mode (default: {BATCH_DEFAULTS['noise_levels']})"
    )
    
    parser.add_argument(
        "--batch-strides",
        type=str,
        nargs='+',
        default=None,
        help="Stride values for batch mode (e.g., 'auto' '50' '25x30'). Default: [None]"
    )
    
    args = parser.parse_args()
    
    # Parse stride values for batch mode
    if args.batch_strides:
        parsed_strides = []
        for s in args.batch_strides:
            if s.lower() == 'auto' or s.lower() == 'none':
                parsed_strides.append(None)
            elif 'x' in s:
                parts = s.split('x')
                if len(parts) == 2:
                    parsed_strides.append((int(parts[0]), int(parts[1])))
            else:
                parsed_strides.append(int(s))
        args.batch_strides = parsed_strides
    
    # Validate strategy-specific arguments
    if args.strategy == 'polygon' and args.threshold is None:
        args.threshold = 0.1
    
    # Run data creation
    creator = GeoglyphDataCreator(
        area_name=args.area,
        strategy=args.strategy,
        window_size=args.window_size,
        output_dir=args.output,
        organize_by_stride=args.organize_by_stride
    )
    
    if args.batch:
        # Run batch mode
        creator.run_batch_pipeline(
            n_crops=args.n_crops,
            window_sizes=args.batch_window_sizes,
            noise_types=args.batch_noise_types,
            noise_levels=args.batch_noise_levels,
            strides=args.batch_strides,
            threshold=args.threshold,
            seed=args.seed
        )
    else:
        # Run single configuration
        creator.run_full_pipeline(
            n_crops=args.n_crops,
            stride=args.stride,
            threshold=args.threshold,
            seed=args.seed,
            fill_noise=args.fill_noise,
            noise_type=args.noise_type
        )


if __name__ == "__main__":
    exit(main())
