"""
Comprehensive geoglyph image analysis test suite.

This script orchestrates the computation of spatial and textural metrics
(Moran's I, Entropy, GLCM Contrast) on geoglyph images from a specified area,
and generates visualizations (histograms, 2D/3D scatter plots) with results
saved to a user-specified output directory.

Usage:
    python repr_test.py --area lluta --methods moran entropy contrast --output ./results
    python repr_test.py --area chugchug --methods moran --output ./my_results
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple

# Import our custom modules
from methods import morans_i, spatial_entropy, compute_glcm_contrast, compute_glcm_homogeneity
from visualize import (
    plot_1d_histogram,
    plot_1d_boxplot,
    plot_2d_scatter,
    plot_3d_scatter
)


class GeoglyphAnalyzer:
    """Orchestrates analysis of geoglyph images."""
    
    def __init__(
        self,
        area_name: str,
        target_size: Tuple[int, int] = (100, 100),
        output_dir: str = "./results"
    ):
        """
        Initialize the analyzer.
        
        Args:
            area_name: Name of the area (e.g., 'lluta', 'chugchug', 'unita').
            target_size: Image resize target as (width, height).
            output_dir: Directory to save results.
        """
        self.area_name = area_name
        self.area_subdir = f"{area_name}_polygons"
        self.target_size = target_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data paths
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.summary_path = self.data_dir / self.area_subdir / "summary.json"
        
        # Validate that the summary exists
        if not self.summary_path.exists():
            raise FileNotFoundError(f"Summary not found: {self.summary_path}")
        
        self.results = []
    
    def load_summary(self) -> dict:
        """Load the area summary JSON."""
        with open(self.summary_path, 'r') as f:
            return json.load(f)
    
    def parse_images(self, summary: dict):
        """Yield image metadata from summary."""
        for polygon in summary['polygons']:
            yield {
                "id": polygon['polygon_index'],
                "class": polygon['class'],
                "img_name": polygon["files"]['ortho_jpeg'],
                "img_sizes": polygon["bbox_size_meters"]
            }
    
    def load_image(self, path: Path) -> np.ndarray:
        """Load and convert image to grayscale numpy array."""
        img = Image.open(path)
        img_gray = img.convert("L").resize(self.target_size, Image.Resampling.BILINEAR)
        return np.array(img_gray)
    
    def compute_metrics(self, image_array: np.ndarray, methods: List[str]) -> Dict[str, float]:
        """
        Compute requested metrics for an image.
        
        Args:
            image_array: Grayscale numpy array.
            methods: List of method names ('moran', 'entropy', 'contrast', 'homogeneity').
        
        Returns:
            Dictionary of {method: value}.
        """
        metrics = {}
        img_df = pd.DataFrame(image_array)
        
        if 'moran' in methods:
            metrics['moran'] = morans_i(img_df, adjacency="rook", wrap=False, nan_policy="omit")
        
        if 'entropy' in methods:
            metrics['entropy'] = spatial_entropy(img_df, adjacency="rook", wrap=False, nan_policy="omit")
        
        if 'contrast' in methods:
            metrics['contrast'] = compute_glcm_contrast(image_array, distances=[1], angles=[0], levels=256)
        
        if 'homogeneity' in methods:
            metrics['homogeneity'] = compute_glcm_homogeneity(image_array, distances=[1], angles=[0], levels=256)
        
        return metrics
    
    def run_analysis(self, methods: List[str]) -> pd.DataFrame:
        """
        Run analysis on all images in the area.
        
        Args:
            methods: List of method names to compute.
        
        Returns:
            DataFrame with results.
        """
        print(f"Processing dataset: {self.area_subdir}")
        print(f"Target size: {self.target_size}")
        print(f"Methods: {', '.join(methods)}")
        print("-" * 50)
        
        summary = self.load_summary()
        results = []
        
        for idx, image_meta in enumerate(self.parse_images(summary)):
            img_path = self.data_dir / self.area_subdir / image_meta["img_name"]
            
            try:
                img_array = self.load_image(img_path)
                metrics = self.compute_metrics(img_array, methods)
                
                result = {
                    "id": image_meta['id'],
                    "class": image_meta['class'],
                    **metrics
                }
                results.append(result)
                
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                print(f"Image ID: {image_meta['id']}, Class: {image_meta['class']}, {metric_str}")
            
            except Exception as e:
                print(f"Error processing image {image_meta['id']}: {e}")
                continue
        
        df_results = pd.DataFrame(results)
        return df_results
    
    def save_results(self, df_results: pd.DataFrame) -> None:
        """Save results DataFrame to CSV."""
        csv_path = self.output_dir / f"{self.area_name}_results.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
    
    def visualize_results(self, df_results: pd.DataFrame, methods: List[str]) -> None:
        """
        Generate visualizations based on available metrics.
        
        Args:
            df_results: DataFrame with computed metrics.
            methods: List of method names computed.
        """
        # Define metric properties (column_name, display_name, unit_label)
        metric_props = {
            'moran': ("moran", "Moran's I", "Moran's I"),
            'entropy': ("entropy", "Spatial Entropy", "Entropy (bits)"),
            'contrast': ("contrast", "GLCM Contrast", "GLCM Contrast"),
            'homogeneity': ("homogeneity", "GLCM Homogeneity", "GLCM Homogeneity")
        }
        
        # 1D plots: histogram and boxplot for each metric
        for method in methods:
            if method not in metric_props:
                continue
            
            col_name, display_name, unit = metric_props[method]
            
            # Histogram
            hist_path = self.output_dir / f"{self.area_name}_{method}_histogram.png"
            plot_1d_histogram(
                df_results, col_name, self.area_name, self.target_size,
                f"{display_name} Distribution", unit, str(hist_path)
            )
            
            # Boxplot
            box_path = self.output_dir / f"{self.area_name}_{method}_boxplot.png"
            plot_1d_boxplot(
                df_results, col_name, self.area_name, self.target_size,
                f"{display_name} by Class", unit, str(box_path)
            )
        
        # 2D scatter plots: pairwise combinations
        if len(methods) >= 2:
            method_pairs = [
                ('moran', 'entropy'),
                ('moran', 'contrast'),
                ('moran', 'homogeneity'),
                ('entropy', 'contrast'),
                ('entropy', 'homogeneity'),
                ('contrast', 'homogeneity')
            ]
            
            for m1, m2 in method_pairs:
                if m1 not in methods or m2 not in methods:
                    continue
                
                props1 = metric_props[m1]
                props2 = metric_props[m2]
                
                scatter_path = self.output_dir / f"{self.area_name}_{m1}_vs_{m2}_scatter.png"
                plot_2d_scatter(
                    df_results, props1[0], props2[0],
                    self.area_name, self.target_size,
                    f"{props1[1]} vs {props2[1]}",
                    props1[2], props2[2],
                    str(scatter_path)
                )
        
        # 3D scatter plot: all three metrics if available
        if len(methods) >= 3 and all(m in methods for m in ['moran', 'entropy', 'contrast', 'homogeneity']):
            scatter_3d_path = self.output_dir / f"{self.area_name}_3d_scatter.png"
            plot_3d_scatter(
                df_results, "entropy", "moran", "contrast",
                self.area_name, self.target_size,
                "3D Feature Space",
                "Entropy (bits)", "Moran's I", "GLCM Contrast",
                str(scatter_3d_path)
            )
    
    def run_full_analysis(self, methods: List[str]) -> None:
        """
        Execute full analysis pipeline: compute metrics, save, and visualize.
        
        Args:
            methods: List of method names to compute.
        """
        print(f"\n{'='*60}")
        print(f"Geoglyph Analysis: {self.area_name.upper()}")
        print(f"{'='*60}\n")
        
        # Run analysis
        df_results = self.run_analysis(methods)
        
        print(f"\nProcessed {len(df_results)} images successfully.")
        print(f"Results shape: {df_results.shape}")
        print(f"\nSample statistics:")
        print(df_results.describe())
        
        # Save results
        self.save_results(df_results)
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        self.visualize_results(df_results, methods)
        
        print(f"\n{'='*60}")
        print(f"Analysis complete! Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """Parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Comprehensive geoglyph image analysis test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python repr_test.py --area lluta --methods moran entropy contrast --output ./results
  python repr_test.py --area chugchug --methods moran --output ./my_results
  python repr_test.py --area unita --methods entropy contrast --output ./output
        """
    )
    
    parser.add_argument(
        "--area",
        type=str,
        required=True,
        help="Area name (e.g., 'lluta', 'chugchug', 'unita')"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        choices=['moran', 'entropy', 'contrast', 'homogeneity'],
        required=True,
        help="Methods to compute (can specify multiple)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[100, 100],
        help="Target image size as 'width height' (default: 100 100)"
    )
    
    args = parser.parse_args()
    target_size = tuple(args.size)
    
    # Validate methods
    valid_methods = {'moran', 'entropy', 'contrast', 'homogeneity'}
    requested_methods = set(args.methods)
    if not requested_methods.issubset(valid_methods):
        parser.error(f"Invalid methods: {requested_methods - valid_methods}")
    
    # Run analysis
    try:
        analyzer = GeoglyphAnalyzer(
            area_name=args.area,
            target_size=target_size,
            output_dir=args.output
        )
        analyzer.run_full_analysis(list(args.methods))
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
