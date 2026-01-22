import json
import matplotlib.pyplot as plt
import numpy as np
import shutil
from pathlib import Path

from handle import (
    PROJECT_DIR,
    UNITA_SUMMARY_PATH,
    CHUG_SUMMARY_PATH,
    LLUTA_SUMMARY_PATH,
    load_json,
    save_json
)

# Aliases for backward compatibility
BASE_DIR = PROJECT_DIR
UNITA_PATH = UNITA_SUMMARY_PATH
CHUG_PATH = CHUG_SUMMARY_PATH
LLUTA_PATH = LLUTA_SUMMARY_PATH

# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def get_geos(summary):
    """Extract all class 1 (geo) polygons from summary."""
    return [polygon for polygon in summary['polygons'] if polygon['class'] == 1]

def get_geos_sizes(geos):
    """Extract relevant size and file information from geos."""
    return [
        {
            "index": geo['polygon_index'],
            "boxmeters": geo['bbox_size_meters'],
            "shape": geo["image_shape"],
            "files": geo["files"]
        }
        for geo in geos
    ]

def extract_metrics(geos_sizes):
    """Extract width, height, and area metrics from geos_sizes."""
    widths = [geo['boxmeters']['width_m'] for geo in geos_sizes]
    heights = [geo['boxmeters']['height_m'] for geo in geos_sizes]
    areas = [geo['boxmeters']['area_m2'] for geo in geos_sizes]
    return widths, heights, areas

def _get_axis_limits_with_padding(data_list, padding=0.05):
    """Calculate min/max limits with padding for consistent axis scaling."""
    flat_data = [val for sublist in data_list for val in sublist]
    data_min, data_max = min(flat_data), max(flat_data)
    padding_val = (data_max - data_min) * padding
    return data_min - padding_val, data_max + padding_val

def _remove_outliers_iqr(data, multiplier=1.5):
    """Remove outliers using IQR method. Returns data without outliers."""
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - multiplier * iqr
    upper_bound = q75 + multiplier * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def calculate_pixel_scale(geos_sizes):
    """Calculate the scale (meters per pixel) for each geo and return mean scale."""
    scales = []
    for geo in geos_sizes:
        bbox_width = geo['boxmeters']['width_m']
        bbox_height = geo['boxmeters']['height_m']
        img_width = geo['shape']['width']
        img_height = geo['shape']['height']

        # Calculate meters per pixel for this geo
        width_scale = bbox_width / img_width if img_width > 0 else 0
        height_scale = bbox_height / img_height if img_height > 0 else 0

        # Use average scale of width and height
        scale = (width_scale + height_scale) / 2
        scales.append(scale)

    return np.mean(scales) if scales else 0

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_geos_statistics(geos_sizes_list, dataset_names):
    """Print average (excluding outliers) dimensions and pixel-based window sizes for each dataset."""
    print("\n" + "="*90)
    print("GEO DIMENSIONS STATISTICS (for sliding window determination)")
    print("="*90)

    for geos_sizes, name in zip(geos_sizes_list, dataset_names):
        widths, heights, areas = extract_metrics(geos_sizes)

        # Remove outliers for mean calculation
        widths_no_outliers = _remove_outliers_iqr(widths)
        heights_no_outliers = _remove_outliers_iqr(heights)
        areas_no_outliers = _remove_outliers_iqr(areas)

        width_mean = np.mean(widths_no_outliers)
        height_mean = np.mean(heights_no_outliers)

        # Calculate scale (meters per pixel)
        scale = calculate_pixel_scale(geos_sizes)

        # Convert meters to pixels
        window_size_m = max(int(np.ceil(width_mean)), int(np.ceil(height_mean)))
        window_size_px = int(np.ceil(window_size_m / scale)) if scale > 0 else 0

        print(f"\n{name}:")
        print(f"  Count:           {len(widths)} (outliers removed from mean calculation)")
        print(f"  Width:  avg={width_mean:.1f}m  min={min(widths):.1f}m  max={max(widths):.1f}m")
        print(f"  Height: avg={height_mean:.1f}m  min={min(heights):.1f}m  max={max(heights):.1f}m")
        print(f"  Area:   avg={np.mean(areas_no_outliers):.1f}m²  min={min(areas):.1f}m²  max={max(areas):.1f}m²")
        print(f"  Pixel scale: {scale:.4f} m/pixel")
        print(f"  Suggested sliding window: {window_size_m}m x {window_size_m}m ({window_size_px}px x {window_size_px}px)")

    print("\n" + "="*90)

def plot_geos_statistics(geos_sizes_list, dataset_names):
    """Create a matplotlib figure displaying geo statistics in three columns for sliding window determination."""
    num_datasets = len(geos_sizes_list)
    fig, axes = plt.subplots(1, num_datasets, figsize=(18, 8))
    fig.suptitle("Geo Dimensions Statistics (for Sliding Window Determination)", fontsize=16, fontweight='bold')

    # Handle single dataset case
    if num_datasets == 1:
        axes = [axes]

    for ax, geos_sizes, name in zip(axes, geos_sizes_list, dataset_names):
        widths, heights, areas = extract_metrics(geos_sizes)

        # Remove outliers
        widths_no_outliers = _remove_outliers_iqr(widths)
        heights_no_outliers = _remove_outliers_iqr(heights)
        areas_no_outliers = _remove_outliers_iqr(areas)

        width_mean = np.mean(widths_no_outliers)
        height_mean = np.mean(heights_no_outliers)

        # Calculate scale and window sizes
        scale = calculate_pixel_scale(geos_sizes)
        window_size_m = max(int(np.ceil(width_mean)), int(np.ceil(height_mean)))
        window_size_px = int(np.ceil(window_size_m / scale)) if scale > 0 else 0

        # Create statistics text
        stats_text = f"{name}\n"
        stats_text += f"{'='*30}\n\n"
        stats_text += f"Count: {len(widths)}\n"
        stats_text += f"(Outliers excluded)\n\n"
        stats_text += f"Width (m):\n"
        stats_text += f"  Avg: {width_mean:.1f}\n"
        stats_text += f"  Min: {min(widths):.1f}\n"
        stats_text += f"  Max: {max(widths):.1f}\n\n"
        stats_text += f"Height (m):\n"
        stats_text += f"  Avg: {height_mean:.1f}\n"
        stats_text += f"  Min: {min(heights):.1f}\n"
        stats_text += f"  Max: {max(heights):.1f}\n\n"
        stats_text += f"Area (m²):\n"
        stats_text += f"  Avg: {np.mean(areas_no_outliers):.1f}\n"
        stats_text += f"  Min: {min(areas):.1f}\n"
        stats_text += f"  Max: {max(areas):.1f}\n\n"
        stats_text += f"Pixel Scale:\n"
        stats_text += f"  {scale:.4f} m/px\n\n"
        stats_text += f"Window Size:\n"
        stats_text += f"  {window_size_m}m × {window_size_m}m\n"
        stats_text += f"  {window_size_px}px × {window_size_px}px"

        # Display text on axis
        ax.axis('off')
        ax.text(0.5, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1.5))

    plt.tight_layout()
    return fig

def plot_distributions(widths, heights, areas, title="Distribution of Geo Dimensions"):
    """Create box plots for widths, heights, and areas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    metrics = [(widths, 'Width (meters)', 0), (heights, 'Height (meters)', 1), (areas, 'Area (m²)', 2)]

    for data, label, idx in metrics:
        axes[idx].boxplot(data)
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f'{label} Distribution')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_histograms(widths, heights, areas, title="Distribution of Geo Dimensions", bins=20):
    """Create histograms for widths, heights, and areas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    colors = ['skyblue', 'lightgreen', 'lightcoral']
    metrics = [(widths, 'Width (meters)', 0), (heights, 'Height (meters)', 1), (areas, 'Area (m²)', 2)]

    for data, label, idx in metrics:
        axes[idx].hist(data, bins=bins, color=colors[idx], edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(label)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{label} Distribution')
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def plot_combined_comparison(datasets, dataset_names, title="Comparison of Geo Dimensions Across Datasets"):
    """Create box plots comparing distributions across multiple datasets with consistent scales."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Extract metrics for all datasets
    widths_data = [extract_metrics(ds)[0] for ds in datasets]
    heights_data = [extract_metrics(ds)[1] for ds in datasets]
    areas_data = [extract_metrics(ds)[2] for ds in datasets]

    # Calculate consistent axis limits
    widths_min, widths_max = _get_axis_limits_with_padding(widths_data)
    heights_min, heights_max = _get_axis_limits_with_padding(heights_data)
    areas_min, areas_max = _get_axis_limits_with_padding(areas_data)

    # Plot metrics
    metrics = [
        (widths_data, 'Width (meters)', widths_min, widths_max, 0),
        (heights_data, 'Height (meters)', heights_min, heights_max, 1),
        (areas_data, 'Area (m²)', areas_min, areas_max, 2)
    ]

    for data, label, y_min, y_max, idx in metrics:
        axes[idx].boxplot(data, labels=dataset_names)
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f'{label} by Dataset')
        axes[idx].set_ylim(y_min, y_max)
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def plot_single_figure_all_datasets(all_data, dataset_names, title="Distribution Comparison", bins=15):
    """Create histograms for all datasets in one figure (rows=datasets, cols=metrics) with consistent scales."""
    num_datasets = len(all_data)
    fig, axes = plt.subplots(num_datasets, 3, figsize=(18, 5*num_datasets))
    fig.suptitle(title, fontsize=16)

    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(1, -1)

    colors = ['skyblue', 'lightgreen', 'lightcoral']

    # Calculate global limits for consistent scaling across all rows
    all_widths = [width for data in all_data for width in extract_metrics(data)[0]]
    all_heights = [height for data in all_data for height in extract_metrics(data)[1]]
    all_areas = [area for data in all_data for area in extract_metrics(data)[2]]

    widths_min, widths_max = _get_axis_limits_with_padding([all_widths])
    heights_min, heights_max = _get_axis_limits_with_padding([all_heights])
    areas_min, areas_max = _get_axis_limits_with_padding([all_areas])

    # Plot each dataset
    for row, (geos_sizes, dataset_name) in enumerate(zip(all_data, dataset_names)):
        widths, heights, areas = extract_metrics(geos_sizes)

        # Width histogram
        axes[row, 0].hist(widths, bins=bins, color=colors[0], edgecolor='black', alpha=0.7)
        axes[row, 0].set_xlabel('Width (meters)')
        axes[row, 0].set_ylabel('Frequency')
        axes[row, 0].set_title(f'{dataset_name} - Width Distribution')
        axes[row, 0].set_xlim(widths_min, widths_max)
        axes[row, 0].grid(True, alpha=0.3, axis='y')

        # Height histogram
        axes[row, 1].hist(heights, bins=bins, color=colors[1], edgecolor='black', alpha=0.7)
        axes[row, 1].set_xlabel('Height (meters)')
        axes[row, 1].set_ylabel('Frequency')
        axes[row, 1].set_title(f'{dataset_name} - Height Distribution')
        axes[row, 1].set_xlim(heights_min, heights_max)
        axes[row, 1].grid(True, alpha=0.3, axis='y')

        # Area histogram
        axes[row, 2].hist(areas, bins=bins, color=colors[2], edgecolor='black', alpha=0.7)
        axes[row, 2].set_xlabel('Area (m²)')
        axes[row, 2].set_ylabel('Frequency')
        axes[row, 2].set_title(f'{dataset_name} - Area Distribution')
        axes[row, 2].set_xlim(areas_min, areas_max)
        axes[row, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def copy_geos_files_by_area(summary_json_path, source_dir, output_base_dir, area_name):
    """
    Copy all class 1 (geo) image files to area-specific output directories.
    
    Args:
        summary_json_path: Path to the summary.json file
        source_dir: Directory where extracted images are stored
        output_base_dir: Base output directory (creates subdirectory for each area)
        area_name: Area name (e.g., 'UNITA', 'CHUG', 'LLUTA')
    """
    summary = load_json(summary_json_path)
    geos = get_geos(summary)
    
    output_dir = Path(output_base_dir) / area_name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir)
    
    copied_count = 0
    for geo in geos:
        polygon_index = geo['polygon_index']
        
        try:
            # Copy image files
            for filename in geo['files'].values():
                source_file = source_dir / filename
                if source_file.exists():
                    shutil.copy2(source_file, output_dir / filename)
            
            # Copy metadata JSON
            base_name = f"geoglif_{polygon_index:04d}"
            metadata_file = source_dir / f"{base_name}_metadata.json"
            if metadata_file.exists():
                shutil.copy2(metadata_file, output_dir / f"{base_name}_metadata.json")
            
            copied_count += 1
        except Exception as e:
            print(f"Error copying polygon {polygon_index}: {e}")
    
    print(f"Copied {copied_count} geos to {output_dir}")


if __name__ == "__main__":
    # Load data from all areas
    unita_summary = load_json(UNITA_PATH)
    chug_summary = load_json(CHUG_PATH)
    lluta_summary = load_json(LLUTA_PATH)
    
    # Extract geos (class 1 polygons) from each area
    summaries = [unita_summary, chug_summary, lluta_summary]
    dataset_names = ['UNITA', 'CHUG', 'LLUTA']
    geos_list = [get_geos(s) for s in summaries]
    geos_sizes_list = [get_geos_sizes(g) for g in geos_list]
    
    # Print statistics for sliding window determination
    print_geos_statistics(geos_sizes_list, dataset_names)
    
    # Generate visualizations
    plot_geos_statistics(geos_sizes_list, dataset_names)
    
    plot_single_figure_all_datasets(
        geos_sizes_list,
        dataset_names,
        "Geo Dimensions Distribution - All Datasets"
    )
    
    plot_combined_comparison(
        geos_sizes_list,
        dataset_names
    )
    
    plt.show()