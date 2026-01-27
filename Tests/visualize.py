"""
Visualization module for geoglyph image analysis.

Provides functions for creating histograms, 2D scatter plots, and 3D scatter plots
with metadata annotations (region name, image size, etc.).
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Optional, Tuple, List


def plot_1d_histogram(
    df_results: pd.DataFrame,
    metric_column: str,
    area_name: str,
    target_size: Tuple[int, int],
    title: str,
    xlabel: str,
    output_path: str,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 20
) -> None:
    """
    Create a 1D histogram by class for a single metric.
    
    Args:
        df_results: DataFrame with 'class' and metric_column columns.
        metric_column: Name of the column to plot.
        area_name: Name of the area (e.g., 'lluta', 'chugchug').
        target_size: Tuple of (width, height) image size.
        title: Main title for the plot.
        xlabel: Label for x-axis.
        output_path: Path to save the figure.
        figsize: Figure size as (width, height).
        bins: Number of histogram bins per class.
    """
    fig, ax = plt.subplots(figsize=figsize)
    classes = sorted(df_results['class'].unique())
    
    for cls in classes:
        subset = df_results[df_results['class'] == cls][metric_column]
        ax.hist(subset, bins=bins, alpha=0.6, label=f"class {cls}")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title} - {area_name.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metadata at bottom right
    fig.text(0.99, 0.01, f"Size: {target_size[0]}x{target_size[1]}",
             ha="right", va="bottom", fontsize=9, color="gray")
    
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=200)
    print(f"Saved histogram to {output_path}")
    plt.close()


def plot_1d_boxplot(
    df_results: pd.DataFrame,
    metric_column: str,
    area_name: str,
    target_size: Tuple[int, int],
    title: str,
    ylabel: str,
    output_path: str,
    figsize: Tuple[int, int] = (8, 5)
) -> None:
    """
    Create a boxplot by class for a single metric.
    
    Args:
        df_results: DataFrame with 'class' and metric_column columns.
        metric_column: Name of the column to plot.
        area_name: Name of the area (e.g., 'lluta', 'chugchug').
        target_size: Tuple of (width, height) image size.
        title: Main title for the plot.
        ylabel: Label for y-axis.
        output_path: Path to save the figure.
        figsize: Figure size as (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)
    classes = sorted(df_results['class'].unique())
    
    data = [df_results[df_results['class'] == cls][metric_column] for cls in classes]
    ax.boxplot(data, tick_labels=[f"class {cls}" for cls in classes], showfliers=True)
    
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {area_name.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.3)
    
    # Add metadata at bottom right
    fig.text(0.99, 0.01, f"Size: {target_size[0]}x{target_size[1]}",
             ha="right", va="bottom", fontsize=9, color="gray")
    
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=200)
    print(f"Saved boxplot to {output_path}")
    plt.close()


def plot_2d_scatter(
    df_results: pd.DataFrame,
    x_column: str,
    y_column: str,
    area_name: str,
    target_size: Tuple[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    figsize: Tuple[int, int] = (10, 8),
    class1_color: str = "green",
    other_color: str = "black"
) -> None:
    """
    Create a 2D scatter plot for two metrics, colored by class.
    
    Args:
        df_results: DataFrame with 'class', x_column, and y_column.
        x_column: Name of x-axis metric column.
        y_column: Name of y-axis metric column.
        area_name: Name of the area (e.g., 'lluta', 'chugchug').
        target_size: Tuple of (width, height) image size.
        title: Main title for the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        output_path: Path to save the figure.
        figsize: Figure size as (width, height).
        class1_color: Color for class 1 (geo) samples.
        other_color: Color for other classes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate by class
    class1 = df_results[df_results["class"] == 1]
    other_classes = df_results[df_results["class"] != 1]
    
    # Plot class 1 as specified color
    ax.scatter(class1[x_column], class1[y_column], 
               c=class1_color, label="Class 1 (Geo)", alpha=0.7, s=50)
    
    # Plot other classes as specified color
    ax.scatter(other_classes[x_column], other_classes[y_column], 
               c=other_color, label="Other Classes", alpha=0.7, s=50)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {area_name.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metadata at bottom right
    fig.text(0.99, 0.01, f"Size: {target_size[0]}x{target_size[1]}",
             ha="right", va="bottom", fontsize=9, color="gray")
    
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=200)
    print(f"Saved 2D scatter plot to {output_path}")
    plt.close()


def plot_3d_scatter(
    df_results: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    area_name: str,
    target_size: Tuple[int, int],
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    output_path: str,
    figsize: Tuple[int, int] = (12, 10),
    class1_color: str = "green",
    other_color: str = "black"
) -> None:
    """
    Create a 3D scatter plot for three metrics, colored by class.
    
    Args:
        df_results: DataFrame with 'class', x_column, y_column, and z_column.
        x_column: Name of x-axis metric column.
        y_column: Name of y-axis metric column.
        z_column: Name of z-axis metric column.
        area_name: Name of the area (e.g., 'lluta', 'chugchug').
        target_size: Tuple of (width, height) image size.
        title: Main title for the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        zlabel: Label for z-axis.
        output_path: Path to save the figure.
        figsize: Figure size as (width, height).
        class1_color: Color for class 1 (geo) samples.
        other_color: Color for other classes.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate by class
    class1 = df_results[df_results["class"] == 1]
    other_classes = df_results[df_results["class"] != 1]
    
    # Plot class 1 as specified color
    ax.scatter(class1[x_column], class1[y_column], class1[z_column],
               c=class1_color, label="Class 1 (Geo)", alpha=0.7, s=50)
    
    # Plot other classes as specified color
    ax.scatter(other_classes[x_column], other_classes[y_column], other_classes[z_column],
               c=other_color, label="Other Classes", alpha=0.7, s=50)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f"{title} - {area_name.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metadata at bottom right
    fig.text(0.99, 0.01, f"Size: {target_size[0]}x{target_size[1]}",
             ha="right", va="bottom", fontsize=9, color="gray")
    
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=200)
    print(f"Saved 3D scatter plot to {output_path}")
    plt.close()
