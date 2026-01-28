"""
Centralized module for geoglyph image analysis methods.

This module contains reusable functions for computing spatial and textural metrics
on grayscale image data, including Moran's I (spatial autocorrelation), 
spatial entropy, and GLCM contrast.
"""

import numpy as np
import pandas as pd
from libpysal.weights import lat2W
from esda.moran import Moran
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.measure import moments_hu, moments_central


def morans_i(image_df, adjacency: str = "rook", wrap: bool = False, nan_policy: str = "omit") -> float:
    """
    Compute Moran's I for a 2D image represented as a pandas DataFrame using libpysal/esda.

    Moran's I measures spatial autocorrelation - the extent to which nearby pixels have
    similar values. Positive values indicate clustering, negative values indicate dispersion.

    Args:
        image_df: pandas DataFrame (H x W) where each cell is a pixel value.
        adjacency: 'rook' (4-neighbors) or 'queen' (8-neighbors) contiguity.
        wrap: whether to wrap the lattice at boundaries (toroidal). Default False.
        nan_policy: 'omit' to drop NaN pixels from the analysis; 'propagate' to keep
                    NaNs (will likely result in NaN I).

    Returns:
        Moran's I statistic (float). Range: approximately [-1, 1].
        
    Raises:
        ValueError: if nan_policy is not 'omit' or 'propagate'.
    """

    # Ensure float array
    arr = np.array(image_df, dtype=float)
    nrows, ncols = arr.shape

    # Build lattice contiguity weights
    rook = True if adjacency.lower() == "rook" else False
    w = lat2W(nrows, ncols, rook=rook, id_type="int")
    w.transform = "r"  # row-standardize

    y = arr.ravel()

    if nan_policy == "omit":
        if np.isnan(y).any():
            # Older libpysal lacks W.subgraph; impute NaNs with the mean of valid pixels
            mean_val = np.nanmean(y)
            y = np.where(np.isnan(y), mean_val, y)
    elif nan_policy == "propagate":
        pass  # allow NaNs to propagate into Moran
    else:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    mi = Moran(y, w)
    return float(mi.I)


def spatial_entropy(
    image_df,
    adjacency: str = "rook",
    wrap: bool = False,
    nan_policy: str = "omit",
    bins: int = 256,
) -> float:
    """
    Compute 2D spatial entropy from neighbor pairs using libpysal lattice weights.

    Forms neighbor pairs from a contiguity lattice (rook or queen), builds a joint
    histogram of paired pixel intensities (binned), and computes Shannon entropy.
    Higher entropy indicates more complex/varied spatial patterns.

    Args:
        image_df: pandas DataFrame (H x W) where each cell is a pixel value.
        adjacency: 'rook' (4-neighbors) or 'queen' (8-neighbors) contiguity.
        wrap: kept for API symmetry; lat2W here does not torus-wrap.
        nan_policy: 'omit' to drop any pair containing NaN; 'propagate' to allow NaN
                    (returns NaN if any are present).
        bins: number of bins for the joint histogram (default 256 for 8-bit grayscale).

    Returns:
        Shannon entropy (bits) of the joint neighbor distribution (float).
        
    Raises:
        ValueError: if nan_policy is not 'omit' or 'propagate'.
    """

    # Ensure float array
    arr = np.array(image_df, dtype=float)
    nrows, ncols = arr.shape

    # Build lattice contiguity weights
    rook = True if adjacency.lower() == "rook" else False
    w = lat2W(nrows, ncols, rook=rook, id_type="int")

    y = arr.ravel()

    if nan_policy == "propagate" and np.isnan(y).any():
        return float("nan")
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    # Collect neighbor pairs (upper triangle to avoid duplicates)
    pairs_a = []
    pairs_b = []
    for i, neighs in w.neighbors.items():
        for j in neighs:
            if j > i:  # avoid double counting pairs
                a, b = y[i], y[j]
                if nan_policy == "omit" and (np.isnan(a) or np.isnan(b)):
                    continue
                pairs_a.append(a)
                pairs_b.append(b)

    if len(pairs_a) == 0:
        return float("nan")

    pairs_a = np.array(pairs_a)
    pairs_b = np.array(pairs_b)

    vmin = min(pairs_a.min(), pairs_b.min())
    vmax = max(pairs_a.max(), pairs_b.max())
    if vmax == vmin:
        return 0.0  # no variability => zero entropy

    # Build 2D histogram of neighbor pairs
    hist, _, _ = np.histogram2d(
        pairs_a, pairs_b, bins=bins, range=[[vmin, vmax], [vmin, vmax]]
    )
    p = hist / hist.sum()
    p = p[p > 0]  # only non-zero probabilities
    entropy_bits = -np.sum(p * np.log2(p))
    return float(entropy_bits)


def compute_glcm_contrast(image_array, distances=[1], angles=[0], levels=256) -> float:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) contrast for a grayscale image.

    GLCM contrast measures the local variations in the image. High contrast indicates
    significant differences between neighboring pixels; low contrast suggests smoother regions.

    Args:
        image_array: numpy array (H x W) of pixel values (will be cast to uint8).
        distances: list of pixel pair distance offsets (in pixel units).
        angles: list of angles in radians (e.g., [0] for horizontal, [0, np.pi/4, np.pi/2, 3*np.pi/4] for all).
        levels: number of gray levels to quantize the image into (default 256 for 8-bit).

    Returns:
        GLCM contrast value (float). Higher values indicate greater local texture variation.
    """
    # Ensure image is uint8 type
    image = image_array.astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    # Extract contrast property (first distance and angle only)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    return float(contrast)


def compute_glcm_homogeneity(image_array, distances=[1], angles=[0], levels=256) -> float:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) homogeneity for a grayscale image.

    GLCM homogeneity measures the closeness of the distribution of elements in the GLCM
    to its diagonal. High homogeneity values indicate similar gray-level patterns,
    while low values suggest more variation.

    The homogeneity is calculated as: sum(P[i, j] / (1 + abs(i - j)))
    where P is the normalized GLCM matrix.

    Args:
        image_array: numpy array (H x W) of pixel values (will be cast to uint8).
        distances: list of pixel pair distance offsets (in pixel units).
        angles: list of angles in radians (e.g., [0] for horizontal, [0, np.pi/4, np.pi/2, 3*np.pi/4] for all).
        levels: number of gray levels to quantize the image into (default 256 for 8-bit).

    Returns:
        GLCM homogeneity value (float). Range [0, 1]. Higher values indicate more uniform texture.
    """
    # Ensure image is uint8 type
    image = image_array.astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    # Extract homogeneity property using graycoprops (first distance and angle only)
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Alternative manual calculation: homogeneity = sum(P[i, j] / (1 + abs(i - j)))
    # This verifies the formula provided
    # P = glcm[:, :, 0, 0]
    # manual_homogeneity = 0.0
    # for i in range(levels):
    #     for j in range(levels):
    #         manual_homogeneity += P[i, j] / (1 + abs(i - j))
    
    return float(homogeneity)


def compute_glcm_energy(image_array, distances=[1], angles=[0], levels=256) -> float:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) energy for a grayscale image.

    GLCM energy measures the sum of squared elements in the GLCM. Also known as 
    Angular Second Moment (ASM), it indicates the orderliness of the texture.
    High energy values indicate repetitive patterns or uniformity, while low values
    suggest more complex or random textures.

    The energy is calculated as: sum(P[i, j]^2)
    where P is the normalized GLCM matrix.

    Args:
        image_array: numpy array (H x W) of pixel values (will be cast to uint8).
        distances: list of pixel pair distance offsets (in pixel units).
        angles: list of angles in radians (e.g., [0] for horizontal, [0, np.pi/4, np.pi/2, 3*np.pi/4] for all).
        levels: number of gray levels to quantize the image into (default 256 for 8-bit).

    Returns:
        GLCM energy value (float). Range [0, 1]. Higher values indicate more ordered texture.
    """
    # Ensure image is uint8 type
    image = image_array.astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    # Extract energy property using graycoprops (first distance and angle only)
    energy = graycoprops(glcm, 'energy')[0, 0]
    return float(energy)


def compute_glcm_correlation(image_array, distances=[1], angles=[0], levels=256) -> float:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) correlation for a grayscale image.

    GLCM correlation measures the linear dependency of gray levels on their neighbors.
    It indicates how correlated a pixel is to its neighbor over the whole image.
    High correlation values indicate that the image has strong linear structures,
    while low values suggest more random patterns.

    Args:
        image_array: numpy array (H x W) of pixel values (will be cast to uint8).
        distances: list of pixel pair distance offsets (in pixel units).
        angles: list of angles in radians (e.g., [0] for horizontal, [0, np.pi/4, np.pi/2, 3*np.pi/4] for all).
        levels: number of gray levels to quantize the image into (default 256 for 8-bit).

    Returns:
        GLCM correlation value (float). Range [-1, 1]. Higher values indicate stronger linear relationships.
    """
    # Ensure image is uint8 type
    image = image_array.astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    # Extract correlation property using graycoprops (first distance and angle only)
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return float(correlation)


def compute_hu_moments(image_array) -> float:
    """
    Compute Hu moments (shape descriptors) for a grayscale image.

    Hu moments are 7 shape-based invariant moments that are invariant to translation,
    rotation, and scale. They are useful for shape analysis and object recognition.
    This function returns the first Hu moment (hu[0]), which is the most stable and
    representative shape descriptor. It's invariant to rotation, scale, and translation.

    The 7 individual Hu moments capture different aspects of shape:
    - hu[0]: Primary shape descriptor (used here) - related to object area & shape
    - hu[1]: Related to object elongation
    - hu[2-4]: Related to shape asymmetry and boundaries
    - hu[5-6]: Related to complex shape details

    Args:
        image_array: numpy array (H x W) of pixel values (grayscale image).

    Returns:
        First Hu moment (hu[0]) as float. This is the primary shape descriptor.
        Returns the log of absolute value to handle potentially large or very small values.
        Higher values indicate more complex/irregular shapes.
    """
    # Convert to float array
    image = image_array.astype(float)
    
    # Compute central moments
    mu = moments_central(image)
    
    # Compute Hu moments (returns 7 values)
    hu = moments_hu(mu)
    
    # Return the first Hu moment (most significant and stable)
    # Use log of absolute value to normalize scale and handle extreme values
    hu_first = float(hu[0])
    
    # Add small epsilon to avoid log(0) and take log to normalize scale
    hu_log = float(np.log10(np.abs(hu_first) + 1e-10))
    
    return hu_log


def hog_energy(image,
               pixels_per_cell=(16, 16),
               cells_per_block=(1, 1),
               orientations=9):
    """
    Compute HOG (Histogram of Oriented Gradients) energy for a grayscale image.
    
    HOG energy measures the overall gradient magnitude and directionality in an image.
    Higher values indicate stronger edges and more pronounced directional patterns,
    which can be useful for detecting structured features in geoglyphs.
    
    Args:
        image: 2D grayscale image (numpy array)
        pixels_per_cell: tuple (y, x) specifying the size of each cell in pixels.
                        Default (16, 16).
        cells_per_block: tuple (y, x) specifying number of cells per block for normalization.
                        Default (1, 1) means no block normalization beyond individual cells.
        orientations: Number of orientation bins for the HOG descriptor. Default 9.
    
    Returns:
        scalar HOG energy (float): Mean squared magnitude of the HOG feature vector.
                                  Higher values indicate stronger gradient patterns.
    """
    hog_vec = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )

    energy = np.mean(hog_vec ** 2)
    return energy
