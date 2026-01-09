import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_dilation, label


def calculate_spatial_autocorrelation(patch):
    """
    Calculate the spatial autocorrelation of an nxn pixel patch using Moran's I statistic.
    
    Moran's I measures the degree of spatial autocorrelation in the patch.
    Values range from -1 (perfect dispersion) to +1 (perfect correlation).
    A value near 0 indicates random spatial pattern.
    
    Parameters:
    -----------
    patch : numpy.ndarray
        An nxn array representing pixel values (grayscale or single channel)
    
    Returns:
    --------
    float
        Moran's I statistic
    """
    if patch.ndim != 2:
        raise ValueError("Patch must be a 2D array")
    
    n, m = patch.shape
    if n != m:
        raise ValueError("Patch must be square (nxn)")
    
    N = n * n  # Total number of pixels
    
    # Flatten the patch for easier computation
    values = patch.flatten()
    
    # Calculate mean
    mean_val = np.mean(values)
    
    # Calculate deviations from mean
    deviations = values - mean_val
    
    # Create spatial weights matrix (using queen contiguity: 8 neighbors)
    # Weight = 1 if pixels are neighbors (including diagonals), 0 otherwise
    weights = np.zeros((N, N))
    
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            
            # Check all 8 neighboring positions
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # Skip the pixel itself
                    
                    ni, nj = i + di, j + dj
                    
                    # Check if neighbor is within bounds
                    if 0 <= ni < n and 0 <= nj < m:
                        neighbor_idx = ni * m + nj
                        weights[idx, neighbor_idx] = 1
    
    # Calculate sum of all weights
    W = np.sum(weights)
    
    if W == 0:
        return 0.0  # No neighbors (shouldn't happen for n > 1)
    
    # Calculate Moran's I
    numerator = 0.0
    for i in range(N):
        for j in range(N):
            numerator += weights[i, j] * deviations[i] * deviations[j]
    
    denominator = np.sum(deviations ** 2)
    
    if denominator == 0:
        return 0.0  # All values are the same
    
    morans_i = (N / W) * (numerator / denominator)
    
    return morans_i


def calculate_spatial_autocorrelation_optimized(patch):
    """
    Optimized version of spatial autocorrelation calculation using vectorized operations.
    
    Parameters:
    -----------
    patch : numpy.ndarray
        An nxn array representing pixel values (grayscale or single channel)
    
    Returns:
    --------
    float
        Moran's I statistic
    """
    if patch.ndim != 2:
        raise ValueError("Patch must be a 2D array")
    
    n, m = patch.shape
    if n != m:
        raise ValueError("Patch must be square (nxn)")
    
    if n < 2:
        return 0.0  # Cannot calculate autocorrelation for 1x1 patch
    
    # Calculate mean and deviations
    mean_val = np.mean(patch)
    deviations = patch - mean_val
    
    # Calculate variance
    variance = np.sum(deviations ** 2)
    
    if variance == 0:
        return 0.0  # All values are the same
    
    # Calculate spatial covariance using neighbor differences
    # Sum over all valid neighbor pairs
    spatial_covariance = 0.0
    weight_sum = 0
    
    # Horizontal neighbors
    for i in range(n):
        for j in range(m - 1):
            spatial_covariance += deviations[i, j] * deviations[i, j + 1]
            weight_sum += 1
    
    # Vertical neighbors
    for i in range(n - 1):
        for j in range(m):
            spatial_covariance += deviations[i, j] * deviations[i + 1, j]
            weight_sum += 1
    
    # Diagonal neighbors (top-left to bottom-right)
    for i in range(n - 1):
        for j in range(m - 1):
            spatial_covariance += deviations[i, j] * deviations[i + 1, j + 1]
            weight_sum += 1
    
    # Diagonal neighbors (top-right to bottom-left)
    for i in range(n - 1):
        for j in range(1, m):
            spatial_covariance += deviations[i, j] * deviations[i + 1, j - 1]
            weight_sum += 1
    
    # Calculate Moran's I
    N = n * m
    morans_i = (N / weight_sum) * (spatial_covariance / variance)
    
    return morans_i


def resize_and_window(image, target_size, window_size):
    """
    Resizes an image to a square and yields successive non-overlapping square windows.
    
    Parameters:
    -----------
    image : PIL.Image or np.ndarray or str
        Input image (PIL Image object, numpy array, or file path)
    target_size : int
        Size to resize the image to (will be target_size x target_size)
    window_size : int
        Size of the square windows to extract
    
    Yields:
    -------
    tuple
        (window_array, row_idx, col_idx) where window_array is the extracted window
        and row_idx, col_idx are the top-left coordinates of the window
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to grayscale for spatial autocorrelation analysis
    image = image.convert('L')
    
    # Resize to square
    resized_image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(resized_image)
    
    # Calculate number of windows in each dimension
    height, width = img_array.shape
    
    # Slide window across the image (non-overlapping)
    for i in range(0, height - window_size + 1, window_size):
        for j in range(0, width - window_size + 1, window_size):
            # Extract window
            window = img_array[i:i+window_size, j:j+window_size]
            yield window, i, j


def morans_i_to_color(morans_i_value):
    """
    Convert a Moran's I value to RGB color.
    -1 (perfect dispersion) = full blue (0, 0, 255)
    0 (random) = black (0, 0, 0)
    +1 (perfect correlation) = full red (255, 0, 0)
    
    Parameters:
    -----------
    morans_i_value : float
        Moran's I value (typically in range [-1, 1])
    
    Returns:
    --------
    tuple
        (R, G, B) values in range [0, 255]
    """
    # Clip to [-1, 1] range
    morans_i_value = np.clip(morans_i_value, -1.0, 1.0)
    
    if morans_i_value > 0:
        # Positive correlation -> Red
        # Scale from 0 to 255 based on how close to +1
        red = int(morans_i_value * 255)
        return (red, 0, 0)
    elif morans_i_value < 0:
        # Negative correlation -> Blue
        # Scale from 0 to 255 based on how close to -1
        blue = int(abs(morans_i_value) * 255)
        return (0, 0, blue)
    else:
        # Zero correlation -> Black (no color)
        return (0, 0, 0)


def create_morans_i_red_mask(image, target_size, window_size, 
                             output_dir="output", save_individual=False,
                             min_morans=None, max_morans=None, red_intensity=0.7,
                             show_results=False):
    """
    Create a color mask visualization of spatial autocorrelation (Moran's I) across an image.
    Negative values (dispersion) shown in blue, positive values (clustering) in red.
    
    Parameters:
    -----------
    image : PIL.Image or np.ndarray or str
        Input image (PIL Image object, numpy array, or file path)
    target_size : int
        Size to resize the image to (will be target_size x target_size)
    window_size : int
        Size of the square windows to extract
    output_dir : str
        Directory to save output images (default: "output")
    save_individual : bool
        Whether to save individual window red masks (default: False)
    min_morans : float, optional
        Not used in new color scheme (kept for compatibility)
    max_morans : float, optional
        Not used in new color scheme (kept for compatibility)
    red_intensity : float, optional
        Intensity of red overlay (0.0 to 1.0). Higher = more red visible (default: 0.7)
    show_results : bool, optional
        If True, automatically display visualization with matplotlib (default: False)
    
    Returns:
    --------
    tuple
        (color_mask_array, morans_values, metadata) where:
        - color_mask_array is the RGB image array (target_size x target_size x 3)
        - morans_values is dict mapping (row, col) -> Moran's I value (with sign)
        - metadata contains min/max Moran's I and other info
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and resize image
    if isinstance(image, str):
        img = Image.open(image)
        image_name = Path(image).stem
    else:
        img = image if isinstance(image, Image.Image) else Image.fromarray(np.array(image))
        image_name = "image"
    
    # Create color mask (RGB image)
    color_mask = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Store Moran's I values (with sign)
    morans_values = {}
    morans_list = []
    
    print(f"\nProcessing image with {window_size}x{window_size} windows...")
    print(f"Target size: {target_size}x{target_size}")
    
    # Process each window
    window_count = 0
    for window, row, col in resize_and_window(img, target_size, window_size):
        window_count += 1
        
        # Calculate Moran's I for this window
        try:
            morans_i = calculate_spatial_autocorrelation_optimized(window)
            # Keep the sign (positive = clustering, negative = dispersion)
            morans_values[(row, col)] = morans_i
            morans_list.append(morans_i)
            
            if window_count % 10 == 0:
                print(f"  Processed {window_count} windows... (latest Moran's I: {morans_i:.4f})")
        except Exception as e:
            print(f"  Warning: Failed to compute Moran's I for window at ({row}, {col}): {e}")
            morans_i = 0.0
            morans_values[(row, col)] = morans_i
    
    print(f"Completed processing {window_count} windows")
    
    # Calculate min/max for reporting
    if morans_list:
        min_morans = min(morans_list)
        max_morans = max(morans_list)
        
        print(f"\nMoran's I range: [{min_morans:.4f}, {max_morans:.4f}]")
    else:
        min_morans = 0.0
        max_morans = 0.0
    
    # Create color mask and optionally save individual windows
    print("\nGenerating color mask...")
    for (row, col), morans_i in morans_values.items():
        # Convert Moran's I to color (blue for negative, red for positive)
        color = morans_i_to_color(morans_i)
        
        # Fill the window area with this color
        row_end = min(row + window_size, target_size)
        col_end = min(col + window_size, target_size)
        color_mask[row:row_end, col:col_end] = color
        
        # Optionally save individual window
        if save_individual:
            window_img = np.full((window_size, window_size, 3), color, dtype=np.uint8)
            window_pil = Image.fromarray(window_img)
            window_path = os.path.join(output_dir, f"window_{row}_{col}_morans_{morans_i:.4f}.png")
            window_pil.save(window_path)
    
    # Save the complete color mask
    color_mask_pil = Image.fromarray(color_mask)
    mask_path = os.path.join(output_dir, f"{image_name}_morans_mask.png")
    color_mask_pil.save(mask_path)
    print(f"Saved Moran's I color mask to: {mask_path}")
    
    # Create folder with image name for results
    image_folder = os.path.join(output_dir, image_name)
    Path(image_folder).mkdir(parents=True, exist_ok=True)
    
    # Create and save a combined visualization (original + mask overlay)
    original_resized = img.convert('RGB').resize((target_size, target_size), Image.Resampling.LANCZOS)
    original_array = np.array(original_resized)
    
    # Blend original with color mask
    red_intensity = np.clip(red_intensity, 0.0, 1.0)
    blended = (original_array * (1.0 - red_intensity) + color_mask * red_intensity).astype(np.uint8)
    blended_pil = Image.fromarray(blended)
    blended_path = os.path.join(output_dir, f"{image_name}_morans_overlay.png")
    blended_pil.save(blended_path)
    print(f"Saved overlay visualization to: {blended_path}")
    
    # Save copy in image-specific folder
    final_overlay_path = os.path.join(image_folder, f"{image_name}_with_morans.png")
    blended_pil.save(final_overlay_path)
    print(f"Saved final result to: {final_overlay_path}")
    
    # Also save the original and mask in the same folder for reference
    original_copy_path = os.path.join(image_folder, f"{image_name}_original.png")
    original_resized.save(original_copy_path)
    
    mask_copy_path = os.path.join(image_folder, f"{image_name}_morans_mask.png")
    color_mask_pil.save(mask_copy_path)
    
    print(f"All files saved to folder: {image_folder}")
    
    # Metadata
    metadata = {
        "image_name": image_name,
        "min_morans": min_morans,
        "max_morans": max_morans,
        "num_windows": len(morans_values),
        "window_size": window_size,
        "target_size": target_size,
        "mean_morans": np.mean(morans_list) if morans_list else 0.0,
        "std_morans": np.std(morans_list) if morans_list else 0.0,
        "red_intensity": red_intensity,
        "output_folder": image_folder
    }
    
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Optionally show visualization
    if show_results:
        print("\nDisplaying visualization...")
        visualize_morans_results(
            image_name=image_name,
            output_dir=output_dir,
            red_mask=color_mask,
            original_array=original_array,
            blended_array=blended
        )
    
    return color_mask, morans_values, metadata


def visualize_morans_results(image_path=None, image_name=None, output_dir="output", 
                             red_mask=None, original_array=None, blended_array=None):
    """
    Display the original image, Moran's I mask, and combined overlay using matplotlib.
    
    Parameters:
    -----------
    image_path : str, optional
        Path to the original image file. If provided, will load from this path.
    image_name : str, optional
        Name of the image (without extension). Used to locate saved files.
    output_dir : str
        Directory where the Moran's I mask files were saved (default: "output")
    red_mask : np.ndarray, optional
        Red mask array. If None, will load from saved file.
    original_array : np.ndarray, optional
        Original image array. If None, will load from saved file.
    blended_array : np.ndarray, optional
        Blended overlay array. If None, will load from saved file.
    """
    # Determine image name
    if image_name is None and image_path is not None:
        image_name = Path(image_path).stem
    elif image_name is None:
        raise ValueError("Either image_path or image_name must be provided")
    
    # Load images if not provided
    image_folder = os.path.join(output_dir, image_name)
    
    if original_array is None:
        original_path = os.path.join(image_folder, f"{image_name}_original.png")
        if os.path.exists(original_path):
            original_array = np.array(Image.open(original_path))
        elif image_path and os.path.exists(image_path):
            # Load and use original image
            img = Image.open(image_path).convert('RGB')
            original_array = np.array(img)
        else:
            raise FileNotFoundError(f"Could not find original image at {original_path}")
    
    if red_mask is None:
        mask_path = os.path.join(image_folder, f"{image_name}_morans_mask.png")
        if os.path.exists(mask_path):
            red_mask = np.array(Image.open(mask_path))
        else:
            raise FileNotFoundError(f"Could not find Moran's I mask at {mask_path}")
    
    if blended_array is None:
        blended_path = os.path.join(image_folder, f"{image_name}_with_morans.png")
        if os.path.exists(blended_path):
            blended_array = np.array(Image.open(blended_path))
        else:
            raise FileNotFoundError(f"Could not find blended image at {blended_path}")
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_array)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Moran's I mask (blue to red)
    axes[1].imshow(red_mask)
    axes[1].set_title("Spatial Autocorrelation Mask (Moran's I)\nBlue=Dispersion, Red=Clustering", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Combined overlay
    axes[2].imshow(blended_array)
    axes[2].set_title("Image + Moran's I Overlay", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f"Spatial Autocorrelation Analysis: {image_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig, axes


# ============================================================================
# GEOGLYPH DETECTION PIPELINE
# ============================================================================

def preprocess_image(image_array, normalize=True, smooth=True, sigma=1.0):
    """
    Step 1: Preprocessing - normalize intensity and optional smoothing.
    
    Parameters:
    -----------
    image_array : np.ndarray
        Grayscale image array (2D)
    normalize : bool
        Whether to normalize intensity to [0, 1] range
    smooth : bool
        Whether to apply Gaussian smoothing
    sigma : float
        Standard deviation for Gaussian kernel (default: 1.0)
    
    Returns:
    --------
    np.ndarray
        Preprocessed image array
    """
    processed = image_array.astype(np.float64)
    
    # Normalize to [0, 1]
    if normalize:
        min_val = processed.min()
        max_val = processed.max()
        if max_val > min_val:
            processed = (processed - min_val) / (max_val - min_val)
    
    # Apply Gaussian smoothing
    if smooth:
        processed = gaussian_filter(processed, sigma=sigma)
    
    return processed


def extract_edges_sobel(image_array):
    """
    Step 2: Edge extraction using Sobel operator.
    
    Parameters:
    -----------
    image_array : np.ndarray
        Preprocessed grayscale image (2D)
    
    Returns:
    --------
    np.ndarray
        Edge magnitude map (same size as input)
    """
    # Sobel operators for horizontal and vertical edges
    sobel_x = ndimage.sobel(image_array, axis=1)
    sobel_y = ndimage.sobel(image_array, axis=0)
    
    # Compute edge magnitude
    edge_magnitude = np.hypot(sobel_x, sobel_y)
    
    return edge_magnitude


def extract_edges_canny(image_array, low_threshold=0.1, high_threshold=0.2):
    """
    Step 2: Edge extraction using Canny edge detector (simplified).
    
    Parameters:
    -----------
    image_array : np.ndarray
        Preprocessed grayscale image (2D)
    low_threshold : float
        Lower threshold for edge detection (0-1)
    high_threshold : float
        Upper threshold for edge detection (0-1)
    
    Returns:
    --------
    np.ndarray
        Binary edge map
    """
    # Compute gradients
    gradient = extract_edges_sobel(image_array)
    
    # Normalize gradient
    if gradient.max() > 0:
        gradient = gradient / gradient.max()
    
    # Apply thresholding
    edges = (gradient > high_threshold).astype(np.uint8)
    
    return edges


def calculate_edge_density(window, edge_map_window):
    """
    Calculate edge density for a window.
    
    Parameters:
    -----------
    window : np.ndarray
        Original image window (2D)
    edge_map_window : np.ndarray
        Corresponding edge map window (2D)
    
    Returns:
    --------
    float
        Edge density (ratio of edge pixels to total pixels)
    """
    total_pixels = edge_map_window.size
    edge_pixels = np.sum(edge_map_window > 0)
    
    return edge_pixels / total_pixels if total_pixels > 0 else 0.0


def calculate_morans_i_multiscale(window, scale_small=1, scale_large=3):
    """
    Step 3: Calculate Moran's I at two spatial scales.
    
    Parameters:
    -----------
    window : np.ndarray
        Image window (2D, square)
    scale_small : int
        Neighborhood distance for small scale (default: 1 = immediate neighbors)
    scale_large : int
        Neighborhood distance for large scale (default: 3)
    
    Returns:
    --------
    tuple
        (I_small, I_large) - Moran's I at small and large scales
    """
    n, m = window.shape
    if n != m or n < scale_large * 2:
        return 0.0, 0.0
    
    # For simplicity, use the optimized version for small scale
    I_small = calculate_spatial_autocorrelation_optimized(window)
    
    # For large scale, we need a modified version that considers larger neighborhoods
    # Simplified approach: subsample and calculate
    step = max(1, scale_large // scale_small)
    if step > 1:
        subsampled = window[::step, ::step]
        if subsampled.shape[0] >= 2 and subsampled.shape[1] >= 2:
            # Make it square
            min_dim = min(subsampled.shape)
            subsampled = subsampled[:min_dim, :min_dim]
            I_large = calculate_spatial_autocorrelation_optimized(subsampled)
        else:
            I_large = I_small
    else:
        I_large = I_small
    
    return I_small, I_large


def detect_geoglyphs_sliding_window(image, window_size=64, stride=None,
                                   threshold_I=0.3, threshold_delta=0.1, threshold_E=0.15,
                                   preprocess_params=None, edge_method='sobel'):
    """
    Step 4: Sliding window analysis with decision rule.
    
    Parameters:
    -----------
    image : np.ndarray or str
        Grayscale image array or path to image
    window_size : int
        Size of sliding window (default: 64)
    stride : int, optional
        Stride for sliding window. If None, uses window_size // 2 (50% overlap)
    threshold_I : float
        Threshold for I_small (default: 0.3)
    threshold_delta : float
        Threshold for (I_small - I_large) (default: 0.1)
    threshold_E : float
        Threshold for edge density (default: 0.15)
    preprocess_params : dict, optional
        Parameters for preprocessing (normalize, smooth, sigma)
    edge_method : str
        Edge extraction method: 'sobel' or 'canny' (default: 'sobel')
    
    Returns:
    --------
    tuple
        (detection_map, candidate_windows, edge_map, metadata)
    """
    # Load image if path provided
    if isinstance(image, str):
        img = Image.open(image).convert('L')
        image_array = np.array(img)
    else:
        image_array = image
    
    # Set default preprocessing parameters
    if preprocess_params is None:
        preprocess_params = {'normalize': True, 'smooth': True, 'sigma': 1.0}
    
    # Step 1: Preprocessing
    print("Step 1/4: Preprocessing image...")
    processed = preprocess_image(image_array, **preprocess_params)
    
    # Step 2: Edge extraction
    print("Step 2/4: Extracting edges...")
    if edge_method == 'canny':
        edge_map = extract_edges_canny(processed)
    else:
        edge_map = extract_edges_sobel(processed)
        # Threshold for binary edge map
        threshold = 0.2 * edge_map.max() if edge_map.max() > 0 else 0
        edge_map = (edge_map > threshold).astype(np.uint8)
    
    # Step 3: Sliding window analysis
    print("Step 3/4: Analyzing windows...")
    height, width = processed.shape
    if stride is None:
        stride = window_size // 2
    
    detection_map = np.zeros((height, width), dtype=np.float32)
    candidate_windows = []
    
    window_count = 0
    detected_count = 0
    
    for row in range(0, height - window_size + 1, stride):
        for col in range(0, width - window_size + 1, stride):
            window_count += 1
            
            # Extract windows
            window = processed[row:row+window_size, col:col+window_size]
            edge_window = edge_map[row:row+window_size, col:col+window_size]
            
            # Calculate features
            I_small, I_large = calculate_morans_i_multiscale(window)
            E = calculate_edge_density(window, edge_window)
            delta_I = I_small - I_large
            
            # Decision rule
            is_candidate = (I_small > threshold_I and 
                          delta_I > threshold_delta and 
                          E > threshold_E)
            
            if is_candidate:
                detected_count += 1
                # Store confidence score (sum of normalized features)
                confidence = (I_small + delta_I + E) / 3.0
                detection_map[row:row+window_size, col:col+window_size] += confidence
                
                candidate_windows.append({
                    'row': row,
                    'col': col,
                    'size': window_size,
                    'I_small': I_small,
                    'I_large': I_large,
                    'delta_I': delta_I,
                    'edge_density': E,
                    'confidence': confidence
                })
            
            if window_count % 100 == 0:
                print(f"  Processed {window_count} windows... ({detected_count} candidates)")
    
    print(f"Completed: {window_count} windows analyzed, {detected_count} candidates found")
    
    metadata = {
        'window_size': window_size,
        'stride': stride,
        'threshold_I': threshold_I,
        'threshold_delta': threshold_delta,
        'threshold_E': threshold_E,
        'total_windows': window_count,
        'detected_windows': detected_count,
        'image_shape': image_array.shape
    }
    
    return detection_map, candidate_windows, edge_map, metadata


def postprocess_detections(detection_map, min_area=100, dilation_size=3):
    """
    Step 5: Post-processing - merge adjacent detections and remove small ones.
    
    Parameters:
    -----------
    detection_map : np.ndarray
        Detection confidence map (2D)
    min_area : int
        Minimum area (in pixels) for a valid detection (default: 100)
    dilation_size : int
        Size of morphological dilation for merging (default: 3)
    
    Returns:
    --------
    tuple
        (binary_mask, labeled_regions, num_detections)
    """
    # Threshold detection map to binary
    threshold = 0.1 * detection_map.max() if detection_map.max() > 0 else 0
    binary_mask = (detection_map > threshold).astype(np.uint8)
    
    # Morphological closing to merge nearby detections
    if dilation_size > 0:
        structure = np.ones((dilation_size, dilation_size))
        binary_mask = binary_dilation(binary_mask, structure=structure)
        binary_mask = binary_dilation(binary_mask, structure=structure)  # Dilation twice for closing effect
    
    # Label connected components
    labeled_regions, num_features = label(binary_mask)
    
    # Remove small detections
    for region_id in range(1, num_features + 1):
        region_mask = (labeled_regions == region_id)
        area = np.sum(region_mask)
        
        if area < min_area:
            labeled_regions[region_mask] = 0
    
    # Relabel after removing small regions
    labeled_regions, num_detections = label(labeled_regions > 0)
    binary_mask = (labeled_regions > 0).astype(np.uint8)
    
    return binary_mask, labeled_regions, num_detections


def run_geoglyph_detection_pipeline(image_path, window_size=64, stride=None,
                                   threshold_I=0.3, threshold_delta=0.1, threshold_E=0.15,
                                   min_area=100, output_dir="output/geoglyph_detection",
                                   visualize=True):
    """
    Complete geoglyph detection pipeline.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    window_size : int
        Size of sliding window (default: 64)
    stride : int, optional
        Stride for sliding window (default: window_size // 2)
    threshold_I : float
        Threshold for I_small (default: 0.3)
    threshold_delta : float
        Threshold for (I_small - I_large) (default: 0.1)
    threshold_E : float
        Threshold for edge density (default: 0.15)
    min_area : int
        Minimum area for valid detection (default: 100)
    output_dir : str
        Directory to save results
    visualize : bool
        Whether to display results
    
    Returns:
    --------
    dict
        Results containing detection mask, candidates, and metadata
    """
    print("=" * 70)
    print("GEOGLYPH DETECTION PIPELINE")
    print("=" * 70)
    print(f"\nInput image: {image_path}")
    print(f"Window size: {window_size}, Stride: {stride or window_size // 2}")
    print(f"Thresholds: I={threshold_I}, ΔI={threshold_delta}, E={threshold_E}")
    
    # Run detection
    detection_map, candidates, edge_map, metadata = detect_geoglyphs_sliding_window(
        image=image_path,
        window_size=window_size,
        stride=stride,
        threshold_I=threshold_I,
        threshold_delta=threshold_delta,
        threshold_E=threshold_E
    )
    
    # Post-processing
    print("\nStep 4/4: Post-processing detections...")
    binary_mask, labeled_regions, num_detections = postprocess_detections(
        detection_map, min_area=min_area
    )
    
    print(f"\nFinal detections: {num_detections} regions")
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_name = Path(image_path).stem
    
    # Save detection map
    detection_map_normalized = (detection_map / detection_map.max() * 255).astype(np.uint8) if detection_map.max() > 0 else detection_map.astype(np.uint8)
    Image.fromarray(detection_map_normalized).save(os.path.join(output_dir, f"{image_name}_detection_map.png"))
    
    # Save binary mask
    Image.fromarray(binary_mask * 255).save(os.path.join(output_dir, f"{image_name}_binary_mask.png"))
    
    # Save edge map
    Image.fromarray(edge_map * 255).save(os.path.join(output_dir, f"{image_name}_edges.png"))
    
    print(f"\nResults saved to: {output_dir}")
    
    # Visualize if requested
    if visualize:
        visualize_geoglyph_detection(image_path, detection_map, binary_mask, edge_map, labeled_regions)
    
    return {
        'detection_map': detection_map,
        'binary_mask': binary_mask,
        'labeled_regions': labeled_regions,
        'candidates': candidates,
        'edge_map': edge_map,
        'num_detections': num_detections,
        'metadata': metadata
    }


def visualize_geoglyph_detection(image_path, detection_map, binary_mask, edge_map, labeled_regions):
    """
    Visualize geoglyph detection results.
    """
    # Load original image
    original = np.array(Image.open(image_path).convert('L'))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Edge map
    axes[0, 1].imshow(edge_map, cmap='gray')
    axes[0, 1].set_title('Edge Map', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Detection map (heatmap)
    im = axes[0, 2].imshow(detection_map, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('Detection Confidence Map', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Binary mask
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title('Binary Detection Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Labeled regions
    axes[1, 1].imshow(labeled_regions, cmap='nipy_spectral')
    axes[1, 1].set_title(f'Labeled Regions ({labeled_regions.max()} detections)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay on original
    overlay = np.stack([original] * 3, axis=-1)
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[binary_mask > 0] = [255, 0, 0]  # Red overlay
    blended = (overlay * 0.7 + mask_rgb * 0.3).astype(np.uint8)
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title('Detections Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Geoglyph Detection Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    img_name = "Test2.jpg"
    
    # Choose mode: 'visualization' or 'detection'
    mode = 'detection'  # Change to 'visualization' for Moran's I visualization
    
    if mode == 'detection':
        print("=" * 70)
        print("GEOGLYPH DETECTION MODE")
        print("=" * 70)
        
        # Check if image exists
        if not os.path.exists(img_name):
            print(f"\nError: Image '{img_name}' not found!")
            print("Please place an image file in the current directory or update img_name.")
        else:
            # Run complete geoglyph detection pipeline
            results = run_geoglyph_detection_pipeline(
                image_path=img_name,
                window_size=256,
                stride=256,
                threshold_I=0.3,      # Moran's I threshold
                threshold_delta=0.1,  # Multi-scale difference threshold
                threshold_E=0.15,     # Edge density threshold
                min_area=100,         # Minimum detection area (pixels)
                output_dir="output/geoglyph_detection",
                visualize=True
            )
            
            print("\n" + "=" * 70)
            print(f"DETECTION COMPLETE: {results['num_detections']} geoglyphs found")
            print("=" * 70)
            
            # Print top candidates
            print("\nTop 5 candidates by confidence:")
            candidates = sorted(results['candidates'], key=lambda x: x['confidence'], reverse=True)
            for i, cand in enumerate(candidates[:5], 1):
                print(f"\n{i}. Position: ({cand['row']}, {cand['col']})")
                print(f"   Confidence: {cand['confidence']:.3f}")
                print(f"   Moran's I (small): {cand['I_small']:.3f}")
                print(f"   ΔI (small-large): {cand['delta_I']:.3f}")
                print(f"   Edge density: {cand['edge_density']:.3f}")
    
    else:  # visualization mode
        print("=" * 70)
        print("SPATIAL AUTOCORRELATION (MORAN'S I) VISUALIZATION MODE")
        print("=" * 70)
        
        # Check if image exists
        if not os.path.exists(img_name):
            print(f"\nError: Image '{img_name}' not found!")
            print("Please place an image file in the current directory or update img_name.")
        else:
            print(f"\nProcessing image: {img_name}")
            
            # Generate Moran's I mask
            print("\n[1/3] Generating spatial autocorrelation mask...")
            red_mask, morans_vals, metadata = create_morans_i_red_mask(
                image=img_name,
                target_size=1024,
                window_size=128,
                output_dir="output/test_results",
                save_individual=False,
                red_intensity=0.5,
                show_results=False,
                min_morans=None,
                max_morans=None
            )
            
            print("\n[2/3] Files saved successfully!")
            print(f"  Output folder: {metadata['output_folder']}")
            print(f"  Number of windows analyzed: {metadata['num_windows']}")
            print(f"  Moran's I range: [{metadata['min_morans']:.4f}, {metadata['max_morans']:.4f}]")
            print(f"  Mean Moran's I: {metadata['mean_morans']:.4f}")
            print(f"  Std Moran's I: {metadata['std_morans']:.4f}")
            
            # Visualize results
            print("\n[3/3] Displaying visualization...")
            visualize_morans_results(
                image_path=img_name,
                output_dir="output/test_results"
            )
            
            print("\n" + "=" * 70)
            print("VISUALIZATION COMPLETED!")
            print("=" * 70)
