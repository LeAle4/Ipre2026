import numpy as np
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt
from main import (compute_sample_entropy, compute_approximate_entropy, 
                  compute_fuzzy_entropy, compute_spectral_entropy)


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
    
    # Resize to square
    resized_image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(resized_image)
    
    # Calculate number of windows in each dimension
    height, width = img_array.shape[:2]
    
    # Slide window across the image (non-overlapping)
    for i in range(0, height - window_size + 1, window_size):
        for j in range(0, width - window_size + 1, window_size):
            # Extract window
            window = img_array[i:i+window_size, j:j+window_size]
            yield window, i, j


def resize_and_window_list(image, target_size, window_size):
    """
    Resizes an image to a square and returns all non-overlapping square windows as a list.
    
    Parameters:
    -----------
    image : PIL.Image or np.ndarray or str
        Input image (PIL Image object, numpy array, or file path)
    target_size : int
        Size to resize the image to (will be target_size x target_size)
    window_size : int
        Size of the square windows to extract
    
    Returns:
    --------
    list of tuples
        List of (window_array, row_idx, col_idx) tuples
    """
    return list(resize_and_window(image, target_size, window_size))


def entropy_to_red_shade(entropy_value, min_entropy=0.0, max_entropy=3.0):
    """
    Convert an entropy value to a shade of red (RGB).
    Low entropy = bright red, High entropy = black
    
    Parameters:
    -----------
    entropy_value : float
        Entropy value to convert
    min_entropy : float
        Minimum entropy value for normalization (default: 0.0)
    max_entropy : float
        Maximum entropy value for normalization (default: 3.0)
    
    Returns:
    --------
    tuple
        (R, G, B) values in range [0, 255]
    """
    # Normalize entropy to [0, 1]
    normalized = np.clip((entropy_value - min_entropy) / (max_entropy - min_entropy), 0.0, 1.0)
    
    # Invert: low entropy = bright red (1 -> 255), high entropy = black (0 -> 0)
    red = int((1.0 - normalized) * 255)
    return (red, 0, 0)


def create_entropy_red_mask(image, target_size, window_size, entropy_func, 
                            entropy_params=None, 
                            output_dir="output", save_individual=False,
                            min_entropy=None, max_entropy=None, red_intensity=0.7,
                            show_results=False):
    """
    Create a red mask visualization of entropy values across an image.
    
    Parameters:
    -----------
    image : PIL.Image or np.ndarray or str
        Input image (PIL Image object, numpy array, or file path)
    target_size : int
        Size to resize the image to (will be target_size x target_size)
    window_size : int
        Size of the square windows to extract
    entropy_func : callable
        Entropy calculation function (e.g., compute_sample_entropy from main.py)
        Should accept a matrix and optional parameters, return (value, info) tuple
    entropy_params : dict, optional
        Parameters to pass to entropy_func (default: None)
    output_dir : str
        Directory to save output images (default: "output")
    save_individual : bool
        Whether to save individual window red masks (default: False)
    min_entropy : float, optional
        Minimum entropy for normalization. If None, auto-calculated from data
    max_entropy : float, optional
        Maximum entropy for normalization. If None, auto-calculated from data
    red_intensity : float, optional
        Intensity of red overlay (0.0 to 1.0). Higher = more red visible (default: 0.7)
    show_results : bool, optional
        If True, automatically display visualization with matplotlib (default: False)
    
    Returns:
    --------
    tuple
        (red_mask_array, entropy_values, metadata) where:
        - red_mask_array is the RGB image array (target_size x target_size x 3)
        - entropy_values is dict mapping (row, col) -> entropy value
        - metadata contains min/max entropy and other info
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set default entropy parameters
    if entropy_params is None:
        entropy_params = {}
    
    # Load and resize image
    if isinstance(image, str):
        img = Image.open(image)
        image_name = Path(image).stem
    else:
        img = image if isinstance(image, Image.Image) else Image.fromarray(np.array(image))
        image_name = "image"
    
    # Create red mask (RGB image)
    red_mask = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Store entropy values
    entropy_values = {}
    entropy_list = []
    
    print(f"\nProcessing image with {window_size}x{window_size} windows...")
    print(f"Target size: {target_size}x{target_size}")
    
    # Process each window
    window_count = 0
    for window, row, col in resize_and_window(img, target_size, window_size):
        window_count += 1
        
        # Calculate entropy for this window
        try:
            entropy_val, info = entropy_func(window, **entropy_params)
            entropy_values[(row, col)] = entropy_val
            entropy_list.append(entropy_val)
            
            if window_count % 10 == 0:
                print(f"  Processed {window_count} windows... (latest entropy: {entropy_val:.4f})")
        except Exception as e:
            print(f"  Warning: Failed to compute entropy for window at ({row}, {col}): {e}")
            entropy_val = 0.0
            entropy_values[(row, col)] = entropy_val
    
    print(f"Completed processing {window_count} windows")
    
    # Calculate min/max for normalization if not provided
    if entropy_list:
        if min_entropy is None:
            min_entropy = min(entropy_list)
        if max_entropy is None:
            max_entropy = max(entropy_list)
        
        print(f"\nEntropy range: [{min_entropy:.4f}, {max_entropy:.4f}]")
    else:
        min_entropy = 0.0
        max_entropy = 1.0
    
    # Create red mask and optionally save individual windows
    print("\nGenerating red mask...")
    for (row, col), entropy_val in entropy_values.items():
        # Convert entropy to red shade
        red_color = entropy_to_red_shade(entropy_val, min_entropy, max_entropy)
        
        # Fill the window area with this red shade
        row_end = min(row + window_size, target_size)
        col_end = min(col + window_size, target_size)
        red_mask[row:row_end, col:col_end] = red_color
        
        # Optionally save individual window
        if save_individual:
            window_img = np.full((window_size, window_size, 3), red_color, dtype=np.uint8)
            window_pil = Image.fromarray(window_img)
            window_path = os.path.join(output_dir, f"window_{row}_{col}_entropy_{entropy_val:.4f}.png")
            window_pil.save(window_path)
    
    # Save the complete red mask
    red_mask_pil = Image.fromarray(red_mask)
    mask_path = os.path.join(output_dir, f"{image_name}_entropy_mask.png")
    red_mask_pil.save(mask_path)
    print(f"Saved red entropy mask to: {mask_path}")
    
    # Create folder with image name for results
    image_folder = os.path.join(output_dir, image_name)
    Path(image_folder).mkdir(parents=True, exist_ok=True)
    
    # Create and save a combined visualization (original + mask overlay)
    original_resized = img.convert('RGB').resize((target_size, target_size), Image.Resampling.LANCZOS)
    original_array = np.array(original_resized)
    
    # Blend original with red mask - make red noticeable (configurable intensity)
    red_intensity = np.clip(red_intensity, 0.0, 1.0)
    blended = (original_array * (1.0 - red_intensity) + red_mask * red_intensity).astype(np.uint8)
    blended_pil = Image.fromarray(blended)
    blended_path = os.path.join(output_dir, f"{image_name}_entropy_overlay.png")
    blended_pil.save(blended_path)
    print(f"Saved overlay visualization to: {blended_path}")
    
    # Save copy with noticeable red overlay in image-specific folder
    final_overlay_path = os.path.join(image_folder, f"{image_name}_with_entropy.png")
    blended_pil.save(final_overlay_path)
    print(f"Saved final result to: {final_overlay_path}")
    
    # Also save the original and mask in the same folder for reference
    original_copy_path = os.path.join(image_folder, f"{image_name}_original.png")
    original_resized.save(original_copy_path)
    
    mask_copy_path = os.path.join(image_folder, f"{image_name}_entropy_mask.png")
    red_mask_pil.save(mask_copy_path)
    
    print(f"All files saved to folder: {image_folder}")
    
    # Metadata
    metadata = {
        "image_name": image_name,
        "min_entropy": min_entropy,
        "max_entropy": max_entropy,
        "num_windows": len(entropy_values),
        "window_size": window_size,
        "target_size": target_size,
        "mean_entropy": np.mean(entropy_list) if entropy_list else 0.0,
        "std_entropy": np.std(entropy_list) if entropy_list else 0.0,
        "red_intensity": red_intensity,
        "output_folder": image_folder
    }
    
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Optionally show visualization
    if show_results:
        print("\nDisplaying visualization...")
        visualize_entropy_results(
            image_name=image_name,
            output_dir=output_dir,
            red_mask=red_mask,
            original_array=original_array,
            blended_array=blended
        )
    
    return red_mask, entropy_values, metadata


def visualize_entropy_results(image_path=None, image_name=None, output_dir="output", 
                              red_mask=None, original_array=None, blended_array=None):
    """
    Display the original image, entropy mask, and combined overlay using matplotlib.
    
    Parameters:
    -----------
    image_path : str, optional
        Path to the original image file. If provided, will load from this path.
    image_name : str, optional
        Name of the image (without extension). Used to locate saved files.
    output_dir : str
        Directory where the entropy mask files were saved (default: "output")
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
        mask_path = os.path.join(image_folder, f"{image_name}_entropy_mask.png")
        if os.path.exists(mask_path):
            red_mask = np.array(Image.open(mask_path))
        else:
            raise FileNotFoundError(f"Could not find entropy mask at {mask_path}")
    
    if blended_array is None:
        blended_path = os.path.join(image_folder, f"{image_name}_with_entropy.png")
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
    
    # Entropy mask (red only)
    axes[1].imshow(red_mask)
    axes[1].set_title('Entropy Mask (Red)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Combined overlay
    axes[2].imshow(blended_array)
    axes[2].set_title('Image + Entropy Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'Entropy Analysis: {image_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig, axes


if __name__ == "__main__":
    """
    Test script: Load an image, generate entropy mask, save and visualize results.
    """
    img_name = "Test2.jpg"
    
    print("=" * 70)
    print("ENTROPY MASK GENERATION TEST")
    print("=" * 70)
    
    # Check if image exists
    if not os.path.exists(img_name):
        print(f"\nError: Image '{img_name}' not found!")
        print("Please place an image file in the current directory or update img_name.")
    else:
        print(f"\nProcessing image: {img_name}")
        
        # Generate entropy mask using Sample Entropy
        print("\n[1/3] Generating entropy mask...")
        red_mask, entropy_vals, metadata = create_entropy_red_mask(
            image=img_name,
            target_size=1024,
            window_size=16,
            entropy_func=compute_sample_entropy,
            entropy_params={'m': 2, 'r': 0.2},
            output_dir="output/test_results",
            save_individual=False,
            red_intensity=0.3,
            show_results=False,  # We'll show manually after
            min_entropy=0,
            max_entropy=3.0
        )
        
        print("\n[2/3] Files saved successfully!")
        print(f"  Output folder: {metadata['output_folder']}")
        print(f"  Number of windows analyzed: {metadata['num_windows']}")
        print(f"  Entropy range: [{metadata['min_entropy']:.4f}, {metadata['max_entropy']:.4f}]")
        print(f"  Mean entropy: {metadata['mean_entropy']:.4f}")
        
        # Visualize results
        print("\n[3/3] Displaying visualization...")
        visualize_entropy_results(
            image_path=img_name,
            output_dir="output/test_results"
        )
        
        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
    