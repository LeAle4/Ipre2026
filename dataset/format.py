import shapely
from pathlib import Path
import numpy as np
import cv2

from handle import (
    BASE_DIR,
    PROJECT_DIR,
    DATA_DIR,
    LLUTA_GEOS_DIR,
    UNITA_GEOS_DIR,
    CHUG_GEOS_DIR
)

TEST_DIR = BASE_DIR / "test_geos"
OUTPUT_DIR = BASE_DIR / "crops_output"
RAND_CROPS_DIR = OUTPUT_DIR / "random_crops"
FIXED_CROPS_DIR = OUTPUT_DIR / "fixed_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fill_with_noise(image, mask, noise_level=0.1, noise_type='gaussian', seed=None):
    """
    Fill out-of-bounds regions in a cropped image with noise.
    
    Args:
        image: Input image (numpy array, H x W or H x W x C)
        mask: Binary mask where 1 indicates valid pixels, 0 indicates out-of-bounds (H x W)
        noise_level: Controls randomness (0-1). 0 = average color, 1 = pure random noise
        noise_type: Type of noise - 'gaussian', 'uniform', or 'perlin'
        seed: Random seed for reproducibility
    
    Returns:
        Image with out-of-bounds regions filled with noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    result = image.copy().astype(np.float32)
    
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    invalid_mask = (1 - mask).astype(bool)
    
    if not np.any(invalid_mask):
        return result.astype(image.dtype)
    
    # Calculate average color of valid pixels
    if len(result.shape) == 3:
        avg_color = np.zeros(result.shape[2])
        for c in range(result.shape[2]):
            channel_values = result[:, :, c][~invalid_mask]
            if len(channel_values) > 0:
                avg_color[c] = np.mean(channel_values)
            else:
                avg_color[c] = 128
    else:
        valid_values = result[~invalid_mask]
        avg_color = np.mean(valid_values) if len(valid_values) > 0 else 128
    
    # Generate noise based on type
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 50, result.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-255, 255, result.shape)
    elif noise_type == 'perlin':
        # Simple pseudo-Perlin-like noise using gaussian blur
        noise = np.random.normal(0, 100, result.shape)
        if len(noise.shape) == 3:
            for c in range(noise.shape[2]):
                noise[:, :, c] = cv2.GaussianBlur(noise[:, :, c], (5, 5), 0)
        else:
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Blend noise: 0 = average color, 1 = full random noise
    blended_noise = avg_color * (1 - noise_level) + noise * noise_level
    
    # Fill invalid regions
    result[invalid_mask] = blended_noise[invalid_mask]
    
    # Clip values to valid range
    result = np.clip(result, 0, 255)
    
    return result.astype(image.dtype)

def crop_image(image, center, window_size):
    """
    Crop an image around a specified center with given window size.
    
    Args:
        image: Input image (numpy array, H x W or H x W x C)
        center: Tuple (x, y) for the center of the crop
        window_size: Size of the crop (int)
    
    Returns:
        Cropped image and binary mask indicating valid pixels
    """
    h, w = image.shape[:2]
    half_size = window_size // 2
    x_center, y_center = center
    
    x_start = x_center - half_size
    y_start = y_center - half_size
    x_end = x_start + window_size
    y_end = y_start + window_size
    
    crop = np.zeros((window_size, window_size) + image.shape[2:], dtype=image.dtype)
    mask = np.zeros((window_size, window_size), dtype=np.uint8)
    
    x_start_img = max(0, x_start)
    y_start_img = max(0, y_start)
    x_end_img = min(w, x_end)
    y_end_img = min(h, y_end)
    
    crop_x_start = x_start_img - x_start
    crop_y_start = y_start_img - y_start
    crop_x_end = crop_x_start + (x_end_img - x_start_img)
    crop_y_end = crop_y_start + (y_end_img - y_start_img)
    
    crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = image[y_start_img:y_end_img, x_start_img:x_end_img]
    mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = 1
    
    return crop, mask

def make_random_crops(image, window_size, n_crops, seed=None):
    """
    Generate random crops from an image.
    
    Args:
        image: Input image (numpy array, H x W or H x W x C)
        window_size: Size of each crop (int)
        n_crops: Number of random crops to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of tuples (crop, mask) where crop is the cropped image and mask indicates valid pixels
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = image.shape[:2]
    crops = []
    
    for _ in range(n_crops):
        # Generate random center coordinates
        x_center = np.random.randint(0, w)
        y_center = np.random.randint(0, h)
        
        crop, mask = crop_image(image, (x_center, y_center), window_size)
        crops.append((crop, mask))
    
    return crops

def make_fixed_crops(window_size, n_crops, stride):
    pass

def make_polygon_thresholds_crops(window_size, n_crops, stride, threshold=0.9):
    pass

def create_negative_samples(area, n_samples, window_size, positive_polygon_coords, seed=None):
    pass


if __name__ == "__main__":
    pass