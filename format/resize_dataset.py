"""
Resize all geoglif images using LCI method with georeferencing preserved.

This script processes all geoglif orthomosaic images from the data folders,
resizes them using the Lagrange-Chebyshev Interpolation method, and saves
them with updated geospatial metadata in the dataset/resized_geos folder.
Also copies and updates the corresponding metadata JSON files.
"""

from pathlib import Path
import json
import sys
import argparse
import numpy as np
from PIL import Image
import rasterio
from scipy.fft import idct
from rasterio.transform import Affine


# ============================================================================
# LCI (Lagrange-Chebyshev Interpolation) Functions
# ============================================================================

def lci(I_in, *args):
    """
    Lagrange–Chebyshev Interpolation (LCI) image resizing.

    Parameters
    ----------
    I_in : array-like
        Input image as (H, W) grayscale or (H, W, 3) RGB.
        dtype can be uint8 or float; output is uint8.
    *args :
        Either:
          - (mi, ni): target rows, target cols
          - (scale,): scale factor applied to both dimensions

    Returns
    -------
    I_fin : np.ndarray
        Resized image as uint8, shape (mi, ni) or (mi, ni, 3).
    """
    I = np.asarray(I_in)
    if I.ndim == 2:
        n1, n2 = I.shape
        c = 1
    elif I.ndim == 3 and I.shape[2] in (3, 4):
        # If RGBA, we keep RGB and drop alpha to match MATLAB behavior (which expects 3 channels)
        n1, n2 = I.shape[:2]
        c = 3
        I = I[:, :, :3]
    else:
        raise ValueError("I_in must be (H,W) or (H,W,3) (optionally (H,W,4) RGBA).")

    # Size computation (matches MATLAB switch nargin)
    if len(args) == 2:
        mi, ni = args
    elif len(args) == 1:
        scale = args[0]
        mi = int(round(scale * n1))
        ni = int(round(scale * n2))
    else:
        raise ValueError("Usage: lci(I_in, mi, ni) or lci(I_in, scale)")

    if mi <= 0 or ni <= 0:
        raise ValueError("Target size must be positive.")

    # Image values transformation
    I = I.astype(np.float64, copy=False)

    # eta and csi computation
    eta = (2 * np.arange(1, mi + 1) - 1) * np.pi / (2 * mi)
    csi = (2 * np.arange(1, ni + 1) - 1) * np.pi / (2 * ni)

    # Chebyshev polynomials with IDCT weights:
    # T(k,i)=cos(k*s_i)*wk with MATLAB-like normalization
    k1 = np.arange(0, n1)[:, None]  # (n1, 1)
    T1 = np.cos(k1 * eta[None, :]) * np.sqrt(2.0 / n1)
    T1[0, :] = np.sqrt(1.0 / n1)

    k2 = np.arange(0, n2)[:, None]  # (n2, 1)
    T2 = np.cos(k2 * csi[None, :]) * np.sqrt(2.0 / n2)
    T2[0, :] = np.sqrt(1.0 / n2)

    # lx and ly computation
    # MATLAB idct operates along columns; here we apply along axis=0
    lx = idct(T1, type=2, norm="ortho", axis=0)  # (n1, mi)
    ly = idct(T2, type=2, norm="ortho", axis=0)  # (n2, ni)

    # Helper: MATLAB double->uint8 conversion rounds and saturates; numpy needs clip to avoid wrap-around
    def to_uint8(x):
        return np.clip(np.rint(x), 0, 255).astype(np.uint8)

    if c == 3:
        I_fin = np.empty((mi, ni, 3), dtype=np.uint8)
        for ch in range(3):
            val = (lx.T @ I[:, :, ch]) @ ly  # (mi,n1)*(n1,n2)*(n2,ni) -> (mi,ni)
            I_fin[:, :, ch] = to_uint8(val)
    else:
        val = (lx.T @ I) @ ly
        I_fin = to_uint8(val)

    return I_fin


def save_georeferenced_tif(image_array, output_path, source_tif_path, scale_factor=None, target_size=None):
    """
    Save a resized image as a georeferenced GeoTIFF with updated metadata.
    
    The function reads geospatial metadata (CRS, transform) from the source .tif,
    adjusts the transform to account for the resize operation, and saves the result
    with proper georeferencing.
    
    Parameters
    ----------
    image_array : np.ndarray
        Resized image array (H, W) or (H, W, C) as uint8.
    output_path : str or Path
        Path where the georeferenced .tif will be saved.
    source_tif_path : str or Path
        Path to the original georeferenced .tif file.
    scale_factor : float, optional
        Scale factor used in resizing (for computing new transform).
        Either scale_factor or target_size must be provided.
    target_size : tuple, optional
        Target (height, width) if explicit size was used.
        Either scale_factor or target_size must be provided.
    
    Returns
    -------
    None
    
    Notes
    -----
    - Preserves CRS from source
    - Updates affine transform to reflect new pixel size
    - Handles both grayscale and RGB images
    """
    output_path = Path(output_path)
    source_tif_path = Path(source_tif_path)
    
    # Read source metadata
    with rasterio.open(source_tif_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_height, src_width = src.height, src.width
    
    # Determine scaling
    if scale_factor is not None:
        scale_x = scale_y = scale_factor
    elif target_size is not None:
        target_height, target_width = target_size
        scale_y = target_height / src_height
        scale_x = target_width / src_width
    else:
        raise ValueError("Either scale_factor or target_size must be provided")
    
    # Compute new transform
    # When resizing, pixel size changes inversely to scale
    # New pixel size = old pixel size / scale
    new_transform = Affine(
        src_transform.a / scale_x,  # pixel width
        src_transform.b,
        src_transform.c,  # x origin
        src_transform.d,
        src_transform.e / scale_y,  # pixel height (usually negative)
        src_transform.f   # y origin
    )
    
    # Prepare image for writing
    if image_array.ndim == 2:
        # Grayscale
        count = 1
        data = image_array
    elif image_array.ndim == 3:
        # RGB or RGBA
        count = image_array.shape[2]
        # Rasterio expects (bands, height, width)
        data = np.transpose(image_array, (2, 0, 1))
    else:
        raise ValueError("image_array must be 2D or 3D")
    
    # Write georeferenced GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=image_array.shape[0],
        width=image_array.shape[1],
        count=count,
        dtype=image_array.dtype,
        crs=src_crs,
        transform=new_transform,
        compress='lzw'  # Optional compression
    ) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            for i in range(count):
                dst.write(data[i], i + 1)


def lci_georeferenced(source_tif_path, output_path, *args):
    """
    Apply LCI resizing to a georeferenced .tif and save with updated metadata.
    
    Parameters
    ----------
    source_tif_path : str or Path
        Path to input georeferenced .tif file.
    output_path : str or Path
        Path where resized georeferenced .tif will be saved.
    *args :
        Same as lci(): either (mi, ni) or (scale,)
    
    Returns
    -------
    I_fin : np.ndarray
        Resized image array.
    
    Examples
    --------
    >>> lci_georeferenced("input.tif", "output.tif", 0.5)  # scale by 0.5
    >>> lci_georeferenced("input.tif", "output.tif", 800, 600)  # resize to 800x600
    """
    # Load the image
    with rasterio.open(source_tif_path) as src:
        # Read RGB bands (1, 2, 3) if available
        if src.count >= 3:
            img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
        else:
            img = src.read(1)
    
    # Apply LCI
    img_resized = lci(img, *args)
    
    # Determine scale/size for metadata
    if len(args) == 1:
        scale_factor = args[0]
        target_size = None
    else:
        scale_factor = None
        target_size = args  # (mi, ni)
    
    # Save with georeferencing
    save_georeferenced_tif(
        img_resized,
        output_path,
        source_tif_path,
        scale_factor=scale_factor,
        target_size=target_size
    )
    
    return img_resized


# ============================================================================
# Dataset Processing Functions
# ============================================================================

# Scale factors for each area (based on analysis)
AREA_SCALES = {
    'unita': 0.886,
    'lluta': 0.218,
    'chugchug': 0.18
}

# Default paths (can be overridden via command-line arguments)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "resized_geos"

# Global variables that will be set by parse_arguments()
DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def get_geos_directories():
    """Find all *_geos2 directories in data folder."""
    return sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.endswith('_geos2')])


def extract_area_name(dir_name):
    """Extract area name from directory name (e.g., 'unita_geos2' -> 'unita')."""
    return dir_name.replace('_geos2', '')


def update_metadata_json(metadata_path, output_path, scale_factor):
    """
    Copy and update metadata JSON file with new image dimensions.
    
    Parameters
    ----------
    metadata_path : Path
        Path to original metadata JSON file
    output_path : Path
        Path where updated metadata JSON will be saved
    scale_factor : float
        Scale factor used for resizing
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Update image_shape if present
    if 'image_shape' in metadata:
        old_width = metadata['image_shape']['width']
        old_height = metadata['image_shape']['height']
        
        metadata['image_shape']['width'] = int(round(old_width * scale_factor))
        metadata['image_shape']['height'] = int(round(old_height * scale_factor))
    
    # Update file size info if present
    if 'file_size_mb' in metadata:
        # Approximate new file size (scales roughly with pixel count)
        metadata['file_size_mb'] = metadata['file_size_mb'] * (scale_factor ** 2)
    
    # Add resize info
    metadata['resized'] = True
    metadata['scale_factor'] = scale_factor
    metadata['original_dimensions'] = {
        'width': old_width if 'image_shape' in metadata else None,
        'height': old_height if 'image_shape' in metadata else None
    }
    
    # Bounds remain the same (same geographic area)
    # CRS remains the same
    # Polygon coordinates remain the same (geographic coordinates)
    
    # Save updated metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Resize geoglif images using LCI method with georeferencing preserved.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resize_dataset.py
  python resize_dataset.py --data-dir /path/to/data
  python resize_dataset.py --output-dir /path/to/output
  python resize_dataset.py --data-dir /path/to/data --output-dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help=f'Path to data directory containing *_geos folders (default: {DEFAULT_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Path to output directory for resized images (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    return parser.parse_args()


def resize_all_geos():
    """Process all geoglif images and resize with georeferencing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    geos_dirs = get_geos_directories()
    
    if not geos_dirs:
        print("No *_geos2 directories found in data folder!")
        return
    
    print("=" * 70)
    print("RESIZING ALL GEOGLIF IMAGES WITH LCI + GEOREFERENCING")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Found {len(geos_dirs)} area(s) to process")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    total_processed = 0
    total_skipped = 0
    
    for geos_dir in geos_dirs:
        area_name = extract_area_name(geos_dir.name)
        
        # Get scale factor for this area
        scale_factor = AREA_SCALES.get(area_name)
        if scale_factor is None:
            print(f"\n[{area_name.upper()}] No scale factor defined, skipping...")
            continue
        
        # Find all .tif files (orthomosaic images)
        tif_files = sorted(geos_dir.glob("*_ortho.tif"))
        
        if not tif_files:
            print(f"\n[{area_name.upper()}] No orthomosaic .tif files found, skipping...")
            continue
        
        print(f"\n[{area_name.upper()}] Processing {len(tif_files)} images (scale: {scale_factor})")
        print("-" * 60)
        
        # Create output subdirectory for this area
        area_output_dir = OUTPUT_DIR / f"{area_name}_geos"
        area_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, tif_path in enumerate(tif_files, 1):
            output_path = area_output_dir / tif_path.name
            png_path = area_output_dir / tif_path.stem.replace('_ortho', '_resized.png')
            
            # Find corresponding metadata JSON
            metadata_name = tif_path.stem.replace('_ortho', '_metadata.json')
            metadata_path = geos_dir / metadata_name
            output_metadata_path = area_output_dir / metadata_name
            
            # Skip if already processed
            if output_path.exists():
                print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - already exists, skipping")
                total_skipped += 1
                continue
            
            try:
                print(f"  [{idx}/{len(tif_files)}] {tif_path.name} - resizing...", end=" ")
                
                # Read source image for PNG saving
                with rasterio.open(str(tif_path)) as src:
                    if src.count >= 3:
                        img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
                    else:
                        img = src.read(1)
                
                # Apply LCI resizing
                img_resized = lci(img, scale_factor)
                
                # Save as GeoTIFF with georeferencing
                lci_georeferenced(str(tif_path), str(output_path), scale_factor)
                
                # Save as PNG (visual record)
                Image.fromarray(img_resized).save(str(png_path))
                
                print("✓", end=" ")
                
                # Copy and update metadata if it exists
                if metadata_path.exists():
                    print("+ metadata...", end=" ")
                    update_metadata_json(metadata_path, output_metadata_path, scale_factor)
                    print("✓")
                else:
                    print("(no metadata)")
                
                total_processed += 1
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {total_processed}")
    print(f"Total images skipped: {total_skipped}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global directories if specified
    if args.data_dir:
        DATA_DIR = Path(args.data_dir).resolve()
        print(f"Using custom data directory: {DATA_DIR}")
    
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).resolve()
        print(f"Using custom output directory: {OUTPUT_DIR}")
    
    # Run the resize process
    resize_all_geos()