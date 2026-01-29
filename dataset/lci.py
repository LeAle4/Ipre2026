import numpy as np
from scipy.fft import idct
import rasterio
from rasterio.transform import Affine
from pathlib import Path


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


# Example usage:
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # Example 1: Simple resize without georeferencing (old method)
    img = np.array(Image.open("../data/unita_geos/geoglif_0000_ortho.tif"))
    out = lci(img, 0.886)
    Image.fromarray(out).save("resized.tif")

    # Example 2: Resize with georeferencing preserved (new method)
    # This reads the .tif, resizes it, and updates the geospatial metadata
    lci_georeferenced(
        "../data/unita_geos/geoglif_0000_ortho.tif",
        "resized_georeferenced.tif",
        0.886  # scale factor
    )

    # Example 3: Resize to specific dimensions with georeferencing
    lci_georeferenced(
        "../data/unita_geos/geoglif_0000_ortho.tif",
        "resized_800x600_georeferenced.tif",
        800, 600  # target height, width
    )