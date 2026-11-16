import math
import numpy as np
from scipy.fft import idct

from datamanager import DataPoint
from parameters import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE

def lci(I_in, *args):
    """
    Lagrange-Chebyshev Interpolation (LCI) image resizing. Implementation by 
    D. Occorsio, G. Ramella, W. Themistoclakis, “Lagrange-Chebyshev Interpolation for image resizing”, Mathematics and Computers in Simulation, ISSN: 0378-4754, DOI: 10.1016/j.matcom.2022.01.017, vol. 197, pp. 105 - 126, 2022
    Python implementation by Natan Brugueras, 2026.

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

        #Width and height scale might be different by a fraction of a pixel
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

def area_weighted_resize(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize *image* via area-weighted averaging (box / bin-averaging).

    This is the physically correct downsampling method for imagery:
    each output pixel is the weighted mean of all source pixels it
    overlaps, preserving radiometric fidelity without Gibbs ringing.

    Parameters
    ----------
    image : ndarray, shape (H, W) or (H, W, C)
        Input image, any numeric dtype.  Output is uint8.
    target_h, target_w : int
        Target spatial dimensions.

    Returns
    -------
    ndarray, shape (target_h, target_w[, C]), dtype uint8
    """
    src_h, src_w = image.shape[:2]
    is_color = image.ndim == 3
    n_channels = image.shape[2] if is_color else 1

    if not is_color:
        image = image[:, :, np.newaxis]

    img = image.astype(np.float64)
    out = np.empty((target_h, target_w, n_channels), dtype=np.float64)

    # Scale ratios (source pixels per output pixel)
    sy = src_h / target_h
    sx = src_w / target_w

    for oy in range(target_h):
        # Source y-interval covered by this output row
        y0 = oy * sy
        y1 = y0 + sy
        iy0 = int(math.floor(y0))
        iy1 = min(int(math.ceil(y1)), src_h)

        for ox in range(target_w):
            x0 = ox * sx
            x1 = x0 + sx
            ix0 = int(math.floor(x0))
            ix1 = min(int(math.ceil(x1)), src_w)

            # Accumulate weighted sum over source block
            total_weight = 0.0
            accum = np.zeros(n_channels, dtype=np.float64)

            for iy in range(iy0, iy1):
                wy = min(y1, iy + 1) - max(y0, iy)  # vertical overlap
                for ix in range(ix0, ix1):
                    wx = min(x1, ix + 1) - max(x0, ix)  # horizontal overlap
                    w = wy * wx
                    accum += img[iy, ix, :] * w
                    total_weight += w

            out[oy, ox, :] = accum / total_weight if total_weight > 0 else 0.0

    result = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if not is_color:
        result = result[:, :, 0]
    return result

def snap_to_crop_grid(dim_px: float, window_size: int, stride: int) -> int:
    """Snaps calculated pixel dimensions to the nearest target grid size:

    window_size + k * stride (for k >= 0).
    """
    if dim_px <= window_size:
        return window_size

    # Find the nearest integer step k
    k = max(0, round((dim_px - window_size) / stride))
    return window_size + k * stride


def resize_datapoint_image(
    datapoint: DataPoint, 
    desired_metric_scale: float,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    resize_fn = area_weighted_resize
) -> None:
    """Resize the image of a DataPoint to the target metric scale.

    The extracted datapoint's image bounds are expected to be already snapped to the
    sliding window grid (window_size + k * stride) during extraction. This function
    verifies that precondition and resizes the image directly to those dimensions
    without performing any snapping.

    Parameters
    ----------
    datapoint : DataPoint
        The datapoint to resize.
    desired_metric_scale : float
        Target resolution in m/px.
    window_size : int, optional
        Window size in pixels for grid validation.
    stride : int, optional
        Stride in pixels for grid validation.
    resize_fn : callable(image, target_h, target_w) -> ndarray, optional
        Resampling function. Defaults to area_weighted_resize.
    """
    # Target dimensions derived directly from the extracted bounding box
    minx, miny, maxx, maxy = datapoint.image_box.bounds
    target_w = int(round((maxx - minx) / desired_metric_scale))
    target_h = int(round((maxy - miny) / desired_metric_scale))

    # Check for the condition: target dimensions must already be snapped
    if target_w < window_size or (target_w - window_size) % stride != 0:
        raise ValueError(
            f"Extracted datapoint width ({target_w} px) is not snapped to crop grid: "
            f"expected window_size ({window_size}) + k * stride ({stride})."
        )
    if target_h < window_size or (target_h - window_size) % stride != 0:
        raise ValueError(
            f"Extracted datapoint height ({target_h} px) is not snapped to crop grid: "
            f"expected window_size ({window_size}) + k * stride ({stride})."
        )

    # Resize directly to target dimensions without snapping
    resized_image = resize_fn(datapoint.image, target_h, target_w)
    
    # Update image and metric scale
    datapoint.modify_image(resized_image, desired_metric_scale)