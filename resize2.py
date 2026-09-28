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
    stride: int = DEFAULT_STRIDE
) -> None:
    """Resize the image of a DataPoint using LCI, snapping dimensions to 

    exact sliding window grid sizes.
    """
    scaling_factor = datapoint.m_px / desired_metric_scale

    raw_height = datapoint.image.shape[0] * scaling_factor
    raw_width = datapoint.image.shape[1] * scaling_factor

    # Snap raw dimensions to exact multiples required by view_as_windows
    # We lose a little bif of precision in the scaling factor, 
    # but this is necessary to ensure that the sliding window crops align perfectly with the resized image.
    new_height = snap_to_crop_grid(raw_height, window_size, stride)
    new_width = snap_to_crop_grid(raw_width, window_size, stride)

    # Resize the image directly to snapped dimensions
    resized_image = lci(datapoint.image, new_height, new_width)
    
    datapoint.modify_image(resized_image, desired_metric_scale)