import argparse
import sys
import numpy as np
    
from PIL import Image
from scipy.fft import idct
from pathlib import Path

# Add parent directory to path to import utils
UTILS_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(UTILS_PATH))

from utils import PATHS, SCALES, Polygon, get_geos_from_summary, load_img_array_from_path, POLYGON_DATA_DIR, tabbed, title, make_resized_path

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

def resize_polygon(polygon:Polygon, scale:float) -> np.ndarray:
    """Resize the polygon's JPEG image to the target size using LCI.
    
    Args:
        polygon: Polygon object with image paths.
        scale: Scale factor to resize the image.
        
    Returns:
        Resized image as a NumPy array.
    """
    img_array = load_img_array_from_path(polygon.tif_path)
    print(tabbed(f"Original shape: {img_array.shape}, resizing with scale {scale}"))
    resized_array = lci(img_array, scale)
    print(tabbed(f"Resized shape: {resized_array.shape}"))
    polygon.shape = resized_array.shape
    return resized_array

def save_resized_polygon(geo:Polygon, resized_array:np.ndarray, save_path:Path) -> None:
    """Save the resized polygon image as a GeoTIFF with adjusted georeferencing.
    
    Args:
        geo: Polygon object with metadata.
        resized_array: Resized image array.
        save_path: Path to save the GeoTIFF file.
    """
    # Save resized image
    img = Image.fromarray(resized_array)
    geo.resized_path = save_path
    geo.save_metadata(POLYGON_DATA_DIR / f"{geo.area}_class{geo.class_id}_{geo.id}_metadata.json")
    img.save(save_path)

"""Code for use in the command line to resize polygon images to a standard size."""
def parse_arguments():
    parser = argparse.ArgumentParser(description="Resize polygon images to a standard size.")
    parser.add_argument(
        "--area",
        type=str,
        nargs = "+",
        choices=["unita", "chugchug", "lluta"],
        required=True,
        help="Study area to process.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    areas = args.area

    POLYGON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for area in areas:
        print(title(f"Resizing polygons in area: {area}"))
        geos = get_geos_from_summary(area)
        for geo in geos:
            print(f"Resizing polygon ID {geo.id}...")
            resized_array = resize_polygon(geo, scale=SCALES[area])
            save_path = make_resized_path(geo, area)
            save_resized_polygon(geo, resized_array, save_path)