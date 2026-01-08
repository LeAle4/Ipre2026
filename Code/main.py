"""
Entropy Analysis of Binary Matrices using EntropyHub

This script generates various types of 400x400 binary matrices and computes
different entropy measures using the EntropyHub library.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import EntropyHub as EH
from pathlib import Path
import os


# ============================================================================
# PERFORMANCE GUARDRAILS
# ============================================================================

# These limits prevent O(N^2) entropy functions from exploding in runtime
# when applied to large flattened images. Adjust as needed.
MAX_LEN_SLOW = 5000   # for SampEn, ApEn, FuzzEn
MAX_LEN_PERM = 50000  # for PermEn


def _prepare_signal(matrix, max_len=None, downsample_step=None):
    """
    Convert matrix to 1D float array and optionally downsample to a target length.
    Returns the signal and a dict with metadata about the transformation.
    """
    sig = np.asarray(matrix, dtype=float).ravel()
    meta = {"orig_len": sig.size, "downsample_step": None}
    
    if downsample_step and downsample_step > 1:
        sig = sig[::int(downsample_step)]
        meta["downsample_step"] = int(downsample_step)
    
    if max_len and sig.size > max_len:
        step = int(np.ceil(sig.size / max_len))
        sig = sig[::step]
        meta["downsample_step"] = step if meta["downsample_step"] is None else meta["downsample_step"] * step
    
    meta["used_len"] = sig.size
    return sig, meta


# ============================================================================
# MATRIX GENERATION FUNCTIONS
# ============================================================================

def generate_random_matrix(size=400, density=0.5, seed=None):
    """
    Generate a random binary matrix.
    
    Parameters:
    -----------
    size : int
        Size of the square matrix (default: 400)
    density : float
        Probability of a pixel being 1 (default: 0.5)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray
        Binary matrix of shape (size, size)
    """
    if seed is not None:
        np.random.seed(seed)
    return (np.random.random((size, size)) < density).astype(np.uint8)


def generate_checkered_matrix(size=400, square_size=32):
    """
    Generate a checkered pattern binary matrix.
    
    Parameters:
    -----------
    size : int
        Size of the square matrix (default: 400)
    square_size : int
        Size of each square in the checkered pattern (default: 32)
        
    Returns:
    --------
    numpy.ndarray
        Binary matrix of shape (size, size) with checkered pattern
    """
    matrix = np.zeros((size, size), dtype=np.uint8)
    
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            # Alternate pattern based on position
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                i_end = min(i + square_size, size)
                j_end = min(j + square_size, size)
                matrix[i:i_end, j:j_end] = 1
                
    return matrix


def generate_striped_matrix(size=400, stripe_width=16, horizontal=True):
    """
    Generate a striped pattern binary matrix.
    
    Parameters:
    -----------
    size : int
        Size of the square matrix (default: 400)
    stripe_width : int
        Width of each stripe (default: 16)
    horizontal : bool
        If True, stripes are horizontal; if False, vertical
        
    Returns:
    --------
    numpy.ndarray
        Binary matrix of shape (size, size) with striped pattern
    """
    matrix = np.zeros((size, size), dtype=np.uint8)
    
    if horizontal:
        for i in range(0, size, stripe_width * 2):
            i_end = min(i + stripe_width, size)
            matrix[i:i_end, :] = 1
    else:
        for j in range(0, size, stripe_width * 2):
            j_end = min(j + stripe_width, size)
            matrix[:, j:j_end] = 1
            
    return matrix


def generate_circle_matrix(size=400, num_circles=5):
    """
    Generate a binary matrix with concentric circles.
    
    Parameters:
    -----------
    size : int
        Size of the square matrix (default: 400)
    num_circles : int
        Number of concentric circles (default: 5)
        
    Returns:
    --------
    numpy.ndarray
        Binary matrix of shape (size, size) with circles
    """
    matrix = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    y, x = np.ogrid[:size, :size]
    distances = np.sqrt((x - center)**2 + (y - center)**2)
    
    max_radius = center
    for i in range(num_circles):
        r_inner = i * max_radius / num_circles
        r_outer = (i + 1) * max_radius / num_circles
        
        if i % 2 == 0:
            mask = (distances >= r_inner) & (distances < r_outer)
            matrix[mask] = 1
            
    return matrix


def generate_simple_face_matrix(size=400):
    """
    Generate a binary matrix with a simple face drawing.
    
    Parameters:
    -----------
    size : int
        Size of the square matrix (default: 400)
        
    Returns:
    --------
    numpy.ndarray
        Binary matrix of shape (size, size) with a face
    """
    matrix = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    y, x = np.ogrid[:size, :size]
    
    # Face circle (outline)
    face_radius = size // 3
    face_dist = np.sqrt((x - center)**2 + (y - center)**2)
    face_outline = (face_dist >= face_radius - 20) & (face_dist <= face_radius)
    matrix[face_outline] = 1
    
    # Left eye
    left_eye_x = center - size // 8
    left_eye_y = center - size // 10
    left_eye_dist = np.sqrt((x - left_eye_x)**2 + (y - left_eye_y)**2)
    matrix[left_eye_dist <= size // 20] = 1
    
    # Right eye
    right_eye_x = center + size // 8
    right_eye_y = center - size // 10
    right_eye_dist = np.sqrt((x - right_eye_x)**2 + (y - right_eye_y)**2)
    matrix[right_eye_dist <= size // 20] = 1
    
    # Smile (arc)
    smile_y = center + size // 10
    for i in range(-size // 6, size // 6):
        smile_x = center + i
        # Parabolic smile
        y_offset = int((i**2) / (size // 3))
        if 0 <= smile_x < size and 0 <= smile_y + y_offset < size:
            for dy in range(-10, 10):
                if 0 <= smile_y + y_offset + dy < size:
                    matrix[smile_y + y_offset + dy, smile_x] = 1
    
    return matrix


def generate_simple_animal_matrix(size=400):
    """
    Generate a binary matrix with a simple animal (cat) drawing.
    
    Parameters:
    -----------
    size : int
        Size of the square matrix (default: 400)
        
    Returns:
    --------
    numpy.ndarray
        Binary matrix of shape (size, size) with an animal
    """
    matrix = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    y, x = np.ogrid[:size, :size]
    
    # Head (circle)
    head_radius = size // 4
    head_dist = np.sqrt((x - center)**2 + (y - center)**2)
    matrix[head_dist <= head_radius] = 1
    
    # Left ear (triangle)
    left_ear_x = center - size // 5
    left_ear_y = center - size // 4
    for i in range(size):
        for j in range(size):
            # Triangle vertices
            if (j - left_ear_x) < 0 and (i - left_ear_y) < 0:
                if abs(j - left_ear_x) + abs(i - left_ear_y) < size // 6:
                    if (j - left_ear_x) > -(i - left_ear_y):
                        matrix[i, j] = 1
    
    # Right ear (triangle)
    right_ear_x = center + size // 5
    right_ear_y = center - size // 4
    for i in range(size):
        for j in range(size):
            if (j - right_ear_x) > 0 and (i - right_ear_y) < 0:
                if abs(j - right_ear_x) + abs(i - right_ear_y) < size // 6:
                    if (j - right_ear_x) < -(i - right_ear_y):
                        matrix[i, j] = 1
    
    return matrix


def load_image_as_grayscale_matrix(image_path, size=400):
    """
    Load an image and convert it to a grayscale matrix with values in [0, 1].
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    size : int
        Target size for the matrix (default: 400)
        
    Returns:
    --------
    numpy.ndarray
        Grayscale matrix of shape (size, size) with float values 0..1
    """
    img = Image.open(image_path).convert('L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    matrix = np.array(img, dtype=float) / 255.0
    return matrix


# ============================================================================
# ENTROPY COMPUTATION FUNCTIONS
# ============================================================================

def compute_sample_entropy(matrix, m=2, r=0.2):
    """
    Compute Sample Entropy of a binary matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix
    m : int
        Embedding dimension (default: 2)
    r : float
        Tolerance (default: 0.2)
        
    Returns:
    --------
    tuple
        (entropy_value, additional_info)
    """
    # Normalize to 1D float signal with length guard
    signal, meta = _prepare_signal(matrix, max_len=MAX_LEN_SLOW)
    
    # Compute Sample Entropy
    Samp, A, B = EH.SampEn(signal, m=m, r=r)
    
    # Ensure scalar output for downstream formatting/plots
    Samp_val = float(np.nanmean(np.asarray(Samp)))
    return Samp_val, {"A": A, "B": B, "raw": Samp, "len_used": meta["used_len"], "len_orig": meta["orig_len"]}


def compute_approximate_entropy(matrix, m=2, r=0.2):
    """
    Compute Approximate Entropy of a binary matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix
    m : int
        Embedding dimension (default: 2)
    r : float
        Tolerance (default: 0.2)
        
    Returns:
    --------
    tuple
        (entropy_value, additional_info)
    """
    signal, meta = _prepare_signal(matrix, max_len=MAX_LEN_SLOW)
    
    # Compute Approximate Entropy
    ApEn, phi = EH.ApEn(signal, m=m, r=r)
    
    ApEn_val = float(np.nanmean(np.asarray(ApEn)))
    return ApEn_val, {"phi": phi, "raw": ApEn, "len_used": meta["used_len"], "len_orig": meta["orig_len"]}


def compute_permutation_entropy(matrix, m=3, tau=1):
    """
    Compute Permutation Entropy of a binary matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix
    m : int
        Embedding dimension (default: 3)
    tau : int
        Time delay (default: 1)
        
    Returns:
    --------
    tuple
        (entropy_value, additional_info)
    """
    signal, meta = _prepare_signal(matrix, max_len=MAX_LEN_PERM)
    
    # Compute Permutation Entropy (EH v2 may return 2 or 3 values)
    out = EH.PermEn(signal, m=m, tau=tau)
    info = {}
    if isinstance(out, (list, tuple)):
        if len(out) == 2:
            PermEn, pnorm = out
            info["pnorm"] = pnorm
        elif len(out) == 3:
            PermEn, pnorm, extra = out
            info["pnorm"] = pnorm
            info["extra"] = extra
        else:
            PermEn = out[0]
            info["extra"] = out[1:]
    else:
        PermEn = out
    
    PermEn_val = float(np.nanmean(np.asarray(PermEn)))
    info["raw"] = PermEn
    info["len_used"] = meta["used_len"]
    info["len_orig"] = meta["orig_len"]
    return PermEn_val, info


def compute_fuzzy_entropy(matrix, m=2, r=(0.2, 2), Fx='Default'):
    """
    Compute Fuzzy Entropy of a binary matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix
    m : int
        Embedding dimension (default: 2)
    r : float
        Tolerance (default: 0.2)
    Fx : str
        Fuzzy function (default: 'default')
        
    Returns:
    --------
    tuple
        (entropy_value, additional_info)
    """
    signal, meta = _prepare_signal(matrix, max_len=MAX_LEN_SLOW)
    
    # Adapt r/Fx per EntropyHub v2.0: when Fx == "Default", r must be a 2-tuple
    Fx_use = Fx
    r_use = r
    if isinstance(Fx_use, str) and Fx_use.lower() == "default":
        # Ensure r is a two-element tuple (tolerance, n)
        if not (isinstance(r_use, (tuple, list)) and len(r_use) == 2):
            # Coerce float to sensible default tuple
            r_use = (float(r_use) if not isinstance(r_use, (tuple, list)) else float(r_use[0]), 2)
    
    # Compute Fuzzy Entropy
    FuzzEn, A, B = EH.FuzzEn(signal, m=m, r=r_use, Fx=Fx_use)
    
    FuzzEn_val = float(np.nanmean(np.asarray(FuzzEn)))
    return FuzzEn_val, {"A": A, "B": B, "raw": FuzzEn, "r_used": r_use, "Fx_used": Fx_use, "len_used": meta["used_len"], "len_orig": meta["orig_len"]}


def compute_spectral_entropy(matrix):
    """
    Compute Spectral Entropy of a binary matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix
        
    Returns:
    --------
    tuple
        (entropy_value, additional_info)
    """
    signal = np.asarray(matrix, dtype=float).ravel()
    
    # Compute Spectral Entropy
    SpecEn, BandEn = EH.SpecEn(signal)
    
    SpecEn_val = float(np.nanmean(np.asarray(SpecEn)))
    return SpecEn_val, {"BandEn": BandEn, "raw": SpecEn}


def compute_all_entropies(matrix, matrix_name="Unknown"):
    """
    Compute core entropy measures (Sample, Approximate, Fuzzy) for a matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix
    matrix_name : str
        Name/description of the matrix
        
    Returns:
    --------
    dict
        Dictionary containing all entropy measures
    """
    print(f"\nComputing entropies for: {matrix_name}")
    print("=" * 60)
    
    results = {
        "name": matrix_name,
        "shape": matrix.shape,
        "density": np.mean(matrix)
    }
    
    # Prepare signal once for the O(N) entropies to reduce overhead
    signal, meta = _prepare_signal(matrix, max_len=MAX_LEN_SLOW)

    # Sample Entropy
    try:
        print("Computing Sample Entropy...")
        Samp, A_s, B_s = EH.SampEn(signal, m=2, r=0.2)
        samp_en_val = float(np.nanmean(np.asarray(Samp)))
        results["SampleEntropy"] = samp_en_val
        print(f"  Sample Entropy: {samp_en_val:.6f}")
    except Exception as e:
        print(f"  Error computing Sample Entropy: {e}")
        results["SampleEntropy"] = None

    # Approximate Entropy
    try:
        print("Computing Approximate Entropy...")
        ApEn, phi = EH.ApEn(signal, m=2, r=0.2)
        app_en_val = float(np.nanmean(np.asarray(ApEn)))
        results["ApproximateEntropy"] = app_en_val
        print(f"  Approximate Entropy: {app_en_val:.6f}")
    except Exception as e:
        print(f"  Error computing Approximate Entropy: {e}")
        results["ApproximateEntropy"] = None

    # Fuzzy Entropy (ensure r tuple handling for Default Fx)
    try:
        print("Computing Fuzzy Entropy...")
        r_use = (0.2, 2)
        Fx_use = 'Default'
        FuzzEn, A_f, B_f = EH.FuzzEn(signal, m=2, r=r_use, Fx=Fx_use)
        fuzz_en_val = float(np.nanmean(np.asarray(FuzzEn)))
        results["FuzzyEntropy"] = fuzz_en_val
        print(f"  Fuzzy Entropy: {fuzz_en_val:.6f}")
    except Exception as e:
        print(f"  Error computing Fuzzy Entropy: {e}")
        results["FuzzyEntropy"] = None
    
    return results


# ============================================================================
# VISUALIZATION AND SAVING FUNCTIONS
# ============================================================================

def visualize_matrix(matrix, title="Matrix", save_path=None, show=True):
    """
    Visualize a matrix (binary or grayscale) as an image.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix to visualize
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(matrix, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_matrix_as_image(matrix, filename, output_dir="output"):
    """
    Save a matrix (binary or grayscale) as an image file.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Binary matrix to save
    filename : str
        Name of the output file (without extension)
    output_dir : str
        Directory to save the image (default: 'output')
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to 0-255 range; matrix may be 0/1 or 0..1 floats
    img_array = np.clip(matrix, 0, 1) * 255
    img_array = img_array.astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(img_array, mode='L')
    output_path = os.path.join(output_dir, f"{filename}.png")
    img.save(output_path)
    
    print(f"Saved image to: {output_path}")
    return output_path


def create_comparison_plot(matrices_dict, save_path=None, show=True):
    """
    Create a comparison plot of multiple matrices.
    
    Parameters:
    -----------
    matrices_dict : dict
        Dictionary with matrix names as keys and matrices as values
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    n_matrices = len(matrices_dict)
    cols = 3
    rows = (n_matrices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, matrix) in enumerate(matrices_dict.items()):
        row = idx // cols
        col = idx % cols
        
        axes[row, col].imshow(matrix, cmap='binary', interpolation='nearest')
        axes[row, col].set_title(name, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(n_matrices, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Ensure parent directory exists before saving
        try:
            from pathlib import Path as _Path
            _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_entropy_comparison(results_list, save_path=None, show=True):
    """
    Create a bar plot comparing entropy measures across different matrices.
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing entropy results
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    entropy_types = ['SampleEntropy', 'ApproximateEntropy', 'FuzzyEntropy']
    
    fig, axes = plt.subplots(len(entropy_types), 1, figsize=(12, 4 * len(entropy_types)))
    
    if len(entropy_types) == 1:
        axes = [axes]
    
    for idx, entropy_type in enumerate(entropy_types):
        names = [r['name'] for r in results_list]
        values = [r.get(entropy_type, 0) if r.get(entropy_type) is not None else 0 
                  for r in results_list]
        
        axes[idx].bar(names, values, color='steelblue', alpha=0.7)
        axes[idx].set_ylabel('Entropy Value', fontsize=12)
        axes[idx].set_title(f'{entropy_type}', fontsize=14, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        try:
            from pathlib import Path as _Path
            _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved entropy comparison plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_matrices_with_entropies(matrices_dict, results_list, save_path=None, show=True):
    """
    Create a comprehensive plot showing each matrix with its entropy values.
    
    Parameters:
    -----------
    matrices_dict : dict
        Dictionary with matrix names as keys and matrices as values
    results_list : list
        List of dictionaries containing entropy results
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    n_matrices = len(matrices_dict)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 4 * n_matrices))
    
    entropy_types = ['SampleEntropy', 'ApproximateEntropy', 'FuzzyEntropy']
    
    for idx, (name, matrix) in enumerate(matrices_dict.items()):
        # Find corresponding results
        result = next((r for r in results_list if r['name'] == name), None)
        
        # Create gridspec for this row (matrix + entropy text)
        row_start = idx * 2
        
        # Plot matrix
        ax_img = plt.subplot2grid((n_matrices * 2, 3), (row_start, 0), rowspan=2, colspan=1)
        ax_img.imshow(matrix, cmap='binary', interpolation='nearest')
        ax_img.set_title(f'{name}\nDensity: {np.mean(matrix):.4f}', 
                        fontsize=12, fontweight='bold')
        ax_img.axis('off')
        
        # Plot entropy values as text
        ax_text = plt.subplot2grid((n_matrices * 2, 3), (row_start, 1), rowspan=2, colspan=2)
        ax_text.axis('off')
        
        if result:
            # Create formatted text with entropy values
            y_pos = 0.9
            spacing = 0.18
            
            ax_text.text(0.05, y_pos, 'Entropy Measures:', 
                        fontsize=14, fontweight='bold', transform=ax_text.transAxes)
            y_pos -= spacing
            
            for entropy_type in entropy_types:
                value = result.get(entropy_type)
                if value is not None:
                    color = 'darkgreen'
                    text = f'{entropy_type}: {value:.6f}'
                else:
                    color = 'red'
                    text = f'{entropy_type}: N/A'
                
                ax_text.text(0.05, y_pos, text, 
                           fontsize=12, color=color, transform=ax_text.transAxes,
                           family='monospace')
                y_pos -= spacing
    
    plt.suptitle('Binary Matrix Entropy Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        try:
            from pathlib import Path as _Path
            _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved matrices with entropies plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_entropy_analysis(size=400, output_dir="output", visualize=True, save_images=True):
    """
    Run complete entropy analysis on various types of binary matrices.
    
    Parameters:
    -----------
    size : int
        Size of the square matrices to generate (default: 400)
    output_dir : str
        Directory to save results (default: 'output')
    visualize : bool
        Whether to show visualizations (default: True)
    save_images : bool
        Whether to save images (default: True)
    """
    print("\n" + "=" * 70)
    print(f"BINARY MATRIX ENTROPY ANALYSIS ({size}x{size})")
    print("=" * 70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate matrices
    print("\n[1/4] Generating matrices...")
    matrices = {}
    
    # Random matrices with different densities
    matrices["Random (50%)"] = generate_random_matrix(size=size, density=0.5, seed=42)
    matrices["Random (25%)"] = generate_random_matrix(size=size, density=0.25, seed=43)
    matrices["Random (75%)"] = generate_random_matrix(size=size, density=0.75, seed=44)
    
    # Pattern matrices
    matrices["Checkered (32px)"] = generate_checkered_matrix(size=size, square_size=32)
    matrices["Checkered (64px)"] = generate_checkered_matrix(size=size, square_size=64)
    matrices["Horizontal Stripes"] = generate_striped_matrix(size=size, horizontal=True)
    matrices["Vertical Stripes"] = generate_striped_matrix(size=size, horizontal=False)
    matrices["Concentric Circles"] = generate_circle_matrix(size=size, num_circles=10)
    
    # Drawing matrices
    matrices["Simple Face"] = generate_simple_face_matrix(size=size)
    matrices["Simple Animal (Cat)"] = generate_simple_animal_matrix(size=size)
    
    # Load image.png if it exists (grayscale 0..1)
    image_path = "image.png"
    if os.path.exists(image_path):
        print(f"Loading custom image: {image_path}")
        try:
            matrices["Custom Image"] = load_image_as_grayscale_matrix(image_path, size=size)
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")
    else:
        print(f"Note: {image_path} not found, skipping custom image.")
    
    print(f"Generated {len(matrices)} matrices")
    
    # Save images
    if save_images:
        print("\n[2/4] Saving matrix images...")
        for name, matrix in matrices.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
            save_matrix_as_image(matrix, safe_name, output_dir)
    
    # Create comparison visualization
    if visualize:
        print("\n[3/4] Creating comparison visualization...")
        create_comparison_plot(matrices, 
                              save_path=os.path.join(output_dir, "comparison.png"),
                              show=False)
    
    # Compute entropies
    print("\n[4/4] Computing entropy measures...")
    results = []
    for name, matrix in matrices.items():
        result = compute_all_entropies(matrix, matrix_name=name)
        results.append(result)
    
    # Create combined matrix-entropy visualization
    if visualize:
        print("\nCreating combined matrix-entropy visualization...")
        plot_matrices_with_entropies(matrices, results,
                                     save_path=os.path.join(output_dir, "matrices_with_entropies.png"),
                                     show=True)
        
        # Also create entropy comparison plot
        print("\nCreating entropy comparison plots...")
        plot_entropy_comparison(results,
                               save_path=os.path.join(output_dir, "entropy_comparison.png"),
                               show=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Density: {result['density']:.4f}")
        for key, value in result.items():
            if key not in ['name', 'shape', 'density'] and value is not None:
                print(f"  {key}: {value:.6f}")
    
    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("=" * 70 + "\n")
    
    return results, matrices


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the complete analysis
    # You can change the size parameter to any value (e.g., 256, 512, 1024)
    results, matrices = run_entropy_analysis(
        size=100,  # Change this to use different matrix sizes
        output_dir="output",
        visualize=True,
        save_images=True
    )
    
    # Optional: Visualize individual matrices
    # for name, matrix in matrices.items():
    #     visualize_matrix(matrix, title=name, show=True)
    
    # Example: Run analysis with different sizes
    # results_small, matrices_small = run_entropy_analysis(size=256, output_dir="output_256")
    # results_large, matrices_large = run_entropy_analysis(size=1024, output_dir="output_1024")
