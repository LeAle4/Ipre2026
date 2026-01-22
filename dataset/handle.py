"""Centralized file handling and path management for the geoglyphs project."""

import json
from pathlib import Path

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# Raw data directories (extracted from geopackages)
UNITA_RAW_DIR = DATA_DIR / "unita_raw"
CHUG_RAW_DIR = DATA_DIR / "chugchug_raw"
LLUTA_RAW_DIR = DATA_DIR / "lluta_raw"
SALVADOR_RAW_DIR = DATA_DIR / "salvador_raw"
GRANLLAMA_RAW_DIR = DATA_DIR / "granllama_raw"

# Processed geos directories (class 1 polygons only)
UNITA_GEOS_DIR = DATA_DIR / "unita_geos"
CHUG_GEOS_DIR = DATA_DIR / "chug_geos"
LLUTA_GEOS_DIR = DATA_DIR / "lluta_geos"

# Polygon directories
UNITA_POLYGONS_DIR = DATA_DIR / "unita_polygons"
CHUG_POLYGONS_DIR = DATA_DIR / "chug_polygons"
LLUTA_POLYGONS_DIR = DATA_DIR / "lluta_polygons"

# Summary JSON paths
UNITA_SUMMARY_PATH = UNITA_RAW_DIR / "summary.json"
CHUG_SUMMARY_PATH = CHUG_RAW_DIR / "summary.json"
LLUTA_SUMMARY_PATH = LLUTA_RAW_DIR / "summary.json"
SALVADOR_SUMMARY_PATH = SALVADOR_RAW_DIR / "summary.json"
GRANLLAMA_SUMMARY_PATH = GRANLLAMA_RAW_DIR / "summary.json"

# Dataset configuration
DATASETS = {
    'unita': {
        'name': 'UNITA',
        'raw_dir': UNITA_RAW_DIR,
        'geos_dir': UNITA_GEOS_DIR,
        'polygons_dir': UNITA_POLYGONS_DIR,
        'summary_path': UNITA_SUMMARY_PATH
    },
    'chug': {
        'name': 'CHUG',
        'raw_dir': CHUG_RAW_DIR,
        'geos_dir': CHUG_GEOS_DIR,
        'polygons_dir': CHUG_POLYGONS_DIR,
        'summary_path': CHUG_SUMMARY_PATH
    },
    'lluta': {
        'name': 'LLUTA',
        'raw_dir': LLUTA_RAW_DIR,
        'geos_dir': LLUTA_GEOS_DIR,
        'polygons_dir': LLUTA_POLYGONS_DIR,
        'summary_path': LLUTA_SUMMARY_PATH
    },
    'salvador': {
        'name': 'SALVADOR',
        'raw_dir': SALVADOR_RAW_DIR,
        'summary_path': SALVADOR_SUMMARY_PATH
    },
    'granllama': {
        'name': 'GRANLLAMA',
        'raw_dir': GRANLLAMA_RAW_DIR,
        'summary_path': GRANLLAMA_SUMMARY_PATH
    }
}

# ============================================================================
# FILE I/O FUNCTIONS
# ============================================================================

def load_json(path):
    """Load JSON file from path.
    
    Args:
        path: Path to JSON file (str or Path)
    
    Returns:
        Parsed JSON data
    """
    with open(path, 'r') as file:
        return json.load(file)

def save_json(data, path, indent=2):
    """Save data as JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        path: Path to save JSON file (str or Path)
        indent: Indentation level for pretty printing (default: 2)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(data, file, indent=indent)

def get_dataset_info(dataset_name):
    """Get information for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('unita', 'chug', 'lluta', etc.)
    
    Returns:
        Dictionary with dataset information
    
    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]

def get_all_datasets(include_incomplete=False):
    """Get list of all available datasets.
    
    Args:
        include_incomplete: If True, include datasets without all directories (default: False)
    
    Returns:
        List of dataset names
    """
    if include_incomplete:
        return list(DATASETS.keys())
    
    # Only return datasets that have geos_dir defined
    return [name for name, info in DATASETS.items() if 'geos_dir' in info]
