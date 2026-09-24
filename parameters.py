from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

# Polygon class mappings: geoglyphs, ground, and road
GEO_CLASS = 1
GROUND_CLASS = 2
ROAD_CLASS = 3
GEO_TYPE = "geo"
GROUND_TYPE = "ground"
ROAD_TYPE = "road"

#Change to be calculated
DEFAULT_TARGET_SCALE = 0.05  # Desired scale in meters per pixel for the resized images
DEFAULT_WINDOW_SIZE = 224
DEFAULT_STRIDE = DEFAULT_WINDOW_SIZE // 2
DEFAULT_THRESHOLD_CROP_CONTENT = 0.4  # Minimum fraction of geoglyph pixels in a crop to be considered valid
DEFAULT_NEGATIVES_RATIO = 3 # Number of negative samples per positive sample
