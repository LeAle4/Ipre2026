from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handle import AREA_NAMES, crops_in_area

for area_name in AREA_NAMES:
    print(f"Number of positive samples in area {area_name}: {crops_in_area(area_name)}")