import json
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent
DATA_DIR = PROJECT_PATH / "data"

RAW_AREA_PATHS = {
    "unita": DATA_DIR / "unita_raw",
    "lluta": DATA_DIR / "lluta_raw",
    "chugchug": DATA_DIR / "chugchug_raw",
}

class GeogliphData:
    def __init__(self, data:dict):
        pass

    def load_from_json(self, json_path:Path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return GeogliphData(data)


class TrainData:
    def __init__(self):
        pass

if __name__ == "__main__":
    pass