import json
import numpy as np

def read_voting(file_path: str) -> np.array:
    with open(file_path) as f:
        voting = json.load(f)["voting"]
        voting = np.array(voting)
        return voting