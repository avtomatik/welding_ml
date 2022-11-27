import pathlib
from functools import cache
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DS_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"


@cache
def get_data_frame(filepath_or_buffer: pathlib.PosixPath = DS_PATH) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer)
