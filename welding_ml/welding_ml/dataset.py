import pathlib
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.resolve().parent
DS_PATH = BASE_DIR / 'data' / 'raw' / 'dataset.csv'


@cache
def get_data_frame() -> pd.DataFrame:
    return pd.read_csv('../data/raw/dataset.csv')


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray]:
    return df.iloc[:, :4].values, df.iloc[:, 4:].values


# =============================================================================
# @cache
# def get_data_frame(filepath_or_buffer: pathlib.PosixPath = DS_PATH) -> pd.DataFrame:
#     return pd.read_csv(filepath_or_buffer)
# =============================================================================

