from functools import cache
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent.resolve().parent
DS_PATH = BASE_DIR / 'data' / 'raw' / 'ebw_data.csv'


# =============================================================================
# @cache
# def get_data_frame(filepath_or_buffer: Path = DS_PATH) -> pd.DataFrame:
#     return pd.read_csv(filepath_or_buffer)
# =============================================================================


@cache
def get_data_frame() -> pd.DataFrame:
    return pd.read_csv('../data/raw/ebw_data.csv')
