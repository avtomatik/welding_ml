from functools import cache

import pandas as pd
from config import DATA_DIR


@cache
def get_data_frame(file_name: str = 'ebw_data.csv') -> pd.DataFrame:
    return pd.read_csv(DATA_DIR.joinpath('raw').joinpath(file_name))
