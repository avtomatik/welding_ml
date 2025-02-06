from functools import cache
from pathlib import Path

import pandas as pd


@cache
def get_data_frame(file_name: str = 'ebw_data.csv') -> pd.DataFrame:
    return pd.read_csv(
        (
            Path(__file__).parent.parent
            .joinpath('data')
            .joinpath('raw')
            .joinpath(file_name)
        )
    )
