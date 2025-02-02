import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dataset import get_data_frame


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray]:
    n_columns = 4
    return df.iloc[:, :n_columns].values, df.iloc[:, n_columns:].values


def fit_scaler(feature: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(feature)
    return scaler


def get_X_y_scalers() -> tuple[StandardScaler]:
    X, y = get_data_frame().pipe(get_X_y)
    return fit_scaler(X), fit_scaler(y)
