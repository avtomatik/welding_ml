import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.make_dataset import get_data_frame


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray]:
    return df.iloc[:, :4].values, df.iloc[:, 4:].values


def fit_scaler(feature: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(feature)
    return scaler


def get_X_y_scalers() -> tuple[StandardScaler]:
    X, y = get_data_frame().pipe(get_X_y)
    return fit_scaler(X), fit_scaler(y)
