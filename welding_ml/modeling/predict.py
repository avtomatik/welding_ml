import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential

from welding_ml.config import MODEL_DIR
from welding_ml.services import validate_input


def load_trained_model() -> Sequential:
    return tf.keras.models.load_model(MODEL_DIR)


def make_prediction_scaled(
    input_features: str,
    model: Sequential,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler
) -> np.ndarray:

    input_features_scaled = scaler_X.transform(
        np.array(validate_input(input_features)).reshape(1, -1)
    )
    y_pred = scaler_y.inverse_transform(
        model.predict(input_features_scaled)
    )

    return y_pred
