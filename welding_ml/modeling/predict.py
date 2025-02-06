import tensorflow as tf
from tensorflow.keras import Sequential

from welding_ml.config import MODEL_DIR


def load_trained_model() -> Sequential:
    return tf.keras.models.load_model(MODEL_DIR)
