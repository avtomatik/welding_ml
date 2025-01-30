from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential


def load_trained_model() -> Sequential:
    return tf.keras.models.load_model(
        Path(__file__).parent.parent.parent.joinpath('models')
    )
