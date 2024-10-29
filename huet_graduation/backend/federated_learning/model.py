import tensorflow as tf
from backend.utils.config import MODEL_CONFIG

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=MODEL_CONFIG['input_shape']),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(MODEL_CONFIG['num_classes'], activation='softmax')
    ])
    return model