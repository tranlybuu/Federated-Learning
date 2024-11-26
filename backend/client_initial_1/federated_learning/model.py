import tensorflow as tf
import os
from ..utils.config import MODEL_CONFIG, MODEL_DIR, INITIAL_MODEL_PATH

def create_model():
    """Create and return CNN model for MNIST."""
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

def compile_model(model):
    """Compile model with default settings."""
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_model_for_mode(mode, round_number=None):
    """Load appropriate model based on training mode."""
    if mode == 'initial':
        # For initial training, create new model
        model = create_model()
        model = compile_model(model)
    elif mode == 'additional':
        # For additional training, load initial model
        if os.path.exists(INITIAL_MODEL_PATH):
            model = tf.keras.models.load_model(INITIAL_MODEL_PATH)
        else:
            raise ValueError("Initial model not found. Please run initial training first.")
    elif mode == 'test-only':
        # For test-only, load the latest available model
        model = load_latest_model()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return model

def load_latest_model():
    """Load the latest available model from models directory."""
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    if not models:
        raise ValueError("No models found in models directory")
    
    # Sort models by creation time
    models.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
    latest_model = os.path.join(MODEL_DIR, models[-1])
    return tf.keras.models.load_model(latest_model)

def save_model(model, mode, round_number=None):
    """Save model with appropriate naming based on mode."""
    if mode == 'initial':
        path = INITIAL_MODEL_PATH
    elif mode == 'additional':
        path = os.path.join(MODEL_DIR, f'additional_model_round_{round_number}.keras')
    elif mode == 'test-only':
        path = os.path.join(MODEL_DIR, 'test_model.keras')
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.save(path)
    print(f"Model saved to: {path}")
    return path