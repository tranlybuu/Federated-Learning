import os

# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.h5')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Cấu hình Federated Learning
FL_CONFIG = {
    'num_rounds': 3,
    'min_fit_clients': 2,
    'min_evaluate_clients': 2,
    'min_available_clients': 2,
    'fraction_fit': 0.5,
    'fraction_evaluate': 0.5,
}

# Model config
MODEL_CONFIG = {
    'num_classes': 10,
    'input_shape': (28, 28, 1),
}

# API server config
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
}

# Data config
DATA_CONFIG = {
    'batch_size': 32,
    'validation_split': 0.1,
    'num_clients': 5,
}

# Random seed
RANDOM_SEED = 42