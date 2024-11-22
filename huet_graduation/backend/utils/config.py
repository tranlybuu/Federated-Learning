import os

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths for models
INITIAL_MODEL_PATH = os.path.join(MODEL_DIR, 'initial_model.keras')
GLOBAL_MODEL_TEMPLATE = os.path.join(MODEL_DIR, 'global_model_round_{}.keras')
CLIENT_MODEL_TEMPLATE = os.path.join(MODEL_DIR, 'client_{}_model.keras')

# Cấu hình Federated Learning
FL_CONFIG = {
    'num_rounds': 5,
    'min_fit_clients': 2,
    'min_evaluate_clients': 2,
    'min_available_clients': 2,
    'fraction_fit': 0.5,
    'fraction_evaluate': 0.5,
}

# Cấu hình training
DATA_CONFIG = {
    'batch_size': 32,
    'local_epochs': 1,
    'num_clients': 5,
    'validation_split': 0.1,
}

# Cấu hình model
MODEL_CONFIG = {
    'num_classes': 10,
    'input_shape': (28, 28, 1),
}

# Cấu hình API
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}