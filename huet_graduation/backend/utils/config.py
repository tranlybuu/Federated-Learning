import os

# Đường dẫn cơ bản
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thư mục models và đường dẫn
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths cho các models
INITIAL_MODEL_PATH = os.path.join(MODEL_DIR, 'initial_model.keras')
ADDITIONAL_MODEL_TEMPLATE = os.path.join(MODEL_DIR, 'additional_model_round_{}.keras')
CLIENT_MODEL_TEMPLATE = os.path.join(MODEL_DIR, 'client_{}_model.keras')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.keras')

# Templates cho tên file models
MODEL_TEMPLATES = {
    # Basic models
    'initial': os.path.join(MODEL_DIR, 'initial_model.keras'),
    'final': os.path.join(MODEL_DIR, 'final_model.keras'),
    'best': os.path.join(MODEL_DIR, 'best_{}_model.keras'),
    
    # Training phase models
    'global': os.path.join(MODEL_DIR, 'global_model_round_{}.keras'),
    'client': os.path.join(MODEL_DIR, 'client_{}_model.keras'),
    'additional': os.path.join(MODEL_DIR, 'additional_model_round_{}.keras')
}

# Training mode và data ranges configuration
TRAINING_CONFIG = {
    'modes': ['initial', 'additional', 'test-only'],
    'data_ranges': {
        'initial': {
            '0': (0, 3),  # Client 1: data 0-2
            '1': (3, 5),  # Client 2: data 3-4
        },
        'additional': {
            '0': (5, 7),  # Client 1: data 5-6
            '1': (7, 9),  # Client 2: data 7-8
            '2': (5, 10), # Client 3: data 5,9
        }
    }
}

# Federated Learning configuration
FL_CONFIG = {
    # Số rounds cho mỗi mode
    'num_rounds': {
        'initial': 3,
        'additional': 5,
    },
    
    # Số clients tối thiểu cho mỗi mode
    'min_fit_clients': {
        'initial': 2,
        'additional': 3,
    },
    
    # Số clients tối thiểu cho evaluation
    'min_evaluate_clients': {
        'initial': 2,
        'additional': 3,
    },
    
    # Số clients tối thiểu cần có
    'min_available_clients': {
        'initial': 2,
        'additional': 3,
    },
    
    # Tỷ lệ clients sử dụng cho training/evaluation
    'fraction_fit': 0.7,
    'fraction_evaluate': 0.7,
}

# Data và training configuration
DATA_CONFIG = {
    # Training hyperparameters
    'batch_size': 32,
    'local_epochs': 1,
    'learning_rate': 0.001,
    'validation_split': 0.1,
    
    # Verbose levels
    'training_verbose': 1,
    'evaluation_verbose': 0,
    
    # Client configuration
    'num_clients': {
        'initial': 2,
        'additional': 3,
    },
}

# Model configuration
MODEL_CONFIG = {
    'input_shape': (28, 28, 1),
    'num_classes': 10,
    'model_architecture': {
        'conv_layers': [
            {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        ],
        'dense_layers': [
            {'units': 64, 'activation': 'relu'},
            {'units': 10, 'activation': 'softmax'},
        ],
        'pooling_size': (2, 2),
    },
    'compile_options': {
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy'],
    }
}

# API Server configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'model_endpoints': {
        'recognition': '/recognize',
        'health': '/health',
        'model_info': '/model-info',
    },
    'max_request_size': 16 * 1024 * 1024,  # 16MB
    'allowed_extensions': ['png', 'jpg', 'jpeg'],
}

# Logging configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'federated_learning.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    }
}

# Results configuration
RESULTS_CONFIG = {
    'save_dir': os.path.join(MODEL_DIR, 'results'),
    'metrics': ['loss', 'accuracy'],
    'save_format': 'json',
    'plot_metrics': True,
    'save_plots': True,
}

# Monitoring configuration
MONITOR_CONFIG = {
    'enabled': True,
    'metrics_interval': 5,  # seconds
    'save_system_metrics': True,
    'monitoring_dir': os.path.join(BASE_DIR, 'monitoring'),
}

# Testing configuration
TEST_CONFIG = {
    'test_split': 0.2,
    'batch_size': 32,
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'confusion_matrix': True,
    'save_predictions': True,
}

# Create necessary directories
for directory in [MODEL_DIR, RESULTS_CONFIG['save_dir'], MONITOR_CONFIG['monitoring_dir']]:
    os.makedirs(directory, exist_ok=True)