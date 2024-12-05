import os

# Đường dẫn cơ bản
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERFACE_DIR = os.path.join(BASE_DIR, 'view')

# Thư mục models và đường dẫn
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths cho các models
INITIAL_MODEL_PATH = os.path.join(MODEL_DIR, 'initial_model.keras')
ADDITIONAL_MODEL_TEMPLATE = os.path.join(MODEL_DIR, 'additional_model_round_{}.keras')
CLIENT_MODEL_TEMPLATE = os.path.join(MODEL_DIR, 'client_{}_model.keras')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')   
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.keras')

DATA_SUMMARY_TEMPLATE = os.path.join(MODEL_DIR, '{}_data_summary.json')

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

# Federated Learning configuration
FL_CONFIG = {
    # Số rounds cho mỗi mode
    'num_rounds': {
        'initial': 3,
        'additional': 3,
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
    'batch_size': 128,
    'local_epochs': 3,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    
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

DATA_RANGES_INFO = {
    'client_ranges': {
            '1': {
                'range': (0, 10),
                'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'description': 'Training with all digits',
                'phase': 'initial'
            },
            '2': {
                'range': (0, 10),
                'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'description': 'Training with all digits',
                'phase': 'initial'
            },
            '3': {
                'range': (0, 10),
                'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'description': 'Training with all digits',
                'phase': 'additional'
            },
            '4': {
                'range': (0, 10),
                'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'description': 'Training with all digits', 
                'phase': 'additional'
            },
            '5': {
                'range': (0, 10),
                'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'description': 'Training with all digits',
                'phase': 'additional'
            }
        },
    'phase_requirements': {
        'initial': {
            'min_clients': 2,
            'max_clients': 2
        },
        'additional': {
            'min_clients': 3,
            'max_clients': 3
        }
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

# Thêm Security Configuration
SECURITY_CONFIG = {
    'secure_aggregation': {
        'enabled': True,
        'key_size': 2048,
        'signature_algorithm': 'SHA256',
        'verification_required': True
    },
    'differential_privacy': {
        'enabled': True,
        'l2_norm_clip': 1.0,
        'noise_multiplier': 1.1,
        'target_delta': 1e-5,
        'target_epsilon': 10.0
    }
}

# Cập nhật FL_CONFIG
FL_CONFIG.update({
    'security': {
        'verify_clients': True,
        'track_privacy_budget': True
    }
})

# Thêm paths cho security
SECURITY_PATHS = {
    'client_keys': os.path.join(BASE_DIR, 'security', 'keys'),
    'privacy_logs': os.path.join(BASE_DIR, 'security', 'privacy_logs'),
    'verification': os.path.join(BASE_DIR, 'security', 'verification'),
    'backups': os.path.join(BASE_DIR, 'security', 'backups')
}

# Create necessary directories
for directory in [MODEL_DIR, RESULTS_CONFIG['save_dir'], MONITOR_CONFIG['monitoring_dir']]:
    os.makedirs(directory, exist_ok=True)