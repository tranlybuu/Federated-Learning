import flwr as fl
import tensorflow as tf
import numpy as np
import json
import argparse
from ..utils.config import (
    MODEL_CONFIG, DATA_CONFIG, DATA_RANGES_INFO, DATA_SUMMARY_TEMPLATE,
    INITIAL_MODEL_PATH, CLIENT_MODEL_TEMPLATE, TEST_CONFIG, MODEL_DIR, PRIVACY_CONFIG
)
from .privacy.secure_aggregation import SecureAggregation
from .privacy.differential_privacy import DPFederatedLearning
from .privacy.privacy_metrics import PrivacyMetrics
import os
import base64

class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = str(cid)
        
        # Load data cho client
        self.x_train, self.y_train, self.x_test, self.y_test = load_data(cid)
        
        # Load model mới nhất từ thư mục models
        self.model = self._load_latest_model()

        # Initialize privacy components if enabled
        if PRIVACY_CONFIG['secure_aggregation']['enabled']:
            total_clients = PRIVACY_CONFIG['secure_aggregation']['min_clients_per_round']
            self.secure_agg = SecureAggregation(num_clients=total_clients)
            self.client_keys = self.secure_agg.generate_client_keys(int(cid))

        if PRIVACY_CONFIG['differential_privacy']['enabled']:
            self.dp_federated = DPFederatedLearning(
                l2_norm_clip=PRIVACY_CONFIG['differential_privacy']['l2_norm_clip'],
                noise_multiplier=PRIVACY_CONFIG['differential_privacy']['noise_multiplier'],
                num_microbatches=PRIVACY_CONFIG['differential_privacy']['num_microbatches']
            )


    def _load_latest_model(self):
        """Load model mới nhất hoặc tạo model mới nếu chưa có."""
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
        if not models:
            # Nếu chưa có model nào, tạo model mới
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), 
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            # Compile với learning rate mới
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=MODEL_CONFIG['compile_options']['optimizer_config']['learning_rate'],
                beta_1=MODEL_CONFIG['compile_options']['optimizer_config']['beta_1'],
                beta_2=MODEL_CONFIG['compile_options']['optimizer_config']['beta_2'],
                epsilon=MODEL_CONFIG['compile_options']['optimizer_config']['epsilon']
            )

            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model

        # Load model mới nhất
        models.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
        latest_model = os.path.join(MODEL_DIR, models[-1])
        return tf.keras.models.load_model(latest_model)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set model parameters 
        self.model.set_weights(parameters)
        
        # Apply DP if enabled
        if hasattr(self, 'dp_federated'):
            model = self.dp_federated.create_dp_model(
                self.model,
                learning_rate=MODEL_CONFIG['compile_options']['optimizer_config']['learning_rate']
            )
        else:
            model = self.model

        # Train model
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=DATA_CONFIG['local_epochs'],
            batch_size=config.get('batch_size', DATA_CONFIG['batch_size']),
            validation_split=config.get('validation_split', DATA_CONFIG['validation_split']),
            verbose=config.get('verbose', 1),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=2,
                    restore_best_weights=True
                )
            ]
        )

        # Save model sau khi train
        save_path = CLIENT_MODEL_TEMPLATE.format(f"client_{self.cid}")
        model.save(save_path)
        print(f"Saved client model to: {save_path}")

        weights = model.get_weights()
        metrics = {
            'accuracy': float(history.history['accuracy'][-1]),  # Convert to float
            'loss': float(history.history['loss'][-1]),  # Convert to float
            'client_id': self.cid
        }

        # Apply secure aggregation if enabled
        if hasattr(self, 'secure_agg'):
            weights, masks = self.secure_agg.mask_weights(
                weights,
                int(self.cid),
                self.client_keys
            )
            # Convert masks to base64 encoded strings
            encoded_masks = [base64.b64encode(m).decode('utf-8') for m in masks]
            metrics['masks'] = json.dumps(encoded_masks)  # Convert list to JSON string

        # Add DP metrics if enabled
        if hasattr(self, 'dp_federated'):
            epsilon, delta = self.dp_federated.compute_privacy_loss(
                len(self.x_train),
                config.get('batch_size', DATA_CONFIG['batch_size']),
                config.get('local_epochs', DATA_CONFIG['local_epochs']),
                PRIVACY_CONFIG['differential_privacy']['target_delta']
            )
            metrics.update({
                'epsilon': float(epsilon),  # Convert to float
                'delta': float(delta),  # Convert to float
                'privacy_spent': float(self.dp_federated.get_privacy_spent())  # Convert to float
            })

        return weights, len(self.x_train), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

class TestOnlyClient:
    def __init__(self):
        """Khởi tạo test-only client với model ban đầu và model mới nhất."""
        self.initial_model = self._load_initial_model()
        self.current_model = self._load_latest_model()
        self._load_test_data()
        print("\nTest-Only Client initialized:")
        print(f"Initial model: {INITIAL_MODEL_PATH}")
        print(f"Current model: {self.current_model_path}")

    def _load_initial_model(self):
        """Load model ban đầu."""
        if not os.path.exists(INITIAL_MODEL_PATH):
            raise ValueError("Initial model not found")
        return tf.keras.models.load_model(INITIAL_MODEL_PATH)

    def _load_latest_model(self):
        """Load model mới nhất từ thư mục models."""
        models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
        if not models:
            print("No additional models found, using initial model")
            self.current_model_path = INITIAL_MODEL_PATH
            return self.initial_model
        
        models.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
        self.current_model_path = os.path.join(MODEL_DIR, models[-1])
        return tf.keras.models.load_model(self.current_model_path)

    def _load_test_data(self):
        """Load dữ liệu test."""
        _, (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_test = self.x_test.reshape(-1, 28, 28, 1) / 255.0

    def compare_predictions(self, data=None):
        """So sánh dự đoán giữa model ban đầu và model hiện tại."""
        if data is None:
            # Sử dụng test data nếu không có data được cung cấp
            data = self.x_test
            actual = self.y_test
        else:
            data = np.array(data).reshape(-1, 28, 28, 1) / 255.0
            actual = None

        # Dự đoán với cả hai model
        initial_pred = self.initial_model.predict(data)
        current_pred = self.current_model.predict(data)

        # Tính toán metrics
        results = {
            'initial_model': {
                'predictions': np.argmax(initial_pred, axis=1).tolist(),
                'confidence': np.max(initial_pred, axis=1).tolist(),
            },
            'current_model': {
                'predictions': np.argmax(current_pred, axis=1).tolist(),
                'confidence': np.max(current_pred, axis=1).tolist(),
            }
        }

        # Thêm actual values và accuracy nếu có
        if actual is not None:
            initial_accuracy = np.mean(np.argmax(initial_pred, axis=1) == actual)
            current_accuracy = np.mean(np.argmax(current_pred, axis=1) == actual)
            results.update({
                'actual': actual.tolist(),
                'initial_accuracy': float(initial_accuracy),
                'current_accuracy': float(current_accuracy),
            })

        return results

    def evaluate_models(self):
        """Đánh giá chi tiết cả hai models trên tập test."""
        # Đánh giá model ban đầu
        initial_loss, initial_accuracy = self.initial_model.evaluate(
            self.x_test, self.y_test,
            batch_size=TEST_CONFIG['batch_size'],
            verbose=0
        )

        # Đánh giá model hiện tại
        current_loss, current_accuracy = self.current_model.evaluate(
            self.x_test, self.y_test,
            batch_size=TEST_CONFIG['batch_size'],
            verbose=0
        )

        evaluation = {
            'initial_model': {
                'loss': float(initial_loss),
                'accuracy': float(initial_accuracy),
            },
            'current_model': {
                'loss': float(current_loss),
                'accuracy': float(current_accuracy),
            },
            'improvement': {
                'accuracy': float(current_accuracy - initial_accuracy),
                'loss': float(initial_loss - current_loss),
            }
        }

        return evaluation

def load_data(cid):
    """Load và phân tích dữ liệu MNIST cho client."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize và reshape
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # Kiểm tra client ID có hợp lệ không
    str_cid = str(cid)
    if str_cid not in DATA_RANGES_INFO['client_ranges']:
        raise ValueError(f"Invalid client ID: {cid}")

    # Lấy thông tin range và labels cho client
    client_info = DATA_RANGES_INFO['client_ranges'][str_cid]
    allowed_labels = client_info['labels']

    # Lọc dữ liệu train theo labels được chỉ định
    train_mask = np.isin(y_train, allowed_labels)
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]

    # Lọc dữ liệu test theo labels được chỉ định
    test_mask = np.isin(y_test, allowed_labels)
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]

    # Tính phân bố chi tiết cho tập train
    train_labels, train_counts = np.unique(y_train_filtered, return_counts=True)
    train_distribution = {
        int(label): int(count) for label, count in zip(train_labels, train_counts)
    }

    # Tính phân bố chi tiết cho tập test
    test_labels, test_counts = np.unique(y_test_filtered, return_counts=True)
    test_distribution = {
        int(label): int(count) for label, count in zip(test_labels, test_counts)
    }

    # Tạo summary chi tiết về dữ liệu
    data_summary = {
        'client_id': cid,
        'train': {
            'total_samples': len(y_train_filtered),
            'samples_per_label': train_distribution,
            'labels_distribution': {
                str(label): f"{(count/len(y_train_filtered)*100):.2f}%"
                for label, count in train_distribution.items()
            }
        },
        'test': {
            'total_samples': len(y_test_filtered),
            'samples_per_label': test_distribution,
            'labels_distribution': {
                str(label): f"{(count/len(y_test_filtered)*100):.2f}%"
                for label, count in test_distribution.items()
            }
        },
        'allowed_labels': allowed_labels,
        'description': client_info['description']
    }

    # Lưu summary
    summary_path = DATA_SUMMARY_TEMPLATE.format(f"client_{cid}")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(data_summary, f, indent=4)

    # In thông tin chi tiết
    print(f"\nClient {cid} Dataset Summary:")
    print("=" * 50)
    print(f"Description: {client_info['description']}")
    print(f"Allowed labels: {allowed_labels}")
    print("\nTraining Data:")
    print(f"Total samples: {len(y_train_filtered)}")
    print("Distribution by label:")
    for label, count in sorted(train_distribution.items()):
        percentage = (count/len(y_train_filtered)*100)
        print(f"  Label {label}: {count} samples ({percentage:.2f}%)")

    print("\nTest Data:")
    print(f"Total samples: {len(y_test_filtered)}")
    print("Distribution by label:")
    for label, count in sorted(test_distribution.items()):
        percentage = (count/len(y_test_filtered)*100)
        print(f"  Label {label}: {count} samples ({percentage:.2f}%)")
    print("=" * 50)

    return x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered

def create_client_parser():
    """Tạo parser với các mô tả chi tiết cho client."""
    parser = argparse.ArgumentParser(
        description="""
Federated Learning Client for Handwriting Recognition

This program runs a Federated Learning client that participates in the training process.
Each client loads a portion of the MNIST dataset and trains the model locally.

Supported modes:
- initial: First training phase with data range 0-4
- additional: Second training phase with data range 5-9
- test-only: For inference only

Example usage:
  Initial training:    python flwr_client.py --mode initial --cid 0
  Additional training: python flwr_client.py --mode additional --cid 0
  Test only:          python flwr_client.py --mode test-only

Note: Make sure the server is running before starting the client.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--cid",
        type=int,
        required=True,
        help="Client ID (required, must be between 1-5)"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="Server address in format host:port (default: 127.0.0.1:8080)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DATA_CONFIG['batch_size'],
        help=f"Batch size for training (default: {DATA_CONFIG['batch_size']})"
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=DATA_CONFIG['local_epochs'],
        help=f"Number of local epochs (default: {DATA_CONFIG['local_epochs']})"
    )

    # Advanced configuration
    advanced_group = parser.add_argument_group('Advanced Configuration')
    advanced_group.add_argument(
        "--validation_split",
        type=float,
        default=DATA_CONFIG['validation_split'],
        help=f"Fraction of data to use for validation (default: {DATA_CONFIG['validation_split']})"
    )

    advanced_group.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level for training (0: silent, 1: progress bar, 2: one line per epoch)"
    )

    return parser

def print_client_config(args):
    """In ra cấu hình hiện tại của client."""
    print("\nClient Configuration:")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Client ID: {args.cid}")
    print(f"Server Address: {args.server_address}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Local Epochs: {args.local_epochs}")
    print(f"Validation Split: {args.validation_split}")
    print(f"Verbose Level: {args.verbose}")
    print("=" * 50)

def start_client(args):
    """Khởi động client dựa trên mode."""
    if args.mode == 'test-only':
        return TestOnlyClient()
        
    # Load data cho client theo mode
    x_train, y_train, x_test, y_test = load_data(args.cid, args.mode)
    
    # Cập nhật DATA_CONFIG với các giá trị từ command line
    DATA_CONFIG.update({
        'batch_size': args.batch_size,
        'local_epochs': args.local_epochs,
        'validation_split': args.validation_split
    })
    
    # Tạo client
    client = MnistClient(args.cid, args.mode, x_train, y_train, x_test, y_test)
    
    # Start Flower client
    print(f"\nConnecting to server at {args.server_address}...")
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )

def main():
    # Parse arguments
    parser = create_client_parser()
    args = parser.parse_args()

    # Print configuration
    print("\nClient Configuration:")
    print("=" * 50)
    print(f"Client ID: {args.cid}")
    print(f"Server Address: {args.server_address}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Local Epochs: {args.local_epochs}")
    print("=" * 50)

    try:
        # Create and start client - chỉ truyền cid
        client = MnistClient(args.cid)

        print(f"\nConnecting to server at {args.server_address}...")
        fl.client.start_client(
            server_address=args.server_address,
            client=client.to_client()
        )

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the server is running")
        print("2. Check if the server address is correct")
        print("3. Verify that your Client ID is valid (1-5)")
        print("4. Ensure you have enough memory for the specified batch size")
        raise

if __name__ == "__main__":
    main()