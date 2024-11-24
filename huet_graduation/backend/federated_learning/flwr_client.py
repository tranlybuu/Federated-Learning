import flwr as fl
import tensorflow as tf
import numpy as np
import argparse
from ..utils.config import (
    TRAINING_CONFIG, DATA_CONFIG, 
    INITIAL_MODEL_PATH, CLIENT_MODEL_TEMPLATE, TEST_CONFIG, MODEL_DIR
)
from .model import load_model_for_mode
import os

class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid, mode, x_train, y_train, x_test, y_test):
        self.cid = cid
        self.mode = mode
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Load model phù hợp với mode
        if mode == 'initial':
            # Với mode initial, load model từ server
            self.model = tf.keras.models.load_model(INITIAL_MODEL_PATH)
        else:
            # Với mode additional, load model cuối cùng từ giai đoạn initial
            self.model = load_model_for_mode(mode)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set trọng số nhận được từ server
        self.model.set_weights(parameters)
        
        # Train local
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=config.get('local_epochs', DATA_CONFIG['local_epochs']),
            batch_size=config.get('batch_size', DATA_CONFIG['batch_size']),
            validation_split=config.get('validation_split', DATA_CONFIG['validation_split']),
            verbose=config.get('verbose', 1)
        )
        
        # Lưu model sau khi train
        save_path = CLIENT_MODEL_TEMPLATE.format(f"{self.mode}_{self.cid}")
        self.model.save(save_path)
        print(f"Saved client model to: {save_path}")
        
        return self.model.get_weights(), len(self.x_train), {
            "accuracy": history.history['accuracy'][-1],
            "loss": history.history['loss'][-1]
        }

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

def load_data(cid, mode):
    """Load và tiền xử lý dữ liệu MNIST cho client theo mode."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize và reshape
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    # Lấy data range cho mode và client hiện tại
    data_range = TRAINING_CONFIG['data_ranges'][mode][str(cid)]
    start, end = data_range
    
    # Tính số lượng samples cho mỗi số (0-9)
    samples_per_digit = len(x_train) // 10
    
    # Lấy dữ liệu theo range
    start_idx = start * samples_per_digit
    end_idx = end * samples_per_digit
    
    client_data = (
        x_train[start_idx:end_idx], 
        y_train[start_idx:end_idx],
        x_test,
        y_test
    )
    
    print(f"\nClient {cid} ({mode} mode) initialized:")
    print(f"Data range: {start}-{end}")
    print(f"Training samples: {end_idx - start_idx}")
    print(f"Test samples: {len(x_test)}")
    
    return client_data

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
        help="Client ID (required, must be unique for each client)"
    )
    
    parser.add_argument(
        "--mode",
        choices=['initial', 'additional', 'test-only'],
        required=True,
        help="Training mode"
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
        help=f"Number of local training epochs (default: {DATA_CONFIG['local_epochs']})"
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

    # In cấu hình
    print_client_config(args)

    try:
        # Khởi động client
        start_client(args)
    except Exception as e:
        print(f"\nError starting client: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the server is running")
        print("2. Check if the server address is correct")
        print("3. Verify that the Client ID is unique")
        print("4. Ensure you have enough memory for the specified batch size")
        if args.mode == 'additional':
            print("5. Verify that initial training has been completed")
        raise

if __name__ == "__main__":
    main()