import flwr as fl
import tensorflow as tf
import argparse
from ..utils.config import INITIAL_MODEL_PATH, CLIENT_MODEL_TEMPLATE, DATA_CONFIG

class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid, x_train, y_train, x_test, y_test):
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Load model từ server
        self.model = tf.keras.models.load_model(INITIAL_MODEL_PATH)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set trọng số nhận được từ server
        self.model.set_weights(parameters)
        
        # Train local
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=DATA_CONFIG['local_epochs'],
            batch_size=DATA_CONFIG['batch_size'],
            validation_split=DATA_CONFIG['validation_split'],
            verbose=1
        )
        
        # Lưu model sau khi train
        save_path = CLIENT_MODEL_TEMPLATE.format(self.cid)
        self.model.save(save_path, save_format='keras')
        
        return self.model.get_weights(), len(self.x_train), {
            "accuracy": history.history['accuracy'][-1],
            "loss": history.history['loss'][-1]
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

def load_data(cid):
    """Load và tiền xử lý dữ liệu MNIST cho client."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize và reshape
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    # Chia dữ liệu cho client cụ thể
    total_clients = DATA_CONFIG['num_clients']
    samples_per_client = len(x_train) // total_clients
    
    # Đảm bảo việc phân phối dữ liệu cho số lượng client bất kỳ
    start_idx = cid * samples_per_client
    end_idx = start_idx + samples_per_client
    if cid == total_clients - 1:
        # Client cuối cùng sẽ lấy tất cả dữ liệu còn lại
        end_idx = len(x_train)
    
    client_data = (
        x_train[start_idx:end_idx], 
        y_train[start_idx:end_idx],
        x_test,
        y_test
    )
    
    print(f"\nClient {cid} initialized:")
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

Example usage:
  Basic usage: python flwr_client.py --cid 0
  Custom configuration: python flwr_client.py --cid 0 --server_address "127.0.0.1:8080" --batch_size 64

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
    print(f"Client ID: {args.cid}")
    print(f"Server Address: {args.server_address}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Local Epochs: {args.local_epochs}")
    print(f"Validation Split: {args.validation_split}")
    print(f"Verbose Level: {args.verbose}")
    print("=" * 50)

def start_client(args):
    """Khởi động Flower client với các cấu hình đã cho."""
    # Load data cho client
    print(f"\nInitializing Client {args.cid}...")
    x_train, y_train, x_test, y_test = load_data(args.cid)
    
    # Cập nhật DATA_CONFIG với các giá trị từ command line
    DATA_CONFIG['batch_size'] = args.batch_size
    DATA_CONFIG['local_epochs'] = args.local_epochs
    DATA_CONFIG['validation_split'] = args.validation_split
    
    # Tạo client
    client = MnistClient(args.cid, x_train, y_train, x_test, y_test)
    
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
        raise

if __name__ == "__main__":
    main()