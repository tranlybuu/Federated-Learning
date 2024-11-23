import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
from .federated_learning.flwr_server import start_server
from .api.server import app
from .utils.config import FL_CONFIG, API_CONFIG, DATA_CONFIG

def create_parser():
    """Tạo parser với các mô tả chi tiết cho từng argument."""
    parser = argparse.ArgumentParser(
        description="""
Federated Learning System for Handwriting Recognition

This program can run in two modes:
1. Federated Learning Server (--mode fl_server)
2. API Server for inference (--mode api)

Example usage:
  Start FL server: python main.py --mode fl_server --min_clients 4 --num_rounds 8
  Start API server: python main.py --mode api
  Start client: python federated_learning/flwr_client.py --cid 0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        choices=["fl_server", "api"],
        required=True,
        help="Operation mode: 'fl_server' for Federated Learning server or 'api' for API server"
    )

    # Optional arguments with defaults from config
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=FL_CONFIG['num_rounds'],
        help=f"Number of training rounds (default: {FL_CONFIG['num_rounds']})"
    )

    parser.add_argument(
        "--min_clients",
        type=int,
        default=FL_CONFIG['min_fit_clients'],
        help=f"Minimum number of clients required for training (default: {FL_CONFIG['min_fit_clients']})"
    )

    # Advanced configuration options
    parser.add_argument(
        "--fraction_fit",
        type=float,
        default=FL_CONFIG['fraction_fit'],
        help=f"Fraction of clients used for training (default: {FL_CONFIG['fraction_fit']})"
    )

    parser.add_argument(
        "--fraction_evaluate",
        type=float,
        default=FL_CONFIG['fraction_evaluate'],
        help=f"Fraction of clients used for evaluation (default: {FL_CONFIG['fraction_evaluate']})"
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

    # Server configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument(
        "--host",
        default=API_CONFIG['host'],
        help=f"Host address for API server (default: {API_CONFIG['host']})"
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=API_CONFIG['port'],
        help=f"Port for API server (default: {API_CONFIG['port']})"
    )

    return parser

def print_configuration(args):
    """In ra cấu hình hiện tại."""
    print("\nCurrent Configuration:")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    if args.mode == "fl_server":
        print(f"Number of rounds: {args.num_rounds}")
        print(f"Minimum clients: {args.min_clients}")
        print(f"Fraction fit: {args.fraction_fit}")
        print(f"Fraction evaluate: {args.fraction_evaluate}")
        print(f"Batch size: {args.batch_size}")
        print(f"Local epochs: {args.local_epochs}")
    else:
        print(f"API Host: {args.host}")
        print(f"API Port: {args.port}")
    print("=" * 50)

def main():
    parser = create_parser()
    args = parser.parse_args()

    # Cập nhật config từ command line arguments
    FL_CONFIG['num_rounds'] = args.num_rounds
    FL_CONFIG['min_fit_clients'] = args.min_clients
    FL_CONFIG['min_evaluate_clients'] = args.min_clients
    FL_CONFIG['min_available_clients'] = args.min_clients
    FL_CONFIG['fraction_fit'] = args.fraction_fit
    FL_CONFIG['fraction_evaluate'] = args.fraction_evaluate
    
    DATA_CONFIG['num_clients'] = args.min_clients
    DATA_CONFIG['batch_size'] = args.batch_size
    DATA_CONFIG['local_epochs'] = args.local_epochs

    API_CONFIG['host'] = args.host
    API_CONFIG['port'] = args.port

    # In cấu hình hiện tại
    print_configuration(args)

    if args.mode == "fl_server":
        print("\nStarting Federated Learning Server...")
        start_server(args.num_rounds, args.min_clients, args.min_clients)
    elif args.mode == "api":
        print("\nStarting API Server...")
        app.run(host=args.host, port=args.port, debug=API_CONFIG['debug'])

if __name__ == "__main__":
    main()