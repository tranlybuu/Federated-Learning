import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import flwr as fl
from .utils.config import (
    FL_CONFIG, API_CONFIG, TRAINING_CONFIG,
    MODEL_DIR, INITIAL_MODEL_PATH
)
from .federated_learning.flwr_server import start_server
from .federated_learning.flwr_client import start_client
from .api.server import app
import os
import json

def create_parser():
    """Create argument parser with detailed help messages."""
    parser = argparse.ArgumentParser(
        description="""
Federated Learning System for Handwriting Recognition

This program supports multiple modes:

1. Initial Training (--mode initial):
   - First phase of training with 2 clients
   - Uses data ranges 0-4
   - Creates initial model

2. Additional Training (--mode additional):
   - Second phase with 3 clients
   - Uses data ranges 5-9
   - Requires initial model from first phase

3. Test Only (--mode test-only):
   - For inference and model comparison
   - Requires at least one trained model

4. API Server (--mode api):
   - Serves trained models via REST API
   - Uses the best available model

Example usage:
  Start initial training server:     python main.py --mode initial --server
  Start additional training server:  python main.py --mode additional --server
  Start client:                     python main.py --mode initial --client --cid 0
  Start API server:                 python main.py --mode api

Note: For client mode, --cid is required.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        choices=['initial', 'additional', 'test-only', 'api'],
        required=True,
        help="Operation mode"
    )

    # Server/Client selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--server",
        action="store_true",
        help="Run as server"
    )
    group.add_argument(
        "--client",
        action="store_true",
        help="Run as client"
    )

    # Client specific arguments
    parser.add_argument(
        "--cid",
        type=int,
        help="Client ID (required for client mode)"
    )

    # Optional configuration
    parser.add_argument(
        "--num_rounds",
        type=int,
        help="Number of training rounds"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=FL_CONFIG.get('batch_size', 32),
        help="Batch size for training"
    )

    return parser

def validate_args(args):
    """Validate command line arguments."""
    if args.mode == 'api':
        if args.server or args.client:
            raise ValueError("API mode doesn't require --server or --client flag")
        return

    if not (args.server or args.client):
        raise ValueError("Must specify either --server or --client")

    if args.client and args.cid is None:
        raise ValueError("Client mode requires --cid")

    if args.mode == 'additional':
        if not os.path.exists(INITIAL_MODEL_PATH):
            raise ValueError("Initial model not found. Please run initial training first.")

def print_configuration(args):
    """Print current configuration."""
    print("\nCurrent Configuration:")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Running as: {'Server' if args.server else 'Client' if args.client else 'API'}")
    
    if args.server:
        print(f"Number of rounds: {args.num_rounds or FL_CONFIG['num_rounds'][args.mode]}")
        print(f"Minimum clients: {FL_CONFIG['min_fit_clients'][args.mode]}")
        print(f"Data ranges: {TRAINING_CONFIG['data_ranges'][args.mode]}")
    elif args.client:
        print(f"Client ID: {args.cid}")
        print(f"Batch size: {args.batch_size}")
    else:  # API mode
        print(f"Host: {API_CONFIG['host']}")
        print(f"Port: {API_CONFIG['port']}")
    
    print("=" * 50)

def initialize_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        MODEL_DIR,
        os.path.join(MODEL_DIR, 'results'),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Validate arguments
        validate_args(args)

        # Create necessary directories
        initialize_directories()

        # Print configuration
        print_configuration(args)

        # Execute based on mode and role
        if args.mode == 'api':
            app.run(
                host=API_CONFIG['host'],
                port=API_CONFIG['port'],
                debug=API_CONFIG['debug']
            )
        elif args.server:
            start_server(
                mode=args.mode,
                num_rounds=args.num_rounds,
                min_fit_clients=FL_CONFIG['min_fit_clients'][args.mode],
                min_evaluate_clients=FL_CONFIG['min_evaluate_clients'][args.mode]
            )
        else:  # client mode
            start_client(args)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        if args.mode == 'additional':
            print("1. Ensure initial training has been completed")
            print("2. Check if initial model exists")
        if args.client:
            print("3. Make sure the server is running")
            print("4. Verify that your Client ID is unique")
            print("5. Check if you have enough memory for the specified batch size")
        print("6. Verify that all directories have write permissions")
        raise

if __name__ == "__main__":
    main()