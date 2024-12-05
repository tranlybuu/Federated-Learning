import argparse
from .utils.config import (
    FL_CONFIG, API_CONFIG, MODEL_DIR, INITIAL_MODEL_PATH,
    SECURITY_CONFIG, SECURITY_PATHS
)
from .federated_learning.flwr_server import start_server 
from .api.server import app
from .security.crypto_utils import CryptoManager
from .security.privacy_metrics import PrivacyMetricsTracker
from .security.verification import ClientVerifier
import os
import sys

def create_parser():
    parser = argparse.ArgumentParser(
        description="""
Federated Learning Server for Handwriting Recognition with Security Features

Modes:
1. Initial Training (--mode initial):
   - First phase with 2 clients
   - Creates initial model

2. Additional Training (--mode additional):
   - Second phase with 3 clients
   - Requires initial model

3. API Server (--mode api):
   - Serves models via REST API
   
Security Features:
--secure-aggregation: Enable secure aggregation
--differential-privacy: Enable differential privacy
--verification: Enable client verification
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=['initial', 'additional', 'api'],
        required=True,
        help="Server operation mode"
    )

    parser.add_argument(
        "--server",
        action="store_true",
        required=True,
        help="Run as server"
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        help="Number of training rounds"
    )

    # Security arguments
    security_group = parser.add_argument_group('Security Configuration')
    security_group.add_argument(
        "--secure-aggregation",
        action="store_true",
        help="Enable secure aggregation"
    )
    security_group.add_argument(
        "--differential-privacy",
        action="store_true",
        help="Enable differential privacy"
    )
    security_group.add_argument(
        "--verification",
        action="store_true",
        help="Enable client verification"
    )
    security_group.add_argument(
        "--privacy-budget",
        type=float,
        default=SECURITY_CONFIG['differential_privacy']['target_epsilon'],
        help="Privacy budget (epsilon) for differential privacy"
    )

    return parser

def initialize_security_components(args):
    """Initialize server-side security components."""
    components = {}
    
    if args.secure_aggregation:
        components['crypto_manager'] = CryptoManager(SECURITY_PATHS['client_keys'])
        
    if args.differential_privacy:
        components['privacy_tracker'] = PrivacyMetricsTracker(
            SECURITY_PATHS['privacy_logs'],
            target_epsilon=args.privacy_budget
        )
        
    if args.verification:
        components['client_verifier'] = ClientVerifier(SECURITY_PATHS['verification'])
        
    return components

def validate_args(args):
    if args.mode == 'api' and args.server:
        raise ValueError("API mode doesn't require --server flag")

    if args.mode == 'additional':
        if not os.path.exists(INITIAL_MODEL_PATH):
            raise ValueError("Initial model not found. Please run initial training first.")

def print_configuration(args, security_components):
    print("\nServer Configuration:")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Number of rounds: {args.num_rounds or FL_CONFIG['num_rounds'][args.mode]}")
    print(f"Minimum clients: {FL_CONFIG['min_fit_clients'][args.mode]}")

    print("\nSecurity Configuration:")
    print(f"Secure Aggregation: {'Enabled' if args.secure_aggregation else 'Disabled'}")
    print(f"Differential Privacy: {'Enabled' if args.differential_privacy else 'Disabled'}")
    print(f"Client Verification: {'Enabled' if args.verification else 'Disabled'}")
    if args.differential_privacy:
        print(f"Privacy Budget (ε): {args.privacy_budget}")
    print("=" * 50)

def initialize_directories():
    directories = [
        MODEL_DIR,
        os.path.join(MODEL_DIR, 'results'),
        SECURITY_PATHS['client_keys'],
        SECURITY_PATHS['privacy_logs'],
        SECURITY_PATHS['verification']
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    parser = create_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        initialize_directories()
        security_components = initialize_security_components(args)
        print_configuration(args, security_components)

        if args.mode == 'api':
            app.run(
                host=API_CONFIG['host'],
                port=API_CONFIG['port'],
                debug=API_CONFIG['debug']
            )
        else:
            start_server(
                mode=args.mode,
                num_rounds=args.num_rounds,
                min_fit_clients=FL_CONFIG['min_fit_clients'][args.mode],
                min_evaluate_clients=FL_CONFIG['min_evaluate_clients'][args.mode],
                security_components=security_components
            )

    except Exception as e:
        print(f"\nServer Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if initial model exists (for additional mode)")
        print("2. Verify security components initialization")
        print("3. Check security paths and permissions")
        raise

if __name__ == "__main__":
    main()