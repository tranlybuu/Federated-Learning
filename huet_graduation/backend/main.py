import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
from backend.federated_learning.flwr_server import start_server
from backend.api.server import app
from backend.utils.config import FL_CONFIG, API_CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting Recognition Backend")
    parser.add_argument("--mode", choices=["fl_server", "api"], required=True)
    parser.add_argument("--num_rounds", type=int, default=FL_CONFIG['num_rounds'])
    parser.add_argument("--min_clients", type=int, default=FL_CONFIG['min_fit_clients'])
    args = parser.parse_args()

    if args.mode == "fl_server":
        start_server(args.num_rounds, args.min_clients, args.min_clients)
    elif args.mode == "api":
        app.run(host=API_CONFIG['host'], port=API_CONFIG['port'], debug=API_CONFIG['debug'])