import flwr as fl
import tensorflow as tf
from .model import create_model, load_model_for_mode, save_model
from ..utils.config import (
    FL_CONFIG, MODEL_DIR, TRAINING_CONFIG,
    INITIAL_MODEL_PATH, MODEL_TEMPLATES
)
import os
import json
import numpy as np

class FederatedServer(fl.server.strategy.FedAvg):
    def __init__(self, mode='initial', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.round_results = []
        self.current_round = 0
        self.best_accuracy = 0.0
        
        # Khởi tạo hoặc load model dựa trên mode
        if mode == 'initial':
            self.model = create_model()
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            # Lưu model ban đầu
            model_path = MODEL_TEMPLATES['initial']
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
        else:
            initial_path = MODEL_TEMPLATES['initial']
            if not os.path.exists(initial_path):
                raise ValueError("Initial model not found. Please run initial training first.")
            self.model = tf.keras.models.load_model(initial_path)
        
        print(f"\nInitializing server in {mode} mode")

    def aggregate_fit(self, server_round, results, failures):
        """Tổng hợp kết quả training từ clients."""
        self.current_round = server_round
        print(f"\nRound {server_round} ({self.mode} mode):")
        print(f"Active clients: {len(results)}")
        print(f"Failures: {len(failures)}")
        
        if not results:
            return None, {}

        # Collect training results
        weights = []
        num_examples = []
        metrics = []
        
        for client_proxy, fit_res in results:
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights.append(client_weights)
            num_examples.append(fit_res.num_examples)
            metrics.append(fit_res.metrics)
            print(f"Client metrics: {fit_res.metrics}")

        total_examples = sum(num_examples)
        if total_examples == 0:
            return None, {}

        # Aggregate weights using weighted average
        weighted_weights = [
            np.sum([
                w[i] * n for w, n in zip(weights, num_examples)
            ], axis=0) / total_examples
            for i in range(len(weights[0]))
        ]

        # Update model and evaluate
        self.model.set_weights(weighted_weights)
        test_loss, test_accuracy = self._evaluate_global_model()
        print(f"Round {server_round} results - Loss: {test_loss}, Accuracy: {test_accuracy}")

        # Save results
        round_metrics = {
            'round': server_round,
            'mode': self.mode,
            'loss': float(test_loss),
            'accuracy': float(test_accuracy),
            'num_clients': len(results),
            'client_metrics': metrics
        }
        self.round_results.append(round_metrics)

        # Save best model if accuracy improved
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            best_model_path = MODEL_TEMPLATES['best'].format(self.mode)
            self.model.save(best_model_path)
            print(f"Saved best model with accuracy {test_accuracy:.4f}")

        # Save current round model
        current_model_path = MODEL_TEMPLATES['global'].format(server_round)
        self.model.save(current_model_path)

        # Save final results if this is the last round
        if server_round == self.num_rounds:
            self._save_final_results()

        # Convert weights back to Parameters
        return fl.common.ndarrays_to_parameters(weighted_weights), {
            "loss": float(test_loss),
            "accuracy": float(test_accuracy)
        }

    def _evaluate_global_model(self):
        """Evaluate model on test data."""
        _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        return self.model.evaluate(x_test, y_test, verbose=0)

    def _save_final_results(self):
        """Save final results and training history."""
        # Create results directory if it doesn't exist
        results_dir = os.path.join(MODEL_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Save detailed results
        results_path = os.path.join(results_dir, f'{self.mode}_training_results.json')
        final_results = {
            'mode': self.mode,
            'round_results': self.round_results,
            'best_accuracy': float(self.best_accuracy),
            'total_rounds': self.current_round,
            'final_accuracy': float(self.round_results[-1]['accuracy']),
            'final_loss': float(self.round_results[-1]['loss']),
            'data_ranges': TRAINING_CONFIG['data_ranges'][self.mode]
        }
        
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
            
        print(f"\nTraining Summary ({self.mode} mode):")
        print("=" * 50)
        print(f"Total Rounds: {self.current_round}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"Final Accuracy: {self.round_results[-1]['accuracy']:.4f}")
        print(f"Final Loss: {self.round_results[-1]['loss']:.4f}")
        print(f"Results saved to: {results_path}")
        print("=" * 50)

    def get_model_parameters(self):
        """Get current model parameters."""
        return self.model.get_weights()

def start_server(mode, num_rounds=None, min_fit_clients=None, min_evaluate_clients=None):
    """Start Flower server with specified configuration."""
    # Set default values if not provided
    if num_rounds is None:
        num_rounds = FL_CONFIG['num_rounds'].get(mode, 3)
    
    if min_fit_clients is None:
        min_fit_clients = FL_CONFIG['min_fit_clients'].get(mode, 2)
    
    if min_evaluate_clients is None:
        min_evaluate_clients = FL_CONFIG['min_evaluate_clients'].get(mode, 2)

    # Print server configuration
    print("\nServer Configuration:")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Minimum fit clients: {min_fit_clients}")
    print(f"Minimum evaluate clients: {min_evaluate_clients}")
    print(f"Data ranges: {TRAINING_CONFIG['data_ranges'][mode]}")
    print("=" * 50)

    # Initialize strategy
    strategy = FederatedServer(
        mode=mode,
        fraction_fit=FL_CONFIG['fraction_fit'],
        fraction_evaluate=FL_CONFIG['fraction_evaluate'],
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=FL_CONFIG['min_available_clients'].get(mode, min_fit_clients)
    )
    strategy.num_rounds = num_rounds

    # Start server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )