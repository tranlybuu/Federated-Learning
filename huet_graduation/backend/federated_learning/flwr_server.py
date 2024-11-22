import flwr as fl
import tensorflow as tf
from .model import create_model
from ..utils.config import (
    FL_CONFIG, INITIAL_MODEL_PATH, 
    GLOBAL_MODEL_TEMPLATE, MODEL_DIR
)
import os
import numpy as np
import json

class FederatedServer(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Khởi tạo model ban đầu
        self.model = create_model()
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # Lưu model ban đầu
        os.makedirs(os.path.dirname(INITIAL_MODEL_PATH), exist_ok=True)
        self.model.save(INITIAL_MODEL_PATH)
        self.round_results = []
        self.current_round = 0
        self.best_accuracy = 0.0

    def aggregate_fit(self, server_round, results, failures):
        """Tổng hợp trọng số từ các clients."""
        self.current_round = server_round
        print(f"Round {server_round}: Aggregating weights from {len(results)} clients")
        
        if not results:
            print("No results received from clients")
            return None, {}

        # Thu thập trọng số từ các clients
        weights = []
        num_examples = []
        
        for client_proxy, fit_res in results:
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights.append(client_weights)
            num_examples.append(fit_res.num_examples)

        total_examples = sum(num_examples)
        if total_examples == 0:
            return None, {}

        # Tính trọng số trung bình có trọng số
        weighted_weights = [
            np.sum([w[i] * n for w, n in zip(weights, num_examples)], axis=0) / total_examples
            for i in range(len(weights[0]))
        ]

        # Cập nhật model và đánh giá
        self.model.set_weights(weighted_weights)
        test_loss, test_accuracy = self._evaluate_global_model()
        print(f"Round {server_round} results - Loss: {test_loss}, Accuracy: {test_accuracy}")
        
        # Lưu kết quả và model
        self.round_results.append({
            'round': server_round,
            'loss': test_loss,
            'accuracy': test_accuracy
        })
        
        # Cập nhật và lưu best model nếu accuracy tốt hơn
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            self.model.save(os.path.join(MODEL_DIR, 'best_model.keras'))
        
        # Lưu model của round hiện tại
        save_path = GLOBAL_MODEL_TEMPLATE.format(server_round)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)

        # Nếu là round cuối cùng, lưu final model và training history
        if server_round == self.num_rounds:
            self._save_final_results()
        
        metrics = {
            "loss": float(test_loss),
            "accuracy": float(test_accuracy)
        }
        return fl.common.ndarrays_to_parameters(weighted_weights), metrics

    def _save_final_results(self):
        """Lưu final model và kết quả training."""
        # Lưu final model
        final_model_path = os.path.join(MODEL_DIR, 'final_model.keras')
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Lưu training history
        history_path = os.path.join(MODEL_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'round_results': self.round_results,
                'best_accuracy': float(self.best_accuracy),
                'total_rounds': self.current_round,
                'final_accuracy': float(self.round_results[-1]['accuracy']) if self.round_results else 0.0,
                'final_loss': float(self.round_results[-1]['loss']) if self.round_results else 0.0,
            }, f, indent=4)
        print(f"Training history saved to: {history_path}")

        # Tạo summary report
        print("\nTraining Summary:")
        print("=" * 50)
        print(f"Total Rounds: {self.current_round}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"Final Accuracy: {self.round_results[-1]['accuracy']:.4f}")
        print(f"Final Loss: {self.round_results[-1]['loss']:.4f}")
        print("=" * 50)

    def aggregate_evaluate(self, server_round, results, failures):
        """Tổng hợp kết quả đánh giá từ clients."""
        if not results:
            return None, {}

        accuracies = []
        losses = []
        examples = []

        for _, eval_res in results:
            accuracies.append(eval_res.metrics.get("accuracy", 0) * eval_res.num_examples)
            losses.append(eval_res.loss * eval_res.num_examples)
            examples.append(eval_res.num_examples)

        if not examples:
            return None, {}

        avg_accuracy = sum(accuracies) / sum(examples)
        avg_loss = sum(losses) / sum(examples)

        metrics = {
            "accuracy": float(avg_accuracy),
            "loss": float(avg_loss)
        }

        return avg_loss, metrics

    def _evaluate_global_model(self):
        """Đánh giá model toàn cục trên tập test."""
        _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        return self.model.evaluate(x_test, y_test, verbose=0)

def start_server(num_rounds, min_fit_clients, min_evaluate_clients):
    strategy = FederatedServer(
        fraction_fit=FL_CONFIG['fraction_fit'],
        fraction_evaluate=FL_CONFIG['fraction_evaluate'],
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=FL_CONFIG['min_available_clients'],
    )
    strategy.num_rounds = num_rounds  # Add this line to track total rounds

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )