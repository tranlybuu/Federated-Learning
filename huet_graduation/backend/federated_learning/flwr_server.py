import flwr as fl
import tensorflow as tf
from .model import create_model
from ..utils.config import FL_CONFIG, MODEL_PATH
import matplotlib.pyplot as plt
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = create_model()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.round_results = []

    def aggregate_fit(self, server_round, results, failures):
        print(f"Round {server_round}: Aggregating fit results")  # Debug print
        print(f"Results: {results}")  # Debug print
        print(f"Failures: {failures}")  # Debug print
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            # Check if the number of weights matches
            if len(aggregated_weights) == len(self.model.get_weights()):
                self.model.set_weights(aggregated_weights)
                
                # Evaluate the model
                _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
                loss, accuracy = self.model.evaluate(x_test, y_test)
                print(f"Round {server_round} : loss = {loss}, accuracy = {accuracy}")
                
                self.round_results.append((server_round, loss, accuracy))
                
                # Save the model after the final round
                if server_round == self.num_rounds:
                    print(f"Saving model after final round {server_round}")
                    self.model.save(MODEL_PATH)
                    self.plot_results()
            else:
                print(f"Warning: Aggregated weights do not match model structure. Expected {len(self.model.get_weights())}, got {len(aggregated_weights)}")
        else:
            print(f"Round {server_round}: No aggregated weights")
        return aggregated_weights

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from clients."""
        if not results:
            return None, {}
        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(server_round, results, failures)

    def plot_results(self):
        rounds, losses, accuracies = zip(*self.round_results)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(rounds, losses, 'b-')
        plt.title('Loss vs. Rounds')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(rounds, accuracies, 'r-')
        plt.title('Accuracy vs. Rounds')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('federated_learning_results.png')
        print("Results visualization saved as 'federated_learning_results.png'")

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "local_epochs": 1,
    }
    return config

def start_server(num_rounds, min_fit_clients, min_evaluate_clients):
    strategy = SaveModelStrategy(
        fraction_fit=FL_CONFIG['fraction_fit'],
        fraction_evaluate=FL_CONFIG['fraction_evaluate'],
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=FL_CONFIG['min_available_clients'],
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

def weighted_average(metrics):
    print(f"Received metrics: {metrics}")  # Debug print
    
    if not metrics:
        return {}

    try:
        # Assuming metrics is a list of tuples (num_examples, dict_of_metrics)
        accuracies = []
        examples = []
        for num_examples, metrics_dict in metrics:
            if 'accuracy' in metrics_dict:
                accuracies.append(num_examples * metrics_dict['accuracy'])
                examples.append(num_examples)
            else:
                print(f"Warning: 'accuracy' not found in metrics: {metrics_dict}")

        if accuracies and examples:
            return {"accuracy": sum(accuracies) / sum(examples)}
        else:
            print("Warning: No valid accuracy metrics found")
            return {}
    except Exception as e:
        print(f"Error in weighted_average: {e}")
        print(f"Metrics structure: {metrics}")
        return {}