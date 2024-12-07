import flwr as fl
import tensorflow as tf
from .model import create_model
from ..utils.config import (
    FL_CONFIG, MODEL_DIR, DATA_SUMMARY_TEMPLATE,
    MODEL_TEMPLATES, DATA_RANGES_INFO, SECURE_AGG_CONFIG
)
from datetime import datetime
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
        self.active_clients = set()
        self.client_id_map = {}  # Theo dõi clients đang tham gia

        # Store client public keys
        self.client_pubkeys = {}
        
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

    def _load_client_pubkeys(self):
        """Load all available client public keys"""
        key_dir = SECURE_AGG_CONFIG['key_storage']
        for filename in os.listdir(key_dir):
            if filename.startswith('client_') and filename.endswith('_pub.pem'):
                client_id = filename.split('_')[1]
                with open(os.path.join(key_dir, filename), 'rb') as f:
                    self.client_pubkeys[client_id] = f.read()

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate masked model updates từ các clients."""
        self.current_round = server_round
        
        print(f"\nRound {server_round} ({self.mode} mode):")
        print(f"Active clients: {len(results)}")
        print(f"Failures: {len(failures)}")

        # Load client public keys 
        self._load_client_pubkeys()

        # Kiểm tra số lượng clients tối thiểu
        if len(results) < SECURE_AGG_CONFIG['min_clients_for_unmasking']:
            print(f"Insufficient clients for unmasking. Need at least {SECURE_AGG_CONFIG['min_clients_for_unmasking']}")
            return None, {}

        # Cập nhật danh sách clients đang hoạt động
        active_client_ids = set()
        for client_proxy, fit_res in results:
            numeric_cid = fit_res.metrics.get('client_id')
            if numeric_cid:
                self.client_id_map[client_proxy.cid] = str(numeric_cid)
                self.active_clients.add(str(numeric_cid))
                active_client_ids.add(str(numeric_cid))
                
        # Kiểm tra client dropouts
        dropout_rate = 1 - (len(active_client_ids) / len(self.active_clients))
        if dropout_rate > SECURE_AGG_CONFIG['dropout_threshold']:
            print(f"High dropout rate detected: {dropout_rate:.2%}")
            if not SECURE_AGG_CONFIG['enable_dropout_recovery']:
                return None, {}

        # Validate yêu cầu của phase
        phase_reqs = DATA_RANGES_INFO['phase_requirements'][self.mode]
        if len(active_client_ids) < phase_reqs['min_clients']:
            raise ValueError(
                f"Phase {self.mode} requires minimum {phase_reqs['min_clients']} clients, "
                f"but only {len(active_client_ids)} are active"
            )
        if len(active_client_ids) > phase_reqs['max_clients']:
            raise ValueError(
                f"Phase {self.mode} allows maximum {phase_reqs['max_clients']} clients, "
                f"but {len(active_client_ids)} are active"
            )

        if not results:
            return None, {}

        # Thu thập masked updates
        weights = []
        num_examples = []
        metrics = []
        masked_updates = {}

        for client_proxy, fit_res in results:
            client_id = self.client_id_map.get(client_proxy.cid, 'unknown')
            
            # Lưu masked update từ client
            masked_update = fl.common.parameters_to_ndarrays(fit_res.parameters)
            masked_updates[client_id] = masked_update
            
            weights.append(masked_update)
            num_examples.append(fit_res.num_examples)
            metrics.append(fit_res.metrics)
            
            print(f"Client {client_id} metrics: {fit_res.metrics}")

        # Tính tổng số examples
        total_examples = sum(num_examples)
        if total_examples == 0:
            return None, {}

        # Tổng hợp masked updates sử dụng weighted average
        aggregated_weights = [
            np.sum(
                [w[i] * n for w, n in zip(weights, num_examples)], 
                axis=0
            ) / total_examples
            for i in range(len(weights[0]))
        ]

        # Update model toàn cục
        self.model.set_weights(aggregated_weights)

        # Đánh giá model mới
        test_loss, test_accuracy = self._evaluate_global_model()
        print(f"Round {server_round} results - Loss: {test_loss}, Accuracy: {test_accuracy}")

        # Lưu kết quả round
        round_metrics = {
            'round': server_round,
            'mode': self.mode,
            'loss': float(test_loss),
            'accuracy': float(test_accuracy),
            'num_clients': len(results),
            'client_metrics': metrics,
            'active_clients': list(active_client_ids),
            'dropout_rate': dropout_rate
        }
        self.round_results.append(round_metrics)

        # Kiểm tra và lưu best model
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            best_model_path = MODEL_TEMPLATES['best'].format(self.mode)
            self.model.save(best_model_path)
            print(f"Saved best model with accuracy {test_accuracy:.4f}")

        # Lưu model của round hiện tại
        current_model_path = MODEL_TEMPLATES['global'].format(server_round)
        self.model.save(current_model_path)

        # Kiểm tra nếu là round cuối
        if server_round == self.num_rounds:
            self._save_final_results()

        # Key rotation check
        need_key_rotation = (
            SECURE_AGG_CONFIG['enable_key_rotation'] and 
            server_round % SECURE_AGG_CONFIG['rotation_frequency'] == 0
        )

        # Chuẩn bị config cho round tiếp theo
        next_round_config = {
            'round_id': server_round + 1,
            'peer_pubkeys': self.client_pubkeys,
            'need_key_rotation': need_key_rotation,
            'active_clients': list(active_client_ids),
            'dropout_threshold': SECURE_AGG_CONFIG['dropout_threshold'],
            'secure_aggregation': True
        }

        return fl.common.ndarrays_to_parameters(aggregated_weights), next_round_config

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

        data_summaries = []
        total_train_samples = 0
        total_test_samples = 0
        train_distribution = {}
        test_distribution = {}

        for numeric_cid in self.active_clients:
            summary_path = DATA_SUMMARY_TEMPLATE.format(f"client_{numeric_cid}")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    client_summary = json.load(f)
                    
                data_summaries.append(client_summary)
                
                # Cộng dồn số lượng mẫu từ train
                train_info = client_summary['train']
                total_train_samples += train_info['total_samples']
                
                # Cộng dồn phân bố từ samples_per_label
                for label, count in train_info['samples_per_label'].items():
                    if label not in train_distribution:
                        train_distribution[label] = 0
                    train_distribution[label] += count

                # Cộng dồn số lượng mẫu từ test
                test_info = client_summary['test']
                total_test_samples += test_info['total_samples']
                
                # Cộng dồn phân bố từ samples_per_label
                for label, count in test_info['samples_per_label'].items():
                    if label not in test_distribution:
                        test_distribution[label] = 0
                    test_distribution[label] += count

        # Lấy thông tin ranges cho các numeric client IDs
        active_client_ranges = {}
        for numeric_cid in self.active_clients:
            if numeric_cid in DATA_RANGES_INFO['client_ranges']:
                active_client_ranges[numeric_cid] = DATA_RANGES_INFO['client_ranges'][numeric_cid]

        # Tạo tổng hợp kết quả cuối cùng
        final_results = {
            'training_info': {
                'mode': self.mode,
                'total_rounds': self.current_round,
                'best_accuracy': float(self.best_accuracy),
                'final_accuracy': float(self.round_results[-1]['accuracy']),
                'final_loss': float(self.round_results[-1]['loss']),
                'training_history': self.round_results
            },
            'active_clients_info': {
                'count': len(self.active_clients),
                'client_ids': list(self.active_clients),
                'client_ranges': active_client_ranges,
                'phase_requirements': DATA_RANGES_INFO['phase_requirements'][self.mode]
            },
            'dataset_statistics': {
                'overall': {
                    'train': {
                        'total_samples': total_train_samples,
                        'samples_per_label': train_distribution,
                        'labels_distribution': {
                            str(label): f"{(count/total_train_samples*100):.2f}%"
                            for label, count in train_distribution.items()
                        }
                    },
                    'test': {
                        'total_samples': total_test_samples,
                        'samples_per_label': test_distribution,
                        'labels_distribution': {
                            str(label): f"{(count/total_test_samples*100):.2f}%"
                            for label, count in test_distribution.items()
                        }
                    }
                },
                'per_client': data_summaries
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Lưu kết quả vào một file duy nhất
        results_path = os.path.join(results_dir, f'best_{self.mode}_model.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        # In tổng kết
        print(f"\nTraining Summary ({self.mode} mode):")
        print("=" * 50)
        print(f"Total Rounds: {self.current_round}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"Final Accuracy: {self.round_results[-1]['accuracy']:.4f}")
        print(f"Final Loss: {self.round_results[-1]['loss']:.4f}")
        print(f"Active Clients: {sorted(list(self.active_clients))}")
        print("\nDataset Statistics:")
        print("Training Data:")
        print(f"Total Samples: {total_train_samples}")
        print("Labels Distribution:")
        for label, count in sorted(train_distribution.items()):
            percentage = count/total_train_samples*100
            print(f"  Label {label}: {count} samples ({percentage:.2f}%)")
        print("\nTest Data:")
        print(f"Total Samples: {total_test_samples}")
        print("Labels Distribution:")
        for label, count in sorted(test_distribution.items()):
            percentage = count/total_test_samples*100
            print(f"  Label {label}: {count} samples ({percentage:.2f}%)")
        print("=" * 50)
        print(f"Results saved to: {results_path}")
    def get_model_parameters(self):
        """Get current model parameters."""
        return self.model.get_weights()

def start_server(mode, num_rounds=None, min_fit_clients=None, min_evaluate_clients=None):
    """Start Flower server with specified configuration."""
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