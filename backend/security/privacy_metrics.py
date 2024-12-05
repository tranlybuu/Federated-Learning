from datetime import datetime
import json
import os
import numpy as np
import math

class PrivacyMetricsTracker:
    def __init__(self, log_dir, target_delta=1e-5, target_epsilon=10.0):
        """
        Khởi tạo Privacy Metrics Tracker.
        
        Args:
            log_dir: Thư mục lưu privacy logs
            target_delta: Mục tiêu delta cho DP guarantee
            target_epsilon: Privacy budget tối đa cho phép
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon
        self.privacy_ledger = {}
        self._load_ledger()
        
    def _load_ledger(self):
        """Load privacy ledger từ file."""
        ledger_path = os.path.join(self.log_dir, 'privacy_ledger.json')
        if os.path.exists(ledger_path):
            with open(ledger_path, 'r') as f:
                self.privacy_ledger = json.load(f)
                
    def _save_ledger(self):
        """Lưu privacy ledger."""
        ledger_path = os.path.join(self.log_dir, 'privacy_ledger.json')
        with open(ledger_path, 'w') as f:
            json.dump(self.privacy_ledger, f, indent=4)

    def compute_privacy_spent(self, client_id, training_params):
        """
        Tính toán privacy đã sử dụng cho một phiên training.
        
        Args:
            client_id: ID của client
            training_params: Dict chứa thông số training:
                - n_samples: Số lượng samples
                - batch_size: Kích thước batch
                - noise_multiplier: Mức độ noise
                - epochs: Số epochs
        """
        n_samples = training_params['n_samples']
        batch_size = training_params['batch_size']
        noise_multiplier = training_params['noise_multiplier']
        epochs = training_params['epochs']

        # Tính số steps
        steps = epochs * n_samples // batch_size
        
        # Tính sampling rate
        q = batch_size / n_samples
        
        # Tính moment using RDP accountant
        alpha = 2 * math.log(1/self.target_delta)
        rdp = steps * q * q * (1/(2 * noise_multiplier * noise_multiplier))
        
        # Convert RDP to (ε,δ)-DP
        epsilon = rdp + (math.log(1/self.target_delta) / alpha)
        
        # Log kết quả
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if client_id not in self.privacy_ledger:
            self.privacy_ledger[client_id] = []
            
        self.privacy_ledger[client_id].append({
            'timestamp': timestamp,
            'epsilon': float(epsilon),
            'training_params': training_params,
            'steps': steps,
            'sampling_rate': float(q),
            'rdp_alpha': float(alpha),
            'rdp_epsilon': float(rdp)
        })
        
        self._save_ledger()
        return epsilon

    def check_privacy_budget(self, client_id):
        """Kiểm tra xem client có vượt quá privacy budget không."""
        if client_id not in self.privacy_ledger:
            return True, self.target_epsilon
            
        total_epsilon = sum(entry['epsilon'] for entry in self.privacy_ledger[client_id])
        remaining_budget = max(0, self.target_epsilon - total_epsilon)
        
        return total_epsilon <= self.target_epsilon, remaining_budget

    def get_client_privacy_report(self, client_id):
        """
        Tạo báo cáo privacy chi tiết cho một client.
        
        Returns:
            Dict chứa thông tin privacy của client
        """
        if client_id not in self.privacy_ledger:
            return {
                'client_id': client_id,
                'total_epsilon': 0,
                'remaining_budget': self.target_epsilon,
                'training_sessions': 0,
                'history': [],
                'status': 'inactive'
            }
            
        history = self.privacy_ledger[client_id]
        total_epsilon = sum(entry['epsilon'] for entry in history)
        remaining_budget = max(0, self.target_epsilon - total_epsilon)
        
        return {
            'client_id': client_id,
            'total_epsilon': total_epsilon,
            'remaining_budget': remaining_budget,
            'training_sessions': len(history),
            'history': history,
            'status': 'active' if remaining_budget > 0 else 'budget_exceeded'
        }
        
    def get_global_privacy_report(self):
        """Tạo báo cáo privacy tổng thể cho toàn hệ thống."""
        clients = list(self.privacy_ledger.keys())
        client_reports = [self.get_client_privacy_report(cid) for cid in clients]
        
        active_clients = [c for c in client_reports if c['status'] == 'active']
        exceeded_clients = [c for c in client_reports if c['status'] == 'budget_exceeded']
        
        average_epsilon = np.mean([report['total_epsilon'] for report in client_reports]) if client_reports else 0
        max_epsilon = max([report['total_epsilon'] for report in client_reports]) if client_reports else 0
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta,
            'system_stats': {
                'total_clients': len(clients),
                'active_clients': len(active_clients),
                'exceeded_clients': len(exceeded_clients),
                'average_epsilon': float(average_epsilon),
                'max_epsilon': float(max_epsilon),
                'global_privacy_health': 'good' if max_epsilon <= self.target_epsilon else 'warning'
            },
            'client_reports': client_reports
        }

    def reset_client_privacy(self, client_id):
        """Reset privacy ledger cho một client cụ thể."""
        if client_id in self.privacy_ledger:
            # Backup trước khi reset
            backup_dir = os.path.join(self.log_dir, 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_path = os.path.join(
                backup_dir,
                f'client_{client_id}_privacy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(backup_path, 'w') as f:
                json.dump(self.privacy_ledger[client_id], f, indent=4)
            
            # Reset ledger
            del self.privacy_ledger[client_id]
            self._save_ledger()
            
            return {
                'status': 'success',
                'message': f'Privacy ledger reset for client {client_id}',
                'backup_path': backup_path
            }
        
        return {
            'status': 'warning',
            'message': f'No privacy ledger found for client {client_id}'
        }