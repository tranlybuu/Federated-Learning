from typing import Dict, Any
import numpy as np
from datetime import datetime

class PrivacyMetrics:
    def __init__(self):
        self.metrics_history = []
        
    def log_privacy_metrics(self, round_num: int, metrics: Dict[str, Any]):
        """Log privacy metrics for a training round."""
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['round'] = round_num
        self.metrics_history.append(metrics)
        
    def get_current_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status including latest metrics."""
        if not self.metrics_history:
            return {
                'status': 'No privacy metrics available',
                'timestamp': datetime.now().isoformat()
            }
            
        latest = self.metrics_history[-1]
        return {
            'current_epsilon': latest.get('epsilon', 0),
            'current_delta': latest.get('delta', 0),
            'num_participating_clients': latest.get('num_clients', 0),
            'secure_aggregation_status': latest.get('secure_agg_success', False),
            'last_update': latest['timestamp']
        }
        
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy metrics across all rounds."""
        if not self.metrics_history:
            return {'status': 'No privacy metrics available'}
            
        epsilons = [m.get('epsilon', 0) for m in self.metrics_history]
        return {
            'total_rounds': len(self.metrics_history),
            'total_epsilon_spent': sum(epsilons),
            'average_epsilon_per_round': np.mean(epsilons),
            'max_epsilon_in_single_round': max(epsilons),
            'total_participating_clients': sum(m.get('num_clients', 0) for m in self.metrics_history),
            'secure_agg_success_rate': sum(m.get('secure_agg_success', False) for m in self.metrics_history) / len(self.metrics_history)
        }