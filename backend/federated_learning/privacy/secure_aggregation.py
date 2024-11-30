import numpy as np
from cryptography.fernet import Fernet
from typing import List, Tuple, Dict
import json

class SecureAggregation:
    def __init__(self, num_clients: int):
        """Initialize secure aggregation with number of participating clients."""
        self.num_clients = num_clients
        self.keys = [Fernet.generate_key() for _ in range(num_clients)]
        self.fernet_objects = [Fernet(key) for key in self.keys]
        
    def generate_client_keys(self, client_id: int) -> Dict:
        """Generate pairwise keys for a client."""
        client_keys = {}
        for i in range(self.num_clients):
            if i != client_id:
                shared_key = Fernet.generate_key()
                client_keys[i] = shared_key
        return client_keys

    def mask_weights(self, weights: List[np.ndarray], client_id: int, 
                    client_keys: Dict) -> Tuple[List[np.ndarray], List[bytes]]:
        """Mask weights using pairwise random numbers."""
        masked_weights = [w.copy() for w in weights]
        masks = []
        
        # Generate random masks using shared keys
        for other_client in client_keys:
            f = Fernet(client_keys[other_client])
            mask = np.random.randn(*weights[0].shape)
            if client_id < other_client:
                masked_weights[0] += mask
            else:
                masked_weights[0] -= mask
            # Encrypt mask for verification
            masks.append(f.encrypt(json.dumps(mask.tolist()).encode()))
            
        return masked_weights, masks

    def unmask_weights(self, masked_weights: List[np.ndarray], 
                      masks: List[bytes]) -> List[np.ndarray]:
        """Unmask the weights using stored masks."""
        unmasked_weights = [w.copy() for w in masked_weights]
        
        for mask_bytes in masks:
            for f in self.fernet_objects:
                try:
                    mask = np.array(json.loads(f.decrypt(mask_bytes).decode()))
                    unmasked_weights[0] -= mask
                    break
                except:
                    continue
                    
        return unmasked_weights

    def aggregate_masked_weights(self, all_masked_weights: List[List[np.ndarray]], 
                               num_examples: List[int]) -> List[np.ndarray]:
        """Aggregate masked weights from all clients."""
        total_examples = sum(num_examples)
        n_layers = len(all_masked_weights[0])
        
        aggregated = [np.zeros_like(all_masked_weights[0][i]) for i in range(n_layers)]
        
        for i in range(len(all_masked_weights)):
            weight = num_examples[i] / total_examples
            for j in range(n_layers):
                aggregated[j] += all_masked_weights[i][j] * weight
                
        return aggregated