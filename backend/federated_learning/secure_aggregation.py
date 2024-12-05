from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import numpy as np

class SecureAggregator:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        # Tạo key pairs cho mỗi client
        self.client_keys = {
            i: rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            ) for i in range(num_clients)
        }
        # Mask secrets cho mỗi cặp clients
        self.pairwise_masks = self._generate_pairwise_masks()
        
    def _generate_pairwise_masks(self):
        """Tạo masks cho mỗi cặp clients."""
        masks = {}
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # Tạo shared secret giữa mỗi cặp clients
                mask = np.random.randn(1).astype(np.float32)
                masks[(i,j)] = mask
                masks[(j,i)] = -mask  # Đảm bảo tổng = 0
        return masks
        
    def mask_weights(self, client_id, weights):
        """Thêm mask vào weights của client."""
        masked_weights = weights.copy()
        for other_id in range(self.num_clients):
            if other_id != client_id:
                key = (client_id, other_id)
                if key in self.pairwise_masks:
                    masked_weights += self.pairwise_masks[key]
        return masked_weights
        
    def aggregate_masked_weights(self, masked_updates):
        """Tổng hợp masked weights từ tất cả clients."""
        # Masks sẽ triệt tiêu nhau khi tổng hợp
        n_clients = len(masked_updates)
        aggregated = sum(masked_updates) / n_clients
        return aggregated

    def verify_client(self, client_id, signature):
        """Xác thực client sử dụng chữ ký số."""
        try:
            public_key = self.client_keys[client_id].public_key()
            public_key.verify(
                signature,
                b"client_verification",
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False