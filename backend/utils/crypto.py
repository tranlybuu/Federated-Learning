import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization
import numpy as np

class CryptoUtils:
    @staticmethod
    def generate_keypair():
        """Generate RSA key pair for client authentication"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def serialize_public_key(public_key):
        """Serialize public key to bytes"""
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    @staticmethod
    def deserialize_public_key(key_bytes):
        """Deserialize public key from bytes"""
        return serialization.load_pem_public_key(key_bytes)

    @staticmethod
    def generate_shared_key(private_key, peer_public_key):
        """Generate shared key using private key and peer's public key"""
        shared_secret = os.urandom(32)  # In practice, use proper key agreement
        return shared_secret

    @staticmethod
    def generate_mask(shared_key, round_id, shape):
        """Generate deterministic mask for model updates"""
        # Use shared key and round ID to seed PRNG
        seed = int.from_bytes(
            HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=str(round_id).encode()
            ).derive(shared_key),
            byteorder='big'
        )
        
        # Generate random mask with same shape as model update
        rng = np.random.RandomState(seed)
        return rng.normal(0, 0.1, size=shape).astype(np.float32)

    @staticmethod
    def apply_mask(weights, mask):
        """Apply mask to model weights"""
        return [w + m for w, m in zip(weights, mask)]

    @staticmethod
    def remove_mask(weights, mask):
        """Remove mask from model weights"""
        return [w - m for w, m in zip(weights, mask)]