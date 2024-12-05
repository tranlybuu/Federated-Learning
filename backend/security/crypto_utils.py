from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.fernet import Fernet
import os
import json
import base64

class CryptoManager:
    def __init__(self, key_directory):
        self.key_directory = key_directory
        os.makedirs(key_directory, exist_ok=True)
        
    def generate_client_keypair(self, client_id):
        """Tạo và lưu key pair cho client."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save keys
        with open(os.path.join(self.key_directory, f'client_{client_id}_private.pem'), 'wb') as f:
            f.write(private_pem)
        with open(os.path.join(self.key_directory, f'client_{client_id}_public.pem'), 'wb') as f:
            f.write(public_pem)
            
        return private_key, public_key

    def load_client_keys(self, client_id):
        """Load key pair cho client."""
        try:
            with open(os.path.join(self.key_directory, f'client_{client_id}_private.pem'), 'rb') as f:
                private_pem = f.read()
            with open(os.path.join(self.key_directory, f'client_{client_id}_public.pem'), 'rb') as f:
                public_pem = f.read()
                
            private_key = serialization.load_pem_private_key(
                private_pem,
                password=None
            )
            public_key = serialization.load_pem_public_key(public_pem)
            
            return private_key, public_key
        except Exception as e:
            print(f"Error loading keys for client {client_id}: {e}")
            return None, None

    def encrypt_weights(self, weights, public_key):
        """Mã hóa model weights."""
        # Convert weights to bytes
        weights_bytes = json.dumps(weights).encode()
        
        # Generate session key
        session_key = Fernet.generate_key()
        f = Fernet(session_key)
        
        # Encrypt weights with session key
        encrypted_weights = f.encrypt(weights_bytes)
        
        # Encrypt session key with public key
        encrypted_session_key = public_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return {
            'encrypted_weights': base64.b64encode(encrypted_weights).decode(),
            'encrypted_session_key': base64.b64encode(encrypted_session_key).decode()
        }

    def decrypt_weights(self, encrypted_data, private_key):
        """Giải mã model weights."""
        try:
            # Decode data
            encrypted_weights = base64.b64decode(encrypted_data['encrypted_weights'])
            encrypted_session_key = base64.b64decode(encrypted_data['encrypted_session_key'])
            
            # Decrypt session key
            session_key = private_key.decrypt(
                encrypted_session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt weights
            f = Fernet(session_key)
            decrypted_weights = f.decrypt(encrypted_weights)
            
            return json.loads(decrypted_weights.decode())
        except Exception as e:
            print(f"Error decrypting weights: {e}")
            return None

    def generate_signature(self, data, private_key):
        """Tạo chữ ký số cho data."""
        signature = private_key.sign(
            json.dumps(data).encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def verify_signature(self, data, signature, public_key):
        """Xác thực chữ ký số."""
        try:
            signature_bytes = base64.b64decode(signature)
            public_key.verify(
                signature_bytes,
                json.dumps(data).encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False