from cryptography.exceptions import InvalidSignature
import json
import os
import time
from datetime import datetime
import hashlib

class ClientVerifier:
    def __init__(self, verification_dir):
        self.verification_dir = verification_dir
        os.makedirs(verification_dir, exist_ok=True)
        self.verified_clients = {}
        self._load_verification_status()
        
    def _load_verification_status(self):
        """Load trạng thái verification."""
        status_path = os.path.join(self.verification_dir, 'verification_status.json')
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                self.verified_clients = json.load(f)
                
    def _save_verification_status(self):
        """Lưu trạng thái verification."""
        status_path = os.path.join(self.verification_dir, 'verification_status.json')
        with open(status_path, 'w') as f:
            json.dump(self.verified_clients, f, indent=4)
            
    def verify_client(self, client_id, signature, crypto_manager):
        """Xác thực client sử dụng chữ ký số."""
        try:
            # Load client's public key
            _, public_key = crypto_manager.load_client_keys(client_id)
            if not public_key:
                return False
                
            # Create verification data
            timestamp = int(time.time())
            verification_data = {
                'client_id': client_id,
                'timestamp': timestamp
            }
            
            # Verify signature
            if not crypto_manager.verify_signature(verification_data, signature, public_key):
                return False
                
            # Update verification status
            self.verified_clients[client_id] = {
                'last_verified': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'verified'
            }
            self._save_verification_status()
            
            return True
        except Exception as e:
            print(f"Verification failed for client {client_id}: {e}")
            return False
            
    def is_client_verified(self, client_id):
        """Kiểm tra xem client đã được xác thực chưa."""
        return (
            client_id in self.verified_clients and
            self.verified_clients[client_id]['status'] == 'verified'
        )
        
    def generate_verification_challenge(self, client_id):
        """Tạo challenge cho client verification."""
        challenge = os.urandom(32)
        challenge_hash = hashlib.sha256(challenge).hexdigest()
        
        challenge_dir = os.path.join(self.verification_dir, 'challenges')
        os.makedirs(challenge_dir, exist_ok=True)
        
        with open(os.path.join(challenge_dir, f'{client_id}_challenge.json'), 'w') as f:
            json.dump({
                'challenge': challenge_hash,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f)
            
        return challenge
        
    def verify_challenge_response(self, client_id, response, crypto_manager):
        """Xác thực response của client với challenge."""
        challenge_path = os.path.join(
            self.verification_dir, 
            'challenges', 
            f'{client_id}_challenge.json'
        )
        
        if not os.path.exists(challenge_path):
            return False
            
        with open(challenge_path, 'r') as f:
            challenge_data = json.load(f)
            
        # Verify response
        try:
            _, public_key = crypto_manager.load_client_keys(client_id)
            if not public_key:
                return False
                
            if not crypto_manager.verify_signature(
                challenge_data['challenge'], 
                response, 
                public_key
            ):
                return False
                
            # Update verification status
            self.verified_clients[client_id] = {
                'last_verified': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'verified',
                'verification_method': 'challenge-response'
            }
            self._save_verification_status()
            
            # Clean up challenge
            os.remove(challenge_path)
            
            return True
        except Exception as e:
            print(f"Challenge verification failed for client {client_id}: {e}")
            return False

    def revoke_client_verification(self, client_id):
        """Thu hồi xác thực của client."""
        if client_id in self.verified_clients:
            self.verified_clients[client_id].update({
                'status': 'revoked',
                'revocation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            self._save_verification_status()
            
    def get_verification_report(self):
            """Tạo báo cáo về trạng thái verification của tất cả clients."""
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_clients': len(self.verified_clients),
                'verified_clients': len([c for c in self.verified_clients.values() 
                                    if c['status'] == 'verified']),
                'revoked_clients': len([c for c in self.verified_clients.values() 
                                    if c['status'] == 'revoked']),
                'client_details': self.verified_clients
            }

    def clean_expired_verifications(self, expiry_hours=24):
        """Xóa các verification đã hết hạn."""
        current_time = datetime.now()
        expired_clients = []
        
        for client_id, info in self.verified_clients.items():
            last_verified = datetime.strptime(
                info['last_verified'], 
                '%Y-%m-%d %H:%M:%S'
            )
            hours_diff = (current_time - last_verified).total_seconds() / 3600
            
            if hours_diff > expiry_hours:
                expired_clients.append(client_id)
                
        for client_id in expired_clients:
            self.verified_clients[client_id].update({
                'status': 'expired',
                'expiration_time': current_time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        self._save_verification_status()
        
        return {
            'expired_count': len(expired_clients),
            'expired_clients': expired_clients
        }

    def verify_client_batch(self, client_data_list, crypto_manager):
        """Xác thực nhiều clients cùng lúc."""
        results = {}
        for client_data in client_data_list:
            client_id = client_data['client_id']
            signature = client_data['signature']
            
            success = self.verify_client(client_id, signature, crypto_manager)
            results[client_id] = {
                'success': success,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'verified' if success else 'failed'
            }
            
        return results

    def get_client_verification_history(self, client_id):
        """Lấy lịch sử verification của một client."""
        history_path = os.path.join(
            self.verification_dir, 
            'history', 
            f'{client_id}_history.json'
        )
        
        if not os.path.exists(history_path):
            return []
            
        with open(history_path, 'r') as f:
            return json.load(f)

    def log_verification_attempt(self, client_id, success, method, details=None):
        """Ghi log cho mỗi lần verification."""
        history_dir = os.path.join(self.verification_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        
        history_path = os.path.join(history_dir, f'{client_id}_history.json')
        
        # Load existing history
        history = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        
        # Add new entry
        history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'success': success,
            'method': method,
            'details': details or {}
        })
        
        # Save updated history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

    def reset_verification_status(self):
        """Reset trạng thái verification của tất cả clients."""
        backup_path = os.path.join(
            self.verification_dir,
            f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        # Backup current status
        with open(backup_path, 'w') as f:
            json.dump(self.verified_clients, f, indent=4)
            
        # Reset status
        self.verified_clients = {}
        self._save_verification_status()
        
        return {
            'backup_path': backup_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'reset_completed'
        }