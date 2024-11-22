import flwr as fl
import tensorflow as tf
from ..utils.config import INITIAL_MODEL_PATH, CLIENT_MODEL_TEMPLATE, DATA_CONFIG

class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid, x_train, y_train, x_test, y_test):
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Load model từ server
        self.model = tf.keras.models.load_model(INITIAL_MODEL_PATH)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set trọng số nhận được từ server
        self.model.set_weights(parameters)
        
        # Train local
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=DATA_CONFIG['local_epochs'],
            batch_size=DATA_CONFIG['batch_size'],
            validation_split=DATA_CONFIG['validation_split'],
            verbose=1
        )
        
        # Lưu model sau khi train
        save_path = CLIENT_MODEL_TEMPLATE.format(self.cid)
        self.model.save(save_path, save_format='keras')
        
        return self.model.get_weights(), len(self.x_train), {
            "accuracy": history.history['accuracy'][-1],
            "loss": history.history['loss'][-1]
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

def load_data(cid):
    """Load và tiền xử lý dữ liệu MNIST cho client."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize và reshape
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    # Chia dữ liệu cho client cụ thể
    total_clients = DATA_CONFIG['num_clients']
    samples_per_client = len(x_train) // total_clients
    
    start_idx = cid * samples_per_client
    end_idx = start_idx + samples_per_client
    
    return (
        x_train[start_idx:end_idx], 
        y_train[start_idx:end_idx],
        x_test,
        y_test
    )

def start_client(cid):
    """Khởi động Flower client."""
    # Load data cho client
    x_train, y_train, x_test, y_test = load_data(cid)
    
    # Tạo client
    client = MnistClient(cid, x_train, y_train, x_test, y_test)
    
    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument("--cid", type=int, required=True)
    args = parser.parse_args()
    
    start_client(args.cid)