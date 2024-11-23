import flwr as fl
import tensorflow as tf
from ..utils.config import INITIAL_MODEL_PATH, CLIENT_MODEL_TEMPLATE, DATA_CONFIG, MODEL_DIR
import os

class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid, x_train, y_train, x_test, y_test):
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        # Với client 3, load best model thay vì initial model
        if self.cid == 2:  # cid = 2 cho client thứ 3
            best_model_path = os.path.join(MODEL_DIR, 'best_model.keras')
            if os.path.exists(best_model_path):
                self.model = tf.keras.models.load_model(best_model_path)
                print(f"Client {cid} loaded best model from {best_model_path}")
            else:
                print(f"Best model not found, using initial model")
                self.model = tf.keras.models.load_model(INITIAL_MODEL_PATH)
        else:
            self.model = tf.keras.models.load_model(INITIAL_MODEL_PATH)

def load_data(cid):
    """Load và tiền xử lý dữ liệu MNIST cho client."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize và reshape
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    if cid == 2:  # Client 3 sẽ nhận toàn bộ dữ liệu
        return (x_train, y_train, x_test, y_test)
    
    # Lọc dữ liệu cho client 1 và 2 theo classes
    mask = y_train <= 4 if cid == 0 else y_train > 4
    return (
        x_train[mask],
        y_train[mask],
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