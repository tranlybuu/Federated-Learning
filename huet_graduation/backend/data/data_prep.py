import tensorflow as tf
import numpy as np
from utils.config import DATA_CONFIG, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def load_and_preprocess_mnist():
    """Load và tiền xử lý dữ liệu MNIST."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Reshape to (samples, height, width, channels)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    
    return (x_train, y_train), (x_test, y_test)

def prepare_data():
    """Phân chia dữ liệu cho các clients theo classes."""
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    
    # Khởi tạo dữ liệu cho 2 clients
    client_data = [
        {
            "x_train": [],
            "y_train": [],
            "x_test": x_test,  # Giữ nguyên tập test cho tất cả clients
            "y_test": y_test
        },
        {
            "x_train": [],
            "y_train": [],
            "x_test": x_test,
            "y_test": y_test
        }
    ]
    
    # Lọc dữ liệu cho từng client
    for i in range(len(x_train)):
        if y_train[i] <= 4:  # Classes 0-4 cho client 1
            client_data[0]["x_train"].append(x_train[i])
            client_data[0]["y_train"].append(y_train[i])
        else:  # Classes 5-9 cho client 2
            client_data[1]["x_train"].append(x_train[i])
            client_data[1]["y_train"].append(y_train[i])
    
    # Chuyển lists thành numpy arrays
    for client in client_data:
        client["x_train"] = np.array(client["x_train"])
        client["y_train"] = np.array(client["y_train"])
    
    # In thống kê về phân phối dữ liệu
    print("\nPhân phối dữ liệu training:")
    for i, client in enumerate(client_data):
        classes, counts = np.unique(client["y_train"], return_counts=True)
        print(f"\nClient {i+1}:")
        print(f"Số lượng samples: {len(client['y_train'])}")
        print("Phân phối classes:")
        for c, count in zip(classes, counts):
            print(f"Class {c}: {count} samples")
    
    return client_data