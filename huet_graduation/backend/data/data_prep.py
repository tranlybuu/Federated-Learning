import tensorflow as tf
import numpy as np
from utils.config import DATA_CONFIG, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Reshape to (samples, height, width, channels)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    
    return (x_train, y_train), (x_test, y_test)

def prepare_data():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    
    # Split training data for clients
    num_clients = DATA_CONFIG['num_clients']
    client_data = []
    shard_size = len(x_train) // num_clients
    
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size
        client_data.append({
            "x_train": x_train[start:end],
            "y_train": y_train[start:end],
            "x_test": x_test,
            "y_test": y_test
        })
    
    return client_data