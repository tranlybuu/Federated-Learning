import tensorflow as tf
import numpy as np
import pandas as pd
from ..utils.config import DATA_CONFIG

def load_and_preprocess_mnist():
    data = pd.read_csv('./dataset/digit_char_dataset.csv')
    y_data = data.iloc[:, -1].values
    x_data = data.iloc[:, :-1].values.reshape(-1, 28, 28, 1) / 255.0

    split_index = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:split_index], y_data[:split_index]
    x_test, y_test = x_data[split_index:], y_data[split_index:]
    
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