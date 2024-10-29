import flwr as fl
import tensorflow as tf
from backend.federated_learning.model import create_model
from backend.utils.config import DATA_CONFIG

class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=DATA_CONFIG['batch_size'], validation_split=0.1)
        print(f"Fit history: {history.history}")  # Debug print
        return self.model.get_weights(), len(self.x_train), {"accuracy": history.history['accuracy'][-1]}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., tf.newaxis]/255.0, x_test[..., tf.newaxis]/255.0
    
    # Simulate data distribution among clients
    num_clients = DATA_CONFIG['num_clients']
    client_id = 0  # You can change this or make it a parameter
    shard_size = len(x_train) // num_clients
    start = client_id * shard_size
    end = start + shard_size
    
    x_train, y_train = x_train[start:end], y_train[start:end]
    
    # Create and compile model
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Create client
    client = MnistClient(model, x_train, y_train, x_test, y_test)

    # Start Flower client
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())

if __name__ == "__main__":
    main()