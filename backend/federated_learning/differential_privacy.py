import tensorflow as tf
import numpy as np

class DPModelWrapper:
    def __init__(
        self,
        enabled=True,
        l2_norm_clip=1.0,
        noise_multiplier=1.1,
        num_microbatches=None,
        target_delta=1e-5,
        target_epsilon=10.0
    ):
        self.enabled = enabled
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.target_delta = target_delta
        self.target_epsilon = target_epsilon

    def add_noise_to_gradients(self, gradients):
        """Thêm nhiễu vào gradients để đảm bảo differential privacy."""
        noisy_gradients = []
        for gradient in gradients:
            if gradient is not None:
                # Clip gradient theo L2 norm
                grad_norm = tf.norm(gradient)
                factor = tf.minimum(1.0, self.l2_norm_clip / (grad_norm + 1e-10))
                clipped_gradient = gradient * factor
                
                # Thêm Gaussian noise
                noise_scale = self.noise_multiplier * self.l2_norm_clip
                noise = tf.random.normal(
                    clipped_gradient.shape,
                    mean=0.0,
                    stddev=noise_scale
                )
                noisy_gradient = clipped_gradient + noise
                noisy_gradients.append(noisy_gradient)
            else:
                noisy_gradients.append(None)
        return noisy_gradients

    def make_model_private(self, model):
        """Wrap model với DP training."""
        class DPModel(tf.keras.Model):
            def __init__(self, base_model, dp_wrapper):
                super().__init__()
                self.base_model = base_model
                self.dp_wrapper = dp_wrapper

            def train_step(self, data):
                x, y = data
                with tf.GradientTape() as tape:
                    y_pred = self.base_model(x, training=True)
                    loss = self.compiled_loss(y, y_pred)

                # Tính toán và add noise vào gradients
                gradients = tape.gradient(loss, self.base_model.trainable_variables)
                noisy_gradients = self.dp_wrapper.add_noise_to_gradients(gradients)
                
                # Apply gradients
                self.optimizer.apply_gradients(
                    zip(noisy_gradients, self.base_model.trainable_variables)
                )
                
                # Update metrics
                self.compiled_metrics.update_state(y, y_pred)
                return {m.name: m.result() for m in self.metrics}

            def call(self, inputs):
                return self.base_model(inputs)

        return DPModel(model, self)

    def compute_epsilon(self, n_samples, batch_size, epochs, delta=1e-5):
        """Ước tính privacy loss."""
        # Gaussian mechanism với advanced composition
        steps = epochs * n_samples // batch_size
        q = batch_size / n_samples  # Sampling rate
        
        # Privacy amplification via sampling
        mu = steps * q * q  # Moment
        
        # Calculate epsilon using Gaussian mechanism bounds
        eps = np.sqrt(2 * mu * np.log(1/delta)) * self.noise_multiplier
        
        return float(eps)

class DPClientUpdate:
    def __init__(self, dp_wrapper):
        self.dp_wrapper = dp_wrapper
        
    def train_with_privacy(self, model, data, epochs, batch_size):
        """Training với DP protection."""
        # Wrap model với DP
        private_model = self.dp_wrapper.make_model_private(model)
        
        # Compile với cùng optimizer và loss
        private_model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=['accuracy']
        )
        
        # Training
        history = private_model.fit(
            data[0], data[1],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        # Tính privacy spent
        eps = self.dp_wrapper.compute_epsilon(
            n_samples=len(data[0]),
            batch_size=batch_size,
            epochs=epochs
        )
        
        return private_model.base_model, history, eps