import tensorflow as tf
import numpy as np
from typing import Tuple

class DPFederatedLearning:
    def __init__(self, l2_norm_clip: float, noise_multiplier: float, num_microbatches: int):
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.privacy_spent = 0.0

    def add_noise_to_gradients(self, gradients):
        """Add Gaussian noise to clipped gradients."""
        noised_gradients = []
        for grad in gradients:
            if grad is not None:
                # Clip gradients
                grad_norm = tf.norm(grad)
                grad = grad * tf.minimum(1.0, self.l2_norm_clip / (grad_norm + 1e-10))
                
                # Add noise
                noise = tf.random.normal(
                    grad.shape,
                    mean=0.0,
                    stddev=self.noise_multiplier * self.l2_norm_clip,
                    dtype=grad.dtype
                )
                noised_grad = grad + noise
                noised_gradients.append(noised_grad)
            else:
                noised_gradients.append(None)
        return noised_gradients

    def create_dp_model(self, base_model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
        """Create model with DP training capability."""
        class DPModel(tf.keras.Model):
            def __init__(self, base_model, dp_noise_fn, **kwargs):
                super().__init__(**kwargs)
                self.base_model = base_model
                self.dp_noise_fn = dp_noise_fn
                # Copy layers from base model
                self.model_layers = [layer for layer in base_model.layers]

            def call(self, inputs):
                x = inputs
                for layer in self.model_layers:
                    x = layer(x)
                return x

            def train_step(self, data):
                x, y = data
                
                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

                # Get gradients
                gradients = tape.gradient(loss, self.trainable_variables)
                
                # Add noise to gradients
                noised_gradients = self.dp_noise_fn(gradients)
                
                # Apply gradients
                self.optimizer.apply_gradients(zip(noised_gradients, self.trainable_variables))
                
                # Update metrics
                self.compiled_metrics.update_state(y, y_pred)
                
                return {m.name: m.result() for m in self.metrics}

        # Create custom model
        dp_model = DPModel(base_model, self.add_noise_to_gradients)
        
        # Copy weights
        dp_model.set_weights(base_model.get_weights())
        
        # Compile model
        dp_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return dp_model

    def compute_privacy_loss(self, num_examples: int, batch_size: int, epochs: int, 
                           target_delta: float = 1e-5) -> Tuple[float, float]:
        """Compute approximate privacy loss using zCDP composition."""
        steps = epochs * num_examples // batch_size
        
        # Simplified privacy computation using zCDP composition
        q = batch_size / float(num_examples)
        T = steps
        sigma = self.noise_multiplier
        
        # Compute epsilon using zCDP composition
        rho = T * q * q / (2 * sigma * sigma)
        epsilon = rho + 2 * np.sqrt(rho * np.log(1/target_delta))
        
        self.privacy_spent += epsilon
        return epsilon, target_delta

    def get_privacy_spent(self) -> float:
        """Get total privacy spent so far."""
        return self.privacy_spent