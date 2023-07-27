"""
Implementation of Layer Normalization.
"""

import tensorflow as tf
from tensorflow.keras import layers

class LayerNormalization(layers.Layer):
    """
    Layer Normalization.
    """
    def __init__(self, epsilon: tf.float32 = 1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer.
        """
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1:],
            initializer=tf.ones_initializer(),
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[-1:],
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.
        """
        mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)
        norm_inputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return norm_inputs * self.gamma + self.beta
