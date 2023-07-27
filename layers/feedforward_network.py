"""
Implementation of the feedforward network layer.
"""

import tensorflow as tf
from tensorflow import keras


class FeedForwardNetwork(keras.layers.Layer):
    """
    Feedforward network layer.
    """
    def __init__(self, d_model: int, dff: int) -> None:
        """
        dff: dimension of feedforward network.
        """
        super(FeedForwardNetwork, self).__init__()
        self.d_model = d_model
        self.dff = dff

        self.dense1 = keras.layers.Dense(dff, activation="relu")
        self.dense2 = keras.layers.Dense(d_model)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.
        """
        outputs = self.dense1(inputs)
        outputs = self.dense2(outputs)
        return outputs
