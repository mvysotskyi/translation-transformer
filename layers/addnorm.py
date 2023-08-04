"""
Add and Normalization layer.
"""

import tensorflow as tf
from tensorflow import keras


class AddNorm(keras.layers.Layer):
    """
    Add and Normalization layer.
    """
    def __init__(self, **kwargs):
        super(AddNorm, self).__init__(**kwargs)

        self.layer_norm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, inputs: tf.Tensor, x_sublayer: tf.Tensor) -> tf.Tensor:
        x = self.add([inputs, x_sublayer])
        x = self.layer_norm(x)

        return x
