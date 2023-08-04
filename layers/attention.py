"""
Scaled Dot-Product Attention.
"""

import tensorflow as tf
from tensorflow import keras


class ScaledDotProductAttention(keras.layers.Layer):
    """
    Scaled Dot-Product Attention.
    """
    def __init__(self, key_dim: int = 512, value_dim: int = 512, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.softmax = keras.layers.Softmax()

        self.key_dim = key_dim
        self.value_dim = value_dim

        self.w_q = keras.layers.Dense(self.key_dim)
        self.w_k = keras.layers.Dense(self.key_dim)
        self.w_v = keras.layers.Dense(self.value_dim)

    def call(self,
             queries: tf.Tensor, keys: tf.Tensor, values: tf.Tensor,
             mask: tf.Tensor | None = None):
        # Project the queries, keys and values
        queries = self.w_q(queries)
        keys = self.w_k(keys)
        values = self.w_v(values)

        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        if mask is not None:
            scores = scores + (mask * -1e9)

        attention = self.softmax(scores)
        output = tf.matmul(attention, values)

        return output
