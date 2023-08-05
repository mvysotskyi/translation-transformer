"""
Encoder layer.
"""

import tensorflow as tf
from tensorflow import keras

from .multihead_attention import MultiHeadAttention
from .feedforward import FeedForward
from .addnorm import AddNorm

# Temporary use Keras MultiHeadAttention
from keras.layers import MultiHeadAttention as KerasMultiHeadAttention


class EncoderLayer(keras.layers.Layer):
    """
    Encoder layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.multihead_attention = KerasMultiHeadAttention(num_heads, d_model)
        self.addnorm_1 = AddNorm()

        self.feedforward = FeedForward(d_model, d_ff)

        # Dropout layers
        self.dropout_1 = keras.layers.Dropout(dropout_rate)
        self.dropout_2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, padding_mask: tf.Tensor | None = None, training: bool = True) -> tf.Tensor:
        attention = self.multihead_attention(inputs, inputs, inputs)
        attention = self.dropout_1(attention, training=training)
        attention = self.addnorm_1(inputs, attention, training=training)

        output = self.feedforward(attention)
        return output