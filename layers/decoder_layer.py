"""
Decoder Layer.
"""

import tensorflow as tf
from tensorflow import keras

from .multihead_attention import MultiHeadAttention
from .feedforward import FeedForward
from .addnorm import AddNorm

# Temporary use Keras MultiHeadAttention
from keras.layers import MultiHeadAttention as KerasMultiHeadAttention


class DecoderLayer(keras.layers.Layer):
    """
    Decoder Layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(DecoderLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.multihead_attention_1 = KerasMultiHeadAttention(num_heads, d_model)
        self.addnorm_1 = AddNorm()

        self.multihead_attention_2 = KerasMultiHeadAttention(num_heads, d_model)
        self.addnorm_2 = AddNorm()

        self.feedforward = FeedForward(d_model, d_ff)

        # Dropout layers
        self.dropout_1 = keras.layers.Dropout(dropout_rate)
        self.dropout_2 = keras.layers.Dropout(dropout_rate)
        self.dropout_3 = keras.layers.Dropout(dropout_rate)

    def call(
        self,
        inputs: tf.Tensor,
        encoder_outputs: tf.Tensor,
        look_ahead_mask: tf.Tensor | None = None,
        padding_mask: tf.Tensor | None = None,
        training: bool = True,
    ) -> tf.Tensor:
        attention = self.multihead_attention_1(
            inputs, inputs, inputs, attention_mask=look_ahead_mask
        )
        attention = self.dropout_1(attention, training=training)
        attention = self.addnorm_1(inputs, attention)

        attention_2 = self.multihead_attention_2(
            attention, encoder_outputs, encoder_outputs, attention_mask=padding_mask
        )
        attention_2 = self.dropout_2(attention_2)
        attention_2 = self.addnorm_2(attention, attention_2, training=training)

        output = self.feedforward(attention_2)
        return output


if __name__ == "__main__":
    decoder = DecoderLayer(512, 8, 2048)
    input_s = tf.random.normal((2, 5))
    encoder_outputs = tf.random.normal((2, 5, 512))

    output = decoder(input_s, encoder_outputs)
    print(output.shape)
