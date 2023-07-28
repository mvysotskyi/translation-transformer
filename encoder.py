"""
Encoder for the Transformer model.
"""

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dropout, Embedding, Dense

from layers.attention import MultiHeadAttention
from layers.positional_encoding import PositionalEncoding
from layers.feedforward_network import FeedForwardNetwork
from layers.layer_normalization import LayerNormalization


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dff: int, dropout_rate: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool, mask: tf.Tensor | None = None) -> tf.Tensor:
        """
        Call the layer.
        """
        outputs1 = self.mha(inputs, inputs, inputs)
        outputs1 = self.dropout1(outputs1, training=training)
        outputs1 = self.layernorm1(inputs + outputs1)

        outputs2 = self.ffn(outputs1)
        outputs2 = self.dropout2(outputs2, training=training)
        outputs2 = self.layernorm2(outputs1 + outputs2)

        return outputs2
    

class Encoder(keras.models.Model):
    def __init__(self, d_model: int, n_heads: int, dff: int, n_layers: int, dropout_rate: float = 0.1,
                 vocab_size: int = 20000,
                 max_seq_len: int = 5000
            ) -> None:
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.enc_layers = [EncoderLayer(d_model, n_heads, dff, dropout_rate) for _ in range(n_layers)]

    def build(self, input_shape):
        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_len)

        super().build(input_shape)


    def call(self, inputs: tf.Tensor, training: bool, mask: tf.Tensor | None = None) -> tf.Tensor:
        """
        Call the layer.
        """
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)

        for i in range(self.n_layers):
            outputs = self.enc_layers[i](outputs, training, mask)

        return outputs

if __name__ == "__main__":
    sample_encoder = Encoder(512, 8, 2048, 6)
    sample_encoder.build((1, 400))

    sample_encoder.summary()
