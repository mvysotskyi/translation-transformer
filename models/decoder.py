"""
Decoder model.
"""

import tensorflow as tf
from tensorflow import keras

from layers.decoder_layer import DecoderLayer
from layers.positional_embedding import PositionEmbeddingFixedWeights


class Decoder(keras.layers.Layer):
    """
    Decoder model.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(Decoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len

        self.pos_embedding = PositionEmbeddingFixedWeights(seq_len, vocab_size, d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.encoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Dropout layer
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(
        self,
        inputs: tf.Tensor,
        encoder_outputs: tf.Tensor,
        look_ahead_mask: tf.Tensor | None = None,
        padding_mask: tf.Tensor | None = None,
        training: bool = True,
    ) -> tf.Tensor:
        output = self.pos_embedding(inputs)
        output = self.dropout(output, training=training)

        for decoder_layer in self.encoder_layers:
            output = decoder_layer(
                output,
                encoder_outputs,
                look_ahead_mask,
                padding_mask,
                training=training,
            )

        return output


if __name__ == "__main__":
    decoder = Decoder(20, 5, 512, 8, 6, 2048)
    input_s = tf.random.normal((2, 5))
    encoder_output = tf.random.normal((2, 5, 512))

    output = decoder(input_s, encoder_output)
    print(output.shape)
