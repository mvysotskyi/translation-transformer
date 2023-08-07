"""
Transformer model.
"""

import tensorflow as tf
from tensorflow import keras

from models.encoder import Encoder
from models.decoder import Decoder


class Transformer(keras.models.Model):
    """
    Transformer model.
    """

    def __init__(
        self,
        src_vocab_size: int,
        target_vocab_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        # self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len

        self.encoder = Encoder(
            src_vocab_size, seq_len, num_layers, d_model, num_heads, d_ff, dropout_rate
        )
        self.decoder = Decoder(
            target_vocab_size,
            seq_len,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            dropout_rate,
        )

        self.final_layer = keras.layers.Dense(target_vocab_size)

    def padding_mask(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Create padding mask.
        """
        mask = tf.math.logical_not(tf.math.equal(inputs, 0))
        mask = tf.cast(mask, dtype=tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def look_ahead_mask(self, size: int) -> tf.Tensor:
        """
        Create look ahead mask.
        """
        mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = True
    ) -> tf.Tensor:
        encoder_input, decoder_input = inputs

        encoder_padding_mask = self.padding_mask(encoder_input)
        decoder_padding_mask = self.padding_mask(decoder_input)
        look_ahead_mask = self.look_ahead_mask(self.seq_len)
        look_ahead_mask = tf.minimum(decoder_padding_mask, look_ahead_mask)
        # print(look_ahead_mask.shape)

        encoder_output = self.encoder(
            encoder_input, encoder_padding_mask, training=training
        )
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            look_ahead_mask,
            encoder_padding_mask,
            training=training,
        )

        output = self.final_layer(decoder_output)

        try:
            del output._keras_mask
        except AttributeError:
            pass

        return output


if __name__ == "__main__":
    transformer = Transformer(20, 20, 5, 512, 8, 6, 2048)
    encoder_input = tf.random.normal((2, 5))
    decoder_input = tf.random.normal((2, 5))

    output = transformer(encoder_input, decoder_input)
    print(output.shape)

    print(tf.argmax(tf.nn.softmax(output, axis=-1), axis=-1))
