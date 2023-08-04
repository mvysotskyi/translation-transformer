"""
Encoder model.
"""

import sys
sys.path.append("../")

import tensorflow as tf
from tensorflow import keras

from layers.encoder_layer import EncoderLayer
from layers.positional_embedding import PositionEmbeddingFixedWeights


class Encoder(keras.layers.Layer):
    def __init__(self, vocab_size: int, seq_len: int, 
                num_layers: int, d_model: int, num_heads: int, d_ff: int,
                dropout_rate: float = 0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
        self.pos_embedding = PositionEmbeddingFixedWeights(seq_len, vocab_size, d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def call(self, inputs: tf.Tensor, padding_mask: tf.Tensor | None = None, training: bool = True) -> tf.Tensor:
        output = self.pos_embedding(inputs)
        output = self.dropout(output, training=training)

        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output, padding_mask, training=training)

        return output

if __name__ == "__main__":
    encoder = Encoder(20, 5, 6, 512, 8, 2048)
    input_s = tf.random.normal((2, 5))

    output = encoder(input_s)
    print(output.shape)
