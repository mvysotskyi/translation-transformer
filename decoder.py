"""
Decoder for Transformer model.
"""

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dropout, Embedding, Dense

from layers.attention import MultiHeadAttention
from layers.positional_encoding import PositionalEncoding
from layers.feedforward_network import FeedForwardNetwork
from layers.layer_normalization import LayerNormalization


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dff: int, dropout_rate: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, n_heads, use_causal_mask=True)
        self.mha2 = MultiHeadAttention(d_model, n_heads)

        self.ffn = FeedForwardNetwork(d_model, dff)

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, enc_outputs: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Call the layer.
        """
        outputs1 = self.mha1(inputs, inputs, inputs)
        outputs1 = self.dropout1(outputs1, training=training)
        outputs1 = self.layernorm1(inputs + outputs1)

        outputs2 = self.mha2(outputs1, enc_outputs, enc_outputs)
        outputs2 = self.dropout2(outputs2, training=training)
        outputs2 = self.layernorm2(outputs1 + outputs2)

        outputs3 = self.ffn(outputs2)
        outputs3 = self.dropout3(outputs3, training=training)
        outputs3 = self.layernorm3(outputs2 + outputs3)

        return outputs3
    

class Decoder(keras.models.Model):
    def __init__(self, d_model: int, n_heads: int, dff: int, n_layers: int, dropout_rate: float = 0.1,
                 vocab_size: int = 20000,
                 max_seq_len: int = 5000
            ) -> None:
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.dec_layers = [DecoderLayer(d_model, n_heads, dff, dropout_rate) for _ in range(n_layers)]

    def build(self, _):
        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
    def call(self, inputs: tf.Tensor, enc_outputs: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Call the layer.
        """
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)

        for i in range(self.n_layers):
            outputs = self.dec_layers[i](outputs, enc_outputs, training)

        return outputs
    
if __name__ == "__main__":
    # Test Decoder
    decoder = Decoder(512, 8, 2048, 6)
    decoder_output = decoder(tf.random.uniform((64, 26)), enc_outputs=tf.random.uniform((64, 62, 512)), training=False)

    print(decoder_output.shape)  # (batch_size, target_seq_len, d_model)