"""
Layers for Transformer.
"""

import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    """
    Positional encoding layer.
    """
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_encoding = self.positional_encoding()

    def get_angles(self, positions: tf.Tensor , indexes: tf.Tensor) -> tf.Tensor:
        """
        Get the angles for the positional encoding.
        """
        angles = 1 / tf.pow(10000, (2 * (indexes // 2)) / tf.cast(self.d_model, tf.float32))
        return positions * angles

    def positional_encoding(self) -> tf.Tensor:
        """
        Compute positional encoding.
        """
        angle_rads = self.get_angles(
            positions=tf.range(self.max_seq_len, dtype=tf.float32)[:, tf.newaxis],
            indexes=tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        )

        # apply sin to even indices in the array and cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

if __name__ == "__main__":
    # test the PositionalEncoding layer
    sample_pos_encoding = PositionalEncoding(512, 2048)
    print(sample_pos_encoding.pos_encoding.shape)

    test_string = tf.random.uniform((1, 60, 512))

    sample_pos_encoding = PositionalEncoding(512, 60)
    a = sample_pos_encoding(test_string)

    print(a.shape)
    print(a[:, 0, :10])
