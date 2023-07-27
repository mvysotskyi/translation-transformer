"""
Implementations of attention layers.
"""

import tensorflow as tf
from tensorflow.keras import layers

from functools import partial


class SingleAttention(layers.Layer):
    """
    Single Attention Layer.
    """
    def __init__(self, d_k: int, d_v: int, use_mask: bool = False):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.use_mask = use_mask

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer.
        """
        self.query = layers.Dense(self.d_k, input_shape=input_shape, dtype=tf.float32)
        self.key = layers.Dense(self.d_k, input_shape=input_shape, dtype=tf.float32)
        self.value = layers.Dense(self.d_v, input_shape=input_shape, dtype=tf.float32)

        super(SingleAttention, self).build(input_shape)

    def generate_mask(self, tensor_shape: tf.TensorShape) -> tf.Tensor:
        """
        Generate the mask.
        """
        seq_len = tensor_shape[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask

    def apply_mask(self, tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Apply mask to the tensor.
        """
        mask = tf.cast(mask, tf.float32)
        tensor = tensor * mask[tf.newaxis, :]
        # tensor[tensor == 0] = tf.float32.min
        tensor = tf.where(tensor == 0.0, tf.float32.min, tensor)
        return tensor

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.
        """
        q = self.query(inputs[0])
        k = self.key(inputs[1])
        v = self.value(inputs[2])

        attention_weights = tf.matmul(q, k, transpose_b=True)
        if self.use_mask:
            mask = self.generate_mask(tf.shape(attention_weights))
            attention_weights = self.apply_mask(attention_weights, mask)

        attention_weights = tf.map_fn(lambda x: x / tf.math.sqrt(tf.cast(self.d_k, tf.float32)), attention_weights)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        outputs = tf.matmul(attention_weights, v)
        return outputs

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, use_mask: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.use_mask = use_mask

        self.attn_heads = []

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer.
        """
        for _ in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v, self.use_mask))

        self.linear = layers.Dense(input_shape[-1], input_shape=input_shape, dtype=tf.float32)

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.
        """
        attn = [self.attn_heads[i]([inputs, inputs, inputs]) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear
    
MaskedMultiHeadAttention = partial(MultiHeadAttention, use_mask=True)
    
if __name__ == "__main__":
    # test the MultiHeadAttention layer

    sample_ffn = MaskedMultiHeadAttention(512, 512, 8)
    temp_input = tf.random.uniform((5, 60, 512))

    sample_ffn_output = sample_ffn(temp_input)

    print(temp_input.shape)
    