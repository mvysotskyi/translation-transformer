"""
Implementations of attention layers.
"""

import tensorflow as tf
from tensorflow.keras import layers

class ScaledDotProductAttention(layers.Layer):
    """
    Scaled dot-product attention layer.
    """
    def __init__(self, d_k: int, d_v: int, use_caual_mask: bool = False):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.use_causal_mask = use_caual_mask

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer.
        """
        self.query = layers.Dense(self.d_k, input_shape=input_shape, dtype=tf.float32)
        self.key = layers.Dense(self.d_k, input_shape=input_shape, dtype=tf.float32)
        self.value = layers.Dense(self.d_v, input_shape=input_shape, dtype=tf.float32)

        super(ScaledDotProductAttention, self).build(input_shape)

    def generate_mask(self, shape: tuple[int, int]) -> tf.Tensor:
        """
        Generate mask.
        """
        mask = 1 - tf.linalg.band_part(tf.ones(shape), -1, 0)
        return mask

    def apply_mask(self, tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Apply mask to the tensor.
        """
        mask = tf.cast(mask, tf.float32)
        tensor = tensor * mask[tf.newaxis, :]
        tensor = tf.where(tensor == 0.0, tf.float32.min, tensor)
        return tensor

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.
        """
        assert len(inputs) == 3
        query, keys, value = inputs

        attention_weights = tf.matmul(query, keys, transpose_b=True)
        attention_weights = attention_weights / tf.math.sqrt(tf.cast(self.d_k, tf.float32))

        if self.use_causal_mask is not None:
            shape = tf.shape(attention_weights)
            attention_weights = self.apply_mask(attention_weights, self.generate_mask(shape[1:]))

        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        outputs = tf.matmul(attention_weights, value)
        return outputs

class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention layer.
    """
    def __init__(self, d_model: int, n_heads: int, use_causal_mask: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.use_causal_mask = use_causal_mask
        self.attn_heads = []

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer.
        """
        for _ in range(self.n_heads):
            self.attn_heads.append(ScaledDotProductAttention(self.d_k, self.d_v, self.use_causal_mask))

        self.linear = layers.Dense(input_shape[-1], input_shape=input_shape, dtype=tf.float32)
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, input1: tf.Tensor, input2: tf.Tensor, input3: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.
        """
        attn = [self.attn_heads[i]([input1, input2, input3]) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear
    
if __name__ == "__main__":
    # test the MultiHeadAttention layer

    sample_ffn = MultiHeadAttention(512, 8)
    temp_input = tf.random.uniform((5, 60, 512))

    sample_ffn_output = sample_ffn(temp_input)

    print(temp_input.shape)
    