"""
Multi-head attention layer.
"""

import tensorflow as tf
from tensorflow import keras

from .attention import ScaledDotProductAttention


# class MultiHeadAttention(keras.layers.Layer):
#     """
#     Multi-head attention layer.
#     """
#     def __init__(self, d_model: int, num_heads: int, **kwargs):
#         super(MultiHeadAttention, self).__init__(**kwargs)

#         self.d_model = d_model
#         self.num_heads = num_heads

#         assert d_model % num_heads == 0

#         self.d_k = d_model // num_heads
#         self.d_v = d_model // num_heads

#         self.w_q = keras.layers.Dense(self.d_k * self.num_heads, use_bias=False)
#         self.w_k = keras.layers.Dense(self.d_k * self.num_heads, use_bias=False)
#         self.w_v = keras.layers.Dense(self.d_v * self.num_heads, use_bias=False)

#         # print(self.w_q.summary())
#         # exit()

#         self.attention = ScaledDotProductAttention()

#         self.output_projection = keras.layers.Dense(d_model)

#     def reshape_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
#         """
#         Reshape the last dimension of tensor to (num_heads, d_k).
#         """
#         tensor = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], self.num_heads, -1))
#         tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])

#         return tensor

#     def reshape_tensor_back(self, tensor: tf.Tensor) -> tf.Tensor:
#         """
#         Reshape the last dimension of tensor back to d_model.
#         """
#         tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
#         tensor = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], self.d_k))

#         return tensor

#     def call(self, queries: tf.Tensor, keys: tf.Tensor, values: tf.Tensor,
#              mask: tf.Tensor | None = None):
#         q_reshaped = self.reshape_tensor(self.w_q(queries))
#         k_reshaped = self.reshape_tensor(self.w_k(keys))
#         v_reshaped = self.reshape_tensor(self.w_v(values))

#         attention = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
#         attention = self.reshape_tensor_back(attention)

#         output = self.output_projection(attention)
#         return output

class MultiHeadAttention(keras.layers.Layer):
    """
    Multi-head attention layer.
    """
    def __init__(self, key_dim: int = 512, n_heads: int = 2, value_dim: int | None = None):
        super(MultiHeadAttention, self).__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.n_heads = n_heads

        self.attn_heads = [ScaledDotProductAttention(self.key_dim, self.value_dim) for _ in range(self.n_heads)]
        self.w_o = keras.layers.Dense(self.key_dim)

    def call(self, input1: tf.Tensor, input2: tf.Tensor, input3: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        """
        Call the layer.
        """
        attn = [self.attn_heads[i](input1, input2, input3, mask) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.w_o(concat_attn)
        return multi_linear
    
if __name__ == "__main__":
    sample_input = tf.random.normal((2, 10, 512))

    multihead_attention = MultiHeadAttention(512, 8)
    output = multihead_attention(sample_input, sample_input, sample_input)

    print(output.shape)
