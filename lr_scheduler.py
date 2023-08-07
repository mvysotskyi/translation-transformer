"""
LR Scheduler implementation from paper:
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf
"""

import tensorflow as tf
from tensorflow import keras


class LearninRateScheduler(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler.
    """
    def __init__(self, d_model: int, warmup_steps: int = 4000, **kwargs):
        super(LearninRateScheduler, self).__init__(**kwargs)

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: int):
        step = tf.cast(step, tf.float32)

        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
