"""
Train transformer model.
"""

# import os

import tensorflow as tf
from tensorflow import keras

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from transformer import Transformer
from lr_scheduler import LearninRateScheduler

from prepare_dataset import BilingualDataset, Tokenizers


train_path = "data/dataset_0"
val_path = "data/dataset_1"


def mask_loss_function(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


if __name__ == "__main__":
    train_ds = BilingualDataset.load(train_path)
    val_ds = BilingualDataset.load(val_path)
    
    transformer = Transformer(7000, 7000, 128, 128, 8, 4, 512)
    transformer.build(input_shape=[(None, 128), (None, 128)])

    transformer.summary()

    optimizer = Adam(learning_rate=LearninRateScheduler(128, 4000), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer.compile(optimizer=optimizer, loss=mask_loss_function, metrics=[masked_accuracy])
    transformer.fit(train_ds, validation_data=val_ds, epochs=1)

    transformer.save_weights("transformer.h5")
