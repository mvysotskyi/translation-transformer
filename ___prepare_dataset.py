import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

tokenizers = tf.saved_model.load("ted_hrlr_translate_pt_en_converter")

def ragged_to_tensor(ragged):
    out = ragged.to_tensor(0)
    # pad to 128
    print(tf.shape(out))
    out = tf.pad(out, [[0, 0], [0, 128-tf.shape(out)[1]]])
    return out

MAX_TOKENS=128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = ragged_to_tensor(pt)  # Convert ragged to tensor.

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = ragged_to_tensor(en[:, :-1])
    en_labels = ragged_to_tensor(en[:, 1:])  # Drop the [START] tokens

    return (pt, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 32



def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

# Save datasets

tf.data.experimental.save(train_batches, "datasets/train")
tf.data.experimental.save(val_batches, "datasets/test")

print(tokenizers.en.get_vocab_size())
print(tokenizers.pt.get_vocab_size())