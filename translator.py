"""
Translator module using pretrained model.
"""

import tensorflow as tf

from prepare_dataset import Tokenizers
from transformer import Transformer


class Translator(tf.Module):
    def __init__(self, transformer: Transformer, tokenizers: Tokenizers, src_trg: dict[str, str]):
        self.tokenizers = tokenizers
        self.src_trg = src_trg
        self.transformer = transformer

    def __call__(self, sentence, max_length=128):
        sentence = self.tokenizers.__getattribute__(self.src_trg["src"]).encode(sentence).ids

        encoder_input = sentence
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences(
            [encoder_input], maxlen=self.transformer.seq_len, padding="post"
        )
   
        start = self.tokenizers.en.token_to_id("[START]")
        end = self.tokenizers.en.token_to_id("[END]")

        output_array = tf.TensorArray(dtype=tf.int64, size=128)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length - 1):
            output = tf.transpose(output_array.stack())
            output = tf.expand_dims(output, axis=0)

            predictions = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, i : i + 1, :]
            
            predicted_id = tf.argmax(predictions, axis=-1)[0]
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id[0] == end:
                break

        tokens_ids = tf.transpose(output_array.stack())
        text = self.tokenizers.__getattribute__(self.src_trg["trg"]).decode(tokens_ids.numpy())

        return text, tokens_ids


if __name__ == "__main__":
    tokenizers = Tokenizers.load("data")

    transformer = Transformer(7000, 7000, 128, 128, 8, 4, 512)
    transformer.build(input_shape=[(None, 128), (None, 128)])
    transformer.load_weights("transformer.h5")

    translator = Translator(transformer, tokenizers, {"src": "uk", "trg": "en"})

    # Sample sentcence in ukranian
    sentence = "він надіслав резервні сили для охорони кордону ."
    sentence = "як ти себе почуваєш ?"
    sentence = "машинне навчання та добування даних часто використовують одні й ті ж методи ."
    sentence = "я не в небезпеці , я і ця небезпека ."
    sentence = "я той хто стукає ."
    sentence = "він був відомий як великий вчений ."
    sentence = "тому що я так сказав ."
    sentence = "він вже ніколи не стане таким яким був раніше ."
    sentence = "як ти знав , що це було тут ?"
    ground_truth = "you know exactly who i am."

    translated, _ = translator(sentence)

    print(f"Input: {sentence}")
    print(f"Ground truth: {ground_truth}")
    print(f"Translated: {translated}")
