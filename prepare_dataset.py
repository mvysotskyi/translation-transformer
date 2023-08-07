"""
File used to prepare the dataset for the training of the model.
"""

import os

import tensorflow as tf

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing


class Tokenizers:
    """
    Class used to manage tokenizers for multiple languages at the same time.
    """
    def __init__(self):
        self._special_tokens = ["[PAD]", "[START]", "[END]"]
        self._langs = []

    def fit(self, texts: dict[str, list[str]], desired_vocab_sizes: dict[str, int]):
        """
        Method used to fit the tokenizer for each language.

        :param texts: dictionary {language: (file1, file2, ...)}
        """
        assert set(texts.keys()) == set(
            desired_vocab_sizes.keys()
        ), "Languages in texts and 'desired_vocab_sizes' must be the same."

        for lang in texts:
            tokenizer = ByteLevelBPETokenizer()

            tokenizer.train(
                texts[lang],
                vocab_size=desired_vocab_sizes[lang],
                special_tokens=self._special_tokens,
            )
            tokenizer.post_processor = TemplateProcessing(
                single="[START] $A [END]",
                special_tokens=[
                    (token, tokenizer.token_to_id(token))
                    for token in self._special_tokens[1:]
                ],
            )

            self.__setattr__(lang, tokenizer)
            self._langs.append(lang)

    def save(self, path: str) -> None:
        """
        Method used to save the tokenizers.
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        for lang in self._langs:
            self.__getattribute__(lang).save(
                os.path.join(path, f"{lang}_tokenizer")
            )

    @classmethod
    def load(cls, path: str) -> "Tokenizers":
        """
        Method used to load the tokenizers.

        :param path: path where the tokenizers are saved
        :return: Tokenizers object
        """
        tokenizers = cls()

        for file in os.listdir(path):
            lang = file.split("_")[0]
            tokenizer = ByteLevelBPETokenizer.from_file(
                os.path.join(path, file), add_prefix_space=True
            )
            tokenizers.__setattr__(lang, tokenizer)

        return tokenizers


class BilingualDataset:
    """
    Class is used to create datasets for translation tasks.

    ** Text files will not be preprocessed and will be imidiatly tokenized "as is".
    """

    def __init__(
        self,
        raw_sets: dict[str, tuple[str]],
        vocab_sizes: dict[str, int],
        src_trg: dict[str, str],
        max_seq_len: int = 128,
        batch_size: int = 64,
        buffer_size: int = 20000,
        shuffle: bool = True,
    ):
        """
        :param raw_sets: dict {language: files_list, ...}
        :param tokenizers: Tokenizers object
        """
        self.raw_sets = raw_sets
        self.tokenizers = Tokenizers()

        self.vocab_sizes = vocab_sizes
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        self.source = src_trg["source"]
        self.target = src_trg["target"]

    def __fit_tokenizers(self):
        """
        Method used to fit the tokenizers.

        :param desired_vocab_sizes: tuple of desired vocab sizes for each language
        """
        self.tokenizers.fit(self.raw_sets, self.vocab_sizes)

    def __make_dataset(self, files: dict[str, str]) -> tf.data.Dataset:
        """
        Method used to create a dataset from a list of files.

        :param files: dict {language: file_path, ...}
        :return: tf.data.Dataset
        """
        assert len(files) == 2, "Only bilingual datasets are supported."

        texts = {}
        for lang in files:
            with open(files[lang], "r", encoding="utf-8") as file:
                texts[lang] = file.readlines()

        assert (
            len(set(map(len, texts.values()))) == 1
        ), "All files must have the same number of lines."

        #  Tokenize texts
        texts = {
            lang: [
                self.tokenizers.__getattribute__(lang).encode(line).ids
                for line in lines
            ]
            for lang, lines in texts.items()
        }

        src = texts[self.source]
        trg = [
            line[:-1] for line in texts[self.target]
        ]  # Remove the last token ([END])
        trg_shifted = [
            line[1:] for line in texts[self.target]
        ]  # Remove the first token ([START])

        del texts

        # Truncate sequences to the maximum length and pad them to the same length
        src = tf.keras.preprocessing.sequence.pad_sequences(
            src, padding="post", maxlen=self.max_seq_len
        )
        trg = tf.keras.preprocessing.sequence.pad_sequences(
            trg, padding="post", maxlen=self.max_seq_len
        )
        trg_shifted = tf.keras.preprocessing.sequence.pad_sequences(
            trg_shifted, padding="post", maxlen=self.max_seq_len
        )

        # Create dataset
        return tf.data.Dataset.from_tensor_slices(((src, trg), trg_shifted))

    def make(self, save_path: str | None = None) -> list[tf.data.Dataset]:
        """
        Method used to create the datasets.

        :param save_path: path where to save the tokenizers
        :return: list of tf.data.Dataset
        """
        self.__fit_tokenizers()
        self.tokenizers.save(save_path)

        # Create datasets
        datasets = []
        for idx in range(len(self.raw_sets[self.source])):
            files = {lang: self.raw_sets[lang][idx] for lang in self.raw_sets}
            datasets.append(self.__make_dataset(files))

        # Shuffle datasets
        if self.shuffle:
            datasets = [dataset.shuffle(self.buffer_size) for dataset in datasets]

        # Batch datasets
        datasets = [
            dataset.batch(self.batch_size, drop_remainder=True) for dataset in datasets
        ]

        # Save datasets
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            for idx, dataset in enumerate(datasets):
                tf.data.experimental.save(
                    dataset, os.path.join(save_path, f"dataset_{idx}")
                )

        return datasets

    @staticmethod
    def load(path: str) -> tf.data.Dataset:
        """
        Method used to load a dataset.

        :param path: path where the dataset is saved
        :return: tf.data.Dataset
        """
        return tf.data.experimental.load(path)


if __name__ == "__main__":
    raw_sets = {
        "en": ["raw_data/train.en", "raw_data/val.en"],
        "uk": ["raw_data/train.uk", "raw_data/val.uk"],
    }

    vocab_sizes = {"en": 7000, "uk": 7000}

    src_trg = {"source": "uk", "target": "en"}

    dataset = BilingualDataset(raw_sets, vocab_sizes, src_trg)
    datasets = dataset.make("data")
