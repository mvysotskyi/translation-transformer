# translation-transformer
Implementation of paper "Attention Is All You Need"


## Overview

This repository contains my implementation of the ["Attention Is All You Need"](
https://arxiv.org/abs/1706.03762) paper using TensorFlow and Keras. The paper introduced the revolutionary Transformer model, which utilizes self-attention mechanisms to achieve state-of-the-art performance in various natural language processing tasks. Our implementation focuses on building a Ukrainian-to-English translator using the Transformer architecture.


### TODO
- Train larger model and use more data
- Try to share weights between Embedding layers
- Try to use label smoothing


### Implementation Details

In our implementation, we deviated from the original paper in a few key aspects. One notable departure is that we did not implement shared weights between the Embedding layers. This decision was driven by the significant linguistic differences between the Ukrainian and English languages, which made it more effective to treat them as separate embeddings.

Additionally, while the original paper introduced label smoothing which I desided not to use.

## Training

This Ukrainian-to-English translator model was trained for 20 epochs using the following parameters:

- **Vocabulary size for each language:** 7000
- **Batch Size:** 64
- **Warmup Steps:** 4000
- **Maximum Sequence Length:** 128
- **Model Dimensions:** 128
- **Number of Heads:** 8
- **Feedforward Dimension:** 512
- **Dropout Rate:** 0.1

This model used Keras MultiHeadAttention layers because they are more efficient than the custom implementation.
Tokenization was performed using Byte-Pair Encoding (BPE), which is known for effectively handling subword-level representations in NLP tasks.

### Model specifications

```
Model: "transformer"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 encoder (Encoder)           multiple                  3534848   
                                                                 
 decoder (Decoder)           multiple                  5645824   
                                                                 
 dense_16 (Dense)            multiple                  903000    
                                                                 
=================================================================
Total params: 10,083,672
Trainable params: 10,083,672
Non-trainable params: 0
_________________________________________________________________
```

## Results

Model is quite good at translating simple sentences, but it has problems with complex sentences.
This can be explained by:
- small dataset(only 52k sentences) => small vocabulary
- small model
- huge linguistic difference

### Examples

```text
Input: ти точно знаєш хто я такий
Translated: you know exactly who i am .
Ground truth: you know exactly who i am .
```

```text
Input: ми два найкращі повари в америці
Translated: we are two best in america .
Ground truth: we are the two best chefs in america .
```

```text
Input: він відчував себе дуже самотнім .
Translated: he felt very lonely .
Ground truth: he felt very lonely .
```

```text
Input: він був відомий як великий вчений в області математики.
Translated: he was known as a big scientist in math .
Ground truth: he was known as a great scientist in the field of mathematics .
```

```text
Input: машинне навчання та добування даних часто використовують одні й ті ж методи та техніки .
Translated: the machine learning and good data is often used those same technique and technical technical techniques .
Ground truth: machine learning and data mining often use the same methods and techniques .
```

```text
Input: cьогодні національна мова набула статусу державної , вона вивчається , відроджується і вдосконалюється .
Translated: cultural language has a status quo , and it learns from broader and improve .
Ground truth: today the national language has the status of the state , it is studied , revived and improved .
```


### Instructions to use model

Requirements:
- TensorFlow 2
- HuggingFace Tokenizers


1. Clone this repository:
   ```
   git clone https://github.com/mvysotskyi/translation-transformer.git
   cd translation-transformer
   ```

2. Install the required dependencies.

3. Use `Translator` class from `translation.py` module and `Tokenizers` class from `prepare_dataset.py` to load tokenizers.

#### Example

```python
from prepare_dataset import Tokenizers
from translation import Translator

tokenizers = Tokenizers.load("data")

transformer = Transformer(7000, 7000, 128, 128, 8, 4, 512)
transformer.build(input_shape=[(None, 128), (None, 128)])
transformer.load_weights("checkpoints/transformer_20.h5")

translator = Translator(transformer, tokenizers, {"src": "uk", "trg": "en"})
sentence = "він був відомий як великий вчений ." # he was known as a great scientist .

translated, _ = translator(sentence)
```

### Notes on `prepare_dataset.py`

This module contains the `Tokenizers` and `BilingualDataset` classes, which are used to prepare the dataset from text files. The `Tokenizers` class is used to train the tokenizers for each language, while the `BilingualDataset` class is used to create a dataset of tokenized sentences. So, it may be useful to use these classes to prepare a dataset for a different task.
