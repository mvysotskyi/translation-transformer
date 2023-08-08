# translation-transformer
Implementation of paper "Attention Is All You Need"

## Overview

This repository contains my implementation of the ["Attention Is All You Need"](
https://arxiv.org/abs/1706.03762) paper using TensorFlow and Keras. The paper introduced the revolutionary Transformer model, which utilizes self-attention mechanisms to achieve state-of-the-art performance in various natural language processing tasks. Our implementation focuses on building a Ukrainian-to-English translator using the Transformer architecture.

### Implementation Details

In our implementation, we deviated from the original paper in a few key aspects. One notable departure is that we did not implement shared weights between the Embedding layers. This decision was driven by the significant linguistic differences between the Ukrainian and English languages, which made it more effective to treat them as separate embeddings.

Additionally, while the original paper introduced label smoothing which I desided not to use.

## Training

This Ukrainian-to-English translator model was trained for 20 epochs using the following parameters:

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
This is due to the fact that the model is itself small and was trained on a small dataset.

... # some examples

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

3. Use 'Translator' class from 'translation.py' module. 