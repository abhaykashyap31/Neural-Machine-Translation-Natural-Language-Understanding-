# English-to-Hindi Neural Machine Translation System

## Overview

This project implements a Neural Machine Translation (NMT) system for translating English sentences to Hindi. It explores and compares three architectures:

* **LSTM-based Seq2Seq model**
* **Transformer model**
* **Hybrid Transformer + BiLSTM model**

Through extensive experiments, ablation studies, and evaluations, we assess the effectiveness of these models in translating natural language using BLEU scores and qualitative analysis.

## Date of Submission

March 31, 2025

## Features

* End-to-end translation from English to Hindi
* Tokenization, vocabulary generation, and embedding initialization (Word2Vec/GloVe or Xavier)
* Efficient batching and padding with PyTorch `collate_fn`
* Custom LSTM-based encoder-decoder with teacher forcing
* Transformer model implementation with multi-head attention and positional encoding
* Hybrid model combining LSTM encoder and Transformer decoder with residual connections
* BLEU score evaluation and ablation analysis
* Visualization of training performance and translation examples

## Dataset

* Parallel corpus of English-Hindi sentence pairs.
* Preprocessing includes:

  * Truncating to 50 tokens
  * Special tokens: `<sos>`, `<eos>`, `<pad>`, `<unk>`

## Architectures

### 1. LSTM-based Seq2Seq

* Bi-directional LSTM encoder
* Unidirectional LSTM decoder
* Autoregressive decoding with teacher forcing
* Performs reasonably but struggles with translation accuracy

### 2. Transformer

* Encoder-decoder with multi-head self-attention
* Positional encoding
* Stronger contextual representation but slow training
* Robust to hyperparameter changes

### 3. Hybrid (Transformer + BiLSTM)

* LSTM encoder + Transformer decoder
* Learnable residual connection with alpha weight
* Best BLEU score (0.69) and training efficiency
* Balanced architecture yields superior performance

## Ablation Studies

### LSTM

* Variants: reduced/increased layers, hidden size, dropout
* Identical BLEU scores (0.4638) across variants
* Translation quality concerns and efficiency trade-offs analyzed

### Transformer

* Variants: reduced layers, attention heads, feedforward dimension
* BLEU score: \~0.4542
* Reduced layers improve training time with no quality drop

### Hybrid

* Variants: base, balanced, transformer-heavy
* Balanced config is best: BLEU 0.69 and fastest training (570s)
* Transformer-heavy failed to produce valid translations

## Evaluation

* **BLEU Score**: Main metric for evaluation
* **Loss Trends**: Tracked across epochs for training convergence
* **Sample Translations**: Provided for qualitative assessment
* **Visualizations**: Bar plots and line graphs for model comparisons

## Key Findings

* Transformer is more robust than LSTM
* LSTM tends to generate off-topic technical text
* Hybrid model offers best translation performance and speed
* Tokenization and evaluation metrics are critical limitations
* BLEU alone may not reflect true translation quality

## Future Work

* Implement subword tokenization (BPE/WordPiece)
* Use human evaluation or chrF for better assessment
* Explore better hybrid ratios and residual mechanisms
* Integrate online learning and dictionary-based augmentation


