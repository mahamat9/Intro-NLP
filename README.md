# Intro-NLP — Classification & Title Generation

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%20HuggingFace-Transformers-yellow?logo=huggingface)
![MIT License](https://img.shields.io/badge/License-MIT-green)

> Fine-tuning pretrained language models for **fake news classification** and **article title generation**.  
> Carried out during the **Batcouzé I.** conference/workshop — February 2025.

---

## Table of Contents

- [Table of Contents](#-table-of-contents)
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [TP1 — Fake News Classification](#-tp1--fake-news-classification)
  - [Dataset](#dataset)
  - [Custom BPE Tokenizer](#custom-bpe-tokenizer-from-scratch)
  - [Word2Vec Embeddings](#word2vec-embeddings-skip-gram)
  - [Model Architectures](#model-architectures)
  - [Training](#training)
  - [Results](#results)
- [TP2 — Title Generation with T5](#️-tp2--title-generation-with-t5)
  - [Dataset](#dataset-1)
  - [Preprocessing](#preprocessing)
  - [Model — T5-small](#model--t5-small)
  - [Training](#training-1)
  - [ROUGE Evaluation](#rouge-evaluation)
  - [HuggingFace Hub Deployment](#huggingface-hub-deployment)
  - [Results](#results-1)
- [Setup](#️-setup)
  - [Requirements](#requirements)
  - [Download Data](#download-data)
- [Author](#-author)

---

## Overview

| Field | Detail |
|---|---|
| **Workshop** | Batcouzé I — Conference (February 2025) |
| **Supervisor** | S. Lamprier |
| **Tasks** | Fake News Classification + Title Generation |
| **Stack** | Python · PyTorch · HuggingFace Transformers · Gensim · TensorBoard |

Two complementary NLP tasks exploring both **supervised classification** (CNN + Word2Vec) and **generative** capabilities (T5-small fine-tuning).

---

## Project Structure

```
Intro-NLP/
├── data/
│   ├── train (2).csv
│   ├── test (1).csv
│   ├── evaluation.csv
│   ├── tp2/
│   │   └── titlegen/
│   │       └── train.csv
│   └── tokenizers/
│       └── tokenizer.json          # Custom BPE tokenizer
├── models_checkpoints/
│   ├── tp1/                        # Classification model weights
│   └── tp2/                        # T5 fine-tuned weights
├── logs/                           # TensorBoard logs
│   ├── tp1/
│   └── tp2/
├── src/
│   ├── module_train_tp1.py         # Shared training utilities
│   ├── tp1_classification_fakenews.ipynb
│   └── tp2_finetune_t5_title_generation.ipynb
├── requirement.txt
└── README.md
```

---

## TP1 — Fake News Classification

### Dataset

- **Source:** [Kaggle — aadyasingh55](https://www.kaggle.com/datasets/aadyasingh55/news-article-classification-with-45k-samples)
- **Size:** ~45,000 English news articles
- **Labels:** `0` = Fake, `1` = True
- **Input:** `content = title + " " + text`

### Custom BPE Tokenizer (from scratch)

Built from scratch using HuggingFace `tokenizers` library:

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=25_000)
tokenizer.train_from_iterator(get_training_corpus(df), trainer=trainer)
tokenizer.save("data/tokenizers/tokenizer.json")
```

### Word2Vec Embeddings (Skip-gram)

Pretrained word embeddings via Gensim to warm-start the classifier:

```python
from gensim.models import Word2Vec

w2v_model = Word2Vec(
    vector_size=100,
    window=5,
    min_count=1,
    sg=1          # Skip-gram
)
w2v_model.build_vocab(sentences_iterator)
w2v_model.train(sentences_iterator, total_examples=w2v_model.corpus_count, epochs=5)
```

Weights are loaded into the embedding layer before training.

### Model Architectures

Two variants tested:

| Variant | Pooling Strategy | Formula |
|---|---|---|
| **Mean Pooling** | Average over sequence | $\frac{1}{L}\sum_{i=1}^L e_i$ |
| **L2-Norm Pooling** | Sum + L2 normalization | $\frac{\sum_i e_i}{\|\sum_i e_i\|}$ |

### Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 15 |
| Early stopping patience | 6 |

### Results

| Metric | Score |
|---|---|
| Accuracy | ~0.92 |
| F1-Score (macro) | ~0.91 |

---

## TP2 — Title Generation with T5

### Dataset

- **Source:** [Kaggle — thejas2002](https://www.kaggle.com/datasets/thejas2002/news-articles-title-generation-dataset)
- **Subset:** ~20% of full dataset
- **Split:** 80/10/10 (train / val / test)

### Preprocessing

```python
import re

def clean_text(text):
    # Strip publisher prefix
    text = re.sub(r'^.*?- ', '', text, flags=re.DOTALL)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    # Normalize whitespace
    return ' '.join(text.split())
```

T5 prompt prefix: `"Generate a title: "`

### Model — T5-small

`google/t5-small` — ~60M parameters, encoder-decoder architecture.

```
Input: "Generate a title: <article text>"
         |
    ┌─────────────────┐
    │  T5 Encoder     │  →  contextualized context
    └─────────────────┘
         |
    ┌─────────────────┐
    │  T5 Decoder     │  →  auto-regressive generation
    └─────────────────┘
         |
Output: "<generated title>"
```

| Property | Value |
|---|---|
| Parameters | ~60M |
| Tokenizer vocab | 32,128 |
| Max input | 512 tokens |
| Max target | 45 tokens |

### Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Train batch | 4 |
| Eval batch | 6 |
| Epochs | 15 |
| Early stopping patience | 5 |
| Beam search | 4 |
| Repetition penalty | 2.5 |

### ROUGE Evaluation

$$ROUGE\text{-}N = \frac{\sum_{s \in \text{Ref}} \sum_{n\text{-gram} \in s} \text{Count}_{\text{match}}(n\text{-gram})}{\sum_{s \in \text{Ref}} \sum_{n\text{-gram} \in s} \text{Count}(n\text{-gram})}$$

### HuggingFace Hub Deployment

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=best_weight_path,
    repo_id="Ivanhoe9/finetune_T5_small_title_generation_NLP_cours",
    repo_type="model"
)
```

**Fine-tuned model:** [Ivanhoe9/finetune\_T5\_small\_title\_generation\_NLP\_cours](https://huggingface.co/Ivanhoe9/finetune_T5_small_title_generation_NLP_cours)

### Results

| Metric | Status |
|---|---|
| ROUGE-1 | Tracked per epoch |
| ROUGE-2 | Tracked per epoch |

---

## Setup

### Requirements

```bash
git clone https://github.com/mahamat9/Intro-NLP
cd Intro-NLP
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0
transformers>=4.30
tokenizers>=0.13
gensim>=4.3
evaluate>=0.4
pandas>=1.5
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
tensorboard>=2.13
huggingface-hub>=0.16
```

### Download Data

Download datasets from Kaggle using the API:

```bash
# Install Kaggle API
pip install kaggle

# Place your kaggle.json credentials in ~/.kaggle/kaggle.json
# Then download the datasets:

# TP1 — Fake News Classification
kaggle datasets download -d aadyasingh55/news-article-classification-with-45k-samples
unzip -o news-article-classification-with-45k-samples.zip -d data/tp1/

# TP2 — Title Generation
kaggle datasets download -d thejas2002/news-articles-title-generation-dataset
unzip -o news-articles-title-generation-dataset.zip -d data/tp2/
```

---

> **Workshop:** Batcouzé I. Conference — February 2025