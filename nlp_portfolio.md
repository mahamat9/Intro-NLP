---
layout: page
title: NLP — Classification & Title Generation
description: Fine-tuning pretrained language models for fake news detection and article title generation
img: assets/img/nlp_thumb.png
importance: 4
category: academic
github: https://github.com/mahamat9/Intro-NLP
---

<p align="center">
  <img src="https://img.shields.io/badge/HuggingFace-T5--small-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/NLP-Classification%20%26%20Generation-blue" />
</p>

---

## Overview

| Field | Detail |
|---|---|
| **Type** | Academic project — Conference Workshop |
| **Supervisor** | Batcouzé I. |
| **Stack** | Python · PyTorch · HuggingFace Transformers · Gensim · tokenizers · TensorBoard |
| **Duration** | February 2025 |

---

## Context

Two complementary NLP tasks exploring both **supervised classification** and **generative** capabilities of modern language models.

| Task | Model | Dataset | Metric |
|---|---|---|---|
| **Fake News Classification** | Custom CNN + Word2Vec | ~45k articles (Kaggle) | Accuracy, F1 |
| **Title Generation** | T5-small (fine-tuned) | TitleGen (Kaggle) | ROUGE-1, ROUGE-2 |

Both were carried out as part of the Batcouzé I. conference-workshop, February 2025.

---

## Part I — Fake News Classification

### Dataset

- ~45,000 English news articles, binary label: `0` = Fake, `1` = True
- Source: [Kaggle — aadyasingh55](https://www.kaggle.com/datasets/aadyasingh55/news-article-classification-with-45k-samples)
- Input: `content = title + " " + text`
- Train / validation / test split applied

### Custom BPE Tokenizer (from scratch)

Using HuggingFace `tokenizers` to build a Byte-Pair Encoding tokenizer:

```python
from tokenizers import Tokenizer, normalizers, pre_tokenizers, trainers, models, processors

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=25_000)
tokenizer.train_from_iterator(get_training_corpus(df), trainer=trainer)
```

Vocabulary size: **25,000 tokens**, normalized with NFD + lowercase stripping.

### Word2Vec Embeddings (Skip-gram)

Word embeddings pre-trained via Gensim to warm-start the embedding layer:

```python
from gensim.models import Word2Vec

w2v_model = Word2Vec(
    vector_size=100,
    window=5,
    min_count=1,
    sg=1           # Skip-gram
)
w2v_model.build_vocab(sentences_iterator)
w2v_model.train(sentences_iterator, total_examples=w2v_model.corpus_count, epochs=5)
```

The learned weights are used to **initialize the embedding layer** of the classifier.

### Model Architectures

Two embedding aggregation variants evaluated:

**Architecture 1 — Mean Pooling**

```python
class TextClassifierMean(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.drop(self.embedding(x))           # (B, seq_len, embed_dim)
        mean = torch.mean(embedded, axis=1)               # (B, embed_dim)
        return self.fc(mean)
```

**Architecture 2 — L2-Normalized Sum Pooling**

```python
class TextClassifierNormL2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.drop(self.embedding(x))           # (B, seq_len, embed_dim)
        summed = torch.sum(embedded, axis=1)              # (B, embed_dim)
        norm = torch.norm(summed, dim=-1, keepdim=True)   # (B, 1)
        return self.fc(summed / (norm + 1e-6))
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 15 |
| Early stopping patience | 6 |

### Visualizations

- **Loss & Accuracy curves** — training vs. validation per epoch
- **F1 score progression** — macro and per-class
- **Confusion matrix** — on held-out test set
- **2D PCA** of sentence embeddings (from the learned Word2Vec space)

---

## Part II — Title Generation with T5

### Dataset

- **TitleGen** — [Kaggle — thejas2002](https://www.kaggle.com/datasets/thejas2002/news-articles-title-generation-dataset)
- ~20% of the full dataset used (feasibility constraint)
- Split: **80/10/10** (train / val / test)

### Preprocessing

```python
import re

def clean_text(text):
    # Remove leading publisher prefix (e.g. "reuters - ")
    text = re.sub(r'^.*?- ', '', text, flags=re.DOTALL)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)
    # Normalize whitespace
    return ' '.join(text.split())
```

Prompt prefix used for T5: `"Generate a title: "`

### Model — T5-small

T5 adopts a **text-to-text** paradigm: both inputs and outputs are raw text strings.

```
Input: "Generate a title: <article text>"
         |
    ┌─────────────────┐
    │  T5 Encoder     │  →  contextualized representations
    └─────────────────┘
         |
    ┌─────────────────┐
    │  T5 Decoder     │  →  auto-regressive token generation
    └─────────────────┘
         |
Output: "<generated title>"
```

| Property | Value |
|---|---|
| Base model | `google/t5-small` |
| Parameters | ~60 million |
| Tokenizer vocab | 32,128 tokens |
| Max input length | 512 tokens |
| Max target length | 45 tokens |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch (train) | 4 |
| Batch (eval) | 6 |
| Epochs | 15 |
| Early stopping patience | 5 |
| Beam search | 4 beams |
| Repetition penalty | 2.5 |

### ROUGE Evaluation

$$ROUGE\text{-}N = \frac{\sum_{s \in \text{Ref}} \sum_{n\text{-gram} \in s} \text{Count}_{\text{match}}(n\text{-gram})}{\sum_{s \in \text{Ref}} \sum_{n\text{-gram} \in s} \text{Count}(n\text{-gram})}$$

- **ROUGE-1** — unigram overlap (word-level)
- **ROUGE-2** — bigram overlap (phrase-level)

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

📦 **Fine-tuned model:** [Ivanhoe9/finetune\_T5\_small\_title\_generation\_NLP\_cours](https://huggingface.co/Ivanhoe9/finetune_T5_small_title_generation_NLP_cours)

---

## Results Summary

| Task | Metric | Score |
|---|---|---|
| Fake News Classification | Accuracy | ~0.92 |
| Fake News Classification | F1-Score (macro) | ~0.91 |
| Title Generation | ROUGE-1 | tracked per epoch |
| Title Generation | ROUGE-2 | tracked per epoch |

---

## Project Structure

```
Intro-NLP/
├── data/
│   ├── tp1/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── evaluation.csv
│   ├── tp2/
│   │   └── titlegen/
│   │       └── train.csv
│   └── tokenizers/
│       └── tokenizer.json
├── models_checkpoints/
│   ├── tp1/
│   └── tp2/
├── logs/
│   ├── tp1/
│   └── tp2/
├── src/
│   ├── module_train_tp1.py
│   ├── tp1_classification/
│   │   └── correction_tp1_classification_fakenews.ipynb
│   └── tp2_title_generation/
│       └── correction_tp2_finetune_t5_title_generation.ipynb
└── README.md
```

---

## Tools & Stack

| Tool | Role |
|---|---|
| **PyTorch** | Model implementation & training |
| **HuggingFace Transformers** | T5-small, tokenizers, trainer |
| **Gensim** | Word2Vec skip-gram pretraining |
| **tokenizers** | Custom BPE tokenizer (from scratch) |
| **TensorBoard** | Training metrics visualization |
| **pandas / NumPy** | Data loading & preprocessing |
| **scikit-learn** | Metrics (F1, confusion matrix), PCA |
| **matplotlib / seaborn** | Plots |

---

<div style="display:flex; gap:1rem; justify-content:center; margin: 2.5rem 0;">
  <a href="https://github.com/mahamat9/Intro-NLP" class="btn btn-primary" role="button" target="_blank">View on GitHub</a>
  <a href="https://huggingface.co/Ivanhoe9/finetune_T5_small_title_generation_NLP_cours" class="btn btn-secondary" role="button" target="_blank">Model on HuggingFace</a>
</div>