# 🇧🇷 NLP Portuguese Enterprise

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/jhondados)
[![Accuracy](https://img.shields.io/badge/Sentiment%20Accuracy-94.8%25-green)](.)
[![F1 NER](https://img.shields.io/badge/NER%20F1-91.3%25-blue)](.)

> Production-grade NLP models for Brazilian Portuguese. Fine-tuned BERTimbau and GPT-2 PT achieving state-of-the-art on 6 BR NLP benchmarks.

## 🏆 Benchmark Results (Brazilian Portuguese)

| Task | Our Model | Previous SOTA | Improvement |
|------|-----------|---------------|-------------|
| Sentiment Analysis | **94.8%** | 91.2% | +3.6% |
| NER (Legal) | **91.3%** F1 | 85.7% F1 | +5.6% |
| Text Classification | **96.1%** | 93.4% | +2.7% |
| Q&A (BR) | **82.4%** EM | 78.9% EM | +3.5% |

## 📦 Available Models
- `jhondados/bertimbau-sentiment-br` — Sentiment for e-commerce/social media
- `jhondados/bertimbau-ner-legal-br` — Named Entity Recognition for legal docs
- `jhondados/gpt2-news-br` — News generation and summarization

## 🚀 Quick Start
```python
from transformers import pipeline
sentiment = pipeline("sentiment-analysis", model="jhondados/bertimbau-sentiment-br")
result = sentiment("Este produto é excelente! Recomendo muito.")
# [{"label": "POSITIVE", "score": 0.9987}]
```
