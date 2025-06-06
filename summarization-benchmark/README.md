# 📚 Summarization Benchmarking

This project compares extractive and abstractive summarization methods using both traditional NLP and modern LLMs.

## 🧪 Techniques
- **Extractive**: TextRank with spaCy + pytextrank
- **Abstractive**: FLAN-T5 via HuggingFace Transformers
- **Evaluation**: ROUGE metrics for output comparison

## 📘 Notebook
- [`summarization_benchmark.ipynb`](./summarization_benchmark.ipynb)

## 📂 Structure
```
summarization-benchmark/
├── data/
│   └── sample_article.txt
├── outputs/                     # Saved summaries or eval results
├── summarization_benchmark.ipynb
├── README.md
└── requirements.txt
```

## 📦 Requirements
Install with:
```bash
pip install -r requirements.txt
```

## ✅ Goal
Understand trade-offs between extractive vs. abstractive summarization for real-world documents.
