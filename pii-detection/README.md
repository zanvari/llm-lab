# 🔐 PII Detection

This project detects Personally Identifiable Information (PII) in text using both traditional NER methods and LLM-based prompts.

## 🧠 Methods
- **spaCy NER** for rule-based entity recognition
- **LLM (OpenAI GPT)** for zero-shot PII detection
- *(Optional)* HuggingFace transformer-based NER models

## 📘 Notebook
- [`pii_detection.ipynb`](./pii_detection.ipynb): Compare traditional and LLM-based PII detection

## 📂 Structure
```
pii-detection/
├── data/
│   └── sample_doc.txt
├── outputs/                     # Annotated or redacted results
├── pii_detection.ipynb
├── README.md
└── requirements.txt
```

## 📦 Requirements
Install with:
```bash
pip install -r requirements.txt
```

## ✅ Goal
Compare conventional and LLM-powered PII detection on real-world samples.
