# 🧾 Key-Value Extraction with LayoutLM

This project demonstrates how to preprocess the FUNSD dataset and fine-tune the LayoutLM model to extract key-value pairs from form-like documents.

## 📚 Highlights

- ✅ Preprocessing for FUNSD annotations
- ✅ Normalized bounding boxes
- ✅ HuggingFace-compatible `Dataset` class
- ✅ Fine-tuning `LayoutLM` with `Trainer`
- ✅ Inference + bounding box visualization

## 🧪 Dataset

This uses the [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/) which contains scanned forms annotated with fields like "question", "answer", and "other".

## 📦 Setup

```bash
pip install -r requirements.txt
```

## 🚀 Run

1. Preprocess samples with `data_preprocessing.py`
2. Fine-tune the model using `layoutlm_funsd.ipynb`
3. Visualize predictions with bounding boxes

## 🧠 Model

Uses `microsoft/layoutlm-base-uncased` from HuggingFace.
