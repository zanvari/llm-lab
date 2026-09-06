# BERT Text Classifier

This project fine-tunes BERT and other transformer-based models for text classification using HuggingFace Transformers and the IMDb dataset.

---

## Structure
- `notebooks/bert_classifier.ipynb` – Fine-tune BERT on binary classification
- `notebooks/bert_comparison.ipynb` – Compare BERT, DistilBERT, and RoBERTa with confusion matrix
- `train.py` – Training script (coming soon)
- `infer.py` – Inference script (coming soon)
- `data/` – Store raw or preprocessed datasets here

---

## Features
- Load and tokenize text datasets (IMDb)
- Fine-tune `bert-base-uncased`, `distilbert-base-uncased`, and `roberta-base`
- Plot confusion matrix and classification report
- Evaluate using accuracy and F1-score

---

## Model Comparison
| Model               | Type       | Notes                      |
|--------------------|------------|----------------------------|
| `bert-base-uncased`| Base BERT  | Standard benchmark         |
| `distilbert-base`  | Lightweight| Faster, smaller version    |
| `roberta-base`     | Robust     | Better performance on many NLP tasks |

---

## Usage
Run the comparison notebook to train and evaluate:
```bash
notebooks/bert_comparison.ipynb
```

---

## Requirements
```bash
transformers
datasets
scikit-learn
torch
seaborn
matplotlib
```
