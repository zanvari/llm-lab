# 🏷️ BERT NER Fine-tuning

Fine-tune BERT for Named Entity Recognition (NER) using HuggingFace and the CoNLL-2003 dataset.

---

## 📁 Structure
- `notebooks/ner_finetune.ipynb` – Main notebook for training + evaluation
- `data/` – Holds datasets or preprocessed splits
- `outputs/` – Stores saved models or predictions

---

## ✅ Goals
- Load CoNLL-2003 NER dataset
- Fine-tune BERT for token classification
- Evaluate with precision, recall, F1-score

---

## 🧪 Models
- `bert-base-cased` (used in CoNLL-2003 benchmarks)

---

## 🔧 Requirements
```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies
- transformers
- datasets
- seqeval
- torch
- pandas
- matplotlib
- scikit-learn
