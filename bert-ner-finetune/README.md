# BERT NER Fine-tuning

Fine-tune BERT for Named Entity Recognition (NER) using HuggingFace and the CoNLL-2003 dataset.

---

## Structure
- `notebooks/ner_finetune.ipynb` – Main notebook for training + evaluation
- `notebooks/ner_visualize.ipynb` – Visualize model predictions on custom text
- `train.py` – Script to fine-tune BERT
- `infer.py` – Inference script for user-defined text
- `evaluate.py` – Evaluation script with F1-score and classification report
- `data/` – Holds datasets or preprocessed splits
- `outputs/` – Stores saved models or predictions

---

## Goals
- Load and preprocess CoNLL-2003 dataset
- Fine-tune BERT for token classification
- Evaluate with precision, recall, F1
- Visualize predictions in plain text and color-coded HTML

---

## Models
- `bert-base-cased` (used in CoNLL-2003 benchmarks)

---

## Setup
```bash
pip install -r requirements.txt
```

---

## Dependencies
- transformers
- datasets
- seqeval
- torch
- pandas
- matplotlib
- scikit-learn
