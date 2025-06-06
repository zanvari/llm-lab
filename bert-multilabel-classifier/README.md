# 🧷 BERT Multi-label Classifier

Fine-tune BERT to classify texts into **multiple categories** using sigmoid output and binary cross-entropy loss.

## 📘 Notebook
- [`multilabel_finetune.ipynb`](./notebooks/multilabel_finetune.ipynb) – Full pipeline from data prep to fine-tuning and inference
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/bert-multilabel-classifier/notebooks/multilabel_finetune.ipynb)

## 🧠 Highlights
- 🏷️ Multi-label binarization with `MultiLabelBinarizer`
- 🧠 BERT fine-tuning with sigmoid layer
- 🧪 Predict multiple labels per document

## 📁 Structure
```
bert-multilabel-classifier/
├── data/
├── notebooks/
├── outputs/
├── README.md
└── requirements.txt
```

## 🚀 Usage
```bash
pip install -r requirements.txt
```

## 📄 License
MIT
