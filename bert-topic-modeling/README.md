# 🧠 BERT + Topic Modeling

Cluster documents using BERT embeddings and extract semantic topics with KeyBERT.

## 📄 Dataset
We use a small synthetic set of 10 documents (`data/documents.json`) covering tech, healthcare, finance, and science.

## 🧠 Features
- Sentence-BERT embeddings (`all-MiniLM-L6-v2`)
- HDBSCAN clustering
- Topic keyword extraction with KeyBERT
- UMAP visualization
- Silhouette score evaluation (optional)

## 📘 Notebook
- [`topic_modeling.ipynb`](./topic_modeling.ipynb): Full pipeline walkthrough

## 📁 Structure
```
bert-topic-modeling/
├── data/
│   └── documents.json         # Sample input docs
├── outputs/                   # Visualizations
├── topic_modeling.ipynb       # Main notebook
├── README.md
└── requirements.txt
```

## 🔧 Requirements
See `requirements.txt` for dependencies.

## 🚀 Future Ideas
- Use HDBSCAN soft cluster probabilities
- Cluster labeling via large LLMs
- Upload your own corpora
