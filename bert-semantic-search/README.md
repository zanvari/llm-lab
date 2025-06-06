# 🔍 BERT Semantic Search

This project demonstrates how to build a semantic search engine using BERT-based sentence embeddings and FAISS.

## 🧠 Features
- Uses `sentence-transformers` for encoding
- Supports FAISS vector search
- Query interface with cosine similarity
- Easy to adapt for FAQs, support tickets, or docs

## 📁 Structure
- `data/corpus.json`: Text entries to index
- `semantic_search.ipynb`: Main demo notebook
- `search_engine.py`: Utility functions

## 🚀 Run
```bash
pip install -r requirements.txt
```

Then launch the notebook:

```bash
jupyter notebook semantic_search.ipynb
```

## 🧪 Example Use Cases
- Legal clause search
- Contract or FAQ match
- Resume/job search
