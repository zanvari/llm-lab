# RAG Document QA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/rag-doc-qa/rag-doc-qa-explained.ipynb)

This project performs Question Answering over PDF documents using LangChain, FAISS, and GPT-4.

## 📄 How it works
- Loads single or multiple PDFs
- Splits them into chunks
- Embeds with OpenAI or HuggingFace
- Stores in FAISS or Chroma vector DB
- Retrieves chunks using Similarity or MMR
- Generates answers using OpenAI GPT

## 🛠 Setup
```bash
pip install -r requirements.txt
```

Add your PDF(s) to the `data/` folder.

## 🚀 Run
```bash
python app.py  # basic run
```

## 📘 Notebook
- [rag-doc-qa-explained.ipynb](./rag-doc-qa-explained.ipynb) – Full walkthrough with markdown explanations, embeddings, retrievers, and LLMs
- 🟢 [Open in Colab](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/rag-doc-qa/rag-doc-qa-explained.ipynb)

## 💬 Streamlit App
Run the interactive web UI:
```bash
streamlit run streamlit_app.py
```

## 🧪 Evaluate with Test Queries
Run all questions listed in `queries.txt` and log answers:
```bash
python test_queries.py
```

Example `queries.txt`:
```
What are the termination conditions?
Who are the parties involved?
What is the monthly fee?
```

## ✅ Features
- Single and multi-document support
- Multiple embedding models (OpenAI, HuggingFace)
- Similarity vs. MMR retrievers
- Streamlit interface
- Automated query evaluation

## 🔧 Technologies Used
LangChain · OpenAI · HuggingFace · FAISS · Chroma · Streamlit · PyPDF · Transformers

## 📄 License
MIT

