# RAG Document QA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/rag-doc-qa/rag-doc-qa.ipynb)

This project performs Question Answering over PDF documents using LangChain, FAISS, and GPT-4.

## 📄 How it works
- Loads a PDF
- Splits it into chunks
- Embeds & stores in FAISS
- Retrieves chunks based on query
- Generates an answer using OpenAI GPT

## 🛠 Setup
```bash
pip install -r requirements.txt
```

Add your PDF to the `data/` folder and update the filename in `app.py`.

## 🚀 Run
```bash
python app.py
```

## 📘 Notebook
- [rag-doc-qa-explained.ipynb](./rag-doc-qa-explained.ipynb) – Full walkthrough with markdown explanations, embeddings, retrievers, and LLMs
- 🟢 [Open in Colab](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/rag-doc-qa/rag-doc-qa-explained.ipynb)

## ✅ Example
```
Query: What are the termination conditions in the contract?
```
```
Answer: <Generated response>
```

## ✅ Features
- Single and multi-document support
- Different retrievers (similarity, MMR)
- OpenAI and HuggingFace embeddings
- Easy-to-follow notebook

## 🔧 Technologies Used
LangChain · OpenAI · HuggingFace · FAISS · PyPDF · Transformers

## 📄 License
MIT

