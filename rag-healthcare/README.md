# 🏥 RAG for Healthcare Document QA

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline to answer questions from healthcare-related documents such as insurance claims and clinical records.

## 📘 Notebook
- [rag-healthcare.ipynb](./rag-healthcare.ipynb) – Full walkthrough: loading docs, embeddings, retrieval, and answering questions.
- 🟢 [Open in Colab](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/rag-healthcare/rag-healthcare.ipynb)

## 💡 Use Case
Given one or more healthcare documents (PDF/text), answer questions such as:
- What is the patient's diagnosis?
- What treatment was administered?
- Who is the policyholder and what amount was claimed?

## 🔧 Pipeline Overview
- 📄 Load PDF(s)
- 📚 Chunk and embed with `sentence-transformers`
- 🔎 Retrieve relevant context with FAISS
- 🤖 Answer using GPT-3.5 via LangChain

## 📦 Install Requirements
```bash
pip install -r requirements.txt
```

## 📂 File Structure
```
rag-healthcare/
├── data/
│   └── health_insurance_claim.pdf
├── rag-healthcare.ipynb
├── requirements.txt
└── README.md
```

## 📄 License
MIT
