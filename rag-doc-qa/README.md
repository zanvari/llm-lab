# RAG Document QA

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

