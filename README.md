# 🧪 LLM Lab

Welcome to `llm-lab` — a curated collection of projects, experiments, and research prototypes focused on **Large Language Models (LLMs)** and **Generative AI**. This lab is designed to showcase hands-on applications and practical workflows across areas like Retrieval-Augmented Generation (RAG), document intelligence, fine-tuning, and more.

> 🧠 This lab includes hands-on projects with:
> **BERT**, **Large Language Models (LLMs)**, **Document Classification**, **Key-Value Extraction**, **Named Entity Recognition (NER)**, **Semantic Search**, **Few-shot Learning**, **RAG**, and **Generative AI** using tools like **HuggingFace**, **LangChain**, **FAISS**, and **Transformers**.

---

## 🚀 Projects

### 📄 [`rag-doc-qa`](./rag-doc-qa)
Perform question answering over PDF documents using LangChain, FAISS, and GPT-4.
- 🔍 RAG pipeline with vector search
- 📄 Supports any local PDF
- 🤖 GPT-4 integration for QA

### 🏥 [`rag-healthcare`](./rag-healthcare)
Healthcare-specific RAG system for answering insurance claim questions from PDFs.
- 📑 Example claims PDFs
- 🔍 Embedding + retriever benchmarking
- 📊 Eval pipeline with F1/BLEU

### 🧷 [`fewshot-doc-classifier`](./fewshot-doc-classifier)
Classify documents using zero-shot or few-shot prompting with OpenAI or open-source LLMs.
- 🔢 Multi-class document classification
- 🧠 Few-shot prompt templates
- 🧪 Comparison with fine-tuned classifiers

### 🧾 [`kv-extraction-funsd`](./kv-extraction-funsd)
Key-value pair extraction from forms using LayoutLM and Donut.
- 🖼️ FUNSD dataset for training
- 📚 Transformers for layout-aware extraction
- 📊 Eval using precision, recall, F1

## 🧠 BERT Projects

### 🗂️ [`bert-doc-clustering`](./bert-doc-clustering)
Cluster documents using Sentence-BERT and visualize with UMAP.
- 🔢 Embeddings via `sentence-transformers`
- 🔁 KMeans + Agglomerative clustering
- 📈 2D UMAP visualization

### 📝 [`bert-text-classifier`](./bert-text-classifier)
Fine-tune BERT on text classification tasks with training, evaluation, and visualization.
- 🔢 Multi-class text classification
- 📊 Confusion matrix and metrics
- 🔁 Compare BERT, DistilBERT, RoBERTa

### 🏷️ [`bert-ner-finetune`](./bert-ner-finetune)
Train BERT for Named Entity Recognition (NER) using token classification.
- 🧾 CoNLL-style NER tasks
- 🔍 Visualize token predictions
- 🧪 Includes training and inference scripts

### 📚 [`bert-qa-squad`](./bert-qa-squad)
Fine-tune BERT on extractive QA with the SQuAD dataset.
- ❓ Question-answering with context
- 📈 EM/F1 metrics and batch inference
- 🕒 Inference time benchmarking

### 🔍 [`bert-semantic-search`](./bert-semantic-search)
Build a semantic search engine using Sentence-BERT and FAISS.
- 🧠 Dense embeddings with `sentence-transformers`
- ⚡ FAISS for fast vector search
- 📄 Notebook + script interface

### 🧷 [`bert-multilabel-classifier`](./bert-multilabel-classifier)
Multi-label classification using BERT with sigmoid activation.
- 🔢 Supports multiple labels per sample
- 📈 Precision, recall, F1
- 📊 Per-label analysis and visualizations

### 🧠 [`bert-sentence-similarity`](./bert-sentence-similarity)
Compare semantic similarity between sentence pairs using BERT.
- 🧪 Cosine similarity of embeddings
- 🔁 Pairwise and batched evaluation
- 🧠 Useful for clustering, deduplication

---

## 📚 Goals
- ✅ Build a hands-on portfolio of GenAI/LLM work
- ✅ Learn and share practical LangChain & vector DB integrations
- ✅ Explore document intelligence use cases with modern models

---

## 🔧 Stack
LangChain · OpenAI · HuggingFace · FAISS · Streamlit · PyPDF · Transformers · Colab · LayoutLM

---

## 📬 Contact
Made with 🧠 by [Zahra Anvari](https://github.com/zanvari)

---

## 📄 License
MIT



