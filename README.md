# 🧪 LLM Lab

Welcome to `llm-lab` — a curated collection of projects, experiments, and research prototypes focused on **Large Language Models (LLMs)** and **Generative AI**. This lab is designed to showcase hands-on applications and practical workflows across areas like Retrieval-Augmented Generation (RAG), document intelligence, fine-tuning, and more.

---

## 🚀 Projects

### 📄 [`rag-doc-qa`](./rag-doc-qa)
Perform question answering over PDF documents using LangChain, FAISS, and GPT-4.
- 🔍 RAG pipeline with vector search
- 📄 Supports any local PDF
- 🤖 GPT-4 integration for QA

### 🧷 [`fewshot-document-classifier`](./fewshot-doc-classifier)
Classify documents using zero-shot or few-shot prompting with OpenAI or open-source LLMs.
- 🔢 Multi-class document classification
- 🧠 Few-shot prompt templates
- 🧪 Comparison with fine-tuned classifiers

### 🧾 [`kv-extraction-funsd`](./kv-extraction-funsd)
Key-value pair extraction from forms using LayoutLM or Donut models.
- 🖼️ FUNSD dataset for training
- 📚 Transformers for layout-aware extraction
- 📊 Eval using precision, recall, F1

### 🏷️ [`bert-ner-finetune`](./bert-ner-finetune)
Fine-tune BERT for Named Entity Recognition using the CoNLL-2003 dataset.
- 🔍 Token-level classification
- 📊 Evaluate F1 and visualize predictions

### ❓ [`bert-qa-squad`](./bert-qa-squad)
Fine-tune BERT on the SQuAD dataset for extractive question answering.
- 📚 Load & preprocess SQuAD v1.1
- 🧠 Train `BertForQuestionAnswering`
- 🔍 Inference on custom questions

### 🧠 `llama-finetune-colab` *(Planned)*
Google Colab notebook to fine-tune LLaMA2 with PEFT and LoRA.
- 🔧 HuggingFace + PEFT
- 🧪 Finetuning small-scale datasets

### 💬 `streamlit-chat-ui` *(Planned)*
Interactive web UI for querying LLMs with memory and custom tools.
- 🧑‍💻 Chat history
- 📥 File upload + doc-based QA

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
