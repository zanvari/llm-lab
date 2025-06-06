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
Key-value pair extraction from forms using LayoutLM and Donut models.
- 🖼️ FUNSD dataset for training
- 🧠 Layout-aware models: LayoutLM & Donut
- 📊 Eval using precision, recall, F1

### 🧠 [`bert-text-classifier`](./bert-text-classifier)
Fine-tune BERT on text classification tasks using HuggingFace Transformers.
- 🧹 Preprocessing with `datasets` and `BertTokenizer`
- 🏋️ Training with `Trainer` API
- 📈 Evaluate using accuracy and F1
- 🔮 Predict on new text samples

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
LangChain · OpenAI · HuggingFace · FAISS · Streamlit · PyPDF · Transformers · Colab · LayoutLM · Donut · BERT

---

## 📬 Contact
Made with 🧠 by [Zahra Anvari](https://github.com/zanvari)

---

## 📄 License
MIT
