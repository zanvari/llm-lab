# 🧠 Few-Shot Document Classifier

This project uses OpenAI's GPT-3.5 to classify short document texts into categories using few-shot prompting, with no training required.

## 🔍 Goal
Given a document snippet, classify it into one of the following:
- `contract`
- `invoice`
- `medical_report`
- `letter`
- `insurance_form`

## 🚀 How it Works
- Prompt-based classification using GPT-3.5
- Few-shot examples embedded directly in the prompt
- No training or labeled dataset required

## 📘 Notebook
- [fewshot_doc_classifier.ipynb](./fewshot_doc_classifier.ipynb) – End-to-end few-shot document classification using GPT-3.5
- 🟢 [Open in Colab](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/fewshot-doc-classifier/fewshot_doc_classifier.ipynb)

## 🧪 Example Prompt
```text
Document:
"Invoice No: 12345. Amount due: $500."
Label:
invoice
```

## 🧠 Technologies Used
- OpenAI GPT-3.5 via LangChain
- Few-shot prompting
- Python + Jupyter Notebook

## 🔧 Setup
```bash
pip install openai langchain tiktoken
```

Set your OpenAI key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## 📂 Future Extensions
- Streamlit or Gradio UI for real-time classification
- Export results to CSV or JSON
- Compare OpenAI vs Mistral models

## 📄 License
MIT

