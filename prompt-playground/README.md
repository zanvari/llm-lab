# 🧪 Prompt Engineering Playground

This project explores different prompt styles and their effect on LLM outputs. It provides a simple interface to experiment with:

- 🔹 Zero-shot prompts
- 🔸 Few-shot prompts
- 🔍 Chain-of-thought reasoning

## 📘 Notebook
- [`prompt_playground.ipynb`](./prompt_playground.ipynb): Compare outputs from multiple LLMs on various prompt strategies.

## 📂 Structure
```
prompt-playground/
├── data/
│   └── prompts.json             # Example prompts (task, type, input)
├── outputs/                     # Saved logs or comparisons
├── prompt_playground.ipynb      # Main notebook
├── README.md
└── requirements.txt
```

## 🧠 Models Used (Simulated)
- OpenAI (e.g., GPT-4)
- Mistral
- LLaMA

You can integrate real APIs by replacing the simulated `query_*` functions in the notebook.

## 📦 Requirements
See `requirements.txt` for dependencies.
