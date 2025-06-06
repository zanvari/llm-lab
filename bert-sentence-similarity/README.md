# 🔗 BERT Sentence Similarity

This project fine-tunes BERT to score similarity between pairs of sentences using the STS-B dataset (GLUE benchmark).

## 📘 Notebook
- [`similarity_finetune.ipynb`](./notebooks/similarity_finetune.ipynb) – Fine-tuning, evaluation, and inference
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/bert-sentence-similarity/notebooks/similarity_finetune.ipynb)

## 📚 Dataset
- **GLUE STS-B**: Semantic Textual Similarity Benchmark
- Sentence pairs with similarity scores from 0 to 5 (normalized to [0, 1])

## 🧠 Model
- `bert-base-uncased` fine-tuned using HuggingFace Trainer
- Regression head (single output neuron)

## 🧪 Evaluation
- Metrics: Pearson and Spearman correlation
- Inference via cosine similarity or regression output

## 📁 Structure
```
bert-sentence-similarity/
├── data/                  # Input datasets
├── notebooks/             # Jupyter notebook
├── outputs/               # Model checkpoints
├── README.md
└── requirements.txt
```

## 🚀 Usage
```bash
pip install -r requirements.txt
```

## 📄 License
MIT


---

## 🛠️ Inference & Evaluation

### 🔍 Single Pair
```bash
python predict.py
```

### 📦 Batch Inference
```bash
python predict_batch.py
```
- Input: `data/batch_input.json`
- Output: `outputs/batch_output.json`

### 📊 Evaluation
```bash
python evaluate.py
```
- Compares predictions with labels
- Reports Pearson and Spearman correlation

---

## 📦 Example Input
```json
[
  {
    "id": "1",
    "sentence1": "A man is playing a guitar.",
    "sentence2": "Someone is performing music.",
    "label": 0.9
  }
]
```
