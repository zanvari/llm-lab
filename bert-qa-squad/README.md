# BERT QA Fine-tuning — SQuAD

Fine-tune `bert-base-uncased` on the Stanford Question Answering Dataset (SQuAD v1.1) using HuggingFace Transformers.

---

## Structure
- `notebooks/qa_finetune.ipynb` — Full notebook: training, evaluation, prediction
- `data/` — (Optional) store preprocessed or downloaded data
- `outputs/` — Saved fine-tuned model checkpoints

---

## Goals
- Load and tokenize SQuAD v1.1
- Train BERT for extractive QA
- Evaluate EM/F1 on the validation set
- Run inference on new questions + context

---

## Model
- `bert-base-uncased`
- Architecture: `BertForQuestionAnswering`

---

## Setup
```bash
pip install -r requirements.txt
```

---

## Dependencies
- transformers
- datasets
- torch
- tqdm
- scikit-learn


---

## Notebook
- [qa_finetune.ipynb](./notebooks/qa_finetune.ipynb) – Full training + inference walkthrough  
- [Open in Colab](https://colab.research.google.com/github/zanvari/llm-lab/blob/main/bert-qa-squad/notebooks/qa_finetune.ipynb)

---

## Batch Inference + Evaluation
Run predictions and evaluate performance on a batch of QA inputs:

```bash
python predict_batch.py
```

- Input: `data/batch_input.json` (list of question-context pairs)
- Output: Exact Match (EM), F1, average inference time
