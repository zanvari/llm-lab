## Notebooks

### `fine_tune_agnews.ipynb` – Fine-Tuning DistilBERT on AG News

This tutorial-style notebook demonstrates how to fine-tune a pretrained `distilbert-base-uncased` model on the AG News dataset using Hugging Face's `Trainer` API.

**Key Highlights:**
- Loads AG News dataset from Kaggle
- Merges and tokenizes news title + description
- Trains DistilBERT using Hugging Face's `Trainer`
- Evaluates with `classification_report` and confusion matrix

Path: `notebooks/fine_tune_agnews.ipynb`

Model: `DistilBERT` (4 classes: World, Sports, Business, Sci/Tech)  
Dataset: [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  

