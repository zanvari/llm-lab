# Key-Value Extraction with LayoutLM and Donut

This project demonstrates how to preprocess the FUNSD dataset and fine-tune or evaluate layout-aware models like LayoutLM and Donut for extracting key-value pairs from scanned forms.

## Highlights

- ✅ Preprocessing for FUNSD annotations
- ✅ Normalized bounding boxes
- ✅ HuggingFace-compatible `Dataset` class
- ✅ Fine-tuning `LayoutLM` with `Trainer`
- ✅ Donut inference without OCR
- ✅ Inference + bounding box visualization

## Dataset

We use the [FUNSD dataset](https://guillaumejaume.github.io/FUNSD/), a benchmark for form understanding with:
- Text annotations
- Bounding boxes
- Entity labels (key, value, other)
- Links between key-value pairs

## Models

### LayoutLM
- Token classification model that uses layout + text
- Fine-tuned using HuggingFace Trainer

### Donut
- OCR-free vision-to-sequence model
- Trained for document QA
- Inference using prompt: "What are the key fields and their values?"

## Structure

```
kv-extraction-funsd/
├── data_preprocessing.py
├── layoutlm_dataset.py
├── donut_infer.py
├── requirements.txt
├── notebooks/
│   ├── layoutlm_funsd.ipynb
│   └── donut_inference.ipynb
```

## Run

### LayoutLM
```bash
# Train LayoutLM with notebook or script
```

### Donut Inference
```bash
python donut_infer.py --image data/images/0000971160.png
```

## Setup

```bash
pip install -r requirements.txt
```
