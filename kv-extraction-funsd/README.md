# 🧾 Key-Value Extraction — FUNSD

This project focuses on extracting key-value pairs from scanned forms using layout-aware models like **LayoutLM** and **Donut**.

## 📄 Dataset

We use the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset, a benchmark for form understanding with:
- Text annotations
- Bounding boxes
- Entities (key, value, other)
- Links between keys and their corresponding values

## 🤖 Models

- 📐 **LayoutLMv2** (transformers-based)
- 🧠 **Donut** (Vision Transformer-based, no OCR needed)

## 🛠️ Tasks

- Preprocess FUNSD annotations
- Visualize layout structure
- Train/test LayoutLM and Donut
- Evaluate precision, recall, F1 on key-value pairs

## 📁 Structure

- `data/` — FUNSD raw + processed files
- `models/` — Saved checkpoints
- `outputs/` — Inference outputs
- `notebooks/` — Analysis + visualization

## ✅ Goals

- Compare layout-aware models for form understanding
- Build a portable and reusable pipeline for KV extraction

