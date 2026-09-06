# BERT Document Clustering

Cluster documents using Sentence-BERT embeddings and visualize them using UMAP.

## Dataset
A small set of 10 example documents (`data/documents.json`) covering topics from healthcare to tech and finance.

## Features
- Sentence-BERT embeddings (`all-MiniLM-L6-v2`)
- KMeans and Agglomerative clustering
- UMAP visualization in 2D
- Silhouette score for cluster quality

## Notebook
- [`clustering.ipynb`](./clustering.ipynb): Full clustering walkthrough

## Structure
```
bert-doc-clustering/
├── data/
│   └── documents.json         # Sample input docs
├── outputs/                   # Visualizations
├── clustering.ipynb           # Main notebook
├── README.md
└── requirements.txt
```

## Requirements
See `requirements.txt` for dependencies.

## Future Ideas
- Plug in custom corpora
- Automatic topic labeling of clusters
- Export cluster groupings

