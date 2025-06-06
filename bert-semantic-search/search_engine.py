
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

class SemanticSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.ids = []

    def load_corpus(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.texts = [item['text'] for item in data]
        self.ids = [item['id'] for item in data]
        return data

    def build_index(self):
        embeddings = self.model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(query_vec, k=top_k)
        results = [(self.texts[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results

if __name__ == "__main__":
    engine = SemanticSearchEngine()
    engine.load_corpus("data/corpus.json")
    engine.build_index()
    query = "How to submit an insurance claim?"
    results = engine.search(query)
    for text, score in results:
        print(f"Score: {score:.4f} | Text: {text}")
