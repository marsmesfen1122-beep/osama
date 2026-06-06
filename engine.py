import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

# Assuming df, embeddings, and index are globally available or passed
# For the purpose of this file, we'll assume df is loaded internally or passed to AIEngine

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

class AIEngine:
    def __init__(self, df):
        self.df = df
        self.index = None
        self.embeddings = None
        self.build_index()

    def build_index(self):
        texts = (self.df["title"] + " " + self.df["authors"].fillna("")).tolist()
        self.embeddings = model.encode(texts)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings).astype("float32"))

    def search(self, query, top_k=5):
        q_vec = model.encode([query]).astype("float32")
        distances, idx = self.index.search(q_vec, top_k)

        results = self.df.iloc[idx[0]].copy()
        results["score"] = distances[0]

        results["rank"] = 1 / (results["score"] + 1e-5)
        return results.sort_values("rank", ascending=False)

# The original search_books function (from _NTUeDkOC_TJ) assumes `model` and `index` are globally available.
# For a standalone engine.py, it should either be a method of AIEngine or take AIEngine as input.
# For simplicity, we'll redefine search_books to use an AIEngine instance.

def search_books(ai_engine_instance, query, top_k=5):
    return ai_engine_instance.search(query, top_k)
