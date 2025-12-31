"""retriever.py
Simple retriever that uses the Embedder + FaissStore to return the top-k relevant chunks.
"""
from embeddings.embedder import Embedder
from embeddings.faiss_store import FaissStore


class Retriever:
    def __init__(self, embedder: Embedder = None, store: FaissStore = None):
        self.embedder = embedder or Embedder()
        self.store = store or FaissStore()
        self.store.load()

    def retrieve(self, query: str, top_k: int = 5):
        """Return top_k candidate chunks for `query` as a list of dicts with score & metadata."""
        vectors = self.embedder.embed_texts([query])
        qv = vectors[0]
        results = self.store.query(qv, top_k=top_k)
        # results are (score, metadata)
        return [{"score": r[0], "metadata": r[1]} for r in results]
