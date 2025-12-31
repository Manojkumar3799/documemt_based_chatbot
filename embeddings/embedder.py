"""embedder.py
Wrap embedding generation. Priority order:
1) OpenAI embeddings (if OPENAI_API_KEY set)
2) sentence-transformers (if installed)
3) TF-IDF vectorizer (scikit-learn) as a fallback - persisted to disk

This keeps the project runnable even on machines without OpenAI or sentence-transformers.
"""
import os
import pickle
from typing import List, Optional

import numpy as np

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TFIDF_PATH = "data/vector_store/tfidf.pkl"


class Embedder:
    def __init__(self, openai_client=None):
        """Initialize embedder and choose available backend.

        If TF-IDF is used, the vectorizer is persisted to `TFIDF_PATH` so queries work after restart.
        """
        self.openai_client = openai_client
        self.mode = None  # 'openai' | 'sbert' | 'tfidf'

        # prefer OpenAI if key is present
        if OPENAI_API_KEY:
            self.mode = "openai"
            return

        # try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.mode = "sbert"
            return
        except Exception:
            pass

        # try scikit-learn TF-IDF fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.TfidfVectorizer = TfidfVectorizer
            self.vectorizer = None  # loaded or fitted vectorizer
            # attempt to load persisted vectorizer
            self._load_vectorizer()
            self.mode = "tfidf"
            return
        except Exception:
            pass

        # if none available, raise an informative error
        raise RuntimeError(
            "No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers or scikit-learn."
        )

    # TF-IDF persistence helpers
    def _save_vectorizer(self):
        try:
            with open(TFIDF_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)
        except Exception as e:
            print("Warning: failed to save TF-IDF vectorizer:", e)

    def _load_vectorizer(self):
        if os.path.exists(TFIDF_PATH):
            try:
                with open(TFIDF_PATH, "rb") as f:
                    self.vectorizer = pickle.load(f)
            except Exception as e:
                print("Warning: failed to load TF-IDF vectorizer:", e)
                self.vectorizer = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts.

        - For OpenAI: calls the embeddings API
        - For sentence-transformers: uses the local model
        - For TF-IDF: fits the vectorizer on supplied texts if needed when indexing; for queries
          it requires a previously fitted vectorizer (i.e., ingest documents first)
        """
        if not texts:
            return []

        if self.mode == "openai":
            import openai

            openai.api_key = OPENAI_API_KEY
            model_name = "text-embedding-3-small"
            vectors = []
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                resp = openai.Embedding.create(input=batch, model=model_name)
                for r in resp.data:
                    vectors.append(r.embedding)
            return vectors

        if self.mode == "sbert":
            embs = self.model.encode(texts, show_progress_bar=False)
            return [emb.tolist() for emb in embs]

        # TF-IDF fallback
        if self.mode == "tfidf":
            # if vectorizer missing and we have >1 text, assume we're indexing and fit
            if self.vectorizer is None:
                if len(texts) == 1:
                    raise RuntimeError(
                        "TF-IDF vectorizer not fitted. Please ingest documents first or install sentence-transformers / set OPENAI_API_KEY."
                    )
                self.vectorizer = self.TfidfVectorizer(ngram_range=(1, 2), max_features=4096)
                X = self.vectorizer.fit_transform(texts)
                self._save_vectorizer()
                # convert to dense arrays
                arrs = X.toarray()
                # normalize rows
                norms = np.linalg.norm(arrs, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                arrs = (arrs / norms).astype(float)
                return [arr.tolist() for arr in arrs]

            # vectorizer exists -> transform
            X = self.vectorizer.transform(texts)
            arrs = X.toarray()
            norms = np.linalg.norm(arrs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            arrs = (arrs / norms).astype(float)
            return [arr.tolist() for arr in arrs]

        raise RuntimeError("No embedding backend available")
