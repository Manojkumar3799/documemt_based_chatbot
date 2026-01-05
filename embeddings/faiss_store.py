"""faiss_store.py
Utilities for storing embeddings in FAISS and persisting metadata.
"""
import os
import pickle

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


class FaissStore:
    """Vector store that uses FAISS when available and a numpy fallback otherwise.

    This lets the application run on platforms where `faiss` is difficult to install
    (e.g., Windows) while still providing a correct though slightly slower search.
    """

    def __init__(self, index_path="data/vector_store/index.faiss", meta_path="data/vector_store/meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None  # if using FAISS this is a faiss.Index; otherwise a numpy array (n, dim)
        self.metadata = []  # list of metadata dicts for each vector
        self.use_faiss = faiss is not None

        if np is None:
            raise RuntimeError("numpy is required. Run `pip install numpy` or `pip install -r requirements.txt`")

        if not os.path.exists(os.path.dirname(self.index_path)):
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    def build_index(self, vectors, metadatas):
        """Build a new index for the given vectors and metadatas."""
        arr = np.array(vectors).astype("float32")

        if self.use_faiss:
            n, dim = arr.shape
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(arr)
            index.add(arr)
            self.index = index
        else:
            # normalize rows to unit length for cosine similarity
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            arr = arr / norms
            self.index = arr

        self.metadata = list(metadatas)
        self.save()

    def save(self):
        # save index and metadata
        try:
            if self.use_faiss and self.index is not None:
                faiss.write_index(self.index, self.index_path)
            elif self.index is not None:
                # save numpy array
                np.save(self.index_path + ".npy", self.index)
        except Exception as e:
            print("Warning: failed to save index:", e)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        try:
            if self.use_faiss and os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                return True

            # fallback: numpy index saved as .npy
            np_path = self.index_path + ".npy"
            if os.path.exists(np_path) and os.path.exists(self.meta_path):
                self.index = np.load(np_path)
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                return True
        except Exception as e:
            print("Warning: failed to load index:", e)
        return False

    def add(self, vectors, metadatas):
        """Add vectors and metadata to an existing index (or create one if needed)."""
        arr = np.array(vectors).astype("float32")

        if self.use_faiss:
            faiss.normalize_L2(arr)
            if self.index is None:
                _, dim = arr.shape
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(arr)
                self.metadata = list(metadatas)
            else:
                self.index.add(arr)
                self.metadata.extend(metadatas)
        else:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            arr = (arr / norms).astype("float32")
            if self.index is None:
                self.index = arr
                self.metadata = list(metadatas)
            else:
                self.index = np.vstack([self.index, arr])
                self.metadata.extend(metadatas)

        self.save()

    def query(self, vector, top_k=5):
        """Query the index with a single vector (list/array) and return top_k results with metadata.

        Returns a list of tuples (score, metadata)
        """
        if self.index is None:
            raise RuntimeError("Index is not built or loaded")

        x = np.array(vector).astype("float32")

        if self.use_faiss:
            try:
                # FAISS expects shape (n, dim)
                x_batch = np.array([x])
                faiss.normalize_L2(x_batch)
                D, I = self.index.search(x_batch, top_k)
                results = []
                # If FAISS returns unexpected shapes, handle gracefully
                if D.shape[0] == 0 or I.shape[0] == 0:
                    return []
                for score, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    # guard metadata index errors
                    if idx >= len(self.metadata):
                        print(f"Warning: FAISS returned idx {idx} >= metadata length {len(self.metadata)}")
                        continue
                    results.append((float(score), self.metadata[idx]))
                return results
            except Exception as e:
                raise RuntimeError(f"FAISS query failed: {e}")

        # numpy-based search: cosine similarity via dot product of normalized vectors
        try:
            norm = np.linalg.norm(x)
            if norm == 0.0:
                norm = 1.0
            x_norm = (x / norm).astype("float32")

            # ensure index is a 2D array
            if self.index.ndim != 2:
                raise RuntimeError("Internal numpy index has invalid shape")

            if self.index.shape[1] != x_norm.shape[0]:
                raise RuntimeError(
                    f"Query vector dimension ({x_norm.shape[0]}) does not match index dimension ({self.index.shape[1]})."
                )

            scores = (self.index @ x_norm)
            idxs = np.argsort(-scores)[:top_k]
            results = []
            for i in idxs:
                if i >= len(self.metadata):
                    print(f"Warning: numpy index returned idx {i} >= metadata length {len(self.metadata)}")
                    continue
                results.append((float(scores[i]), self.metadata[i]))
            return results
        except Exception as e:
            raise RuntimeError(f"Numpy-based query failed: {e}")
