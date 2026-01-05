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
        """Return top_k candidate chunks for `query` as a list of dicts with score & metadata.

        This method is defensive: it validates that embeddings are returned and that the
        vector dimensionality matches the index. If something is wrong we raise a
        clear RuntimeError with instructions for remediation (e.g., re-ingest documents).
        """
        vectors = self.embedder.embed_texts([query])
        if not vectors or not vectors[0]:
            raise RuntimeError(
                "Failed to compute embedding for the query. "
                "If you're using the TF-IDF fallback, ensure you have ingested PDFs so the vectorizer is fitted, "
                "or install sentence-transformers / set OPENAI_API_KEY to use another backend."
            )

        qv = vectors[0]

        # validate dimensionality if index exists
        if self.store.index is not None:
            try:
                if getattr(self.store, "use_faiss", False):
                    dim = getattr(self.store.index, "d", None)
                    if dim is None:
                        # some FAISS indexes expose `ntotal` and `d` differently; attempt a safe probe
                        arr = np.array([qv]).astype("float32")
                        # if dimension mismatch faiss will raise when searching; we catch that below
                    else:
                        if len(qv) != dim:
                            raise RuntimeError(
                                f"Query vector dimension ({len(qv)}) does not match FAISS index dimension ({dim}). "
                                "Try re-ingesting documents to rebuild the index."
                            )
                else:
                    # numpy index
                    idx_shape = getattr(self.store.index, "shape", None)
                    if idx_shape and len(qv) != idx_shape[1]:
                        raise RuntimeError(
                            f"Query vector dimension ({len(qv)}) does not match index dimension ({idx_shape[1]}). "
                            "Try re-ingesting documents to rebuild the index."
                        )
            except RuntimeError:
                raise
            except Exception:
                # fall through; we'll attempt the query and surface a clearer error if it fails
                pass

        # perform the query and return structured results
        try:
            results = self.store.query(qv, top_k=top_k)
        except Exception as e:
            raise RuntimeError(f"Failed to query vector store: {e}")

        return [{"score": r[0], "metadata": r[1]} for r in results]
