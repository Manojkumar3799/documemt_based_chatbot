"""Main Flask app for RAG-based document chatbot.

Endpoints:
- GET / : serve the UI
- POST /upload : upload a PDF; it will be saved and indexed into the vector store
- POST /chat : send a question and session_id (optional) → returns an answer grounded in documents
- POST /ingest_all : (dev) ingest all PDFs from data/pdfs into vector store
- POST /clear_session : clear a session's chat history
"""
from flask import Flask, render_template, request, jsonify
import os
import uuid

from ingestion.pdf_loader import extract_text_from_pdf, save_uploaded_pdf
from ingestion.text_splitter import split_text_into_chunks
from ingestion.summarizer import summarize_text
from embeddings.embedder import Embedder
from embeddings.faiss_store import FaissStore
from chat.chat_loop import ChatLoop

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["DATA_PDFS"] = "data/pdfs"

# initialize components
embedder = Embedder()
store = FaissStore()
# try to load existing index (if present)
loaded = store.load()
chat_loop = ChatLoop()

# If we are using TF-IDF fallback and there is no index loaded, try to ingest PDFs automatically
if getattr(embedder, "mode", None) == "tfidf" and (not loaded or store.index is None):
    print("No vector index found and TF-IDF embedding backend is active — attempting to ingest PDFs under data/pdfs/...")
    ok, msg = ingest_all_pdfs(app.config["DATA_PDFS"])
    if ok:
        print("Auto-ingest completed:", msg)
    else:
        print("Auto-ingest did not run:", msg)


@app.route("/")
def index():
    # use the existing template; it will make chat requests to /chat
    return render_template("Index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Save uploaded PDF and process it immediately: extract text, split into chunks, embed, and add to FAISS."""
    if "pdf" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    pdf = request.files["pdf"]
    saved_path = save_uploaded_pdf(pdf, app.config["DATA_PDFS"])

    # extract text
    text = extract_text_from_pdf(saved_path)
    # split into chunks
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)

    # prepare texts and metadata (include a short summary for each chunk)
    texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        summary = summarize_text(c["text"])
        metadatas.append({
            "source": os.path.basename(saved_path),
            "chunk_id": c["id"],
            "text": c["text"],
            "summary": summary,
        })

    if texts:
        vectors = embedder.embed_texts(texts)
        # add to FAISS store
        try:
            store.add(vectors, metadatas)
        except Exception:
            # if index empty or incompatible, build a new one
            store.build_index(vectors, metadatas)

    return jsonify({"message": "PDF uploaded and indexed"})


def ingest_all_pdfs(root: str) -> (bool, str):
    """Ingest all PDFs in `root` into the vector store.

    Returns (success, message)
    """
    all_texts = []
    all_meta = []
    if not os.path.exists(root):
        return False, "PDFs folder not found"

    for fname in os.listdir(root):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(root, fname)
        try:
            text = extract_text_from_pdf(path)
        except Exception as e:
            print(f"Warning: failed to extract {path}: {e}")
            continue
        chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)
        for c in chunks:
            all_texts.append(c["text"])
            # compute a short summary for the chunk (fast fallback used if OpenAI not configured)
            summary = summarize_text(c["text"])
            all_meta.append({"source": fname, "chunk_id": c["id"], "text": c["text"], "summary": summary})

    if not all_texts:
        return False, "No PDFs found"

    vectors = embedder.embed_texts(all_texts)
    # build a fresh index
    try:
        store.build_index(vectors, all_meta)
    except Exception as e:
        return False, f"Failed to build index: {e}"

    return True, f"Ingested {len(all_texts)} chunks from {len(set([m['source'] for m in all_meta]))} PDFs"


@app.route("/ingest_all", methods=["POST"])
def ingest_all():
    success, message = ingest_all_pdfs(app.config["DATA_PDFS"])
    if success:
        return jsonify({"message": message})
    return jsonify({"message": message}), 400


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint used by the UI.

    Request JSON: {"question": str, "session_id": str (optional)}
    Response JSON: {"answer": str, "sources": [{...}], "session_id": str}
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())

    try:
        answer, sources = chat_loop.handle(session_id, question, top_k=5)
        return jsonify({"answer": answer, "sources": sources, "session_id": session_id})
    except Exception as e:
        # log the full traceback to server stdout for debugging
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e) or "Internal server error"}), 500


@app.route("/clear_session", methods=["POST"])
def clear_session():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    chat_loop.history.clear(session_id)
    return jsonify({"message": "session cleared"})


if __name__ == "__main__":
    app.run(debug=True)
