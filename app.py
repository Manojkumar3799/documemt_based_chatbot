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
store.load()
chat_loop = ChatLoop()


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

    # prepare texts and metadata
    texts = [c["text"] for c in chunks]
    metadatas = [
        {"source": os.path.basename(saved_path), "chunk_id": c["id"], "text": c["text"]}
        for c in chunks
    ]

    if texts:
        vectors = embedder.embed_texts(texts)
        # add to FAISS store
        try:
            store.add(vectors, metadatas)
        except Exception:
            # if index empty or incompatible, build a new one
            store.build_index(vectors, metadatas)

    return jsonify({"message": "PDF uploaded and indexed"})


@app.route("/ingest_all", methods=["POST"])
def ingest_all():
    """(Developer helper) Ingest all PDFs found under data/pdfs/ into vector store.

    This endpoint processes files in data/pdfs, extracts text, chunks, and builds a fresh index.
    """
    root = app.config["DATA_PDFS"]
    all_texts = []
    all_meta = []
    for fname in os.listdir(root):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(root, fname)
        text = extract_text_from_pdf(path)
        chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)
        for c in chunks:
            all_texts.append(c["text"])
            all_meta.append({"source": fname, "chunk_id": c["id"], "text": c["text"]})

    if not all_texts:
        return jsonify({"message": "No PDFs found"})

    vectors = embedder.embed_texts(all_texts)
    store.build_index(vectors, all_meta)
    return jsonify({"message": "All PDFs ingested and indexed"})


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

    answer, sources = chat_loop.handle(session_id, question, top_k=5)

    return jsonify({"answer": answer, "sources": sources, "session_id": session_id})


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
