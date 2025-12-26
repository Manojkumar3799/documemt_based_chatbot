from flask import Flask, render_template, request, jsonify
import os
from pdf_reader import extract_text_from_pdf

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

document_text = ""

def load_pdfs_from_folder(folder):
    """Load all PDF files from `folder` and append their extracted text into `document_text`."""
    global document_text
    if not os.path.exists(folder):
        os.makedirs(folder)
    for fname in os.listdir(folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder, fname)
            try:
                text = extract_text_from_pdf(path)
                if text:
                    document_text += "\n\n" + text.strip()
            except Exception as e:
                print(f"Failed to load {path}: {e}")

# Load existing PDFs from the uploads folder at startup
load_pdfs_from_folder(app.config["UPLOAD_FOLDER"])

import re

def find_paragraph(text, question):
    """Return the full paragraph containing `question` (case-insensitive) if found.
    Falls back to returning the matched sentence plus adjacent sentences if paragraphs are not available."""
    if not text or not question:
        return None

    normalized = text.replace("\r\n", "\n")
    # Split into paragraphs by blank lines
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

    for p in paragraphs:
        if question.lower() in p.lower():
            return p

    # Fallback: split into sentences and expand window
    sentences = re.split(r'(?<=[.!?])\s+', normalized)
    for idx, s in enumerate(sentences):
        if question.lower() in s.lower():
            start = max(0, idx - 1)
            end = min(len(sentences), idx + 2)
            return " ".join(sentences[start:end]).strip()

    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global document_text

    if "pdf" not in request.files:
        return jsonify({"message": "No file uploaded"})

    pdf = request.files["pdf"]
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf.filename)
    pdf.save(pdf_path)

    try:
        new_text = extract_text_from_pdf(pdf_path)
        if new_text:
            document_text += "\n\n" + new_text.strip()
    except Exception as e:
        print(f"Failed to extract text from uploaded PDF: {e}")

    return jsonify({"message": "PDF uploaded successfully"})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    if not document_text:
        return jsonify({"answer": "Please upload a PDF first"})

    paragraph = find_paragraph(document_text, question)
    if paragraph:
        return jsonify({"answer": paragraph})

    return jsonify({"answer": "Answer not found in document"})

if __name__ == "__main__":
    app.run(debug=True)
