from flask import Flask, render_template, request, jsonify
import os
from pdf_reader import extract_text_from_pdf

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

document_text = ""

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

    document_text = extract_text_from_pdf(pdf_path)

    return jsonify({"message": "PDF uploaded successfully"})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    if not document_text:
        return jsonify({"answer": "Please upload a PDF first"})

    # Simple keyword matching (basic logic)
    for sentence in document_text.split("."):
        if question.lower() in sentence.lower():
            return jsonify({"answer": sentence})

    return jsonify({"answer": "Answer not found in document"})

if __name__ == "__main__":
    app.run(debug=True)
