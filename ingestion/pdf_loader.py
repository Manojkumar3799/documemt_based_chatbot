"""pdf_loader.py
Utilities to load PDFs and extract text in a clean way.
"""
from PyPDF2 import PdfReader
import os


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF and return cleaned text.

    Returns a string with page breaks turned into blank lines for paragraph separation.
    """
    reader = PdfReader(pdf_path)
    text = []

    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            # Keep a small amount of structure: separate pages with blank lines
            text.append(page_text.strip())

    return "\n\n".join(text).strip()


def save_uploaded_pdf(uploaded_file, dest_folder):
    """Save a Werkzeug FileStorage (Flask upload) into dest_folder and return path."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    fname = uploaded_file.filename
    dest_path = os.path.join(dest_folder, fname)
    uploaded_file.save(dest_path)
    return dest_path
