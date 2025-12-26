from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Add a blank line between pages so we get paragraph separation
            text += page_text + "\n\n"

    return text.strip()
