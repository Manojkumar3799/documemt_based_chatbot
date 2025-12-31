"""text_splitter.py
Simple text splitter to break long document text into overlapping chunks.
This is intentionally simple so developers with no NLP background can read and modify it.
"""
import math


def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split `text` into chunks of approximately `chunk_size` characters with `overlap` characters overlap.

    Returns a list of dicts: {"id": int, "text": str, "start": int, "end": int}
    """
    if not text:
        return []

    text = text.strip()
    n = len(text)
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    idx = 0
    counter = 0
    while idx < n:
        start = idx
        end = min(idx + chunk_size, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": counter,
                "text": chunk_text,
                "start": start,
                "end": end,
            })
            counter += 1
        idx += step

    return chunks
