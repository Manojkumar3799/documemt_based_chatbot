"""summarizer.py
Utilities to summarize text chunks at ingestion time.

If an OpenAI API key is available, use ChatCompletion to produce a concise paraphrased summary.
Otherwise, fall back to a lightweight extractive summarizer (first N sentences).
"""
import os
import re

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def _split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def summarize_text(text: str, max_sentences: int = 2) -> str:
    """Return a short summary for `text` (up to `max_sentences`).

    Uses OpenAI ChatCompletion when `OPENAI_API_KEY` is set; otherwise returns
    the first `max_sentences` sentences as a conservative fallback.
    """
    if not text:
        return ""

    # fast heuristic fallback (used if OpenAI is not available or call fails)
    def _fallback():
        s = _split_sentences(text)
        if not s:
            return (text.strip()[: max(128, max_sentences * 100)])
        return " ".join(s[:max_sentences])

    if not OPENAI_API_KEY:
        return _fallback()

    try:
        import openai

        openai.api_key = OPENAI_API_KEY
        system = {
            "role": "system",
            "content": (
                "You are a helpful summarizer. Create a concise paraphrased summary of the provided text. "
                f"Keep it to at most {max_sentences} short sentences, avoid quoting long passages verbatim, "
                "and produce a clear, standalone summary suitable for feeding into a knowledge retrieval prompt."
            ),
        }
        user = {"role": "user", "content": text}
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[system, user],
            max_tokens=150,
            temperature=0.0,
        )
        summary = resp.choices[0].message.content.strip()
        # sanitize and keep only the first max_sentences if model returns more
        sents = _split_sentences(summary)
        if not sents:
            return summary
        return " ".join(sents[:max_sentences])
    except Exception:
        return _fallback()
