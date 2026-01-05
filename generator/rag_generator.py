"""rag_generator.py
Combine retrieved document chunks + chat history + user question and ask an LLM to produce a conversational answer.
We use OpenAI ChatCompletion by default. Make sure `OPENAI_API_KEY` is set in environment.
"""
import os
from typing import List

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def build_prompt(retrieved_chunks: List[dict], chat_history: List[dict], question: str) -> List[dict]:
    """Return messages in OpenAI Chat format (list of dicts) for ChatCompletion.

    - retrieved_chunks: list of metadata dicts containing 'text' and 'source'
    - chat_history: list of dicts {role: 'user'|'assistant', 'content': str}
    - question: the new user question
    """
    system = {
        "role": "system",
        "content": (
            "You are an assistant that answers questions using ONLY the provided document excerpts for reference. "
            "Do NOT copy or paste the excerpts verbatim — instead, synthesize and paraphrase the information into a concise, conversational answer. "
            "If you need to quote, keep quotes very short (under 100 characters) and include a short citation like [source]. "
            "If the answer is not contained in the excerpts, say you don't know. Be concise and helpful."
        ),
    }

    # Attach retrieved document chunks as a single context message. Prefer short summaries when available.
    context_texts = []
    for i, r in enumerate(retrieved_chunks):
        meta = r.get("metadata", {})
        src = meta.get("source", "unknown")
        summary = meta.get("summary")
        txt = summary if summary else meta.get("text", "")
        # indicate whether this is a summary to make intent explicit to the LLM
        label = "SUMMARY" if summary else "EXCERPT"
        context_texts.append(f"[{src}] ({label}) {txt}")

    context_msg = {
        "role": "system",
        "content": "DOCUMENT EXCERPTS (summaries preferred):\n\n" + "\n\n".join(context_texts),
    }

    messages = [system, context_msg]

    # add last N turns of chat history
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # finally add the new user question
    messages.append({"role": "user", "content": question})
    return messages


def _fallback_generate_answer(retrieved_chunks: List[dict], chat_history: List[dict], question: str, max_sentences: int = 3) -> str:
    """Simple extractive fallback when an LLM is not available.

    This selects the top sentences from the retrieved chunks by simple token overlap with the question.
    It is intentionally conservative and clearly labelled so users understand this is a fallback.
    """
    import re

    def split_sentences(text: str):
        # crude sentence splitter (keeps it dependency-free)
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if s.strip()]

    q_tokens = set([w.lower() for w in re.findall(r"\w+", question) if len(w) > 2])
    if not q_tokens:
        q_tokens = set()

    scored = []  # list of (score, sentence, source)
    for r in retrieved_chunks:
        meta = r.get("metadata", {})
        text = meta.get("text", "")
        source = meta.get("source", "unknown")
        for sent in split_sentences(text):
            tokens = set([w.lower() for w in re.findall(r"\w+", sent)])
            score = len(tokens & q_tokens)
            # small boost if exact question tokens appear
            if question.lower().strip() in sent.lower():
                score += 2
            # record
            scored.append((score, sent, source))

    # pick top sentences
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    top = [s for sc, s, src in scored if sc > 0][:max_sentences]

    if not top:
        # if nothing matched, return a neutral reply pointing to sources
        sources = sorted(set([r.get("metadata", {}).get("source", "unknown") for r in retrieved_chunks]))
        return (
            "I couldn't find a specific sentence that answers your question exactly, "
            "but I found relevant sections in the documents: " + ", ".join(sources)
        )

    answer = " ".join(top)
    return f"(Extractive fallback) {answer}"


def generate_answer(messages: List[dict], model: str = "gpt-3.5-turbo", max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Call OpenAI ChatCompletion with prepared messages and return assistant text.

    If OPENAI_API_KEY is not configured, a conservative extractive fallback is used.
    """
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY

            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # if OpenAI call fails, log and fall back
            print("Warning: OpenAI call failed, using fallback generator:", e)

    # Build a lightweight fallback answer using the retrieved chunks embedded in the messages
    # Extract retrieved chunk metadata from messages (we put them in the second system message)
    retrieved_chunks = []
    if len(messages) >= 2 and messages[1]["role"] == "system" and messages[1]["content"].startswith("DOCUMENT EXCERPTS"):
        # parse the DOCUMENT EXCERPTS block we created in build_prompt
        content = messages[1]["content"].replace("DOCUMENT EXCERPTS:\n\n", "")
        # split by blank lines into items like "[source] text"
        entries = [e.strip() for e in content.split("\n\n") if e.strip()]
        for e in entries:
            if e.startswith("["):
                try:
                    src_end = e.index("]")
                    src = e[1:src_end]
                    txt = e[src_end + 1 :].strip()
                except Exception:
                    src = "unknown"
                    txt = e
                retrieved_chunks.append({"metadata": {"source": src, "text": txt}})

    # get the last user message (question)
    question = ""
    for m in reversed(messages):
        if m["role"] == "user":
            question = m["content"]
            break

    return _fallback_generate_answer(retrieved_chunks, [], question)
