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
            "You are an assistant that answers questions using ONLY the provided document excerpts. "
            "Do NOT hallucinate or invent facts. If the answer is not contained in the excerpts, say you don't know. "
            "Be concise and conversational. Provide short citations in square brackets like [source]."
        ),
    }

    # Attach retrieved document chunks as a single context message
    context_texts = []
    for i, r in enumerate(retrieved_chunks):
        meta = r.get("metadata", {})
        txt = meta.get("text", "")
        src = meta.get("source", "unknown")
        context_texts.append(f"[{src}] {txt}")

    context_msg = {
        "role": "system",
        "content": "DOCUMENT EXCERPTS:\n\n" + "\n\n".join(context_texts),
    }

    messages = [system, context_msg]

    # add last N turns of chat history
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # finally add the new user question
    messages.append({"role": "user", "content": question})
    return messages


def generate_answer(messages: List[dict], model: str = "gpt-3.5-turbo", max_tokens: int = 512, temperature: float = 0.0):
    """Call OpenAI ChatCompletion with prepared messages and return assistant text.

    Keep temperature low (deterministic) so answers stick to the documents.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it to use the RAG generator.")

    import openai
    openai.api_key = OPENAI_API_KEY

    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
