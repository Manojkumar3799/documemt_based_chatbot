"""chat_loop.py
Orchestrate: receive a question, retrieve relevant chunks, call generator, update chat history.
"""
from typing import Tuple
from embeddings.embedder import Embedder
from embeddings.faiss_store import FaissStore
from retriever.retriever import Retriever
from generator.rag_generator import build_prompt, generate_answer
from chat.chat_history import ChatHistory


class ChatLoop:
    def __init__(self):
        self.embedder = Embedder()
        self.store = FaissStore()
        # ensure store is loaded
        self.store.load()
        self.retriever = Retriever(self.embedder, self.store)
        self.history = ChatHistory()

    def handle(self, session_id: str, question: str, top_k: int = 5) -> Tuple[str, list]:
        """Process the user question and return (answer, sources).

        sources is a list of metadata dicts for each retrieved chunk.
        This method wraps steps in a try/except to give clearer errors back to the API
        and to ensure chat history is only updated after a successful answer generation.
        """
        try:
            # ensure session history exists
            self.history.load(session_id)
            # add user message
            self.history.add_user_message(session_id, question)

            # retrieve relevant chunks
            candidates = self.retriever.retrieve(question, top_k=top_k)

            if not candidates:
                return (
                    "I couldn't find relevant content in the indexed documents. "
                    "Please upload or re-ingest your PDFs and try again.",
                    [],
                )

            # prepare list of chunks for prompt
            retrieved = [c for c in candidates]

            # build messages
            messages = build_prompt(retrieved, self.history.get(session_id), question)

            # call generator (openai chat or fallback)
            answer = generate_answer(messages)

            # add assistant message to history
            self.history.add_ai_message(session_id, answer)

            # return answer and list of sources
            sources = [c["metadata"] for c in candidates]
            return answer, sources

        except Exception as e:
            # wrap and re-raise with context so the Flask handler prints a helpful message
            raise RuntimeError(f"Chat handling failed: {e}") from e
