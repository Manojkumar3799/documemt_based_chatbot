"""chat_history.py
Simple in-memory chat history manager keyed by session_id.
Designed to be readable for people with no NLP background.
"""
import os
import json
from typing import List

# Optionally persist chat history to disk under data/chat_history/
HISTORY_DIR = "data/chat_history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR, exist_ok=True)


class ChatHistory:
    def __init__(self):
        # map session_id -> list of turns (each turn: {role, content})
        self.sessions = {}

    def _path(self, session_id: str):
        return os.path.join(HISTORY_DIR, f"{session_id}.json")

    def load(self, session_id: str):
        """Load session history from disk if available"""
        path = self._path(session_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.sessions[session_id] = json.load(f)
        else:
            self.sessions.setdefault(session_id, [])

    def save(self, session_id: str):
        path = self._path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.sessions.get(session_id, []), f, ensure_ascii=False, indent=2)

    def get(self, session_id: str):
        self.sessions.setdefault(session_id, [])
        return self.sessions[session_id]

    def add_user_message(self, session_id: str, content: str):
        self.sessions.setdefault(session_id, [])
        self.sessions[session_id].append({"role": "user", "content": content})
        self.save(session_id)

    def add_ai_message(self, session_id: str, content: str):
        self.sessions.setdefault(session_id, [])
        self.sessions[session_id].append({"role": "assistant", "content": content})
        self.save(session_id)

    def clear(self, session_id: str):
        self.sessions[session_id] = []
        self.save(session_id)
