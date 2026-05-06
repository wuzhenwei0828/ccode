from typing import Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

from config import get_settings
from config.settings import MemoryConfig
from models.chat_message import ChatMessage
from utils.db import SessionLocal


class MemoryService:
    """Dual-layer memory: short-term sliding window + long-term MySQL persistence."""

    def __init__(self):
        self._config: Optional[MemoryConfig] = None
        # In-memory cache: session_id -> list of recent messages
        self._short_term: dict[str, list[BaseMessage]] = {}

    @property
    def config(self) -> MemoryConfig:
        if self._config is None:
            self._config = get_settings().get_memory_config()
        return self._config

    def get_context(self, session_id: str) -> list[BaseMessage]:
        """Get messages for LLM context (short-term sliding window)."""
        messages = self._short_term.get(session_id, [])
        window = self.config.short_term_window
        return messages[-window:] if len(messages) > window else messages

    def add_message(self, session_id: str, role: str, content: str, db: Optional[Session] = None):
        """Add a message to both short-term cache and long-term storage."""
        # Short-term
        if session_id not in self._short_term:
            self._short_term[session_id] = []
        msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        self._short_term[session_id].append(msg)

        # Long-term (MySQL)
        should_close = db is None
        if db is None:
            db = SessionLocal()
        try:
            # Get current max sequence
            max_seq = (
                db.query(ChatMessage.sequence)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.sequence.desc())
                .first()
            )
            next_seq = (max_seq[0] + 1) if max_seq else 0

            chat_msg = ChatMessage(
                id=str(uuid4()),
                session_id=session_id,
                role=role,
                content=content,
                sequence=next_seq,
            )
            db.add(chat_msg)
            db.commit()
        finally:
            if should_close:
                db.close()

    def load_history(self, session_id: str, limit: Optional[int] = None) -> list[BaseMessage]:
        """Load full history from MySQL for a session."""
        db = SessionLocal()
        try:
            query = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.sequence.asc())
            )
            if limit:
                query = query.limit(limit)
            messages = query.all()
            result = []
            for m in messages:
                if m.role == "user":
                    result.append(HumanMessage(content=m.content))
                else:
                    result.append(AIMessage(content=m.content))
            return result
        finally:
            db.close()

    def clear_session(self, session_id: str):
        """Clear all memory for a session."""
        self._short_term.pop(session_id, None)
        db = SessionLocal()
        try:
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            db.commit()
        finally:
            db.close()


# Singleton
memory_service = MemoryService()
