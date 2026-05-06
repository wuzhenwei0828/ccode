import logging
import threading
from typing import Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from config import get_settings
from config.settings import MemoryConfig
from models.chat_message import ChatMessage
from models.chat_session import ChatSession
from services.llm_factory import LLMFactory
from utils.db import SessionLocal

logger = logging.getLogger(__name__)


class MemoryService:
    """Three-layer memory:
    - Long-term: MySQL chat_messages + chat_sessions.summary
    - Short-term active buffer: S1 stores the current stable context window
    - Short-term overflow buffer: S2 stores new messages while background compression is running
    """

    def __init__(self):
        self._config: Optional[MemoryConfig] = None
        self._buffers_s1: dict[str, list[BaseMessage]] = {}
        self._buffers_s2: dict[str, list[BaseMessage]] = {}
        self._summary_cache: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._summary_in_progress: set[str] = set()
        self._compressing: set[str] = set()

    def _get_lock(self, session_id: str) -> threading.Lock:
        with self._locks_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]

    @property
    def config(self) -> MemoryConfig:
        if self._config is None:
            self._config = get_settings().get_memory_config()
        return self._config

    def get_context_parts(self, session_id: str) -> tuple[str, list[BaseMessage]]:
        """Return summary text and history messages separately for prompt composition."""
        lock = self._get_lock(session_id)
        with lock:
            s1 = self._buffers_s1.get(session_id, [])
            s2 = self._buffers_s2.get(session_id, [])
            summary = self._summary_cache.get(session_id, "")
            if s1 or s2:
                return summary, [*s1, *s2]
        return self._recover_context_parts(session_id)

    def _recover_context_parts(self, session_id: str) -> tuple[str, list[BaseMessage]]:
        """Recover summary and history separately from MySQL when in-memory cache is empty."""
        db = SessionLocal()
        try:
            session = ChatSession.get_by_id(db, session_id)
            summary = ""
            summary_sequence = 0
            if session and session.summary:
                summary = session.summary
                summary_sequence = session.summary_sequence

            msgs = ChatMessage.get_after_sequence(db, session_id, summary_sequence)
            converted = []
            for m in msgs:
                if m.role == "user":
                    converted.append(HumanMessage(content=m.content))
                else:
                    converted.append(AIMessage(content=m.content))

            half = max(1, self.config.short_term_window // 2)
            cached = converted if len(converted) <= half else converted[-half:]

            lock = self._get_lock(session_id)
            with lock:
                self._buffers_s1[session_id] = cached
                self._buffers_s2[session_id] = []
                if summary:
                    self._summary_cache[session_id] = summary

            if len(converted) > half:
                self._start_compression(session_id)

            return summary, cached
        finally:
            db.close()

    def get_context(self, session_id: str) -> list[BaseMessage]:
        """Compatibility wrapper for legacy callers.

        New code should prefer get_context_parts(), which returns summary text and
        history messages separately for prompt composition.
        """
        summary, history = self.get_context_parts(session_id)
        if summary:
            return [SystemMessage(content=f"以下是之前对话的摘要：{summary}")] + history
        return history

    def add_message(self, session_id: str, role: str, content: str, db: Optional[Session] = None):
        """Add a message to both short-term cache and long-term storage."""
        msg_obj = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        half = max(1, self.config.short_term_window // 2)
        should_start_compression = False

        lock = self._get_lock(session_id)
        with lock:
            s1 = self._buffers_s1.setdefault(session_id, [])
            if session_id in self._compressing or len(s1) >= half:
                self._buffers_s2.setdefault(session_id, []).append(msg_obj)
            else:
                s1.append(msg_obj)
                if len(s1) >= half:
                    should_start_compression = True

        should_close = db is None
        if db is None:
            db = SessionLocal()
        try:
            max_seq = ChatMessage.get_max_sequence(db, session_id)
            next_seq = (max_seq[0] + 1) if max_seq else 0

            chat_msg = ChatMessage(
                id=str(uuid4()),
                session_id=session_id,
                role=role,
                content=content,
                sequence=next_seq,
            )
            db.add(chat_msg)
            db.flush()

            session = ChatSession.get_by_id(db, session_id)
            if session:
                session.message_count = next_seq + 1
                db.flush()

            db.commit()
        except Exception:
            with lock:
                s2 = self._buffers_s2.get(session_id, [])
                if s2 and s2[-1] is msg_obj:
                    s2.pop()
                else:
                    s1 = self._buffers_s1.get(session_id, [])
                    if s1 and s1[-1] is msg_obj:
                        s1.pop()
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()

        if should_start_compression:
            self._start_compression(session_id)

    def _start_compression(self, session_id: str):
        """Mark a session as compressing and schedule background compression once."""
        with self._locks_lock:
            if session_id in self._summary_in_progress:
                return
            self._summary_in_progress.add(session_id)
            self._compressing.add(session_id)

        def _run_compression():
            bg_db = SessionLocal()
            try:
                bg_session = ChatSession.get_by_id(bg_db, session_id)
                if not bg_session:
                    return
                self._generate_summary(bg_db, session_id, bg_session)
                self._on_compression_complete(session_id)
                bg_db.commit()
            except Exception as e:
                logger.error("compression failed for %s: %s", session_id, e)
            finally:
                bg_db.close()
                with self._locks_lock:
                    self._summary_in_progress.discard(session_id)

        threading.Thread(target=_run_compression, daemon=True).start()

    def _on_compression_complete(self, session_id: str):
        """Move S2 into S1 after compression completes and clear S2."""
        lock = self._get_lock(session_id)
        with lock:
            self._buffers_s1[session_id] = list(self._buffers_s2.get(session_id, []))
            self._buffers_s2[session_id] = []
            self._compressing.discard(session_id)

    def _generate_summary(self, db: Session, session_id: str, session: ChatSession):
        """Generate conversation summary using LLM and persist it."""
        try:
            llm = LLMFactory.create()
            old_summary = session.summary
            window = self.config.short_term_window

            if old_summary:
                new_msgs = ChatMessage.get_after_sequence(db, session_id, session.summary_sequence)
                new_text = "\n".join(f"[{m.role}] {m.content}" for m in new_msgs)
                prompt = (
                    f"以下是之前对话的摘要：\n{old_summary}\n\n"
                    f"以下是新的对话内容：\n{new_text}\n\n"
                    f"请生成一段新的摘要，整合之前的摘要和新内容。"
                    f"摘要要简短精炼，只保留关键信息。"
                    f"控制在 {self.config.summary_max_length} 字以内。"
                )
                latest_seq = new_msgs[-1].sequence if new_msgs else session.summary_sequence
            else:
                msgs = ChatMessage.get_recent(db, session_id, window)
                msgs = list(reversed(msgs))
                all_text = "\n".join(f"[{m.role}] {m.content}" for m in msgs)
                prompt = (
                    f"请总结以下对话的主要内容：\n{all_text}\n\n"
                    f"摘要要简短精炼，只保留关键信息。"
                    f"控制在 {self.config.summary_max_length} 字以内。"
                )
                latest_seq = msgs[-1].sequence if msgs else 0

            summary_resp = llm.invoke(prompt)
            summary = summary_resp.content if hasattr(summary_resp, "content") else str(summary_resp)
            session.summary = summary
            session.summary_sequence = latest_seq

            lock = self._get_lock(session_id)
            with lock:
                self._summary_cache[session_id] = summary

            logger.info(
                "summary generated session=%s sequence=%d summary=%s",
                session_id,
                session.summary_sequence,
                summary,
            )
        except Exception as e:
            logger.error("failed to generate summary for %s: %s", session_id, e)

    def load_history(self, session_id: str, limit: Optional[int] = None) -> list[BaseMessage]:
        """Load full history from MySQL for a session."""
        db = SessionLocal()
        try:
            messages = ChatMessage.get_all(db, session_id, limit)
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
        lock = self._get_lock(session_id)
        with lock:
            self._buffers_s1.pop(session_id, None)
            self._buffers_s2.pop(session_id, None)
            self._summary_cache.pop(session_id, None)
        db = SessionLocal()
        try:
            ChatMessage.delete_by_session(db, session_id)
            ChatSession.delete_by_id(db, session_id)
            db.commit()
        finally:
            db.close()


# Singleton
memory_service = MemoryService()
