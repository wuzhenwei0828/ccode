# Import all models so they are registered with Base.metadata
from models.chat_session import ChatSession  # noqa: F401
from models.chat_message import ChatMessage  # noqa: F401
from models.knowledge_base import KnowledgeBase, DocumentMeta  # noqa: F401

from utils.db import Base

__all__ = ["Base", "ChatSession", "ChatMessage", "KnowledgeBase", "DocumentMeta"]
