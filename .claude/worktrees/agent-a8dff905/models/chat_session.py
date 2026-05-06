from datetime import datetime

from sqlalchemy import Column, String, DateTime, Integer, func

from utils.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, comment="UUID for session")
    title = Column(String(255), default="New Chat", comment="Session title")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="Last update time")
    message_count = Column(Integer, default=0, comment="Total messages in session")
