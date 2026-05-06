from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, Integer, func

from utils.db import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, comment="UUID for message")
    session_id = Column(String(36), nullable=False, index=True, comment="Associated session ID")
    role = Column(String(20), nullable=False, comment="Message role: user / assistant / system")
    content = Column(Text, nullable=False, comment="Message content")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    sequence = Column(Integer, default=0, comment="Message order within session")
