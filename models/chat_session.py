from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Session

from utils.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment session ID")
    user_id = Column(BigInteger, nullable=False, index=True, comment="Owner user ID")
    title = Column(String(255), default="New Chat", comment="Session title")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="Last update time")
    message_count = Column(Integer, default=0, comment="Total messages in session")
    summary = Column(Text, nullable=True, default=None, comment="Conversation summary for long-term memory")
    summary_sequence = Column(Integer, default=0, comment="Last message sequence covered by summary")

    @classmethod
    def get_by_id(cls, db: Session, session_id: int):
        return db.query(cls).filter(cls.id == session_id).first()

    @classmethod
    def get_by_user_and_id(cls, db: Session, user_id: int, session_id: int):
        return db.query(cls).filter(cls.user_id == user_id).filter(cls.id == session_id).first()

    @classmethod
    def list_by_user(cls, db: Session, user_id: int):
        return db.query(cls).filter(cls.user_id == user_id).order_by(cls.updated_at.desc()).all()

    @classmethod
    def delete_by_id(cls, db: Session, session_id: int):
        return db.query(cls).filter(cls.id == session_id).delete()
