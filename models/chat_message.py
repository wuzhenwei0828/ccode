from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Session

from utils.db import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment message ID")
    session_id = Column(BigInteger, nullable=False, index=True, comment="Associated session ID")
    role = Column(String(20), nullable=False, comment="Message role: user / assistant / system")
    content = Column(Text, nullable=False, comment="Message content")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    sequence = Column(Integer, default=0, comment="Message order within session")

    @classmethod
    def get_after_sequence(cls, db: Session, session_id: int, sequence: int) -> list["ChatMessage"]:
        return (
            db.query(cls)
            .filter(cls.session_id == session_id)
            .filter(cls.sequence > sequence)
            .order_by(cls.sequence.asc())
            .all()
        )

    @classmethod
    def get_max_sequence(cls, db: Session, session_id: int):
        return (
            db.query(cls.sequence)
            .filter(cls.session_id == session_id)
            .order_by(cls.sequence.desc())
            .first()
        )

    @classmethod
    def get_recent(cls, db: Session, session_id: int, limit: int) -> list["ChatMessage"]:
        return (
            db.query(cls)
            .filter(cls.session_id == session_id)
            .order_by(cls.sequence.desc())
            .limit(limit)
            .all()
        )

    @classmethod
    def get_all(cls, db: Session, session_id: int, limit: int | None = None) -> list["ChatMessage"]:
        query = (
            db.query(cls)
            .filter(cls.session_id == session_id)
            .order_by(cls.sequence.asc())
        )
        if limit is not None:
            query = query.limit(limit)
        return query.all()

    @classmethod
    def delete_by_session(cls, db: Session, session_id: int):
        return db.query(cls).filter(cls.session_id == session_id).delete()
