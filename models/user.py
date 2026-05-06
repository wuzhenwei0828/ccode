from sqlalchemy import BigInteger, Column, DateTime, String, func
from sqlalchemy.orm import Session

from utils.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment user ID")
    name = Column(String(255), nullable=False, comment="Display name")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")

    @classmethod
    def get_by_id(cls, db: Session, user_id: int):
        return db.query(cls).filter(cls.id == user_id).first()

    @classmethod
    def list_all(cls, db: Session):
        return db.query(cls).order_by(cls.created_at.asc()).all()
