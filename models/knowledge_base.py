from sqlalchemy import BigInteger, Column, DateTime, String, Text, func

from utils.db import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment knowledge base ID")
    name = Column(String(255), nullable=False, comment="Knowledge base name")
    description = Column(Text, default="", comment="Description")
    created_at = Column(DateTime, server_default=func.now())


class DocumentMeta(Base):
    __tablename__ = "document_metas"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment document ID")
    kb_id = Column(BigInteger, nullable=False, index=True, comment="Associated knowledge base ID")
    file_name = Column(String(255), nullable=False, comment="Original file name")
    file_path = Column(String(512), default="", comment="Stored file path")
    status = Column(String(20), default="pending", comment="pending / indexed / failed")
    created_at = Column(DateTime, server_default=func.now())
