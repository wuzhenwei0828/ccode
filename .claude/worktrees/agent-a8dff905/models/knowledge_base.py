from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, Integer, func

from utils.db import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, comment="UUID for knowledge base")
    name = Column(String(255), nullable=False, comment="Knowledge base name")
    description = Column(Text, default="", comment="Description")
    created_at = Column(DateTime, server_default=func.now())


class DocumentMeta(Base):
    __tablename__ = "document_metas"

    id = Column(String(36), primary_key=True, comment="UUID for document")
    kb_id = Column(String(36), nullable=False, index=True, comment="Associated knowledge base ID")
    file_name = Column(String(255), nullable=False, comment="Original file name")
    file_path = Column(String(512), default="", comment="Stored file path")
    status = Column(String(20), default="pending", comment="pending / indexed / failed")
    created_at = Column(DateTime, server_default=func.now())
