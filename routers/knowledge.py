import os
import tempfile
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.knowledge_base import DocumentMeta, KnowledgeBase
from services.document_loader import process_document
from services.rag_service import rag_service
from utils.db import get_db

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class KBCreateRequest(BaseModel):
    name: str
    description: str = ""


class KBResponse(BaseModel):
    id: str
    name: str
    description: str


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    status: str


@router.post("/base", response_model=KBResponse)
def create_kb(req: KBCreateRequest, db: Session = Depends(get_db)):
    """Create a new knowledge base."""
    kb = KnowledgeBase(
        id=str(uuid4()),
        name=req.name,
        description=req.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return KBResponse(id=kb.id, name=kb.name, description=kb.description)


@router.get("/base", response_model=list[KBResponse])
def list_kbs(db: Session = Depends(get_db)):
    """List all knowledge bases."""
    kbs = db.query(KnowledgeBase).all()
    return [KBResponse(id=kb.id, name=kb.name, description=kb.description) for kb in kbs]


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """Upload a document to a knowledge base and index it."""
    # Verify KB exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Save file temporarily
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process and chunk
        chunks = process_document(tmp_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document is empty or unsupported format")

        # Create metadata for each chunk
        doc_id = str(uuid4())
        metadatas = [
            {"doc_id": doc_id, "kb_id": kb_id, "file_name": file.filename}
            for _ in chunks
        ]

        # Index into Milvus
        rag_service.add_texts(chunks, metadatas=metadatas)

        # Save to DB
        doc_meta = DocumentMeta(
            id=doc_id,
            kb_id=kb_id,
            file_name=file.filename,
            file_path=tmp_path,
            status="indexed",
        )
        db.add(doc_meta)
        db.commit()

        return DocumentResponse(id=doc_meta.id, file_name=doc_meta.file_name, status=doc_meta.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)
