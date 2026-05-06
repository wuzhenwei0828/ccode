from uuid import uuid4

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.chat_message import ChatMessage
from models.chat_session import ChatSession
from utils.db import get_db

router = APIRouter(prefix="/api/session", tags=["session"])


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


class SessionResponse(BaseModel):
    id: str
    title: str
    message_count: int


class MessageResponse(BaseModel):
    role: str
    content: str
    sequence: int


@router.post("", response_model=SessionResponse)
def create_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    """Create a new chat session."""
    session = ChatSession(
        id=str(uuid4()),
        title=req.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse(
        id=session.id,
        title=session.title,
        message_count=session.message_count,
    )


@router.get("", response_model=list[SessionResponse])
def list_sessions(db: Session = Depends(get_db)):
    """List all chat sessions."""
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    return [
        SessionResponse(id=s.id, title=s.title, message_count=s.message_count)
        for s in sessions
    ]


@router.get("/{session_id}/history", response_model=list[MessageResponse])
def get_history(session_id: str, db: Session = Depends(get_db)):
    """Get message history for a session."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.sequence.asc())
        .all()
    )
    return [
        MessageResponse(role=m.role, content=m.content, sequence=m.sequence)
        for m in messages
    ]
