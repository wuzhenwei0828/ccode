from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.chat_message import ChatMessage
from models.chat_session import ChatSession
from utils.db import get_db

router = APIRouter(prefix="/api/session", tags=["session"])


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    user_id: int


class SessionResponse(BaseModel):
    id: int
    title: str
    message_count: int
    summary_sequence: int = 0


class MessageResponse(BaseModel):
    role: str
    content: str
    sequence: int


@router.post("", response_model=SessionResponse)
def create_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    """Create a new chat session."""
    session = ChatSession(
        user_id=req.user_id,
        title=req.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse(
        id=session.id,
        title=session.title,
        message_count=session.message_count,
        summary_sequence=session.summary_sequence or 0,
    )


@router.get("", response_model=list[SessionResponse])
def list_sessions(user_id: int = Query(...), db: Session = Depends(get_db)):
    """List chat sessions by user."""
    sessions = ChatSession.list_by_user(db, user_id)
    return [
        SessionResponse(id=s.id, title=s.title, message_count=s.message_count, summary_sequence=s.summary_sequence or 0)
        for s in sessions
    ]


@router.get("/{session_id}/history", response_model=list[MessageResponse])
def get_history(session_id: int, user_id: int = Query(...), db: Session = Depends(get_db)):
    """Get message history for a user-owned session."""
    session = ChatSession.get_by_user_and_id(db, user_id, session_id)
    if not session:
        return []
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
