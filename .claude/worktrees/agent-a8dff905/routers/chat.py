import json
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from chains.chat_chain import ChatChain
from chains.rag_chain import RAGChain
from services.memory_service import memory_service
from services.rag_service import rag_service
from utils.db import get_db

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = False
    provider: Optional[str] = None
    rag_k: int = 4


class ChatResponse(BaseModel):
    session_id: str
    response: str


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Send a message and receive a complete response."""
    if req.use_rag:
        chain = RAGChain(memory_service, rag_service, provider=req.provider)
        response = chain.invoke(req.session_id, req.message, k=req.rag_k)
    else:
        chain = ChatChain(memory_service, provider=req.provider)
        response = chain.invoke(req.session_id, req.message)

    return ChatResponse(session_id=req.session_id, response=response)


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """Send a message and receive a streaming SSE response."""

    from fastapi.responses import StreamingResponse

    async def event_generator():
        if req.use_rag:
            chain = RAGChain(memory_service, rag_service, provider=req.provider)
            async for chunk in chain.astream(req.session_id, req.message, k=req.rag_k):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        else:
            chain = ChatChain(memory_service, provider=req.provider)
            async for chunk in chain.astream(req.session_id, req.message):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
