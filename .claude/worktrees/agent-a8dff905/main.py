from fastapi import FastAPI

from config import get_settings
from utils.db import init_db

app = FastAPI(title="Chatbot Service")


@app.on_event("startup")
def on_startup():
    """Initialize database tables on startup."""
    init_db()


@app.get("/")
async def root():
    settings = get_settings()
    return {"service": settings.app_name, "status": "running"}


# Register routers
from routers.chat import router as chat_router  # noqa: E402
from routers.session import router as session_router  # noqa: E402
from routers.knowledge import router as knowledge_router  # noqa: E402

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(knowledge_router)
