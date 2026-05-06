import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from utils.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Chatbot Service")


@app.on_event("startup")
def on_startup():
    """Initialize database tables on startup."""
    init_db()
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    """Serve the frontend chat interface."""
    return FileResponse("frontend/index.html", media_type="text/html")


# Register routers
from routers.chat import router as chat_router  # noqa: E402
from routers.session import router as session_router  # noqa: E402
from routers.knowledge import router as knowledge_router  # noqa: E402
from routers.user import router as user_router  # noqa: E402

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(knowledge_router)
app.include_router(user_router)
