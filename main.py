import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from services.tools.registry import ToolRegistry
from utils.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="Chatbot Service")
logger = logging.getLogger(__name__)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.on_event("startup")
async def on_startup():
    """Initialize runtime dependencies on startup."""
    init_db()
    await ToolRegistry.default().abuild_tools()
    logger.info("startup tool preload complete")


@app.get("/")
async def root():
    """Serve the frontend chat interface."""
    return FileResponse(str(FRONTEND_DIR / "index.html"), media_type="text/html")


# Register routers
from routers.chat import router as chat_router  # noqa: E402
from routers.session import router as session_router  # noqa: E402
from routers.knowledge import router as knowledge_router  # noqa: E402
from routers.user import router as user_router  # noqa: E402

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(knowledge_router)
app.include_router(user_router)
