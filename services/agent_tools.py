import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from services.rag_service import RAGService
logger = logging.getLogger(__name__)

@dataclass
class ChatTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


def _compact_text(value: str, limit: int = 240) -> str:
    text = (value or "").strip()
    return text[:limit]


def _build_knowledge_search_tool() -> ChatTool:
    rag_service = RAGService()

    def handler(question: str, k: int = 4) -> dict[str, Any]:
        logger.info("调用工具 knowledge_search")
        documents = rag_service.query(question, k=k)
        results = []
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            results.append({
                "content": _compact_text(getattr(document, "page_content", "")),
                "source": metadata.get("source") or metadata.get("file_name"),
                "metadata": {
                    key: metadata[key]
                    for key in ("kb_id", "doc_id")
                    if key in metadata
                },
            })
        return {
            "ok": True,
            "count": len(results),
            "results": results,
        }

    return ChatTool(
        name="knowledge_search",
        description="Search internal knowledge for supporting context before answering.",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to search for."},
                "k": {"type": "integer", "description": "Maximum number of results.", "default": 4},
            },
            "required": ["question"],
        },
        handler=handler,
    )


def _build_current_time_tool() -> ChatTool:
    def handler() -> dict[str, Any]:
        logger.info("调用工具 current_time")
        return {
            "ok": True,
            "current_time": datetime.now().isoformat(),
        }

    return ChatTool(
        name="current_time",
        description="获取当前时间，包含年月日时分秒",
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=handler,
    )


def build_chat_tools() -> list[ChatTool]:
    """Return the tool list for ChatChain agent execution."""
    return [_build_knowledge_search_tool(), _build_current_time_tool()]
