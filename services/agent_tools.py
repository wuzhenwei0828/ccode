import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

from config.settings import get_settings
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


def _normalize_web_search_snippet(item: Any) -> str:
    for field in ("summary", "snippet", "body"):
        value = item.get(field) if isinstance(item, dict) else getattr(item, field, "")
        if isinstance(value, str) and value.strip():
            return value

    return ""


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


def _extract_web_search_results(response: Any) -> list[dict[str, str]]:
    if response is None:
        return []

    if isinstance(response, tuple) and response:
        response = response[0]

    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]

    return []


def _run_web_search(query: str):
    settings = get_settings()
    configured_max_results = getattr(settings, "search_top_k", 5)
    max_results = max(configured_max_results, 1)
    search_tool = DuckDuckGoSearchResults(
        max_results=max_results,
        output_format="list",
    )
    return search_tool.invoke(query)


def _build_web_search_error_result(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": "web_search_error",
            "message": message,
        },
    }


def _build_web_search_tool() -> ChatTool:
    def handler(query: str, limit: int = 5) -> dict[str, Any]:
        logger.info("调用工具 web_search")

        normalized_query = (query or "").strip()
        if not normalized_query:
            return _build_web_search_error_result("query must not be blank")

        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return _build_web_search_error_result("limit must be a positive integer")

        try:
            raw_results = _run_web_search(normalized_query)
            normalized_results = _extract_web_search_results(raw_results)
        except Exception:
            logger.exception("web_search backend failure query=%s", normalized_query)
            return _build_web_search_error_result("web search backend unavailable")

        results = []
        for item in normalized_results[:limit]:
            results.append({
                "title": str(item.get("title") or ""),
                "snippet": _compact_text(_normalize_web_search_snippet(item)),
                "url": str(item.get("url") or item.get("link") or ""),
            })
        return {
            "ok": True,
            "count": len(results),
            "results": results,
        }

    return ChatTool(
        name="web_search",
        description="Search public web results for external or real-time information.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query keywords."},
                "limit": {"type": "integer", "description": "Maximum number of results.", "default": 5},
            },
            "required": ["query"],
        },
        handler=handler,
    )


def build_chat_tools() -> list[ChatTool]:
    """Return the tool list for ChatChain agent execution."""
    return [
        _build_knowledge_search_tool(),
        _build_current_time_tool(),
        _build_web_search_tool(),
    ]
