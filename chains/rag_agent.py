import logging

from services.llm.llm_factory import LLMFactory
from services.memory.memory_service import MemoryService
from services.planners.factory import PlannerFactory
from services.rag_service import RAGService
from services.token_usage import TokenUsage, normalize_token_usage

logger = logging.getLogger(__name__)


class RAGAgent:
    """Thin RAG agent wrapper around retrieval, planner, and memory."""

    def __init__(
        self,
        memory_service: MemoryService,
        rag_service: RAGService,
        provider: str | None = None,
        planner=None,
        planner_mode: str = "react",
    ):
        self.memory = memory_service
        self.rag = rag_service
        self.provider = provider
        self._planner = planner or PlannerFactory.create(mode=planner_mode, provider=provider)

    @staticmethod
    def _usage_payload(usage: TokenUsage | dict | None) -> dict:
        if isinstance(usage, dict):
            return {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "analysis_tokens": int(usage.get("analysis_tokens", 0)),
            }
        normalized = normalize_token_usage(usage)
        return {
            "input_tokens": normalized.input_tokens,
            "output_tokens": normalized.output_tokens,
            "analysis_tokens": normalized.analysis_tokens,
        }

    @staticmethod
    def _build_context(docs: list) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, session_id: str | int, message: str, k: int = 4) -> dict:
        docs = self.rag.query(message, k=k)
        context = self._build_context(docs)
        summary, history = self.memory.get_context_parts(session_id)

        logger.info("rag invoke session=%s provider=%s docs=%d", session_id, self.provider, len(docs))

        llm = LLMFactory.create(self.provider)
        result = self._planner.invoke(
            llm=llm,
            summary=summary,
            history=history,
            message=message,
            context=context,
            tools=[],
        )
        response = result["response"]
        usage = self._usage_payload(result.get("usage") or result.get("raw_response"))

        logger.info("rag response session=%s len=%d preview=%s", session_id, len(response), response[:120])
        logger.info(
            "rag usage session=%s provider=%s input_tokens=%d output_tokens=%d analysis_tokens=%d",
            session_id,
            self.provider,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["analysis_tokens"],
        )

        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response)

        return {
            "response": response,
            "usage": usage,
        }

    async def astream(self, session_id: str | int, message: str, k: int = 4):
        docs = self.rag.query(message, k=k)
        context = self._build_context(docs)
        summary, history = self.memory.get_context_parts(session_id)

        logger.info("rag astream session=%s provider=%s docs=%d", session_id, self.provider, len(docs))

        self.memory.add_message(session_id, "user", message)

        full_response = []
        usage_payload = self._usage_payload(None)
        llm = LLMFactory.create(self.provider)
        async for event in self._planner.astream(
            llm=llm,
            summary=summary,
            history=history,
            message=message,
            context=context,
            tools=[],
        ):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "usage" or (event_type is None and "usage" in event):
                usage_payload = self._usage_payload(event.get("usage"))
                continue
            if event_type == "chunk" or (event_type is None and "chunk" in event):
                text = event.get("chunk", "")
                if text:
                    full_response.append(text)
                    yield {"chunk": text}

        full = "".join(full_response)
        logger.info("rag astream done session=%s total_len=%d preview=%s", session_id, len(full), full[:120])
        logger.info(
            "rag astream usage session=%s provider=%s input_tokens=%d output_tokens=%d analysis_tokens=%d",
            session_id,
            self.provider,
            usage_payload["input_tokens"],
            usage_payload["output_tokens"],
            usage_payload["analysis_tokens"],
        )

        self.memory.add_message(session_id, "assistant", full)
        yield {"usage": usage_payload}
