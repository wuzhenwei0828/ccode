import logging

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.llm_factory import LLMFactory
from services.memory_service import MemoryService
from services.token_usage import normalize_token_usage

logger = logging.getLogger(__name__)


class ChatChain:
    """Plain conversation chain with memory context."""

    def __init__(self, memory_service: MemoryService, provider: str | None = None):
        self.memory = memory_service
        self.provider = provider
        self._prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant."),
            ("human", "以下是历史对话摘要，仅供参考：\n{summary}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

    @staticmethod
    def _extract_text(response) -> str:
        if isinstance(response, str):
            return response
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(getattr(item, "text", ""))
            return "".join(parts)
        return str(content or "")

    def _build_messages(self, summary: str, history: list[BaseMessage], message: str):
        return self._prompt.format_messages(
            summary=summary or "无",
            input=message,
            history=history,
        )

    def invoke(self, session_id: str, message: str) -> dict:
        """Run a conversation turn."""
        llm = LLMFactory.create(self.provider)
        summary, history = self.memory.get_context_parts(session_id)
        messages = self._build_messages(summary, history, message)

        logger.info("invoke session=%s provider=%s messages=%s", session_id, self.provider, history)

        raw_response = llm.invoke(messages)
        response = self._extract_text(raw_response)
        usage = normalize_token_usage(raw_response)

        logger.info("response session=%s len=%d preview=%s", session_id, len(response), response[:120])
        logger.info(
            "usage session=%s provider=%s input_tokens=%d output_tokens=%d analysis_tokens=%d",
            session_id,
            self.provider,
            usage.input_tokens,
            usage.output_tokens,
            usage.analysis_tokens,
        )

        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response)

        return {
            "response": response,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "analysis_tokens": usage.analysis_tokens,
            },
        }

    async def astream(self, session_id: str, message: str):
        """Stream a conversation response."""
        llm = LLMFactory.create(self.provider)
        summary, history = self.memory.get_context_parts(session_id)
        messages = self._build_messages(summary, history, message)

        logger.info("astream session=%s provider=%s messages=%d", session_id, self.provider, len(history))

        self.memory.add_message(session_id, "user", message)

        full_response = []
        last_chunk = None
        logger.info("astream messages=%s history=%s", message, history)
        async for chunk in llm.astream(messages):
            text = self._extract_text(chunk)
            if chunk.usage_metadata:
                last_chunk = chunk
            if text:
                full_response.append(text)
                yield {"chunk": text}

        full = "".join(full_response)
        usage = normalize_token_usage(last_chunk)
        logger.info("astream done session=%s total_len=%d preview=%s", session_id, len(full), full[:120])
        logger.info(
            "astream usage session=%s provider=%s input_tokens=%d output_tokens=%d analysis_tokens=%d",
            session_id,
            self.provider,
            usage.input_tokens,
            usage.output_tokens,
            usage.analysis_tokens,
        )

        self.memory.add_message(session_id, "assistant", full)
        yield {
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "analysis_tokens": usage.analysis_tokens,
            },
        }
