import logging

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.agent_service import AgentService
from services.agent_tools import build_chat_tools
from services.memory_service import MemoryService
from services.token_usage import TokenUsage, normalize_token_usage

logger = logging.getLogger(__name__)


class ChatChain:
    """Plain conversation chain with memory context."""

    def __init__(self, memory_service: MemoryService, provider: str | None = None):
        self.memory = memory_service
        self.provider = provider
        self._prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant.回答要简洁明了，不要废话。"),
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

    def _build_messages(self, summary: str, history: list[BaseMessage], message: str):
        return self._prompt.format_messages(
            summary=summary or "无",
            input=message,
            history=history,
        )

    def invoke(self, session_id: str, message: str) -> dict:
        """Run a conversation turn."""
        agent_service = AgentService(provider=self.provider)
        summary, history = self.memory.get_context_parts(session_id)
        messages = self._build_messages(summary, history, message)
        tools = build_chat_tools()

        logger.info("invoke session=%s provider=%s messages=%s", session_id, self.provider, history)

        result = agent_service.invoke(messages=messages, tools=tools)
        response = result["response"]
        usage = self._usage_payload(result.get("usage") or result.get("raw_response"))

        logger.info("response session=%s len=%d preview=%s", session_id, len(response), response[:120])
        logger.info(
            "usage session=%s provider=%s input_tokens=%d output_tokens=%d analysis_tokens=%d",
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

    async def astream(self, session_id: str, message: str):
        """Stream a conversation response."""
        agent_service = AgentService(provider=self.provider)
        summary, history = self.memory.get_context_parts(session_id)
        messages = self._build_messages(summary, history, message)
        tools = build_chat_tools()

        logger.info("astream session=%s provider=%s messages=%d", session_id, self.provider, len(history))

        self.memory.add_message(session_id, "user", message)

        full_response = []
        usage_payload = self._usage_payload(None)
        logger.info("astream messages=%s history=%s", message, history)
        async for event in agent_service.astream(messages=messages, tools=tools):
            if "usage" in event:
                usage_payload = self._usage_payload(event.get("usage"))
                continue
            text = event.get("text") or event.get("chunk", "")
            if text:
                full_response.append(text)
                yield {"chunk": text}

        full = "".join(full_response)
        logger.info("astream done session=%s total_len=%d preview=%s", session_id, len(full), full[:120])
        logger.info(
            "astream usage session=%s provider=%s input_tokens=%d output_tokens=%d analysis_tokens=%d",
            session_id,
            self.provider,
            usage_payload["input_tokens"],
            usage_payload["output_tokens"],
            usage_payload["analysis_tokens"],
        )

        self.memory.add_message(session_id, "assistant", full)
        yield {"usage": usage_payload}
