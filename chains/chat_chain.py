import logging

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.llm_factory import LLMFactory
from services.memory_service import MemoryService

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
        self._parser = StrOutputParser()

    def invoke(self, session_id: str, message: str) -> str:
        """Run a conversation turn."""
        llm = LLMFactory.create(self.provider)
        chain = self._prompt | llm | self._parser

        summary, history = self.memory.get_context_parts(session_id)

        logger.info("invoke session=%s provider=%s messages=%s", session_id, self.provider, history)

        response = chain.invoke({
            "summary": summary or "无",
            "input": message,
            "history": history,
        })

        logger.info("response session=%s len=%d preview=%s", session_id, len(response), response[:120])

        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response)

        return response

    async def astream(self, session_id: str, message: str):
        """Stream a conversation response."""
        llm = LLMFactory.create(self.provider)
        chain = self._prompt | llm | self._parser

        summary, history = self.memory.get_context_parts(session_id)

        logger.info("astream session=%s provider=%s messages=%d", session_id, self.provider, len(history))

        self.memory.add_message(session_id, "user", message)

        full_response = []
        logger.info("astream messages=%s history=%s", message, history)
        async for chunk in chain.astream({
            "summary": summary or "无",
            "input": message,
            "history": history,
        }):
            full_response.append(chunk)
            yield chunk

        full = "".join(full_response)
        logger.info("astream done session=%s total_len=%d preview=%s", session_id, len(full), full[:120])

        self.memory.add_message(session_id, "assistant", full)
