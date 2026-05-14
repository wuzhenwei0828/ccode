from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.llm.llm_factory import LLMFactory
from services.memory.memory_service import MemoryService


class ChatChain:
    """Plain conversation chain with memory context."""

    def __init__(self, memory_service: MemoryService, provider: str | None = None):
        self.memory = memory_service
        self.provider = provider
        self._prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self._parser = StrOutputParser()

    def invoke(self, session_id: str, message: str) -> str:
        """Run a conversation turn.

        Args:
            session_id: The conversation session ID.
            message: User's input message.

        Returns:
            Assistant's response text.
        """
        llm = LLMFactory.create(self.provider)
        chain = self._prompt | llm | self._parser

        # Get context from memory
        history = self.memory.get_context(session_id)

        # Run chain
        response = chain.invoke({"input": message, "history": history})

        # Save to memory
        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response)

        return response

    async def astream(self, session_id: str, message: str):
        """Stream a conversation response.

        Args:
            session_id: The conversation session ID.
            message: User's input message.

        Yields:
            Text chunks of the response.
        """
        llm = LLMFactory.create(self.provider)
        chain = self._prompt | llm | self._parser

        history = self.memory.get_context(session_id)

        # Save user message immediately
        self.memory.add_message(session_id, "user", message)

        # Stream response
        full_response = []
        async for chunk in chain.astream({"input": message, "history": history}):
            full_response.append(chunk)
            yield chunk

        # Save assistant response after streaming completes
        self.memory.add_message(session_id, "assistant", "".join(full_response))
