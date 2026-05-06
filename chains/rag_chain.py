from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.llm_factory import LLMFactory
from services.memory_service import MemoryService
from services.rag_service import RAGService


class RAGChain:
    """RAG conversation chain: retrieve context, then answer with LLM."""

    _RAG_PROMPT = """You are a helpful assistant. Answer the question based on the provided context.
If the context does not contain relevant information, say "Based on the provided knowledge base, I cannot find relevant information."

Context:
{context}

Question: {input}
"""

    def __init__(self, memory_service: MemoryService, rag_service: RAGService, provider: str | None = None):
        self.memory = memory_service
        self.rag = rag_service
        self.provider = provider

    def invoke(self, session_id: str, message: str, k: int = 4) -> str:
        """Run a RAG conversation turn."""
        llm = LLMFactory.create(self.provider)
        parser = StrOutputParser()

        docs = self.rag.query(message, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer based on the provided context."),
            ("human", "以下是历史对话摘要，仅供参考：\n{summary}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", self._RAG_PROMPT),
        ])

        chain = prompt | llm | parser
        summary, history = self.memory.get_context_parts(session_id)

        response = chain.invoke({
            "summary": summary or "无",
            "input": message,
            "history": history,
            "context": context,
        })

        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response)

        return response

    async def astream(self, session_id: str, message: str, k: int = 4):
        """Stream a RAG response."""
        llm = LLMFactory.create(self.provider)
        parser = StrOutputParser()

        docs = self.rag.query(message, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer based on the provided context."),
            ("human", "以下是历史对话摘要，仅供参考：\n{summary}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", self._RAG_PROMPT),
        ])

        chain = prompt | llm | parser
        summary, history = self.memory.get_context_parts(session_id)

        self.memory.add_message(session_id, "user", message)

        full_response = []
        async for chunk in chain.astream({
            "summary": summary or "无",
            "input": message,
            "history": history,
            "context": context,
        }):
            full_response.append(chunk)
            yield chunk

        self.memory.add_message(session_id, "assistant", "".join(full_response))
