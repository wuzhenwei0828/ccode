import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from chains.rag_chain import RAGChain


class TestRAGChainSummaryPrompt(unittest.TestCase):
    @patch("chains.rag_chain.LLMFactory.create")
    def test_invoke_should_request_summary_and_history_separately(self, mock_create):
        memory = MagicMock()
        history = [HumanMessage(content="msg1")]
        memory.get_context_parts.return_value = ("summary text", history)

        rag_service = MagicMock()
        doc = MagicMock()
        doc.page_content = "kb-content"
        rag_service.query.return_value = [doc]

        llm = MagicMock()
        mock_create.return_value = llm

        chain = RAGChain(memory, rag_service, provider="siliconflow")

        with patch("chains.rag_chain.ChatPromptTemplate.from_messages") as mock_from_messages:
            prompt = MagicMock()
            parser_chain = MagicMock()
            full_chain = MagicMock()
            mock_from_messages.return_value = prompt
            prompt.__or__.return_value = parser_chain
            parser_chain.__or__.return_value = full_chain
            full_chain.invoke.return_value = "rag-answer"

            result = chain.invoke("sess-1", "hi", k=4)

        self.assertEqual(result, "rag-answer")
        memory.get_context_parts.assert_called_once_with("sess-1")
        full_chain.invoke.assert_called_once_with({
            "summary": "summary text",
            "history": history,
            "input": "hi",
            "context": "kb-content",
        })


if __name__ == "__main__":
    unittest.main()
