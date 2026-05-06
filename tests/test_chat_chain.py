import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from chains.chat_chain import ChatChain


class TestChatChainSummaryPrompt(unittest.TestCase):
    @patch("chains.chat_chain.LLMFactory.create")
    def test_invoke_should_request_summary_and_history_separately(self, mock_create):
        memory = MagicMock()
        history = [HumanMessage(content="msg1")]
        memory.get_context_parts.return_value = ("summary text", history)

        llm = MagicMock()
        mock_create.return_value = llm

        chain = ChatChain(memory, provider="siliconflow")

        try:
            chain.invoke("sess-1", "hi")
        except Exception:
            pass

        memory.get_context_parts.assert_called_once_with("sess-1")

    @patch("chains.chat_chain.LLMFactory.create")
    def test_invoke_accepts_integer_session_id(self, mock_create):
        history = [HumanMessage(content="hi")]
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", history)

        llm = MagicMock()
        mock_create.return_value = llm

        chain = ChatChain(memory, provider="siliconflow")

        with patch("chains.chat_chain.ChatPromptTemplate.from_messages") as mock_from_messages:
            prompt = MagicMock()
            parser_chain = MagicMock()
            full_chain = MagicMock()
            mock_from_messages.return_value = prompt
            prompt.__or__.return_value = parser_chain
            parser_chain.__or__.return_value = full_chain
            full_chain.invoke.return_value = "answer"

            result = chain.invoke(123, "hello")

        self.assertEqual(result, "answer")
        memory.get_context_parts.assert_called_with(123)


if __name__ == "__main__":
    unittest.main()
