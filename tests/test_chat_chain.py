import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from chains.chat_chain import ChatChain


class TestChatChainSummaryPrompt(unittest.TestCase):
    @patch("chains.chat_chain.LLMFactory.create")
    def test_invoke_uses_summary_and_history_in_prompt_messages(self, mock_create):
        history = [HumanMessage(content="msg1")]
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", history)

        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        mock_create.return_value = llm

        chain = ChatChain(memory, provider="siliconflow")
        chain.invoke("sess-1", "hi")

        memory.get_context_parts.assert_called_once_with("sess-1")
        prompt_messages = llm.invoke.call_args.args[0]
        self.assertEqual(prompt_messages[1].content, "以下是历史对话摘要，仅供参考：\nsummary text")
        self.assertEqual(prompt_messages[2].content, "msg1")
        self.assertEqual(prompt_messages[3].content, "hi")

    @patch("chains.chat_chain.LLMFactory.create")
    def test_invoke_accepts_integer_session_id(self, mock_create):
        history = [HumanMessage(content="hi")]
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", history)

        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        mock_create.return_value = llm

        chain = ChatChain(memory, provider="siliconflow")
        result = chain.invoke(123, "hello")

        self.assertEqual(result, {
            "response": "answer",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "analysis_tokens": 0,
            },
        })
        memory.get_context_parts.assert_called_with(123)


class TestChatChainUsage(unittest.TestCase):
    @patch("chains.chat_chain.LLMFactory.create")
    def test_invoke_returns_response_and_usage(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        llm = MagicMock()
        mock_create.return_value = llm

        response = SimpleNamespace(content="answer", usage_metadata={"input_tokens": 10, "output_tokens": 4})
        llm.invoke.return_value = response

        chain = ChatChain(memory, provider="siliconflow")
        result = chain.invoke("sess-1", "hello")

        self.assertEqual(result, {
            "response": "answer",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "analysis_tokens": 0,
            },
        })
        memory.add_message.assert_any_call("sess-1", "user", "hello")
        memory.add_message.assert_any_call("sess-1", "assistant", "answer")

    @patch("chains.chat_chain.LLMFactory.create")
    def test_astream_yields_chunks_and_usage_event(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        llm = MagicMock()
        mock_create.return_value = llm

        async def fake_astream(_messages):
            yield SimpleNamespace(content="hel")
            yield SimpleNamespace(content="lo", response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "output_tokens_details": {"reasoning_tokens": 2}}})

        llm.astream = fake_astream

        chain = ChatChain(memory, provider="siliconflow")

        async def collect_events():
            events = []
            async for event in chain.astream("sess-1", "hello"):
                events.append(event)
            return events

        result = asyncio.run(collect_events())

        self.assertEqual(result, [
            {"chunk": "hel"},
            {"chunk": "lo"},
            {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}},
        ])
        memory.add_message.assert_any_call("sess-1", "user", "hello")
        memory.add_message.assert_any_call("sess-1", "assistant", "hello")

    @patch("chains.chat_chain.LLMFactory.create")
    def test_astream_returns_zero_usage_when_no_chunks_are_emitted(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        llm = MagicMock()
        mock_create.return_value = llm

        async def fake_astream(_messages):
            if False:
                yield None

        llm.astream = fake_astream

        chain = ChatChain(memory, provider="siliconflow")

        async def collect_events():
            events = []
            async for event in chain.astream("sess-1", "hello"):
                events.append(event)
            return events

        result = asyncio.run(collect_events())

        self.assertEqual(result, [
            {"usage": {"input_tokens": 0, "output_tokens": 0, "analysis_tokens": 0}},
        ])
        memory.add_message.assert_any_call("sess-1", "user", "hello")
        memory.add_message.assert_any_call("sess-1", "assistant", "")


if __name__ == "__main__":
    unittest.main()
