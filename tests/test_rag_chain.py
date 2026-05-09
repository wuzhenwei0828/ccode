import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from chains.rag_chain import RAGChain


class TestRAGChainSummaryPrompt(unittest.TestCase):
    @patch("chains.rag_chain.LLMFactory.create")
    def test_invoke_passes_summary_history_context_and_question_to_model(self, mock_create):
        memory = MagicMock()
        history = [HumanMessage(content="msg1")]
        memory.get_context_parts.return_value = ("summary text", history)

        rag_service = MagicMock()
        doc = SimpleNamespace(page_content="kb-content")
        rag_service.query.return_value = [doc]

        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="rag-answer", usage_metadata={})
        mock_create.return_value = llm

        chain = RAGChain(memory, rag_service, provider="siliconflow")
        result = chain.invoke("sess-1", "hi", k=4)

        self.assertEqual(result, {
            "response": "rag-answer",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "analysis_tokens": 0,
            },
        })
        memory.get_context_parts.assert_called_once_with("sess-1")
        rag_service.query.assert_called_once_with("hi", k=4)
        prompt_messages = llm.invoke.call_args.args[0]
        joined_content = "\n".join(getattr(message, "content", "") for message in prompt_messages)
        self.assertIn("summary text", joined_content)
        self.assertIn("msg1", joined_content)
        self.assertIn("kb-content", joined_content)
        self.assertIn("Question: hi", joined_content)


class TestRAGChainUsage(unittest.TestCase):
    @patch("chains.rag_chain.LLMFactory.create")
    def test_invoke_returns_response_and_usage(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        rag_service = MagicMock()
        rag_service.query.return_value = [SimpleNamespace(page_content="kb-content")]

        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(
            content="rag-answer",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 8,
                    "output_tokens_details": {"reasoning_tokens": 3},
                }
            },
        )
        mock_create.return_value = llm

        chain = RAGChain(memory, rag_service, provider="siliconflow")
        result = chain.invoke("sess-1", "hi", k=4)

        self.assertEqual(result, {
            "response": "rag-answer",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 8,
                "analysis_tokens": 3,
            },
        })
        memory.add_message.assert_any_call("sess-1", "user", "hi")
        memory.add_message.assert_any_call("sess-1", "assistant", "rag-answer")

    @patch("chains.rag_chain.LLMFactory.create")
    def test_astream_yields_chunks_and_usage_event(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        rag_service = MagicMock()
        rag_service.query.return_value = [SimpleNamespace(page_content="kb-content")]

        llm = MagicMock()
        mock_create.return_value = llm

        async def fake_astream(_messages):
            yield SimpleNamespace(content="rag-")
            yield SimpleNamespace(content="answer", response_metadata={"token_usage": {"prompt_tokens": 12, "completion_tokens": 6, "output_tokens_details": {"reasoning_tokens": 2}}})

        llm.astream = fake_astream

        chain = RAGChain(memory, rag_service, provider="siliconflow")

        async def collect_events():
            events = []
            async for event in chain.astream("sess-1", "hi", k=4):
                events.append(event)
            return events

        result = asyncio.run(collect_events())

        self.assertEqual(result, [
            {"chunk": "rag-"},
            {"chunk": "answer"},
            {"usage": {"input_tokens": 12, "output_tokens": 6, "analysis_tokens": 2}},
        ])
        memory.add_message.assert_any_call("sess-1", "user", "hi")
        memory.add_message.assert_any_call("sess-1", "assistant", "rag-answer")

    @patch("chains.rag_chain.LLMFactory.create")
    def test_astream_returns_zero_usage_when_no_chunks_are_emitted(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        rag_service = MagicMock()
        rag_service.query.return_value = [SimpleNamespace(page_content="kb-content")]

        llm = MagicMock()
        mock_create.return_value = llm

        async def fake_astream(_messages):
            if False:
                yield None

        llm.astream = fake_astream

        chain = RAGChain(memory, rag_service, provider="siliconflow")

        async def collect_events():
            events = []
            async for event in chain.astream("sess-1", "hi", k=4):
                events.append(event)
            return events

        result = asyncio.run(collect_events())

        self.assertEqual(result, [
            {"usage": {"input_tokens": 0, "output_tokens": 0, "analysis_tokens": 0}},
        ])
        memory.add_message.assert_any_call("sess-1", "user", "hi")
        memory.add_message.assert_any_call("sess-1", "assistant", "")


if __name__ == "__main__":
    unittest.main()
