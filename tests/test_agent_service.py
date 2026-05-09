import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from services.agent_service import AgentService
from services.agent_tools import build_chat_tools


class StubTool:
    def __init__(self, name, handler):
        self.name = name
        self.handler = handler


class TestAgentService(unittest.TestCase):
    @patch("services.agent_tools.RAGService", create=True)
    def test_build_chat_tools_registers_knowledge_and_current_time_tools(self, mock_rag_service_cls):
        mock_rag_service_cls.return_value.query.return_value = []

        tools = build_chat_tools()

        self.assertTrue(tools)
        knowledge_tool = next(tool for tool in tools if tool.name == "knowledge_search")
        current_time_tool = next(tool for tool in tools if tool.name == "current_time")
        self.assertIn("knowledge", knowledge_tool.description.lower())
        self.assertTrue(callable(knowledge_tool.handler))
        self.assertIn("time", current_time_tool.description.lower())
        self.assertTrue(callable(current_time_tool.handler))

    def test_current_time_tool_returns_serializable_result(self):
        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "current_time")

        result = tool.handler()

        self.assertEqual(result["ok"], True)
        self.assertIsInstance(result["current_time"], str)

    @patch("services.agent_tools.RAGService", create=True)
    def test_knowledge_tool_returns_serializable_result(self, mock_rag_service_cls):
        rag_service = MagicMock()
        rag_service.query.return_value = [
            Document(page_content="alpha" * 80, metadata={"source": "doc-a", "kb_id": "kb-1"}),
            Document(page_content="beta", metadata={"source": "doc-b"}),
        ]
        mock_rag_service_cls.return_value = rag_service

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "knowledge_search")

        result = tool.handler(question="What is ReAct?", k=2)

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["results"]), 2)
        self.assertIsInstance(result["results"][0]["content"], str)
        self.assertLessEqual(len(result["results"][0]["content"]), 240)
        self.assertEqual(result["results"][0]["source"], "doc-a")
        self.assertNotIsInstance(result["results"][0], Document)

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_returns_final_answer_without_tool_calls(self, mock_create):
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[])

        self.assertEqual(result["response"], "answer")
        self.assertIs(result["raw_response"], llm.invoke.return_value)
        llm.invoke.assert_called_once_with(["m1"])

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_bind_tools_before_first_model_call(self, mock_create):
        llm = MagicMock()
        bound_llm = MagicMock()
        bound_llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        llm.bind_tools.return_value = bound_llm
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        tool = StubTool("knowledge_search", MagicMock(return_value={"ok": True}))
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["response"], "answer")
        llm.bind_tools.assert_called_once()
        bound_llm.invoke.assert_called_once_with(["m1"])
        llm.invoke.assert_not_called()

    @patch("services.agent_service.LLMFactory.create")
    @patch("services.agent_tools.RAGService", create=True)
    def test_invoke_schema_to_bind_tools(self, mock_rag_service_cls, mock_create):
        mock_rag_service_cls.return_value.query.return_value = []
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        llm.bind_tools.return_value = llm
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        tools = build_chat_tools()
        service.invoke(messages=["m1"], tools=tools)

        llm.bind_tools.assert_called_once()
        payload = llm.bind_tools.call_args.args[0]
        tool_names = {item["name"] for item in payload}
        self.assertEqual(tool_names, {"knowledge_search", "current_time"})
        current_time_tool = next(item for item in payload if item["name"] == "current_time")
        self.assertEqual(current_time_tool["name"], "current_time")
        self.assertTrue(current_time_tool["description"])
        self.assertIsInstance(current_time_tool["input_schema"], dict)
        self.assertEqual(current_time_tool["input_schema"]["type"], "object")

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_uses_raw_model_when_bind_tools_is_missing(self, mock_create):
        llm = MagicMock(spec=["invoke"])
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[StubTool("knowledge_search", MagicMock())])

        self.assertEqual(result["response"], "answer")
        llm.invoke.assert_called_once_with(["m1"])

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_uses_raw_model_when_bind_tools_raises(self, mock_create):
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        llm.bind_tools.side_effect = NotImplementedError()
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[StubTool("knowledge_search", MagicMock())])

        self.assertEqual(result["response"], "answer")
        llm.bind_tools.assert_called_once()
        llm.invoke.assert_called_once_with(["m1"])

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_keeps_tool_call_text_fallback_when_bind_tools_raises(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.side_effect = ValueError("unsupported")
        llm.invoke.side_effect = [
            SimpleNamespace(
                content='TOOL_CALL {"name": "knowledge_search", "args": {"question": "What is ReAct?", "k": 2}}',
                usage_metadata={},
            ),
            SimpleNamespace(content="Final answer", usage_metadata={}),
        ]
        mock_create.return_value = llm

        tool = StubTool(
            "knowledge_search",
            MagicMock(return_value={"ok": True, "results": [{"content": "ReAct is a tool loop."}]}),
        )

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["response"], "Final answer")
        tool.handler.assert_called_once_with(question="What is ReAct?", k=2)
        llm.invoke.assert_called()

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_executes_tool_then_returns_final_answer(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"question": "What is ReAct?", "k": 2}}],
                usage_metadata={},
            ),
            SimpleNamespace(content="Final answer", usage_metadata={}),
        ]
        mock_create.return_value = llm

        tool = StubTool(
            "knowledge_search",
            MagicMock(return_value={"ok": True, "results": [{"content": "ReAct is a tool loop."}]}),
        )

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["response"], "Final answer")
        tool.handler.assert_called_once_with(question="What is ReAct?", k=2)
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_executes_zero_arg_current_time_tool_calls(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "current_time", "args": {}}],
                usage_metadata={},
            ),
            SimpleNamespace(content="现在是 2026-05-09T10:00:00", usage_metadata={}),
        ]
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        current_time_tool = next(tool for tool in build_chat_tools() if tool.name == "current_time")
        current_time_tool.handler = MagicMock(return_value={"ok": True, "current_time": "2026-05-09T10:00:00"})

        result = service.invoke(messages=["m1"], tools=[current_time_tool])

        self.assertEqual(result["response"], "现在是 2026-05-09T10:00:00")
        llm.bind_tools.assert_called_once()
        current_time_tool.handler.assert_called_once_with()
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_stops_at_max_iterations(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(content="", tool_calls=[{"name": "knowledge_search", "args": {"question": "loop"}}], usage_metadata={}),
            SimpleNamespace(content="", tool_calls=[{"name": "knowledge_search", "args": {"question": "loop"}}], usage_metadata={}),
        ]
        mock_create.return_value = llm

        tool = StubTool("knowledge_search", MagicMock(return_value={"ok": True, "results": []}))

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool], max_iterations=2)

        self.assertEqual(result["response"], "抱歉，我暂时无法完成这次工具调用。")
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_converts_tool_error_into_recoverable_result(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"question": "boom"}}],
                usage_metadata={},
            ),
            SimpleNamespace(content="Recovered answer", usage_metadata={}),
        ]
        mock_create.return_value = llm

        tool = StubTool("knowledge_search", MagicMock(side_effect=RuntimeError("tool failed")))

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["response"], "Recovered answer")
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_handles_unknown_or_invalid_tool_calls(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(content="", tool_calls=[{"name": "missing_tool", "args": {"question": "loop"}}], usage_metadata={}),
            SimpleNamespace(content="Recovered after unknown tool", usage_metadata={}),
        ]
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[])

        self.assertEqual(result["response"], "Recovered after unknown tool")
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_falls_back_when_provider_has_no_native_tool_calls(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content='TOOL_CALL {"name": "knowledge_search", "args": {"question": "What is ReAct?", "k": 2}}',
                usage_metadata={},
            ),
            SimpleNamespace(content="Final answer", usage_metadata={}),
        ]
        mock_create.return_value = llm

        tool = StubTool(
            "knowledge_search",
            MagicMock(return_value={"ok": True, "results": [{"content": "ReAct is a tool loop."}]}),
        )

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["response"], "Final answer")
        tool.handler.assert_called_once_with(question="What is ReAct?", k=2)
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_returns_text_and_raw_response_without_tools(self, mock_create):
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={})
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[])

        self.assertEqual(result["response"], "answer")
        self.assertIs(result["raw_response"], llm.invoke.return_value)
        llm.invoke.assert_called_once_with(["m1"])

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_aggregates_usage_across_model_turns(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"question": "What is ReAct?", "k": 2}}],
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "reasoning_tokens": 1},
            ),
            SimpleNamespace(
                content="Final answer",
                usage_metadata={"input_tokens": 4, "output_tokens": 6, "reasoning_tokens": 2},
            ),
        ]
        mock_create.return_value = llm

        tool = StubTool(
            "knowledge_search",
            MagicMock(return_value={"ok": True, "results": [{"content": "ReAct is a tool loop."}]}),
        )

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["usage"], {
            "input_tokens": 14,
            "output_tokens": 9,
            "analysis_tokens": 3,
        })

    @patch("services.agent_service.LLMFactory.create")
    def test_astream_only_yields_final_answer_chunks_after_internal_tool_loop(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"question": "What is ReAct?"}}],
                usage_metadata={"input_tokens": 12, "output_tokens": 5, "reasoning_tokens": 2},
            ),
            SimpleNamespace(content="Final answer", usage_metadata={}),
        ]
        mock_create.return_value = llm

        tool = StubTool("knowledge_search", MagicMock(return_value={"ok": True, "results": [{"content": "ctx"}]}))

        async def fake_astream(messages):
            yield SimpleNamespace(content="Final ")
            yield SimpleNamespace(content="answer", usage_metadata={"input_tokens": 999, "output_tokens": 999, "reasoning_tokens": 999})

        llm.astream = fake_astream

        service = AgentService(provider="siliconflow")

        async def collect_events():
            events = []
            async for event in service.astream(messages=["m1"], tools=[tool]):
                events.append(event)
            return events

        events = asyncio.run(collect_events())

        self.assertEqual(events, [
            {"chunk": "Final ", "text": "Final "},
            {"chunk": "answer", "text": "answer"},
            {"usage": {"input_tokens": 12, "output_tokens": 5, "analysis_tokens": 2}},
        ])
        self.assertEqual(llm.invoke.call_count, 2)

    @patch("services.agent_service.LLMFactory.create")
    def test_astream_aggregates_usage_across_internal_turns(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"question": "What is ReAct?"}}],
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "reasoning_tokens": 1},
            ),
            SimpleNamespace(content="Final answer", usage_metadata={"input_tokens": 4, "output_tokens": 6, "reasoning_tokens": 2}),
        ]
        mock_create.return_value = llm

        tool = StubTool("knowledge_search", MagicMock(return_value={"ok": True, "results": [{"content": "ctx"}]}))

        async def fake_astream(messages):
            yield SimpleNamespace(content="Final ")
            yield SimpleNamespace(content="answer")

        llm.astream = fake_astream

        service = AgentService(provider="siliconflow")

        async def collect_events():
            events = []
            async for event in service.astream(messages=["m1"], tools=[tool]):
                events.append(event)
            return events

        events = asyncio.run(collect_events())

        self.assertEqual(events[-1], {
            "usage": {"input_tokens": 14, "output_tokens": 9, "analysis_tokens": 3}
        })

    @patch("services.agent_service.LLMFactory.create")
    def test_astream_keeps_last_chunk_when_final_chunk_has_only_usage(self, mock_create):
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="answer", usage_metadata={"input_tokens": 10})
        mock_create.return_value = llm

        final_chunk = SimpleNamespace(content="", response_metadata={"token_usage": {"prompt_tokens": 999}})

        async def fake_astream(messages):
            self.assertEqual(messages, ["m1"])
            yield SimpleNamespace(content="hel")
            yield final_chunk

        llm.astream = fake_astream

        service = AgentService(provider="siliconflow")

        async def collect_events():
            events = []
            async for event in service.astream(messages=["m1"], tools=[]):
                events.append(event)
            return events

        events = asyncio.run(collect_events())

        self.assertEqual(events[0]["chunk"], "hel")
        self.assertEqual(events[1], {
            "usage": {"input_tokens": 10, "output_tokens": 0, "analysis_tokens": 0}
        })

        llm = MagicMock()
        mock_create.return_value = llm
        llm.invoke.return_value = SimpleNamespace(content="hello", usage_metadata={"input_tokens": 10, "output_tokens": 2})

        async def fake_astream(messages):
            self.assertEqual(messages, ["m1"])
            yield SimpleNamespace(content="hel")
            yield SimpleNamespace(content="lo", usage_metadata={"output_tokens": 999})

        llm.astream = fake_astream

        service = AgentService(provider="siliconflow")

        async def collect_events():
            events = []
            async for event in service.astream(messages=["m1"], tools=[]):
                events.append(event)
            return events

        events = asyncio.run(collect_events())

        self.assertEqual(events[0]["chunk"], "hel")
        self.assertEqual(events[0]["text"], "hel")
        self.assertEqual(events[1]["chunk"], "lo")
        self.assertEqual(events[1]["text"], "lo")
        self.assertEqual(events[2], {
            "usage": {"input_tokens": 10, "output_tokens": 2, "analysis_tokens": 0}
        })


if __name__ == "__main__":
    unittest.main()
