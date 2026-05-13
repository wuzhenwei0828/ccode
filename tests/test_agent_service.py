import asyncio
import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from services.agent_service import AgentService
from services.agent_tools import build_chat_tools


class StubTool:
    def __init__(self, name, handler):
        self.name = name
        self.handler = handler


class TestAgentService(unittest.TestCase):
    @patch("services.agent_tools.RAGService", create=True)
    def test_build_chat_tools_registers_knowledge_current_time_and_web_search_tools(self, mock_rag_service_cls):
        mock_rag_service_cls.return_value.query.return_value = []

        tools = build_chat_tools()

        tool_names = [tool.name for tool in tools]
        self.assertIn("knowledge_search", tool_names)
        self.assertIn("current_time", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertLess(tool_names.index("knowledge_search"), tool_names.index("current_time"))
        self.assertLess(tool_names.index("current_time"), tool_names.index("web_search"))
        knowledge_tool = next(tool for tool in tools if tool.name == "knowledge_search")
        current_time_tool = next(tool for tool in tools if tool.name == "current_time")
        web_search_tool = next(tool for tool in tools if tool.name == "web_search")
        self.assertIn("knowledge", knowledge_tool.description.lower())
        self.assertTrue(callable(knowledge_tool.handler))
        self.assertTrue(current_time_tool.description)
        self.assertTrue(callable(current_time_tool.handler))
        self.assertTrue(web_search_tool.description)
        self.assertTrue(callable(web_search_tool.handler))

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

    @patch("services.agent_tools.DuckDuckGoSearchResults")
    @patch("services.agent_tools.RAGService", create=True)
    def test_web_search_tool_returns_serializable_results(self, mock_rag_service_cls, mock_ddg_cls):
        mock_rag_service_cls.return_value.query.return_value = []
        ddg_tool = MagicMock()
        ddg_tool.invoke.return_value = [
            {
                "title": "Result A",
                "snippet": "alpha" * 80,
                "link": "https://example.com/a",
                "source": "ignored",
            },
            {
                "title": "Result B",
                "body": "beta",
                "link": "https://example.com/b",
                "score": 0.9,
            },
        ]
        mock_ddg_cls.return_value = ddg_tool

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "web_search")

        result = tool.handler(query="langchain react", limit=2)

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0], {
            "title": "Result A",
            "snippet": ("alpha" * 80)[:240],
            "url": "https://example.com/a",
        })
        self.assertEqual(result["results"][1], {
            "title": "Result B",
            "snippet": "beta",
            "url": "https://example.com/b",
        })
        self.assertEqual(set(result["results"][0].keys()), {"title", "snippet", "url"})
        mock_ddg_cls.assert_called_once_with(max_results=5, output_format="list")
        ddg_tool.invoke.assert_called_once_with("langchain react")

    @patch("services.agent_tools.DuckDuckGoSearchResults")
    @patch("services.agent_tools.RAGService", create=True)
    def test_web_search_tool_normalizes_duckduckgo_result_fields(self, mock_rag_service_cls, mock_ddg_cls):
        mock_rag_service_cls.return_value.query.return_value = []
        ddg_tool = MagicMock()
        ddg_tool.invoke.return_value = [
            {
                "title": "Summary First",
                "summary": "fresh summary text",
                "body": "older body text",
                "link": "https://example.com/live",
            }
        ]
        mock_ddg_cls.return_value = ddg_tool

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "web_search")

        result = tool.handler(query="duckduckgo summary", limit=1)

        self.assertEqual(result, {
            "ok": True,
            "count": 1,
            "results": [{
                "title": "Summary First",
                "snippet": "fresh summary text",
                "url": "https://example.com/live",
            }],
        })

    @patch("services.agent_tools._run_web_search")
    @patch("services.agent_tools.RAGService", create=True)
    def test_web_search_tool_rejects_blank_query(self, mock_rag_service_cls, mock_run_web_search):
        mock_rag_service_cls.return_value.query.return_value = []

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "web_search")

        result = tool.handler(query="   ", limit=2)

        self.assertEqual(result, {
            "ok": False,
            "error": {
                "type": "web_search_error",
                "message": "query must not be blank",
            },
        })
        mock_run_web_search.assert_not_called()

    @patch("services.agent_tools._run_web_search")
    @patch("services.agent_tools.RAGService", create=True)
    def test_web_search_tool_rejects_invalid_limit(self, mock_rag_service_cls, mock_run_web_search):
        mock_rag_service_cls.return_value.query.return_value = []

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "web_search")

        for invalid_limit in (0, -1, True, "2"):
            with self.subTest(limit=invalid_limit):
                result = tool.handler(query="langchain", limit=invalid_limit)
                self.assertEqual(result, {
                    "ok": False,
                    "error": {
                        "type": "web_search_error",
                        "message": "limit must be a positive integer",
                    },
                })

        mock_run_web_search.assert_not_called()

    @patch("services.agent_tools.DuckDuckGoSearchResults")
    @patch("services.agent_tools.RAGService", create=True)
    def test_web_search_tool_returns_empty_results_when_search_finds_nothing(self, mock_rag_service_cls, mock_ddg_cls):
        mock_rag_service_cls.return_value.query.return_value = []
        ddg_tool = MagicMock()
        ddg_tool.invoke.return_value = []
        mock_ddg_cls.return_value = ddg_tool

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "web_search")

        result = tool.handler(query="no matches", limit=5)

        self.assertEqual(result, {
            "ok": True,
            "count": 0,
            "results": [],
        })

    @patch("services.agent_tools._run_web_search")
    @patch("services.agent_tools.RAGService", create=True)
    def test_web_search_tool_returns_structured_error_when_search_raises(self, mock_rag_service_cls, mock_run_web_search):
        mock_rag_service_cls.return_value.query.return_value = []
        mock_run_web_search.side_effect = RuntimeError("search backend unavailable")

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "web_search")

        with self.assertLogs("services.agent_tools", level=logging.ERROR) as captured_logs:
            result = tool.handler(query="langchain", limit=3)

        self.assertEqual(result, {
            "ok": False,
            "error": {
                "type": "web_search_error",
                "message": "web search backend unavailable",
            },
        })
        self.assertEqual(len(captured_logs.output), 1)
        self.assertIn("web_search backend failure query=langchain", captured_logs.output[0])
        self.assertIn("RuntimeError: search backend unavailable", captured_logs.output[0])

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
        self.assertTrue({"knowledge_search", "current_time", "web_search"}.issubset(tool_names))
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
    def test_invoke_replays_assistant_tool_call_and_tool_observation(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        first_response = AIMessage(
            content="Thinking",
            tool_calls=[{"name": "knowledge_search", "args": {"question": "What is ReAct?", "k": 2}, "id": "call-1"}],
        )
        first_response.usage_metadata = {}
        llm.invoke.side_effect = [
            first_response,
            SimpleNamespace(content="Final answer", usage_metadata={}),
        ]
        mock_create.return_value = llm

        tool_result = {"ok": True, "results": [{"content": "ReAct is a tool loop."}]}
        tool = StubTool("knowledge_search", MagicMock(return_value=tool_result))

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["response"], "Final answer")
        tool.handler.assert_called_once_with(question="What is ReAct?", k=2)
        second_call_messages = llm.invoke.call_args_list[1].args[0]
        self.assertEqual(second_call_messages[0], "m1")
        self.assertIs(second_call_messages[1], first_response)
        self.assertIsInstance(second_call_messages[2], ToolMessage)
        self.assertEqual(second_call_messages[2].tool_call_id, "call-1")
        self.assertEqual(second_call_messages[2].content, json.dumps(tool_result, ensure_ascii=False))

    @patch("services.agent_service.LLMFactory.create")
    def test_invoke_returns_fallback_when_final_response_is_empty(self, mock_create):
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="   ", usage_metadata={})
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[])

        self.assertEqual(result["response"], service.FALLBACK_RESPONSE)

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
    def test_invoke_replays_text_tool_call_fallback_as_ai_message(self, mock_create):
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

        tool_result = {"ok": True, "results": [{"content": "ReAct is a tool loop."}]}
        tool = StubTool("knowledge_search", MagicMock(return_value=tool_result))

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["m1"], tools=[tool])

        self.assertEqual(result["response"], "Final answer")
        tool.handler.assert_called_once_with(question="What is ReAct?", k=2)
        second_call_messages = llm.invoke.call_args_list[1].args[0]
        self.assertEqual(second_call_messages[0], "m1")
        self.assertIsInstance(second_call_messages[1], AIMessage)
        self.assertEqual(
            second_call_messages[1].content,
            'TOOL_CALL {"name": "knowledge_search", "args": {"question": "What is ReAct?", "k": 2}}',
        )
        self.assertEqual(len(second_call_messages[1].tool_calls), 1)
        self.assertEqual(second_call_messages[1].tool_calls[0]["name"], "knowledge_search")
        self.assertEqual(
            second_call_messages[1].tool_calls[0]["args"],
            {"question": "What is ReAct?", "k": 2},
        )
        self.assertEqual(second_call_messages[1].tool_calls[0]["id"], "knowledge_search")
        self.assertIsInstance(second_call_messages[2], ToolMessage)
        self.assertEqual(second_call_messages[2].tool_call_id, "knowledge_search")
        self.assertEqual(second_call_messages[2].content, json.dumps(tool_result, ensure_ascii=False))

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
    def test_astream_uses_post_loop_messages_for_final_stream(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"id": "call-1", "name": "knowledge_search", "args": {"question": "What is ReAct?"}}],
                usage_metadata={"input_tokens": 8, "output_tokens": 2, "reasoning_tokens": 1},
            ),
            SimpleNamespace(
                content="Final answer",
                usage_metadata={"input_tokens": 3, "output_tokens": 4, "reasoning_tokens": 1},
            ),
        ]
        mock_create.return_value = llm

        tool = StubTool("knowledge_search", MagicMock(return_value={"ok": True, "results": [{"content": "ctx"}]}))

        async def fake_astream(messages):
            self.assertEqual(messages[0].content, "What is ReAct?")
            self.assertIsInstance(messages[1], AIMessage)
            self.assertEqual(messages[1].tool_calls[0]["id"], "call-1")
            self.assertIsInstance(messages[-1], ToolMessage)
            self.assertEqual(messages[-1].tool_call_id, "call-1")
            yield SimpleNamespace(content="Final ")
            yield SimpleNamespace(content="answer")

        llm.astream = fake_astream

        service = AgentService(provider="siliconflow")

        async def collect_events():
            return [event async for event in service.astream(messages=[HumanMessage(content="What is ReAct?")], tools=[tool])]

        events = asyncio.run(collect_events())

        self.assertEqual(events, [
            {"chunk": "Final ", "text": "Final "},
            {"chunk": "answer", "text": "answer"},
            {"usage": {"input_tokens": 11, "output_tokens": 6, "analysis_tokens": 2}},
        ])

    @patch("services.agent_service.LLMFactory.create")
    def test_astream_emits_fallback_when_tool_loop_is_exhausted(self, mock_create):
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"question": "loop"}}],
                usage_metadata={},
            ),
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"question": "loop"}}],
                usage_metadata={},
            ),
        ]
        mock_create.return_value = llm

        tool = StubTool("knowledge_search", MagicMock(return_value={"ok": True, "results": []}))
        service = AgentService(provider="siliconflow")

        async def collect_events():
            return [event async for event in service.astream(messages=["m1"], tools=[tool], max_iterations=2)]

        events = asyncio.run(collect_events())

        self.assertEqual(events, [
            {"chunk": service.FALLBACK_RESPONSE, "text": service.FALLBACK_RESPONSE},
            {"usage": {"input_tokens": 0, "output_tokens": 0, "analysis_tokens": 0}},
        ])

    @patch("services.agent_service.LLMFactory.create")
    def test_astream_emits_normalized_final_text_when_stream_has_no_chunks(self, mock_create):
        llm = MagicMock()
        llm.invoke.return_value = SimpleNamespace(content="   ", usage_metadata={"input_tokens": 7, "output_tokens": 1})
        mock_create.return_value = llm

        async def fake_astream(messages):
            self.assertEqual(messages, ["m1"])
            if False:
                yield None

        llm.astream = fake_astream

        service = AgentService(provider="siliconflow")

        async def collect_events():
            events = []
            async for event in service.astream(messages=["m1"], tools=[]):
                events.append(event)
            return events

        events = asyncio.run(collect_events())

        self.assertEqual(events, [
            {"chunk": service.FALLBACK_RESPONSE, "text": service.FALLBACK_RESPONSE},
            {"usage": {"input_tokens": 7, "output_tokens": 1, "analysis_tokens": 0}},
        ])

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
