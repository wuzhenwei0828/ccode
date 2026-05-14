import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from services.agent_service import AgentService
from services.planners.react_planner import ReActPlanner
from services.tools.agent_tools import build_chat_tools


class TestAgentService(unittest.TestCase):
    def test_service_uses_react_planner_by_default(self):
        service = AgentService(provider="siliconflow")

        self.assertIsInstance(service._planner, ReActPlanner)
        self.assertEqual(service._planner.provider, "siliconflow")

    @patch("services.agent_service.LLMFactory.create")
    @patch("services.agent_service.ReActPlanner")
    def test_invoke_delegates_structured_arguments_to_planner(self, mock_planner_cls, mock_create):
        planner = MagicMock()
        planner.invoke.return_value = {
            "response": "answer",
            "raw_response": object(),
            "usage": {"input_tokens": 1, "output_tokens": 2, "analysis_tokens": 3},
        }
        mock_planner_cls.return_value = planner
        tools = [MagicMock(name="knowledge_search")]
        llm = MagicMock()
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")
        result = service.invoke(messages=["history", "question"], tools=tools, max_iterations=2)

        self.assertEqual(result, planner.invoke.return_value)
        mock_planner_cls.assert_called_once_with(provider="siliconflow")
        mock_create.assert_called_once_with("siliconflow")
        planner.invoke.assert_called_once_with(
            llm=llm,
            summary="",
            history=["history"],
            message="question",
            tools=tools,
            context=None,
            max_iterations=2,
        )

    @patch("services.agent_service.LLMFactory.create")
    @patch("services.agent_service.ReActPlanner")
    def test_astream_delegates_structured_arguments_to_planner(self, mock_planner_cls, mock_create):
        planner = MagicMock()

        async def fake_astream(**kwargs):
            yield {"chunk": "hello"}
            yield {"usage": {"input_tokens": 1, "output_tokens": 1, "analysis_tokens": 0}}

        planner.astream.side_effect = fake_astream
        mock_planner_cls.return_value = planner
        llm = MagicMock()
        mock_create.return_value = llm

        service = AgentService(provider="siliconflow")

        async def collect_events():
            return [event async for event in service.astream(messages=["history", "question"], tools=[], max_iterations=3)]

        events = asyncio.run(collect_events())

        self.assertEqual(events, [
            {"chunk": "hello"},
            {"usage": {"input_tokens": 1, "output_tokens": 1, "analysis_tokens": 0}},
        ])
        mock_create.assert_called_once_with("siliconflow")
        planner.astream.assert_called_once_with(
            llm=llm,
            summary="",
            history=["history"],
            message="question",
            tools=[],
            context=None,
            max_iterations=3,
        )

    @patch("services.tools.agent_tools.RAGService")
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

        self.assertTrue(result["ok"])
        self.assertIsInstance(result["current_time"], str)

    @patch("services.tools.agent_tools.RAGService")
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

        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["results"]), 2)
        self.assertIsInstance(result["results"][0]["content"], str)
        self.assertLessEqual(len(result["results"][0]["content"]), 240)
        self.assertEqual(result["results"][0]["source"], "doc-a")
        self.assertEqual(result["results"][0]["metadata"], {"kb_id": "kb-1"})

    @patch("services.tools.agent_tools.DuckDuckGoSearchResults")
    @patch("services.tools.agent_tools.RAGService")
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

        self.assertEqual(result["count"], 2)
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
        mock_ddg_cls.assert_called_once_with(max_results=5, output_format="list")
        ddg_tool.invoke.assert_called_once_with("langchain react")

    @patch("services.tools.agent_tools.DuckDuckGoSearchResults")
    @patch("services.tools.agent_tools.RAGService")
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

    @patch("services.tools.agent_tools._run_web_search")
    @patch("services.tools.agent_tools.RAGService")
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

    @patch("services.tools.agent_tools._run_web_search")
    @patch("services.tools.agent_tools.RAGService")
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

    @patch("services.tools.agent_tools.DuckDuckGoSearchResults")
    @patch("services.tools.agent_tools.RAGService")
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

    @patch("services.tools.agent_tools._run_web_search")
    @patch("services.tools.agent_tools.RAGService")
    def test_web_search_tool_returns_structured_error_when_search_raises(self, mock_rag_service_cls, mock_run_web_search):
        mock_rag_service_cls.return_value.query.return_value = []
        mock_run_web_search.side_effect = RuntimeError("search backend unavailable")

        tools = build_chat_tools()
        tool = next(tool for tool in tools if tool.name == "web_search")

        with self.assertLogs("services.tools.agent_tools", level=logging.ERROR) as captured_logs:
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


if __name__ == "__main__":
    unittest.main()
