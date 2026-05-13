import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from chains.chat_chain import ChatChain
from services.agent_tools import build_chat_tools


class TestChatChainSummaryPrompt(unittest.TestCase):
    def test_build_chat_tools_registers_knowledge_current_time_and_web_search_tools(self):
        tools = build_chat_tools()

        tool_names = [tool.name for tool in tools]
        self.assertIn("knowledge_search", tool_names)
        self.assertIn("current_time", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertLess(tool_names.index("knowledge_search"), tool_names.index("current_time"))
        self.assertLess(tool_names.index("current_time"), tool_names.index("web_search"))

    @patch("chains.chat_chain.AgentService")
    def test_invoke_uses_react_system_prompt_and_preserves_message_order(self, mock_agent_service_cls):
        history = [HumanMessage(content="msg1")]
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", history)

        agent_service = MagicMock()
        agent_service.invoke.return_value = {
            "response": "answer",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "analysis_tokens": 0,
            },
        }
        mock_agent_service_cls.return_value = agent_service

        chain = ChatChain(memory, provider="siliconflow")
        chain.invoke("sess-1", "hi")

        memory.get_context_parts.assert_called_once_with("sess-1")
        prompt_messages = agent_service.invoke.call_args.kwargs["messages"]
        self.assertIsInstance(prompt_messages[0], SystemMessage)
        self.assertIn("你是一个会使用工具完成任务的 agent", prompt_messages[0].content)
        self.assertIn("如果需要工具，必须优先调用工具", prompt_messages[0].content)
        self.assertIn("调用工具后，必须基于工具结果继续完成任务", prompt_messages[0].content)
        self.assertIn("不要向用户暴露内部推理过程", prompt_messages[0].content)
        self.assertEqual(prompt_messages[1].content, "以下是历史对话摘要，仅供参考：\nsummary text")
        self.assertEqual(prompt_messages[2].content, "msg1")
        self.assertEqual(prompt_messages[3].content, "hi")

    @patch("chains.chat_chain.AgentService")
    def test_invoke_accepts_integer_session_id(self, mock_agent_service_cls):
        history = [HumanMessage(content="hi")]
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", history)

        agent_service = MagicMock()
        agent_service.invoke.return_value = {
            "response": "answer",
            "raw_response": SimpleNamespace(usage_metadata={}),
        }
        mock_agent_service_cls.return_value = agent_service

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

    @patch("chains.chat_chain.AgentService")
    def test_invoke_passes_runtime_web_search_tool_to_agent_service(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        agent_service = MagicMock()
        agent_service.invoke.return_value = {
            "response": "answer",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "analysis_tokens": 0,
            },
        }
        mock_agent_service_cls.return_value = agent_service

        chain = ChatChain(memory, provider="siliconflow")
        chain.invoke("sess-1", "hello")

        tools = agent_service.invoke.call_args.kwargs["tools"]
        tool_names = {tool.name for tool in tools}
        self.assertIn("web_search", tool_names)


class TestChatChainUsage(unittest.TestCase):
    @patch("chains.chat_chain.AgentService")
    def test_invoke_returns_aggregated_usage_from_agent_loop(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        agent_service = MagicMock()
        agent_service.invoke.return_value = {
            "response": "answer",
            "usage": {
                "input_tokens": 14,
                "output_tokens": 9,
                "analysis_tokens": 3,
            },
            "raw_response": SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1}),
        }
        mock_agent_service_cls.return_value = agent_service

        chain = ChatChain(memory, provider="siliconflow")
        result = chain.invoke("sess-1", "hello")

        self.assertEqual(result, {
            "response": "answer",
            "usage": {
                "input_tokens": 14,
                "output_tokens": 9,
                "analysis_tokens": 3,
            },
        })

    @patch("chains.chat_chain.AgentService")
    def test_invoke_persists_only_user_and_final_assistant_after_tool_loop(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        agent_service = MagicMock()
        agent_service.invoke.return_value = {
            "response": "final answer",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 7,
                "analysis_tokens": 2,
            },
        }
        mock_agent_service_cls.return_value = agent_service

        chain = ChatChain(memory, provider="siliconflow")
        result = chain.invoke("sess-1", "hello")

        self.assertEqual(result, {
            "response": "final answer",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 7,
                "analysis_tokens": 2,
            },
        })
        self.assertEqual(memory.add_message.call_count, 2)
        memory.add_message.assert_any_call("sess-1", "user", "hello")
        memory.add_message.assert_any_call("sess-1", "assistant", "final answer")

    @patch("chains.chat_chain.AgentService")
    def test_astream_hides_internal_tool_events(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        agent_service = MagicMock()

        async def fake_astream(messages, tools):
            yield {"type": "tool_call", "tool": "knowledge_search"}
            yield {"type": "tool_result", "tool": "knowledge_search", "result": {"ok": True}}
            yield {"chunk": "final ", "text": "final "}
            yield {"chunk": "answer", "text": "answer"}
            yield {
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 6,
                    "analysis_tokens": 3,
                }
            }

        agent_service.astream = fake_astream
        mock_agent_service_cls.return_value = agent_service

        chain = ChatChain(memory, provider="siliconflow")

        async def collect_events():
            events = []
            async for event in chain.astream("sess-1", "hello"):
                events.append(event)
            return events

        result = asyncio.run(collect_events())

        self.assertEqual(result, [
            {"chunk": "final "},
            {"chunk": "answer"},
            {"usage": {"input_tokens": 12, "output_tokens": 6, "analysis_tokens": 3}},
        ])
        self.assertEqual(memory.add_message.call_count, 2)
        memory.add_message.assert_any_call("sess-1", "user", "hello")
        memory.add_message.assert_any_call("sess-1", "assistant", "final answer")

    @patch("chains.chat_chain.AgentService")
    def test_invoke_returns_response_and_usage(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        response = SimpleNamespace(content="answer", usage_metadata={"input_tokens": 10, "output_tokens": 4})
        agent_service = MagicMock()
        agent_service.invoke.return_value = {
            "response": "answer",
            "raw_response": response,
        }
        mock_agent_service_cls.return_value = agent_service

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

    @patch("chains.chat_chain.AgentService")
    @patch("chains.chat_chain.build_chat_tools", create=True)
    def test_invoke_routes_through_agent_service_with_tools(self, mock_build_chat_tools, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])
        mock_build_chat_tools.return_value = []

        agent_service = MagicMock()
        agent_service.invoke.return_value = {
            "response": "answer",
            "raw_response": SimpleNamespace(usage_metadata={}),
        }
        mock_agent_service_cls.return_value = agent_service

        chain = ChatChain(memory, provider="siliconflow")
        result = chain.invoke("sess-1", "hello")

        self.assertEqual(result["response"], "answer")
        mock_build_chat_tools.assert_called_once_with()
        mock_agent_service_cls.assert_called_once_with(provider="siliconflow")
        agent_service.invoke.assert_called_once()

    @patch("chains.chat_chain.AgentService")
    @patch("chains.chat_chain.build_chat_tools", create=True)
    def test_astream_routes_through_agent_service_with_tools(self, mock_build_chat_tools, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])
        mock_build_chat_tools.return_value = []

        agent_service = MagicMock()

        async def fake_astream(messages, tools):
            yield {"chunk": "hel", "text": "hel"}
            yield {"chunk": "lo", "text": "lo"}
            yield {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}}

        agent_service.astream = fake_astream
        mock_agent_service_cls.return_value = agent_service

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
        mock_build_chat_tools.assert_called_once_with()
        mock_agent_service_cls.assert_called_once_with(provider="siliconflow")

    @patch("chains.chat_chain.AgentService")
    def test_astream_yields_chunks_and_usage_event(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        agent_service = MagicMock()

        async def fake_astream(messages, tools):
            yield {"chunk": "hel", "text": "hel"}
            yield {"chunk": "lo", "text": "lo"}
            yield {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}}

        agent_service.astream = fake_astream
        mock_agent_service_cls.return_value = agent_service

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

    @patch("chains.chat_chain.AgentService")
    def test_astream_returns_zero_usage_when_no_chunks_are_emitted(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        agent_service = MagicMock()

        async def fake_astream(messages, tools):
            if False:
                yield None

        agent_service.astream = fake_astream
        mock_agent_service_cls.return_value = agent_service

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


    @patch("chains.chat_chain.AgentService")
    def test_astream_uses_metadata_only_final_chunk_for_usage(self, mock_agent_service_cls):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", [])

        final_usage = {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}
        agent_service = MagicMock()

        async def fake_astream(messages, tools):
            yield {"chunk": "hel", "text": "hel"}
            yield {"usage": final_usage}

        agent_service.astream = fake_astream
        mock_agent_service_cls.return_value = agent_service

        chain = ChatChain(memory, provider="siliconflow")

        async def collect_events():
            events = []
            async for event in chain.astream("sess-1", "hello"):
                events.append(event)
            return events

        result = asyncio.run(collect_events())

        self.assertEqual(result, [
            {"chunk": "hel"},
            {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}},
        ])

