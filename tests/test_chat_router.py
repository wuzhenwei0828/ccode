import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app


class TestChatRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("routers.chat.ChatChain")
    def test_chat_returns_usage(self, mock_chain_cls):
        chain = MagicMock()
        chain.invoke.return_value = {
            "response": "answer",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "analysis_tokens": 2,
            },
        }
        mock_chain_cls.return_value = chain

        response = self.client.post("/api/chat", json={
            "session_id": 1,
            "message": "hello",
            "use_rag": False,
            "provider": "siliconflow",
            "rag_k": 4,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "session_id": 1,
            "response": "answer",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "analysis_tokens": 2,
            },
        })

    @patch("routers.chat.RAGChain")
    def test_chat_returns_usage_for_rag_requests(self, mock_chain_cls):
        chain = MagicMock()
        chain.invoke.return_value = {
            "response": "rag-answer",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "analysis_tokens": 3,
            },
        }
        mock_chain_cls.return_value = chain

        response = self.client.post("/api/chat", json={
            "session_id": 1,
            "message": "hello",
            "use_rag": True,
            "provider": "siliconflow",
            "rag_k": 4,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["usage"], {
            "input_tokens": 12,
            "output_tokens": 6,
            "analysis_tokens": 3,
        })

    @patch("routers.chat.ChatChain")
    def test_chat_stream_keeps_public_sse_contract_when_agent_uses_tools(self, mock_chain_cls):
        async def fake_stream(*args, **kwargs):
            yield {"chunk": "final "}
            yield {"chunk": "answer"}
            yield {"usage": {"input_tokens": 12, "output_tokens": 6, "analysis_tokens": 3}}

        chain = MagicMock()
        chain.astream = fake_stream
        mock_chain_cls.return_value = chain

        with self.client.stream("POST", "/api/chat/stream", json={
            "session_id": 1,
            "message": "hello",
            "use_rag": False,
            "provider": "siliconflow",
            "rag_k": 4,
        }) as response:
            body = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"chunk": "final "}', body)
        self.assertIn('data: {"chunk": "answer"}', body)
        self.assertIn('data: {"usage": {"input_tokens": 12, "output_tokens": 6, "analysis_tokens": 3}}', body)
        self.assertNotIn("tool_call", body)
        self.assertNotIn("tool_result", body)
        self.assertTrue(body.strip().endswith('data: [DONE]'))

    @patch("routers.chat.ChatChain")
    def test_chat_stream_emits_usage_before_done(self, mock_chain_cls):
        async def fake_stream(*args, **kwargs):
            yield {"chunk": "hel"}
            yield {"chunk": "lo"}
            yield {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}}

        chain = MagicMock()
        chain.astream = fake_stream
        mock_chain_cls.return_value = chain

        with self.client.stream("POST", "/api/chat/stream", json={
            "session_id": 1,
            "message": "hello",
            "use_rag": False,
            "provider": "siliconflow",
            "rag_k": 4,
        }) as response:
            body = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"chunk": "hel"}', body)
        self.assertIn('data: {"chunk": "lo"}', body)
        self.assertIn('data: {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}}', body)
        self.assertTrue(body.strip().endswith('data: [DONE]'))


    @patch("routers.chat.RAGChain")
    def test_chat_stream_uses_rag_chain_for_rag_requests(self, mock_chain_cls):
        async def fake_stream(*args, **kwargs):
            yield {"chunk": "rag"}
            yield {"usage": {"input_tokens": 12, "output_tokens": 6, "analysis_tokens": 3}}

        chain = MagicMock()
        chain.astream = fake_stream
        mock_chain_cls.return_value = chain

        with self.client.stream("POST", "/api/chat/stream", json={
            "session_id": 1,
            "message": "hello",
            "use_rag": True,
            "provider": "siliconflow",
            "rag_k": 4,
        }) as response:
            body = "".join(chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn('data: {"chunk": "rag"}', body)
        self.assertIn('data: {"usage": {"input_tokens": 12, "output_tokens": 6, "analysis_tokens": 3}}', body)
        self.assertTrue(body.strip().endswith('data: [DONE]'))

