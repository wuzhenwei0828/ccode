import unittest
from unittest.mock import MagicMock, patch

from routers.chat import ChatRequest


class TestChatRequestBigintContract(unittest.TestCase):
    def test_chat_request_accepts_integer_session_id(self):
        req = ChatRequest(session_id=2, message="hello", use_rag=False, provider="siliconflow", rag_k=4)
        self.assertEqual(req.session_id, 2)


if __name__ == "__main__":
    unittest.main()
