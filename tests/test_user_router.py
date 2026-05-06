import unittest

from models.chat_message import ChatMessage
from models.chat_session import ChatSession
from models.user import User


class TestChatSessionUserOwnership(unittest.TestCase):
    def test_chat_session_has_user_id_field(self):
        self.assertIn("user_id", ChatSession.__table__.columns)


class TestBigintModelIds(unittest.TestCase):
    def test_user_id_is_integer_column(self):
        self.assertEqual(User.__table__.columns["id"].type.python_type, int)

    def test_chat_session_user_id_is_integer_column(self):
        self.assertEqual(ChatSession.__table__.columns["user_id"].type.python_type, int)

    def test_chat_message_session_id_is_integer_column(self):
        self.assertEqual(ChatMessage.__table__.columns["session_id"].type.python_type, int)
