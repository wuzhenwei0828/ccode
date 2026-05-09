from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from routers.session import CreateSessionRequest, create_session, get_history, list_sessions
from routers.user import CreateUserRequest, UserResponse, create_user, list_users


class TestUserRouter(unittest.TestCase):
    @patch("routers.user.User")
    def test_list_users_returns_integer_ids(self, mock_user):
        db = MagicMock()
        mock_user.list_all.return_value = [SimpleNamespace(id=1, name="wzw", created_at=None)]

        result = list_users(db)

        self.assertEqual(result[0].id, 1)

    def test_user_response_rejects_string_id_after_recovery(self):
        with self.assertRaises(Exception):
            UserResponse(id="uuid-value", name="wzw", created_at=None)

    def test_create_user_returns_created_user(self):
        db = MagicMock()
        req = CreateUserRequest(name="Alice")

        def refresh_side_effect(user):
            user.id = 1
            user.created_at = None

        db.refresh.side_effect = refresh_side_effect

        result = create_user(req, db)

        self.assertEqual(result.name, "Alice")
        self.assertEqual(result.id, 1)


class TestSessionRouter(unittest.TestCase):
    @patch("routers.session.ChatSession")
    def test_list_sessions_filters_by_user_id(self, mock_chat_session):
        db = MagicMock()
        mock_session = MagicMock(id=1, title="T1", message_count=0, summary_sequence=0)
        mock_chat_session.list_by_user.return_value = [mock_session]

        result = list_sessions(1, db)

        mock_chat_session.list_by_user.assert_called_once_with(db, 1)
        self.assertEqual(len(result), 1)

    @patch("routers.session.ChatSession")
    def test_list_sessions_returns_integer_session_id(self, mock_chat_session):
        db = MagicMock()
        mock_session = MagicMock(id=101, title="Recovered", message_count=58, summary_sequence=14)
        mock_chat_session.list_by_user.return_value = [mock_session]

        result = list_sessions(1, db)

        self.assertEqual(result[0].id, 101)

    def test_create_session_does_not_require_explicit_string_id(self):
        db = MagicMock()
        req = CreateSessionRequest(title="Chat", user_id=1)

        def add_side_effect(session):
            self.assertIsNone(getattr(session, "id", None))
            self.assertEqual(session.user_id, 1)

        def refresh_side_effect(session):
            session.id = 101
            session.message_count = 0
            session.summary_sequence = 0

        db.add.side_effect = add_side_effect
        db.refresh.side_effect = refresh_side_effect

        result = create_session(req, db)

        self.assertEqual(result.id, 101)


class TestBigintRouterIds(unittest.TestCase):
    def test_create_session_request_accepts_integer_user_id(self):
        req = CreateSessionRequest(title="Chat", user_id=123)
        self.assertEqual(req.user_id, 123)

    def test_user_response_accepts_integer_id(self):
        from routers.user import UserResponse

        resp = UserResponse(id=1, name="Alice", created_at=None)
        self.assertEqual(resp.id, 1)
