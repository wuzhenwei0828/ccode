import unittest
from unittest.mock import MagicMock, patch

from routers.session import CreateSessionRequest, create_session, get_history, list_sessions
from routers.user import CreateUserRequest, create_user


class TestUserRouter(unittest.TestCase):
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


class TestBigintRouterIds(unittest.TestCase):
    def test_create_session_request_accepts_integer_user_id(self):
        req = CreateSessionRequest(title="Chat", user_id=123)
        self.assertEqual(req.user_id, 123)

    def test_user_response_accepts_integer_id(self):
        from routers.user import UserResponse

        resp = UserResponse(id=1, name="Alice", created_at=None)
        self.assertEqual(resp.id, 1)
