"""Tests for MemoryService with mocked MySQL (SessionLocal)."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from services.memory.memory_service import MemoryService


class TestMemoryService(unittest.TestCase):
    """Tests for MemoryService dual-layer memory."""

    def setUp(self):
        """Create a fresh MemoryService instance for each test."""
        self.service = MemoryService()

    def _mock_db_session(self):
        """Create a mock DB session with common query chain set up."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        return mock_session, mock_filter

    # ------------------------------------------------------------------
    # get_context
    # ------------------------------------------------------------------

    @patch("services.memory_service.get_settings")
    def test_get_context_returns_all_messages_when_under_window(self, mock_get_settings):
        """get_context returns all messages when count < window size."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 5

        session_id = "sess-1"
        # Add 3 messages (under window of 5)
        self.service._short_term[session_id] = [
            HumanMessage(content="msg1"),
            HumanMessage(content="msg2"),
            HumanMessage(content="msg3"),
        ]

        result = self.service.get_context(session_id)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], HumanMessage)
        self.assertEqual(result[0].content, "msg1")

    @patch("services.memory_service.get_settings")
    def test_get_context_returns_sliding_window(self, mock_get_settings):
        """get_context returns only the most recent N messages (window size)."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-2"
        # Add 5 messages, window is 3
        self.service._short_term[session_id] = [
            HumanMessage(content=f"msg{i}") for i in range(1, 6)
        ]

        result = self.service.get_context(session_id)
        self.assertEqual(len(result), 3)
        # Should return the last 3 messages
        self.assertEqual(result[0].content, "msg3")
        self.assertEqual(result[1].content, "msg4")
        self.assertEqual(result[2].content, "msg5")

    @patch("services.memory_service.get_settings")
    def test_get_context_empty_session(self, mock_get_settings):
        """get_context returns empty list for unknown session."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10

        result = self.service.get_context("nonexistent-session")
        self.assertEqual(result, [])

    # ------------------------------------------------------------------
    # add_message
    # ------------------------------------------------------------------

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_user_to_short_term(self, mock_get_settings, mock_session_local):
        """add_message stores a user message correctly in short-term cache."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.first.return_value = None  # no existing messages

        session_id = "sess-3"
        self.service.add_message(session_id, "user", "hello", db=mock_db)

        messages = self.service._short_term[session_id]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertEqual(messages[0].content, "hello")

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_assistant_to_short_term(self, mock_get_settings, mock_session_local):
        """add_message stores an assistant message correctly in short-term cache."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.first.return_value = None

        session_id = "sess-4"
        self.service.add_message(session_id, "assistant", "hi there", db=mock_db)

        messages = self.service._short_term[session_id]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], AIMessage)
        self.assertEqual(messages[0].content, "hi there")

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_persists_to_mysql(self, mock_get_settings, mock_session_local):
        """add_message creates and commits a ChatMessage in MySQL."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        # Chain: query -> filter -> order_by -> first
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.first.return_value = None  # no existing messages, sequence starts at 0

        session_id = "sess-5"
        self.service.add_message(session_id, "user", "persist me", db=mock_db)

        # Verify add() was called on the session (a ChatMessage was added)
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

        # Verify the saved message has correct fields
        saved_msg = mock_db.add.call_args[0][0]
        self.assertEqual(saved_msg.session_id, session_id)
        self.assertEqual(saved_msg.role, "user")
        self.assertEqual(saved_msg.content, "persist me")
        self.assertEqual(saved_msg.sequence, 0)

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_sequence_increment(self, mock_get_settings, mock_session_local):
        """add_message increments sequence number correctly."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        session_id = "sess-6"

        # Setup mock chain: query -> filter -> order_by -> first
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order

        # First message: max_seq is None, so sequence = 0
        mock_order.first.return_value = None
        self.service.add_message(session_id, "user", "first", db=mock_db)
        first_msg = mock_db.add.call_args[0][0]
        self.assertEqual(first_msg.sequence, 0)

        # Reset mock add calls
        mock_db.add.reset_mock()
        mock_db.commit.reset_mock()

        # Second message: max_seq is 0, so sequence = 1
        mock_order.first.return_value = (0,)
        self.service.add_message(session_id, "assistant", "second", db=mock_db)
        second_msg = mock_db.add.call_args[0][0]
        self.assertEqual(second_msg.sequence, 1)

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_multiple_messages_same_session(self, mock_get_settings, mock_session_local):
        """add_message accumulates multiple messages in short-term cache."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order

        session_id = "sess-7"
        mock_order.first.return_value = None

        self.service.add_message(session_id, "user", "q1", db=mock_db)
        # Update max_seq for second message
        mock_filter.first.return_value = (0,)
        self.service.add_message(session_id, "assistant", "a1", db=mock_db)

        messages = self.service._short_term[session_id]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)

    # ------------------------------------------------------------------
    # load_history
    # ------------------------------------------------------------------

    @patch("services.memory_service.SessionLocal")
    def test_load_history_returns_messages_in_order(self, mock_session_local):
        """load_history returns messages ordered by sequence ascending."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        # Create mock ChatMessage objects
        mock_msg1 = MagicMock()
        mock_msg1.role = "user"
        mock_msg1.content = "hello"
        mock_msg1.sequence = 0

        mock_msg2 = MagicMock()
        mock_msg2.role = "assistant"
        mock_msg2.content = "hi back"
        mock_msg2.sequence = 1

        # Chain: query -> filter -> order_by -> all
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.all.return_value = [mock_msg1, mock_msg2]

        result = self.service.load_history("sess-8")

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], HumanMessage)
        self.assertEqual(result[0].content, "hello")
        self.assertIsInstance(result[1], AIMessage)
        self.assertEqual(result[1].content, "hi back")
        mock_db.close.assert_called_once()

    @patch("services.memory_service.SessionLocal")
    def test_load_history_with_limit(self, mock_session_local):
        """load_history respects the limit parameter."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        mock_limit_chain = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.limit.return_value = mock_limit_chain
        mock_limit_chain.all.return_value = []

        self.service.load_history("sess-9", limit=5)

        mock_order.limit.assert_called_once_with(5)
        mock_db.close.assert_called_once()

    @patch("services.memory_service.SessionLocal")
    def test_load_history_empty_session(self, mock_session_local):
        """load_history returns empty list for session with no messages."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.all.return_value = []

        result = self.service.load_history("empty-session")
        self.assertEqual(result, [])

    # ------------------------------------------------------------------
    # clear_session
    # ------------------------------------------------------------------

    @patch("services.memory_service.SessionLocal")
    def test_clear_session_removes_short_term_and_db(self, mock_session_local):
        """clear_session removes session from both short-term cache and MySQL."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        # Pre-populate short-term cache
        session_id = "sess-10"
        self.service._short_term[session_id] = [HumanMessage(content="should be gone")]

        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter

        self.service.clear_session(session_id)

        # Short-term should be cleared
        self.assertNotIn(session_id, self.service._short_term)

        # DB delete should have been called
        mock_filter.delete.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("services.memory_service.SessionLocal")
    def test_clear_session_unknown_session(self, mock_session_local):
        """clear_session on unknown session does not raise and still cleans DB."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter

        # Should not raise
        self.service.clear_session("nonexistent")

        # DB operations still happen
        mock_filter.delete.assert_called_once()
        mock_db.commit.assert_called_once()

    # ------------------------------------------------------------------
    # get_context respects short_term_window after add_message
    # ------------------------------------------------------------------

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_get_context_reflects_added_messages(self, mock_get_settings, mock_session_local):
        """get_context returns messages that were added via add_message."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.first.return_value = None

        session_id = "sess-11"
        self.service.add_message(session_id, "user", "test input", db=mock_db)

        context = self.service.get_context(session_id)
        self.assertEqual(len(context), 1)
        self.assertIsInstance(context[0], HumanMessage)
        self.assertEqual(context[0].content, "test input")


if __name__ == "__main__":
    unittest.main()
