"""Tests for MemoryService with mocked MySQL (SessionLocal)."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from models.chat_message import ChatMessage
from models.chat_session import ChatSession
from services.memory_service import MemoryService


class TestMemoryService(unittest.TestCase):
    """Tests for MemoryService behavior."""

    def setUp(self):
        self.service = MemoryService()

    @patch("services.memory_service.get_settings")
    def test_get_context_returns_all_messages_when_under_window(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 5

        session_id = "sess-1"
        self.service._buffers_s1[session_id] = [
            HumanMessage(content="msg1"),
            HumanMessage(content="msg2"),
            HumanMessage(content="msg3"),
        ]

        result = self.service.get_context(session_id)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].content, "msg1")

    @patch("services.memory_service.get_settings")
    def test_get_context_returns_all_cached_s1_messages(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-2"
        self.service._buffers_s1[session_id] = [
            HumanMessage(content=f"msg{i}") for i in range(1, 6)
        ]

        result = self.service.get_context(session_id)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0].content, "msg1")
        self.assertEqual(result[-1].content, "msg5")

    @patch("services.memory_service.get_settings")
    def test_get_context_empty_session(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        with patch("services.memory_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db
            mock_session = MagicMock()
            mock_session.summary = None
            mock_session.summary_sequence = 0

            session_query = MagicMock()
            msg_query = MagicMock()
            msg_filter = MagicMock()
            msg_order = MagicMock()

            def query_side_effect(model):
                if model is ChatSession:
                    session_query.filter.return_value.first.return_value = None
                    return session_query
                msg_query.filter.return_value = msg_filter
                msg_filter.filter.return_value = msg_filter
                msg_filter.order_by.return_value = msg_order
                msg_order.all.return_value = []
                return msg_query

            mock_db.query.side_effect = query_side_effect
            result = self.service.get_context("nonexistent-session")
        self.assertEqual(result, [])

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_user_to_s1(self, mock_get_settings, mock_session_local):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        msg_query = MagicMock()
        msg_filter = MagicMock()
        msg_order = MagicMock()
        msg_filter.order_by.return_value = msg_order
        msg_order.first.return_value = None

        session_query = MagicMock()
        session_filter = MagicMock()
        session_obj = MagicMock()
        session_filter.first.return_value = session_obj

        def query_side_effect(model):
            if model is ChatMessage:
                msg_query.filter.return_value = msg_filter
                return msg_query
            session_query.filter.return_value = session_filter
            return session_query

        mock_db.query.side_effect = query_side_effect

        session_id = "sess-3"
        self.service.add_message(session_id, "user", "hello", db=mock_db)

        messages = self.service._buffers_s1[session_id]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertEqual(messages[0].content, "hello")

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_assistant_to_s1(self, mock_get_settings, mock_session_local):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        msg_query = MagicMock()
        msg_filter = MagicMock()
        msg_order = MagicMock()
        msg_filter.order_by.return_value = msg_order
        msg_order.first.return_value = None

        session_query = MagicMock()
        session_filter = MagicMock()
        session_obj = MagicMock()
        session_filter.first.return_value = session_obj

        def query_side_effect(model):
            if model is ChatMessage:
                msg_query.filter.return_value = msg_filter
                return msg_query
            session_query.filter.return_value = session_filter
            return session_query

        mock_db.query.side_effect = query_side_effect

        session_id = "sess-4"
        self.service.add_message(session_id, "assistant", "hi there", db=mock_db)

        messages = self.service._buffers_s1[session_id]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], AIMessage)
        self.assertEqual(messages[0].content, "hi there")

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_persists_to_mysql(self, mock_get_settings, mock_session_local):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        msg_query = MagicMock()
        msg_filter = MagicMock()
        msg_order = MagicMock()
        msg_filter.order_by.return_value = msg_order
        msg_order.first.return_value = None
        msg_query.filter.return_value = msg_filter

        session_query = MagicMock()
        session_filter = MagicMock()
        session_obj = MagicMock()
        session_filter.first.return_value = session_obj
        session_query.filter.return_value = session_filter

        def query_side_effect(model):
            if model is ChatSession:
                return session_query
            return msg_query

        mock_db.query.side_effect = query_side_effect

        session_id = "sess-5"
        self.service.add_message(session_id, "user", "persist me", db=mock_db)

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        saved_msg = mock_db.add.call_args[0][0]
        self.assertEqual(saved_msg.session_id, session_id)
        self.assertEqual(saved_msg.role, "user")
        self.assertEqual(saved_msg.content, "persist me")
        self.assertEqual(saved_msg.sequence, 0)

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_sequence_increment(self, mock_get_settings, mock_session_local):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        msg_query = MagicMock()
        msg_filter = MagicMock()
        msg_order = MagicMock()
        msg_filter.order_by.return_value = msg_order
        msg_query.filter.return_value = msg_filter

        session_query = MagicMock()
        session_filter = MagicMock()
        session_obj = MagicMock()
        session_filter.first.return_value = session_obj
        session_query.filter.return_value = session_filter

        def query_side_effect(model):
            if model is ChatSession:
                return session_query
            return msg_query

        mock_db.query.side_effect = query_side_effect

        session_id = "sess-6"
        msg_order.first.return_value = None
        self.service.add_message(session_id, "user", "first", db=mock_db)
        first_msg = mock_db.add.call_args[0][0]
        self.assertEqual(first_msg.sequence, 0)

        mock_db.add.reset_mock()
        mock_db.commit.reset_mock()

        msg_order.first.return_value = (0,)
        self.service.add_message(session_id, "assistant", "second", db=mock_db)
        second_msg = mock_db.add.call_args[0][0]
        self.assertEqual(second_msg.sequence, 1)

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_add_message_multiple_messages_same_session(self, mock_get_settings, mock_session_local):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        msg_query = MagicMock()
        msg_filter = MagicMock()
        msg_order = MagicMock()
        msg_filter.order_by.return_value = msg_order
        msg_query.filter.return_value = msg_filter

        session_query = MagicMock()
        session_filter = MagicMock()
        session_obj = MagicMock()
        session_filter.first.return_value = session_obj
        session_query.filter.return_value = session_filter

        def query_side_effect(model):
            if model is ChatSession:
                return session_query
            return msg_query

        mock_db.query.side_effect = query_side_effect

        session_id = "sess-7"
        msg_order.first.return_value = None
        self.service.add_message(session_id, "user", "q1", db=mock_db)

        msg_order.first.return_value = (0,)
        self.service.add_message(session_id, "assistant", "a1", db=mock_db)

        s1_messages = self.service._buffers_s1.get(session_id, [])
        s2_messages = self.service._buffers_s2.get(session_id, [])
        all_messages = s1_messages + s2_messages
        self.assertEqual(len(all_messages), 2)
        self.assertIsInstance(all_messages[0], HumanMessage)
        self.assertIsInstance(all_messages[1], AIMessage)

    @patch("services.memory_service.SessionLocal")
    def test_load_history_returns_messages_in_order(self, mock_session_local):
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        mock_msg1 = MagicMock()
        mock_msg1.role = "user"
        mock_msg1.content = "hello"
        mock_msg1.sequence = 0

        mock_msg2 = MagicMock()
        mock_msg2.role = "assistant"
        mock_msg2.content = "hi back"
        mock_msg2.sequence = 1

        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_order = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_filter
        mock_filter.order_by.return_value = mock_order
        mock_order.all.return_value = [mock_msg1, mock_msg2]

        result = self.service.load_history("sess-8")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "hello")
        self.assertEqual(result[1].content, "hi back")
        mock_db.close.assert_called_once()

    @patch("services.memory_service.SessionLocal")
    def test_load_history_with_limit(self, mock_session_local):
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

    @patch("services.memory_service.SessionLocal")
    def test_clear_session_removes_short_term_and_db(self, mock_session_local):
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        session_id = "sess-10"
        self.service._buffers_s1[session_id] = [HumanMessage(content="should be gone")]

        msg_query = MagicMock()
        session_query = MagicMock()
        msg_filter = MagicMock()
        session_filter = MagicMock()

        def query_side_effect(model):
            if model is ChatMessage:
                msg_query.filter.return_value = msg_filter
                return msg_query
            session_query.filter.return_value = session_filter
            return session_query

        mock_db.query.side_effect = query_side_effect

        self.service.clear_session(session_id)

        self.assertNotIn(session_id, self.service._buffers_s1)
        msg_filter.delete.assert_called_once()
        session_filter.delete.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.close.assert_called_once()

    @patch("services.memory_service.SessionLocal")
    def test_clear_session_unknown_session(self, mock_session_local):
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        msg_query = MagicMock()
        session_query = MagicMock()
        msg_filter = MagicMock()
        session_filter = MagicMock()

        def query_side_effect(model):
            if model is ChatMessage:
                msg_query.filter.return_value = msg_filter
                return msg_query
            session_query.filter.return_value = session_filter
            return session_query

        mock_db.query.side_effect = query_side_effect

        self.service.clear_session("nonexistent")

        msg_filter.delete.assert_called_once()
        session_filter.delete.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch("services.memory_service.SessionLocal")
    @patch("services.memory_service.get_settings")
    def test_get_context_reflects_added_messages(self, mock_get_settings, mock_session_local):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        msg_query = MagicMock()
        msg_filter = MagicMock()
        msg_order = MagicMock()
        msg_filter.order_by.return_value = msg_order
        msg_order.first.return_value = None

        session_query = MagicMock()
        session_filter = MagicMock()
        session_obj = MagicMock()
        session_filter.first.return_value = session_obj

        def query_side_effect(model):
            if model is ChatMessage:
                msg_query.filter.return_value = msg_filter
                return msg_query
            session_query.filter.return_value = session_filter
            return session_query

        mock_db.query.side_effect = query_side_effect

        session_id = "sess-11"
        self.service.add_message(session_id, "user", "test input", db=mock_db)

        context = self.service.get_context(session_id)
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0].content, "test input")


class TestDualBufferCompression(unittest.TestCase):
    def setUp(self):
        self.service = MemoryService()

    @patch("services.memory_service.get_settings")
    def test_get_context_returns_summary_plus_s1_plus_s2(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-dual-1"
        self.service._summary_cache[session_id] = "summary text"
        self.service._buffers_s1[session_id] = [HumanMessage(content="s1-msg")]
        self.service._buffers_s2[session_id] = [AIMessage(content="s2-msg")]

        context = self.service.get_context(session_id)
        self.assertEqual(len(context), 3)
        self.assertEqual(context[0].content, "以下是之前对话的摘要：summary text")
        self.assertEqual(context[1].content, "s1-msg")
        self.assertEqual(context[2].content, "s2-msg")

    @patch("services.memory_service.get_settings")
    def test_add_message_writes_to_s2_when_s1_is_full(self, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 2
        cfg.summary_update_interval = 2
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        session_id = "sess-dual-3"
        self.service._buffers_s1[session_id] = [
            HumanMessage(content="m1"),
            AIMessage(content="m2"),
        ]

        with patch("services.memory_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db

            mock_msg_query = MagicMock()
            mock_msg_filter = MagicMock()
            mock_msg_order = MagicMock()
            mock_msg_filter.order_by.return_value = mock_msg_order
            mock_msg_order.first.return_value = (1,)

            mock_session_query = MagicMock()
            mock_session_filter = MagicMock()
            mock_session_obj = MagicMock()
            mock_session_obj.message_count = 2
            mock_session_filter.first.return_value = mock_session_obj

            def query_side_effect(model):
                if model is ChatMessage:
                    mock_msg_query.filter.return_value = mock_msg_filter
                    return mock_msg_query
                mock_session_query.filter.return_value = mock_session_filter
                return mock_session_query

            mock_db.query.side_effect = query_side_effect
            self.service.add_message(session_id, "user", "m3", db=mock_db)

        self.assertEqual([m.content for m in self.service._buffers_s1[session_id]], ["m1", "m2"])
        self.assertEqual([m.content for m in self.service._buffers_s2[session_id]], ["m3"])

    @patch("services.memory_service.get_settings")
    def test_clear_session_removes_s1_and_s2_buffers(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-dual-2"
        self.service._buffers_s1[session_id] = [HumanMessage(content="s1")]
        self.service._buffers_s2[session_id] = [AIMessage(content="s2")]

        with patch("services.memory_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db
            msg_query = MagicMock()
            session_query = MagicMock()
            msg_filter = MagicMock()
            session_filter = MagicMock()

            def query_side_effect(model):
                if model is ChatMessage:
                    msg_query.filter.return_value = msg_filter
                    return msg_query
                session_query.filter.return_value = session_filter
                return session_query

            mock_db.query.side_effect = query_side_effect
            self.service.clear_session(session_id)

        self.assertNotIn(session_id, self.service._buffers_s1)
        self.assertNotIn(session_id, self.service._buffers_s2)

    @patch("services.memory_service.get_settings")
    def test_on_compression_complete_moves_s2_to_s1_and_clears_s2(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-dual-4"
        self.service._buffers_s1[session_id] = [HumanMessage(content="old-s1")]
        self.service._buffers_s2[session_id] = [HumanMessage(content="new-s2"), AIMessage(content="new-s2-reply")]

        self.service._on_compression_complete(session_id)

        self.assertEqual([m.content for m in self.service._buffers_s1[session_id]], ["new-s2", "new-s2-reply"])
        self.assertEqual(self.service._buffers_s2[session_id], [])

    @patch("services.memory_service.get_settings")
    def test_start_compression_marks_session_as_compressing_and_is_idempotent(self, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 2
        cfg.summary_update_interval = 2
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        session_id = "sess-dual-5"

        with patch("services.memory_service.threading.Thread") as mock_thread:
            thread_instance = MagicMock()
            mock_thread.return_value = thread_instance
            self.service._start_compression(session_id)
            self.service._start_compression(session_id)

        self.assertIn(session_id, self.service._summary_in_progress)
        self.assertEqual(mock_thread.call_count, 1)
        thread_instance.start.assert_called_once()

    @patch("services.memory_service.get_settings")
    def test_start_compression_runs_summary_and_completion_flow(self, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 2
        cfg.summary_update_interval = 2
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        session_id = "sess-dual-6"

        with patch("services.memory_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db
            mock_session_query = MagicMock()
            mock_session_filter = MagicMock()
            mock_session_obj = MagicMock()
            mock_session_filter.first.return_value = mock_session_obj
            mock_session_query.filter.return_value = mock_session_filter
            mock_db.query.return_value = mock_session_query

            class ImmediateThread:
                def __init__(self, target=None, daemon=None):
                    self._target = target

                def start(self):
                    if self._target:
                        self._target()

            with patch("services.memory_service.threading.Thread", ImmediateThread):
                with patch.object(self.service, "_generate_summary") as mock_generate_summary:
                    with patch.object(self.service, "_on_compression_complete") as mock_on_complete:
                        self.service._start_compression(session_id)

        mock_generate_summary.assert_called_once()
        mock_on_complete.assert_called_once_with(session_id)
        self.assertNotIn(session_id, self.service._summary_in_progress)

    @patch("services.memory_service.get_settings")
    def test_add_message_triggers_compression_when_s1_reaches_window(self, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 4
        cfg.summary_update_interval = 4
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        session_id = "sess-dual-7"
        self.service._buffers_s1[session_id] = [HumanMessage(content="m1")]

        with patch.object(self.service, "_start_compression") as mock_start_compression:
            with patch("services.memory_service.SessionLocal") as mock_session_local:
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                mock_msg_query = MagicMock()
                mock_msg_filter = MagicMock()
                mock_msg_order = MagicMock()
                mock_msg_filter.order_by.return_value = mock_msg_order
                mock_msg_order.first.return_value = (0,)

                mock_session_query = MagicMock()
                mock_session_filter = MagicMock()
                mock_session_obj = MagicMock()
                mock_session_obj.message_count = 1
                mock_session_filter.first.return_value = mock_session_obj

                def query_side_effect(model):
                    if model is ChatMessage:
                        mock_msg_query.filter.return_value = mock_msg_filter
                        return mock_msg_query
                    mock_session_query.filter.return_value = mock_session_filter
                    return mock_session_query

                mock_db.query.side_effect = query_side_effect
                self.service.add_message(session_id, "assistant", "m2", db=mock_db)

        mock_start_compression.assert_called_once_with(session_id)



    @patch("services.memory_service.get_settings")
    def test_recover_context_restores_all_uncompressed_messages_to_s1_when_within_half_window(self, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 10
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        session_id = "sess-dual-8"

        with patch("services.memory_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db

            session_obj = MagicMock()
            session_obj.summary = "summary text"
            session_obj.summary_sequence = 10

            msg11 = MagicMock(role="user", content="m11", sequence=11)
            msg12 = MagicMock(role="assistant", content="m12", sequence=12)
            msg13 = MagicMock(role="user", content="m13", sequence=13)
            msg14 = MagicMock(role="assistant", content="m14", sequence=14)
            msg15 = MagicMock(role="user", content="m15", sequence=15)

            with patch.object(ChatSession, "get_by_id", return_value=session_obj):
                with patch.object(ChatMessage, "get_after_sequence", return_value=[msg11, msg12, msg13, msg14, msg15]):
                    with patch.object(self.service, "_generate_summary") as mock_generate_summary:
                        context = self.service.get_context(session_id)

        self.assertEqual([m.content for m in self.service._buffers_s1[session_id]], ["m11", "m12", "m13", "m14", "m15"])
        self.assertEqual(self.service._buffers_s2[session_id], [])
        self.assertEqual(len(context), 6)
        self.assertEqual(context[0].content, "以下是之前对话的摘要：summary text")
        mock_generate_summary.assert_not_called()

    @patch("services.memory_service.get_settings")
    def test_recover_context_restores_last_half_window_and_triggers_compression_when_uncompressed_exceeds_half(self, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 10
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        session_id = "sess-dual-9"

        with patch("services.memory_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db

            session_obj = MagicMock()
            session_obj.summary = "summary text"
            session_obj.summary_sequence = 10

            msgs = [MagicMock(role="user" if i % 2 else "assistant", content=f"m{i}", sequence=i) for i in range(11, 19)]

            with patch.object(ChatSession, "get_by_id", return_value=session_obj):
                with patch.object(ChatMessage, "get_after_sequence", return_value=msgs):
                    with patch.object(self.service, "_start_compression") as mock_start_compression:
                        context = self.service.get_context(session_id)

        self.assertEqual([m.content for m in self.service._buffers_s1[session_id]], ["m14", "m15", "m16", "m17", "m18"])
        self.assertEqual(self.service._buffers_s2[session_id], [])
        self.assertEqual(len(context), 6)

    @patch("services.memory_service.get_settings")
    def test_recover_context_without_summary_uses_half_window_policy(self, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 10
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        session_id = "sess-dual-10"

        with patch("services.memory_service.SessionLocal") as mock_session_local:
            mock_db = MagicMock()
            mock_session_local.return_value = mock_db

            session_obj = MagicMock()
            session_obj.summary = None
            session_obj.summary_sequence = 0

            msgs = [MagicMock(role="user" if i % 2 else "assistant", content=f"m{i}", sequence=i) for i in range(1, 9)]

            with patch.object(ChatSession, "get_by_id", return_value=session_obj):
                with patch.object(ChatMessage, "get_after_sequence", return_value=msgs):
                    with patch.object(self.service, "_start_compression") as mock_start_compression:
                        context = self.service.get_context(session_id)

        self.assertEqual([m.content for m in self.service._buffers_s1[session_id]], ["m4", "m5", "m6", "m7", "m8"])
        self.assertEqual(self.service._buffers_s2[session_id], [])
        self.assertEqual(len(context), 5)
        mock_start_compression.assert_called_once_with(session_id)


    @patch("services.memory_service.get_settings")
    def test_get_context_parts_returns_summary_and_history_separately(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-dual-11"
        self.service._summary_cache[session_id] = "summary text"
        self.service._buffers_s1[session_id] = [HumanMessage(content="s1-msg")]
        self.service._buffers_s2[session_id] = [AIMessage(content="s2-msg")]

        summary, history = self.service.get_context_parts(session_id)

        self.assertEqual(summary, "summary text")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].content, "s1-msg")
        self.assertEqual(history[1].content, "s2-msg")


    @patch("services.memory_service.get_settings")
    @patch("services.memory_service.LLMFactory.create")
    def test_generate_summary_uses_shorter_length_and_concise_prompt(self, mock_create, mock_get_settings):
        cfg = MagicMock()
        cfg.short_term_window = 10
        cfg.summary_max_length = 1000
        mock_get_settings.return_value.get_memory_config.return_value = cfg

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="简短摘要")
        mock_create.return_value = mock_llm

        db = MagicMock()
        session = MagicMock()
        session.summary = None
        session.summary_sequence = 0

        msg1 = MagicMock(role="user", content="问题一", sequence=0)
        msg2 = MagicMock(role="assistant", content="回答一", sequence=1)

        with patch.object(ChatMessage, "get_recent", return_value=[msg1, msg2]):
            self.service._generate_summary(db, "sess-summary", session)

        prompt = mock_llm.invoke.call_args[0][0]

    @patch("services.memory_service.get_settings")
    def test_memory_uses_integer_session_id_keys(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        session_id = 123
        self.service._buffers_s1[session_id] = [HumanMessage(content="msg")]

        summary, history = self.service.get_context_parts(session_id)

        self.assertEqual(summary, "")
        self.assertEqual(history[0].content, "msg")
