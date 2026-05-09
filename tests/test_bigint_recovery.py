import sys
import unittest
from unittest.mock import MagicMock

from sqlalchemy.sql.elements import TextClause

from scripts.recover_bigint_schema import (
    RecoveryStep,
    _ensure_project_root_on_path,
    build_recovery_steps,
    run_recovery_steps,
)


class TestBigintRecoveryPlan(unittest.TestCase):
    def test_ensure_project_root_on_path_adds_root_once(self):
        original_path = list(sys.path)
        try:
            sys.path = [entry for entry in sys.path if entry != "/project-root"]
            _ensure_project_root_on_path("/project-root")
            _ensure_project_root_on_path("/project-root")

            self.assertEqual(sys.path.count("/project-root"), 1)
            self.assertEqual(sys.path[0], "/project-root")
        finally:
            sys.path = original_path

    def test_build_recovery_steps_includes_user_session_message_migration(self):
        steps = build_recovery_steps(single_user_name="wzw")
        sql_text = "\n".join(step.sql for step in steps)

        self.assertIn("ALTER TABLE users ADD COLUMN new_id BIGSERIAL", sql_text)
        self.assertIn("ALTER TABLE chat_sessions ADD COLUMN new_user_id BIGINT", sql_text)
        self.assertIn("ALTER TABLE chat_messages ADD COLUMN new_session_id BIGINT", sql_text)
        self.assertIn("UPDATE chat_sessions", sql_text)
        self.assertIn("UPDATE chat_messages", sql_text)
        self.assertIn("ALTER TABLE chat_sessions RENAME COLUMN new_user_id TO user_id", sql_text)

    def test_build_recovery_steps_requires_single_user_name(self):
        with self.assertRaises(ValueError):
            build_recovery_steps(single_user_name="")

    def test_build_recovery_steps_swaps_uuid_columns_to_bigint_contract(self):
        steps = build_recovery_steps(single_user_name="wzw")
        sql_text = "\n".join(step.sql for step in steps)

        self.assertIn("ALTER TABLE users ADD COLUMN new_id BIGSERIAL", sql_text)
        self.assertIn("ALTER TABLE chat_sessions ADD COLUMN new_id BIGSERIAL", sql_text)
        self.assertIn("ALTER TABLE chat_messages ADD COLUMN new_id BIGSERIAL", sql_text)
        self.assertIn("ALTER TABLE chat_sessions ADD COLUMN new_user_id BIGINT", sql_text)
        self.assertIn("ALTER TABLE chat_messages ADD COLUMN new_session_id BIGINT", sql_text)
        self.assertIn("UPDATE chat_sessions SET new_user_id = users.new_id", sql_text)
        self.assertIn("UPDATE chat_messages SET new_session_id = chat_sessions.new_id", sql_text)
        self.assertIn("ALTER TABLE users DROP COLUMN id", sql_text)
        self.assertIn("ALTER TABLE chat_sessions DROP COLUMN id", sql_text)
        self.assertIn("ALTER TABLE chat_messages DROP COLUMN session_id", sql_text)

    def test_build_recovery_steps_restores_primary_key_constraints(self):
        steps = build_recovery_steps(single_user_name="wzw")
        sql_text = "\n".join(step.sql for step in steps)

        self.assertIn("ALTER TABLE users ADD PRIMARY KEY (id)", sql_text)
        self.assertIn("ALTER TABLE chat_sessions ADD PRIMARY KEY (id)", sql_text)
        self.assertIn("ALTER TABLE chat_messages ADD PRIMARY KEY (id)", sql_text)
        self.assertIn("ALTER TABLE knowledge_bases ADD PRIMARY KEY (id)", sql_text)
        self.assertIn("ALTER TABLE document_metas ADD PRIMARY KEY (id)", sql_text)

    def test_build_recovery_steps_includes_knowledge_base_and_document_meta_migration(self):
        steps = build_recovery_steps(single_user_name="wzw")
        sql_text = "\n".join(step.sql for step in steps)

        self.assertIn("ALTER TABLE knowledge_bases ADD COLUMN new_id BIGSERIAL", sql_text)
        self.assertIn("ALTER TABLE document_metas ADD COLUMN new_id BIGSERIAL", sql_text)
        self.assertIn("ALTER TABLE document_metas ADD COLUMN new_kb_id BIGINT", sql_text)
        self.assertIn(
            "UPDATE document_metas SET new_kb_id = knowledge_bases.new_id FROM knowledge_bases WHERE document_metas.kb_id = knowledge_bases.id::text",
            sql_text,
        )
        self.assertIn("ALTER TABLE knowledge_bases DROP COLUMN id", sql_text)
        self.assertIn("ALTER TABLE document_metas DROP COLUMN kb_id", sql_text)
        self.assertIn("ALTER TABLE document_metas RENAME COLUMN new_kb_id TO kb_id", sql_text)
        self.assertIn("CREATE INDEX IF NOT EXISTS ix_document_metas_kb_id ON document_metas (kb_id)", sql_text)

    def test_run_recovery_steps_executes_all_sql_in_order(self):
        session = MagicMock()
        steps = [
            RecoveryStep("step1", "SELECT 1"),
            RecoveryStep("step2", "SELECT 2"),
        ]

        run_recovery_steps(session, steps, {"single_user_name": "wzw"})

        self.assertEqual(len(session.execute.call_args_list), 2)
        self.assertIsInstance(session.execute.call_args_list[0].args[0], TextClause)
        self.assertEqual(str(session.execute.call_args_list[0].args[0]), "SELECT 1")
        self.assertEqual(session.execute.call_args_list[0].args[1], {"single_user_name": "wzw"})
        self.assertIsInstance(session.execute.call_args_list[1].args[0], TextClause)
        self.assertEqual(str(session.execute.call_args_list[1].args[0]), "SELECT 2")
        self.assertEqual(session.execute.call_args_list[1].args[1], {"single_user_name": "wzw"})
        session.commit.assert_called_once()

    def test_run_recovery_steps_rolls_back_on_failure(self):
        session = MagicMock()
        session.execute.side_effect = [None, RuntimeError("boom")]
        steps = [RecoveryStep("step1", "SELECT 1"), RecoveryStep("step2", "SELECT 2")]

        with self.assertRaises(RuntimeError):
            run_recovery_steps(session, steps, {})

        session.rollback.assert_called_once()
