# API User Bigint Schema Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover the local database schema from string/UUID identifiers back to the bigint contract expected by the current models and routers, while preserving the existing user, sessions, and messages by assigning all current sessions to the single existing user.

**Architecture:** Implement this as a one-off database recovery script that performs additive bigint-column migration, data backfill, and final column swap inside controlled SQL steps. Keep the runtime application contract on `int`/`BigInteger`, then add regression tests that verify the migration script produces a schema and API behavior aligned with the existing ORM and router design.

**Tech Stack:** Python 3.11, SQLAlchemy, FastAPI, Pydantic, PostgreSQL-compatible SQL, unittest/pytest

---

### Task 1: Add migration-script tests for bigint recovery

**Files:**
- Create: `tests/test_bigint_recovery.py`
- Test: `tests/test_bigint_recovery.py`

**Step 1: Write the failing test**

Create `tests/test_bigint_recovery.py` with a focused schema/data recovery test. Use a fake executor to capture SQL statements instead of touching the real database in the first test.

```python
import unittest
from unittest.mock import MagicMock

from scripts.recover_bigint_schema import build_recovery_steps


class TestBigintRecoveryPlan(unittest.TestCase):
    def test_build_recovery_steps_includes_user_session_message_migration(self):
        steps = build_recovery_steps(single_user_name="wzw")
        sql_text = "\n".join(step.sql for step in steps)

        self.assertIn("ALTER TABLE users ADD COLUMN new_id BIGINT", sql_text)
        self.assertIn("ALTER TABLE chat_sessions ADD COLUMN new_user_id BIGINT", sql_text)
        self.assertIn("ALTER TABLE chat_messages ADD COLUMN new_session_id BIGINT", sql_text)
        self.assertIn("UPDATE chat_sessions", sql_text)
        self.assertIn("UPDATE chat_messages", sql_text)
        self.assertIn("ALTER TABLE chat_sessions RENAME COLUMN new_user_id TO user_id", sql_text)
```

Add a second test for guardrails:

```python
def test_build_recovery_steps_requires_single_user_name(self):
    with self.assertRaises(ValueError):
        build_recovery_steps(single_user_name="")
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_bigint_recovery.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.recover_bigint_schema'`.

**Step 3: Write minimal implementation**

Create `scripts/recover_bigint_schema.py` with:

```python
from dataclasses import dataclass


@dataclass
class RecoveryStep:
    name: str
    sql: str


def build_recovery_steps(single_user_name: str) -> list[RecoveryStep]:
    if not single_user_name:
        raise ValueError("single_user_name is required")
    return [
        RecoveryStep("add users.new_id", "ALTER TABLE users ADD COLUMN new_id BIGINT"),
        RecoveryStep("add sessions.new_id", "ALTER TABLE chat_sessions ADD COLUMN new_id BIGINT"),
        RecoveryStep("add sessions.new_user_id", "ALTER TABLE chat_sessions ADD COLUMN new_user_id BIGINT"),
        RecoveryStep("add messages.new_id", "ALTER TABLE chat_messages ADD COLUMN new_id BIGINT"),
        RecoveryStep("add messages.new_session_id", "ALTER TABLE chat_messages ADD COLUMN new_session_id BIGINT"),
        RecoveryStep("backfill sessions user", "UPDATE chat_sessions SET new_user_id = 1"),
        RecoveryStep("backfill messages session", "UPDATE chat_messages SET new_session_id = 1"),
        RecoveryStep("rename session user", "ALTER TABLE chat_sessions RENAME COLUMN new_user_id TO user_id"),
    ]
```

This is intentionally incomplete but enough to go green on the first red test.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_bigint_recovery.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_bigint_recovery.py scripts/recover_bigint_schema.py
git commit -m "test: scaffold bigint recovery migration plan"
```

---

### Task 2: Make the recovery plan encode the actual migration steps

**Files:**
- Modify: `scripts/recover_bigint_schema.py`
- Modify: `tests/test_bigint_recovery.py`
- Read: `docs/plans/2026-05-08-api-user-bigint-recovery-design.md`

**Step 1: Write the failing test**

Extend `tests/test_bigint_recovery.py` to lock in the actual SQL plan requirements from the design:

```python
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
```

Add an ordering test to ensure the destructive drops happen after backfill steps:

```python
def test_drop_steps_happen_after_backfill_steps(self):
    steps = build_recovery_steps(single_user_name="wzw")
    names = [step.name for step in steps]

    self.assertLess(names.index("backfill sessions user mapping"), names.index("swap chat_sessions columns"))
    self.assertLess(names.index("backfill messages session mapping"), names.index("swap chat_messages columns"))
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_bigint_recovery.py -v
```

Expected: FAIL because the current placeholder SQL does not meet the design.

**Step 3: Write minimal implementation**

Update `scripts/recover_bigint_schema.py` so `build_recovery_steps(...)` returns a concrete ordered plan, for example:

```python
return [
    RecoveryStep("add users.new_id", "ALTER TABLE users ADD COLUMN new_id BIGSERIAL"),
    RecoveryStep("add chat_sessions.new_id", "ALTER TABLE chat_sessions ADD COLUMN new_id BIGSERIAL"),
    RecoveryStep("add chat_sessions.new_user_id", "ALTER TABLE chat_sessions ADD COLUMN new_user_id BIGINT"),
    RecoveryStep("add chat_messages.new_id", "ALTER TABLE chat_messages ADD COLUMN new_id BIGSERIAL"),
    RecoveryStep("add chat_messages.new_session_id", "ALTER TABLE chat_messages ADD COLUMN new_session_id BIGINT"),
    RecoveryStep(
        "backfill sessions user mapping",
        """
        UPDATE chat_sessions
        SET new_user_id = users.new_id
        FROM users
        WHERE users.name = :single_user_name
        """.strip(),
    ),
    RecoveryStep(
        "backfill messages session mapping",
        """
        UPDATE chat_messages
        SET new_session_id = chat_sessions.new_id
        FROM chat_sessions
        WHERE chat_messages.session_id = chat_sessions.id::text
        """.strip(),
    ),
    RecoveryStep("swap users columns", "..."),
    RecoveryStep("swap chat_sessions columns", "..."),
    RecoveryStep("swap chat_messages columns", "..."),
]
```

Keep the swap SQL explicit and in one place. Do not optimize yet.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_bigint_recovery.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_bigint_recovery.py scripts/recover_bigint_schema.py
git commit -m "feat: define bigint recovery migration steps"
```

---

### Task 3: Execute the recovery script against the local database

**Files:**
- Modify: `scripts/recover_bigint_schema.py`
- Test: local database via script execution
- Read: `utils/db.py:20-33`

**Step 1: Write the failing test**

Add a runner-level unit test to verify the script executes steps transactionally and fails loudly if the target user is missing.

```python
from unittest.mock import MagicMock, call

from scripts.recover_bigint_schema import RecoveryStep, run_recovery_steps


def test_run_recovery_steps_executes_all_sql_in_order():
    session = MagicMock()
    steps = [
        RecoveryStep("step1", "SELECT 1"),
        RecoveryStep("step2", "SELECT 2"),
    ]

    run_recovery_steps(session, steps, {"single_user_name": "wzw"})

    self.assertEqual(session.execute.call_args_list, [
        call("SELECT 1", {"single_user_name": "wzw"}),
        call("SELECT 2", {"single_user_name": "wzw"}),
    ])
    session.commit.assert_called_once()
```

Add another test:

```python
def test_run_recovery_steps_rolls_back_on_failure():
    session = MagicMock()
    session.execute.side_effect = [None, RuntimeError("boom")]
    steps = [RecoveryStep("step1", "SELECT 1"), RecoveryStep("step2", "SELECT 2")]

    with self.assertRaises(RuntimeError):
        run_recovery_steps(session, steps, {})

    session.rollback.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_bigint_recovery.py -v
```

Expected: FAIL because `run_recovery_steps` is missing.

**Step 3: Write minimal implementation**

In `scripts/recover_bigint_schema.py`, add:

```python
from sqlalchemy import text
from utils.db import SessionLocal


def run_recovery_steps(session, steps, params):
    try:
        for step in steps:
            session.execute(text(step.sql), params)
        session.commit()
    except Exception:
        session.rollback()
        raise


def main():
    steps = build_recovery_steps(single_user_name="wzw")
    with SessionLocal() as session:
        run_recovery_steps(session, steps, {"single_user_name": "wzw"})
```

Then run the real script once:

```bash
python scripts/recover_bigint_schema.py
```

Expected: exit 0 and no traceback.

**Step 4: Verify the database shape after execution**

Run:

```bash
python - <<'PY'
from sqlalchemy import text
from utils.db import SessionLocal

sqls = [
    ("users_columns", "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users' ORDER BY ordinal_position"),
    ("chat_sessions_columns", "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'chat_sessions' ORDER BY ordinal_position"),
    ("chat_messages_columns", "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'chat_messages' ORDER BY ordinal_position"),
]
with SessionLocal() as db:
    for name, sql in sqls:
        print(f'=== {name} ===')
        for row in db.execute(text(sql)).fetchall():
            print(dict(row._mapping))
PY
```

Expected:
- `users.id` is bigint
- `chat_sessions.id` is bigint
- `chat_sessions.user_id` exists and is bigint
- `chat_messages.id` is bigint
- `chat_messages.session_id` is bigint

**Step 5: Commit**

```bash
git add scripts/recover_bigint_schema.py tests/test_bigint_recovery.py
git commit -m "feat: execute bigint schema recovery"
```

---

### Task 4: Align router and model behavior with recovered schema

**Files:**
- Modify: `models/user.py:6-19`
- Modify: `models/chat_session.py:8-34`
- Modify: `models/chat_message.py:8-60`
- Modify: `routers/user.py:14-32`
- Modify: `routers/session.py:13-74`
- Modify: `tests/test_session_router.py`
- Modify: `tests/test_user_router.py`

**Step 1: Write the failing tests**

Add API-level behavior tests that assume the recovered bigint schema is now real.

In `tests/test_session_router.py`, add:

```python
def test_list_sessions_returns_integer_session_id(self):
    mock_session = MagicMock(id=101, title="Recovered", message_count=58, summary_sequence=14)
    ...
    result = list_sessions(1, db)
    self.assertEqual(result[0].id, 101)
```

In `tests/test_user_router.py`, add a stronger list-user response test:

```python
from routers.user import UserResponse


def test_user_response_rejects_string_id_after_recovery(self):
    with self.assertRaises(Exception):
        UserResponse(id="uuid-value", name="wzw", created_at=None)
```

Add a `list_users` router behavior test in a new or existing router test file:

```python
@patch("routers.user.User")
def test_list_users_returns_integer_ids(self, mock_user):
    db = MagicMock()
    mock_user.list_all.return_value = [MagicMock(id=1, name="wzw", created_at=None)]

    result = list_users(db)

    self.assertEqual(result[0].id, 1)
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_session_router.py tests/test_user_router.py -v
```

Expected: FAIL if any code still assumes the old drifted schema or if list-user behavior is not explicitly covered.

**Step 3: Write minimal implementation**

Keep models/router types on `int`/`BigInteger`, but tighten any assumptions needed after the recovery:

- Ensure router response models remain `int`
- Ensure model helper methods still query with integer IDs
- Ensure no string-ID fallback logic is introduced
- Only add the minimum code needed to support post-recovery behavior and tests

If current code already matches, the implementation step is just the test additions and any tiny cleanup necessary.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_session_router.py tests/test_user_router.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add models/user.py models/chat_session.py models/chat_message.py routers/user.py routers/session.py tests/test_session_router.py tests/test_user_router.py
git commit -m "test: lock bigint router and model contract"
```

---

### Task 5: Verify recovered data through the live application contract

**Files:**
- Modify: `tests/test_chat_router.py` only if needed for recovery verification helpers
- Test: live local API plus database inspection

**Step 1: Write the failing verification script**

Create a one-off command sequence to prove the recovered data is visible through existing endpoints.

Run:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In another terminal-equivalent command sequence, prepare checks:

```bash
curl -s http://127.0.0.1:8000/api/user
curl -s "http://127.0.0.1:8000/api/session?user_id=1"
curl -s "http://127.0.0.1:8000/api/session/1/history?user_id=1"
```

Expected current failure before migration completion: at least one endpoint returns 500 or empty data for preserved records.

**Step 2: Implement the minimal missing pieces**

If the migration assigned a user ID other than `1`, do not hardcode `1` in the actual verification logic. Instead, first fetch `/api/user`, read the integer ID from the response, then use it for the session/history checks.

If needed, add a very small helper script:

- Create: `scripts/verify_bigint_recovery.py`

Example minimal helper:

```python
import requests

base = "http://127.0.0.1:8000"
users = requests.get(f"{base}/api/user").json()
assert users and isinstance(users[0]["id"], int)
user_id = users[0]["id"]
sessions = requests.get(f"{base}/api/session", params={"user_id": user_id}).json()
assert len(sessions) == 2
history = requests.get(f"{base}/api/session/{sessions[0]['id']}/history", params={"user_id": user_id}).json()
assert history
```

**Step 3: Run verification to prove it passes**

Run:

```bash
python scripts/verify_bigint_recovery.py
```

Expected: exit 0 with no traceback.

**Step 4: Stop the server and record evidence**

Record the observed facts:
- `/api/user` returns integer IDs
- the preserved user exists
- `2` sessions are visible for that user
- message history is still readable

**Step 5: Commit**

```bash
git add scripts/verify_bigint_recovery.py
 git commit -m "test: verify recovered bigint data through api"
```

---

### Task 6: Run full regression and compare against the design

**Files:**
- Modify: none unless fixes are required
- Review: `docs/plans/2026-05-08-api-user-bigint-recovery-design.md`
- Test: full targeted regression suite

**Step 1: Run the recovery-specific automated tests**

Run:

```bash
pytest tests/test_bigint_recovery.py tests/test_session_router.py tests/test_user_router.py -v
```

Expected: PASS.

**Step 2: Run the broader application regression slice**

Run:

```bash
pytest tests/test_token_usage.py tests/test_chat_chain.py tests/test_rag_chain.py tests/test_chat_router.py tests/test_llm_factory.py tests/test_session_router.py tests/test_user_router.py -v
```

Expected: PASS.

**Step 3: Compare implementation against the approved design**

Re-read `docs/plans/2026-05-08-api-user-bigint-recovery-design.md` and verify each requirement:

- `/api/user` no longer returns 500
- all ID contracts remain integer-based
- `chat_sessions.user_id` exists after migration
- existing 2 sessions and 58 messages are preserved
- all preserved sessions are assigned to the single existing user
- no token-usage codepaths were modified

**Step 4: Fix any remaining issue only if evidence shows a gap**

If a test or verification step fails, make the smallest possible fix and rerun the failing command first, then rerun the full Task 6 suite.

**Step 5: Commit**

```bash
git add scripts/recover_bigint_schema.py scripts/verify_bigint_recovery.py tests/test_bigint_recovery.py tests/test_session_router.py tests/test_user_router.py models/user.py models/chat_session.py models/chat_message.py routers/user.py routers/session.py
git commit -m "fix: recover bigint schema for user session data"
```

---

## Notes for the implementing engineer

- Keep SQL inside the recovery script, not in routers or services.
- Do not introduce string-ID compatibility in the runtime code; the chosen design is to restore bigint, not support both forms.
- The migration script must be explicit and readable; correctness matters more than cleverness.
- Verify the live API after migration, not just unit tests.
- Before claiming completion, use `superpowers:verification-before-completion`.
