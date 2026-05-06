# Bigint ID Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current UUID-string primary key system with bigint auto-increment IDs across users, chat sessions, and chat messages, while preserving existing data relationships.

**Architecture:** Migrate in controlled phases: first add bigint model semantics and tests, then switch APIs and services to integer IDs, and finally update frontend and verification paths so every layer consistently treats `id`, `user_id`, and `session_id` as bigint-backed integers. Keep business behavior unchanged; only the identifier system changes.

**Tech Stack:** FastAPI, SQLAlchemy ORM, MySQL, plain browser JavaScript, unittest

---

## File Structure

- **Modify** `models/user.py`
  - Change `id` to bigint-compatible primary key and query signatures to `int`
- **Modify** `models/chat_session.py`
  - Change `id` and `user_id` to bigint-compatible fields and update helper methods
- **Modify** `models/chat_message.py`
  - Change `id` and `session_id` to bigint-compatible fields and update helper method signatures
- **Modify** `routers/user.py`
  - Make user response and request handling use integer IDs
- **Modify** `routers/session.py`
  - Make session creation/list/history use integer `user_id` / `session_id`
- **Modify** `services/memory_service.py`
  - Replace string session ID assumptions with integer session IDs and stop generating UUID message IDs
- **Modify** `chains/chat_chain.py`
  - Update `session_id` argument typing to integer
- **Modify** `chains/rag_chain.py`
  - Update `session_id` argument typing to integer
- **Modify** `frontend/index.html`
  - Ensure current user/session IDs are handled consistently as numeric IDs in requests and localStorage restoration
- **Modify** tests:
  - `tests/test_user_router.py`
  - `tests/test_session_router.py`
  - `tests/test_memory_service.py`
  - `tests/test_chat_chain.py`
  - `tests/test_rag_chain.py`

---

### Task 1: Convert ORM ID fields to bigint semantics

**Files:**
- Modify: `models/user.py`
- Modify: `models/chat_session.py`
- Modify: `models/chat_message.py`
- Test: `tests/test_user_router.py`

- [ ] **Step 1: Write the failing model tests**

Append these tests to `tests/test_user_router.py`:

```python
import unittest

from models.chat_message import ChatMessage
from models.chat_session import ChatSession
from models.user import User


class TestBigintModelIds(unittest.TestCase):
    def test_user_id_is_integer_column(self):
        self.assertEqual(User.__table__.columns["id"].type.python_type, int)

    def test_chat_session_user_id_is_integer_column(self):
        self.assertEqual(ChatSession.__table__.columns["user_id"].type.python_type, int)

    def test_chat_message_session_id_is_integer_column(self):
        self.assertEqual(ChatMessage.__table__.columns["session_id"].type.python_type, int)
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run: `python -m unittest tests.test_user_router.TestBigintModelIds -v`
Expected: FAIL because the current ORM columns are still `String(36)`-based

- [ ] **Step 3: Change the `User` model to bigint ID semantics**

Modify `models/user.py` to:

```python
from sqlalchemy import BigInteger, Column, DateTime, String, func
from sqlalchemy.orm import Session

from utils.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment user ID")
    name = Column(String(255), nullable=False, comment="Display name")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")

    @classmethod
    def get_by_id(cls, db: Session, user_id: int):
        return db.query(cls).filter(cls.id == user_id).first()

    @classmethod
    def list_all(cls, db: Session):
        return db.query(cls).order_by(cls.created_at.asc()).all()
```

- [ ] **Step 4: Change the `ChatSession` model to bigint ownership semantics**

Modify `models/chat_session.py` to:

```python
from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Session

from utils.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment session ID")
    user_id = Column(BigInteger, nullable=False, index=True, comment="Owner user ID")
    title = Column(String(255), default="New Chat", comment="Session title")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="Last update time")
    message_count = Column(Integer, default=0, comment="Total messages in session")
    summary = Column(Text, nullable=True, default=None, comment="Conversation summary for long-term memory")
    summary_sequence = Column(Integer, default=0, comment="Last message sequence covered by summary")

    @classmethod
    def get_by_id(cls, db: Session, session_id: int):
        return db.query(cls).filter(cls.id == session_id).first()

    @classmethod
    def get_by_user_and_id(cls, db: Session, user_id: int, session_id: int):
        return db.query(cls).filter(cls.user_id == user_id).filter(cls.id == session_id).first()

    @classmethod
    def list_by_user(cls, db: Session, user_id: int):
        return db.query(cls).filter(cls.user_id == user_id).order_by(cls.updated_at.desc()).all()

    @classmethod
    def delete_by_id(cls, db: Session, session_id: int):
        return db.query(cls).filter(cls.id == session_id).delete()
```

- [ ] **Step 5: Change the `ChatMessage` model to bigint session references**

Modify `models/chat_message.py` to:

```python
from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Session

from utils.db import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="Auto increment message ID")
    session_id = Column(BigInteger, nullable=False, index=True, comment="Associated session ID")
    role = Column(String(20), nullable=False, comment="Message role: user / assistant / system")
    content = Column(Text, nullable=False, comment="Message content")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    sequence = Column(Integer, default=0, comment="Message order within session")

    @classmethod
    def get_after_sequence(cls, db: Session, session_id: int, sequence: int) -> list["ChatMessage"]:
        return (
            db.query(cls)
            .filter(cls.session_id == session_id)
            .filter(cls.sequence > sequence)
            .order_by(cls.sequence.asc())
            .all()
        )

    @classmethod
    def get_max_sequence(cls, db: Session, session_id: int):
        return (
            db.query(cls.sequence)
            .filter(cls.session_id == session_id)
            .order_by(cls.sequence.desc())
            .first()
        )

    @classmethod
    def get_recent(cls, db: Session, session_id: int, limit: int) -> list["ChatMessage"]:
        return (
            db.query(cls)
            .filter(cls.session_id == session_id)
            .order_by(cls.sequence.desc())
            .limit(limit)
            .all()
        )

    @classmethod
    def get_all(cls, db: Session, session_id: int, limit: int | None = None) -> list["ChatMessage"]:
        query = (
            db.query(cls)
            .filter(cls.session_id == session_id)
            .order_by(cls.sequence.asc())
        )
        if limit is not None:
            query = query.limit(limit)
        return query.all()

    @classmethod
    def delete_by_session(cls, db: Session, session_id: int):
        return db.query(cls).filter(cls.session_id == session_id).delete()
```

- [ ] **Step 6: Re-run the model tests to verify they pass**

Run: `python -m unittest tests.test_user_router.TestBigintModelIds -v`
Expected: PASS

- [ ] **Step 7: Commit the model ID migration changes**

```bash
git add models/user.py models/chat_session.py models/chat_message.py tests/test_user_router.py
git commit -m "feat: switch core models to bigint ids"
```

---

### Task 2: Switch router APIs to integer IDs

**Files:**
- Modify: `routers/user.py`
- Modify: `routers/session.py`
- Modify: `main.py`
- Test: `tests/test_session_router.py`
- Test: `tests/test_user_router.py`

- [ ] **Step 1: Write the failing router ID type tests**

Append these tests to `tests/test_session_router.py`:

```python
import unittest
from unittest.mock import MagicMock, patch

from routers.session import CreateSessionRequest, SessionResponse, create_session, get_history, list_sessions
from routers.user import UserResponse


class TestBigintRouterIds(unittest.TestCase):
    def test_create_session_request_accepts_integer_user_id(self):
        req = CreateSessionRequest(title="Chat", user_id=123)
        self.assertEqual(req.user_id, 123)

    def test_user_response_accepts_integer_id(self):
        resp = UserResponse(id=1, name="Alice", created_at=None)
        self.assertEqual(resp.id, 1)
```

- [ ] **Step 2: Run the router tests to verify they fail**

Run: `python -m unittest tests.test_session_router.TestBigintRouterIds -v`
Expected: FAIL because the current Pydantic models still use string ID semantics implicitly or are not consistently updated

- [ ] **Step 3: Update `routers/user.py` response schema to integer IDs**

Modify `routers/user.py` so `UserResponse` becomes:

```python
class UserResponse(BaseModel):
    id: int
    name: str
    created_at: object | None = None
```

Keep creation logic the same, but it now returns bigint-backed integer IDs from the ORM object.

- [ ] **Step 4: Update `routers/session.py` request and response schemas to integer IDs**

Modify the route schemas in `routers/session.py` to:

```python
class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    user_id: int


class SessionResponse(BaseModel):
    id: int
    title: str
    message_count: int
    summary_sequence: int = 0
```

Keep `list_sessions(user_id: int = Query(...), ...)` and `get_history(session_id: int, user_id: int = Query(...), ...)` using integer query/path semantics.

- [ ] **Step 5: Register the current user router if not already registered**

Ensure `main.py` still contains:

```python
from routers.user import router as user_router  # noqa: E402

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(knowledge_router)
app.include_router(user_router)
```

If already present and correct, make no change.

- [ ] **Step 6: Re-run router tests to verify they pass**

Run: `python -m unittest tests.test_user_router tests.test_session_router -v`
Expected: PASS

- [ ] **Step 7: Commit the router bigint changes**

```bash
git add routers/user.py routers/session.py main.py tests/test_session_router.py tests/test_user_router.py
git commit -m "feat: switch user and session APIs to bigint ids"
```

---

### Task 3: Switch memory and chain layers to integer session IDs

**Files:**
- Modify: `services/memory_service.py`
- Modify: `chains/chat_chain.py`
- Modify: `chains/rag_chain.py`
- Test: `tests/test_memory_service.py`
- Test: `tests/test_chat_chain.py`
- Test: `tests/test_rag_chain.py`

- [ ] **Step 1: Write the failing session-id type tests**

Append this test to `tests/test_memory_service.py`:

```python
    @patch("services.memory_service.get_settings")
    def test_memory_uses_integer_session_id_keys(self, mock_get_settings):
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 10
        session_id = 123
        self.service._buffers_s1[session_id] = [HumanMessage(content="msg")]

        summary, history = self.service.get_context_parts(session_id)

        self.assertEqual(summary, "")
        self.assertEqual(history[0].content, "msg")
```

Append this test to `tests/test_chat_chain.py`:

```python
    def test_invoke_accepts_integer_session_id(self, mock_create):
        history = [HumanMessage(content="hi")]
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary text", history)
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="answer")
        mock_create.return_value = llm

        from chains.chat_chain import ChatChain
        chain = ChatChain(memory_service=memory)
        result = chain.invoke(123, "hello")

        self.assertEqual(result["session_id"], 123)
```

- [ ] **Step 2: Run the focused tests to verify they fail if any string assumptions remain**

Run: `python -m unittest tests.test_memory_service tests.test_chat_chain tests.test_rag_chain -v`
Expected: at least one failure if current signatures or returned values still assume string-based IDs

- [ ] **Step 3: Remove UUID generation from `MemoryService.add_message` and switch signatures to `int`**

Modify `services/memory_service.py` so:
- remove `from uuid import uuid4`
- update all `session_id: str` signatures to `session_id: int`
- update cache types to integer-keyed dicts:

```python
self._buffers_s1: dict[int, list[BaseMessage]] = {}
self._buffers_s2: dict[int, list[BaseMessage]] = {}
self._summary_cache: dict[int, str] = {}
self._locks: dict[int, threading.Lock] = {}
```

In `add_message`, replace message construction with:

```python
chat_msg = ChatMessage(
    session_id=session_id,
    role=role,
    content=content,
    sequence=next_seq,
)
```

- [ ] **Step 4: Switch chain method signatures to integer session IDs**

Update `chains/chat_chain.py` and `chains/rag_chain.py` method signatures from:

```python
def invoke(self, session_id: str, ...)
async def astream(self, session_id: str, ...)
```

To:

```python
def invoke(self, session_id: int, ...)
async def astream(self, session_id: int, ...)
```

Preserve all existing business logic.

- [ ] **Step 5: Re-run the focused memory/chain tests to verify they pass**

Run: `python -m unittest tests.test_memory_service tests.test_chat_chain tests.test_rag_chain -v`
Expected: PASS

- [ ] **Step 6: Commit the service and chain bigint changes**

```bash
git add services/memory_service.py chains/chat_chain.py chains/rag_chain.py tests/test_memory_service.py tests/test_chat_chain.py tests/test_rag_chain.py
git commit -m "feat: switch memory and chains to bigint session ids"
```

---

### Task 4: Switch frontend user/session handling to numeric IDs

**Files:**
- Modify: `frontend/index.html`

- [ ] **Step 1: Write the failing browser console checks**

Before changing the frontend, open the page and run:

```js
typeof state.currentUserId
typeof state.sessionId
```

Expected before implementation: user/session IDs may still be absent or handled as plain strings without explicit numeric normalization

- [ ] **Step 2: Normalize restored user IDs to numbers**

In `restoreCurrentUser()`, replace the current logic with numeric normalization:

```js
function restoreCurrentUser() {
  const stored = localStorage.getItem(USER_STORAGE_KEY);
  const storedId = stored ? Number(stored) : null;
  const fallback = state.users[0]?.id ?? null;
  const valid = state.users.some(u => u.id === storedId) ? storedId : fallback;
  if (valid == null) return;
  state.currentUserId = valid;
  localStorage.setItem(USER_STORAGE_KEY, String(valid));
  const el = document.getElementById('userSelect');
  if (el) el.value = String(valid);
  loadSessions();
}
```

- [ ] **Step 3: Normalize user switching to numeric IDs**

Replace `handleUserChange` with:

```js
function handleUserChange(userId) {
  const numericUserId = Number(userId);
  if (!Number.isFinite(numericUserId)) return;
  state.currentUserId = numericUserId;
  localStorage.setItem(USER_STORAGE_KEY, String(numericUserId));
  state.sessionId = null;
  state.sessions = [];
  state.messages = [];
  document.getElementById('messages').style.display = 'none';
  document.getElementById('welcome').style.display = 'flex';
  document.getElementById('chatTitle').textContent = 'New Chat';
  loadSessions();
}
```

- [ ] **Step 4: Normalize selected session IDs to numeric IDs**

Update `selectSession` and session creation flow to keep numeric semantics:

```js
function selectSession(id) {
  state.sessionId = Number(id);
  state.messages = [];
  const session = state.sessions.find(s => s.id === state.sessionId);
  if (session) document.getElementById('chatTitle').textContent = session.title;
  document.getElementById('welcome').style.display = 'none';
  document.getElementById('messages').style.display = 'block';
  renderSessions();
  loadHistory(state.sessionId);
}
```

In `renderSessions()`, use string interpolation but pass numeric IDs through:

```js
<div class="session-item ${s.id === state.sessionId ? 'active' : ''}" onclick="selectSession(${s.id})">
```

- [ ] **Step 5: Verify browser behavior manually**

Run the app and verify:
- create a user and confirm `localStorage.getItem('chat_user_id')` stores a numeric string like `"1"`
- switch user and confirm `state.currentUserId` is a number in the console
- create a session and confirm `state.sessionId` is a number in the console
- load session history successfully after switching users

- [ ] **Step 6: Commit the frontend bigint ID handling**

```bash
git add frontend/index.html
git commit -m "feat: switch frontend user and session ids to bigint semantics"
```

---

### Task 5: Final regression verification and migration-readiness review

**Files:**
- Modify: only if verification exposes a minimal bug

- [ ] **Step 1: Run the full focused Python suite**

Run: `python -m unittest tests.test_user_router tests.test_session_router tests.test_memory_service tests.test_chat_chain tests.test_rag_chain -v`
Expected: PASS

- [ ] **Step 2: Run syntax verification for all touched Python modules**

Run: `python -m py_compile /Users/zyb/PycharmProjects/CCode/models/user.py /Users/zyb/PycharmProjects/CCode/models/chat_session.py /Users/zyb/PycharmProjects/CCode/models/chat_message.py /Users/zyb/PycharmProjects/CCode/routers/user.py /Users/zyb/PycharmProjects/CCode/routers/session.py /Users/zyb/PycharmProjects/CCode/services/memory_service.py /Users/zyb/PycharmProjects/CCode/chains/chat_chain.py /Users/zyb/PycharmProjects/CCode/chains/rag_chain.py /Users/zyb/PycharmProjects/CCode/main.py`
Expected: exit code 0

- [ ] **Step 3: Review the final diff for intended files only**

Run: `git diff -- models/user.py models/chat_session.py models/chat_message.py routers/user.py routers/session.py services/memory_service.py chains/chat_chain.py chains/rag_chain.py frontend/index.html tests/test_user_router.py tests/test_session_router.py tests/test_memory_service.py tests/test_chat_chain.py tests/test_rag_chain.py`
Expected: only bigint migration work appears

- [ ] **Step 4: Verify against the spec checklist**

Confirm this checklist against code and manual verification:

```text
[ ] users.id is bigint/autoincrement in ORM
[ ] chat_sessions.id is bigint/autoincrement in ORM
[ ] chat_messages.id is bigint/autoincrement in ORM
[ ] chat_sessions.user_id is bigint in ORM
[ ] chat_messages.session_id is bigint in ORM
[ ] router request/response IDs are integer semantics
[ ] memory service uses integer session IDs
[ ] chains use integer session IDs
[ ] frontend uses numeric user/session IDs
[ ] user/session/history flows still work
```

- [ ] **Step 5: Do not claim DB migration complete without actual migration script execution**

If the repository still lacks executed schema migration scripts, explicitly report that the code-level bigint migration is done but live database migration remains to be executed against the target MySQL instance.

- [ ] **Step 6: Commit only if a final verification fix was needed**

If no additional fix was required, do not create another commit.
If a minimal verification fix was required, commit it with:

```bash
git add models/user.py models/chat_session.py models/chat_message.py routers/user.py routers/session.py services/memory_service.py chains/chat_chain.py chains/rag_chain.py frontend/index.html tests/test_user_router.py tests/test_session_router.py tests/test_memory_service.py tests/test_chat_chain.py tests/test_rag_chain.py
git commit -m "fix: polish bigint id migration flow"
```

---

## Self-Review

- **Spec coverage:** The plan covers bigint conversion for models, references, routers, memory, chains, frontend handling, and regression verification. It also explicitly calls out that actual DB migration execution is a separate verified step.
- **Placeholder scan:** No TBD/TODO placeholders remain; each task contains exact files, commands, and code.
- **Type consistency:** Uses `int` semantics consistently across `id`, `user_id`, and `session_id` in all described code paths.
