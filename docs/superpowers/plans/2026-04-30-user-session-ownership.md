# User Session Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight user layer so one user can own multiple chat sessions, with frontend user switching/creation and browser persistence of the most recently used user.

**Architecture:** Add a minimal `User` model and persist `user_id` on each chat session, then require all session reads and writes to be scoped by `user_id`. Keep memory and message storage session-scoped, while the frontend maintains the current user in `localStorage` and includes `user_id` in all session-related requests.

**Tech Stack:** FastAPI, SQLAlchemy ORM, Pydantic models, plain browser JavaScript, localStorage, unittest

---

## File Structure

- **Create** `models/user.py`
  - Define the `users` table and helper query methods
- **Modify** `models/chat_session.py`
  - Add `user_id` column and user-scoped lookup helpers
- **Modify** `routers/session.py`
  - Require `user_id` for session creation/list/history and enforce ownership filtering
- **Create** `routers/user.py`
  - Add user creation and listing endpoints
- **Modify** `main.py`
  - Register the new user router
- **Modify** `frontend/index.html`
  - Add current-user UI, localStorage handling, user creation flow, and pass `user_id` on session requests
- **Modify** `tests/test_memory_service.py` only if existing setup requires fixture compatibility after schema change
- **Create** focused tests for new behavior in `tests/test_session_router.py` and `tests/test_user_router.py`

---

### Task 1: Add the user model and session ownership field

**Files:**
- Create: `models/user.py`
- Modify: `models/chat_session.py`

- [ ] **Step 1: Write the failing model test**

Create `tests/test_user_router.py` with a minimal schema/behavior smoke test scaffold:

```python
import unittest

from models.chat_session import ChatSession


class TestChatSessionUserOwnership(unittest.TestCase):
    def test_chat_session_has_user_id_field(self):
        self.assertIn("user_id", ChatSession.__table__.columns)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_user_router.TestChatSessionUserOwnership.test_chat_session_has_user_id_field -v`
Expected: FAIL because `user_id` is not yet defined on `ChatSession`

- [ ] **Step 3: Create the user model**

Create `models/user.py` with:

```python
from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.orm import Session

from utils.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, comment="UUID for user")
    name = Column(String(255), nullable=False, comment="Display name")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")

    @classmethod
    def get_by_id(cls, db: Session, user_id: str):
        return db.query(cls).filter(cls.id == user_id).first()

    @classmethod
    def list_all(cls, db: Session):
        return db.query(cls).order_by(cls.created_at.asc()).all()
```

- [ ] **Step 4: Add `user_id` and helpers to `ChatSession`**

Modify `models/chat_session.py` to:

```python
from sqlalchemy import Column, String, DateTime, Integer, Text, func
from sqlalchemy.orm import Session

from utils.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, comment="UUID for session")
    user_id = Column(String(36), nullable=False, index=True, comment="Owner user ID")
    title = Column(String(255), default="New Chat", comment="Session title")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="Last update time")
    message_count = Column(Integer, default=0, comment="Total messages in session")
    summary = Column(Text, nullable=True, default=None, comment="Conversation summary for long-term memory")
    summary_sequence = Column(Integer, default=0, comment="Last message sequence covered by summary")

    @classmethod
    def get_by_id(cls, db: Session, session_id: str):
        return db.query(cls).filter(cls.id == session_id).first()

    @classmethod
    def get_by_user_and_id(cls, db: Session, user_id: str, session_id: str):
        return db.query(cls).filter(cls.user_id == user_id).filter(cls.id == session_id).first()

    @classmethod
    def list_by_user(cls, db: Session, user_id: str):
        return db.query(cls).filter(cls.user_id == user_id).order_by(cls.updated_at.desc()).all()

    @classmethod
    def delete_by_id(cls, db: Session, session_id: str):
        return db.query(cls).filter(cls.id == session_id).delete()
```

- [ ] **Step 5: Re-run the model test to verify it passes**

Run: `python -m unittest tests.test_user_router.TestChatSessionUserOwnership.test_chat_session_has_user_id_field -v`
Expected: PASS

- [ ] **Step 6: Commit the model changes**

```bash
git add models/user.py models/chat_session.py tests/test_user_router.py
git commit -m "feat: add user ownership to chat sessions"
```

---

### Task 2: Add user APIs and user-scoped session APIs

**Files:**
- Create: `routers/user.py`
- Modify: `routers/session.py`
- Modify: `main.py`
- Test: `tests/test_user_router.py`
- Test: `tests/test_session_router.py`

- [ ] **Step 1: Write the failing API tests**

Create `tests/test_user_router.py` with:

```python
import unittest
from unittest.mock import MagicMock, patch

from routers.user import CreateUserRequest, create_user, list_users


class TestUserRouter(unittest.TestCase):
    def test_create_user_returns_created_user(self):
        db = MagicMock()
        req = CreateUserRequest(name="Alice")

        result = create_user(req, db)

        self.assertEqual(result.name, "Alice")
```

Create `tests/test_session_router.py` with:

```python
import unittest
from unittest.mock import MagicMock, patch

from routers.session import CreateSessionRequest, create_session, list_sessions, get_history


class TestSessionRouter(unittest.TestCase):
    @patch("routers.session.ChatSession")
    def test_list_sessions_filters_by_user_id(self, mock_chat_session):
        db = MagicMock()
        mock_session = MagicMock(id="s1", title="T1", message_count=0, summary_sequence=0)
        mock_chat_session.list_by_user.return_value = [mock_session]

        result = list_sessions("u1", db)

        mock_chat_session.list_by_user.assert_called_once_with(db, "u1")
        self.assertEqual(len(result), 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_user_router tests.test_session_router -v`
Expected: FAIL because `routers.user` does not exist and session router signatures are not yet user-scoped

- [ ] **Step 3: Create the user router**

Create `routers/user.py` with:

```python
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.user import User
from utils.db import get_db

router = APIRouter(prefix="/api/user", tags=["user"])


class CreateUserRequest(BaseModel):
    name: str


class UserResponse(BaseModel):
    id: str
    name: str
    created_at: object | None = None


@router.post("", response_model=UserResponse)
def create_user(req: CreateUserRequest, db: Session = Depends(get_db)):
    user = User(id=str(uuid4()), name=req.name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(id=user.id, name=user.name, created_at=user.created_at)


@router.get("", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    users = User.list_all(db)
    return [UserResponse(id=u.id, name=u.name, created_at=u.created_at) for u in users]
```

- [ ] **Step 4: Make session endpoints user-scoped**

Modify `routers/session.py` so `CreateSessionRequest` becomes:

```python
class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    user_id: str
```

Change `create_session` to:

```python
def create_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    session = ChatSession(
        id=str(uuid4()),
        user_id=req.user_id,
        title=req.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse(
        id=session.id,
        title=session.title,
        message_count=session.message_count,
        summary_sequence=session.summary_sequence or 0,
    )
```

Change list endpoint to:

```python
@router.get("", response_model=list[SessionResponse])
def list_sessions(user_id: str = Query(...), db: Session = Depends(get_db)):
    sessions = ChatSession.list_by_user(db, user_id)
    return [
        SessionResponse(id=s.id, title=s.title, message_count=s.message_count, summary_sequence=s.summary_sequence or 0)
        for s in sessions
    ]
```

Change history endpoint to:

```python
@router.get("/{session_id}/history", response_model=list[MessageResponse])
def get_history(session_id: str, user_id: str = Query(...), db: Session = Depends(get_db)):
    session = ChatSession.get_by_user_and_id(db, user_id, session_id)
    if not session:
        return []
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.sequence.asc())
        .all()
    )
    return [MessageResponse(role=m.role, content=m.content, sequence=m.sequence) for m in messages]
```

- [ ] **Step 5: Register the user router**

Modify `main.py` imports and router registration:

```python
from routers.user import router as user_router  # noqa: E402

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(knowledge_router)
app.include_router(user_router)
```

- [ ] **Step 6: Re-run API tests to verify they pass**

Run: `python -m unittest tests.test_user_router tests.test_session_router -v`
Expected: PASS

- [ ] **Step 7: Commit the API changes**

```bash
git add routers/user.py routers/session.py main.py tests/test_user_router.py tests/test_session_router.py
git commit -m "feat: add user scoped session APIs"
```

---

### Task 3: Add frontend user switching, creation, and persistence

**Files:**
- Modify: `frontend/index.html`

- [ ] **Step 1: Write the failing browser behavior check**

Open the app and run this in the browser console before implementation:

```js
localStorage.getItem('chat_user_id')
```

Expected before implementation: no user selector exists in the UI and session requests are not scoped by a stored user ID

- [ ] **Step 2: Extend frontend state for users**

In the `state` object, add:

```js
  currentUserId: null,
  users: [],
```

So the full state block includes these fields alongside the existing session/message fields.

- [ ] **Step 3: Add user controls to the header**

In `frontend/index.html`, inside the header control area, insert a compact user switcher and create button above or beside provider controls using existing styling patterns:

```html
<select class="provider-select" id="userSelect" onchange="handleUserChange(this.value)"></select>
<button class="rag-toggle" onclick="promptCreateUser()" title="Create User">USER</button>
```

- [ ] **Step 4: Add user loading, selection, and persistence helpers**

Add these functions into the script section:

```js
const USER_STORAGE_KEY = 'chat_user_id';

async function loadUsers() {
  const res = await fetch(`${API}/api/user`);
  if (!res.ok) return;
  state.users = await res.json();
  renderUserSelect();
  restoreCurrentUser();
}

function renderUserSelect() {
  const el = document.getElementById('userSelect');
  if (!el) return;
  el.innerHTML = state.users.map(u => `<option value="${u.id}">${escHtml(u.name)}</option>`).join('');
  if (state.currentUserId) {
    el.value = state.currentUserId;
  }
}

function restoreCurrentUser() {
  const stored = localStorage.getItem(USER_STORAGE_KEY);
  const fallback = state.users[0]?.id || null;
  const valid = state.users.some(u => u.id === stored) ? stored : fallback;
  if (!valid) return;
  state.currentUserId = valid;
  localStorage.setItem(USER_STORAGE_KEY, valid);
  const el = document.getElementById('userSelect');
  if (el) el.value = valid;
  loadSessions();
}

function handleUserChange(userId) {
  state.currentUserId = userId;
  localStorage.setItem(USER_STORAGE_KEY, userId);
  state.sessionId = null;
  state.sessions = [];
  state.messages = [];
  document.getElementById('messages').style.display = 'none';
  document.getElementById('welcome').style.display = 'flex';
  document.getElementById('chatTitle').textContent = 'New Chat';
  loadSessions();
}

async function promptCreateUser() {
  const name = window.prompt('Enter user name');
  if (!name || !name.trim()) return;
  const res = await fetch(`${API}/api/user`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name.trim() }),
  });
  if (!res.ok) {
    toast('Failed to create user', 'error');
    return;
  }
  const user = await res.json();
  state.users.push(user);
  renderUserSelect();
  handleUserChange(user.id);
  toast('User created', 'success');
}
```

- [ ] **Step 5: Make initialization and session requests user-aware**

Update `DOMContentLoaded` init to call `loadUsers()` instead of directly calling `loadSessions()`:

```js
document.addEventListener('DOMContentLoaded', () => {
  loadUsers();
  const input = document.getElementById('messageInput');
  input.addEventListener('input', () => {
    document.getElementById('sendBtn').disabled = !input.value.trim();
    document.getElementById('charCount').textContent = input.value.length;
  });
});
```

Update session list loading:

```js
async function loadSessions() {
  if (!state.currentUserId) return;
  try {
    const res = await fetch(`${API}/api/session?user_id=${encodeURIComponent(state.currentUserId)}`);
    if (res.ok) {
      state.sessions = await res.json();
      renderSessions();
    }
  } catch (e) { /* no sessions yet */ }
}
```

Update session creation body:

```js
body: JSON.stringify({ title: 'New Chat', user_id: state.currentUserId }),
```

Update history loading:

```js
const res = await fetch(`${API}/api/session/${sessionId}/history?user_id=${encodeURIComponent(state.currentUserId)}`);
```

- [ ] **Step 6: Verify the frontend behavior manually**

Run the app and verify these exact flows:
- create two users from the UI
- switch between users and confirm the session list changes
- create a session under user A, then switch to user B and confirm user A's session is not listed
- refresh the browser and confirm the last selected user is restored from `localStorage`

- [ ] **Step 7: Commit the frontend user UX changes**

```bash
git add frontend/index.html
git commit -m "feat: add frontend user switching"
```

---

### Task 4: Final regression verification

**Files:**
- Modify: only if verification reveals a minimal bug

- [ ] **Step 1: Run the full focused test suite**

Run: `python -m unittest tests.test_user_router tests.test_session_router tests.test_memory_service -v`
Expected: PASS

- [ ] **Step 2: Run syntax verification for touched Python modules**

Run: `python -m py_compile /Users/zyb/PycharmProjects/CCode/models/user.py /Users/zyb/PycharmProjects/CCode/models/chat_session.py /Users/zyb/PycharmProjects/CCode/routers/user.py /Users/zyb/PycharmProjects/CCode/routers/session.py /Users/zyb/PycharmProjects/CCode/main.py`
Expected: exit code 0

- [ ] **Step 3: Review the final diff for intended files only**

Run: `git diff -- models/user.py models/chat_session.py routers/user.py routers/session.py main.py frontend/index.html tests/test_user_router.py tests/test_session_router.py`
Expected: only user/session ownership work appears

- [ ] **Step 4: Verify against the spec checklist**

Confirm this checklist against code and manual verification:

```text
[ ] users table/model added
[ ] chat_sessions has user_id
[ ] create user API works
[ ] list users API works
[ ] session creation requires user_id
[ ] session listing is user-scoped
[ ] history lookup validates user ownership
[ ] frontend can create users
[ ] frontend can switch users
[ ] localStorage restores last user
[ ] existing memory logic remains session-scoped
```

- [ ] **Step 5: Commit only if a final verification fix was needed**

If no additional code changes were required, do not create another commit.
If a final bug fix was required, commit it with:

```bash
git add models/user.py models/chat_session.py routers/user.py routers/session.py main.py frontend/index.html tests/test_user_router.py tests/test_session_router.py
git commit -m "fix: polish user session ownership flow"
```

---

## Self-Review

- **Spec coverage:** The plan covers the new user model, session ownership, user APIs, user-scoped session APIs, frontend user create/switch UX, localStorage persistence, and explicit non-goals around auth and user-level memory.
- **Placeholder scan:** No TBD/TODO placeholders remain; each task includes exact files, commands, code, and expected outcomes.
- **Type consistency:** Uses consistent names across tasks: `User`, `user_id`, `CreateUserRequest`, `UserResponse`, `loadUsers`, `renderUserSelect`, `handleUserChange`, `promptCreateUser`.
