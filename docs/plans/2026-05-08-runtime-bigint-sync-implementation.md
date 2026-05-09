# Runtime Bigint ID Synchronization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Synchronize runtime code with the already-migrated bigint database schema for chat/user/session/knowledge tables while preserving string-based Milvus `doc_id` metadata.

**Architecture:** Keep database-generated bigint primary keys as the source of truth for `users`, `chat_sessions`, `chat_messages`, `knowledge_bases`, and `document_metas`, and stop Python runtime code from explicitly writing UUID/string primary keys into those tables. Preserve the separate business `doc_id` string in Milvus metadata so vector-search identifiers remain decoupled from database primary keys.

**Tech Stack:** Python 3.11, SQLAlchemy, FastAPI, Pydantic, pytest/unittest, PostgreSQL-compatible runtime database

---

### Task 1: Add failing tests for bigint runtime writes in memory and session flows

**Files:**
- Modify: `tests/test_chat_chain.py`
- Modify: `tests/test_session_router.py`
- Test: `tests/test_chat_chain.py`
- Test: `tests/test_session_router.py`

**Step 1: Write the failing tests**

Add a focused test in `tests/test_chat_chain.py` that proves runtime message persistence no longer writes UUID IDs explicitly. Since `ChatChain.invoke()` calls memory persistence, patch the DB layer around the persistence point if needed, or write the lower-level test in `tests/test_memory_service.py` if that file already owns DB persistence behavior.

Preferred concrete test in `tests/test_memory_service.py` if message persistence is implemented there:

```python
def test_add_message_does_not_set_explicit_uuid_id(self):
    db = MagicMock()
    session = MagicMock(id=1, message_count=0, summary_sequence=0)
    db.query.return_value.filter.return_value.first.return_value = session

    memory_service.add_message(1, "user", "hello", db=db)

    saved_message = db.add.call_args.args[0]
    assert saved_message.session_id == 1
    assert saved_message.role == "user"
    assert saved_message.content == "hello"
    assert getattr(saved_message, "id", None) is None
```

Add a session-router contract test in `tests/test_session_router.py`:

```python
def test_create_session_does_not_require_explicit_string_id(self):
    db = MagicMock()
    req = CreateSessionRequest(title="Chat", user_id=1)

    def refresh_side_effect(session):
        session.id = 101
        session.message_count = 0
        session.summary_sequence = 0

    db.refresh.side_effect = refresh_side_effect

    result = create_session(req, db)

    created_session = db.add.call_args.args[0]
    assert getattr(created_session, "id", None) is None
    assert created_session.user_id == 1
    assert result.id == 101
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_memory_service.py tests/test_session_router.py -v
```

Expected: FAIL because the current runtime path still sets string/UUID IDs explicitly in at least one place.

**Step 3: Write minimal implementation**

Modify the actual runtime write path so database-managed bigint IDs are not set manually:

- In the message persistence path, remove `id=str(uuid4())` when constructing `ChatMessage(...)`
- In session creation, ensure `ChatSession(...)` is created without explicit `id`
- Keep `session_id` / `user_id` integer usage unchanged

Minimal target shape:

```python
chat_msg = ChatMessage(
    session_id=session_id,
    role=role,
    content=content,
    sequence=next_sequence,
)
```

and:

```python
session = ChatSession(
    user_id=req.user_id,
    title=req.title,
)
```

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_memory_service.py tests/test_session_router.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_memory_service.py tests/test_session_router.py services/memory_service.py routers/session.py
git commit -m "fix: stop explicit uuid writes in chat runtime"
```

---

### Task 2: Add failing tests for bigint user and knowledge-base runtime writes

**Files:**
- Modify: `tests/test_session_router.py`
- Create: `tests/test_knowledge_router.py`
- Test: `tests/test_session_router.py`
- Test: `tests/test_knowledge_router.py`

**Step 1: Write the failing tests**

Add a user-router creation test if not already present, strengthening it to verify no string ID is set before refresh:

```python
def test_create_user_does_not_set_explicit_string_id(self):
    db = MagicMock()
    req = CreateUserRequest(name="wzw")

    def refresh_side_effect(user):
        user.id = 1
        user.created_at = None

    db.refresh.side_effect = refresh_side_effect

    result = create_user(req, db)

    created_user = db.add.call_args.args[0]
    assert getattr(created_user, "id", None) is None
    assert result.id == 1
```

Create `tests/test_knowledge_router.py` to verify bigint DB writes but string Milvus `doc_id`:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from routers.knowledge import KBCreateRequest, create_kb, upload_document


class TestKnowledgeRouter(unittest.TestCase):
    def test_create_kb_does_not_set_explicit_uuid_id(self):
        db = MagicMock()
        req = KBCreateRequest(name="KB", description="desc")

        def refresh_side_effect(kb):
            kb.id = 10
            kb.name = "KB"
            kb.description = "desc"

        db.refresh.side_effect = refresh_side_effect

        result = create_kb(req, db)

        created_kb = db.add.call_args.args[0]
        self.assertIsNone(getattr(created_kb, "id", None))
        self.assertEqual(result.id, 10)
```

Add a second test for upload persistence semantics:

```python
@patch("routers.knowledge.rag_service")
@patch("routers.knowledge.process_document")
def test_upload_document_uses_string_doc_id_for_metadata_but_not_db_primary_key(self, mock_process_document, mock_rag_service):
    ...
    saved_doc = db.add.call_args.args[0]
    self.assertIsNone(getattr(saved_doc, "id", None))
    metadata = mock_rag_service.add_texts.call_args.kwargs["metadatas"]
    self.assertIsInstance(metadata[0]["doc_id"], str)
    self.assertNotEqual(metadata[0]["doc_id"], "")
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_session_router.py tests/test_knowledge_router.py -v
```

Expected: FAIL because knowledge creation/upload still writes UUID DB IDs explicitly.

**Step 3: Write minimal implementation**

Modify runtime write paths only where bigint tables are now database-managed:

- `routers/user.py` should keep creating `User(name=...)` with no explicit `id`
- `routers/knowledge.py` should change:

```python
kb = KnowledgeBase(
    name=req.name,
    description=req.description,
)
```

and:

```python
doc_id = str(uuid4())
doc_meta = DocumentMeta(
    kb_id=kb_id,
    file_name=file.filename,
    file_path=tmp_path,
    status="indexed",
)
```

Then, after `db.flush()` or `db.commit()/refresh()`, if needed, use `doc_meta.id` for API response while keeping `doc_id` only inside Milvus metadata.

If `kb_id` is now bigint, update request/response models accordingly:

```python
class KBResponse(BaseModel):
    id: int
    ...

class DocumentResponse(BaseModel):
    id: int
    ...
```

and:

```python
kb_id: int = Form(...)
```

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_session_router.py tests/test_knowledge_router.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_session_router.py tests/test_knowledge_router.py routers/user.py routers/knowledge.py models/knowledge_base.py
git commit -m "fix: align user and knowledge writes with bigint ids"
```

---

### Task 3: Update model and router contracts for bigint knowledge IDs

**Files:**
- Modify: `models/knowledge_base.py`
- Modify: `routers/knowledge.py`
- Modify: `tests/test_knowledge_router.py`
- Test: `tests/test_knowledge_router.py`

**Step 1: Write the failing tests**

Strengthen `tests/test_knowledge_router.py` so it locks the bigint contract for DB-facing IDs:

```python
def test_kb_response_accepts_integer_id(self):
    resp = KBResponse(id=10, name="KB", description="desc")
    self.assertEqual(resp.id, 10)


def test_document_response_accepts_integer_id(self):
    resp = DocumentResponse(id=20, file_name="a.txt", status="indexed")
    self.assertEqual(resp.id, 20)
```

Add a list behavior test:

```python
@patch("routers.knowledge.KnowledgeBase")
def test_list_kbs_returns_integer_ids(self, mock_kb_cls):
    db = MagicMock()
    db.query.return_value.all.return_value = [SimpleNamespace(id=10, name="KB", description="desc")]

    result = list_kbs(db)

    self.assertEqual(result[0].id, 10)
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_knowledge_router.py -v
```

Expected: FAIL while response models or handler types still assume string IDs.

**Step 3: Write minimal implementation**

Update knowledge model/router contracts to align with bigint DB IDs:

```python
class KnowledgeBase(Base):
    id = Column(BigInteger, primary_key=True, autoincrement=True, ...)

class DocumentMeta(Base):
    id = Column(BigInteger, primary_key=True, autoincrement=True, ...)
    kb_id = Column(BigInteger, nullable=False, index=True, ...)
```

Router updates:

```python
class KBResponse(BaseModel):
    id: int

class DocumentResponse(BaseModel):
    id: int

async def upload_document(..., kb_id: int = Form(...), ...):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
```

Keep Milvus metadata `doc_id` as string business ID:

```python
metadata_doc_id = str(uuid4())
metadatas = [{"doc_id": metadata_doc_id, "kb_id": str(kb_id), "file_name": file.filename} for _ in chunks]
```

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_knowledge_router.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add models/knowledge_base.py routers/knowledge.py tests/test_knowledge_router.py
git commit -m "feat: switch knowledge db ids to bigint"
```

---

### Task 4: Verify `/api/user`, `/api/session`, and message persistence after runtime sync

**Files:**
- Modify: `tests/test_user_router.py`
- Modify: `tests/test_session_router.py`
- Modify: `tests/test_memory_service.py`
- Test: `tests/test_user_router.py`
- Test: `tests/test_session_router.py`
- Test: `tests/test_memory_service.py`

**Step 1: Write the failing tests**

Add explicit regression tests for the runtime contract after schema migration:

```python
def test_user_response_accepts_integer_id(self):
    resp = UserResponse(id=1, name="wzw", created_at=None)
    self.assertEqual(resp.id, 1)
```

```python
def test_create_session_response_returns_integer_id(self):
    ...
    self.assertIsInstance(result.id, int)
```

```python
def test_persisted_chat_message_uses_integer_session_id(self):
    ...
    saved_message = db.add.call_args.args[0]
    self.assertEqual(saved_message.session_id, 1)
```

If these already exist, strengthen them so they also verify no explicit string primary key assignment is present.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_user_router.py tests/test_session_router.py tests/test_memory_service.py -v
```

Expected: FAIL if any remaining runtime path still writes or assumes string IDs.

**Step 3: Write minimal implementation**

Make only the smallest changes needed so all router/model/runtime paths align with the bigint schema:

- Keep `UserResponse.id`, `CreateSessionRequest.user_id`, `SessionResponse.id` as `int`
- Ensure no explicit string ID assignment remains in user/session/message creation code
- Preserve all existing logic unrelated to ID generation

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_user_router.py tests/test_session_router.py tests/test_memory_service.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_user_router.py tests/test_session_router.py tests/test_memory_service.py services/memory_service.py routers/user.py routers/session.py
 git commit -m "test: lock bigint runtime contract"
```

---

### Task 5: Run live verification for recovered bigint runtime behavior

**Files:**
- Create: `scripts/verify_runtime_bigint_sync.py`
- Test: live local API

**Step 1: Write the failing verification helper**

Create `scripts/verify_runtime_bigint_sync.py` that checks the real runtime contract through the live API:

```python
import requests

base = "http://127.0.0.1:8000"
users = requests.get(f"{base}/api/user").json()
assert users and isinstance(users[0]["id"], int)
user_id = users[0]["id"]

sessions = requests.get(f"{base}/api/session", params={"user_id": user_id}).json()
assert all(isinstance(item["id"], int) for item in sessions)

if sessions:
    history = requests.get(f"{base}/api/session/{sessions[0]['id']}/history", params={"user_id": user_id}).json()
    assert isinstance(history, list)
```

Add a small runtime message-write check if there is already an API path available for sending a message without frontend involvement.

**Step 2: Run verification to observe current failure if any remains**

Run server:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then run:

```bash
python scripts/verify_runtime_bigint_sync.py
```

Expected before final fixes: if any runtime write path still uses UUIDs, the verification or a manual `/api/chat` call will fail.

**Step 3: Write minimal implementation**

If verification reveals any remaining bigint/runtime mismatch, fix only that exact path.

Examples:
- remove the last explicit UUID write
- correct a response model type
- convert a request field type from `str` to `int`

**Step 4: Run verification to prove it passes**

Run:

```bash
python scripts/verify_runtime_bigint_sync.py
```

Expected: exit 0 with no traceback.

Also manually verify a message write path if available:

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": 1, "message": "hello", "use_rag": false, "provider": "siliconflow", "rag_k": 4}'
```

Expected: no UUID/bigint insert error from `chat_messages`.

**Step 5: Commit**

```bash
git add scripts/verify_runtime_bigint_sync.py
 git commit -m "test: verify bigint runtime sync through live api"
```

---

### Task 6: Run full regression and compare against the agreed design

**Files:**
- Review: `docs/plans/2026-05-08-api-user-bigint-recovery-design.md`
- Test: all affected suites

**Step 1: Run targeted automated tests**

Run:

```bash
pytest tests/test_memory_service.py tests/test_session_router.py tests/test_user_router.py tests/test_knowledge_router.py -v
```

Expected: PASS.

**Step 2: Run broader application regression**

Run:

```bash
pytest tests/test_token_usage.py tests/test_chat_chain.py tests/test_rag_chain.py tests/test_chat_router.py tests/test_llm_factory.py tests/test_memory_service.py tests/test_session_router.py tests/test_user_router.py tests/test_knowledge_router.py -v
```

Expected: PASS.

**Step 3: Compare implementation against the agreed design**

Verify each requirement from the confirmed design:

- bigint DB IDs are used for `users`, `chat_sessions`, `chat_messages`, `knowledge_bases`, `document_metas`
- runtime no longer writes explicit UUID/string primary keys into bigint tables
- Milvus `doc_id` remains a string business identifier
- `/api/user` works again
- session and message flows still work
- knowledge-base creation/upload still works
- unrelated token-usage code was not modified

**Step 4: Fix only if evidence shows a gap**

If any test or live verification fails, make the smallest possible correction, rerun the failing command first, then rerun the full Task 6 suite.

**Step 5: Commit**

```bash
git add services/memory_service.py routers/session.py routers/user.py routers/knowledge.py models/knowledge_base.py tests/test_memory_service.py tests/test_session_router.py tests/test_user_router.py tests/test_knowledge_router.py scripts/verify_runtime_bigint_sync.py
 git commit -m "fix: sync runtime writes with bigint schema"
```

---

## Notes for the implementing engineer

- Keep SQL in the model/migration layer where appropriate; do not move SQL into routers.
- Do not touch Milvus `doc_id` semantics beyond preserving it as a string metadata field.
- Remove only the runtime UUID writes that target bigint DB columns.
- Prefer database-generated IDs over Python-side manual ID generation.
- Before claiming completion, use `superpowers:verification-before-completion`.
