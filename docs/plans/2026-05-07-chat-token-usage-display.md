# Chat Token Usage Display Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add stable input/output/analysis token usage reporting for chat responses across synchronous API responses, streaming SSE responses, server logs, and the frontend chat UI.

**Architecture:** Keep the existing chat and RAG flow intact, but change the chain layer to return both response text and normalized usage data instead of only raw strings. Add one lightweight usage normalizer shared by `ChatChain` and `RAGChain`, expose the normalized structure from `routers/chat.py`, and teach the frontend stream handler and renderer to attach usage to assistant messages.

**Tech Stack:** Python 3.11, FastAPI, LangChain, Pydantic, unittest/pytest, vanilla HTML/CSS/JavaScript frontend

---

### Task 1: Add normalized usage model and extractor

**Files:**
- Create: `services/token_usage.py`
- Test: `tests/test_token_usage.py`

**Step 1: Write the failing tests**

Create `tests/test_token_usage.py` with focused cases for usage normalization:

```python
import unittest

from services.token_usage import TokenUsage, normalize_token_usage


class TestNormalizeTokenUsage(unittest.TestCase):
    def test_returns_zero_usage_for_none(self):
        usage = normalize_token_usage(None)

        self.assertEqual(usage, TokenUsage(input_tokens=0, output_tokens=0, analysis_tokens=0))

    def test_reads_usage_metadata_fields(self):
        response = type("Resp", (), {
            "usage_metadata": {
                "input_tokens": 11,
                "output_tokens": 7,
            }
        })()

        usage = normalize_token_usage(response)

        self.assertEqual(usage.input_tokens, 11)
        self.assertEqual(usage.output_tokens, 7)
        self.assertEqual(usage.analysis_tokens, 0)

    def test_reads_response_metadata_token_usage_and_reasoning(self):
        response = type("Resp", (), {
            "response_metadata": {
                "token_usage": {
                    "prompt_tokens": 23,
                    "completion_tokens": 9,
                    "output_tokens_details": {
                        "reasoning_tokens": 4,
                    },
                }
            }
        })()

        usage = normalize_token_usage(response)

        self.assertEqual(usage, TokenUsage(input_tokens=23, output_tokens=9, analysis_tokens=4))

    def test_prefers_reasoning_tokens_when_present(self):
        response = type("Resp", (), {
            "response_metadata": {
                "reasoning_tokens": 5,
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                },
            }
        })()

        usage = normalize_token_usage(response)

        self.assertEqual(usage.analysis_tokens, 5)
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_token_usage.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing `TokenUsage` / `normalize_token_usage`.

**Step 3: Write minimal implementation**

Create `services/token_usage.py` with a small, explicit implementation:

```python
from dataclasses import dataclass
from typing import Any


@dataclass(eq=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    analysis_tokens: int = 0


def _read_path(data: Any, *path: str):
    current = data
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
            continue
        current = getattr(current, key, None)
    return current


def _to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def normalize_token_usage(response: Any) -> TokenUsage:
    input_tokens = (
        _read_path(response, "usage_metadata", "input_tokens")
        or _read_path(response, "response_metadata", "token_usage", "prompt_tokens")
        or _read_path(response, "token_usage", "prompt_tokens")
        or _read_path(response, "response_metadata", "input_tokens")
        or 0
    )
    output_tokens = (
        _read_path(response, "usage_metadata", "output_tokens")
        or _read_path(response, "response_metadata", "token_usage", "completion_tokens")
        or _read_path(response, "token_usage", "completion_tokens")
        or _read_path(response, "response_metadata", "output_tokens")
        or 0
    )
    analysis_tokens = (
        _read_path(response, "response_metadata", "reasoning_tokens")
        or _read_path(response, "response_metadata", "output_tokens_details", "reasoning_tokens")
        or _read_path(response, "response_metadata", "token_usage", "output_tokens_details", "reasoning_tokens")
        or _read_path(response, "usage_metadata", "reasoning_tokens")
        or 0
    )
    return TokenUsage(
        input_tokens=_to_int(input_tokens),
        output_tokens=_to_int(output_tokens),
        analysis_tokens=_to_int(analysis_tokens),
    )
```

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_token_usage.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_token_usage.py services/token_usage.py
git commit -m "feat: normalize chat token usage metadata"
```

---

### Task 2: Return response text and usage from ChatChain

**Files:**
- Modify: `chains/chat_chain.py`
- Modify: `tests/test_chat_chain.py`
- Test: `tests/test_token_usage.py`

**Step 1: Write the failing tests**

Extend `tests/test_chat_chain.py` with explicit return-shape checks:

```python
class TestChatChainUsage(unittest.TestCase):
    @patch("chains.chat_chain.LLMFactory.create")
    def test_invoke_returns_response_and_usage(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary", [])

        llm = MagicMock()
        mock_create.return_value = llm

        response = MagicMock()
        response.content = "answer"
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 4}

        with patch("chains.chat_chain.ChatPromptTemplate.from_messages") as mock_from_messages:
            prompt = MagicMock()
            model_chain = MagicMock()
            mock_from_messages.return_value = prompt
            prompt.__or__.return_value = model_chain
            model_chain.invoke.return_value = response

            chain = ChatChain(memory, provider="siliconflow")
            result = chain.invoke("sess-1", "hello")

        self.assertEqual(result["response"], "answer")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 4)
        self.assertEqual(result["usage"]["analysis_tokens"], 0)

    @patch("chains.chat_chain.LLMFactory.create")
    async def test_astream_yields_chunks_and_returns_usage(self, mock_create):
        ...
```

For streaming, prefer a dedicated helper pattern instead of trying to assert a generator return value directly. Add a helper expectation such as:

```python
result = []
async for event in chain.astream("sess-1", "hello"):
    result.append(event)

self.assertEqual(result, [
    {"chunk": "hel"},
    {"chunk": "lo"},
    {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}},
])
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_chat_chain.py -v
```

Expected: FAIL because `invoke()` still returns `str` and `astream()` still yields raw strings.

**Step 3: Write minimal implementation**

Modify `chains/chat_chain.py` to:

1. Replace `StrOutputParser`-based output-only flow with direct model invocation.
2. Build prompt messages first, then call `llm.invoke(messages)`.
3. Read text from the model response with a small extractor that supports `.content` and string fallbacks.
4. Call `normalize_token_usage(...)` from `services/token_usage.py`.
5. Return this exact shape from `invoke()`:

```python
{
    "response": response_text,
    "usage": {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "analysis_tokens": usage.analysis_tokens,
    },
}
```

6. Change `astream()` to yield structured events instead of plain strings:

```python
yield {"chunk": chunk_text}
...
yield {"usage": usage_dict}
```

7. Keep memory writes unchanged in semantics:
   - add user message before stream generation
   - add assistant message after full text is assembled

8. Add usage logging with the existing `logger.info(...)` style.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_chat_chain.py tests/test_token_usage.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add chains/chat_chain.py tests/test_chat_chain.py tests/test_token_usage.py
git commit -m "feat: include token usage in chat chain responses"
```

---

### Task 3: Return response text and usage from RAGChain

**Files:**
- Modify: `chains/rag_chain.py`
- Modify: `tests/test_rag_chain.py`
- Test: `tests/test_token_usage.py`

**Step 1: Write the failing tests**

Extend `tests/test_rag_chain.py` to mirror the chat-chain contract:

```python
class TestRAGChainUsage(unittest.TestCase):
    @patch("chains.rag_chain.LLMFactory.create")
    def test_invoke_returns_response_and_usage(self, mock_create):
        memory = MagicMock()
        memory.get_context_parts.return_value = ("summary", [])

        rag_service = MagicMock()
        doc = MagicMock(page_content="kb-content")
        rag_service.query.return_value = [doc]

        response = MagicMock()
        response.content = "rag-answer"
        response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "output_tokens_details": {"reasoning_tokens": 3},
            }
        }

        ...
        self.assertEqual(result["response"], "rag-answer")
        self.assertEqual(result["usage"], {
            "input_tokens": 15,
            "output_tokens": 8,
            "analysis_tokens": 3,
        })
```

Add a streaming test similar to `ChatChain` so `astream()` yields chunk events followed by one usage event.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_rag_chain.py -v
```

Expected: FAIL because `RAGChain` still returns text-only results.

**Step 3: Write minimal implementation**

Modify `chains/rag_chain.py` to match the `ChatChain` contract:

1. Stop parsing directly to `str` before usage is captured.
2. Invoke the model with prompt messages.
3. Extract final text.
4. Normalize usage with `normalize_token_usage(...)`.
5. Return `{"response": ..., "usage": ...}` from `invoke()`.
6. Yield `{"chunk": ...}` events followed by `{"usage": ...}` from `astream()`.
7. Add usage logging in the same style as `ChatChain`.
8. Keep RAG retrieval and memory side effects unchanged.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_rag_chain.py tests/test_token_usage.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add chains/rag_chain.py tests/test_rag_chain.py tests/test_token_usage.py
git commit -m "feat: include token usage in rag chain responses"
```

---

### Task 4: Expose usage in chat router responses and SSE events

**Files:**
- Modify: `routers/chat.py`
- Create: `tests/test_chat_router.py`
- Test: `main.py`

**Step 1: Write the failing tests**

Create `tests/test_chat_router.py` using `fastapi.testclient.TestClient` against `main.app`:

```python
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from main import app


class TestChatRouter(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("routers.chat.ChatChain")
    def test_chat_returns_usage(self, mock_chain_cls):
        chain = MagicMock()
        chain.invoke.return_value = {
            "response": "answer",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "analysis_tokens": 2,
            },
        }
        mock_chain_cls.return_value = chain

        response = self.client.post("/api/chat", json={
            "session_id": "sess-1",
            "message": "hello",
            "use_rag": False,
            "provider": "siliconflow",
            "rag_k": 4,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["usage"]["analysis_tokens"], 2)

    @patch("routers.chat.ChatChain")
    def test_chat_stream_emits_usage_before_done(self, mock_chain_cls):
        async def fake_stream(*args, **kwargs):
            yield {"chunk": "hel"}
            yield {"chunk": "lo"}
            yield {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}}

        chain = MagicMock()
        chain.astream = fake_stream
        mock_chain_cls.return_value = chain

        with self.client.stream("POST", "/api/chat/stream", json={...}) as response:
            body = b"".join(response.iter_bytes()).decode()

        self.assertIn('data: {"chunk": "hel"}', body)
        self.assertIn('data: {"usage": {"input_tokens": 10, "output_tokens": 5, "analysis_tokens": 2}}', body)
        self.assertTrue(body.strip().endswith('data: [DONE]'))
```

Include one RAG-path test to confirm the router returns the same response shape when `use_rag=True`.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_chat_router.py -v
```

Expected: FAIL because `ChatResponse` does not include `usage` and the stream emits raw chunks only.

**Step 3: Write minimal implementation**

Modify `routers/chat.py` to:

1. Add a Pydantic model for usage:

```python
class TokenUsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    analysis_tokens: int
```

2. Extend `ChatResponse`:

```python
class ChatResponse(BaseModel):
    session_id: str
    response: str
    usage: TokenUsageResponse
```

3. Update the sync endpoint to use the chain result directly:

```python
result = chain.invoke(...)
return ChatResponse(session_id=req.session_id, response=result["response"], usage=result["usage"])
```

4. Update the stream endpoint to serialize the structured events returned by the chain:

```python
async for event in chain.astream(...):
    yield f"data: {json.dumps(event)}\n\n"
yield "data: [DONE]\n\n"
```

5. Keep route selection logic (`use_rag`) unchanged.

**Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_chat_router.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add routers/chat.py tests/test_chat_router.py
git commit -m "feat: expose token usage in chat api responses"
```

---

### Task 5: Render usage in the frontend message list and stream updates

**Files:**
- Modify: `frontend/index.html`
- Test: `frontend/index.html` (manual browser verification)

**Step 1: Write the failing behavior checklist**

Document the expected failures before coding:

1. Existing stream handler ignores any `{usage: ...}` SSE event.
2. `state.messages` assistant entries do not store usage.
3. `renderMessages()` cannot render token metadata.
4. `updateLastMessage()` only updates content, not usage.

Manual reproduction command:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected current behavior: assistant messages render text only; no token usage line appears.

**Step 2: Implement the minimal frontend changes**

Modify `frontend/index.html` in place:

1. Add a small style block for usage metadata near the message styles:

```css
.message-usage {
  margin-top: 8px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-muted);
}
```

2. Add a helper:

```javascript
function renderUsage(usage) {
  if (!usage) return '';
  return `<div class="message-usage">Token：输入 ${usage.input_tokens ?? 0} / 输出 ${usage.output_tokens ?? 0} / 分析 ${usage.analysis_tokens ?? 0}</div>`;
}
```

3. Update assistant rendering in `renderMessages()`:

```javascript
<div class="message-content ...">...</div>
${m.role === 'assistant' ? renderUsage(m.usage) : ''}
```

4. When inserting the assistant placeholder during streaming, include usage storage:

```javascript
state.messages.push({ role: 'assistant', content: '', usage: null, sequence: state.messages.length });
```

5. In the stream parser, handle usage events:

```javascript
if (parsed.usage) {
  const lastMsg = state.messages[state.messages.length - 1];
  lastMsg.usage = parsed.usage;
  updateLastMessage(lastMsg.content, lastMsg.usage);
}
```

6. Update `updateLastMessage` signature so it can refresh both content and usage in the last assistant message DOM node.

7. When loading history, keep behavior backward compatible: if older history entries have no usage, render nothing.

**Step 3: Run manual verification**

Run:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Manual checks:

1. Send a normal chat message.
2. Confirm assistant response shows `Token：输入 X / 输出 Y / 分析 Z`.
3. Enable RAG and send another message.
4. Confirm the usage line still appears.
5. Confirm streaming text still updates progressively before the final usage line appears.

Expected: all checks PASS.

**Step 4: Commit**

```bash
git add frontend/index.html
git commit -m "feat: display token usage in chat ui"
```

---

### Task 6: Run regression checks and review against the spec

**Files:**
- Modify: none unless fixes are required
- Test: `tests/test_token_usage.py`
- Test: `tests/test_chat_chain.py`
- Test: `tests/test_rag_chain.py`
- Test: `tests/test_chat_router.py`
- Review: `docs/superpowers/specs/2026-05-07-chat-chain-token-usage-design.md`

**Step 1: Run the targeted automated test suite**

Run:

```bash
pytest tests/test_token_usage.py tests/test_chat_chain.py tests/test_rag_chain.py tests/test_chat_router.py -v
```

Expected: PASS.

**Step 2: Run a broader regression slice**

Run:

```bash
pytest tests/test_llm_factory.py tests/test_session_router.py tests/test_user_router.py -v
```

Expected: PASS.

**Step 3: Review implementation against the approved spec**

Verify each approved requirement from `docs/superpowers/specs/2026-05-07-chat-chain-token-usage-design.md`:

- sync chat endpoint returns `usage`
- stream endpoint emits `usage` before `[DONE]`
- logs include input/output/analysis tokens
- frontend shows token usage below assistant messages
- provider compatibility falls back to zero when reasoning tokens are absent
- no database or memory semantics changed

**Step 4: Fix any regression only if tests or review fail**

If any check fails, make the smallest possible fix in the affected file and rerun only the failing command first, then rerun the full Task 6 suite.

**Step 5: Commit**

```bash
git add chains/chat_chain.py chains/rag_chain.py routers/chat.py frontend/index.html services/token_usage.py tests/test_token_usage.py tests/test_chat_chain.py tests/test_rag_chain.py tests/test_chat_router.py
git commit -m "feat: add chat token usage reporting"
```

---

## Notes for the implementing engineer

- Do not add database columns or persistence for usage.
- Do not move SQL anywhere; this feature does not require SQL changes.
- Prefer one shared normalization module over duplicating token extraction logic in both chains.
- Keep frontend changes minimal and local to the current message rendering flow.
- Preserve current memory side effects and stream completion semantics.
- Before claiming completion, use `superpowers:verification-before-completion`.
