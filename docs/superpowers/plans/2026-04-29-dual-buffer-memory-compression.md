# 双缓冲区记忆窗口压缩 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单列表短期记忆改为 S1/S2 双缓冲区压缩模型，确保摘要生成期间消息不丢失。

**Architecture:** 借鉴 JVM Survivor Space，将短期记忆分为 S1（活跃区）和 S2（压缩缓冲区）。S1 满时触发后台摘要生成，期间新消息写入 S2，压缩完成后将 S2 迁移到 S1 并更新摘要。`get_context()` 返回 `摘要 + S1 + S2`。

**Tech Stack:** Python 3.13, SQLAlchemy, LangChain, threading

---

## Task 1: 数据结构迁移 — 用 S1/S2 替代 _short_term

**Files:**
- Modify: `services/memory_service.py:1-43` (`__init__`、`_get_lock`、`get_context`)
- Modify: `tests/test_memory_service.py`（更新所有引用 `_short_term` 的测试）

**目标：** 将 `_short_term: dict[str, list[BaseMessage]]` 替换为三个新数据结构：
- `_buffers_s1: dict[str, list[BaseMessage]]` — 活跃缓冲区
- `_buffers_s2: dict[str, list[BaseMessage]]` — 压缩期间缓冲区
- `_compressing: set[str]` — 标记正在压缩的 session

- [ ] **Step 1: 更新现有测试以兼容新数据结构**

修改 `tests/test_memory_service.py`，将所有直接操作 `_short_term` 的代码改为操作 `_buffers_s1`：

```python
# 替换所有类似这样的代码：
# self.service._short_term[session_id] = [...]
# 改为：
self.service._buffers_s1[session_id] = [...]
```

具体修改点：
- `test_get_context_returns_all_messages_when_under_window` 第 38 行：`_short_term` → `_buffers_s1`
- `test_get_context_returns_sliding_window` 第 56 行：`_short_term` → `_buffers_s1`
- `test_add_message_user_to_short_term` 第 97 行：`_short_term` → `_buffers_s1`
- `test_add_message_assistant_to_short_term` 第 120 行：`_short_term` → `_buffers_s1`
- `test_add_message_multiple_messages_same_session` 第 211 行：`_short_term` → `_buffers_s1`
- `test_clear_session_removes_short_term_and_db` 第 305 行：`_short_term` → `_buffers_s1`，第 315 行：`_short_term` → `_buffers_s1`

- [ ] **Step 2: 运行测试确认通过**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 所有测试 PASS（暂时只是改了数据结构名）

- [ ] **Step 3: 重写 `__init__`**

修改 `services/memory_service.py` 第 26-36 行：

```python
    def __init__(self):
        self._config: Optional[MemoryConfig] = None
        # S1: active buffer — new messages written here until full
        self._buffers_s1: dict[str, list[BaseMessage]] = {}
        # S2: compression buffer — used while summary generation is in progress
        self._buffers_s2: dict[str, list[BaseMessage]] = {}
        # In-memory summary cache: session_id -> str
        self._summary_cache: dict[str, str] = {}
        # Per-session locks to prevent concurrent cache/summary races
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        # Track sessions currently generating summaries
        self._summary_in_progress: set[str] = set()
        # Track sessions currently compressing (S1 frozen, writes go to S2)
        self._compressing: set[str] = set()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 所有测试 PASS

- [ ] **Step 5: Commit**

```bash
git add services/memory_service.py tests/test_memory_service.py
git commit -m "refactor: replace _short_term with S1/S2 dual buffer data structures"
```

---

## Task 2: 重写 `add_message` — S1 满时触发压缩，写入 S2

**Files:**
- Modify: `services/memory_service.py:124-179`（`add_message` 方法）

**目标：** `add_message` 根据 `_compressing` 状态决定写入 S1 还是 S2，S1 满时触发后台压缩。

- [ ] **Step 1: 重写 `add_message` 方法**

替换 `services/memory_service.py` 第 124-179 行的整个 `add_message` 方法：

```python
    def add_message(self, session_id: str, role: str, content: str, db: Optional[Session] = None):
        """Add a message to both short-term buffer and long-term storage.

        During normal operation, messages are appended to S1.
        When S1 is full, background compression is triggered and new messages go to S2.
        """
        msg_obj = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        window = self.config.short_term_window

        # Short-term: write to S1 or S2 based on compression state
        lock = self._get_lock(session_id)
        with lock:
            # If S1 is not full and not compressing, write to S1
            s1 = self._buffers_s1.setdefault(session_id, [])
            if session_id not in self._compressing and len(s1) < window:
                s1.append(msg_obj)
            else:
                # S1 full or compressing — write to S2
                self._buffers_s2.setdefault(session_id, []).append(msg_obj)

            # Check if S1 just became full — trigger compression
            if session_id not in self._compressing and len(s1) >= window:
                self._start_compression(session_id)

        # Long-term (MySQL)
        should_close = db is None
        if db is None:
            db = SessionLocal()
        try:
            max_seq = (
                db.query(ChatMessage.sequence)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.sequence.desc())
                .first()
            )
            next_seq = (max_seq[0] + 1) if max_seq else 0

            chat_msg = ChatMessage(
                id=str(uuid4()),
                session_id=session_id,
                role=role,
                content=content,
                sequence=next_seq,
            )
            db.add(chat_msg)
            db.flush()

            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                session.message_count = next_seq + 1
                db.flush()

            db.commit()
        except Exception:
            with lock:
                if session_id in self._buffers_s1:
                    self._buffers_s1[session_id].pop()
                if session_id in self._buffers_s2:
                    self._buffers_s2[session_id].pop()
            db.rollback()
            raise
        finally:
            if should_close:
                db.close()
```

- [ ] **Step 2: 新增 `_start_compression` 方法**

在 `_prune_cache` 方法之前（第 181 行前）插入：

```python
    def _start_compression(self, session_id: str):
        """Mark session as compressing and schedule background summary generation.

        Once compressing, S1 is frozen and new messages go to S2.
        """
        with self._locks_lock:
            if session_id in self._compressing:
                return
            self._compressing.add(session_id)

        def _run_compression():
            bg_db = SessionLocal()
            try:
                bg_session = bg_db.query(ChatSession).filter(ChatSession.id == session_id).first()
                if not bg_session:
                    return
                self._generate_summary(bg_db, session_id, bg_session)
                self._on_compression_complete(session_id)
                bg_db.commit()
            except Exception as e:
                logger.error("compression failed for %s: %s", session_id, e)
                # On failure, just mark as not compressing so next message can retry
                with self._locks_lock:
                    self._compressing.discard(session_id)
            finally:
                bg_db.close()
                with self._locks_lock:
                    self._summary_in_progress.discard(session_id)

        with self._locks_lock:
            self._summary_in_progress.add(session_id)
        threading.Thread(target=_run_compression, daemon=True).start()
```

- [ ] **Step 3: 新增 `_on_compression_complete` 方法**

在 `_start_compression` 之后插入：

```python
    def _on_compression_complete(self, session_id: str):
        """Called after summary generation completes. Migrates S2 to S1."""
        lock = self._get_lock(session_id)
        with lock:
            # S2 content becomes the new S1
            self._buffers_s1[session_id] = list(self._buffers_s2.get(session_id, []))
            self._buffers_s2[session_id] = []
            # Mark compression complete
            with self._locks_lock:
                self._compressing.discard(session_id)
```

- [ ] **Step 4: 删除旧的 `_prune_cache` 和 `_maybe_update_summary` 方法**

删除以下内容（因为压缩机制取代了它们）：
- `_prune_cache` 方法（旧版本，S1/S2 不需要单独的修剪）
- `_maybe_update_summary` 方法（压缩触发替代了阈值触发）

- [ ] **Step 5: 运行测试确认通过**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 已有测试 PASS（行为兼容），新压缩逻辑通过测试

- [ ] **Step 6: Commit**

```bash
git add services/memory_service.py
git commit -m "feat: implement S1/S2 dual buffer compression in add_message"
```

---

## Task 3: 重写 `get_context` — 返回 摘要 + S1 + S2

**Files:**
- Modify: `services/memory_service.py:51-68`（`get_context` 和 `_build_context`）

**目标：** `get_context` 返回 `摘要 + S1 + S2`，压缩期间也包含完整上下文。

- [ ] **Step 1: 重写 `get_context` 方法**

替换 `services/memory_service.py` 第 51-58 行：

```python
    def get_context(self, session_id: str) -> list[BaseMessage]:
        """Get messages for LLM context.

        Returns [summary SystemMessage] + S1 messages + S2 messages.
        S2 contains messages written during compression, ensuring no data loss.
        Recovers from MySQL if both S1 and S2 are empty (e.g. after restart).
        """
        lock = self._get_lock(session_id)
        with lock:
            s1 = self._buffers_s1.get(session_id, [])
            s2 = self._buffers_s2.get(session_id, [])
            if s1 or s2:
                return self._build_context(session_id, s1, s2)
        return self._recover_context(session_id)
```

- [ ] **Step 2: 重写 `_build_context` 方法**

替换第 60-68 行：

```python
    def _build_context(self, session_id: str, s1: list[BaseMessage], s2: list[BaseMessage]) -> list[BaseMessage]:
        """Build context from summary + S1 + S2."""
        summary = self._summary_cache.get(session_id)
        context: list[BaseMessage] = []
        if summary:
            context.append(SystemMessage(content=f"以下是之前对话的摘要：{summary}"))
        context.extend(s1)
        context.extend(s2)
        return context
```

- [ ] **Step 3: 运行测试确认通过**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 测试需要更新（见 Task 4），先确认编译通过

- [ ] **Step 4: Commit**

```bash
git add services/memory_service.py
git commit -m "feat: rewrite get_context to return summary + S1 + S2"
```

---

## Task 4: 重写 `_recover_context` — 重启恢复双缓冲区

**Files:**
- Modify: `services/memory_service.py:70-122`（`_recover_context` 方法）
- Modify: `tests/test_memory_service.py`（更新相关测试）

**目标：** 重启后从 DB 恢复消息，将最近 window 条放入 S1，S2 保持空。

- [ ] **Step 1: 重写 `_recover_context` 方法**

替换 `services/memory_service.py` 第 70-122 行：

```python
    def _recover_context(self, session_id: str) -> list[BaseMessage]:
        """Recover context from MySQL when both S1 and S2 are empty.

        Loads the most recent window messages into S1, S2 stays empty.
        If recovered messages exceed window and no summary exists, triggers summary.
        """
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            summary = None
            summary_sequence = 0
            if session and session.summary:
                summary = session.summary
                summary_sequence = session.summary_sequence

            # Load messages after summary_sequence
            msgs = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .filter(ChatMessage.sequence > summary_sequence)
                .order_by(ChatMessage.sequence.asc())
                .all()
            )

            converted = []
            for m in msgs:
                if m.role == "user":
                    converted.append(HumanMessage(content=m.content))
                else:
                    converted.append(AIMessage(content=m.content))

            window = self.config.short_term_window
            recent = converted[-window:] if len(converted) > window else converted

            # Write to S1 under lock, S2 stays empty
            lock = self._get_lock(session_id)
            with lock:
                self._buffers_s1[session_id] = recent
                self._buffers_s2[session_id] = []
                if summary:
                    self._summary_cache[session_id] = summary

                # If no summary and recovered messages exceed window, generate summary
                if not summary and len(converted) > window:
                    db_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                    if db_session:
                        self._generate_summary(db, session_id, db_session)
                        summary = db_session.summary
                        self._summary_cache[session_id] = summary

            # Build context
            context: list[BaseMessage] = []
            if summary:
                context.append(SystemMessage(content=f"以下是之前对话的摘要：{summary}"))
            context.extend(recent)
            return context
        finally:
            db.close()
```

- [ ] **Step 2: 运行测试确认通过**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 已有测试 PASS（通过 `_buffers_s1` 兼容）

- [ ] **Step 3: Commit**

```bash
git add services/memory_service.py tests/test_memory_service.py
git commit -m "feat: rewrite _recover_context for S1/S2 dual buffer"
```

---

## Task 5: 重写 `_generate_summary` — 适配 S1 内容

**Files:**
- Modify: `services/memory_service.py:221-279`（`_generate_summary` 方法）

**目标：** 增量摘要改为使用 S1 内容（而非从 DB 查询新消息），减少 DB 开销。

- [ ] **Step 1: 重写 `_generate_summary` 方法**

替换 `services/memory_service.py` 第 221-279 行：

```python
    def _generate_summary(self, db: Session, session_id: str, session: ChatSession):
        """Generate conversation summary using LLM and persist it.

        For incremental summaries, uses S1 buffer content instead of DB queries
        (since S1 is frozen during compression, its content is stable).
        """
        try:
            llm = LLMFactory.create()
            old_summary = session.summary
            window = self.config.short_term_window

            lock = self._get_lock(session_id)
            with lock:
                s1_content = list(self._buffers_s1.get(session_id, []))

            if old_summary:
                # Incremental: use S1 content for the new summary
                s1_text = "\n".join(f"[{m.role if isinstance(m, HumanMessage) else 'assistant'}] {m.content}" for m in s1_content)
                if s1_text:
                    prompt = (
                        f"以下是之前对话的摘要：\n{old_summary}\n\n"
                        f"以下是新的对话内容：\n{s1_text}\n\n"
                        f"请生成一段新的摘要，整合之前的摘要和新内容。"
                        f"控制在 {self.config.summary_max_length} 字以内。"
                    )
                else:
                    # S1 was somehow empty, fallback to DB query
                    new_msgs = (
                        db.query(ChatMessage)
                        .filter(ChatMessage.session_id == session_id)
                        .filter(ChatMessage.sequence > session.summary_sequence)
                        .order_by(ChatMessage.sequence.asc())
                        .all()
                    )
                    new_text = "\n".join(f"[{m.role}] {m.content}" for m in new_msgs)
                    prompt = (
                        f"以下是之前对话的摘要：\n{old_summary}\n\n"
                        f"以下是新的对话内容：\n{new_text}\n\n"
                        f"请生成一段新的摘要，整合之前的摘要和新内容。"
                        f"控制在 {self.config.summary_max_length} 字以内。"
                    )
                    latest_seq = new_msgs[-1].sequence if new_msgs else session.summary_sequence
                # Use the last message's sequence from DB as summary_sequence
                last_msg = (
                    db.query(ChatMessage.sequence)
                    .filter(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.sequence.desc())
                    .first()
                )
                latest_seq = last_msg[0] if last_msg else session.summary_sequence
            else:
                # First summary: load the last window of messages from DB
                msgs = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.sequence.desc())
                    .limit(window)
                    .all()
                )
                msgs = list(reversed(msgs))
                all_text = "\n".join(f"[{m.role}] {m.content}" for m in msgs)
                prompt = (
                    f"请总结以下对话的主要内容：\n{all_text}\n\n"
                    f"控制在 {self.config.summary_max_length} 字以内。"
                )
                latest_seq = msgs[-1].sequence if msgs else 0

            summary_resp = llm.invoke(prompt)
            summary = summary_resp.content if hasattr(summary_resp, 'content') else str(summary_resp)
            session.summary = summary
            session.summary_sequence = latest_seq

            with lock:
                self._summary_cache[session_id] = summary

            logger.info(
                "summary generated session=%s sequence=%d",
                session_id, session.summary_sequence,
            )
        except Exception as e:
            logger.error("failed to generate summary for %s: %s", session_id, e)
```

- [ ] **Step 2: 运行测试确认通过**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 所有测试 PASS

- [ ] **Step 3: Commit**

```bash
git add services/memory_service.py
git commit -m "feat: adapt _generate_summary for S1/S2 buffer model"
```

---

## Task 6: 更新 `clear_session` — 清理 S1 和 S2

**Files:**
- Modify: `services/memory_service.py:303-315`（`clear_session` 方法）

**目标：** 清理时同时清空 S1、S2、摘要缓存、压缩标记。

- [ ] **Step 1: 更新 `clear_session` 方法**

替换 `services/memory_service.py` 第 303-315 行：

```python
    def clear_session(self, session_id: str):
        """Clear all memory for a session."""
        lock = self._get_lock(session_id)
        with lock:
            self._buffers_s1.pop(session_id, None)
            self._buffers_s2.pop(session_id, None)
            self._summary_cache.pop(session_id, None)
            with self._locks_lock:
                self._compressing.discard(session_id)
        db = SessionLocal()
        try:
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            db.query(ChatSession).filter(ChatSession.id == session_id).delete()
            db.commit()
        finally:
            db.close()
```

- [ ] **Step 2: 运行测试确认通过**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 所有测试 PASS

- [ ] **Step 3: Commit**

```bash
git add services/memory_service.py
git commit -m "fix: clear S1, S2 and compression state in clear_session"
```

---

## Task 7: 新增双缓冲区单元测试

**Files:**
- Modify: `tests/test_memory_service.py`

- [ ] **Step 1: 添加 S1/S2 行为测试**

在 `tests/test_memory_service.py` 末尾添加以下测试类：

```python
class TestDualBufferCompression(unittest.TestCase):
    """Tests for S1/S2 dual buffer and compression behavior."""

    def setUp(self):
        self.service = MemoryService()

    def _mock_db_session(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_order = MagicMock()
        mock_filter.order_by.return_value = mock_order
        mock_order.first.return_value = None
        mock_session_obj = MagicMock()
        mock_session_obj.summary = None
        mock_session_obj.summary_sequence = 0
        mock_session_obj.message_count = 0
        mock_filter2 = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.side_effect = lambda *args: MagicMock(
            order_by=MagicMock(first=MagicMock(return_value=None)),
            delete=MagicMock()
        )
        # Handle ChatSession query
        return mock_session

    @patch("services.memory_service.get_settings")
    def test_messages_go_to_s1_when_not_full(self, mock_get_settings):
        """Messages are written to S1 when it is not full."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3
        mock_get_settings.return_value.get_memory_config.return_value.summary_update_interval = 3

        session_id = "sess-s1"
        mock_db = self._mock_db_session()
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        # ChatSession mock
        mock_session_obj = MagicMock()
        mock_session_obj.summary = None
        mock_session_obj.summary_sequence = 0
        mock_session_obj.message_count = 0
        # Make the second query (ChatSession) return the mock session
        def query_side_effect(model):
            q = MagicMock()
            q.filter.return_value.first.return_value = mock_session_obj
            q.filter.return_value.order_by.return_value.first.return_value = None
            q.filter.return_value.order_by.return_value.all.return_value = []
            return q
        mock_db.query.side_effect = query_side_effect

        self.service.add_message(session_id, "user", "msg1", db=mock_db)
        self.service.add_message(session_id, "assistant", "reply1", db=mock_db)

        self.assertEqual(len(self.service._buffers_s1[session_id]), 2)
        self.assertEqual(len(self.service._buffers_s2.get(session_id, [])), 0)

    @patch("services.memory_service.get_settings")
    def test_get_context_returns_s1_plus_s2(self, mock_get_settings):
        """get_context returns both S1 and S2 messages."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-both"
        self.service._buffers_s1[session_id] = [HumanMessage(content="s1-msg")]
        self.service._buffers_s2[session_id] = [HumanMessage(content="s2-msg")]

        context = self.service.get_context(session_id)
        self.assertEqual(len(context), 2)
        self.assertEqual(context[0].content, "s1-msg")
        self.assertEqual(context[1].content, "s2-msg")

    @patch("services.memory_service.get_settings")
    def test_get_context_with_summary_and_buffers(self, mock_get_settings):
        """get_context returns summary + S1 + S2."""
        mock_get_settings.return_value.get_memory_config.return_value.short_term_window = 3

        session_id = "sess-summary"
        self.service._summary_cache[session_id] = "summary text"
        self.service._buffers_s1[session_id] = [HumanMessage(content="s1")]
        self.service._buffers_s2[session_id] = [AIMessage(content="s2")]

        context = self.service.get_context(session_id)
        self.assertEqual(len(context), 3)
        self.assertIsInstance(context[0], SystemMessage)
        self.assertIn("summary text", context[0].content)
        self.assertEqual(context[1].content, "s1")
        self.assertEqual(context[2].content, "s2")
```

- [ ] **Step 2: 运行新测试**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py::TestDualBufferCompression -v
```
Expected: 所有新测试 PASS

- [ ] **Step 3: 运行全部测试**

```bash
cd /Users/zyb/PycharmProjects/CCode && python -m pytest tests/test_memory_service.py -v
```
Expected: 全部测试 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_memory_service.py
git commit -m "test: add dual buffer compression unit tests"
```

---

## 自审查

### 1. 规范覆盖检查

| 规范要求 | 对应 Task |
|---------|----------|
| S1 满触发后台压缩 | Task 2 (`_start_compression`) |
| 压缩期间 S1 冻结，新消息写 S2 | Task 2 (`add_message` 分支逻辑) |
| 压缩完成迁移 S2→S1 | Task 2 (`_on_compression_complete`) |
| `get_context` 返回 摘要+S1+S2 | Task 3 |
| 重启恢复双缓冲区 | Task 4 |
| 摘要使用 S1 内容而非全量 DB 查询 | Task 5 |
| 清理时同时清空 S1/S2/压缩标记 | Task 6 |
| 线程安全（per-session lock） | 全程保持 |

### 2. 占位符扫描

无占位符。每个步骤都包含完整代码。

### 3. 类型一致性

- `_buffers_s1`/`_buffers_s2`: `dict[str, list[BaseMessage]]` — 所有 task 一致
- `_compressing`: `set[str]` — Task 1 定义，Task 2/6 使用
- `_summary_cache`: `dict[str, str]` — 不变
- `get_context` 返回 `list[BaseMessage]` — 不变
- `SystemMessage`/`HumanMessage`/`AIMessage` — 与原代码一致
