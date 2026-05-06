# 记忆窗口压缩优化设计文档

## 当前问题

旧实现中，短期记忆是单一列表，超过窗口大小时直接截断最老消息：

```text
[0 ... 9]  → 新消息到达 → [1 ... 10]  → 新消息到达 → [2 ... 11]
```

这种方式有两个明显问题：

1. **摘要期间可能丢消息**：如果后台摘要还没覆盖旧消息，截断会直接丢失它们
2. **压缩与写入互相干扰**：一边压缩一边继续写同一个列表，边界不清晰

## 最终方案：双缓冲区压缩（S1 / S2）

### 核心思想

将短期记忆拆成两个缓冲区：

```text
┌─────────────┬─────────────┐
│    S1       │     S2      │
│ 活跃上下文区 │ 压缩期间新消息区 │
│ 容量=half    │ 动态增长        │
└─────────────┴─────────────┘
```

其中：

- `window = short_term_window`
- `half = window // 2`
- **S1 的目标容量是 half，不是整个 window**
- **S2 只在压缩期间承接新增消息**
- 最终上下文 = `summary + S1 + S2`

## 状态流转

### 状态 A：正常写入

- 新消息写入 `S1`
- 只要 `S1` 未达到 `half`，不会触发压缩

### 状态 B：S1 达到 half

- 当前消息先进入 `S1`
- 然后触发 `_start_compression(session_id)`
- session 进入 `_compressing`
- 后续新消息进入 `S2`

### 状态 C：压缩进行中

- 后台线程执行：
  1. 读取当前 `ChatSession`
  2. 生成摘要
  3. 调用 `_on_compression_complete(session_id)`
  4. `commit`
  5. 清理 `_summary_in_progress`

### 状态 D：压缩完成

`_on_compression_complete()` 做三件事：

```python
self._buffers_s1[session_id] = list(self._buffers_s2.get(session_id, []))
self._buffers_s2[session_id] = []
self._compressing.discard(session_id)
```

也就是：
- `S2 -> S1`
- `S2` 清空
- session 退出压缩状态

## 重启恢复策略（half-window 版本）

服务重启后，恢复逻辑不再是“把最近 window 条全塞进 S1”，而是：

### 恢复输入

从持久化层拿到：

- `summary_sequence`：摘要已覆盖到哪一条消息
- `latest_sequence`：当前最新消息 sequence
- `uncompressed_messages`：所有 `sequence > summary_sequence` 的消息

设：

```text
half = short_term_window // 2
uncompressed_count = len(uncompressed_messages)
```

### 情况 1：未压缩消息数 `<= half`

说明积压不多。

恢复结果：

- `S1 = 全部未压缩消息`
- `S2 = []`
- **不触发压缩**

### 情况 2：未压缩消息数 `> half`

说明摘要之后积压的原始消息已经超过半窗。

恢复结果：

- `S1 = uncompressed_messages[-half:]`
- `S2 = []`
- **触发一次压缩**

注意：

> 这次压缩的输入范围不是 `S1`，而是 `summary_sequence` 之后的**所有未压缩消息**。

也就是说，恢复后触发压缩时：

- 展示给 LLM 的实时上下文是 `summary + S1`
- 但后台压缩覆盖范围是 `all uncompressed messages`

这样不会丢掉“未压缩但未进入 S1”的前半段消息。

## 当前上下文读取规则

### 内存命中

如果内存里已有 `S1` 或 `S2`：

```python
s1 = self._buffers_s1.get(session_id, [])
s2 = self._buffers_s2.get(session_id, [])
if s1 or s2:
    return self._build_context(session_id, s1, s2)
```

### 上下文拼装

```text
summary + S1 + S2
```

其中：
- `summary` 是长程记忆
- `S1` 是稳定窗口
- `S2` 是压缩期间增量消息

## 数据结构

```python
class MemoryService:
    _buffers_s1: dict[str, list[BaseMessage]]
    _buffers_s2: dict[str, list[BaseMessage]]
    _summary_cache: dict[str, str]
    _compressing: set[str]
    _summary_in_progress: set[str]
    _locks: dict[str, threading.Lock]
```

## 线程安全策略

当前实现使用 **per-session lock**：

- `get_context()` 读取 `S1/S2`
- `add_message()` 写入 `S1/S2`
- `_on_compression_complete()` 做 `S2 -> S1`
- `clear_session()` 清理缓存
- `_generate_summary()` 更新 `_summary_cache`

都通过 `self._get_lock(session_id)` 做保护。

此外：
- `_locks_lock` 用于保护 `_locks` 字典本身
- `_summary_in_progress` 用于防止同一 session 重复启动压缩线程

## SQL 分层

按项目规则：

> 所有 SQL 语句都应当在 model 层完成

当前实现要求 `MemoryService` 不直接拼接 query，统一通过 model 方法访问。

### `models/chat_message.py`

提供：
- `get_after_sequence()`
- `get_max_sequence()`
- `get_recent()`
- `get_all()`
- `delete_by_session()`

### `models/chat_session.py`

提供：
- `get_by_id()`
- `delete_by_id()`

## 预期效果

最终目标：

- 服务重启后可恢复上下文
- 长对话通过摘要保留历史信息
- 恢复时严格遵循 half-window 语义
- 压缩期间不会丢失新消息
- 后台压缩覆盖所有未压缩消息，而不是只覆盖当前 S1
- 双缓冲避免摘要和写入互相污染
- `memory_service` 层有完整自动化测试保护
