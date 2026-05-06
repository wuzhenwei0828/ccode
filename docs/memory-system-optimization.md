# 记忆系统优化设计文档

## 问题描述

当前短期记忆系统存在三个核心问题：

| 问题 | 描述 | 影响 |
|------|------|------|
| 1. 上下文丢失 | 服务重启后 `_short_term` 内存缓存清空 | 所有会话的历史上下文丢失 |
| 2. 内存浪费 | `add_message()` 无限追加到 `_short_term`，从不修剪 | 聊 1000 轮缓存就有 1000 条（虽然只取最后 10 条给 LLM） |
| 3. 历史被截断 | 超过窗口大小时旧消息直接丢弃，LLM 完全遗忘 | 长对话丢失早期关键信息 |

## 根因分析

`services/memory_service.py` 中存在两个存储层：

- **短期缓存** `_short_term`：纯内存字典，进程生命周期内有效
- **长期存储** MySQL：消息已持久化到 `chat_messages` 表

但 `get_context()` 方法只读内存缓存：

```python
def get_context(self, session_id: str) -> list[BaseMessage]:
    messages = self._short_term.get(session_id, [])  # 仅读内存
    window = self.config.short_term_window
    return messages[-window:] if len(messages) > window else messages
```

`load_history()` 方法已实现但从未被调用，无法回填缓存。

## 解决方案

### 方案一：MySQL 回源 + 缓存修剪（解决 问题 1、2）

修改 `get_context()` 方法，增加 MySQL 回源逻辑：

```
get_context(session_id):
  1. 如果 _short_term[session_id] 非空 → 返回窗口内的最近消息
  2. 如果为空（重启/新实例/首次访问）
     → 从 MySQL 加载最近 short_term_window 条消息
     → 写入 _short_term 缓存
     → 返回
```

修改 `add_message()`，写入后修剪缓存：

```
add_message(session_id, role, content):
  1. 写入缓存 + MySQL（保持双写）
  2. 如果缓存长度 > short_term_window → 保留最后 short_term_window 条，丢弃旧消息
```

### 方案二：对话摘要持久化（解决 问题 3）

当对话超过窗口大小时，自动生成历史摘要并持久化到 `chat_sessions.summary` 字段。

`get_context()` 最终返回结构：

```
[system: 对话摘要: xxx...]    ← 从 DB 加载的摘要（如存在）
[最近 N 轮具体对话...]         ← 滑动窗口内的原始消息
[当前用户问题...]             ← 新消息
```

**摘要生成时机：**

- 当消息数量首次超过 `short_term_window` 时触发首次摘要
- 后续每累积 `short_term_window` 条新消息触发一次增量摘要
- 即：每满一个窗口大小的新消息就更新一次摘要
- 摘要内容 = 旧摘要 + 新消息的总结

**摘要持久化：**

- 存储在 `chat_sessions` 表新增的 `summary` 和 `summary_sequence` 字段（Text 类型 + Integer）
- `summary_sequence` 记录摘要覆盖到的最后一条消息的 sequence 值
- 内存中也缓存 `_summary_cache: dict[str, str]`，避免频繁读 DB
- 服务重启后摘要从 DB 恢复，不会丢失

**重启后加载短期记忆的完整流程：**

```
get_context(session_id):
  1. 如果 _short_term[session_id] 非空 → 返回窗口内的最近消息
  2. 如果为空（重启/首次访问）：
     a. 从 chat_sessions 读取 summary 和 summary_sequence
     b. 从 chat_messages 加载 sequence > summary_sequence 的所有消息
     c. 如果数量 > window_size → 只取最后 window_size 条
     d. 写入 _short_term 缓存
     e. 返回 [摘要] + [恢复后的短期消息]
```

### 变更范围

| 文件 | 变更 |
|------|------|
| `models/chat_session.py` | 新增 `summary` 和 `summary_sequence` 列 |
| `services/memory_service.py` | 修改 `get_context()` 增加 DB 回源和摘要加载；修改 `add_message()` 增加缓存修剪和摘要更新；新增 `_generate_summary()` 方法 |
| `config/settings.py` | MemoryConfig 新增 `summary_max_length`（摘要最大字数，默认 2000）、`summary_trigger_size`（触发摘要的阈值，默认同窗口大小）、`summary_update_interval`（增量摘要间隔，默认同窗口大小） |
| `chains/chat_chain.py` | 无需变更 |
| `chains/rag_chain.py` | 无需变更 |

### 预期效果

- 服务重启后自动恢复：摘要（长期记忆）+ 最近 10 条（短期上下文）
- 内存中始终只保留窗口大小的原始消息 + 一段摘要文本，不会无限增长
- 长对话不会丢失早期信息，LLM 通过摘要了解历史脉络
- 完整原始消息仍保存在 MySQL 中，可随时查询

### 成本考量

| 项目 | 说明 |
|------|------|
| 首次摘要 | 约 1 次 LLM 调用（短会话，token 少） |
| 增量摘要 | 每 short_term_window 轮 1 次 LLM 调用 |
| 存储 | `summary` 字段约 2000 字，`summary_sequence` 一个整数，几乎不占空间 |
| 读开销 | 每次请求多一次 `chat_sessions` 查询（可被内存缓存优化） |
