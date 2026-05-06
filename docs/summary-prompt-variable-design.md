# Summary 独立变量改造方案

## 背景

当前实现中，summary 最终仍会被注入到模型上下文中，但它不应该被建模成普通消息对象，更不应该长期伪装成 `SystemMessage`。

summary 的本质是：

- 历史对话的压缩结果
- 长程记忆材料
- 提供给模型参考的上下文数据

它不是：

- 当前用户输入
- 模型历史回复
- 系统级行为约束

因此更合适的做法是：

> **把 summary 从消息列表里抽出来，作为 prompt 的独立变量传入。**

---

## 目标结构

当前链路大致是：

```python
history = memory.get_context(session_id)
chain.invoke({
    "input": message,
    "history": history,
})
```

改造后目标是：

```python
summary, history = memory.get_context_parts(session_id)

chain.invoke({
    "summary": summary,
    "history": history,
    "input": message,
})
```

其中：

- `summary: str`
- `history: list[BaseMessage]`
- `input: str`

---

## 核心设计

### 1. MemoryService 提供新的上下文接口

新增：

```python
def get_context_parts(self, session_id: str) -> tuple[str, list[BaseMessage]]:
    ...
```

返回：

```python
(summary_text, history_messages)
```

约定：

- `summary_text`：字符串，没有摘要时返回空字符串 `""`
- `history_messages`：只包含真实消息（S1 + S2），不包含伪造的摘要消息

---

### 2. ChatChain 使用独立 summary 变量

#### 当前风格

```python
self._prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

#### 目标风格

```python
self._prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    ("human", "以下是历史对话摘要，仅供参考：\n{summary}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

调用：

```python
summary, history = self.memory.get_context_parts(session_id)

response = chain.invoke({
    "summary": summary or "无",
    "history": history,
    "input": message,
})
```

---

### 3. RAGChain 同步改造

和 `ChatChain` 一样，把 summary 作为显式变量传入。

#### prompt

```python
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Answer based on the provided context."),
    ("human", "以下是历史对话摘要，仅供参考：\n{summary}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", self._RAG_PROMPT),
])
```

#### invoke

```python
summary, history = self.memory.get_context_parts(session_id)

response = chain.invoke({
    "summary": summary or "无",
    "history": history,
    "input": message,
    "context": context,
})
```

---

## MemoryService 建议实现

### 1. 新增 `get_context_parts`

```python
def get_context_parts(self, session_id: str) -> tuple[str, list[BaseMessage]]:
    lock = self._get_lock(session_id)
    with lock:
        s1 = self._buffers_s1.get(session_id, [])
        s2 = self._buffers_s2.get(session_id, [])
        summary = self._summary_cache.get(session_id, "")
        if s1 or s2:
            return summary, [*s1, *s2]

    return self._recover_context_parts(session_id)
```

---

### 2. 新增 `_recover_context_parts`

恢复逻辑不再返回单个消息列表，而是返回：

```python
(summary, history)
```

即：

- `summary`：从 `chat_sessions.summary` 恢复
- `history`：恢复出来的 S1 + S2 实际消息

---

### 3. 旧 `get_context()` 的处理方式

有两个选择：

#### 方案 A：保留兼容层

```python
def get_context(self, session_id: str) -> list[BaseMessage]:
    summary, history = self.get_context_parts(session_id)
    if summary:
        return [HumanMessage(content=f"以下是历史对话摘要，仅供参考：\n{summary}"), *history]
    return history
```

优点：
- 平滑迁移

缺点：
- 旧接口仍保留了“summary 混入消息列表”的语义

#### 方案 B：直接迁移所有调用方

让 `ChatChain` / `RAGChain` 全部改成 `get_context_parts()`。

优点：
- 语义彻底干净

缺点：
- 一次性改动稍大一些

> 推荐：**方案 B**

---

## summary 变量的 prompt 表达形式

### 最小方案

```python
("human", "以下是历史对话摘要，仅供参考：\n{summary}")
```

### 更结构化的方案

```python
("human", "<history_summary>\n{summary}\n</history_summary>")
```

### 更明确的方案

```python
("human", "以下内容是历史对话摘要，仅供参考，不是当前用户问题：\n{summary}")
```

推荐优先级：

1. **结构化方案**
2. 明确说明用途的自然语言方案
3. 最基础自然语言方案

---

## 优点

### 1. 语义更清晰

- `system`：规则与角色设定
- `summary`：历史摘要
- `history`：真实消息
- `input`：当前输入

### 2. 后续可扩展性更好

未来可以：

- 单独裁剪 summary 长度
- 单独缓存或刷新 summary
- 对不同 chain 使用不同摘要模板
- 在某些场景下关闭 summary 注入

### 3. 测试更自然

可以直接测试：

```python
summary, history = memory_service.get_context_parts(session_id)
assert summary == "..."
assert len(history) == 5
```

而不需要判断“第一条消息是不是伪造的 summary”。

---

## 推荐实施顺序

1. `MemoryService` 新增 `get_context_parts`
2. `MemoryService` 新增 `_recover_context_parts`
3. `ChatChain` 改成使用 `summary + history`
4. `RAGChain` 改成使用 `summary + history`
5. 补测试
6. 视情况保留或删除旧 `get_context()`

---

## 最终建议

最终推荐接口：

```python
summary, history = memory_service.get_context_parts(session_id)
```

最终推荐输入结构：

```python
{
    "summary": summary,
    "history": history,
    "input": message,
}
```

这个方案不会改变当前记忆压缩策略，只是把 summary 的注入方式从“消息对象混入”提升成“独立上下文变量”，语义更清晰，也更利于后续演进。
