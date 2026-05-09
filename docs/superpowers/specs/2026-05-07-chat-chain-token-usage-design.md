# chat_chain Token 使用量展示设计

- 日期：2026-05-07
- 主题：为 `chat_chain` 增加 token 使用量展示，包含输入 token、输出 token、分析 token，并同步到接口、日志和前端

## 1. 背景

当前聊天接口与对话链路中，普通对话 `ChatChain` 只返回文本响应，流式接口只返回 chunk 事件，未对模型返回的 token 使用量做统一采集与展示。

用户本次明确要求：

1. 为 `chat_chain` 增加 token 使用量展示
2. 展示范围包括：后端接口、服务端日志、前端页面
3. 展示字段包括：
   - `input_tokens`
   - `output_tokens`
   - `analysis_tokens`
4. `analysis_tokens` 采用兼容提取策略：优先从不同 provider 的原生或扩展 usage 字段中读取；若不存在，则兜底为 `0`

当前代码现状：

- `chains/chat_chain.py` 中 `ChatChain.invoke()` 只返回 `str`
- `routers/chat.py` 中 `ChatResponse` 仅包含 `session_id` 和 `response`
- `routers/chat.py` 中流式接口只输出 `chunk` 和 `[DONE]`
- 代码中尚未存在统一的 usage 提取逻辑
- 项目支持多 provider（OpenAI、Claude、兼容 OpenAI 的 provider），因此不能只依赖单一字段结构

本次目标是在尽量少改动现有结构的前提下，为普通对话链路补齐 usage 数据采集、统一结构输出和前端展示能力。

## 2. 目标

1. 为普通聊天链路增加统一 usage 结构输出
2. 在同步聊天接口中返回 usage 字段
3. 在流式聊天接口中于结束前补发 usage 事件
4. 在服务端日志中记录 usage 信息
5. 在前端 assistant 消息下展示 usage 信息
6. 对不同 provider 的 usage 元数据做轻量兼容提取
7. 在 provider 不支持 analysis token 的情况下，稳定返回 `0`

## 3. 非目标

以下内容不在本次范围内：

- 不修改数据库结构
- 不将 usage 持久化到 MySQL
- 不改 memory 逻辑、摘要逻辑或 RAG 检索策略
- 不对 provider 工厂做大规模重构
- 不增加复杂的 provider SDK 适配抽象层
- 不调整现有流式 chunk 输出格式
- 不新增统计报表、监控聚合或成本分析页面

## 4. 现状分析

### 4.1 ChatChain 当前返回值边界

`chains/chat_chain.py` 中：

- `invoke()` 当前直接构建 `prompt | llm | parser` 链并返回文本
- `astream()` 当前按 chunk 流式输出文本，并在结束后拼接完整结果保存到 memory

这意味着：

1. 当前调用链以纯文本为中心
2. usage 信息如果存在于模型原始响应中，经过 `StrOutputParser()` 后很可能无法直接透传到路由层
3. 若只在 `routers/chat.py` 做包装，无法稳定获取 usage，尤其不利于流式场景统一处理

因此最合适的改动位置是 chain 层，而不是仅改路由层。

### 4.2 路由层当前输出边界

`routers/chat.py` 中：

- `ChatResponse` 当前只定义 `session_id` 与 `response`
- `/api/chat/stream` 当前事件只有 `{chunk: ...}` 和 `[DONE]`

因此：

- 同步接口需要扩展 response model
- 流式接口需要增加 usage 事件，但应保持已有 chunk 消费方式不变，以减少前端和调用方回归风险

### 4.3 多 provider 兼容性要求

项目通过 `services/llm_factory.py` 支持多 provider。不同 provider 或 LangChain 适配层返回的 usage 结构可能不同，例如：

- `usage_metadata`
- `response_metadata`
- `token_usage`
- `output_tokens_details.reasoning_tokens`
- `reasoning_tokens`

因此本次不能把 usage 提取逻辑写死为某一个字段路径，而应采用轻量兼容提取方式，并统一转换为项目内部稳定结构。

## 5. 设计方案

### 5.1 统一 usage 数据结构

本次统一对外输出以下结构：

```json
{
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "analysis_tokens": 0
  }
}
```

约束如下：

1. 三个字段都固定存在
2. 字段类型为整数
3. provider 未返回对应字段时使用 `0`
4. 所有接口、日志和前端都基于该统一结构消费

这样可以避免前端和调用方对 provider 差异做分支判断。

### 5.2 chain 层统一产出“文本 + usage”

改动位置：

- `chains/chat_chain.py`
- `chains/rag_chain.py`

设计原则：

1. 不再让 chain 只处理纯字符串概念
2. chain 内部在拿到模型结果后，提取文本与 usage
3. 再由路由层决定如何组装 HTTP 响应或 SSE 事件

目标结果：

- 同步链路返回：`response + usage`
- 流式链路在完成后能拿到完整 usage

这样同步与流式可以复用同一套 usage 定义，避免一套接口一个口径。

### 5.3 轻量 usage 提取逻辑

本次会增加一个轻量的 usage 提取函数，职责仅限：

1. 从模型返回对象中兼容提取 token 统计信息
2. 统一映射到项目标准结构
3. 提供默认值 `0`

兼容读取方向包括但不限于：

- `usage_metadata.input_tokens`
- `usage_metadata.output_tokens`
- `response_metadata.token_usage.prompt_tokens`
- `response_metadata.token_usage.completion_tokens`
- `response_metadata.reasoning_tokens`
- `response_metadata.output_tokens_details.reasoning_tokens`
- 其他等价字段

映射规则：

- `input_tokens`：优先取 prompt/input 类型字段
- `output_tokens`：优先取 completion/output 类型字段
- `analysis_tokens`：优先取 reasoning 类型字段
- 缺失时兜底 `0`

本次不做过度抽象，不做完整 provider 注册表，只做项目内够用的兼容解析。

### 5.4 同步接口输出设计

改动位置：`routers/chat.py`

当前：

```json
{
  "session_id": "xxx",
  "response": "xxx"
}
```

调整后：

```json
{
  "session_id": "xxx",
  "response": "xxx",
  "usage": {
    "input_tokens": 123,
    "output_tokens": 45,
    "analysis_tokens": 6
  }
}
```

设计要求：

1. 保留原有 `session_id` 与 `response` 字段
2. 仅新增 `usage` 字段，不改原字段语义
3. `response_model` 明确声明 usage 结构，保证接口契约清晰

### 5.5 流式接口事件设计

改动位置：`routers/chat.py`

当前流式接口输出：

```text
data: {"chunk":"..."}
...
data: [DONE]
```

调整后：

```text
data: {"chunk":"..."}
...
data: {"usage":{"input_tokens":123,"output_tokens":45,"analysis_tokens":6}}

data: [DONE]
```

设计原则：

1. 保持既有 chunk 事件结构不变
2. usage 作为单独事件在结束前发送
3. `[DONE]` 保持最后一个终止标记

这样可以最大程度兼容现有前端的流式文本拼接逻辑，只需额外识别 usage 事件。

### 5.6 日志设计

改动位置：`chains/chat_chain.py` 与 `chains/rag_chain.py`

在对话完成后增加 usage 日志，日志内容包含：

- `session_id`
- `provider`
- `input_tokens`
- `output_tokens`
- `analysis_tokens`

设计原则：

1. 不替换现有日志，只补充 usage 相关信息
2. 保持日志风格与现有 `logger.info(...)` 调用一致
3. 同步与流式完成时都打印 usage

这样便于排查 provider 差异和问题定位。

### 5.7 前端展示设计

改动位置：前端聊天页面相关渲染代码

本次采用最小展示方案：

1. 仅在 assistant 消息下展示 usage
2. 展示文案固定为：

```text
Token：输入 X / 输出 Y / 分析 Z
```

3. 不新增复杂浮层、面板或折叠交互
4. 不改变消息主内容结构，仅在消息下方追加 usage 信息

这样实现成本低、回归风险小，且能满足“展示 token 使用量”的需求。

### 5.8 RAGChain 一致性

虽然本次需求由 `chat_chain` 提出，但 `routers/chat.py` 同时支持普通对话与 RAG 对话。

为保证接口结构统一，本次应同步处理 `RAGChain`：

1. 与 `ChatChain` 产出一致的 usage 结构
2. `/api/chat` 与 `/api/chat/stream` 无论 `use_rag` 是否开启，都返回相同 usage 结构
3. 避免前端根据 `use_rag` 分叉处理 usage

## 6. 文件变更边界

本次预期修改文件：

- 修改：`chains/chat_chain.py`
- 修改：`chains/rag_chain.py`
- 修改：`routers/chat.py`
- 修改：前端聊天页面文件
- 修改：相关测试文件（至少包括 chat chain / chat router 测试）

本次不应修改：

- `models/`
- `services/memory_service.py`
- 数据库结构
- provider 配置文件
- `utils/db.py`
- 与本需求无关的前端样式或布局模块

## 7. 验收标准

### 7.1 同步接口验收

需要确认：

- `POST /api/chat` 返回 `usage` 字段
- `usage.input_tokens`、`usage.output_tokens`、`usage.analysis_tokens` 都存在
- 字段类型为整数
- provider 无 reasoning 字段时，`analysis_tokens` 返回 `0`

### 7.2 流式接口验收

需要确认：

- 原有 chunk 事件继续正常输出
- 在 `[DONE]` 之前存在一条 usage 事件
- usage 事件结构为统一格式

### 7.3 日志验收

需要确认：

- 同步请求完成后打印 usage 日志
- 流式请求完成后打印 usage 日志
- 日志中包含输入、输出、分析 token

### 7.4 前端验收

需要确认：

- assistant 消息下显示 `Token：输入 X / 输出 Y / 分析 Z`
- 普通对话与 RAG 对话展示一致
- 同步返回与流式返回都能展示 usage

### 7.5 回归验收

需要确认：

- 聊天主文本内容展示不受影响
- memory 写入逻辑不变
- SSE 流式输出结束逻辑不变
- 现有 provider 仍可正常响应

## 8. 实施原则

1. 只做与 usage 展示直接相关的最小改动
2. 统一接口结构，避免前端分支逻辑扩散
3. 兼容多 provider，但不做过度抽象
4. 保持现有代码风格与日志风格一致
5. 流式接口优先保持向后兼容

## 9. 实施结论

本次将按以下方案进入实现计划阶段：

1. 在 chain 层统一提取并返回 usage
2. 增加轻量 usage 提取逻辑，兼容多 provider usage 字段
3. 同步接口返回 `usage`
4. 流式接口在 `[DONE]` 前发送 usage 事件
5. 日志记录输入/输出/分析 token
6. 前端 assistant 消息展示 token 信息
7. 补齐 usage 提取、同步接口、流式接口与前端展示相关测试

## 10. 自检结果

- 无 TBD / TODO / 占位内容
- 目标、非目标、改动边界、接口结构与验收标准已明确
- 范围聚焦于 chat token usage 展示，可直接进入 implementation plan
- 同步、流式、日志、前端四处口径一致，避免后续重复设计
