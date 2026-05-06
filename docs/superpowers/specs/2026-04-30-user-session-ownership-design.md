# 用户与会话归属设计

- 日期：2026-04-30
- 主题：增加用户机制，使一个用户对应多个会话，并支持前端切换/创建用户

## 1. 背景

当前系统只有 `chat_sessions` 与 `chat_messages` 两层关系，没有用户维度。所有会话都在同一个空间下管理，前端也没有“当前用户”的概念。用户希望新增一个最小可用的用户机制，满足：

1. 增加独立用户表
2. 一个用户可对应多个 session
3. 不做登录/注册/鉴权体系
4. 前端可以切换用户
5. 前端可以创建新用户
6. 浏览器缓存最近使用的用户

本次目标是在不引入完整账号体系的前提下，为现有聊天与会话系统补上明确的用户归属关系。

## 2. 目标

1. 新增 `users` 表与 `User` 模型
2. 为 `chat_sessions` 增加 `user_id`
3. 所有 session 读写操作基于 `user_id` 约束
4. 前端支持创建用户、切换用户、缓存最近使用用户
5. 不改变现有聊天消息与 memory 的核心处理方式

## 3. 非目标

以下内容不在本次范围内：

- 不实现登录/注册接口语义
- 不实现 token / cookie / session 鉴权
- 不实现权限体系
- 不做用户级跨 session 摘要合并或长期记忆共享
- 不扩展成完整账号系统

## 4. 现状分析

### 4.1 当前数据模型

当前主要关系为：

- `chat_sessions`
- `chat_messages`

其中：
- `ChatSession` 只有 `id`、`title`、`message_count`、`summary` 等字段
- `ChatMessage` 通过 `session_id` 归属于某个 session
- 没有用户归属字段

### 4.2 当前接口行为

当前 `routers/session.py` 中：

- `POST /api/session` 直接创建 session，不带用户维度
- `GET /api/session` 返回所有 session
- `GET /api/session/{session_id}/history` 直接按 session 查询历史

这意味着：
- 所有会话对所有调用者“可见”
- 无法实现“切换用户后只看自己的会话”

### 4.3 当前前端行为

当前前端：
- 默认直接加载全部 session 列表
- 创建 session 时不携带用户信息
- 浏览器端没有“当前用户”状态

因此，要实现一用户多 session，后端接口和前端状态都需要显式引入 user 维度。

## 5. 设计方案

### 5.1 数据模型设计

新增 `User` 模型，对应 `users` 表，字段最小化为：

- `id`
- `name`
- `created_at`

同时为 `ChatSession` 增加：

- `user_id`

关系为：

- 一个 `User` 拥有多个 `ChatSession`
- 一个 `ChatSession` 仍拥有多条 `ChatMessage`

本次不要求在 ORM 层必须建立复杂 relationship，只需保证查询和写入逻辑明确基于 `user_id`。

### 5.2 用户接口设计

新增用户接口：

#### `POST /api/user`
用途：创建用户

请求体最小字段：
- `name`

返回：
- `id`
- `name`
- `created_at`

#### `GET /api/user`
用途：列出所有用户

返回用户列表，供前端切换使用。

本次不增加删除用户、更新用户接口。

### 5.3 Session 接口调整

#### `POST /api/session`
创建 session 时必须传：
- `title`
- `user_id`

后端创建时将 `user_id` 写入 `chat_sessions`。

#### `GET /api/session`
查询 session 列表时必须传：
- `user_id`

后端按 `user_id` 过滤，只返回当前用户的 session。

#### `GET /api/session/{session_id}/history`
查询历史时必须传：
- `user_id`

后端先检查该 session 是否属于该 `user_id`，只有归属正确才返回历史记录。

这样能确保用户切换后只看到自己的会话与历史。

### 5.4 前端用户切换设计

前端新增用户切换区，具备两个能力：

1. 切换已有用户
2. 创建新用户

推荐行为：

- 页面初始化先加载用户列表
- 从 `localStorage` 读取最近使用的 `user_id`
- 若存在且有效，则设为当前用户
- 若不存在但用户列表非空，则默认选第一个用户
- 若用户列表为空，则提示用户先创建用户

一旦当前用户切换：
- 清空当前会话态
- 重新加载该用户的 session 列表
- 只有选中用户后才能创建 session 或进入聊天

### 5.5 浏览器缓存设计

只缓存一个最近使用的用户标识，建议键名固定为：

- `chat_user_id`

缓存内容仅为：
- `user_id`

本次不缓存：
- 用户对象完整信息
- session 列表
- 历史消息

这样能降低前端缓存不一致风险。

### 5.6 Memory 影响边界

当前 `MemoryService` 仍以 `session_id` 为主键管理缓存、摘要与消息恢复。本次不改变这一层语义。

原因：
- 当前需求是“用户拥有多个 session”
- 不是“多个 session 共用用户级 memory”

因此本次只要求：
- session 归属于 user
- memory 仍按 session 隔离

## 6. 文件变更边界

本次预期修改文件：

- 新增：`models/user.py`
- 修改：`models/chat_session.py`
- 修改：`routers/session.py`
- 新增：用户 router 文件（如 `routers/user.py`）
- 修改：`main.py`（注册用户 router）
- 修改：`frontend/index.html`
- 可能修改：与数据库初始化相关的文件

本次不应修改：

- `services/memory_service.py` 的核心逻辑
- `chains/` 下聊天链核心逻辑
- `models/chat_message.py` 消息结构
- 鉴权相关机制（本次不存在也不新增）

## 7. 验收标准

### 7.1 数据模型验收

需要确认：

- 存在 `users` 表
- `chat_sessions.user_id` 存在
- 新建 session 时会正确写入 `user_id`

### 7.2 接口验收

需要确认：

- 可以创建用户
- 可以列出用户
- 按 `user_id` 查询 session 时只返回该用户的会话
- 查询 session history 时会校验 session 归属当前用户

### 7.3 前端验收

需要确认：

- 前端可以创建用户
- 前端可以切换用户
- 切换用户后 session 列表发生变化
- 浏览器刷新后能恢复最近使用的用户

### 7.4 回归验收

需要确认：

- 原有聊天流程仍可用
- 原有消息持久化仍可用
- memory 与摘要逻辑仍按 session 维度工作
- 不引入与登录/鉴权相关的新依赖或复杂逻辑

## 8. 实施原则

1. 只实现最小可用用户体系
2. 用户与会话关系明确，但不扩展到完整账号系统
3. session 相关接口统一按 `user_id` 约束
4. 前端只缓存最近使用用户，不缓存更多业务状态
5. 保持现有代码风格与结构一致

## 9. 实施结论

本次将按以下方案进入实现计划阶段：

- 增加 `users` 表与 `User` 模型
- 给 `chat_sessions` 增加 `user_id`
- 新增用户创建/列表接口
- 修改 session 创建、列表、历史接口，全部纳入 `user_id`
- 前端支持创建/切换用户，并缓存最近使用用户
- 不引入登录鉴权和用户级 memory 聚合

## 10. 自检结果

- 无 TBD / TODO / 占位内容
- 目标、非目标、数据关系、接口行为、前端行为和边界都已明确
- 范围聚焦于“一用户多 session”的最小完整落地方案
- 各章节描述一致：均以“独立用户表 + session 归属用户 + 前端切换/缓存用户”为中心
