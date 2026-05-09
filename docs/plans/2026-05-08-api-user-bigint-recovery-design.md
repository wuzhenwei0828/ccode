# /api/user bigint schema recovery design

- 日期：2026-05-08
- 主题：修复 `/api/user` 因数据库 schema 漂移导致的 500，并将用户、会话、消息表从字符串/UUID 体系迁回 bigint 体系

## 1. 背景

手工验证前端 token usage 展示时，发现 `/api/user` 返回 500。排查结果表明，这不是本次 token usage 改动引入的问题，而是数据库真实 schema 与当前代码约定整体漂移：

- `users.id` 实际为 `character varying`
- `chat_sessions.id` 实际为 `character varying`
- `chat_messages.id` 实际为 `character varying`
- `chat_messages.session_id` 实际为 `character varying`
- `chat_sessions` 实际缺少 `user_id` 列

而当前代码明确按 bigint/int 体系工作：

- `models/user.py`：`User.id` 为 `BigInteger`
- `models/chat_session.py`：`ChatSession.id`、`ChatSession.user_id` 为 `BigInteger`
- `models/chat_message.py`：`ChatMessage.id`、`ChatMessage.session_id` 为 `BigInteger`
- `routers/user.py`、`routers/session.py` 也都按 `int` 对外暴露接口

因此，这不是单个 `/api/user` 的局部问题，而是底层库结构与应用契约已经不一致。

## 2. 当前数据现状

通过实际查询数据库确认：

- `users` 当前只有 1 条记录：
  - `id = '67e3e351-602a-4aad-8fa1-a5748af49e87'`
  - `name = 'wzw'`
- `chat_sessions` 当前有 2 条记录：
  - `id` 都是 UUID 字符串
  - 无 `user_id` 列
- `chat_messages` 当前有 58 条记录：
  - `session_id` 指向 UUID 形式的 `chat_sessions.id`

可恢复关系：
- `chat_messages.session_id -> chat_sessions.id` 仍然可用

不可恢复关系：
- `chat_sessions -> users` 的归属关系已经不存在，因为真实表中没有 `user_id`
- schema 中也没有其他可替代的 owner/user 关联字段

因此，如果要“保留全部数据并迁回 bigint”，唯一可行的恢复方式是：

**把现有 2 个 session 全部挂到当前唯一用户 `wzw` 名下。**

## 3. 目标

1. 修复 `/api/user` 的 500 问题
2. 将 `users`、`chat_sessions`、`chat_messages` 三张表迁回 bigint 体系
3. 为 `chat_sessions` 补回 `user_id`
4. 保留现有 2 个 session 和 58 条 message 数据
5. 迁移后继续沿用当前代码里的 `int`/`BigInteger` 契约，不改接口设计

## 4. 非目标

以下内容不在本次范围内：

- 不切换成 string ID 方案
- 不做 int/string 双栈兼容
- 不尝试恢复不可得的原始 session-user 归属信息
- 不修改 token usage 相关逻辑
- 不顺手重构其他路由或模型

## 5. 设计方案

### 5.1 总体策略

采用 **新列迁移 + 数据回填 + 正式切换**，不做原地改列类型。

原因：
- 现有 UUID/string 值无法直接安全转换为 bigint
- 新列迁移可以保留映射关系，便于核对和回滚
- 能分阶段验证数据是否正确

### 5.2 用户表迁移

对 `users`：

1. 新增 bigint 新主键列，例如 `new_id`
2. 为当前唯一用户生成新的 bigint ID
3. 保留原有 `name`、`created_at`
4. 暂时保留旧字符串 `id` 用于映射和核对

最终切换后：
- `users.id` 成为 bigint 主键

### 5.3 会话表迁移

对 `chat_sessions`：

1. 新增 bigint 新主键列，例如 `new_id`
2. 新增 bigint 新关联列，例如 `new_user_id`
3. 为每条现有 session 生成新的 bigint session ID
4. 因原始归属丢失，将所有 session 的 `new_user_id` 全部指向唯一用户 `wzw` 的新 bigint ID
5. 保留 `title`、`created_at`、`updated_at`、`message_count`、`summary`、`summary_sequence`

最终切换后：
- `chat_sessions.id` 为 bigint
- `chat_sessions.user_id` 为 bigint

### 5.4 消息表迁移

对 `chat_messages`：

1. 新增 bigint 新主键列，例如 `new_id`
2. 新增 bigint 新外键列，例如 `new_session_id`
3. 为每条 message 生成新的 bigint message ID
4. 通过旧 `chat_sessions.id(UUID)` -> 新 `chat_sessions.new_id(bigint)` 的映射，回填 `new_session_id`
5. 保留 `role`、`content`、`created_at`、`sequence`

最终切换后：
- `chat_messages.id` 为 bigint
- `chat_messages.session_id` 为 bigint

### 5.5 正式切换

当三张表新列回填完成且数据核对通过后：

1. 将新 bigint 列切换为正式字段
2. 移除旧字符串 ID 列
3. 为正式字段恢复：
   - 主键约束
   - 非空约束
   - 必要索引
4. 确保 ORM 与真实库结构重新一致

## 6. 数据语义说明

本次迁移有一个明确且已确认的语义变化：

- 旧库中 session 原始归属信息已经丢失
- 迁移后，现有所有 session 都会归到唯一用户 `wzw` 名下

这是当前“保留数据 + 恢复 bigint 契约”下唯一合理且可执行的恢复策略。

## 7. 文件与变更边界

预计涉及：

- `models/user.py`
- `models/chat_session.py`
- `models/chat_message.py`
- `routers/user.py`
- `routers/session.py`
- 迁移脚本/一次性修复脚本（待实现阶段确定具体路径）
- 相关测试文件

数据库层面需要执行 schema 修复与数据迁移。

本次不应修改：

- `chains/`
- `services/token_usage.py`
- `routers/chat.py`
- `frontend/index.html`
- RAG / memory 逻辑

## 8. 验收标准

### 8.1 接口验收

需要确认：

- `GET /api/user` 不再返回 500
- `GET /api/user` 返回的 `id` 为整数
- `POST /api/user` 创建的新用户 ID 为整数
- `GET /api/session?user_id=<id>` 可返回原有 2 个 session
- `GET /api/session/{id}/history?user_id=<id>` 可返回原有 message 数据

### 8.2 数据验收

需要确认：

- `users.id` 为 bigint
- `chat_sessions.id` 为 bigint
- `chat_sessions.user_id` 存在且为 bigint
- `chat_messages.id` 为 bigint
- `chat_messages.session_id` 为 bigint
- 原有 2 个 session 与 58 条 message 仍可读

### 8.3 代码验收

需要确认：

- ORM 模型与真实数据库结构一致
- `UserResponse.id`、`CreateSessionRequest.user_id`、`SessionResponse.id` 等 int 契约继续成立
- 现有相关测试通过，必要处补充新的迁移后行为测试

## 9. 实施结论

本次将按以下策略进入实现规划：

1. 通过新列迁移方式将三张表迁回 bigint 体系
2. 保留现有用户、session、message 数据
3. 将全部现有 session 统一挂到唯一用户 `wzw` 名下
4. 完成 schema 切换后，保持当前代码侧 int 体系不变
5. 用接口与测试双重验证迁移结果
