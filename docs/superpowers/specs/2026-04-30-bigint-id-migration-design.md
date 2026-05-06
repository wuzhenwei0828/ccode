# 全表主键切换为 BIGINT 自增设计

- 日期：2026-04-30
- 主题：将现有所有表的主键 `id` 统一改为 `BIGINT AUTO_INCREMENT`，并同步迁移历史数据与所有引用关系

## 1. 背景

当前系统中的核心表主键与关联字段主要使用字符串 UUID 体系：

- `users.id`
- `chat_sessions.id`
- `chat_sessions.user_id`
- `chat_messages.id`
- `chat_messages.session_id`

用户希望统一改为：

- 所有表主键 `id` 均使用 `BIGINT` 自增
- 所有关联字段同步切换为 `BIGINT`
- 历史数据库数据也一并迁移
- 不保留 UUID 与 bigint 双轨兼容

这意味着本次不是单纯 ORM 字段修改，而是一轮涉及**数据模型、历史数据迁移、接口类型、前端参数传递、缓存 key 语义和测试数据**的全链路切换。

## 2. 目标

1. 将所有核心表主键统一改为 `BIGINT AUTO_INCREMENT`
2. 将所有相关外键/引用字段统一改为 `BIGINT`
3. 保留旧数据中的 user / session / message 归属关系
4. 同步更新后端模型、路由、服务、前端与测试
5. 不保留旧 UUID 兼容逻辑

## 3. 非目标

以下内容不在本次范围内：

- 不保留 UUID / bigint 双轨兼容
- 不引入新的登录鉴权机制
- 不重构 memory 架构
- 不扩展用户级跨 session 记忆共享
- 不顺带重构与主键类型无关的模块

## 4. 现状分析

### 4.1 当前核心数据表

当前至少包含以下核心表：

- `users`
- `chat_sessions`
- `chat_messages`

其中：
- `users.id` 当前为字符串类型主键
- `chat_sessions.id` 当前为字符串类型主键
- `chat_sessions.user_id` 当前为字符串类型引用
- `chat_messages.id` 当前为字符串类型主键
- `chat_messages.session_id` 当前为字符串类型引用

### 4.2 当前代码中的 ID 语义

当前代码中 `session_id` / `user_id` 广泛作为字符串处理，涉及：

- `models/`
- `routers/`
- `services/memory_service.py`
- `chains/chat_chain.py`
- `chains/rag_chain.py`
- `frontend/index.html`
- 测试代码中的 mock / fixture / 断言

因此这次迁移必须同时修改：
- 数据库存储类型
- Python 类型语义
- JS 端参数传递语义
- 缓存与查询逻辑

### 4.3 直接改列的风险

不能简单地把现有 `VARCHAR(36)` 主键直接改成 `BIGINT`，因为：

1. UUID 文本无法直接转换成 bigint
2. 关联表之间的引用关系需要同步映射
3. 如果先改主表不改子表，会导致关系断裂

因此本次必须采用**分阶段迁移**，通过映射关系回填新列，再完成最终切换。

## 5. 设计方案

### 5.1 目标主键策略

统一采用：

- 主键：`BIGINT AUTO_INCREMENT`
- 外键/引用字段：`BIGINT`

适用对象：

- `users.id`
- `chat_sessions.id`
- `chat_messages.id`
- `chat_sessions.user_id`
- `chat_messages.session_id`

不再使用字符串 UUID 作为主键或核心引用键。

### 5.2 迁移总体策略

采用四阶段迁移：

1. **新增 bigint 列**
   - 为主键和引用字段增加临时 bigint 列
2. **回填映射关系**
   - 为旧 UUID 与新 bigint 建立映射，并回填引用列
3. **代码切换**
   - 代码层统一改为 bigint 语义
4. **最终清理**
   - 移除旧 UUID 列，切换正式主键/引用关系

这种方式能保证历史数据关系不丢失，并降低一次性改结构导致的数据损坏风险。

### 5.3 数据迁移顺序

建议按依赖顺序迁移：

#### 第一步：迁移 `users`
- 为 `users` 建立新的 bigint 主键
- 生成旧 `user_uuid -> new_user_id` 映射

#### 第二步：迁移 `chat_sessions`
- 为 `chat_sessions` 建立新的 bigint 主键
- 根据用户映射，把 `chat_sessions.user_id` 回填为新的 bigint 用户 ID
- 生成旧 `session_uuid -> new_session_id` 映射

#### 第三步：迁移 `chat_messages`
- 为 `chat_messages` 建立新的 bigint 主键
- 根据 session 映射，把 `chat_messages.session_id` 回填为新的 bigint session ID

顺序不能反，否则引用关系无法正确回填。

### 5.4 代码层切换原则

#### 模型层
修改：
- `models/user.py`
- `models/chat_session.py`
- `models/chat_message.py`

要求：
- 所有主键改为 `BigInteger`
- 所有关联字段改为 `BigInteger`
- 所有模型查询方法签名同步改为 `int`

#### 路由层
修改：
- `routers/user.py`
- `routers/session.py`
- 可能涉及 chat 入口相关 router

要求：
- 请求体和响应体中的 `id` / `user_id` / `session_id` 改成整数语义
- query/path 参数按 `int` 处理

#### 服务层
修改：
- `services/memory_service.py`

要求：
- 缓存 dict 的 key 从字符串 session_id 切到整数 session_id
- 所有 `session_id` 相关方法签名同步改成 `int`
- 摘要恢复、消息写入、缓存读取逻辑保持不变，只切换 ID 类型

#### Chain 层
修改：
- `chains/chat_chain.py`
- `chains/rag_chain.py`

要求：
- 所有 `session_id` 参数改为整数语义
- 不改变业务流程，仅改变 ID 类型处理

#### 前端
修改：
- `frontend/index.html`

要求：
- `currentUserId`、`sessionId` 按数字使用
- `localStorage` 取出的用户 ID 需要与后端返回的 bigint 语义一致
- 所有 session/user 请求继续工作

#### 测试
修改：
- 所有使用 UUID 字符串的 fixture、mock、断言

要求：
- 全部切换为整数 ID 语义
- 保留原测试覆盖范围

## 6. 文件变更边界

本次预期修改文件：

- `models/user.py`
- `models/chat_session.py`
- `models/chat_message.py`
- `routers/user.py`
- `routers/session.py`
- `services/memory_service.py`
- `chains/chat_chain.py`
- `chains/rag_chain.py`
- `frontend/index.html`
- 相关测试文件
- 数据库初始化或迁移相关文件

本次不应修改：

- 与主键迁移无关的业务功能逻辑
- 与 UI 样式无关的展示细节
- 与摘要策略无关的 memory 行为

## 7. 风险点

### 7.1 历史数据映射风险
若映射步骤错误，可能导致：
- session 丢失所属 user
- message 丢失所属 session

### 7.2 代码类型不一致风险
如果只改模型，不改调用链，会出现：
- 查询不到数据
- 缓存命中失败
- 前端请求参数类型错位

### 7.3 浏览器缓存失效
现有 `localStorage` 中保存的旧字符串 `user_id` 将不再可用。前端需要在切换后自然覆盖新值，旧值可视为失效缓存。

### 7.4 外部调用方不兼容
任何仍按 UUID 传参的调用都会直接失效，因为本次明确不做兼容。

## 8. 验收标准

### 8.1 数据结构验收
需要确认：

- `users.id` 为 bigint 自增
- `chat_sessions.id` 为 bigint 自增
- `chat_messages.id` 为 bigint 自增
- `chat_sessions.user_id` 为 bigint
- `chat_messages.session_id` 为 bigint

### 8.2 数据迁移验收
需要确认：

- 历史 user 数据保留
- 历史 session 数据保留
- 历史 message 数据保留
- user-session-message 归属关系保持正确

### 8.3 代码验收
需要确认：

- 所有 `id` / `user_id` / `session_id` 相关模型、接口、服务、前端语义已改为 bigint
- 自动化测试更新并通过
- 基础语法检查通过

### 8.4 功能验收
需要确认：

- 用户创建正常
- 用户切换正常
- 新建 session 正常
- 历史消息加载正常
- 聊天和 RAG 调用正常
- memory 摘要与上下文恢复正常

## 9. 实施原则

1. 不保留旧 UUID 兼容逻辑
2. 先完成数据结构映射，再切换代码
3. 所有引用字段必须同步切换
4. 不在本次迁移中顺带重构无关功能
5. 保持现有业务流程不变，只替换主键体系

## 10. 实施结论

本次将按以下方案进入实现计划阶段：

- 全量切换为 bigint 自增主键
- 所有关联字段同步改为 bigint
- 通过映射和分阶段迁移保留历史数据关系
- 同步修改模型、接口、前端、memory、chain 和测试
- 不保留 UUID 双轨兼容

## 11. 自检结果

- 无 TBD / TODO / 占位内容
- 迁移策略、数据顺序、代码影响范围和风险已明确
- 范围聚焦于“全表 bigint 主键迁移”这一单一目标
- 各章节描述一致：均以“全量切换 bigint、自增主键、同步迁移引用关系”为中心
