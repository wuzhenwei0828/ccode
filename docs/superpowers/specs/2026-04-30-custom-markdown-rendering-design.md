# 前端自写 Markdown 转 HTML 设计

- 日期：2026-04-30
- 主题：用自写基础子集 Markdown 转 HTML 函数替换当前 markdown-it 渲染

## 1. 背景

当前前端 assistant 消息渲染已经从 `marked` 切换到 `markdown-it`，并保留了 `DOMPurify` 做安全清洗。之后在实际使用中暴露出渲染链路问题，用户进一步明确提出：不再使用第三方 Markdown 渲染库，而是直接在前端自己实现一个 Markdown 转 HTML 的函数。

用户已确认：

- 使用自写转换函数
- 仅支持基础子集 Markdown
- 仅 assistant 消息走 Markdown 渲染
- 用户消息继续保持纯文本展示
- 保留 `DOMPurify` 做输出清洗

本次目标是将前端 Markdown 渲染能力收敛为一个边界明确、可控的小型实现，满足聊天展示场景，不追求完整 Markdown 规范兼容。

## 2. 目标

1. 去除 `markdown-it` 依赖与静态资源引用
2. 在前端实现一个基础子集 Markdown 转 HTML 函数
3. 仅用于 assistant 消息渲染
4. 保留 `DOMPurify.sanitize(...)` 做最终 HTML 清洗
5. 保持历史消息、新消息、流式更新三条路径的渲染入口一致

## 3. 非目标

以下内容不在本次范围内：

- 不实现完整 CommonMark 或 GFM 规范
- 不支持表格、任务列表、HTML 原样透传
- 不支持复杂嵌套列表和复杂混合语法
- 不改后端接口与消息数据结构
- 不改用户消息的渲染方式
- 不重构整页前端架构

## 4. 现状分析

### 4.1 当前渲染链路

当前 `frontend/index.html` 中：

- `renderMessages()` 用于历史消息与本地新消息整体渲染
- `updateLastMessage()` 用于流式更新 assistant 最后一条消息
- `renderMd(text)` 作为 assistant 渲染入口
- 当前 `renderMd(text)` 依赖 `window.markdownit(...)`
- 输出结果再经过 `DOMPurify.sanitize(...)`

因此，本次最合适的修改点不是改调用方，而是替换 `renderMd` 内部实现。

### 4.2 当前静态资源

当前前端静态资源中已有：

- `frontend/js/markdown-it.min.js`
- `frontend/js/purify.min.js`
- 历史上还保留过 `frontend/js/marked.min.js`

本次完成后：

- `markdown-it.min.js` 应删除
- `purify.min.js` 保留
- 不再依赖 Markdown 第三方转换库

## 5. 设计方案

### 5.1 对外接口保持不变

保留现有调用方式：

- `renderMessages()` 继续调用 `renderMd(m.content)`
- `updateLastMessage()` 继续调用 `renderMd(content)`

这样历史消息、新消息、流式消息三条路径都能复用同一入口，减少回归风险。

### 5.2 函数边界

建议将实现分成两层，但都保留在 `frontend/index.html` 中：

#### `renderMd(text)`
职责：
- 规范化输入换行
- 调用 Markdown 转 HTML 主函数
- 调用 `DOMPurify.sanitize(...)`
- 返回最终 HTML 字符串

#### `markdownToHtml(text)`
职责：
- 实现基础子集 Markdown 到 HTML 的转换
- 内部按“先块级、后行内”的顺序处理

这样可以把“入口控制”和“解析逻辑”分开，方便后续排查问题。

### 5.3 支持的块级语法

本次支持的块级语法限定为：

1. fenced code block：
   - 形如 ```` ```lang ... ``` ````
   - 输出 `<pre><code>...</code></pre>`
   - 代码内容需要先做 HTML 转义，再包装

2. 标题：
   - 支持 `#` 到 `####`
   - 输出对应 `h1` 到 `h4`

3. 引用：
   - 以 `>` 开头
   - 输出 `<blockquote>...</blockquote>`

4. 无序列表：
   - 支持 `- ` 或 `* ` 开头
   - 输出 `<ul><li>...</li></ul>`

5. 有序列表：
   - 支持 `1. ` 形式
   - 输出 `<ol><li>...</li></ol>`

6. 普通段落：
   - 作为默认块级内容
   - 输出 `<p>...</p>`

### 5.4 支持的行内语法

在块级文本内容内部再处理以下行内语法：

1. 行内代码：
   - `` `code` ``
   - 输出 `<code>...</code>`

2. 链接：
   - `[text](url)`
   - 输出 `<a href="...">...</a>`

3. 粗体：
   - `**text**`
   - 输出 `<strong>...</strong>`

4. 斜体：
   - `*text*`
   - 输出 `<em>...</em>`

### 5.5 处理顺序

为了降低语法相互干扰，建议处理顺序如下：

1. 先按行扫描，识别 fenced code block
2. 对非代码块部分做块级拆分
3. 每个块内部再做行内语法替换
4. 统一拼接为 HTML
5. 交给 `DOMPurify.sanitize(...)`

关键原则：
- 代码块内部不再做行内 Markdown 替换
- 行内代码优先于粗体和斜体处理，避免误替换
- 链接文本内部不额外扩展复杂嵌套语法

## 6. 安全策略

### 6.1 输出清洗保留

即使是自写函数输出的 HTML，也必须继续经过：

```js
DOMPurify.sanitize(html)
```

不允许直接将未经清洗的字符串写入 DOM。

### 6.2 原始文本转义

在自写解析过程中：

- 普通文本片段应先做 HTML 转义
- 只有明确识别出的 Markdown 语法，才生成对应 HTML 标签
- 代码块和行内代码中的原始内容必须转义

这能避免把原始消息内容当作可执行 HTML 插入页面。

## 7. 文件变更边界

本次预期修改文件：

- 修改：`frontend/index.html`
- 修改：`package.json`
- 修改：`package-lock.json`
- 删除：`frontend/js/markdown-it.min.js`

本次保留文件：

- `frontend/js/purify.min.js`

本次不应改动：

- 后端 Python 文件
- 历史消息接口与流式接口结构
- 会话逻辑、RAG、provider 逻辑

## 8. 验收标准

### 8.1 assistant 消息功能验收

assistant 消息需要支持并正确展示：

- 标题
- 普通段落
- 换行
- 粗体
- 斜体
- 行内代码
- fenced code block
- 引用
- 无序列表
- 有序列表
- 链接

### 8.2 用户消息回归验收

用户消息必须继续满足：

- 纯文本展示
- 不做 Markdown 渲染

### 8.3 渲染链路验收

需要确认以下三条路径行为一致：

- 历史消息加载
- 新消息展示
- 流式输出更新

### 8.4 安全验收

需要确认：

- 输出 HTML 经过 `DOMPurify.sanitize(...)`
- 用户或模型返回的恶意脚本片段不会直接注入 DOM

### 8.5 依赖验收

需要确认：

- 页面不再依赖 `markdown-it`
- 删除 `frontend/js/markdown-it.min.js`
- 删除 `package.json` / `package-lock.json` 中的 `markdown-it`
- 保留 `purify.min.js`

## 9. 实施原则

1. 只实现用户确认的基础子集语法
2. 保持外部调用入口不变
3. 不扩展为完整 Markdown 引擎
4. 不在本次任务中做无关前端重构
5. 保持与现有代码风格一致

## 10. 实施结论

本次将按以下方案进入实现计划阶段：

- 移除 `markdown-it` 依赖与静态资源
- 在 `frontend/index.html` 中自写 `markdownToHtml` + `renderMd`
- 仅 assistant 消息使用该函数
- 保留 `DOMPurify`
- 历史消息、新消息、流式更新统一走 `renderMd`
- 支持基础子集 Markdown，不支持完整规范

## 11. 自检结果

- 无 TBD / TODO / 占位内容
- 目标、非目标、支持语法、处理顺序、改动边界和验收标准已明确
- 范围聚焦单一前端子目标，可直接进入 implementation plan
- 各章节描述一致：均以“自写基础子集 Markdown 转 HTML、保留 DOMPurify、仅 assistant 使用”为中心
