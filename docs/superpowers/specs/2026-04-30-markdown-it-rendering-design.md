# 前端 Markdown 渲染切换设计

- 日期：2026-04-30
- 主题：将前端 assistant 消息渲染从 marked 切换为 markdown-it

## 1. 背景

当前前端页面在 `frontend/index.html` 中通过 `marked.min.js` 渲染 assistant 消息，并额外依赖 `frontend/js/autoDivBlock.js` 做自定义块级分段处理。用户希望改为使用 `markdown-it` 渲染前端页面，并已确认：

- 选择方案 2：使用 `markdown-it`，同时移除现有自定义分块扩展逻辑
- 仅 assistant 消息使用 Markdown 渲染
- 用户消息继续保持纯文本展示

本次变更目标是以最小范围完成前端 Markdown 渲染库替换，不改动后端接口与非渲染逻辑。

## 2. 目标

1. 将 assistant 消息的 Markdown 渲染库由 `marked` 替换为 `markdown-it`
2. 删除或停用 `frontend/js/autoDivBlock.js`
3. 保留 `DOMPurify` 作为 HTML 安全清洗层
4. 保持现有页面结构与样式体系尽量稳定
5. 不改变用户消息、会话、RAG、provider 等既有逻辑

## 3. 非目标

以下内容不在本次范围内：

- 不改后端接口返回格式
- 不引入额外 Markdown 插件体系
- 不重构整个前端脚本结构
- 不修改用户消息的渲染方式
- 不新增代码高亮、目录、锚点等增强能力
- 不调整会话管理、知识库面板、RAG 开关、模型切换相关逻辑

## 4. 现状分析

### 4.1 当前依赖

- `frontend/index.html` 当前引入 `/static/js/marked.min.js`
- 页面已使用 `DOMPurify`
- `package.json` 当前包含 `dompurify` 依赖
- `frontend/js/autoDivBlock.js` 为 `marked` 自定义块扩展逻辑

### 4.2 当前样式基础

`frontend/index.html` 中已经存在 `.markdown-body` 及其子元素样式，覆盖了：

- 段落 `p`
- 代码块 `pre`
- 行内代码 `code`
- 其他常见 Markdown 内容样式

因此本次应优先复用现有样式，只在发现 `markdown-it` 输出标签与现有样式不兼容时做最小调整。

## 5. 设计方案

### 5.1 渲染范围

- **assistant 消息**：使用 `markdown-it` 将原始文本渲染为 HTML
- **用户消息**：继续使用纯文本展示，不走 Markdown 解析

这样可以把行为变化限制在 assistant 回复区域，降低回归风险。

### 5.2 渲染链路

assistant 消息的新渲染流程为：

1. 读取后端返回的原始文本
2. 使用 `markdown-it.render(text)` 生成 HTML
3. 使用 `DOMPurify.sanitize(html)` 对 HTML 做清洗
4. 将清洗后的 HTML 写入 assistant 消息容器
5. 保持 `.markdown-body` 作为渲染内容的样式包装节点

即：

`assistant 原始文本 -> markdown-it.render -> DOMPurify.sanitize -> 插入 .message-content`

用户消息仍保持：

`user 原始文本 -> 纯文本插入`

### 5.3 自定义分块逻辑处理

本次明确移除 `autoDivBlock.js` 的参与：

- 不再继续维护 `marked` 的自定义扩展行为
- 不迁移这段扩展逻辑到 `markdown-it`
- 以 `markdown-it` 的标准段落和块级语义作为最终展示依据

这样做的结果是：

- assistant 消息将按标准 Markdown 规则分段
- 原先依赖连续文本自动包裹自定义 `<div>` 的行为不再保留
- 展示结果更标准，也更容易维护

### 5.4 前端资源调整

涉及的资源调整如下：

#### `frontend/index.html`
- 删除 `marked.min.js` 的引入
- 删除 `autoDivBlock.js` 的引入
- 增加 `markdown-it` 的引入
- 将 assistant 消息渲染入口从 `marked` 替换为 `markdown-it`

#### `frontend/js/autoDivBlock.js`
- 停止使用
- 若确认项目内没有其他地方引用，可直接删除

#### `package.json`
- 新增 `markdown-it` 依赖
- 保留 `dompurify`

## 6. 文件变更边界

本次预期改动文件：

- 修改：`frontend/index.html`
- 修改：`package.json`
- 可能删除：`frontend/js/autoDivBlock.js`

本次不应改动：

- `main.py`
- `routers/`
- `services/`
- `models/`
- `chains/`
- 其他后端逻辑与接口文件

## 7. 兼容性与风险

### 7.1 段落表现差异

由于移除了 `autoDivBlock.js`，assistant 消息的段落切分将完全遵循标准 Markdown。可能出现：

- 某些原本连续文本的视觉分组方式变化
- 段落间距与之前略有不同

应通过现有 `.markdown-body` 样式做最小兼容调整，而不是重新设计排版体系。

### 7.2 样式兼容性

`markdown-it` 输出的 HTML 结构可能与当前 `marked` + 自定义扩展组合略有差异，因此需要重点确认以下元素样式：

- `p`
- `ul` / `ol` / `li`
- `blockquote`
- `pre` / `code`
- `a`
- 标题标签如 `h1` ~ `h6`

如有问题，仅补充必要样式，不扩散到整页视觉重构。

### 7.3 安全性

`markdown-it` 渲染产生的是 HTML 字符串，必须继续经过 `DOMPurify.sanitize` 后再插入页面。不得直接将未经清洗的 HTML 写入 DOM。

## 8. 测试与验收标准

### 8.1 功能验收

以下内容需要能正常显示于 assistant 消息区域：

- 普通段落文本
- 标题
- 无序列表与有序列表
- 引用块
- 行内代码
- 代码块
- 链接

### 8.2 用户消息回归

需要确认：

- 用户消息仍以纯文本方式展示
- 用户输入中的 Markdown 标记不会被渲染成 HTML

### 8.3 历史与实时消息一致性

需要确认：

- 历史消息加载时的 assistant 渲染行为正确
- 新发送后的 assistant 实时回复渲染行为正确
- 两类路径展示一致

### 8.4 安全验收

需要确认：

- assistant 返回的潜在 HTML 片段不会未经清洗直接注入页面
- `DOMPurify` 仍在 assistant 渲染链路中生效

### 8.5 样式验收

需要确认：

- `.markdown-body` 在常见 Markdown 内容下排版可读
- 段落、列表、代码块间距无明显错乱
- 不因移除 `autoDivBlock.js` 产生消息粘连或布局塌陷

## 9. 实施原则

1. 仅做完成目标所需的最小修改
2. 优先复用现有页面脚本和样式
3. 不扩展到无关前端重构
4. 不调整后端行为来适配前端渲染库切换
5. 保持代码风格与现有项目一致

## 10. 实施结论

本次将按以下方案进入实现计划阶段：

- 用 `markdown-it` 替换 `marked`
- 仅 assistant 消息使用 Markdown 渲染
- 删除或停用 `frontend/js/autoDivBlock.js`
- 保留 `DOMPurify` 清洗链路
- 只修改前端渲染相关文件，控制影响范围

## 11. 自检结果

- 无 TBD / TODO / 占位描述
- 目标、非目标、改动边界、风险和验收标准已明确
- 方案范围聚焦于单一子目标，可直接进入 implementation plan
- 各章节描述一致：均以“仅 assistant 使用 markdown-it、移除 autoDivBlock、保留 DOMPurify”为中心
