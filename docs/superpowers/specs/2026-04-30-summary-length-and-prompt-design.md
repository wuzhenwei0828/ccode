# 摘要长度与提示词优化设计

- 日期：2026-04-30
- 主题：将 `summary_max_length` 调整为 1000，并优化摘要提示词使其更简短精炼

## 1. 背景

当前对话摘要逻辑位于 `services/memory_service.py`，其最大摘要长度由 `config/settings.py` 中的 `MemoryConfig.summary_max_length` 控制，当前默认值为 `2000`。用户希望：

1. 将 `summary_max_length` 改为 `1000`
2. 优化摘要生成提示词，明确要求“摘要要简短精炼”

用户已确认本次范围仅限于：

- 修改摘要最大长度默认值
- 修改摘要 prompt 文案

不涉及摘要触发时机、数据库结构、缓存恢复逻辑或其他 memory 机制的调整。

## 2. 目标

1. 将 `MemoryConfig.summary_max_length` 默认值从 `2000` 调整为 `1000`
2. 在首次摘要和增量摘要两处 prompt 中增加“摘要要简短精炼，只保留关键信息”的要求
3. 保持现有摘要生成链路、调用方式和持久化逻辑不变

## 3. 非目标

以下内容不在本次范围内：

- 不修改摘要触发阈值或更新间隔
- 不增加摘要结果截断逻辑
- 不改数据库表结构
- 不改缓存策略或恢复逻辑
- 不改 LLM 调用方式
- 不调整日志行为

## 4. 现状分析

### 4.1 配置位置

`config/settings.py` 中：

- `MemoryConfig.summary_max_length` 默认值当前为 `2000`
- `MemoryService.config` 通过 `get_settings().get_memory_config()` 读取配置

因此只要调整 `MemoryConfig` 默认值，摘要逻辑中所有 `self.config.summary_max_length` 的引用都会自动生效。

### 4.2 提示词位置

`services/memory_service.py` 中 `_generate_summary()` 包含两类摘要 prompt：

1. **增量摘要场景**
   - 旧摘要 + 新消息
   - 让模型生成整合后的新摘要

2. **首次摘要场景**
   - 直接总结最近一段对话

当前两类 prompt 都只强调“控制在 X 字以内”，没有显式约束输出风格必须简短精炼。

## 5. 设计方案

### 5.1 配置修改

改动位置：`config/settings.py`

将：

```python
summary_max_length: int = Field(default=2000, description="Max characters for conversation summary")
```

修改为：

```python
summary_max_length: int = Field(default=1000, description="Max characters for conversation summary")
```

这样可以在不改变调用方的前提下，将摘要目标长度统一收紧到 `1000`。

### 5.2 增量摘要提示词修改

改动位置：`services/memory_service.py` 中旧摘要 + 新消息整合 prompt。

当前目标保持不变：
- 整合旧摘要和新内容
- 生成新的摘要

新增风格要求：
- 摘要要简短精炼
- 只保留关键信息

期望语义为：

- 先整合历史摘要与新增消息
- 再压缩表达
- 避免冗长复述和细枝末节

### 5.3 首次摘要提示词修改

改动位置：`services/memory_service.py` 中首次总结 prompt。

当前目标保持不变：
- 总结对话主要内容

新增风格要求：
- 摘要要简短精炼
- 只保留关键信息

期望效果与增量摘要保持一致，避免首次摘要偏长、偏散。

### 5.4 统一落地原则

本次 prompt 调整遵循以下原则：

1. 不改变 prompt 的核心任务，只增强风格约束
2. 继续使用 `self.config.summary_max_length` 控制字数上限
3. 不新增新的配置字段
4. 不在 Python 侧追加字符串裁剪逻辑
5. 让模型优先通过提示词自我约束输出长度与精炼度

## 6. 文件变更边界

本次预期修改文件：

- 修改：`config/settings.py`
- 修改：`services/memory_service.py`

本次不应修改：

- `models/`
- `routers/`
- `chains/`
- `main.py`
- 数据库结构与 migration
- 其他 memory 相关方法

## 7. 验收标准

### 7.1 配置验收

需要确认：

- `MemoryConfig.summary_max_length` 默认值为 `1000`

### 7.2 提示词验收

需要确认：

- 首次摘要 prompt 明确要求“摘要要简短精炼，只保留关键信息”
- 增量摘要 prompt 明确要求“摘要要简短精炼，只保留关键信息”
- 仍保留长度限制描述

### 7.3 回归验收

需要确认：

- `_generate_summary()` 的调用链未改变
- `self.config.summary_max_length` 仍正常参与 prompt 拼接
- 不引入其他逻辑变更

### 7.4 代码验收

需要确认：

- 代码通过基础语法检查
- 变更仅限本次需求相关文件

## 8. 实施原则

1. 只做最小必要改动
2. 不扩展到摘要链路其他部分
3. 配置改动与提示词改动保持一致
4. 保持现有代码结构和风格不变

## 9. 实施结论

本次将按以下方案进入实现计划阶段：

- 将 `summary_max_length` 默认值修改为 `1000`
- 优化首次摘要与增量摘要两处 prompt
- 明确要求摘要“简短精炼，只保留关键信息”
- 不修改摘要生成时机、存储、缓存、恢复或截断逻辑

## 10. 自检结果

- 无 TBD / TODO / 占位内容
- 目标、非目标、改动位置、边界与验收标准已明确
- 范围聚焦于单一摘要配置优化目标，可直接进入 implementation plan
- 各章节描述一致：均以“1000 字上限 + 更精炼的摘要提示词”为中心
