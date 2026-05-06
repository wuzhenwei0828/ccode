# Chatbot Service

基于 FastAPI 和 LangChain 构建的生产级对话式 AI 服务，支持检索增强生成（RAG）、双层记忆（短期滑动窗口 + 长期 MySQL 持久化），以及可插拔的 LLM 提供商架构（支持 OpenAI、Anthropic Claude、百度文心和阿里通义千问）。

**核心特性**

- **可插拔 LLM 架构** — 基于注册表的工厂模式，通过 YAML 配置或环境变量在 OpenAI、Claude、文心、千问之间无缝切换，无需修改代码
- **RAG 知识库** — 文档解析（PDF、DOCX、TXT、Markdown）、`RecursiveCharacterTextSplitter` 自动分块、Milvus 向量存储语义检索、基于上下文的知识问答
- **双层记忆** — 内存滑动窗口用于 LLM 上下文（窗口大小可配置）+ MySQL 持久化对话历史，支持跨会话检索
- **SSE 流式输出** — Server-Sent Events 实现逐字流式响应
- **多格式文档支持** — 解析 PDF（PyPDF）、Word 文档（python-docx）以及纯文本/Markdown 文件
- **Docker Compose 部署** — 一条命令启动 MySQL、Milvus、etcd 和 MinIO 全栈基础设施
- **生产级前端** — 暗色主题 SPA 聊天界面，支持会话管理、RAG 模式切换、提供商选择和知识库面板

---

## 目录

- [技术栈](#技术栈)
- [前置条件](#前置条件)
- [快速开始](#快速开始)
- [架构设计](#架构设计)
- [API 接口](#api-接口)
- [配置说明](#配置说明)
- [环境变量](#环境变量)
- [常用命令](#常用命令)
- [测试](#测试)
- [部署](#部署)
- [前端](#前端)
- [故障排查](#故障排查)

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **语言** | Python 3.11+ |
| **Web 框架** | FastAPI 0.115+ |
| **LLM 编排** | LangChain 0.3+ |
| **LLM 提供商** | OpenAI (GPT-4o)、Anthropic Claude (Sonnet 4)、百度文心 4.0、阿里通义千问-Max |
| **Embedding** | OpenAI text-embedding-3-small（可配置，支持 Ollama/本地模型） |
| **向量数据库** | Milvus 2.4 |
| **关系型数据库** | MySQL 8.0（通过 SQLAlchemy 2.0 + PyMySQL） |
| **文档解析** | PyPDF、python-docx、Markdown |
| **文本分块** | LangChain RecursiveCharacterTextSplitter |
| **容器化** | Docker + Docker Compose |
| **前端** | 原生 HTML/CSS/JS SPA |

---

## 前置条件

- **Python 3.11+**（通过 pyenv、asdf 或系统 Python）
- **MySQL 8.0+**（或使用 Docker Compose 运行）
- **Milvus 2.4+**（或使用 Docker Compose 配合 etcd + MinIO 运行）
- **Docker & Docker Compose**（推荐用于基础设施部署）
- **Git**

---

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd CCode
```

### 2. 创建并激活虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

这将安装所有依赖：FastAPI、Uvicorn、LangChain、SQLAlchemy、PyMySQL、Milvus SDK、文档解析器和测试工具。

### 4. 环境变量配置

复制环境变量示例文件：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置以下变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `DB_URL` | MySQL 连接字符串 | `mysql+pymysql://root:root@localhost:3306/chatbot` |
| `OPENAI_API_KEY` | OpenAI API 密钥（用于 LLM 和 Embedding） | `sk-xxx` |
| `MILVUS_HOST` | Milvus 服务主机名 | `localhost` |
| `MILVUS_PORT` | Milvus 服务端口 | `19530` |

完整的环境变量参考见 [环境变量](#环境变量) 章节。

### 5. 启动基础设施

最简单的方式是使用 Docker Compose，一次性启动 MySQL、Milvus、etcd 和 MinIO：

```bash
docker compose up -d mysql milvus-standalone
```

等待健康检查通过（Milvus 首次启动约需 60 秒）：

```bash
docker compose ps
```

验证 MySQL 是否正常运行：

```bash
docker compose exec mysql mysqladmin ping -h localhost
# 预期输出: mysqld is alive
```

### 6. 初始化数据库

应用启动时会自动创建表结构。如需手动初始化：

```python
from utils.db import init_db
init_db()
```

这将创建以下 MySQL 表：
- `chat_sessions` — 对话会话
- `chat_messages` — 会话中的消息记录
- `knowledge_bases` — 知识库容器
- `document_metas` — 已索引文档的元数据

### 7. 配置 LLM 提供商

编辑 `config/llm_config.yaml` 设置所需的模型和 API 密钥。环境变量（`OPENAI_API_KEY`、`CLAUDE_API_KEY` 等）会覆盖 YAML 文件中的值。

修改默认提供商，编辑文件底部的 `default_provider` 字段：

```yaml
default_provider: openai  # 或 claude, ernie, qwen
```

### 8. 启动开发服务器

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API 现在运行在 `http://localhost:8000`。

**验证服务：**

```bash
curl http://localhost:8000/
# 预期返回: {"service": "Chatbot Service", "status": "running"}
```

**交互式 API 文档** 可在 `http://localhost:8000/docs` 访问（Swagger UI）。

---

## 架构设计

### 目录结构

```
├── main.py                      # FastAPI 应用入口
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── Dockerfile                   # 生产用 Docker 镜像
├── docker-compose.yml           # 基础设施编排 (MySQL + Milvus)
│
├── config/
│   ├── __init__.py              # 配置包导出
│   ├── settings.py              # Pydantic 配置 (YAML + ENV 合并)
│   └── llm_config.yaml          # LLM 提供商定义
│
├── models/                      # SQLAlchemy ORM 模型
│   ├── __init__.py              # 模型注册 + Base 导出
│   ├── chat_session.py          # ChatSession 模型
│   ├── chat_message.py          # ChatMessage 模型
│   └── knowledge_base.py        # KnowledgeBase + DocumentMeta 模型
│
├── services/                    # 业务逻辑层
│   ├── __init__.py
│   ├── llm_factory.py           # 可插拔 LLM 工厂模式
│   ├── openai_llm.py            # OpenAI 提供商实现
│   ├── claude_llm.py            # Anthropic Claude 提供商
│   ├── chinese_llm.py           # 文心 / 千问 (OpenAI 兼容 API)
│   ├── embedding_service.py     # Embedding 模型工厂
│   ├── memory_service.py        # 双层记忆 (滑动窗口 + MySQL)
│   ├── rag_service.py           # RAG 流水线 (索引 + 检索)
│   └── document_loader.py       # 多格式文档解析与分块
│
├── chains/                      # LangChain 对话链
│   ├── __init__.py
│   ├── chat_chain.py            # 普通对话 (带记忆)
│   └── rag_chain.py             # RAG 增强对话
│
├── routers/                     # FastAPI API 接口
│   ├── __init__.py
│   ├── chat.py                  # POST /api/chat, POST /api/chat/stream
│   ├── session.py               # 会话管理 + 历史记录
│   └── knowledge.py             # 知识库管理 + 文档上传
│
├── utils/                       # 基础设施工具
│   ├── __init__.py
│   ├── db.py                    # SQLAlchemy 引擎 + 会话管理
│   └── milvus_client.py         # Milvus 向量存储封装
│
├── tests/                       # 测试套件
│   ├── __init__.py
│   ├── conftest.py              # Pytest 测试夹具
│   ├── test_llm_factory.py      # LLM 工厂测试
│   ├── test_memory_service.py   # 记忆服务测试
│   └── test_document_loader.py  # 文档加载器测试
│
└── frontend/
    └── index.html               # SPA 聊天前端
```

### 设计模式

**工厂模式（LLM 提供商）**

`LLMFactory` 使用基于注册表的工厂模式。每个提供商在模块加载时自动注册：

```python
LLMFactory.register("openai", _create_openai_model)
LLMFactory.register("claude", _create_claude_model)
LLMFactory.register("ernie", _create_chinese_model)
LLMFactory.register("qwen", _create_chinese_model)
```

当请求到达时，`LLMFactory.create(provider_name)` 查找已注册的工厂，从 YAML 加载提供商配置（带环境变量覆盖），然后实例化对应的 `BaseChatModel`。

**单例模式**

- `get_settings()` — 全局配置单例
- `memory_service` — 记忆服务单例
- `rag_service` — RAG 服务单例

**仓储模式（记忆层）**

`MemoryService` 抽象了两层存储：
1. **短期记忆**：内存字典（`_short_term`）用于快速滑动窗口访问
2. **长期记忆**：MySQL（`ChatMessage` 表）用于持久化历史

### 请求生命周期

```
客户端请求
    │
    ▼
FastAPI 路由 (routers/chat.py)
    │
    ▼
链选择 (ChatChain 或 RAGChain)
    │
    ├── RAG 路径: RAGService.query(Milvus) → 上下文文档
    │
    ▼
LLM 工厂 (LLMFactory.create)
    │
    ▼
记忆服务 (get_context → 滑动窗口)
    │
    ▼
LangChain 管道 (prompt | LLM | parser)
    │
    ├── 同步: invoke() → 完整响应
    ├── 异步: astream() → 逐字 SSE 流
    │
    ▼
记忆服务 (add_message → 保存到缓存 + MySQL)
    │
    ▼
响应客户端 (JSON 或 SSE)
```

### 数据流

**普通对话：**

```
用户消息 → MemoryService.add_message() → [短期缓存 + MySQL]
    │
    ▼
MemoryService.get_context() → 最近 N 条消息 (滑动窗口)
    │
    ▼
ChatPromptTemplate + LLM → StrOutputParser → 响应文本
    │
    ▼
MemoryService.add_message() → 保存助手回复
```

**RAG 对话：**

```
用户提问 → RAGService.query(question, k=4) → Milvus 语义搜索
    │
    ▼
检索 top-k 文档分块 → 构建上下文字符串
    │
    ▼
RAG 提示词 (上下文 + 历史 + 问题) → LLM → 响应
    │
    ▼
保存到记忆 (与普通对话相同)
```

**文档索引：**

```
文件上传 → DocumentLoader.process_document() → 解析 + 分块
    │
    ▼
RAGService.add_texts(chunks, metadatas) → Milvus.add_texts()
    │
    ▼
DocumentMeta 保存到 MySQL (状态: "indexed")
```

### 数据库 Schema

```
chat_sessions (对话会话表)
├── id (VARCHAR(36), 主键, UUID)
├── title (VARCHAR(255), 默认: "New Chat")
├── created_at (DATETIME, 服务器默认 NOW())
├── updated_at (DATETIME, 服务器默认 NOW(), 自动更新)
└── message_count (INT, 默认: 0)

chat_messages (对话消息表)
├── id (VARCHAR(36), 主键, UUID)
├── session_id (VARCHAR(36), 非空, 索引, FK → chat_sessions)
├── role (VARCHAR(20), 非空: "user" / "assistant" / "system")
├── content (TEXT, 非空)
├── created_at (DATETIME, 服务器默认 NOW())
└── sequence (INT, 默认: 0, 消息序号)

knowledge_bases (知识库表)
├── id (VARCHAR(36), 主键, UUID)
├── name (VARCHAR(255), 非空)
├── description (TEXT, 默认: "")
└── created_at (DATETIME, 服务器默认 NOW())

document_metas (文档元数据表)
├── id (VARCHAR(36), 主键, UUID)
├── kb_id (VARCHAR(36), 非空, 索引, FK → knowledge_bases)
├── file_name (VARCHAR(255), 非空)
├── file_path (VARCHAR(512), 默认: "")
├── status (VARCHAR(20), 默认: "pending", 值: pending/indexed/failed)
└── created_at (DATETIME, 服务器默认 NOW())
```

### Milvus 集合结构

默认集合（`knowledge_base`）存储：
- 文本分块（嵌入向量）
- 元数据：`doc_id`、`kb_id`、`file_name`

Embedding 向量维度默认为 1536（OpenAI `text-embedding-3-small`）。

---

## API 接口

### 对话接口

#### `POST /api/chat`

发送消息并获取完整响应。

**请求体：**

```json
{
  "session_id": "abc-123",
  "message": "这个项目是做什么的？",
  "use_rag": false,
  "provider": "openai",
  "rag_k": 4
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | 会话 UUID |
| `message` | string | 是 | 用户消息内容 |
| `use_rag` | boolean | 否 | 是否使用 RAG 模式（默认 `false`） |
| `provider` | string | 否 | LLM 提供商标识（默认：从 YAML 配置读取） |
| `rag_k` | integer | 否 | RAG 检索的上下文文档数量（默认 `4`） |

**响应：**

```json
{
  "session_id": "abc-123",
  "response": "这是一个基于 FastAPI 和 LangChain 构建的聊天机器人服务..."
}
```

#### `POST /api/chat/stream`

参数同上。返回 SSE 流：

```
data: {"chunk": "这是"}

data: {"chunk": "一个"}

data: {"chunk": "基于"}

data: [DONE]
```

### 会话接口

#### `POST /api/session`

创建新会话。

**请求：**

```json
{ "title": "项目讨论" }
```

**响应：**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "项目讨论",
  "message_count": 0
}
```

#### `GET /api/session`

列出所有会话（按时间倒序）。

**响应：**

```json
[
  { "id": "...", "title": "项目讨论", "message_count": 12 },
  { "id": "...", "title": "新对话", "message_count": 3 }
]
```

#### `GET /api/session/{session_id}/history`

获取会话的历史消息。

**响应：**

```json
[
  { "role": "user", "content": "你好", "sequence": 0 },
  { "role": "assistant", "content": "你好！有什么可以帮你的？", "sequence": 1 }
]
```

### 知识库接口

#### `POST /api/knowledge/base`

创建知识库。

**请求：**

```json
{ "name": "产品文档", "description": "产品相关文档" }
```

#### `GET /api/knowledge/base`

列出所有知识库。

#### `POST /api/knowledge/upload`

上传文档到知识库并自动索引。

**表单数据：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | file | 文档文件（PDF、DOCX、TXT、MD） |
| `kb_id` | string | 目标知识库 ID |

**响应：**

```json
{
  "id": "doc-uuid",
  "file_name": "guide.pdf",
  "status": "indexed"
}
```

---

## 配置说明

### LLM 提供商配置 (`config/llm_config.yaml`)

每个提供商条目支持以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `provider` | string | 提供商标识: `openai`、`claude`、`ernie`、`qwen` |
| `api_key` | string | API 密钥（如设置环境变量则会被覆盖） |
| `api_base` | string | 自定义 API 端点 URL |
| `model_name` | string | 模型标识（如 `gpt-4o`） |
| `temperature` | float | 采样温度 (0.0 - 1.0) |
| `max_tokens` | integer | 最大响应 token 数 |

**提供商实现细节：**

- **OpenAI**: 使用 `langchain_openai.ChatOpenAI`，支持可选 `base_url` 覆盖
- **Claude**: 使用 `langchain_anthropic.ChatAnthropic`
- **文心 / 千问**: 均使用 `ChatOpenAI` + 自定义 `base_url`，因为它们提供 OpenAI 兼容 API

### Embedding 配置

通过 `.env` 环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_PROVIDER` | `openai` | `openai` 或 `local`（Ollama 兼容） |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 模型标识 |
| `EMBEDDING_API_KEY` | - | API 密钥（回退到 `OPENAI_API_KEY`） |
| `EMBEDDING_API_BASE` | - | 自定义端点（本地模型: `http://localhost:11434/v1`） |
| `EMBEDDING_DIMENSIONS` | `1536` | 向量维度 |

### 记忆配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEMORY_WINDOW_SIZE` | `10` | 短期滑动窗口保留的消息数量 |

---

## 环境变量

### 开发必需

| 变量 | 说明 | 示例 |
|------|------|------|
| `DB_URL` | MySQL 连接字符串 | `mysql+pymysql://root:root@localhost:3306/chatbot` |
| `OPENAI_API_KEY` | OpenAI API 密钥 (LLM + Embedding) | `sk-xxx` |

### 可选

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEBUG` | 开启调试模式 | `false` |
| `DB_ECHO` | 打印 SQL 日志 | `false` |
| `MILVUS_HOST` | Milvus 服务主机名 | `localhost` |
| `MILVUS_PORT` | Milvus 服务端口 | `19530` |
| `MILVUS_COLLECTION` | Milvus 集合名称 | `knowledge_base` |
| `CLAUDE_API_KEY` | Anthropic API 密钥 (覆盖 YAML) | - |
| `CLAUDE_API_BASE` | 自定义 Claude API 端点 | - |
| `ERNIE_API_KEY` | 百度文心 API 密钥 | - |
| `ERNIE_API_BASE` | 自定义文心 API 端点 | - |
| `QWEN_API_KEY` | 阿里千问 API 密钥 | - |
| `QWEN_API_BASE` | 自定义千问 API 端点 | - |
| `EMBEDDING_PROVIDER` | Embedding 提供商 | `openai` |
| `EMBEDDING_MODEL` | Embedding 模型名称 | `text-embedding-3-small` |
| `EMBEDDING_API_KEY` | Embedding API 密钥 | (回退到 OPENAI_API_KEY) |
| `EMBEDDING_API_BASE` | Embedding API 端点 | - |
| `EMBEDDING_DIMENSIONS` | 向量维度 | `1536` |
| `MEMORY_WINDOW_SIZE` | 滑动窗口大小 | `10` |
| `LLM_CONFIG_PATH` | LLM 配置 YAML 路径 | `./config/llm_config.yaml` |

---

## 常用命令

| 命令 | 说明 |
|------|------|
| `uvicorn main:app --reload` | 启动开发服务器（自动重载） |
| `uvicorn main:app --host 0.0.0.0 --port 8000` | 启动生产服务器 |
| `python -m pytest` | 运行所有测试 |
| `python -m pytest -v` | 运行测试并显示详细输出 |
| `python -m pytest tests/test_document_loader.py -v` | 运行指定测试文件 |
| `python -m pytest -k "chunk"` | 运行包含关键字的测试 |
| `docker compose up -d` | 启动所有基础设施服务 |
| `docker compose down` | 停止并移除所有容器 |
| `docker compose up --build` | 重新构建并启动所有服务 |
| `docker build -t chatbot .` | 构建 Docker 镜像 |

---

## 测试

### 运行测试

项目使用 pytest。所有测试位于 `tests/` 目录。

```bash
# 运行所有测试
python -m pytest

# 运行测试并显示详细输出
python -m pytest -v

# 运行指定测试文件
python -m pytest tests/test_document_loader.py -v

# 运行包含关键字的测试
python -m pytest -k "memory" -v

# 运行测试覆盖率报告 (需安装 pytest-cov)
python -m pytest --cov=services --cov=chains --cov-report=term-missing
```

### 测试套件

| 测试文件 | 覆盖范围 | 数量 |
|---------|---------|------|
| `tests/test_memory_service.py` | MemoryService: 滑动窗口、添加消息、加载历史、清除会话 | 14 个测试 |
| `tests/test_document_loader.py` | 文档加载、文本分块、格式验证 | 6 个测试 |
| `tests/test_llm_factory.py` | 提供商注册、未知提供商错误 | 2 个测试 |

### 测试结构

```
tests/
├── __init__.py
├── conftest.py              # Pytest 测试夹具 (LLM 提供商配置)
├── test_llm_factory.py      # 工厂模式测试
├── test_memory_service.py   # 双层记忆测试 (Mock MySQL)
└── test_document_loader.py  # 文档解析与分块测试
```

所有测试中的数据库交互均通过 `unittest.mock.patch` 进行 Mock — 不需要真实的数据库连接。

---

## 部署

### Docker Compose（推荐）

全栈部署，包含所有基础设施：

```bash
# 启动全部服务 (MySQL, Milvus, etcd, MinIO, 应用)
docker compose up -d

# 查看日志
docker compose logs -f app

# 查看 Milvus 日志 (向量搜索失败时排查)
docker compose logs -f milvus-standalone

# 停止全部服务
docker compose down
```

启动 5 个容器：
- **app** — FastAPI 应用，端口 8000
- **mysql** — MySQL 8.0，端口 3306，持久化数据卷
- **etcd** — etcd v3.5（Milvus 依赖）
- **minio** — MinIO 对象存储（Milvus 依赖）
- **milvus-standalone** — Milvus 2.4 standalone，端口 19530

### 手动构建 Docker 镜像

```bash
# 构建镜像
docker build -t chatbot-service .

# 运行 (需要 MySQL 和 Milvus 已就绪)
docker run -p 8000:8000 \
  -e DB_URL=mysql+pymysql://root:root@host.docker.internal:3306/chatbot \
  -e OPENAI_API_KEY=sk-xxx \
  -e MILVUS_HOST=host.docker.internal \
  chatbot-service
```

### 生产环境注意事项

1. **反向代理**：在 Uvicorn 前放置 Nginx 或负载均衡器
2. **多 Worker**：使用 Gunicorn + Uvicorn worker：
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UVicornWorker --bind 0.0.0.0:8000
   ```
3. **数据库**：使用托管 MySQL 服务并配置连接池
4. **Milvus**：生产环境部署 Milvus 集群（非 standalone 模式）
5. **密钥管理**：使用密钥管理服务替代 `.env` 文件
6. **监控**：添加日志采集、指标监控和健康检查端点
7. **HTTPS**：在反向代理处终止 TLS

---

## 前端

聊天界面位于 `frontend/index.html`。单文件 SPA，无需构建步骤。

**功能：**
- 会话管理（创建、切换、列出对话）
- 实时 SSE 流式响应
- RAG 模式开关
- LLM 提供商选择器
- 知识库面板与文档上传
- 自适应高度文本输入框（Enter 发送，Shift+Enter 换行）
- 暗色主题 + 琥珀色强调色

**设计选择：**
- **排版**：Fraunces（衬线体）用于 AI 回复、Outfit（无衬线体）用于 UI 元素、IBM Plex Mono 用于元数据
- **色彩**：深炭色画布 (#0a0a0b) + 琥珀色强调 (#d4a24e)
- **布局**：三栏网格 — 会话侧边栏 | 对话区 | 可展开知识库面板

### 提供前端服务

方式一 — Python HTTP 服务：

```bash
cd frontend
python3 -m http.server 3000
```

在浏览器中打开 `http://localhost:3000`。

方式二 — 通过 FastAPI 提供服务：

在 `main.py` 中添加：

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/chat")
async def chat_ui():
    return FileResponse("frontend/index.html")
```

前端默认连接同源 API。如果后端在不同主机，编辑 `frontend/index.html` 中的 `API` 常量：

```javascript
const API = 'http://localhost:8000';
```

---

## 故障排查

### MySQL 连接被拒绝

**错误：** `(pymysql.err.OperationalError) (2003, "Can't connect to MySQL server")`

**解决方案：**
1. 验证 MySQL 是否运行：`docker compose ps mysql`
2. 检查连接字符串格式：`mysql+pymysql://USER:PASSWORD@HOST:PORT/DATABASE`
3. 确保数据库已创建：`docker compose exec mysql mysql -uroot -proot -e "CREATE DATABASE IF NOT EXISTS chatbot;"`

### Milvus 连接失败

**错误：** `MilvusException: connection failed`

**解决方案：**
1. 检查 Milvus 状态：`docker compose ps milvus-standalone`
2. 等待健康检查通过（首次启动可能需要 60-90 秒）
3. 验证 etcd 和 MinIO 正在运行（Milvus 依赖它们）：`docker compose ps etcd minio`
4. 查看 Milvus 日志：`docker compose logs milvus-standalone`

### 模块未找到错误

**错误：** `ModuleNotFoundError: No module named 'langchain_xxx'`

**解决方案：**
确保虚拟环境已激活且所有依赖已安装：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### LLM 提供商未找到

**错误：** `ValueError: Provider 'xxx' not found in configuration`

**解决方案：**
1. 确认提供商已列在 `config/llm_config.yaml` 中
2. 检查 `default_provider` 设置为有效的提供商
3. 确保提供商已在 `services/llm_factory.py` 中注册

### Embedding 服务问题

**错误：** 文档索引或 Milvus 集合创建时出错

**解决方案：**
1. 验证 `.env` 中 `OPENAI_API_KEY` 设置正确
2. 直接测试 Embedding：
   ```python
   from services.embedding_service import get_embeddings
   emb = get_embeddings()
   result = emb.embed_query("test")
   print(len(result))  # 应与 EMBEDDING_DIMENSIONS 一致
   ```

### API 密钥错误

**错误：** LLM 提供商 API 返回 401

**解决方案：**
环境变量（`OPENAI_API_KEY` 等）会覆盖 `llm_config.yaml` 中的值。确保 `.env` 文件中的密钥正确：

```bash
# 检查环境变量是否已设置
echo $OPENAI_API_KEY
```

如果希望使用 YAML 文件中的值，请清空或删除对应的环境变量。

### Docker 镜像构建失败

**错误：** Docker 内 `pip install` 失败

**解决方案：**
部分包（如 `unstructured`）有系统依赖。如需可更新 `Dockerfile`：

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
```

### 前端跨域 (CORS) 问题

如果前端和 API 不在同一域名下，在 `main.py` 中添加 CORS 中间件：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 贡献指南

1. 创建功能分支
2. 为新功能编写测试
3. 实现代码
4. 运行测试套件：`python -m pytest`
5. 提交并创建 Pull Request

遵循代码库中已有的编码规范：
- 所有 SQL 逻辑放在 `models/` 层（SQLAlchemy ORM）
- 适当使用单例模式管理服务
- 配置通过 Pydantic Settings 管理，YAML + ENV 合并
- LLM 提供商使用工厂注册表模式

---

## 许可证

私有项目。
