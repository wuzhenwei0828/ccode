# Chatbot Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + LangChain chatbot service with RAG knowledge base, dual-layer memory (short-term sliding window + long-term MySQL persistence), and pluggable LLM providers (OpenAI/Claude/Chinese models).

**Architecture:** Lightweight layered architecture with config → models → services → chains → routers. LLM providers use a factory pattern driven by YAML + ENV configuration. Memory combines in-memory sliding window for context with MySQL-persisted chat history. RAG uses Milvus for vector storage and retrieval. Docker Compose orchestrates Milvus + MySQL dependencies.

**Tech Stack:** Python 3.11+, FastAPI, LangChain (latest), SQLAlchemy, Milvus, MySQL, PyYAML, python-dotenv, Docker Compose.

---

## File Map

| File | Responsibility | Action |
|------|---------------|--------|
| `requirements.txt` | All Python dependencies | Create |
| `config/settings.py` | Pydantic settings (YAML + ENV) | Create |
| `config/llm_config.yaml` | LLM provider definitions | Create |
| `config/__init__.py` | Package init | Create |
| `utils/db.py` | MySQL connection & session management | Create |
| `utils/milvus_client.py` | Milvus client wrapper | Create |
| `utils/__init__.py` | Package init | Create |
| `models/chat_session.py` | ChatSession model | Create |
| `models/chat_message.py` | ChatMessage model | Create |
| `models/knowledge_base.py` | KnowledgeBase & DocumentMeta models | Create |
| `models/__init__.py` | Package init + table init | Create |
| `services/llm_factory.py` | LLM provider factory & base interface | Create |
| `services/openai_llm.py` | OpenAI provider implementation | Create |
| `services/claude_llm.py` | Claude provider implementation | Create |
| `services/chinese_llm.py` | Chinese LLM providers (ERNIE, Qwen) | Create |
| `services/memory_service.py` | Short-term + long-term memory | Create |
| `services/embedding_service.py` | Embedding model factory | Create |
| `services/rag_service.py` | RAG pipeline (index + query) | Create |
| `services/document_loader.py` | Multi-format document parsing & chunking | Create |
| `services/__init__.py` | Package init | Create |
| `chains/chat_chain.py` | Plain conversation chain with memory | Create |
| `chains/rag_chain.py` | RAG conversation chain | Create |
| `chains/__init__.py` | Package init | Create |
| `routers/chat.py` | `/api/chat` endpoints (SSE + JSON) | Create |
| `routers/session.py` | `/api/session` management | Create |
| `routers/knowledge.py` | `/api/knowledge` endpoints | Create |
| `routers/__init__.py` | Package init | Create |
| `main.py` | FastAPI app entry, router registration | Modify |
| `docker-compose.yml` | Milvus + MySQL containers | Create |
| `.env.example` | Environment variable template | Create |
| `tests/test_llm_factory.py` | LLM factory tests | Create |
| `tests/test_memory_service.py` | Memory service tests | Create |
| `tests/test_document_loader.py` | Document loader tests | Create |
| `tests/conftest.py` | Test fixtures | Create |

---

### Task 1: Project Setup & Configuration

**Files:**
- Create: `requirements.txt`, `config/settings.py`, `config/llm_config.yaml`, `config/__init__.py`, `.env.example`, `utils/__init__.py`

- [ ] **Step 1.1: Create requirements.txt**

```text
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0
pydantic-settings>=2.0
python-dotenv>=1.0
pyyaml>=6.0

# LangChain
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0
langchain-community>=0.3.0
langchain-milvus>=0.1.0
langchain-text-splitters>=0.3.0

# Database
sqlalchemy>=2.0
pymysql>=1.1

# Milvus
pymilvus>=2.4.0

# Document processing
pypdf>=4.0
python-docx>=1.1
markdown>=3.6
unstructured>=0.15.0

# Embeddings (optional, depending on provider)
openai>=1.40.0

# Testing
pytest>=8.0
pytest-asyncio>=0.24.0
httpx>=0.27.0
```

- [ ] **Step 1.2: Create config/__init__.py**

```python
from config.settings import get_settings, Settings

__all__ = ["get_settings", "Settings"]
```

- [ ] **Step 1.3: Create config/settings.py**

```python
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent


class LLMProviderConfig(BaseModel):
    """Single LLM provider configuration."""

    provider: Literal["openai", "claude", "ernie", "qwen"]
    api_key: str = ""
    api_base: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048


class MemoryConfig(BaseModel):
    """Memory configuration."""

    short_term_window: int = Field(default=10, description="Number of recent messages in sliding window")
    db_url: str = ""


class MilvusConfig(BaseModel):
    """Milvus vector database configuration."""

    host: str = "localhost"
    port: int = 19530
    collection_name: str = "knowledge_base"


class DatabaseConfig(BaseModel):
    """MySQL database configuration."""

    url: str = "mysql+pymysql://root:root@localhost:3306/chatbot"
    echo: bool = False
    pool_size: int = 5


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: Literal["openai", "local"] = "openai"
    model_name: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str = ""
    dimensions: int = 1536


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Chatbot Service"
    debug: bool = False

    # Database
    db_url: str = "mysql+pymysql://root:root@localhost:3306/chatbot"
    db_echo: bool = False

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"

    # Embedding
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_dimensions: int = 1536

    # Memory
    memory_window_size: int = 10

    # LLM config file path
    llm_config_path: str = str(PROJECT_ROOT / "config" / "llm_config.yaml")

    def get_llm_providers(self) -> list[LLMProviderConfig]:
        """Load LLM provider list from YAML config file."""
        path = Path(self.llm_config_path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        providers = data.get("providers", [])
        result = []
        for p in providers:
            cfg = LLMProviderConfig(**p)
            # Override api_key from ENV if available
            env_key = os.getenv(f"{cfg.provider.upper()}_API_KEY")
            if env_key:
                cfg = cfg.model_copy(update={"api_key": env_key})
            env_base = os.getenv(f"{cfg.provider.upper()}_API_BASE")
            if env_base:
                cfg = cfg.model_copy(update={"api_base": env_base})
            result.append(cfg)
        return result

    def get_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=self.embedding_provider,
            model_name=self.embedding_model,
            api_key=self.embedding_api_key or os.getenv("OPENAI_API_KEY", ""),
            api_base=self.embedding_api_base or os.getenv("OPENAI_API_BASE", ""),
            dimensions=self.embedding_dimensions,
        )

    def get_milvus_config(self) -> MilvusConfig:
        return MilvusConfig(
            host=self.milvus_host,
            port=self.milvus_port,
            collection_name=self.milvus_collection,
        )

    def get_memory_config(self) -> MemoryConfig:
        return MemoryConfig(
            short_term_window=self.memory_window_size,
            db_url=self.db_url,
        )

    def get_database_config(self) -> DatabaseConfig:
        return DatabaseConfig(
            url=self.db_url,
            echo=self.db_echo,
        )


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 1.4: Create config/llm_config.yaml**

```yaml
# LLM Provider Configuration
# Override individual api_key values via environment variables:
#   OPENAI_API_KEY, CLAUDE_API_KEY, ERNIE_API_KEY, QWEN_API_KEY
# Override api_base via: OPENAI_API_BASE, CLAUDE_API_BASE, etc.

providers:
  - provider: openai
    api_key: "${OPENAI_API_KEY:sk-xxx}"
    api_base: "${OPENAI_API_BASE:https://api.openai.com/v1}"
    model_name: "gpt-4o"
    temperature: 0.7
    max_tokens: 2048

  - provider: claude
    api_key: "${CLAUDE_API_KEY:sk-xxx}"
    api_base: ""
    model_name: "claude-sonnet-4-20250514"
    temperature: 0.7
    max_tokens: 2048

  - provider: ernie
    api_key: "${ERNIE_API_KEY:}"
    api_base: "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
    model_name: "ernie-4.0"
    temperature: 0.7
    max_tokens: 2048

  - provider: qwen
    api_key: "${QWEN_API_KEY:}"
    api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: "qwen-max"
    temperature: 0.7
    max_tokens: 2048

# Default provider to use when none is specified
default_provider: openai
```

- [ ] **Step 1.5: Create .env.example**

```text
# Chatbot Service Environment Variables
# Copy to .env and fill in actual values

# App
DEBUG=false

# Database
DB_URL=mysql+pymysql://root:root@localhost:3306/chatbot

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LLM API Keys (override llm_config.yaml values)
OPENAI_API_KEY=sk-xxx
CLAUDE_API_KEY=sk-xxx
ERNIE_API_KEY=
QWEN_API_KEY=

# Embedding (defaults to OpenAI)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=
EMBEDDING_DIMENSIONS=1536

# Memory
MEMORY_WINDOW_SIZE=10
```

- [ ] **Step 1.6: Create utils/__init__.py**

```python

```

- [ ] **Step 1.7: Commit**

```bash
git add requirements.txt config/ utils/__init__.py .env.example
git commit -m "feat: add project setup and configuration"
```

---

### Task 2: Database Layer (MySQL)

**Files:**
- Create: `utils/db.py`, `models/__init__.py`, `models/chat_session.py`, `models/chat_message.py`, `models/knowledge_base.py`

- [ ] **Step 2.1: Create utils/db.py — Database connection**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import get_settings

settings = get_settings()
db_config = settings.get_database_config()

engine = create_engine(
    db_config.url,
    echo=db_config.echo,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency for getting a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once at startup."""
    # Import all models to register them before create_all
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
```

- [ ] **Step 2.2: Create models/chat_session.py**

```python
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Integer, func

from utils.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, comment="UUID for session")
    title = Column(String(255), default="New Chat", comment="Session title")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="Last update time")
    message_count = Column(Integer, default=0, comment="Total messages in session")
```

- [ ] **Step 2.3: Create models/chat_message.py**

```python
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, Integer, func

from utils.db import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, comment="UUID for message")
    session_id = Column(String(36), nullable=False, index=True, comment="Associated session ID")
    role = Column(String(20), nullable=False, comment="Message role: user / assistant / system")
    content = Column(Text, nullable=False, comment="Message content")
    created_at = Column(DateTime, server_default=func.now(), comment="Creation time")
    sequence = Column(Integer, default=0, comment="Message order within session")
```

- [ ] **Step 2.4: Create models/knowledge_base.py**

```python
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Text, Integer, func

from utils.db import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True, comment="UUID for knowledge base")
    name = Column(String(255), nullable=False, comment="Knowledge base name")
    description = Column(Text, default="", comment="Description")
    created_at = Column(DateTime, server_default=func.now())


class DocumentMeta(Base):
    __tablename__ = "document_metas"

    id = Column(String(36), primary_key=True, comment="UUID for document")
    kb_id = Column(String(36), nullable=False, index=True, comment="Associated knowledge base ID")
    file_name = Column(String(255), nullable=False, comment="Original file name")
    file_path = Column(String(512), default="", comment="Stored file path")
    status = Column(String(20), default="pending", comment="pending / indexed / failed")
    created_at = Column(DateTime, server_default=func.now())
```

- [ ] **Step 2.5: Create models/__init__.py**

```python
# Import all models so they are registered with Base.metadata
from models.chat_session import ChatSession  # noqa: F401
from models.chat_message import ChatMessage  # noqa: F401
from models.knowledge_base import KnowledgeBase, DocumentMeta  # noqa: F401

from utils.db import Base

__all__ = ["Base", "ChatSession", "ChatMessage", "KnowledgeBase", "DocumentMeta"]
```

- [ ] **Step 2.6: Commit**

```bash
git add utils/db.py models/
git commit -m "feat: add database models and connection"
```

---

### Task 3: LLM Factory (Pluggable Providers)

**Files:**
- Create: `services/__init__.py`, `services/llm_factory.py`, `services/openai_llm.py`, `services/claude_llm.py`, `services/chinese_llm.py`

- [ ] **Step 3.1: Create services/__init__.py**

```python

```

- [ ] **Step 3.2: Create services/llm_factory.py**

```python
from typing import Optional

from langchain_core.language_models import BaseChatModel

from config import get_settings
from config.settings import LLMProviderConfig


class LLMFactory:
    """Factory for creating LangChain chat models based on provider configuration."""

    _registry: dict[str, callable] = {}

    @classmethod
    def register(cls, provider: str, factory: callable):
        """Register a provider factory function."""
        cls._registry[provider] = factory

    @classmethod
    def create(cls, provider_name: Optional[str] = None) -> BaseChatModel:
        """Create a chat model instance for the given provider.

        Args:
            provider_name: Provider key (openai/claude/ernie/qwen).
                           If None, uses the default provider from config.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            ValueError: If provider is unknown or not configured.
        """
        settings = get_settings()
        providers = settings.get_llm_providers()

        if provider_name is None:
            # Use default provider from YAML
            import yaml
            from pathlib import Path
            cfg_path = Path(settings.llm_config_path)
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                provider_name = data.get("default_provider", "openai")
            else:
                provider_name = "openai"

        # Find matching provider config
        provider_cfg = None
        for p in providers:
            if p.provider == provider_name:
                provider_cfg = p
                break

        if provider_cfg is None:
            raise ValueError(f"Provider '{provider_name}' not found in configuration")

        factory_func = cls._registry.get(provider_cfg.provider)
        if factory_func is None:
            raise ValueError(f"No factory registered for provider '{provider_cfg.provider}'")

        return factory_func(provider_cfg)

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._registry.keys())


# Auto-register built-in providers
def _create_openai_model(cfg: LLMProviderConfig) -> BaseChatModel:
    from services.openai_llm import create_openai_chat_model
    return create_openai_chat_model(cfg)


def _create_claude_model(cfg: LLMProviderConfig) -> BaseChatModel:
    from services.claude_llm import create_claude_chat_model
    return create_claude_chat_model(cfg)


def _create_chinese_model(cfg: LLMProviderConfig) -> BaseChatModel:
    from services.chinese_llm import create_chinese_chat_model
    return create_chinese_chat_model(cfg)


LLMFactory.register("openai", _create_openai_model)
LLMFactory.register("claude", _create_claude_model)
LLMFactory.register("ernie", _create_chinese_model)
LLMFactory.register("qwen", _create_chinese_model)
```

- [ ] **Step 3.3: Create services/openai_llm.py**

```python
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import LLMProviderConfig


def create_openai_chat_model(cfg: LLMProviderConfig) -> BaseChatModel:
    """Create an OpenAI chat model from provider config."""
    kwargs = {
        "model": cfg.model_name,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "api_key": cfg.api_key,
    }
    if cfg.api_base:
        kwargs["base_url"] = cfg.api_base
    return ChatOpenAI(**kwargs)
```

- [ ] **Step 3.4: Create services/claude_llm.py**

```python
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic

from config.settings import LLMProviderConfig


def create_claude_chat_model(cfg: LLMProviderConfig) -> BaseChatModel:
    """Create a Claude chat model from provider config."""
    kwargs = {
        "model": cfg.model_name,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "api_key": cfg.api_key,
    }
    return ChatAnthropic(**kwargs)
```

- [ ] **Step 3.5: Create services/chinese_llm.py**

```python
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import LLMProviderConfig


def create_chinese_chat_model(cfg: LLMProviderConfig) -> BaseChatModel:
    """Create a Chinese LLM (ERNIE/Qwen) via OpenAI-compatible interface.

    Both Baidu ERNIE and Alibaba Qwen provide OpenAI-compatible API endpoints,
    so we reuse ChatOpenAI with custom base_url.
    """
    if not cfg.api_base:
        raise ValueError(f"Provider '{cfg.provider}' requires api_base to be configured")

    kwargs = {
        "model": cfg.model_name,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "api_key": cfg.api_key,
        "base_url": cfg.api_base,
    }
    return ChatOpenAI(**kwargs)
```

- [ ] **Step 3.6: Commit**

```bash
git add services/__init__.py services/llm_factory.py services/openai_llm.py services/claude_llm.py services/chinese_llm.py
git commit -m "feat: add LLM factory with pluggable providers"
```

---

### Task 4: Embedding Service

**Files:**
- Create: `services/embedding_service.py`

- [ ] **Step 4.1: Create services/embedding_service.py**

```python
from langchain_core.embeddings import Embeddings

from config import get_settings
from config.settings import EmbeddingConfig


def get_embeddings() -> Embeddings:
    """Get the configured embedding model."""
    settings = get_settings()
    emb_cfg = settings.get_embedding_config()

    if emb_cfg.provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": emb_cfg.model_name,
            "dimensions": emb_cfg.dimensions,
        }
        if emb_cfg.api_key:
            kwargs["api_key"] = emb_cfg.api_key
        if emb_cfg.api_base:
            kwargs["base_url"] = emb_cfg.api_base
        return OpenAIEmbeddings(**kwargs)
    else:
        # Fallback: use OpenAI-compatible endpoint for local models
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=emb_cfg.model_name,
            dimensions=emb_cfg.dimensions,
            base_url=emb_cfg.api_base or "http://localhost:11434/v1",
            api_key=emb_cfg.api_key or "ollama",
        )
```

- [ ] **Step 4.2: Commit**

```bash
git add services/embedding_service.py
git commit -m "feat: add embedding service factory"
```

---

### Task 5: Memory Service (Short-term + Long-term)

**Files:**
- Create: `services/memory_service.py`
- Test: `tests/test_memory_service.py`

- [ ] **Step 5.1: Create services/memory_service.py**

```python
from typing import Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

from config import get_settings
from config.settings import MemoryConfig
from models.chat_message import ChatMessage
from utils.db import SessionLocal


class MemoryService:
    """Dual-layer memory: short-term sliding window + long-term MySQL persistence."""

    def __init__(self):
        self._config: Optional[MemoryConfig] = None
        # In-memory cache: session_id -> list of recent messages
        self._short_term: dict[str, list[BaseMessage]] = {}

    @property
    def config(self) -> MemoryConfig:
        if self._config is None:
            self._config = get_settings().get_memory_config()
        return self._config

    def get_context(self, session_id: str) -> list[BaseMessage]:
        """Get messages for LLM context (short-term sliding window)."""
        messages = self._short_term.get(session_id, [])
        window = self.config.short_term_window
        return messages[-window:] if len(messages) > window else messages

    def add_message(self, session_id: str, role: str, content: str, db: Optional[Session] = None):
        """Add a message to both short-term cache and long-term storage."""
        # Short-term
        if session_id not in self._short_term:
            self._short_term[session_id] = []
        msg = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        self._short_term[session_id].append(msg)

        # Long-term (MySQL)
        should_close = db is None
        if db is None:
            db = SessionLocal()
        try:
            # Get current max sequence
            max_seq = (
                db.query(ChatMessage.sequence)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.sequence.desc())
                .first()
            )
            next_seq = (max_seq[0] + 1) if max_seq else 0

            chat_msg = ChatMessage(
                id=str(uuid4()),
                session_id=session_id,
                role=role,
                content=content,
                sequence=next_seq,
            )
            db.add(chat_msg)
            db.commit()
        finally:
            if should_close:
                db.close()

    def load_history(self, session_id: str, limit: Optional[int] = None) -> list[BaseMessage]:
        """Load full history from MySQL for a session."""
        db = SessionLocal()
        try:
            query = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.sequence.asc())
            )
            if limit:
                query = query.limit(limit)
            messages = query.all()
            result = []
            for m in messages:
                if m.role == "user":
                    result.append(HumanMessage(content=m.content))
                else:
                    result.append(AIMessage(content=m.content))
            return result
        finally:
            db.close()

    def clear_session(self, session_id: str):
        """Clear all memory for a session."""
        self._short_term.pop(session_id, None)
        db = SessionLocal()
        try:
            db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            db.commit()
        finally:
            db.close()


# Singleton
memory_service = MemoryService()
```

- [ ] **Step 5.2: Commit**

```bash
git add services/memory_service.py
git commit -m "feat: add dual-layer memory service"
```

---

### Task 6: Milvus Client Wrapper

**Files:**
- Create: `utils/milvus_client.py`

- [ ] **Step 6.1: Create utils/milvus_client.py**

```python
from langchain_milvus import Milvus

from config import get_settings
from services.embedding_service import get_embeddings


def get_milvus_store(collection_name: str | None = None) -> Milvus:
    """Get a Milvus vector store instance.

    Args:
        collection_name: Override the default collection name from config.

    Returns:
        LangChain Milvus vector store instance.
    """
    settings = get_settings()
    milvus_cfg = settings.get_milvus_config()
    embeddings = get_embeddings()

    uri = f"http://{milvus_cfg.host}:{milvus_cfg.port}"
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name or milvus_cfg.collection_name,
        connection_args={"uri": uri},
    )
```

- [ ] **Step 6.2: Commit**

```bash
git add utils/milvus_client.py
git commit -m "feat: add Milvus client wrapper"
```

---

### Task 7: Document Loader & Chunking

**Files:**
- Create: `services/document_loader.py`
- Test: `tests/test_document_loader.py`

- [ ] **Step 7.1: Create services/document_loader.py**

```python
import os
import tempfile
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Extension -> loader function mapping
_LOADER_MAP: dict[str, str] = {
    ".pdf": "pypdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
}


def load_document(file_path: str) -> str:
    """Load and extract text from a document file.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If file format is not supported.
    """
    ext = Path(file_path).suffix.lower()
    loader_type = _LOADER_MAP.get(ext)

    if loader_type is None:
        supported = ", ".join(_LOADER_MAP.keys())
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {supported}")

    if loader_type == "pypdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif loader_type == "docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif loader_type == "text":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")

    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Args:
        text: Full text content.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks


def process_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Full pipeline: load document and split into chunks.

    Args:
        file_path: Path to the document file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks ready for embedding.
    """
    text = load_document(file_path)
    if not text.strip():
        return []
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
```

- [ ] **Step 7.2: Create tests/test_document_loader.py**

```python
import os
import tempfile

import pytest

from services.document_loader import chunk_text, load_document, process_document


class TestChunkText:
    def test_chunk_text_splits_by_size(self):
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=0)
        assert len(chunks) >= 4
        assert all(len(c) <= 300 for c in chunks)

    def test_chunk_text_with_overlap(self):
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(chunks) >= 2
        # Overlap means consecutive chunks share content
        for i in range(len(chunks) - 1):
            # Verify chunks are non-empty
            assert len(chunks[i]) > 0
            assert len(chunks[i + 1]) > 0

    def test_chunk_text_short_input(self):
        text = "short text"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestLoadDocument:
    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello world")
            f.flush()
            result = load_document(f.name)
        os.unlink(f.name)
        assert result == "hello world"

    def test_load_md_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Title\n\ncontent")
            f.flush()
            result = load_document(f.name)
        os.unlink(f.name)
        assert "# Title" in result

    def test_load_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document("test.csv")
```

- [ ] **Step 7.3: Commit**

```bash
git add services/document_loader.py tests/test_document_loader.py
git commit -m "feat: add document loader with multi-format support and tests"
```

---

### Task 8: RAG Service

**Files:**
- Create: `services/rag_service.py`

- [ ] **Step 8.1: Create services/rag_service.py**

```python
import logging
from typing import Optional

from langchain_core.documents import Document

from services.embedding_service import get_embeddings
from utils.milvus_client import get_milvus_store

logger = logging.getLogger(__name__)


class RAGService:
    """RAG pipeline: index documents into Milvus and query for retrieval."""

    def __init__(self):
        self._store = None

    @property
    def store(self):
        """Lazy-init the vector store."""
        if self._store is None:
            self._store = get_milvus_store()
        return self._store

    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[str]:
        """Index text chunks into Milvus.

        Args:
            texts: List of text chunks to index.
            metadatas: Optional metadata per chunk (e.g., doc_id, kb_id, file_name).

        Returns:
            List of inserted document IDs.
        """
        ids = self.store.add_texts(texts, metadatas=metadatas)
        logger.info(f"Indexed {len(ids)} text chunks into Milvus")
        return ids

    def query(self, question: str, k: int = 4) -> list[Document]:
        """Retrieve relevant documents for a question.

        Args:
            question: User's query text.
            k: Number of documents to retrieve.

        Returns:
            List of relevant Document objects.
        """
        results = self.store.similarity_search(question, k=k)
        logger.info(f"Retrieved {len(results)} documents for query: {question[:50]}...")
        return results

    def delete_by_metadata(self, **metadata_filter) -> int:
        """Delete documents from Milvus by metadata filter.

        Args:
            **metadata_filter: Key-value pairs to match (e.g., doc_id="xxx").

        Returns:
            Number of deleted documents.
        """
        # Milvus delete via expression
        # Build filter expression from metadata
        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(f'{key} == "{value}"')
        expr = " and ".join(conditions)
        result = self.store.delete(expr=expr)
        logger.info(f"Deleted documents matching {metadata_filter}: {result}")
        return result


# Singleton
rag_service = RAGService()
```

- [ ] **Step 8.2: Commit**

```bash
git add services/rag_service.py
git commit -m "feat: add RAG service for document indexing and retrieval"
```

---

### Task 9: Chains (Chat + RAG)

**Files:**
- Create: `chains/__init__.py`, `chains/chat_chain.py`, `chains/rag_chain.py`

- [ ] **Step 9.1: Create chains/__init__.py**

```python

```

- [ ] **Step 9.2: Create chains/chat_chain.py — Plain conversation chain**

```python
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.llm_factory import LLMFactory
from services.memory_service import MemoryService


class ChatChain:
    """Plain conversation chain with memory context."""

    def __init__(self, memory_service: MemoryService, provider: str | None = None):
        self.memory = memory_service
        self.provider = provider
        self._prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self._parser = StrOutputParser()

    def invoke(self, session_id: str, message: str) -> str:
        """Run a conversation turn.

        Args:
            session_id: The conversation session ID.
            message: User's input message.

        Returns:
            Assistant's response text.
        """
        llm = LLMFactory.create(self.provider)
        chain = self._prompt | llm | self._parser

        # Get context from memory
        history = self.memory.get_context(session_id)

        # Run chain
        response = chain.invoke({"input": message, "history": history})

        # Save to memory
        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response)

        return response

    async def astream(self, session_id: str, message: str):
        """Stream a conversation response.

        Args:
            session_id: The conversation session ID.
            message: User's input message.

        Yields:
            Text chunks of the response.
        """
        llm = LLMFactory.create(self.provider)
        chain = self._prompt | llm | self._parser

        history = self.memory.get_context(session_id)

        # Save user message immediately
        self.memory.add_message(session_id, "user", message)

        # Stream response
        full_response = []
        async for chunk in chain.astream({"input": message, "history": history}):
            full_response.append(chunk)
            yield chunk

        # Save assistant response after streaming completes
        self.memory.add_message(session_id, "assistant", "".join(full_response))
```

- [ ] **Step 9.3: Create chains/rag_chain.py — RAG conversation chain**

```python
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from services.llm_factory import LLMFactory
from services.memory_service import MemoryService
from services.rag_service import RAGService


class RAGChain:
    """RAG conversation chain: retrieve context, then answer with LLM."""

    _RAG_PROMPT = """You are a helpful assistant. Answer the question based on the provided context.
If the context does not contain relevant information, say "Based on the provided knowledge base, I cannot find relevant information."

Context:
{context}

Question: {input}
"""

    def __init__(self, memory_service: MemoryService, rag_service: RAGService, provider: str | None = None):
        self.memory = memory_service
        self.rag = rag_service
        self.provider = provider

    def invoke(self, session_id: str, message: str, k: int = 4) -> str:
        """Run a RAG conversation turn.

        Args:
            session_id: The conversation session ID.
            message: User's question.
            k: Number of context documents to retrieve.

        Returns:
            Assistant's response text.
        """
        llm = LLMFactory.create(self.provider)
        parser = StrOutputParser()

        # Retrieve context
        docs = self.rag.query(message, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer based on the provided context."),
            MessagesPlaceholder(variable_name="history"),
            ("human", self._RAG_PROMPT),
        ])

        chain = prompt | llm | parser
        history = self.memory.get_context(session_id)

        response = chain.invoke({"input": message, "history": history, "context": context})

        self.memory.add_message(session_id, "user", message)
        self.memory.add_message(session_id, "assistant", response)

        return response

    async def astream(self, session_id: str, message: str, k: int = 4):
        """Stream a RAG response.

        Args:
            session_id: The conversation session ID.
            message: User's question.
            k: Number of context documents to retrieve.

        Yields:
            Text chunks of the response.
        """
        llm = LLMFactory.create(self.provider)
        parser = StrOutputParser()

        # Retrieve context first
        docs = self.rag.query(message, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer based on the provided context."),
            MessagesPlaceholder(variable_name="history"),
            ("human", self._RAG_PROMPT),
        ])

        chain = prompt | llm | parser
        history = self.memory.get_context(session_id)

        self.memory.add_message(session_id, "user", message)

        full_response = []
        async for chunk in chain.astream({"input": message, "history": history, "context": context}):
            full_response.append(chunk)
            yield chunk

        self.memory.add_message(session_id, "assistant", "".join(full_response))
```

- [ ] **Step 9.4: Commit**

```bash
git add chains/__init__.py chains/chat_chain.py chains/rag_chain.py
git commit -m "feat: add chat and RAG conversation chains"
```

---

### Task 10: API Routers

**Files:**
- Create: `routers/__init__.py`, `routers/chat.py`, `routers/session.py`, `routers/knowledge.py`
- Modify: `main.py`

- [ ] **Step 10.1: Create routers/__init__.py**

```python

```

- [ ] **Step 10.2: Create routers/chat.py**

```python
import json
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from chains.chat_chain import ChatChain
from chains.rag_chain import RAGChain
from services.memory_service import memory_service
from services.rag_service import rag_service
from utils.db import get_db

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = False
    provider: Optional[str] = None
    rag_k: int = 4


class ChatResponse(BaseModel):
    session_id: str
    response: str


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Send a message and receive a complete response."""
    if req.use_rag:
        chain = RAGChain(memory_service, rag_service, provider=req.provider)
        response = chain.invoke(req.session_id, req.message, k=req.rag_k)
    else:
        chain = ChatChain(memory_service, provider=req.provider)
        response = chain.invoke(req.session_id, req.message)

    return ChatResponse(session_id=req.session_id, response=response)


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """Send a message and receive a streaming SSE response."""

    from fastapi.responses import StreamingResponse

    async def event_generator():
        if req.use_rag:
            chain = RAGChain(memory_service, rag_service, provider=req.provider)
            async for chunk in chain.astream(req.session_id, req.message, k=req.rag_k):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        else:
            chain = ChatChain(memory_service, provider=req.provider)
            async for chunk in chain.astream(req.session_id, req.message):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

- [ ] **Step 10.3: Create routers/session.py**

```python
from uuid import uuid4

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.chat_message import ChatMessage
from models.chat_session import ChatSession
from utils.db import get_db

router = APIRouter(prefix="/api/session", tags=["session"])


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"


class SessionResponse(BaseModel):
    id: str
    title: str
    message_count: int


class MessageResponse(BaseModel):
    role: str
    content: str
    sequence: int


@router.post("", response_model=SessionResponse)
def create_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    """Create a new chat session."""
    session = ChatSession(
        id=str(uuid4()),
        title=req.title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse(
        id=session.id,
        title=session.title,
        message_count=session.message_count,
    )


@router.get("", response_model=list[SessionResponse])
def list_sessions(db: Session = Depends(get_db)):
    """List all chat sessions."""
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    return [
        SessionResponse(id=s.id, title=s.title, message_count=s.message_count)
        for s in sessions
    ]


@router.get("/{session_id}/history", response_model=list[MessageResponse])
def get_history(session_id: str, db: Session = Depends(get_db)):
    """Get message history for a session."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.sequence.asc())
        .all()
    )
    return [
        MessageResponse(role=m.role, content=m.content, sequence=m.sequence)
        for m in messages
    ]
```

- [ ] **Step 10.4: Create routers/knowledge.py**

```python
import os
import tempfile
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.knowledge_base import DocumentMeta, KnowledgeBase
from services.document_loader import process_document
from services.rag_service import rag_service
from utils.db import get_db

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class KBCreateRequest(BaseModel):
    name: str
    description: str = ""


class KBResponse(BaseModel):
    id: str
    name: str
    description: str


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    status: str


@router.post("/base", response_model=KBResponse)
def create_kb(req: KBCreateRequest, db: Session = Depends(get_db)):
    """Create a new knowledge base."""
    kb = KnowledgeBase(
        id=str(uuid4()),
        name=req.name,
        description=req.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return KBResponse(id=kb.id, name=kb.name, description=kb.description)


@router.get("/base", response_model=list[KBResponse])
def list_kbs(db: Session = Depends(get_db)):
    """List all knowledge bases."""
    kbs = db.query(KnowledgeBase).all()
    return [KBResponse(id=kb.id, name=kb.name, description=kb.description) for kb in kbs]


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    db: Session = Depends(get_db),
):
    """Upload a document to a knowledge base and index it."""
    # Verify KB exists
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Save file temporarily
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process and chunk
        chunks = process_document(tmp_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document is empty or unsupported format")

        # Create metadata for each chunk
        doc_id = str(uuid4())
        metadatas = [
            {"doc_id": doc_id, "kb_id": kb_id, "file_name": file.filename}
            for _ in chunks
        ]

        # Index into Milvus
        rag_service.add_texts(chunks, metadatas=metadatas)

        # Save to DB
        doc_meta = DocumentMeta(
            id=doc_id,
            kb_id=kb_id,
            file_name=file.filename,
            file_path=tmp_path,
            status="indexed",
        )
        db.add(doc_meta)
        db.commit()

        return DocumentResponse(id=doc_meta.id, file_name=doc_meta.file_name, status=doc_meta.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)
```

- [ ] **Step 10.5: Modify main.py**

```python
from fastapi import FastAPI

from config import get_settings
from utils.db import init_db

app = FastAPI(title="Chatbot Service")


@app.on_event("startup")
def on_startup():
    """Initialize database tables on startup."""
    init_db()


@app.get("/")
async def root():
    settings = get_settings()
    return {"service": settings.app_name, "status": "running"}


# Register routers
from routers.chat import router as chat_router  # noqa: E402
from routers.session import router as session_router  # noqa: E402
from routers.knowledge import router as knowledge_router  # noqa: E402

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(knowledge_router)
```

- [ ] **Step 10.6: Commit**

```bash
git add routers/ main.py
git commit -m "feat: add API routers for chat, session, and knowledge"
```

---

### Task 11: Docker Compose & Deployment

**Files:**
- Create: `docker-compose.yml`, `Dockerfile`, `.dockerignore`

- [ ] **Step 11.1: Create docker-compose.yml**

```yaml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - mysql
      - milvus-standalone
    volumes:
      - ./config:/app/config:ro

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: chatbot
    ports:
      - "3306:3306"
    volumes:
      - mysql-data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5

  etcd:
    image: quay.io/coreos/etcd:v3.5.11
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio
    ports:
      - "19530:19530"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3

volumes:
  mysql-data:
```

- [ ] **Step 11.2: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps for document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 11.3: Create .dockerignore**

```text
__pycache__
*.pyc
.venv
.env
.idea
.git
.gitignore
*.md
docs/
tests/
```

- [ ] **Step 11.4: Commit**

```bash
git add docker-compose.yml Dockerfile .dockerignore
git commit -m "feat: add Docker deployment configuration"
```

---

### Task 12: Tests & Final Integration

**Files:**
- Create: `tests/conftest.py`, `tests/test_llm_factory.py`

- [ ] **Step 12.1: Create tests/conftest.py**

```python
import pytest

from config.settings import LLMProviderConfig


@pytest.fixture
def openai_provider_config():
    return LLMProviderConfig(
        provider="openai",
        api_key="sk-test-key",
        api_base="https://api.openai.com/v1",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.fixture
def claude_provider_config():
    return LLMProviderConfig(
        provider="claude",
        api_key="sk-ant-test-key",
        model_name="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1024,
    )
```

- [ ] **Step 12.2: Create tests/test_llm_factory.py**

```python
import pytest

from services.llm_factory import LLMFactory


class TestLLMFactory:
    def test_list_providers(self):
        providers = LLMFactory.list_providers()
        assert "openai" in providers
        assert "claude" in providers
        assert "ernie" in providers
        assert "qwen" in providers

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="not found in configuration"):
            LLMFactory.create("nonexistent")
```

- [ ] **Step 12.3: Commit**

```bash
git add tests/conftest.py tests/test_llm_factory.py
git commit -m "test: add test fixtures and LLM factory tests"
```

---

## Self-Review: Spec Coverage Check

| Requirement | Covered By |
|-------------|-----------|
| FastAPI base | `main.py`, `routers/*` |
| LangChain chatbot | `chains/chat_chain.py`, `chains/rag_chain.py` |
| RAG knowledge base | `services/rag_service.py`, `services/embedding_service.py`, `utils/milvus_client.py`, `routers/knowledge.py` |
| Memory (short + long term) | `services/memory_service.py` |
| Pluggable LLM config | `services/llm_factory.py`, `services/openai_llm.py`, `services/claude_llm.py`, `services/chinese_llm.py`, `config/llm_config.yaml` |
| Milvus vector DB | `utils/milvus_client.py`, `services/rag_service.py` |
| MySQL persistence | `utils/db.py`, `models/chat_message.py`, `models/chat_session.py` |
| Multi-format documents | `services/document_loader.py` |
| Docker deployment | `docker-compose.yml`, `Dockerfile` |
| YAML + ENV config | `config/settings.py`, `config/llm_config.yaml`, `.env.example` |

**Placeholder scan:** No TBD/TODO/fill-in patterns found. All code steps contain actual implementation.

**Type consistency:** Function signatures and class names are consistent across tasks. `memory_service`, `rag_service` use singleton pattern consistently. `LLMFactory.create()` returns `BaseChatModel` across all providers.

---

## Task Summary

| Task | Focus | Estimated Files |
|------|-------|----------------|
| 1 | Project setup & config | 5 files |
| 2 | Database layer (MySQL) | 5 files |
| 3 | LLM factory (pluggable) | 6 files |
| 4 | Embedding service | 1 file |
| 5 | Memory service | 2 files |
| 6 | Milvus wrapper | 1 file |
| 7 | Document loader | 2 files |
| 8 | RAG service | 1 file |
| 9 | Chains (chat + RAG) | 3 files |
| 10 | API routers | 5 files |
| 11 | Docker deployment | 3 files |
| 12 | Tests & integration | 2 files |

Total: ~37 files, each task independently commitable and testable.
