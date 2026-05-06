import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent


class LLMProviderConfig(BaseModel):
    """Single LLM provider configuration."""

    provider: Literal["openai", "claude", "ernie", "qwen", "siliconflow"]
    api_key: str = ""
    api_base: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048


class MemoryConfig(BaseModel):
    """Memory configuration."""

    short_term_window: int = Field(default=10, description="Number of recent messages in sliding window")
    summary_max_length: int = Field(default=1000, description="Max characters for conversation summary")
    summary_update_interval: int = Field(default=10, description="Number of new messages to trigger summary update")
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
    milvus_collection: str = "xxsy_wzw"

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
