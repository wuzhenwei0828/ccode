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
