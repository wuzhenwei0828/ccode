from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import LLMProviderConfig

def create_siliconflow_chat_model(cfg: LLMProviderConfig) -> BaseChatModel:
    """Create a SiliconFlow LLM via OpenAI-compatible interface."""
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