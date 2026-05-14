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
