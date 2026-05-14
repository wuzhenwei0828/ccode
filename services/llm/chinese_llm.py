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
