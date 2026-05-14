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
