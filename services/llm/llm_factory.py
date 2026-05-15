from typing import Optional

from langchain_core.language_models import BaseChatModel

from config import get_settings
from config.settings import LLMProviderConfig
from services.llm.logged_chat_model import LoggedChatModel


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
    from services.llm.openai_llm import create_openai_chat_model
    return create_openai_chat_model(cfg)


def _create_claude_model(cfg: LLMProviderConfig) -> BaseChatModel:
    from services.llm.claude_llm import create_claude_chat_model
    return create_claude_chat_model(cfg)


def _create_chinese_model(cfg: LLMProviderConfig) -> BaseChatModel:
    from services.llm.chinese_llm import create_chinese_chat_model
    return create_chinese_chat_model(cfg)


def _create_siliconflow_model(cfg: LLMProviderConfig) -> BaseChatModel:
    from services.llm.siliconflow_llm import create_siliconflow_chat_model
    return create_siliconflow_chat_model(cfg)


LLMFactory.register("openai", _create_openai_model)
LLMFactory.register("claude", _create_claude_model)
LLMFactory.register("ernie", _create_chinese_model)
LLMFactory.register("qwen", _create_chinese_model)
LLMFactory.register("siliconflow", _create_siliconflow_model)
