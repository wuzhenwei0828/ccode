from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.llm.claude_llm import create_claude_chat_model
from services.llm.chinese_llm import create_chinese_chat_model
from services.llm.llm_factory import LLMFactory
from services.llm.openai_llm import create_openai_chat_model
from services.llm.siliconflow_llm import create_siliconflow_chat_model


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

    def test_create_returns_logged_wrapper_for_openai_provider(self):
        LoggedChatModel = __import__("services.llm.logged_chat_model", fromlist=["LoggedChatModel"]).LoggedChatModel
        cfg = MagicMock()
        cfg.provider = "openai"
        cfg.model_name = "gpt-4o-mini"
        cfg.temperature = 0.1
        cfg.max_tokens = 128
        cfg.api_key = "k"
        cfg.api_base = None
        settings = MagicMock()
        settings.get_llm_providers.return_value = [cfg]
        inner = MagicMock()

        with patch("services.llm.llm_factory.get_settings", return_value=settings), patch(
            "services.llm.openai_llm.ChatOpenAI"
        ) as mock_chat_openai:
            mock_chat_openai.return_value = inner
            model = LLMFactory.create("openai")

        assert isinstance(model, LoggedChatModel)
        assert model.inner is inner
        assert model.provider == "openai"
        assert model.model_name == "gpt-4o-mini"

    @pytest.mark.parametrize(
        ("factory", "patch_target", "provider", "api_base"),
        [
            (create_openai_chat_model, "services.llm.openai_llm.ChatOpenAI", "openai", None),
            (create_claude_chat_model, "services.llm.claude_llm.ChatAnthropic", "claude", None),
            (create_chinese_chat_model, "services.llm.chinese_llm.ChatOpenAI", "ernie", "https://ernie.example.com"),
            (
                create_siliconflow_chat_model,
                "services.llm.siliconflow_llm.ChatOpenAI",
                "siliconflow",
                "https://siliconflow.example.com",
            ),
        ],
    )
    def test_provider_creators_return_real_model_instances(self, factory, patch_target, provider, api_base):
        inner = MagicMock()
        cfg = SimpleNamespace(
            provider=provider,
            model_name=f"{provider}-model",
            temperature=0.3,
            max_tokens=256,
            api_key="secret",
            api_base=api_base,
        )

        with patch(patch_target) as mock_chat_model:
            mock_chat_model.return_value = inner
            model = factory(cfg)

        assert model is inner
