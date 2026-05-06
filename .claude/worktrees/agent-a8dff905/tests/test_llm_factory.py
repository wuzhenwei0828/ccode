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
