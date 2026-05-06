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
