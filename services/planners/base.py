from abc import ABC, abstractmethod


class BasePlanner(ABC):
    FALLBACK_RESPONSE = "抱歉，我暂时无法完成这次工具调用。"

    def __init__(self, provider: str | None = None):
        self.provider = provider

    @abstractmethod
    def invoke(
        self,
        llm,
        summary: str,
        history: list,
        message: str,
        tools: list | None = None,
        context: str | None = None,
        max_iterations: int = 4,
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def astream(
        self,
        llm,
        summary: str,
        history: list,
        message: str,
        tools: list | None = None,
        context: str | None = None,
        max_iterations: int = 4,
    ):
        raise NotImplementedError
