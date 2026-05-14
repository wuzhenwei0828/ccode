from services.llm.llm_factory import LLMFactory
from services.planners.react_planner import ReActPlanner


class AgentService:
    """Compatibility facade delegating agent behavior to ReActPlanner."""

    FALLBACK_RESPONSE = ReActPlanner.FALLBACK_RESPONSE

    def __init__(self, provider: str | None = None):
        self.provider = provider
        self._planner = ReActPlanner(provider=provider)

    @staticmethod
    def _split_messages(messages: list) -> tuple[str, list, str]:
        if not messages:
            return "", [], ""
        if len(messages) == 1:
            return "", [], messages[0]
        return "", list(messages[:-1]), messages[-1]

    def invoke(self, messages: list, tools: list | None = None, max_iterations: int = 4) -> dict:
        summary, history, message = self._split_messages(messages)
        llm = LLMFactory.create(self.provider)
        return self._planner.invoke(
            llm=llm,
            summary=summary,
            history=history,
            message=message,
            tools=tools,
            context=None,
            max_iterations=max_iterations,
        )

    async def astream(self, messages: list, tools: list | None = None, max_iterations: int = 4):
        summary, history, message = self._split_messages(messages)
        llm = LLMFactory.create(self.provider)
        async for event in self._planner.astream(
            llm=llm,
            summary=summary,
            history=history,
            message=message,
            tools=tools,
            context=None,
            max_iterations=max_iterations,
        ):
            yield event
