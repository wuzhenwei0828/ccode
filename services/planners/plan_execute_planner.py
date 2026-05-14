from services.planners.base import BasePlanner


class PlanAndExecutePlanner(BasePlanner):
    """Placeholder planner for future plan-and-execute mode."""

    NOT_IMPLEMENTED_MESSAGE = "plan_and_execute mode is not implemented yet."

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
        raise NotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)
