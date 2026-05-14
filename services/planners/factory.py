from services.planners.plan_execute_planner import PlanAndExecutePlanner
from services.planners.react_planner import ReActPlanner


class PlannerFactory:
    @staticmethod
    def create(mode: str = "react", provider: str | None = None):
        if mode == "react":
            return ReActPlanner(provider=provider)
        if mode == "plan_and_execute":
            return PlanAndExecutePlanner(provider=provider)
        raise ValueError(f"Unsupported planner mode: {mode}")
