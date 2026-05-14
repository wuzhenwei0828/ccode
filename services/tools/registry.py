from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from services.tools.agent_tools import ChatTool


@dataclass(frozen=True)
class ToolRegistry:
    _builders: tuple[Callable[[], ChatTool], ...]

    @classmethod
    def default(cls) -> "ToolRegistry":
        from services.tools.agent_tools import (
            _build_current_time_tool,
            _build_knowledge_search_tool,
            _build_web_search_tool,
        )

        return cls(
            _builders=(
                _build_knowledge_search_tool,
                _build_current_time_tool,
                _build_web_search_tool,
            )
        )

    def build_tools(self) -> list[ChatTool]:
        return [builder() for builder in self._builders]
