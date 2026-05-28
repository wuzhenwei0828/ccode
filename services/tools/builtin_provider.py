from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.tools.agent_tools import ChatTool


@dataclass(frozen=True)
class BuiltinToolProvider:
    label: str = "builtin"

    def build_tools(self) -> list["ChatTool"]:
        from services.tools.agent_tools import (
            _build_current_time_tool,
            _build_knowledge_search_tool,
            _build_web_search_tool,
        )

        return [
            _build_knowledge_search_tool(),
            _build_current_time_tool(),
            _build_web_search_tool(),
        ]
