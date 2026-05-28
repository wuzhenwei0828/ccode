from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from services.tools.builtin_provider import BuiltinToolProvider
from services.tools.mcp_provider import MCPToolProvider

if TYPE_CHECKING:
    from services.tools.agent_tools import ChatTool

logger = logging.getLogger(__name__)


class ToolProvider(Protocol):
    def build_tools(self) -> list["ChatTool"]: ...


def _provider_label(provider: ToolProvider) -> str:
    label = getattr(provider, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    return provider.__class__.__name__.lower()


@dataclass(frozen=True)
class ToolRegistry:
    providers: tuple[ToolProvider, ...]

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls(
            providers=(
                BuiltinToolProvider(),
                MCPToolProvider(),
            )
        )

    def build_tools(self) -> list["ChatTool"]:
        merged: dict[str, "ChatTool"] = {}
        for provider in self.providers:
            provider_name = _provider_label(provider)
            try:
                tools = provider.build_tools()
            except Exception:
                logger.exception("tool provider build failed provider=%s", provider_name)
                continue
            self._merge_provider_tools(merged, provider_name, tools)
        return self._finalize_merged_tools(merged)

    async def abuild_tools(self) -> list["ChatTool"]:
        merged: dict[str, "ChatTool"] = {}
        for provider in self.providers:
            provider_name = _provider_label(provider)
            try:
                async_builder = getattr(provider, "abuild_tools", None)
                if callable(async_builder):
                    tools = await async_builder()
                else:
                    tools = provider.build_tools()
            except Exception:
                logger.exception("tool provider build failed provider=%s", provider_name)
                continue
            self._merge_provider_tools(merged, provider_name, tools)
        return self._finalize_merged_tools(merged)

    @staticmethod
    def _merge_provider_tools(merged: dict[str, "ChatTool"], provider_name: str, tools: list["ChatTool"]) -> None:
        loaded_names = ",".join(tool.name for tool in tools) or "none"
        logger.info(f"{provider_name} tools loaded: %s", loaded_names)

        for tool in tools:
            previous = merged.get(tool.name)
            if previous is not None:
                logger.warning(
                    "tool override name=%s old_description=%s new_description=%s",
                    tool.name,
                    previous.description,
                    tool.description,
                )
            merged[tool.name] = tool

    @staticmethod
    def _finalize_merged_tools(merged: dict[str, "ChatTool"]) -> list["ChatTool"]:
        merged_names = ",".join(merged) or "none"
        logger.info("merged tools loaded: %s", merged_names)
        return list(merged.values())
