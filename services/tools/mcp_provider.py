from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.settings import PROJECT_ROOT
from services.tools.mcp_adapter import adapt_mcp_tool
from services.tools.mcp_config import load_mcp_server_configs

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:  # pragma: no cover - exercised indirectly via downgrade behavior
    MultiServerMCPClient = None

if TYPE_CHECKING:
    from services.tools.agent_tools import ChatTool

logger = logging.getLogger(__name__)


@dataclass
class MCPToolProvider:
    project_root: Path = PROJECT_ROOT
    label: str = "mcp"
    _tools: list["ChatTool"] | None = field(default=None, init=False, repr=False)
    _load_lock: asyncio.Lock | None = field(default=None, init=False, repr=False)
    _dependency_unavailable: bool = field(default=False, init=False, repr=False)

    def build_tools(self) -> list["ChatTool"]:
        if self._tools is not None:
            return list(self._tools)
        return []

    async def abuild_tools(self) -> list["ChatTool"]:
        if self._tools is not None:
            return list(self._tools)

        async with self._get_load_lock():
            if self._tools is not None:
                return list(self._tools)
            tools, cacheable = await self._aload_tools()
            if cacheable:
                self._tools = tools
            return list(tools)

    async def _aload_tools(self) -> tuple[list["ChatTool"], bool]:
        if self._dependency_unavailable:
            return [], True

        configs = load_mcp_server_configs(self.project_root)
        if not configs:
            return [], True

        if MultiServerMCPClient is None:
            logger.warning("langchain_mcp_adapters is unavailable; skipping mcp tools")
            self._dependency_unavailable = True
            return [], True

        try:
            client = MultiServerMCPClient(self._build_client_config(configs))
            raw_tools = await client.get_tools()
        except Exception:
            logger.exception("failed to load mcp tools")
            return [], False

        return [adapt_mcp_tool(raw_tool) for raw_tool in raw_tools], True

    def _get_load_lock(self) -> asyncio.Lock:
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

    @staticmethod
    def _build_client_config(configs: list[Any]) -> dict[str, dict[str, Any]]:
        return {
            config.name: MCPToolProvider._build_single_client_config(config)
            for config in configs
        }

    @staticmethod
    def _build_single_client_config(config: Any) -> dict[str, Any]:
        if config.transport == "stdio":
            if not config.command:
                raise ValueError(f"mcp server {config.name} missing command")
            return {
                "transport": "stdio",
                "command": config.command,
                "args": list(config.args),
            }

        if config.transport not in {"http", "sse"}:
            raise ValueError(f"mcp server {config.name} has unsupported transport {config.transport}")
        if not config.url:
            raise ValueError(f"mcp server {config.name} missing url")

        remote_config = {
            "transport": config.transport,
            "url": config.url,
        }
        if config.headers:
            remote_config["headers"] = dict(config.headers)
        return remote_config
