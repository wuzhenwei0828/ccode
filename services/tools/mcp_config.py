from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ALLOWED_TRANSPORTS = {"http", "sse", "stdio"}


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    url: str | None
    command: str | None
    args: tuple[str, ...]
    headers: dict[str, str]


def load_mcp_server_configs(project_root: Path) -> list[MCPServerConfig]:
    config_path = project_root / ".mcp.json"
    if not config_path.exists():
        return []

    payload = json.loads(config_path.read_text(encoding="utf-8")) or {}
    servers = payload.get("mcpServers") or {}
    configs: list[MCPServerConfig] = []

    for name, raw in servers.items():
        if not isinstance(raw, dict):
            raise ValueError(f"mcp server {name} must be an object")

        transport = str(raw.get("type") or "").strip()
        url = str(raw.get("url") or "").strip() or None
        command = str(raw.get("command") or "").strip() or None
        args = tuple(str(item) for item in raw.get("args") or ())
        headers = {str(k): str(v) for k, v in (raw.get("headers") or {}).items()}

        if not transport:
            raise ValueError(f"mcp server {name} missing type")
        if transport not in ALLOWED_TRANSPORTS:
            raise ValueError(f"mcp server {name} has unsupported type {transport}")
        if transport in {"http", "sse"} and not url:
            raise ValueError(f"mcp server {name} missing url")
        if transport == "stdio" and not command:
            raise ValueError(f"mcp server {name} missing command")

        configs.append(
            MCPServerConfig(
                name=name,
                transport=transport,
                url=url,
                command=command,
                args=args,
                headers=headers,
            )
        )

    return configs
