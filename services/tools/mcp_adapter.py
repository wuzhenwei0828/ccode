from __future__ import annotations

from typing import Any

from services.tools.agent_tools import ChatTool


def _normalize_input_schema(raw_tool: Any) -> dict[str, Any]:
    schema = getattr(raw_tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema

    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            result = model_json_schema()
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    return {"type": "object", "properties": {}}


def adapt_mcp_tool(raw_tool: Any) -> ChatTool:
    def handler(**kwargs):
        return raw_tool.invoke(kwargs)

    return ChatTool(
        name=str(raw_tool.name),
        description=str(getattr(raw_tool, "description", "") or ""),
        input_schema=_normalize_input_schema(raw_tool),
        handler=handler,
    )
