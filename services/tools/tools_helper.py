import json


class ToolsHelper:
    @classmethod
    def bind_tools_if_supported(cls, llm, tools: list | None):
        if not tools:
            return llm
        try:
            bind_tools = getattr(llm, "bind_tools")
        except AttributeError:
            return llm
        if not callable(bind_tools):
            return llm
        try:
            return bind_tools([cls.to_model_tool(tool) for tool in tools])
        except (NotImplementedError, ValueError, TypeError, AttributeError):
            return llm

    @staticmethod
    def to_model_tool(tool) -> dict:
        return {
            "name": tool.name,
            "description": getattr(tool, "description", ""),
            "input_schema": getattr(tool, "input_schema", {"type": "object", "properties": {}}),
        }

    @classmethod
    def extract_tool_calls(cls, response) -> list[dict]:
        tool_calls = getattr(response, "tool_calls", None)
        if isinstance(tool_calls, list):
            return tool_calls

        text = cls.extract_text(response).strip()
        if not text.startswith("TOOL_CALL "):
            return []

        payload = text[len("TOOL_CALL "):].strip()
        try:
            tool_call = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(tool_call, dict) and tool_call.get("name"):
            return [tool_call]
        return []

    @staticmethod
    def extract_text(response) -> str:
        if isinstance(response, str):
            return response
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(getattr(item, "text", ""))
            return "".join(parts)
        return str(content or "")