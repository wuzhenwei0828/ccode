import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage

from services.llm_factory import LLMFactory
from services.token_usage import TokenUsage, merge_token_usage, normalize_token_usage


class AgentService:
    """Provider-agnostic tool loop for chat models."""

    FALLBACK_RESPONSE = "抱歉，我暂时无法完成这次工具调用。"

    def __init__(self, provider: str | None = None):
        self.provider = provider

    @staticmethod
    def _extract_text(response) -> str:
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

    @staticmethod
    def _build_tool_map(tools: list | None) -> dict[str, object]:
        return {tool.name: tool for tool in tools or []}

    @staticmethod
    def _to_model_tool(tool) -> dict:
        return {
            "name": tool.name,
            "description": getattr(tool, "description", ""),
            "input_schema": getattr(tool, "input_schema", {"type": "object", "properties": {}}),
        }

    def _bind_tools_if_supported(self, llm, tools: list | None):
        if not tools:
            return llm
        try:
            bind_tools = getattr(llm, "bind_tools")
        except AttributeError:
            return llm
        if not callable(bind_tools):
            return llm
        try:
            return bind_tools([self._to_model_tool(tool) for tool in tools])
        except (NotImplementedError, ValueError, TypeError, AttributeError):
            return llm

    @staticmethod
    def _extract_tool_calls(response) -> list[dict]:
        tool_calls = getattr(response, "tool_calls", None)
        if isinstance(tool_calls, list):
            return tool_calls

        text = AgentService._extract_text(response).strip()
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
    def _normalize_tool_args(args) -> dict:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            return json.loads(args)
        raise TypeError("Tool args must be a dict or JSON string")

    @staticmethod
    def _usage_dict(usage: TokenUsage) -> dict:
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "analysis_tokens": usage.analysis_tokens,
        }

    def _execute_tool_call(self, tool_map: dict[str, object], tool_call: dict) -> tuple[str, str]:
        name = tool_call.get("name") or "unknown_tool"
        call_id = tool_call.get("id") or name
        tool = tool_map.get(name)
        if tool is None:
            return call_id, json.dumps({"ok": False, "error": f"Unknown tool: {name}"}, ensure_ascii=False)

        try:
            args = self._normalize_tool_args(tool_call.get("args"))
            result = tool.handler(**args)
            return call_id, json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return call_id, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    def _run_loop(self, llm, messages: list, tools: list | None = None, max_iterations: int = 4) -> tuple[list, object, TokenUsage, bool]:
        working_messages = list(messages)
        tool_map = self._build_tool_map(tools)
        total_usage = TokenUsage()
        raw_response = None

        for _ in range(max_iterations):
            raw_response = llm.invoke(working_messages)
            total_usage = merge_token_usage(total_usage, normalize_token_usage(raw_response))
            tool_calls = self._extract_tool_calls(raw_response)
            if not tool_calls:
                return working_messages, raw_response, total_usage, False

            assistant_text = self._extract_text(raw_response)
            if assistant_text:
                working_messages.append(HumanMessage(content=assistant_text))

            for tool_call in tool_calls:
                call_id, tool_result = self._execute_tool_call(tool_map, tool_call)
                working_messages.append(ToolMessage(content=tool_result, tool_call_id=call_id))

        return working_messages, raw_response, total_usage, True

    def invoke(self, messages: list, tools: list | None = None, max_iterations: int = 4) -> dict:
        llm = LLMFactory.create(self.provider)
        effective_llm = self._bind_tools_if_supported(llm, tools)
        _, raw_response, total_usage, exhausted = self._run_loop(
            llm=effective_llm,
            messages=messages,
            tools=tools,
            max_iterations=max_iterations,
        )
        if exhausted:
            return {
                "response": self.FALLBACK_RESPONSE,
                "raw_response": raw_response,
                "usage": self._usage_dict(total_usage),
            }

        return {
            "response": self._extract_text(raw_response),
            "raw_response": raw_response,
            "usage": self._usage_dict(total_usage),
        }

    async def astream(self, messages: list, tools: list | None = None, max_iterations: int = 4):
        llm = LLMFactory.create(self.provider)
        effective_llm = self._bind_tools_if_supported(llm, tools)
        working_messages, raw_response, total_usage, exhausted = self._run_loop(
            llm=effective_llm,
            messages=messages,
            tools=tools,
            max_iterations=max_iterations,
        )

        if exhausted:
            yield {"chunk": self.FALLBACK_RESPONSE, "text": self.FALLBACK_RESPONSE}
            yield {"usage": self._usage_dict(total_usage)}
            return

        async for chunk in effective_llm.astream(working_messages):
            text = self._extract_text(chunk)
            if text:
                yield {"chunk": text, "text": text}

        yield {"usage": self._usage_dict(total_usage)}
