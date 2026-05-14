import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from services.planners.base import BasePlanner
from services.token_usage import TokenUsage, merge_token_usage, normalize_token_usage

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = (
    "你是一个会使用工具完成任务的 agent。如果需要工具，必须优先调用工具，不要猜测工具返回结果。"
    "调用工具后，必须基于工具结果继续完成任务，直到获取的信息足以回答问题。"
    "禁止从历史对话或记忆中推断当前时间，所有时间相关问题必须通过工具查询实时值。"
    "未通过工具获取足够多的信息前，不要给出最终结论。"
    "最终回答要简洁明了，不要废话，不要向用户暴露内部推理过程。"
)


class ReActPlanner(BasePlanner):
    """Provider-agnostic ReAct planner for chat models."""

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

    @classmethod
    def _extract_tool_calls(cls, response) -> list[dict]:
        tool_calls = getattr(response, "tool_calls", None)
        if isinstance(tool_calls, list):
            return tool_calls

        text = cls._extract_text(response).strip()
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

    @classmethod
    def _build_assistant_replay_message(cls, response, tool_calls: list[dict]):
        if isinstance(response, AIMessage):
            return response
        normalized_tool_calls = []
        for tool_call in tool_calls:
            normalized_tool_call = dict(tool_call)
            normalized_tool_call.setdefault("id", normalized_tool_call.get("name") or "tool_call")
            normalized_tool_calls.append(normalized_tool_call)
        return AIMessage(content=cls._extract_text(response), tool_calls=normalized_tool_calls)

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
        raw_args = tool_call.get("args")

        try:
            args = self._normalize_tool_args(raw_args) if raw_args is not None else {}
        except Exception as exc:
            logger.info("agent tool error name=%s args=%s error=%s", name, raw_args, exc)
            return call_id, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        tool = tool_map.get(name)
        if tool is None:
            logger.info("agent tool error name=%s args=%s error=%s", name, args, f"Unknown tool: {name}")
            return call_id, json.dumps({"ok": False, "error": f"Unknown tool: {name}"}, ensure_ascii=False)

        try:
            logger.info("agent tool call name=%s args=%s", name, args)
            result = tool.handler(**args)
            logger.info("agent tool result name=%s result=%s", name, result)
            return call_id, json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.info("agent tool error name=%s args=%s error=%s", name, args, exc)
            return call_id, json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    @classmethod
    def _normalize_final_response_text(cls, response) -> str:
        text = cls._extract_text(response).strip()
        return text or cls.FALLBACK_RESPONSE

    def _run_react_loop(self, llm, messages: list, tools: list | None = None, max_iterations: int = 4) -> tuple[list, object, TokenUsage, bool]:
        working_messages = list(messages)
        tool_map = self._build_tool_map(tools)
        total_usage = TokenUsage()
        raw_response = None
        tool_names = [tool.name for tool in tools or []]

        for iteration in range(max_iterations):
            logger.info(
                "agent invoke request provider=%s iteration=%d messages=%s tools=%s",
                self.provider,
                iteration,
                working_messages,
                tool_names,
            )
            raw_response = llm.invoke(working_messages)
            total_usage = merge_token_usage(total_usage, normalize_token_usage(raw_response))
            tool_calls = self._extract_tool_calls(raw_response)
            logger.info(
                "agent invoke response provider=%s iteration=%d raw_response=%s tool_calls=%s",
                self.provider,
                iteration,
                raw_response,
                tool_calls,
            )
            if not tool_calls:
                return working_messages, raw_response, total_usage, False

            if raw_response is not None:
                working_messages.append(self._build_assistant_replay_message(raw_response, tool_calls))

            for tool_call in tool_calls:
                call_id, tool_result = self._execute_tool_call(tool_map, tool_call)
                working_messages.append(ToolMessage(content=tool_result, tool_call_id=call_id))

        return working_messages, raw_response, total_usage, True

    @staticmethod
    def _build_user_message(message: str, context: str | None = None) -> HumanMessage:
        if not context:
            return HumanMessage(content=message)
        return HumanMessage(
            content=(
                "以下是检索到的参考资料，仅在相关时使用：\n"
                f"{context}\n\n"
                f"问题：{message}"
            )
        )

    @classmethod
    def _build_messages(cls, summary: str, history: list, message: str, context: str | None = None) -> list:
        return [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=f"以下是历史对话摘要，仅供参考：\n{summary or '无'}"),
            *(history or []),
            cls._build_user_message(message=message, context=context),
        ]

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
        effective_llm = self._bind_tools_if_supported(llm, tools)
        messages = self._build_messages(summary, history, message, context)
        _, raw_response, total_usage, exhausted = self._run_react_loop(
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
            "response": self._normalize_final_response_text(raw_response),
            "raw_response": raw_response,
            "usage": self._usage_dict(total_usage),
        }

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
        effective_llm = self._bind_tools_if_supported(llm, tools)
        messages = self._build_messages(summary, history, message, context)
        working_messages, raw_response, total_usage, exhausted = self._run_react_loop(
            llm=effective_llm,
            messages=messages,
            tools=tools,
            max_iterations=max_iterations,
        )

        if exhausted or raw_response is None:
            yield {"chunk": self.FALLBACK_RESPONSE, "text": self.FALLBACK_RESPONSE}
            yield {"usage": self._usage_dict(total_usage)}
            return

        final_text = self._normalize_final_response_text(raw_response)
        yield {"chunk": final_text, "text": final_text}
        yield {"usage": self._usage_dict(total_usage)}
