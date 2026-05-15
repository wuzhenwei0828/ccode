import logging

from services.agent_service import AgentService
from services.llm import LLMFactory
from services.planners.base import BasePlanner
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from services.token_usage import merge_token_usage, normalize_token_usage, TokenUsage, usage_dict, _to_int
from services.tools.tools_helper import ToolsHelper

logger = logging.getLogger(__name__)


class PlanAndExecutePlanner(BasePlanner):
    """Placeholder planner for future plan-and-execute mode."""

    NOT_IMPLEMENTED_MESSAGE = "plan_and_execute mode is not implemented yet."
    SYSTEM_PROMPT = """
    你是任务的规划器，负责将大任务拆分成原子级的可执行步骤。你需要输出一个编号列表，每一步都要具体、可执行。
    你的输出格式必须严格遵守：
      - 只输出编号列表，严禁输出思考过程
      - 每一行代表一步，以数字编号开始，以句号结束。
      - 拆成多少步就输出多少行，便于后续解析
      
    示例：
      - 问：明天天气怎么样？
        答：1.获取今天是什么时候。
            2.根据今天的日期推出明天的日期。
            3.获取当前的地理位置。
            4.根据日期和位置查询天气情况。
    """

    def __init__(self, provider: str):
        super().__init__()
        self.executorLlm = LLMFactory.create(provider)

    @classmethod
    def normalize_final_response_text(cls, response) -> str:
        text = ToolsHelper.extract_text(response).strip()
        return text

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
        raise NotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)

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
        plan = llm.invoke(input=[
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"以下是历史对话摘要，仅供参考：\n{summary or '无'}"),
            *(history or []),
            HumanMessage(content=f"请为以下任务制定一个分步执行计划：{message}")
        ])
        logger.info(f"plan: {plan}")

        steps = self.parse_plan(plan)
        react_executor = AgentService(provider=self.provider)

        results = []
        total_usage = TokenUsage()
        for i, step in enumerate(steps):
            step_message = (
                f"你是整体任务中一个小任务的执行器，整体任务共{len(steps)}步，当前是第{i + 1}步，"
                f"前面步骤的执行结果是: {results}。\n\n"
                f"你需要完成的任务是：{step}"
            )
            step_result = react_executor.invoke1(
                "",
                [],
                message=step_message,
                tools=tools
            )
            logger.info(f"step {i + 1}: {step_result.get('response')}")
            usage = step_result.get("usage")
            token_usage = TokenUsage(
                input_tokens=_to_int(usage.get("input_tokens")),
                output_tokens=_to_int(usage.get("output_tokens")),
                analysis_tokens=_to_int(usage.get("analysis_tokens"))
            )
            total_usage = merge_token_usage(total_usage, token_usage)
            results.append(step_result.get("response"))

        # 根据执行结果生成最终回答（流式输出）
        final_prompt = f"""基于以下任务执行步骤的结果，请综合回答用户的原始问题。

执行步骤和结果：
{chr(10).join([f'步骤{i + 1}: {result}' for i, result in enumerate(results)])}

用户原始问题：{message}

请给出综合、准确的回答："""

        raw_response_stream = llm.stream(final_prompt)

        # 流式收集响应文本和 token usage
        accumulated_text = ""
        final_usage = TokenUsage()

        for chunk in raw_response_stream:
            chunk_text = self.normalize_final_response_text(chunk)
            if chunk_text:
                accumulated_text += chunk_text
                yield {"chunk": chunk_text, "text": chunk_text}

            # 收集每个 chunk 的 token usage
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                chunk_usage = TokenUsage(
                    input_tokens=_to_int(chunk.usage_metadata.get('input_tokens')),
                    output_tokens=_to_int(chunk.usage_metadata.get('output_tokens')),
                    analysis_tokens=_to_int(chunk.usage_metadata.get('analysis_tokens', 0))
                )
                final_usage = merge_token_usage(final_usage, chunk_usage)

        # 合并最终的 token usage
        total_usage = merge_token_usage(total_usage, final_usage)
        yield {"usage": usage_dict(total_usage)}

    def parse_plan(self, plan):
        content = plan.content.strip()
        steps = []
        for line in content.split("\n"):
            steps.append(line.strip())
        return steps
