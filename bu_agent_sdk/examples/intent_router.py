"""
多类型意图路由示例 - 基于 BU Agent SDK 实现 Skills/Tools/Flows/Message 分发

这个示例展示了如何在 BU Agent SDK 的设计体系下实现多种意图类型的自动匹配：
- Skills: 复杂的多步骤能力（如"帮我写一篇博客"）
- Tools: 单一功能调用（如"搜索天气"）
- Flows: 固定模式匹配的工作流（如"我要请假"触发请假流程）
- Message: 纯文本对话（闲聊、问答）

运行方式:
    python -m bu_agent_sdk.examples.intent_router
"""

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import TaskComplete
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import Depends, tool


# =============================================================================
# 1. 意图类型定义
# =============================================================================


class IntentType(str, Enum):
    """意图类型枚举"""

    SKILL = "skill"  # 复杂技能，需要子 Agent 处理
    TOOL = "tool"  # 单一工具调用
    FLOW = "flow"  # 固定流程
    MESSAGE = "message"  # 纯文本对话


# =============================================================================
# 2. Skills 定义 - 复杂多步骤能力
# =============================================================================


@dataclass
class SkillDefinition:
    """技能定义"""

    name: str
    description: str
    system_prompt: str
    tools: list  # 技能可用的工具列表


# 示例技能：博客写作助手
BLOG_WRITER_SKILL = SkillDefinition(
    name="blog_writer",
    description="撰写博客文章，包括选题、大纲、正文撰写、润色等完整流程",
    system_prompt="""你是一个专业的博客写作助手。
按以下流程撰写博客：
1. 确认主题和目标读者
2. 生成大纲
3. 撰写正文
4. 润色和优化
完成后调用 done 工具。""",
    tools=[],  # 会在运行时填充
)

# 示例技能：代码审查助手
CODE_REVIEW_SKILL = SkillDefinition(
    name="code_reviewer",
    description="进行代码审查，分析代码质量、潜在问题、改进建议",
    system_prompt="""你是一个专业的代码审查助手。
分析代码时关注：
1. 代码风格和规范
2. 潜在的 bug 和安全问题
3. 性能优化建议
4. 可读性和可维护性
完成后调用 done 工具。""",
    tools=[],
)

# 技能注册表
SKILLS_REGISTRY: dict[str, SkillDefinition] = {
    "blog_writer": BLOG_WRITER_SKILL,
    "code_reviewer": CODE_REVIEW_SKILL,
}


# =============================================================================
# 3. Tools 定义 - 单一功能工具
# =============================================================================


@tool("搜索天气信息")
async def search_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 模拟天气 API
    return f"{city}天气：晴，温度 25°C，湿度 60%"


@tool("搜索新闻")
async def search_news(topic: str, limit: int = 5) -> str:
    """搜索指定主题的新闻"""
    # 模拟新闻搜索
    return f"关于'{topic}'的最新新闻：\n1. 新闻标题1\n2. 新闻标题2\n3. 新闻标题3"


@tool("计算数学表达式")
async def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 简单的安全计算（生产环境需要更严格的沙箱）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool("翻译文本")
async def translate(text: str, target_lang: str = "英文") -> str:
    """将文本翻译为目标语言"""
    # 模拟翻译
    return f"[翻译为{target_lang}]: {text} -> (translated text)"


# 工具注册表
TOOLS_REGISTRY: dict[str, Any] = {
    "search_weather": search_weather,
    "search_news": search_news,
    "calculate": calculate,
    "translate": translate,
}


# =============================================================================
# 4. Flows 定义 - 固定模式匹配工作流
# =============================================================================


@dataclass
class FlowDefinition:
    """流程定义"""

    name: str
    description: str
    trigger_patterns: list[str]  # 正则匹配模式
    steps: list[str]  # 流程步骤


# 示例流程：请假申请
LEAVE_REQUEST_FLOW = FlowDefinition(
    name="leave_request",
    description="请假申请流程",
    trigger_patterns=[
        r"我要请假",
        r"申请.*假",
        r"请.*天假",
        r"休假申请",
    ],
    steps=[
        "1. 确认请假类型（年假/病假/事假）",
        "2. 确认请假日期范围",
        "3. 填写请假原因",
        "4. 提交审批",
        "5. 等待审批结果",
    ],
)

# 示例流程：报销申请
REIMBURSEMENT_FLOW = FlowDefinition(
    name="reimbursement",
    description="费用报销流程",
    trigger_patterns=[
        r"我要报销",
        r"申请报销",
        r"费用报销",
        r"报销.*费用",
    ],
    steps=[
        "1. 选择报销类型（差旅/办公/其他）",
        "2. 填写报销金额",
        "3. 上传发票凭证",
        "4. 填写报销说明",
        "5. 提交审批",
    ],
)

# 流程注册表
FLOWS_REGISTRY: dict[str, FlowDefinition] = {
    "leave_request": LEAVE_REQUEST_FLOW,
    "reimbursement": REIMBURSEMENT_FLOW,
}


def match_flow(user_input: str) -> FlowDefinition | None:
    """通过正则匹配检查是否触发流程"""
    for flow in FLOWS_REGISTRY.values():
        for pattern in flow.trigger_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return flow
    return None


# =============================================================================
# 5. 上下文管理
# =============================================================================


@dataclass
class RouterContext:
    """路由器上下文，用于依赖注入"""

    llm: Any  # BaseChatModel
    skills: dict[str, SkillDefinition] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    flows: dict[str, FlowDefinition] = field(default_factory=dict)
    current_skill_agent: Agent | None = None
    current_flow: FlowDefinition | None = None
    flow_state: dict = field(default_factory=dict)


def get_router_context() -> RouterContext:
    """依赖注入标记"""
    raise RuntimeError("Must be overridden via dependency_overrides")


# =============================================================================
# 6. 路由工具定义 - 核心意图分发逻辑
# =============================================================================


class SkillRequest(BaseModel):
    """技能调用请求"""

    skill_name: str = Field(description="要调用的技能名称，可选：blog_writer, code_reviewer")
    user_request: str = Field(description="用户的具体请求内容")


@tool("调用复杂技能来完成多步骤任务，如写博客、代码审查等")
async def dispatch_to_skill(
    request: SkillRequest,
    ctx: Annotated[RouterContext, Depends(get_router_context)],
) -> str:
    """当用户需要完成复杂的多步骤任务时，调用对应的技能"""
    skill = ctx.skills.get(request.skill_name)
    if not skill:
        available = ", ".join(ctx.skills.keys())
        return f"未找到技能 '{request.skill_name}'，可用技能: {available}"

    # 创建子 Agent 执行技能
    @tool("标记技能任务完成")
    async def skill_done(result: str) -> str:
        raise TaskComplete(result)

    skill_agent = Agent(
        llm=ctx.llm,
        tools=[skill_done],
        system_prompt=skill.system_prompt,
        max_iterations=20,
    )

    # 执行技能
    result = await skill_agent.query(request.user_request)
    return f"[技能 {skill.name} 完成]\n{result}"


class ToolRequest(BaseModel):
    """工具调用请求"""

    tool_name: str = Field(
        description="要调用的工具名称，可选：search_weather, search_news, calculate, translate"
    )
    arguments: dict = Field(description="工具参数，如 {'city': '北京'} 或 {'expression': '2+3'}")


@tool("调用单一功能工具，如搜索天气、计算、翻译等")
async def dispatch_to_tool(
    request: ToolRequest,
    ctx: Annotated[RouterContext, Depends(get_router_context)],
) -> str:
    """当用户需要执行简单的单一功能时，调用对应的工具"""
    tool_func = ctx.tools.get(request.tool_name)
    if not tool_func:
        available = ", ".join(ctx.tools.keys())
        return f"未找到工具 '{request.tool_name}'，可用工具: {available}"

    # 执行工具
    try:
        result = await tool_func.execute(**request.arguments)
        return f"[工具 {request.tool_name} 执行结果]\n{result}"
    except Exception as e:
        return f"工具执行错误: {e}"


class FlowRequest(BaseModel):
    """流程启动请求"""

    flow_name: str = Field(description="要启动的流程名称，可选：leave_request, reimbursement")
    initial_data: dict = Field(default_factory=dict, description="流程初始数据")


@tool("启动固定流程，如请假申请、费用报销等标准化流程")
async def dispatch_to_flow(
    request: FlowRequest,
    ctx: Annotated[RouterContext, Depends(get_router_context)],
) -> str:
    """当用户需要执行标准化流程时，启动对应的流程"""
    flow = ctx.flows.get(request.flow_name)
    if not flow:
        available = ", ".join(ctx.flows.keys())
        return f"未找到流程 '{request.flow_name}'，可用流程: {available}"

    # 返回流程信息（实际应用中会启动状态机）
    steps_text = "\n".join(flow.steps)
    return f"""[启动流程: {flow.name}]
{flow.description}

流程步骤：
{steps_text}

请按步骤提供所需信息。"""


class MessageResponse(BaseModel):
    """消息响应"""

    content: str = Field(description="回复给用户的消息内容")


@tool("直接回复用户消息，用于闲聊、问答等不需要调用工具或技能的场景")
async def respond_message(response: MessageResponse) -> str:
    """当用户只是闲聊或提问，不需要执行任何动作时使用"""
    raise TaskComplete(response.content)


@tool("结束对话")
async def done(message: str) -> str:
    """当任务完成时调用"""
    raise TaskComplete(message)


# =============================================================================
# 7. 意图路由 Agent
# =============================================================================


def create_intent_router(llm: Any) -> Agent:
    """创建意图路由 Agent"""

    # 准备上下文
    ctx = RouterContext(
        llm=llm,
        skills=SKILLS_REGISTRY,
        tools=TOOLS_REGISTRY,
        flows=FLOWS_REGISTRY,
    )

    # 路由器的系统提示
    system_prompt = """你是一个智能意图路由器，负责理解用户意图并分发到正确的处理器。

## 意图类型

1. **Skills (技能)** - 复杂的多步骤任务
   - blog_writer: 撰写博客文章
   - code_reviewer: 代码审查
   → 使用 dispatch_to_skill

2. **Tools (工具)** - 单一功能调用
   - search_weather: 查询天气
   - search_news: 搜索新闻
   - calculate: 数学计算
   - translate: 文本翻译
   → 使用 dispatch_to_tool

3. **Flows (流程)** - 标准化工作流
   - leave_request: 请假申请
   - reimbursement: 费用报销
   → 使用 dispatch_to_flow

4. **Message (消息)** - 闲聊/问答
   → 使用 respond_message 直接回复

## 路由规则

- 优先匹配 Flow（如果用户意图明确匹配标准流程）
- 其次匹配 Tool（如果是简单的单一功能需求）
- 再次匹配 Skill（如果是复杂的多步骤任务）
- 最后是 Message（纯对话）

请准确理解用户意图并选择最合适的处理方式。"""

    # 创建路由 Agent
    router_agent = Agent(
        llm=llm,
        tools=[
            dispatch_to_skill,
            dispatch_to_tool,
            dispatch_to_flow,
            respond_message,
            done,
        ],
        system_prompt=system_prompt,
        dependency_overrides={get_router_context: lambda: ctx},
        require_done_tool=False,  # 允许 respond_message 抛出 TaskComplete
    )

    return router_agent


# =============================================================================
# 8. 高级版本：带预匹配的路由器
# =============================================================================


class IntentRouterWithPreMatch:
    """
    带预匹配的意图路由器

    在调用 LLM 之前，先进行规则匹配（适用于 Flow 类型的固定模式）
    这样可以：
    1. 减少 LLM 调用成本
    2. 对于明确的流程触发，保证 100% 准确率
    """

    def __init__(self, llm: Any):
        self.llm = llm
        self.router_agent = create_intent_router(llm)
        self.flows = FLOWS_REGISTRY

    async def route(self, user_input: str) -> str:
        """路由用户输入到正确的处理器"""

        # Step 1: 预匹配 - 检查是否触发固定流程
        matched_flow = match_flow(user_input)
        if matched_flow:
            steps_text = "\n".join(matched_flow.steps)
            return f"""[预匹配触发流程: {matched_flow.name}]
{matched_flow.description}

流程步骤：
{steps_text}

请按步骤提供所需信息。"""

        # Step 2: LLM 路由 - 让 LLM 决定意图类型
        result = await self.router_agent.query(user_input)
        return result


# =============================================================================
# 9. 演示
# =============================================================================


async def demo():
    """演示意图路由"""
    print("=" * 60)
    print("多类型意图路由演示")
    print("=" * 60)

    # 创建 LLM（使用 mock 或真实的）
    try:
        llm = ChatOpenAI(model="gpt-4o")
    except Exception:
        print("⚠️  无法创建 LLM，使用模拟模式")
        return

    # 创建路由器
    router = IntentRouterWithPreMatch(llm)

    # 测试用例
    test_cases = [
        # Flow 触发（预匹配）
        "我要请假三天",
        # Tool 调用
        "北京今天天气怎么样？",
        # Skill 调用
        "帮我写一篇关于 AI 的博客文章",
        # Message 回复
        "你好，你是谁？",
        # 计算
        "帮我算一下 123 * 456",
    ]

    for user_input in test_cases:
        print(f"\n{'─' * 60}")
        print(f"📝 用户输入: {user_input}")
        print(f"{'─' * 60}")

        try:
            result = await router.route(user_input)
            print(f"🤖 响应:\n{result}")
        except Exception as e:
            print(f"❌ 错误: {e}")


async def interactive_demo():
    """交互式演示"""
    print("=" * 60)
    print("多类型意图路由 - 交互模式")
    print("=" * 60)
    print("\n支持的意图类型:")
    print("  • Skills: 写博客、代码审查")
    print("  • Tools: 天气查询、新闻搜索、计算、翻译")
    print("  • Flows: 请假申请、费用报销")
    print("  • Message: 闲聊问答")
    print("\n输入 'quit' 退出\n")

    try:
        llm = ChatOpenAI(model="gpt-4o")
    except Exception as e:
        print(f"⚠️  无法创建 LLM: {e}")
        return

    router = IntentRouterWithPreMatch(llm)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见!")
                break
            if not user_input:
                continue

            result = await router.route(user_input)
            print(f"\n🤖 Assistant: {result}")

        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(demo())
