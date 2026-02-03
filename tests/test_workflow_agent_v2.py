"""
WorkflowAgentV2 单元测试

测试覆盖：
1. 基于 Agent 的统一 LLM 交互
2. Tool 转换（HTTP tools, skills, flows, system actions）
3. System prompt 构建
4. Token 使用统计
5. 流式响应

运行测试：
    pytest tests/test_workflow_agent_v2.py -v -s
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from bu_agent_sdk.agent.workflow_agent_v2 import (
    WorkflowAgentV2,
    _create_http_tool_wrapper,
    _create_skill_tool,
    _create_flow_tool,
    _create_system_tool,
)
from bu_agent_sdk.agent.service import TaskComplete
from bu_agent_sdk.tools.actions import (
    WorkflowConfigSchema,
    FlowDefinition,
)
from bu_agent_sdk.llm.base import ToolDefinition
from bu_agent_sdk.llm.messages import ToolCall, Function
from bu_agent_sdk.llm.views import ChatInvokeCompletion


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """创建 Mock LLM"""
    llm = MagicMock()
    llm.model = "gpt-4o"
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def basic_config():
    """创建基础配置"""
    return WorkflowConfigSchema(
        basic_settings={
            "name": "Test Agent",
            "description": "A test workflow agent",
            "language": "English",
            "tone": "professional",
        },
        instructions="Follow the SOP step by step.",
        constraints="Be helpful and concise.",
        max_iterations=5,
        iteration_strategy="sop_driven",
    )


@pytest.fixture
def config_with_tools():
    """创建带 tools 的配置"""
    return WorkflowConfigSchema(
        basic_settings={"name": "Tool Agent", "tone": "friendly"},
        instructions="Use tools to help users.",
        tools=[
            {
                "name": "save_customer_info",
                "description": "Save customer information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "name": {"type": "string"},
                    },
                    "required": ["email"],
                },
                "endpoint": {
                    "url": "http://api.example.com/save",
                    "method": "POST",
                },
            },
        ],
        max_iterations=5,
    )


@pytest.fixture
def config_with_all_actions():
    """创建包含所有 action 类型的配置"""
    return WorkflowConfigSchema(
        basic_settings={"name": "Full Agent"},
        instructions="Handle all types of actions.",
        tools=[
            {
                "name": "weather_tool",
                "description": "Get weather information",
                "parameters": {"type": "object", "properties": {}},
                "endpoint": {"url": "http://api.weather.com/get", "method": "GET"},
            },
        ],
        skills=[
            {
                "skill_id": "customer_service",
                "name": "Customer Service",
                "description": "Handle customer service requests",
            },
        ],
        flows=[
            FlowDefinition(
                flow_id="order_process",
                name="Order Process",
                description="Process customer orders",
            ),
        ],
        system_actions=["handoff"],
        max_iterations=5,
    )


# =============================================================================
# 测试 Tool 创建函数
# =============================================================================


class TestToolCreation:
    """测试 Tool 创建函数"""

    def test_create_skill_tool(self):
        """验证 skill tool 创建"""
        tool = _create_skill_tool("test_skill", "Test skill description", None)

        assert tool.name == "skill__test_skill"
        assert "Test skill description" in tool.description

    def test_create_flow_tool(self):
        """验证 flow tool 创建"""
        tool = _create_flow_tool("test_flow", "Test flow description", None)

        assert tool.name == "flow__test_flow"
        assert "Test flow description" in tool.description

    def test_create_system_tool(self):
        """验证 system tool 创建"""
        tool = _create_system_tool("handoff", "Transfer to Human", False)

        assert tool.name == "system__handoff"
        assert "Transfer to Human" in tool.description

    @pytest.mark.asyncio
    async def test_system_tool_handoff_raises_task_complete(self):
        """验证 handoff 系统动作抛出 TaskComplete"""
        tool = _create_system_tool("handoff", "Transfer", False)

        with pytest.raises(TaskComplete) as exc_info:
            await tool.execute(reason="User requested")

        assert "Transferred to human agent" in str(exc_info.value)


# =============================================================================
# 测试 WorkflowAgentV2 初始化
# =============================================================================


class TestWorkflowAgentV2Init:
    """测试 WorkflowAgentV2 初始化"""

    def test_init_with_basic_config(self, mock_llm, basic_config):
        """验证基础配置初始化"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)

            assert agent.config == basic_config
            assert agent._llm == mock_llm

    def test_init_builds_system_prompt(self, mock_llm, basic_config):
        """验证 system prompt 构建"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)

            assert "Test Agent" in agent._system_prompt
            assert "Follow the SOP step by step" in agent._system_prompt
            assert "Be helpful and concise" in agent._system_prompt

    def test_init_builds_workflow_tools(self, mock_llm, config_with_all_actions):
        """验证 workflow tools 构建"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=config_with_all_actions, llm=mock_llm)

            tool_names = [t.name for t in agent._workflow_tools]

            # 验证所有类型的 tools 都被创建
            assert "weather_tool" in tool_names
            assert "skill__customer_service" in tool_names
            assert "flow__order_process" in tool_names
            assert "system__handoff" in tool_names

            print(f"\n[Workflow Tools] {tool_names}")


# =============================================================================
# 测试 System Prompt 构建
# =============================================================================


class TestSystemPromptBuilding:
    """测试 System Prompt 构建"""

    def test_prompt_includes_agent_profile(self, mock_llm, basic_config):
        """验证 prompt 包含 agent profile"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)

            assert "## Agent Profile" in agent._system_prompt
            assert "Test Agent" in agent._system_prompt
            assert "professional" in agent._system_prompt

    def test_prompt_includes_sop_instructions(self, mock_llm, basic_config):
        """验证 prompt 包含 SOP instructions"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)

            assert "## SOP Instructions" in agent._system_prompt
            assert "Follow the SOP step by step" in agent._system_prompt

    def test_prompt_includes_constraints(self, mock_llm, basic_config):
        """验证 prompt 包含 constraints"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)

            assert "## Constraints" in agent._system_prompt
            assert "Be helpful and concise" in agent._system_prompt

    def test_prompt_includes_available_actions(self, mock_llm, config_with_all_actions):
        """验证 prompt 包含可用 actions"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=config_with_all_actions, llm=mock_llm)

            assert "## Available Actions" in agent._system_prompt
            assert "weather_tool" in agent._system_prompt
            assert "customer_service" in agent._system_prompt
            assert "order_process" in agent._system_prompt
            assert "handoff" in agent._system_prompt

    def test_prompt_includes_response_guidelines(self, mock_llm, basic_config):
        """验证 prompt 包含响应指南"""
        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent"):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)

            assert "## Response Guidelines" in agent._system_prompt
            assert "Follow the SOP step by step" in agent._system_prompt


# =============================================================================
# 测试 Query 方法
# =============================================================================


class TestQuery:
    """测试 query 方法"""

    @pytest.mark.asyncio
    async def test_query_calls_agent(self, mock_llm, basic_config):
        """验证 query 调用内部 Agent"""
        mock_agent = MagicMock()
        mock_agent.query = AsyncMock(return_value="Hello! How can I help?")

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)
            agent._agent = mock_agent

            result = await agent.query(
                message="Hello",
                session_id="test_session",
            )

            assert result == "Hello! How can I help?"
            mock_agent.query.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_query_with_context(self, mock_llm, basic_config):
        """验证 query 支持注入 context"""
        mock_agent = MagicMock()
        mock_agent.query = AsyncMock(return_value="I remember our conversation.")
        mock_agent.load_history = MagicMock()

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)
            agent._agent = mock_agent

            context = [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]

            result = await agent.query(
                message="Remember me?",
                session_id="test_session",
                context=context,
            )

            # 验证 load_history 被调用
            mock_agent.load_history.assert_called_once()
            assert result == "I remember our conversation."

    @pytest.mark.asyncio
    async def test_query_handles_task_complete(self, mock_llm, basic_config):
        """验证 query 处理 TaskComplete 异常"""
        mock_agent = MagicMock()
        mock_agent.query = AsyncMock(side_effect=TaskComplete("Transferred to human"))

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)
            agent._agent = mock_agent

            result = await agent.query(
                message="Transfer me",
                session_id="test_session",
            )

            assert result == "Transferred to human"


# =============================================================================
# 测试 Usage 统计
# =============================================================================


class TestUsageStatistics:
    """测试 usage 统计"""

    @pytest.mark.asyncio
    async def test_get_usage(self, mock_llm, basic_config):
        """验证 get_usage 方法"""
        mock_usage = MagicMock()
        mock_usage.total_tokens = 1000
        mock_usage.total_cost = 0.01

        mock_agent = MagicMock()
        mock_agent.get_usage = AsyncMock(return_value=mock_usage)

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)
            agent._agent = mock_agent

            usage = await agent.get_usage()

            assert usage.total_tokens == 1000
            assert usage.total_cost == 0.01


# =============================================================================
# 测试 History 管理
# =============================================================================


class TestHistoryManagement:
    """测试 history 管理"""

    def test_clear_history(self, mock_llm, basic_config):
        """验证 clear_history 方法"""
        mock_agent = MagicMock()

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)
            agent._agent = mock_agent

            agent.clear_history()

            mock_agent.clear_history.assert_called_once()

    def test_messages_property(self, mock_llm, basic_config):
        """验证 messages 属性"""
        mock_messages = [MagicMock(), MagicMock()]
        mock_agent = MagicMock()
        mock_agent.messages = mock_messages

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)
            agent._agent = mock_agent

            assert agent.messages == mock_messages


# =============================================================================
# 集成测试
# =============================================================================


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow_with_tool_call(self, mock_llm, config_with_tools):
        """测试完整的工作流（包含 tool call）"""
        # 模拟 Agent 的响应
        mock_agent = MagicMock()
        mock_agent.query = AsyncMock(return_value="I've saved your information successfully.")

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=config_with_tools, llm=mock_llm)
            agent._agent = mock_agent

            result = await agent.query(
                message="Save my email: test@example.com",
                session_id="test_session",
            )

            assert "saved" in result.lower() or "information" in result.lower()
            print(f"\n[Integration Test] Result: {result}")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_llm, basic_config):
        """测试多轮对话"""
        mock_agent = MagicMock()
        mock_agent.query = AsyncMock(side_effect=[
            "Hello! How can I help you today?",
            "Sure, I can help with that.",
            "Is there anything else?",
        ])

        with patch("bu_agent_sdk.agent.workflow_agent_v2.Agent", return_value=mock_agent):
            agent = WorkflowAgentV2(config=basic_config, llm=mock_llm)
            agent._agent = mock_agent

            # 多轮对话
            r1 = await agent.query("Hello", "sess_1")
            r2 = await agent.query("Help me", "sess_1")
            r3 = await agent.query("Thanks", "sess_1")

            assert "Hello" in r1
            assert "help" in r2.lower()
            assert mock_agent.query.call_count == 3

            print(f"\n[Multi-turn] R1: {r1}")
            print(f"[Multi-turn] R2: {r2}")
            print(f"[Multi-turn] R3: {r3}")
