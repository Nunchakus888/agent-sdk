"""
Unit tests for workflow executors.

Tests cover:
- SkillExecutor (agent mode + function mode)
- FlowExecutor (API call)
- SystemExecutor (silent mode)
- TimerScheduler
- KBEnhancer
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from bu_agent_sdk.workflow.executors import (
    SkillExecutor,
    FlowExecutor,
    SystemExecutor,
    TimerScheduler,
    KBEnhancer,
)
from bu_agent_sdk.tools.action_books import (
    SkillDefinition,
    FlowDefinition,
    SystemAction,
    TimerConfig,
    WorkflowConfigSchema,
)
from bu_agent_sdk.agent.workflow_state import Session, WorkflowState


# =============================================================================
# Test SkillExecutor
# =============================================================================


@pytest.fixture
def skill_config():
    """Create test skill configuration."""
    return WorkflowConfigSchema(
        basic_settings={"name": "Test Agent"},
        skills=[
            SkillDefinition(
                skill_id="test_agent_skill",
                name="Test Agent Skill",
                description="Test skill in agent mode",
                execution_mode="agent",
                system_prompt="You are a test assistant",
                tools=["test_tool"],
                max_iterations=5,
                require_done_tool=True
            ),
            SkillDefinition(
                skill_id="test_function_skill",
                name="Test Function Skill",
                description="Test skill in function mode",
                execution_mode="function",
                endpoint={
                    "url": "http://test.example.com/api",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"input": "{input}"}
                },
                output_parser="json"
            )
        ]
    )


@pytest.mark.asyncio
async def test_skill_executor_agent_mode(skill_config):
    """Test SkillExecutor in agent mode."""
    mock_llm = AsyncMock()
    executor = SkillExecutor(skill_config, mock_llm)

    # Mock agent query
    with patch.object(executor, '_get_or_create_agent') as mock_get_agent:
        mock_agent = AsyncMock()
        mock_agent.query.return_value = "Agent completed the task"
        mock_get_agent.return_value = mock_agent

        result = await executor.execute(
            skill_id="test_agent_skill",
            user_request="Do something",
            parameters={}
        )

        assert "Test Agent Skill" in result
        assert "Completed" in result
        mock_agent.query.assert_called_once_with("Do something")


@pytest.mark.asyncio
async def test_skill_executor_function_mode(skill_config):
    """Test SkillExecutor in function mode."""
    mock_llm = AsyncMock()
    executor = SkillExecutor(skill_config, mock_llm)

    # Mock HTTP client
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {"status": "success", "data": "test"}
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = await executor.execute(
            skill_id="test_function_skill",
            user_request="Test input",
            parameters={}
        )

        assert "Test Function Skill" in result
        assert "Completed" in result
        mock_client.request.assert_called_once()


@pytest.mark.asyncio
async def test_skill_executor_not_found(skill_config):
    """Test SkillExecutor with non-existent skill."""
    mock_llm = AsyncMock()
    executor = SkillExecutor(skill_config, mock_llm)

    result = await executor.execute(
        skill_id="non_existent_skill",
        user_request="Test",
        parameters={}
    )

    assert "not found" in result.lower()


# =============================================================================
# Test FlowExecutor
# =============================================================================


@pytest.fixture
def flow_config():
    """Create test flow configuration."""
    return WorkflowConfigSchema(
        basic_settings={"name": "Test Agent"},
        flows=[
            FlowDefinition(
                flow_id="test_flow",
                name="Test Flow",
                description="Test flow",
                trigger_patterns=["test.*flow"],
                endpoint={
                    "url": "http://flow.example.com/api",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": {
                        "user_message": "{user_message}",
                        "session_id": "{session_id}"
                    }
                },
                response_template="✅ Flow completed: {result}"
            )
        ]
    )


@pytest.mark.asyncio
async def test_flow_executor_success(flow_config):
    """Test FlowExecutor successful execution."""
    executor = FlowExecutor(flow_config)

    # Create test session
    session = Session(
        session_id="test_session",
        agent_id="test_agent",
        workflow_state=WorkflowState(),
        messages=[]
    )

    # Mock HTTP client
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {"status": "completed"}
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = await executor.execute(
            flow_id="test_flow",
            user_message="Test message",
            parameters={},
            session=session
        )

        assert "Flow completed" in result
        mock_client.request.assert_called_once()


@pytest.mark.asyncio
async def test_flow_executor_not_found(flow_config):
    """Test FlowExecutor with non-existent flow."""
    executor = FlowExecutor(flow_config)
    session = Session(
        session_id="test_session",
        agent_id="test_agent",
        workflow_state=WorkflowState(),
        messages=[]
    )

    result = await executor.execute(
        flow_id="non_existent_flow",
        user_message="Test",
        parameters={},
        session=session
    )

    assert "not found" in result.lower()


# =============================================================================
# Test SystemExecutor
# =============================================================================


@pytest.fixture
def system_config():
    """Create test system action configuration."""
    return WorkflowConfigSchema(
        basic_settings={"name": "Test Agent"},
        system_actions=[
            SystemAction(
                action_id="test_handoff",
                name="Test Handoff",
                handler="handoff",
                silent=False,
                response_template="Transferring to human..."
            ),
            SystemAction(
                action_id="test_silent_action",
                name="Test Silent Action",
                handler="update_profile",
                silent=True
            )
        ]
    )


@pytest.mark.asyncio
async def test_system_executor_non_silent(system_config):
    """Test SystemExecutor non-silent action."""
    executor = SystemExecutor(system_config)

    result = await executor.execute(
        action_id="test_handoff",
        parameters={}
    )

    assert result is not None
    assert "Transferring" in result


@pytest.mark.asyncio
async def test_system_executor_silent(system_config):
    """Test SystemExecutor silent action."""
    executor = SystemExecutor(system_config)

    result = await executor.execute(
        action_id="test_silent_action",
        parameters={}
    )

    # Silent action should return None
    assert result is None


@pytest.mark.asyncio
async def test_system_executor_not_found(system_config):
    """Test SystemExecutor with non-existent action."""
    executor = SystemExecutor(system_config)

    result = await executor.execute(
        action_id="non_existent_action",
        parameters={}
    )

    assert "not found" in result.lower()


# =============================================================================
# Test TimerScheduler
# =============================================================================


@pytest.mark.asyncio
async def test_timer_scheduler():
    """Test TimerScheduler basic functionality."""
    mock_agent = AsyncMock()
    scheduler = TimerScheduler(mock_agent)

    # Create test timer
    timer = TimerConfig(
        timer_id="test_timer",
        delay_seconds=0.1,  # Short delay for testing
        action_type="system",
        action_target="test_action",
        message="Timer triggered"
    )

    # Schedule timer
    await scheduler.schedule("test_session", [timer])

    # Wait for timer to trigger
    await asyncio.sleep(0.2)

    # Verify agent.query was called
    mock_agent.query.assert_called_once()


@pytest.mark.asyncio
async def test_timer_scheduler_cancel():
    """Test TimerScheduler cancellation."""
    mock_agent = AsyncMock()
    scheduler = TimerScheduler(mock_agent)

    # Create test timer with longer delay
    timer = TimerConfig(
        timer_id="test_timer",
        delay_seconds=1.0,
        action_type="system",
        action_target="test_action"
    )

    # Schedule timer
    await scheduler.schedule("test_session", [timer])

    # Cancel immediately
    await scheduler.cancel_session_timers("test_session")

    # Wait a bit
    await asyncio.sleep(0.1)

    # Verify agent.query was NOT called
    mock_agent.query.assert_not_called()


# =============================================================================
# Test KBEnhancer
# =============================================================================


@pytest.fixture
def kb_config():
    """Create test KB configuration."""
    return WorkflowConfigSchema(
        basic_settings={"name": "Test Agent"},
        kb_config={
            "enabled": True,
            "tool_name": "search_kb",
            "auto_enhance": True,
            "enhance_conditions": ["skill", "tool"]
        }
    )


@pytest.mark.asyncio
async def test_kb_enhancer_query(kb_config):
    """Test KBEnhancer query functionality."""
    mock_kb_tool = AsyncMock()
    mock_kb_tool.execute.return_value = "KB result: relevant information"

    enhancer = KBEnhancer(kb_config, mock_kb_tool)

    result = await enhancer.query_kb("test query")

    assert "relevant information" in result
    mock_kb_tool.execute.assert_called_once_with(query="test query")


@pytest.mark.asyncio
async def test_kb_enhancer_disabled():
    """Test KBEnhancer when disabled."""
    config = WorkflowConfigSchema(
        basic_settings={"name": "Test Agent"},
        kb_config={"enabled": False}
    )

    mock_kb_tool = AsyncMock()
    enhancer = KBEnhancer(config, mock_kb_tool)

    result = await enhancer.query_kb("test query")

    assert result == ""
    mock_kb_tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_kb_enhancer_enhance(kb_config):
    """Test KBEnhancer enhance functionality."""
    mock_kb_tool = AsyncMock()
    mock_kb_tool.execute.return_value = "KB knowledge"

    enhancer = KBEnhancer(kb_config, mock_kb_tool)

    result = await enhancer.enhance(
        user_message="test query",
        action_type="skill",
        execution_result="Original result"
    )

    assert "Original result" in result
    assert "KB knowledge" in result
    assert "Related knowledge" in result


# =============================================================================
# Test Parameter Substitution
# =============================================================================


def test_parameter_substitution():
    """Test parameter substitution in templates."""
    from bu_agent_sdk.workflow.executors import SkillExecutor

    mock_llm = AsyncMock()
    config = WorkflowConfigSchema(basic_settings={})
    executor = SkillExecutor(config, mock_llm)

    # Test string substitution
    template = "Hello {name}, you are {age} years old"
    params = {"name": "Alice", "age": "25"}
    result = executor._substitute_parameters(template, params)
    assert result == "Hello Alice, you are 25 years old"

    # Test dict substitution
    template = {"greeting": "Hello {name}", "info": {"age": "{age}"}}
    result = executor._substitute_parameters(template, params)
    assert result == {"greeting": "Hello Alice", "info": {"age": "25"}}

    # Test list substitution
    template = ["Hello {name}", "Age: {age}"]
    result = executor._substitute_parameters(template, params)
    assert result == ["Hello Alice", "Age: 25"]
