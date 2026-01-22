"""
Integration tests for WorkflowAgent.

Tests cover:
- Complete workflow execution
- SOP-driven iteration
- Silent action handling
- KB parallel query
- Session management
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.tools.action_books import (
    WorkflowConfigSchema,
    SkillDefinition,
    FlowDefinition,
    SystemAction,
)


@pytest.fixture
def test_config():
    """Create comprehensive test configuration."""
    return WorkflowConfigSchema(
        basic_settings={
            "name": "Test Workflow Agent",
            "description": "Test agent for integration testing",
            "language": "English",
            "tone": "Professional"
        },
        greeting="Hello! How can I help you?",
        sop="1. Understand user needs\n2. Execute appropriate action\n3. Provide feedback",
        constraints="Be helpful and accurate",
        tools=[
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                },
                "endpoint": {
                    "url": "http://test.example.com/tool",
                    "method": "POST",
                    "body": {"data": "{input}"}
                }
            }
        ],
        skills=[
            SkillDefinition(
                skill_id="test_skill",
                name="Test Skill",
                description="Test skill for testing",
                execution_mode="agent",
                system_prompt="You are a test assistant",
                tools=["test_tool"],
                max_iterations=5
            )
        ],
        flows=[
            FlowDefinition(
                flow_id="test_flow",
                name="Test Flow",
                description="Test flow",
                trigger_patterns=[r"test flow"],
                endpoint={
                    "url": "http://test.example.com/flow",
                    "method": "POST"
                }
            )
        ],
        system_actions=[
            SystemAction(
                action_id="test_silent",
                name="Test Silent Action",
                handler="update_profile",
                silent=True
            ),
            SystemAction(
                action_id="test_non_silent",
                name="Test Non-Silent Action",
                handler="handoff",
                silent=False,
                response_template="Action completed"
            )
        ],
        kb_config={
            "enabled": True,
            "tool_name": "test_tool",
            "auto_enhance": True
        },
        max_iterations=3,
        iteration_strategy="sop_driven"
    )


@pytest.mark.asyncio
async def test_workflow_agent_greeting(test_config):
    """Test greeting message on first interaction."""
    mock_llm = AsyncMock()
    agent = WorkflowAgent(test_config, mock_llm)

    response = await agent.query(
        message="Hello",
        session_id="test_session_1"
    )

    assert response == "Hello! How can I help you?"


@pytest.mark.asyncio
async def test_workflow_agent_rule_matching(test_config):
    """Test rule-based flow matching."""
    mock_llm = AsyncMock()
    agent = WorkflowAgent(test_config, mock_llm)

    # First call to consume greeting
    await agent.query(message="Hi", session_id="test_session_2")

    # Mock HTTP client for flow execution
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.json.return_value = {"status": "success"}
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        response = await agent.query(
            message="test flow please",
            session_id="test_session_2"
        )

        assert "Test Flow" in response
        mock_client.request.assert_called_once()


@pytest.mark.asyncio
async def test_workflow_agent_sop_driven_execution(test_config):
    """Test SOP-driven multi-step execution."""
    mock_llm = AsyncMock()

    # Mock LLM decision responses
    decision_responses = [
        # First iteration: decide to execute tool
        Mock(content='''
        {
            "should_continue": true,
            "should_respond": false,
            "next_action": {
                "type": "tool",
                "target": "test_tool",
                "params": {"input": "test"}
            },
            "reasoning": "Execute tool first"
        }
        '''),
        # Second iteration: decide to respond
        Mock(content='''
        {
            "should_continue": false,
            "should_respond": true,
            "next_action": null,
            "reasoning": "Task completed"
        }
        ''')
    ]

    # Mock final response generation
    final_response = Mock(content="Task completed successfully")

    mock_llm.ainvoke.side_effect = decision_responses + [final_response]

    agent = WorkflowAgent(test_config, mock_llm)

    # Consume greeting
    await agent.query(message="Hi", session_id="test_session_3")

    # Mock tool execution
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.text = "Tool result"
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        response = await agent.query(
            message="Do something",
            session_id="test_session_3"
        )

        assert "completed" in response.lower()
        # Verify multiple LLM calls (decisions + final response)
        assert mock_llm.ainvoke.call_count >= 2


@pytest.mark.asyncio
async def test_workflow_agent_silent_action(test_config):
    """Test silent action exits iteration immediately."""
    mock_llm = AsyncMock()

    # Mock LLM decision to execute silent action
    decision_response = Mock(content='''
    {
        "should_continue": true,
        "should_respond": false,
        "next_action": {
            "type": "system",
            "target": "test_silent",
            "params": {}
        },
        "reasoning": "Execute silent action"
    }
    ''')

    # Mock final response
    final_response = Mock(content="Action executed silently")

    mock_llm.ainvoke.side_effect = [decision_response, final_response]

    agent = WorkflowAgent(test_config, mock_llm)

    # Consume greeting
    await agent.query(message="Hi", session_id="test_session_4")

    response = await agent.query(
        message="Execute silent action",
        session_id="test_session_4"
    )

    # Should generate response after silent action
    assert response is not None
    # Should only have 2 LLM calls (decision + response), not continue iteration
    assert mock_llm.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_workflow_agent_max_iterations(test_config):
    """Test max iterations limit."""
    mock_llm = AsyncMock()

    # Mock LLM to always want to continue
    decision_response = Mock(content='''
    {
        "should_continue": true,
        "should_respond": false,
        "next_action": {
            "type": "tool",
            "target": "test_tool",
            "params": {"input": "test"}
        },
        "reasoning": "Continue"
    }
    ''')

    final_response = Mock(content="Max iterations reached")

    # Return decision for max_iterations times, then final response
    mock_llm.ainvoke.side_effect = [decision_response] * test_config.max_iterations + [final_response]

    agent = WorkflowAgent(test_config, mock_llm)

    # Consume greeting
    await agent.query(message="Hi", session_id="test_session_5")

    # Mock tool execution
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.text = "Tool result"
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        response = await agent.query(
            message="Keep going",
            session_id="test_session_5"
        )

        # Should stop after max_iterations
        assert response is not None
        # Verify LLM was called max_iterations times for decisions + 1 for final response
        assert mock_llm.ainvoke.call_count == test_config.max_iterations + 1


@pytest.mark.asyncio
async def test_workflow_agent_config_change_detection(test_config):
    """Test config change detection and state reset."""
    mock_llm = AsyncMock()
    agent = WorkflowAgent(test_config, mock_llm)

    # First interaction
    response1 = await agent.query(message="Hi", session_id="test_session_6")
    assert response1 == test_config.greeting

    # Get session
    session = agent._sessions["test_session_6"]
    assert session.workflow_state.need_greeting is False

    # Change config hash to simulate config change
    old_hash = agent.config_hash
    agent.config_hash = "new_hash_value"

    # Next interaction should reset state
    response2 = await agent.query(message="Hi again", session_id="test_session_6")

    # State should be reset
    session = agent._sessions["test_session_6"]
    assert session.workflow_state.config_hash == "new_hash_value"


@pytest.mark.asyncio
async def test_workflow_agent_session_isolation(test_config):
    """Test that different sessions are isolated."""
    mock_llm = AsyncMock()
    agent = WorkflowAgent(test_config, mock_llm)

    # Session 1
    response1 = await agent.query(message="Hi", session_id="session_1")
    assert response1 == test_config.greeting

    # Session 2 should also get greeting
    response2 = await agent.query(message="Hi", session_id="session_2")
    assert response2 == test_config.greeting

    # Session 1 should not get greeting again
    mock_llm.ainvoke.return_value = Mock(content='''
    {
        "should_continue": false,
        "should_respond": true,
        "next_action": null,
        "reasoning": "Just respond"
    }
    ''')

    response3 = await agent.query(message="Hi again", session_id="session_1")
    assert response3 != test_config.greeting


@pytest.mark.asyncio
async def test_workflow_agent_single_shot_mode(test_config):
    """Test single-shot execution mode."""
    # Change to single-shot mode
    test_config.iteration_strategy = "single_shot"

    mock_llm = AsyncMock()

    # Mock intent matcher response
    mock_response = Mock(content='''
    {
        "matched": true,
        "action_type": "tool",
        "action_target": "test_tool",
        "parameters": {"input": "test"},
        "confidence": 0.9,
        "reasoning": "User wants to use tool"
    }
    ''')
    mock_llm.ainvoke.return_value = mock_response

    agent = WorkflowAgent(test_config, mock_llm)

    # Consume greeting
    await agent.query(message="Hi", session_id="test_session_7")

    # Mock tool execution
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.text = "Tool executed"
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        response = await agent.query(
            message="Use the tool",
            session_id="test_session_7"
        )

        assert "test_tool" in response
        # In single-shot mode, should only call LLM once for intent matching
        assert mock_llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_workflow_agent_kb_parallel_query(test_config):
    """Test KB parallel query optimization."""
    mock_llm = AsyncMock()

    # Mock decision to respond immediately
    decision_response = Mock(content='''
    {
        "should_continue": false,
        "should_respond": true,
        "next_action": null,
        "reasoning": "Respond with KB info"
    }
    ''')

    final_response = Mock(content="Response with KB knowledge")

    mock_llm.ainvoke.side_effect = [decision_response, final_response]

    agent = WorkflowAgent(test_config, mock_llm)

    # Consume greeting
    await agent.query(message="Hi", session_id="test_session_8")

    # Mock KB tool
    mock_kb_tool = AsyncMock()
    mock_kb_tool.execute.return_value = "KB result"
    agent.kb_enhancer.kb_tool = mock_kb_tool

    response = await agent.query(
        message="Tell me about something",
        session_id="test_session_8"
    )

    # KB should be queried
    mock_kb_tool.execute.assert_called_once()
    assert response is not None
