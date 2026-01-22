"""
Unit tests for intent matcher.

Tests cover:
- Rule-based matching (Flow patterns)
- LLM-based matching
- Hybrid matching strategy
"""

import pytest
from unittest.mock import AsyncMock, Mock

from bu_agent_sdk.tools.intent_matcher import IntentMatcher, IntentMatchResult
from bu_agent_sdk.tools.action_books import (
    ActionType,
    FlowDefinition,
    SkillDefinition,
    WorkflowConfigSchema,
)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return WorkflowConfigSchema(
        basic_settings={"name": "Test Agent"},
        flows=[
            FlowDefinition(
                flow_id="leave_request",
                name="Leave Request",
                description="Employee leave request flow",
                trigger_patterns=[
                    r"我要请假",
                    r"申请.*假",
                    r"请.*天假"
                ],
                endpoint={
                    "url": "http://api.example.com/leave",
                    "method": "POST"
                }
            )
        ],
        skills=[
            SkillDefinition(
                skill_id="blog_writer",
                name="Blog Writer",
                description="Write blog articles",
                execution_mode="agent",
                system_prompt="You are a blog writer"
            )
        ],
        tools=[
            {
                "name": "search_weather",
                "description": "Search weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                },
                "endpoint": {
                    "url": "http://api.example.com/weather",
                    "method": "GET"
                }
            }
        ],
        action_books=[]
    )


@pytest.mark.asyncio
async def test_rule_matching_flow(test_config):
    """Test rule-based matching for flows."""
    mock_llm = AsyncMock()
    matcher = IntentMatcher(test_config, mock_llm)

    # Test pattern matching
    result = await matcher.match("我要请假3天", [])

    assert result.matched is True
    assert result.action_type == ActionType.FLOW
    assert result.action_target == "leave_request"
    assert result.confidence == 1.0
    assert "Pattern matched" in result.reasoning


@pytest.mark.asyncio
async def test_rule_matching_multiple_patterns(test_config):
    """Test rule matching with multiple patterns."""
    mock_llm = AsyncMock()
    matcher = IntentMatcher(test_config, mock_llm)

    # Test different patterns
    test_cases = [
        "我要请假",
        "申请年假",
        "请3天假"
    ]

    for message in test_cases:
        result = await matcher.match(message, [])
        assert result.matched is True
        assert result.action_type == ActionType.FLOW
        assert result.action_target == "leave_request"


@pytest.mark.asyncio
async def test_llm_matching_skill(test_config):
    """Test LLM-based matching for skills."""
    mock_llm = AsyncMock()

    # Mock LLM response
    mock_response = Mock()
    mock_response.content = '''
    {
        "matched": true,
        "action_type": "skill",
        "action_target": "blog_writer",
        "parameters": {},
        "confidence": 0.9,
        "reasoning": "User wants to write a blog"
    }
    '''
    mock_llm.ainvoke.return_value = mock_response

    matcher = IntentMatcher(test_config, mock_llm)

    result = await matcher.match("Help me write a blog about AI", [])

    assert result.matched is True
    assert result.action_type == ActionType.SKILL
    assert result.action_target == "blog_writer"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_llm_matching_tool(test_config):
    """Test LLM-based matching for tools."""
    mock_llm = AsyncMock()

    # Mock LLM response
    mock_response = Mock()
    mock_response.content = '''
    {
        "matched": true,
        "action_type": "tool",
        "action_target": "search_weather",
        "parameters": {"city": "Beijing"},
        "confidence": 0.95,
        "reasoning": "User wants to check weather"
    }
    '''
    mock_llm.ainvoke.return_value = mock_response

    matcher = IntentMatcher(test_config, mock_llm)

    result = await matcher.match("What's the weather in Beijing?", [])

    assert result.matched is True
    assert result.action_type == ActionType.TOOL
    assert result.action_target == "search_weather"
    assert result.parameters == {"city": "Beijing"}


@pytest.mark.asyncio
async def test_llm_matching_message(test_config):
    """Test LLM-based matching for general messages."""
    mock_llm = AsyncMock()

    # Mock LLM response for general chat
    mock_response = Mock()
    mock_response.content = '''
    {
        "matched": true,
        "action_type": "message",
        "action_target": "",
        "parameters": {},
        "confidence": 0.8,
        "reasoning": "User is just chatting"
    }
    '''
    mock_llm.ainvoke.return_value = mock_response

    matcher = IntentMatcher(test_config, mock_llm)

    result = await matcher.match("Hello, how are you?", [])

    assert result.matched is True
    assert result.action_type == "message"
    assert result.action_target == ""


@pytest.mark.asyncio
async def test_hybrid_matching_priority(test_config):
    """Test that rule matching has priority over LLM matching."""
    mock_llm = AsyncMock()

    # Even if LLM would return something, rule matching should win
    mock_response = Mock()
    mock_response.content = '''
    {
        "matched": true,
        "action_type": "skill",
        "action_target": "blog_writer",
        "parameters": {},
        "confidence": 0.9,
        "reasoning": "Test"
    }
    '''
    mock_llm.ainvoke.return_value = mock_response

    matcher = IntentMatcher(test_config, mock_llm)

    # This should match the flow pattern first
    result = await matcher.match("我要请假", [])

    assert result.action_type == ActionType.FLOW
    assert result.action_target == "leave_request"
    # LLM should not be called
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_llm_matching_with_context(test_config):
    """Test LLM matching with conversation context."""
    mock_llm = AsyncMock()

    mock_response = Mock()
    mock_response.content = '''
    {
        "matched": true,
        "action_type": "tool",
        "action_target": "search_weather",
        "parameters": {},
        "confidence": 0.85,
        "reasoning": "Following up on previous question"
    }
    '''
    mock_llm.ainvoke.return_value = mock_response

    matcher = IntentMatcher(test_config, mock_llm)

    # Provide context
    context = [
        Mock(content="What's the weather like?"),
        Mock(content="In Beijing")
    ]

    result = await matcher.match("How about tomorrow?", context)

    assert result.matched is True
    # Verify context was passed to LLM
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_invalid_regex_pattern():
    """Test handling of invalid regex patterns."""
    config = WorkflowConfigSchema(
        basic_settings={"name": "Test Agent"},
        flows=[
            FlowDefinition(
                flow_id="test_flow",
                name="Test Flow",
                description="Test",
                trigger_patterns=[
                    r"valid_pattern",
                    r"[invalid(pattern"  # Invalid regex
                ],
                endpoint={"url": "http://test.com", "method": "POST"}
            )
        ]
    )

    mock_llm = AsyncMock()
    # Should not raise exception, just log warning
    matcher = IntentMatcher(config, mock_llm)

    # Valid pattern should still work
    result = await matcher.match("valid_pattern test", [])
    assert result.action_type == ActionType.FLOW


@pytest.mark.asyncio
async def test_llm_parse_error(test_config):
    """Test handling of LLM parse errors."""
    mock_llm = AsyncMock()

    # Mock invalid JSON response
    mock_response = Mock()
    mock_response.content = "This is not valid JSON"
    mock_llm.ainvoke.return_value = mock_response

    matcher = IntentMatcher(test_config, mock_llm)

    result = await matcher.match("Test message", [])

    # Should return default result on parse error
    assert result.matched is False
    assert result.action_type is None
    assert result.confidence == 0.0
    assert "Parse error" in result.reasoning


def test_build_matching_prompt(test_config):
    """Test prompt building for LLM matching."""
    mock_llm = AsyncMock()
    matcher = IntentMatcher(test_config, mock_llm)

    prompt = matcher._build_matching_prompt()

    # Verify prompt contains all necessary information
    assert "Skills" in prompt
    assert "blog_writer" in prompt
    assert "Tools" in prompt
    assert "search_weather" in prompt
    assert "Output Format" in prompt
    assert "JSON" in prompt
