"""
配置 Schema 模块

定义 Workflow Agent 的配置数据模型。
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Action types for workflow routing."""
    SKILL = "skill"      # Complex multi-step tasks
    TOOL = "tool"        # Single tool calls (includes system actions)
    FLOW = "flow"        # Fixed business process APIs
    GENERATE_RESPONSE = "generate_response"  # Response to user


class FlowType(str, Enum):
    """Flow type for matching strategy."""
    KEYWORD = "keyword"  # Code-based keyword matching (fast, deterministic)
    INTENT = "intent"    # LLM semantic matching (flexible, context-aware)


class MatchType(str, Enum):
    """Match type for keyword flows."""
    EXACT = "exact"        # Exact match
    CONTAINS = "contains"  # Contains match
    REGEX = "regex"        # Regex match


class FlowDefinition(BaseModel):
    """Flow definition - direct API call, black box service."""
    name: str | None = Field(default=None, description="Flow name")
    flow_id: str | None = Field(default=None, description="Flow identifier (alternative to name)")
    description: str | None = Field(default=None, description="Flow description")

    type: FlowType = Field(
        default=FlowType.INTENT,
        description="Flow type: keyword (code matching) or intent (LLM semantic matching)"
    )
    trigger_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns for keyword matching (used when type=keyword)"
    )
    match_type: MatchType = Field(
        default=MatchType.CONTAINS,
        description="Match type for keyword flows: exact | contains | regex"
    )

    endpoint: dict | None = Field(
        default=None,
        description="Flow execution HTTP endpoint config"
    )

    parameter_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter mapping: {api_param: user_message_field}"
    )

    response_template: str | None = Field(
        default=None,
        description="Response template, supports {result} placeholder"
    )

    def get_identifier(self) -> str:
        """Get the flow identifier (name or flow_id)."""
        return self.name or self.flow_id or "unknown"


class TimerConfig(BaseModel):
    """Timer configuration - unified Tool model."""
    timer_id: str = Field(description="Timer unique identifier")
    delay_seconds: int = Field(default=300, description="Delay in seconds before timer triggers")
    max_triggers: int = Field(default=1, description="Max trigger count, 0=unlimited")

    tool_name: str = Field(default="generate_response", description="Tool name to execute")
    tool_params: dict = Field(default_factory=dict, description="Tool parameters")
    message: str | None = Field(default=None, description="Message content (for generate_response)")

    def model_post_init(self, __context) -> None:
        """Validate timer_id."""
        if not self.timer_id:
            raise ValueError("timer_id is required")


class WorkflowConfigSchema(BaseModel):
    """
    Workflow Agent 配置 Schema。

    分离固定配置和 LLM 解析配置：
    - 固定配置: basic_settings, system_actions, agent_actions, skills, tools, flows
    - LLM 解析: instructions, timers, need_greeting, constraints
    """

    # Fixed Configuration
    basic_settings: dict = Field(
        default_factory=dict,
        description="Basic settings - name, description, background, language, tone, chatbot_id"
    )

    system_actions: list[str] | None = Field(
        default=None,
        description="System actions with concurrent execution"
    )

    agent_actions: list[str] | None = Field(
        default=None,
        description="Agent actions with serial execution"
    )

    skills: list[dict] = Field(
        default_factory=list,
        description="Skills list - structure: {condition, action, tools}"
    )

    action_books: list[dict] = Field(
        default_factory=list,
        description="Action book list - structure: {condition, action, tools}"
    )

    tools: list[dict] = Field(
        default_factory=list,
        description="Tool config list"
    )

    flows: list[FlowDefinition] = Field(
        default_factory=list,
        description="Fixed flow definitions"
    )

    # LLM-Parsed Configuration
    instructions: str | None = Field(
        default=None,
        description="Workflow instructions (replaces 'sop' field)"
    )

    timers: list[TimerConfig] = Field(
        default_factory=list,
        description="Timer task config"
    )

    need_greeting: str = Field(
        default="",
        description="Greeting message if needed, empty if not"
    )

    constraints: str | None = Field(
        default=None,
        description="Conversation constraints and boundary rules"
    )

    # Knowledge Base
    retrieve_knowledge_url: str | None = Field(
        default=None,
        description="Knowledge base URL"
    )

    # Runtime
    max_iterations: int = Field(default=5, description="Max iterations")
    iteration_strategy: str = Field(default="sop_driven", description="Iteration strategy")


__all__ = [
    "ActionType",
    "FlowType",
    "MatchType",
    "FlowDefinition",
    "TimerConfig",
    "WorkflowConfigSchema",
]
