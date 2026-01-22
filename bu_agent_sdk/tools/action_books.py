"""
Action Books - Workflow data models and configuration schema.

Based on workflow-agent-v9.md design.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Action types for workflow routing."""
    SKILL = "skill"      # Complex multi-step tasks
    TOOL = "tool"        # Single tool calls
    FLOW = "flow"        # Fixed business process APIs
    SYSTEM = "system"    # System actions (handoff, close, etc.)


class ActionBookEntry(BaseModel):
    """Intent to action mapping rule."""
    condition: str = Field(description="Natural language condition that triggers this action")
    action_type: ActionType
    action_target: str = Field(description="skill_id | tool_name | flow_id | system_action")
    parameters: dict = Field(default_factory=dict, description="Default parameters")
    priority: int = Field(default=0, description="Higher priority matched first")


class SkillExecutionMode(str, Enum):
    """Skill execution modes."""
    AGENT = "agent"          # Sub-agent execution (multi-turn)
    FUNCTION = "function"    # External function call (single I/O)


class SkillDefinition(BaseModel):
    """Skill definition - supports multiple execution modes."""
    skill_id: str
    name: str
    description: str

    # Execution mode
    execution_mode: SkillExecutionMode = Field(
        default=SkillExecutionMode.AGENT,
        description="Execution mode: agent or function"
    )

    # Agent mode config
    system_prompt: str | None = Field(
        default=None,
        description="System prompt (required for agent mode)"
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Available tool names (agent mode)"
    )
    max_iterations: int = Field(
        default=20,
        description="Max iterations (agent mode)"
    )
    require_done_tool: bool = Field(
        default=True,
        description="Require explicit completion (agent mode)"
    )

    # Function mode config
    endpoint: dict | None = Field(
        default=None,
        description="HTTP endpoint config (required for function mode)"
    )
    input_schema: dict | None = Field(
        default=None,
        description="Input parameter schema (function mode)"
    )
    output_parser: str | None = Field(
        default=None,
        description="Output parser type: json | text (function mode)"
    )


class FlowDefinition(BaseModel):
    """Flow definition - direct API call, black box service."""
    flow_id: str
    name: str
    description: str

    # Trigger rules
    trigger_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for fast matching"
    )

    # API endpoint config
    endpoint: dict = Field(
        description="Flow execution HTTP endpoint config"
    )

    # Parameter mapping
    parameter_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter mapping: {api_param: user_message_field}"
    )

    # Response handling
    response_template: str | None = Field(
        default=None,
        description="Response template, supports {result} placeholder"
    )


class SystemAction(BaseModel):
    """System action definition."""
    action_id: str
    name: str
    handler: str = Field(description="Handler type: handoff | close | update_profile")
    silent: bool = Field(
        default=False,
        description="Silent execution (no response to user)"
    )
    response_template: str | None = Field(
        default=None,
        description="Response template (non-silent)"
    )


class TimerConfig(BaseModel):
    """Timer configuration."""
    timer_id: str
    delay_seconds: int
    action_type: ActionType
    action_target: str
    message: str | None = Field(default=None, description="Timer message")


class IterationDecision(BaseModel):
    """Iteration decision result - LLM-driven decision."""
    should_continue: bool = Field(
        description="Whether to continue executing next action"
    )
    should_respond: bool = Field(
        description="Whether to generate final response to user"
    )
    next_action: dict | None = Field(
        default=None,
        description="Next action to execute: {type, target, params}"
    )
    reasoning: str = Field(
        description="Decision reasoning (for debugging and logging)"
    )


class WorkflowConfigSchema(BaseModel):
    """Complete workflow configuration schema."""

    # Basic settings
    basic_settings: dict = Field(
        description="Name, language, tone, etc.",
        default_factory=dict
    )

    # Tool definitions (HTTP API)
    tools: list[dict] = Field(
        default_factory=list,
        description="HTTP tool config list (reuse ConfigToolLoader)"
    )

    # Skill definitions
    skills: list[SkillDefinition] = Field(
        default_factory=list,
        description="Complex multi-step skill list"
    )

    # Flow definitions
    flows: list[FlowDefinition] = Field(
        default_factory=list,
        description="Fixed flow definition list"
    )

    # System actions
    system_actions: list[SystemAction] = Field(
        default_factory=list,
        description="System-level actions (handoff, close, etc.)"
    )

    # Intent mapping rules
    action_books: list[ActionBookEntry] = Field(
        default_factory=list,
        description="Intent matching rule list"
    )

    # Timers
    timers: list[TimerConfig] = Field(
        default_factory=list,
        description="Timer task config"
    )

    # Greeting
    greeting: str | None = Field(
        default=None,
        description="Greeting message for first conversation"
    )

    # SOP (optional)
    sop: str | None = Field(
        default=None,
        description="Standard Operating Procedure description"
    )

    # Constraints (optional)
    constraints: str | None = Field(
        default=None,
        description="Conversation constraints and boundary rules"
    )

    # KB config (optional)
    kb_config: dict | None = Field(
        default=None,
        description="Knowledge base config: enable KB enhancement, KB tool name, etc."
    )

    # Iteration control config
    max_iterations: int = Field(
        default=5,
        description="Max iterations (prevent infinite loop)"
    )

    iteration_strategy: str = Field(
        default="sop_driven",
        description="Iteration strategy: sop_driven (multi-step) | single_shot (single execution)"
    )
