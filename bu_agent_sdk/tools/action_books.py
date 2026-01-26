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
    """
    Complete workflow configuration schema.

    Separates fixed configuration (no LLM parsing) from LLM-parsed configuration.
    - Fixed config: kb_config (hardcoded pattern execution, appended to instructions)
    - LLM-parsed: instructions, skills, tools, flows, timers, need_greeting, constraints
    - Environment-driven: max_iterations, iteration_strategy

    Note: Knowledge base info is automatically appended to instructions after LLM parsing.
    """

    # =========================================================================
    # Fixed Configuration (no LLM parsing)
    # =========================================================================

    kb_config: dict = Field(
        default_factory=dict,
        description="Knowledge base config - fixed pattern execution (no LLM parsing)"
    )

    # =========================================================================
    # LLM-Parsed Configuration (enhanced through LLM)
    # =========================================================================

    # Backward compatibility
    instructions: str | None = Field(
        default=None,
        description="Workflow instructions (LLM-parsed, integrates basic_settings and extracts workflow). Replaces 'sop' field."
    )

    skills: list[dict] = Field(
        default_factory=list,
        description="Skills list (LLM-parsed) - structure: {condition, action, tools}"
    )

    tools: list[dict] = Field(
        default_factory=list,
        description="Tool config list (LLM-parsed) - includes both regular and silent tools"
    )

    flows: list[FlowDefinition] = Field(
        default_factory=list,
        description="Fixed flow definition list (LLM-parsed)"
    )

    timers: list[TimerConfig] = Field(
        default_factory=list,
        description="Timer task config (LLM-parsed)"
    )

    need_greeting: str = Field(
        default="",
        description="Greeting message if needed (LLM-parsed), empty if not needed"
    )

    # =========================================================================
    # Other Configuration
    # =========================================================================

    system_actions: list[SystemAction] = Field(
        default_factory=list,
        description="System-level actions (handoff, close, etc.)"
    )

    action_books: list[ActionBookEntry] = Field(
        default_factory=list,
        description="Intent matching rule list"
    )

    constraints: str | None = Field(
        default=None,
        description="Conversation constraints and boundary rules (LLM-inferred if necessary)"
    )

    # Iteration control config
    max_iterations: int = Field(
        default=5,
        description="Max iterations (from env var MAX_ITERATIONS or config)"
    )

    iteration_strategy: str = Field(
        default="sop_driven",
        description="Iteration strategy (from env var ITERATION_STRATEGY or config): sop_driven | single_shot"
    )
