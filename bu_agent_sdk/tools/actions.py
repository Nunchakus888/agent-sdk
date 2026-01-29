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
    # Primary identifier - supports both 'name' and 'flow_id' for backward compatibility
    name: str | None = Field(default=None, description="Flow name")
    flow_id: str | None = Field(default=None, description="Flow identifier (alternative to name)")
    description: str | None = Field(default=None, description="Flow description")

    # Trigger rules
    trigger_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for fast matching"
    )
    match_type: str | None = Field(
        default=None,
        description="Match type: exact | contains | regex"
    )

    # API endpoint config
    endpoint: dict | None = Field(
        default=None,
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

    def get_identifier(self) -> str:
        """Get the flow identifier (name or flow_id)."""
        return self.name or self.flow_id or "unknown"


class SystemAction(BaseModel):
    """System action definition."""
    action_id: str | None = Field(default=None, description="Action ID, defaults to name if not provided")
    name: str
    handler: str = Field(default=None, description="Handler type: handoff | close | update_profile")
    silent: bool = Field(
        default=False,
        description="Silent execution (no response to user)"
    )
    response_template: str | None = Field(
        default=None,
        description="Response template (non-silent)"
    )

    def model_post_init(self, __context) -> None:
        """Auto-set action_id from name if not provided."""
        if self.action_id is None:
            object.__setattr__(self, 'action_id', self.name)


class TimerConfig(BaseModel):
    """Timer configuration."""
    timer_id: str | None = Field(default=None, description="Timer ID, defaults to name if not provided")
    name: str | None = Field(default=None, description="Timer name (alternative to timer_id)")
    delay_seconds: int = Field(default=300, description="Delay in seconds before timer triggers")
    action_type: ActionType | str = Field(default=ActionType.SYSTEM, description="Action type when timer triggers")
    action_target: str = Field(default="send_message", description="Action target when timer triggers")
    message: str | None = Field(default=None, description="Timer message")

    def model_post_init(self, __context) -> None:
        """Auto-set timer_id from name if not provided."""
        if self.timer_id is None and self.name:
            object.__setattr__(self, 'timer_id', self.name)
        elif self.timer_id is None:
            object.__setattr__(self, 'timer_id', 'default_timer')


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
    - Fixed config: 
      - basic_settings
      - system_actions
      - agent_actions
      - skills
      - tools
      - flows
      (hardcoded pattern execution, appended to instructions)
    - LLM-parsed: 
      - instructions
      - timers
      - need_greeting
      - constraints
    - Environment-driven: max_iterations, iteration_strategy
    """

    # =========================================================================
    # Fixed Configuration (no LLM parsing)
    # =========================================================================

    basic_settings: dict = Field(
        default_factory=dict,
        description="Basic settings - name, description, background, language, tone, chatbot_id"
    )

    system_actions: list[str] | None = Field(
      default=None,
      description="System actions that can be executed by the system with concurrent execution"
    )

    agent_actions: list[str] | None = Field(
      default=None,
      description="Agent actions that can be executed by the agent with serial execution"
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

    # =========================================================================
    # LLM-Parsed Configuration (enhanced through LLM)
    # =========================================================================

    # Backward compatibility
    instructions: str | None = Field(
        default=None,
        description="Workflow instructions (LLM-parsed, integrates basic_settings and extracts workflow). Replaces 'sop' field."
    )


    timers: list[TimerConfig] = Field(
        default_factory=list,
        description="Timer task config (LLM-parsed)"
    )

    need_greeting: str = Field(
        default="",
        description="Greeting message if needed (LLM-parsed), empty if not needed"
    )


    constraints: str | None = Field(
        default=None,
        description="Conversation constraints and boundary rules (LLM-inferred if necessary)"
    )

    # =========================================================================
    # Knowledge Base Configuration
    # =========================================================================

    kb_config: dict = Field(
        default_factory=dict,
        description="Knowledge base configuration - enabled, retrieve_url, chatbot_id, auto_retrieve"
    )

    # =========================================================================
    # Other Configuration
    # =========================================================================

    # Iteration control config
    max_iterations: int = Field(
        default=5,
        description="Max iterations (from env var MAX_ITERATIONS or config)"
    )

    iteration_strategy: str = Field(
        default="sop_driven",
        description="Iteration strategy (from env var ITERATION_STRATEGY or config): sop_driven | single_shot"
    )
