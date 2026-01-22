"""
Workflow executors for handling different types of actions.

Design Philosophy:
- Fully configuration-driven (no hardcoded tools)
- LLM-visible actions (Skills, Tools) → Use existing Tool module
- LLM-invisible actions (Flow, System) → Manual HTTP implementation
- Silent actions → Return None to exit iteration

This module provides:
1. Dynamic tool loading from configuration
2. FlowExecutor - For fixed business process APIs (manual HTTP)
3. SystemExecutor - For system actions (manual HTTP, supports silent mode)
4. SkillMatcher - For complex intent matching
5. WorkflowOrchestrator - Unified orchestration
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from bu_agent_sdk.llm.base import ToolDefinition
from bu_agent_sdk.tools.decorator import Tool, ToolContent
from bu_agent_sdk.tools.config_loader import (
    ConfigToolLoader,
    AgentConfigSchema,
    HttpTool,
    ToolConfig,
)


# =============================================================================
# Configuration Schema for Workflow (extends existing schema)
# =============================================================================


class SkillConfig(BaseModel):
    """
    Skill configuration from sop.json.

    Example from sop.json:
    {
      "condition": "Customer wants to schedule a demo...",
      "action": "Persuade customer to provide info...",
      "tools": ["save_customer_information"]
    }
    """

    condition: str = Field(description="Condition that triggers this skill")
    action: str = Field(description="Action to take when condition is met")
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names available for this skill"
    )


class WorkflowConfigSchema(BaseModel):
    """
    Complete workflow configuration schema matching sop.json structure.

    This schema is fully dynamic - all tools are loaded from configuration,
    no hardcoding required.
    """

    # Basic settings
    basic_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent basic settings (name, description, etc.)"
    )

    # Flow URL - single endpoint for flow API (optional)
    flow_url: str | None = Field(
        default=None,
        description="Flow API endpoint URL (if configured)"
    )

    # Knowledge retrieval URL - single endpoint for KB (optional)
    retrieve_knowledge_url: str | None = Field(
        default=None,
        description="Knowledge retrieval API endpoint URL (if configured)"
    )

    # Skills - complex intents with conditions
    skills: list[SkillConfig] = Field(
        default_factory=list,
        description="Skill definitions with conditions and tool mappings"
    )

    # System tools - all tools are loaded dynamically from this array
    system_tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool definitions (dynamically loaded, no hardcoding)"
    )


# =============================================================================
# Flow Executor (Manual HTTP Implementation)
# =============================================================================


class FlowExecutor:
    """
    Flow executor for flow_url endpoint.

    Design:
    - NOT registered as Tool (LLM-invisible)
    - Single endpoint for all flow operations
    - Triggered by pattern matching or explicit routing
    - Direct HTTP calls without Tool abstraction
    """

    def __init__(
        self,
        flow_url: str | None,
        http_client: httpx.AsyncClient | None = None,
        context_vars: dict[str, Any] | None = None,
    ):
        self.flow_url = flow_url
        self.http_client = http_client
        self.context_vars = context_vars or {}

    async def execute(
        self,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Execute flow API call.

        Args:
            user_message: User's message
            parameters: Additional parameters

        Returns:
            str: Response content
            None: If flow_url not configured
        """
        if not self.flow_url:
            return None

        try:
            # Build request body
            params = {
                "message": user_message,
                **(parameters or {}),
                **self.context_vars,
            }

            # Send request
            client = self.http_client or httpx.AsyncClient()
            should_close = self.http_client is None

            try:
                response = await client.request(
                    method="POST",
                    url=self.flow_url,
                    headers={"Content-Type": "application/json"},
                    json=params,
                    timeout=30.0,
                )

                # Handle response
                if response.is_success:
                    try:
                        data = response.json()
                        return json.dumps(data, ensure_ascii=False, indent=2)
                    except Exception:
                        return response.text
                else:
                    return f"❌ Flow API failed: HTTP {response.status_code}"

            finally:
                if should_close:
                    await client.aclose()

        except Exception as e:
            return f"❌ Flow API error: {e}"


# =============================================================================
# System Executor (Manual HTTP Implementation)
# =============================================================================


class SystemExecutor:
    """
    System action executor for special operations.

    Design:
    - NOT registered as Tool (LLM-invisible)
    - Handles special system operations that may need silent mode
    - Can be extended for custom handlers
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        context_vars: dict[str, Any] | None = None,
    ):
        self.http_client = http_client
        self.context_vars = context_vars or {}

    async def execute(
        self,
        action_type: str,
        endpoint_url: str,
        parameters: dict[str, Any] | None = None,
        silent: bool = False,
    ) -> str | None:
        """
        Execute a system action.

        Args:
            action_type: Type of action (handoff, close, update_profile, etc.)
            endpoint_url: API endpoint URL
            parameters: Action parameters
            silent: If True, returns None (silent execution)

        Returns:
            str: Response content (non-silent)
            None: Silent execution
        """
        try:
            # Build request body
            params = {
                **(parameters or {}),
                **self.context_vars,
            }

            # Send request
            client = self.http_client or httpx.AsyncClient()
            should_close = self.http_client is None

            try:
                response = await client.request(
                    method="POST",
                    url=endpoint_url,
                    headers={"Content-Type": "application/json"},
                    json=params,
                    timeout=30.0,
                )

                # Silent mode: return None regardless of result
                if silent:
                    return None

                # Non-silent: return response
                if response.is_success:
                    try:
                        data = response.json()
                        return json.dumps(data, ensure_ascii=False, indent=2)
                    except Exception:
                        return response.text
                else:
                    return f"❌ System action failed: HTTP {response.status_code}"

            finally:
                if should_close:
                    await client.aclose()

        except Exception as e:
            if silent:
                return None
            return f"❌ System action error: {e}"


# =============================================================================
# Skill Matcher (Intent Matching)
# =============================================================================


class SkillMatcher:
    """
    Skill condition matcher for complex intent detection.

    Extracts conditions from skills configuration and provides
    matching capabilities for intent routing.
    """

    def __init__(self, skills: list[SkillConfig]):
        self.skills = skills

    def get_all_conditions(self) -> list[str]:
        """Get all skill conditions for LLM context."""
        return [skill.condition for skill in self.skills]

    def get_skill_by_condition(self, condition: str) -> SkillConfig | None:
        """Get skill configuration by condition."""
        return next(
            (s for s in self.skills if s.condition == condition),
            None
        )

    def get_tools_for_condition(self, condition: str) -> list[str]:
        """Get tool names for a specific skill condition."""
        skill = self.get_skill_by_condition(condition)
        return skill.tools if skill else []

    def build_skill_prompt(self) -> str:
        """Build a prompt section describing all skills."""
        if not self.skills:
            return ""

        parts = ["## Skills (Complex Intents)", ""]

        for i, skill in enumerate(self.skills, 1):
            parts.append(f"### Skill {i}")
            parts.append(f"**Condition:** {skill.condition}")
            parts.append(f"**Action:** {skill.action}")
            if skill.tools:
                parts.append(f"**Available Tools:** {', '.join(skill.tools)}")
            parts.append("")

        return "\n".join(parts)


# =============================================================================
# Workflow Orchestrator (Fully Dynamic)
# =============================================================================


class WorkflowOrchestrator:
    """
    Workflow orchestrator with fully dynamic tool loading.

    Key Features:
    - NO hardcoded tools - all loaded from configuration
    - Supports arbitrary tool configurations
    - Automatically handles flow_url and retrieve_knowledge_url
    - Skills are mapped to existing tools via tool names

    Design:
    - LLM-visible: All tools in system_tools → HttpTool (via ConfigToolLoader)
    - LLM-invisible: flow_url → FlowExecutor (manual)
    - Skills: Conditions + tool name mappings (no separate registration)
    """

    def __init__(
        self,
        config: WorkflowConfigSchema,
        context_vars: dict[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.context_vars = context_vars or {}
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)

        # Manual executors (LLM-invisible)
        self.flow_executor = FlowExecutor(
            config.flow_url,
            self.http_client,
            self.context_vars
        )
        self.system_executor = SystemExecutor(
            self.http_client,
            self.context_vars
        )

        # Skill matcher
        self.skill_matcher = SkillMatcher(config.skills)

        # Tool loader (dynamic)
        self._tool_loader = self._create_tool_loader()

    def _create_tool_loader(self) -> ConfigToolLoader:
        """
        Create ConfigToolLoader from workflow configuration.

        This dynamically loads ALL tools from system_tools array,
        including retrieve_knowledge if configured.
        """
        # Build tools list
        tools = []

        # 1. Add all system_tools (dynamically)
        tools.extend(self.config.system_tools)

        # 2. Add retrieve_knowledge if URL is configured
        if self.config.retrieve_knowledge_url:
            # Check if retrieve_knowledge already exists in system_tools
            has_retrieve_knowledge = any(
                tool.get("name") == "retrieve_knowledge"
                for tool in self.config.system_tools
            )

            if not has_retrieve_knowledge:
                # Add retrieve_knowledge tool dynamically
                tools.append({
                    "name": "retrieve_knowledge",
                    "description": "Retrieve relevant information from the knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "string",
                                "description": "Keywords to search in knowledge base"
                            }
                        },
                        "required": ["keywords"]
                    },
                    "endpoint": {
                        "url": self.config.retrieve_knowledge_url,
                        "method": "GET",
                        "headers": {"Content-Type": "application/json"},
                        "body": {
                            "chatbotId": "{chatbotId}",
                            "tenantId": "{tenantId}",
                            "keywords": "{keywords}"
                        }
                    }
                })

        # Create AgentConfigSchema for ConfigToolLoader
        agent_config = AgentConfigSchema(
            basic_settings=self.config.basic_settings,
            tools=[ToolConfig.model_validate(tool) for tool in tools]
        )

        return ConfigToolLoader(agent_config)

    def get_tools(self) -> list[HttpTool]:
        """
        Get all LLM-visible tools (dynamically loaded from configuration).

        Returns:
            List of HttpTool instances
        """
        return self._tool_loader.get_tools(
            context_vars=self.context_vars,
            http_client=self.http_client
        )

    def get_tool_by_name(self, tool_name: str) -> HttpTool | None:
        """
        Get a specific tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            HttpTool instance or None if not found
        """
        tools = self.get_tools()
        return next((t for t in tools if t.name == tool_name), None)

    def build_system_prompt(self, include_skills: bool = True) -> str:
        """
        Build complete system prompt with workflow instructions.

        Args:
            include_skills: Whether to include skill descriptions

        Returns:
            Formatted system prompt
        """
        parts = []

        # Basic settings
        basic = self.config.basic_settings
        if basic:
            if basic.get("name"):
                parts.append(f"# {basic['name']}")
                parts.append("")
            if basic.get("description"):
                parts.append(basic["description"])
                parts.append("")
            if basic.get("background"):
                parts.append("## Background")
                parts.append(basic["background"])
                parts.append("")

        # Skills (if enabled)
        if include_skills:
            skill_prompt = self.skill_matcher.build_skill_prompt()
            if skill_prompt:
                parts.append(skill_prompt)

        # Communication guidelines
        if basic.get("language") or basic.get("tone"):
            parts.append("## Communication Guidelines")
            if basic.get("language"):
                parts.append(f"- Language: {basic['language']}")
            if basic.get("tone"):
                parts.append(f"- Tone: {basic['tone']}")
            parts.append("")

        return "\n".join(parts)

    async def execute_flow(
        self,
        user_message: str,
        parameters: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Execute flow API (LLM-invisible).

        Args:
            user_message: User's message
            parameters: Additional parameters

        Returns:
            str: Response content
            None: If flow_url not configured
        """
        return await self.flow_executor.execute(user_message, parameters)

    async def execute_system_action(
        self,
        action_type: str,
        endpoint_url: str,
        parameters: dict[str, Any] | None = None,
        silent: bool = False,
    ) -> str | None:
        """
        Execute system action (LLM-invisible).

        Args:
            action_type: Type of action
            endpoint_url: API endpoint URL
            parameters: Action parameters
            silent: If True, returns None

        Returns:
            str: Response content (non-silent)
            None: Silent execution
        """
        return await self.system_executor.execute(
            action_type,
            endpoint_url,
            parameters,
            silent
        )

    def get_skill_tools(self, condition: str) -> list[HttpTool]:
        """
        Get tools for a specific skill condition.

        Args:
            condition: Skill condition

        Returns:
            List of HttpTool instances for this skill
        """
        tool_names = self.skill_matcher.get_tools_for_condition(condition)
        all_tools = self.get_tools()

        return [
            tool for tool in all_tools
            if tool.name in tool_names
        ]


# =============================================================================
# Configuration Loaders
# =============================================================================


def load_workflow_config(config_dict: dict[str, Any]) -> WorkflowConfigSchema:
    """
    Load workflow configuration from dictionary.

    Args:
        config_dict: Configuration dictionary (e.g., from sop.json)

    Returns:
        WorkflowConfigSchema instance
    """
    return WorkflowConfigSchema.model_validate(config_dict)


def load_workflow_config_from_file(file_path: str) -> WorkflowConfigSchema:
    """
    Load workflow configuration from JSON file.

    Args:
        file_path: Path to JSON configuration file (e.g., sop.json)

    Returns:
        WorkflowConfigSchema instance
    """
    with open(file_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    return load_workflow_config(config_dict)


# =============================================================================
# Helper Functions
# =============================================================================


def get_tool_names_from_config(config: WorkflowConfigSchema) -> list[str]:
    """
    Get all tool names from configuration.

    Args:
        config: Workflow configuration

    Returns:
        List of tool names
    """
    tool_names = []

    # From system_tools
    for tool in config.system_tools:
        if "name" in tool:
            tool_names.append(tool["name"])

    # Add retrieve_knowledge if configured
    if config.retrieve_knowledge_url:
        has_retrieve_knowledge = any(
            tool.get("name") == "retrieve_knowledge"
            for tool in config.system_tools
        )
        if not has_retrieve_knowledge:
            tool_names.append("retrieve_knowledge")

    return tool_names


def validate_skill_tools(config: WorkflowConfigSchema) -> dict[str, list[str]]:
    """
    Validate that all tools referenced in skills exist in configuration.

    Args:
        config: Workflow configuration

    Returns:
        Dict mapping skill conditions to missing tool names
    """
    available_tools = set(get_tool_names_from_config(config))
    missing_tools = {}

    for skill in config.skills:
        missing = [
            tool_name for tool_name in skill.tools
            if tool_name not in available_tools
        ]
        if missing:
            missing_tools[skill.condition] = missing

    return missing_tools
