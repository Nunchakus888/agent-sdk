"""
Configuration-driven tool loader for dynamic HTTP API tools.

This module provides support for loading tools from JSON configuration files,
enabling SaaS-style deployments where tools are defined declaratively.

Usage:
    from bu_agent_sdk.tools.config_loader import ConfigToolLoader, AgentConfig

    # Load from file
    config = ConfigToolLoader.load_from_file("config/agent.json")

    # Or from dict
    config = ConfigToolLoader.load_from_dict(config_dict)

    # Get tools for Agent
    tools = config.get_tools()

    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt=config.build_system_prompt(),
    )
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from bu_agent_sdk.llm.base import ToolDefinition
from bu_agent_sdk.tools.decorator import Tool, ToolContent


# =============================================================================
# 1. Configuration Schema (JSON Schema 定义)
# =============================================================================


class EndpointConfig(BaseModel):
    """HTTP endpoint configuration for a tool.

    JSON Schema:
    ```json
    {
      "url": "http://api.example.com/endpoint",
      "method": "POST",
      "headers": {"Content-Type": "application/json"},
      "body": {"param1": "{param1}", "static_field": "value"}
    }
    ```
    """

    url: str = Field(description="The HTTP endpoint URL")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = Field(
        default="POST", description="HTTP method"
    )
    headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"},
        description="HTTP headers",
    )
    body: dict[str, Any] | None = Field(
        default=None,
        description="Request body template with {param} placeholders",
    )
    query_params: dict[str, str] | None = Field(
        default=None,
        description="Query parameters template with {param} placeholders",
    )
    timeout: float = Field(default=30.0, description="Request timeout in seconds")


class ToolParameterProperty(BaseModel):
    """JSON Schema property definition for a tool parameter.

    JSON Schema:
    ```json
    {
      "type": "string",
      "description": "The email of the customer",
      "default": null,
      "examples": ["example@email.com"]
    }
    ```
    """

    type: Literal["string", "integer", "number", "boolean", "array", "object"] = Field(
        default="string"
    )
    description: str | None = Field(default=None)
    default: Any = Field(default=None)
    examples: list[Any] | None = Field(default=None)
    enum: list[Any] | None = Field(default=None)
    items: dict[str, Any] | None = Field(
        default=None, description="For array types, defines item schema"
    )


class ToolParameters(BaseModel):
    """JSON Schema for tool parameters.

    JSON Schema:
    ```json
    {
      "type": "object",
      "properties": {
        "email": {"type": "string", "description": "Customer email"},
        "name": {"type": "string", "description": "Customer name"}
      },
      "required": ["email"]
    }
    ```
    """

    type: Literal["object"] = "object"
    properties: dict[str, ToolParameterProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolConfig(BaseModel):
    """Configuration for a single tool.

    JSON Schema:
    ```json
    {
      "name": "save_customer_information",
      "description": "Save the customer's information",
      "parameters": {
        "type": "object",
        "properties": {...},
        "required": ["email"]
      },
      "endpoint": {
        "url": "http://api.example.com/save",
        "method": "POST",
        "body": {"email": "{email}"}
      }
    }
    ```
    """

    name: str = Field(description="Tool name (unique identifier)")
    description: str = Field(description="Tool description for LLM")
    parameters: ToolParameters = Field(
        default_factory=ToolParameters, description="JSON Schema for parameters"
    )
    endpoint: EndpointConfig = Field(description="HTTP endpoint configuration")


class ActionBookEntry(BaseModel):
    """Conditional action rule.

    JSON Schema:
    ```json
    {
      "condition": "Customer wants to schedule a demo",
      "action": "Use save_customer_information to save their info",
      "tools": ["save_customer_information"]
    }
    ```
    """

    condition: str = Field(description="When this condition is met")
    action: str = Field(description="What action to take")
    tools: list[str] = Field(
        default_factory=list, description="Tools to use for this action"
    )


class BasicSettings(BaseModel):
    """Basic agent settings.

    JSON Schema:
    ```json
    {
      "name": "Customer Service Agent",
      "description": "Help customers with their inquiries",
      "background": "Company background info",
      "language": "English",
      "tone": "Friendly and professional",
      "chatbot_id": "abc123"
    }
    ```
    """

    name: str = Field(description="Agent name")
    description: str = Field(description="Agent description/role")
    background: str = Field(default="", description="Company/context background")
    language: str = Field(default="English", description="Primary language")
    tone: str = Field(default="Professional", description="Communication tone")
    chatbot_id: str | None = Field(default=None, description="External chatbot ID")


class AgentConfigSchema(BaseModel):
    """Complete agent configuration schema.

    JSON Schema:
    ```json
    {
      "basic_settings": {...},
      "action_books": [...],
      "tools": [...]
    }
    ```
    """

    basic_settings: BasicSettings
    action_books: list[ActionBookEntry] = Field(default_factory=list)
    tools: list[ToolConfig] = Field(default_factory=list)


# =============================================================================
# 2. HTTP Tool Implementation
# =============================================================================


@dataclass
class HttpTool:
    """A dynamically created HTTP API tool from configuration.

    This wraps a ToolConfig and provides Tool-compatible interface.
    """

    config: ToolConfig
    http_client: httpx.AsyncClient | None = None
    context_vars: dict[str, Any] = field(default_factory=dict)
    """Context variables for auto-fill (e.g., dialogId, tenantId)"""

    _definition: ToolDefinition | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    @property
    def ephemeral(self) -> bool:
        return False

    @property
    def definition(self) -> ToolDefinition:
        """Generate ToolDefinition from config."""
        if self._definition is not None:
            return self._definition

        # Convert ToolParameters to dict schema
        properties: dict[str, Any] = {}
        for prop_name, prop in self.config.parameters.properties.items():
            prop_schema: dict[str, Any] = {"type": prop.type}
            if prop.description:
                prop_schema["description"] = prop.description
            if prop.enum:
                prop_schema["enum"] = prop.enum
            if prop.examples:
                prop_schema["examples"] = prop.examples
            properties[prop_name] = prop_schema

        schema = {
            "type": "object",
            "properties": properties,
            "required": self.config.parameters.required,
            "additionalProperties": False,
        }

        self._definition = ToolDefinition(
            name=self.config.name,
            description=self.config.description,
            parameters=schema,
            strict=True,
        )
        return self._definition

    def _substitute_placeholders(
        self, template: Any, params: dict[str, Any]
    ) -> Any:
        """Substitute {param} placeholders in template with actual values.

        Handles:
        - String values: "{param}" -> actual value
        - Nested dicts and lists
        - Special marker "todo_autofill_by_system" -> context_vars
        """
        if isinstance(template, str):
            # Check for "todo_autofill_by_system" marker
            if template == "todo_autofill_by_system":
                # Try to find matching key in context_vars
                return None  # Will be filled by context

            # Check for full placeholder replacement: "{param}"
            match = re.fullmatch(r"\{(\w+)\}", template)
            if match:
                param_name = match.group(1)
                if param_name in params:
                    return params[param_name]
                if param_name in self.context_vars:
                    return self.context_vars[param_name]
                return template  # Keep as-is if not found

            # Partial substitution for strings containing placeholders
            def replacer(m: re.Match) -> str:
                param_name = m.group(1)
                if param_name in params:
                    return str(params[param_name])
                if param_name in self.context_vars:
                    return str(self.context_vars[param_name])
                return m.group(0)

            return re.sub(r"\{(\w+)\}", replacer, template)

        elif isinstance(template, dict):
            result = {}
            for k, v in template.items():
                substituted = self._substitute_placeholders(v, params)
                # Skip None values (unfilled todo_autofill_by_system)
                if substituted is not None:
                    result[k] = substituted
            return result

        elif isinstance(template, list):
            return [self._substitute_placeholders(item, params) for item in template]

        else:
            return template

    async def execute(
        self, _overrides: dict | None = None, **kwargs: Any
    ) -> ToolContent:
        """Execute the HTTP tool with given parameters.

        Args:
            _overrides: Ignored (for compatibility with Tool interface)
            **kwargs: Tool parameters from LLM

        Returns:
            HTTP response content as string
        """
        endpoint = self.config.endpoint

        # Merge kwargs with defaults from parameter definitions
        params = {}
        for prop_name, prop in self.config.parameters.properties.items():
            if prop_name in kwargs:
                params[prop_name] = kwargs[prop_name]
            elif prop.default is not None:
                params[prop_name] = prop.default

        # Add context variables
        params.update(self.context_vars)

        # Build request
        url = self._substitute_placeholders(endpoint.url, params)
        headers = self._substitute_placeholders(endpoint.headers, params)

        # Build body or query params
        body = None
        query_params = None

        if endpoint.body:
            body = self._substitute_placeholders(endpoint.body, params)

        if endpoint.query_params:
            query_params = self._substitute_placeholders(endpoint.query_params, params)

        # Use provided client or create new one
        client = self.http_client or httpx.AsyncClient()
        should_close = self.http_client is None

        try:
            response = await client.request(
                method=endpoint.method,
                url=url,
                headers=headers,
                json=body if body else None,
                params=query_params,
                timeout=endpoint.timeout,
            )

            # Return response
            if response.is_success:
                try:
                    return json.dumps(response.json(), ensure_ascii=False)
                except json.JSONDecodeError:
                    return response.text
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"

        except httpx.TimeoutException:
            return f"Error: Request timeout after {endpoint.timeout}s"
        except httpx.RequestError as e:
            return f"Error: Request failed - {str(e)}"
        finally:
            if should_close:
                await client.aclose()


# =============================================================================
# 3. Config Loader
# =============================================================================


class ConfigToolLoader:
    """Loader for configuration-driven agent tools.

    Example:
        ```python
        # Load configuration
        config = ConfigToolLoader.load_from_file("config/agent.json")

        # Get tools with context variables
        tools = config.get_tools(context_vars={
            "dialogId": "123",
            "tenantId": "456",
            "chatbotId": "789"
        })

        # Build system prompt
        system_prompt = config.build_system_prompt()

        # Create agent
        agent = Agent(llm=llm, tools=tools, system_prompt=system_prompt)
        ```
    """

    def __init__(self, config: AgentConfigSchema):
        self.config = config
        self._http_client: httpx.AsyncClient | None = None

    @classmethod
    def load_from_file(cls, path: str | Path) -> "ConfigToolLoader":
        """Load configuration from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.load_from_dict(data)

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> "ConfigToolLoader":
        """Load configuration from a dictionary."""
        config = AgentConfigSchema.model_validate(data)
        return cls(config)

    def get_tools(
        self,
        context_vars: dict[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> list[HttpTool]:
        """Get all tools from configuration.

        Args:
            context_vars: Variables for auto-fill placeholders
                (e.g., dialogId, tenantId, chatbotId)
            http_client: Shared HTTP client for all tools

        Returns:
            List of HttpTool instances
        """
        tools = []
        for tool_config in self.config.tools:
            tool = HttpTool(
                config=tool_config,
                http_client=http_client,
                context_vars=context_vars or {},
            )
            tools.append(tool)
        return tools

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get ToolDefinitions for all configured tools.

        Useful for inspecting tool schemas without creating executable tools.
        """
        return [
            HttpTool(config=tc, context_vars={}).definition
            for tc in self.config.tools
        ]

    def build_system_prompt(self, include_action_books: bool = True) -> str:
        """Build a system prompt from configuration.

        Args:
            include_action_books: Whether to include action rules in prompt

        Returns:
            System prompt string
        """
        settings = self.config.basic_settings
        parts = []

        # Role definition
        parts.append(f"You are {settings.name}.")
        parts.append("")
        parts.append(f"## Role Description")
        parts.append(settings.description)

        # Background
        if settings.background:
            parts.append("")
            parts.append("## Background")
            parts.append(settings.background)

        # Communication style
        parts.append("")
        parts.append("## Communication Guidelines")
        parts.append(f"- Language: {settings.language}")
        parts.append(f"- Tone: {settings.tone}")

        # Action books
        if include_action_books and self.config.action_books:
            parts.append("")
            parts.append("## Action Rules")
            for i, ab in enumerate(self.config.action_books, 1):
                parts.append(f"")
                parts.append(f"### Rule {i}")
                parts.append(f"**When:** {ab.condition}")
                parts.append(f"**Action:** {ab.action}")
                if ab.tools:
                    parts.append(f"**Tools:** {', '.join(ab.tools)}")

        return "\n".join(parts)

    @property
    def basic_settings(self) -> BasicSettings:
        """Access basic settings."""
        return self.config.basic_settings

    @property
    def action_books(self) -> list[ActionBookEntry]:
        """Access action books."""
        return self.config.action_books


# =============================================================================
# 4. JSON Schema Export (for documentation)
# =============================================================================


def export_config_schema() -> dict[str, Any]:
    """Export the complete JSON Schema for agent configuration.

    Use this to document the expected configuration format.

    Returns:
        JSON Schema dict
    """
    return AgentConfigSchema.model_json_schema()


def export_tool_schema() -> dict[str, Any]:
    """Export the JSON Schema for a single tool configuration.

    Returns:
        JSON Schema dict
    """
    return ToolConfig.model_json_schema()
