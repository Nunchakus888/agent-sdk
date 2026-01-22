"""
Tools framework for building agentic applications.

This module provides:
- @tool decorator for creating type-safe tools from functions
- Depends for dependency injection
- ConfigToolLoader for loading tools from JSON configuration
- MCPClient and MCPServiceLoader for MCP (Model Context Protocol) integration

Example:
    from bu_agent_sdk.tools import tool, Depends

    # Define a simple tool
    @tool("Add two numbers together")
    async def add(a: int, b: int) -> int:
        return a + b

    # Define a tool with dependency injection
    async def get_db():
        return DatabaseConnection()

    @tool("Query the database")
    async def query(sql: str, db: Depends(get_db)) -> str:
        return await db.execute(sql)

    # Load tools from JSON configuration
    from bu_agent_sdk.tools import ConfigToolLoader

    config = ConfigToolLoader.load_from_file("config/agent.json")
    tools = config.get_tools(context_vars={"dialogId": "123"})

    # Connect to MCP servers
    from bu_agent_sdk.tools import MCPClient, MCPServiceLoader

    async with MCPClient.from_url("http://localhost:8080") as client:
        mcp_tools = await client.get_tools()
"""

from bu_agent_sdk.tools.config_loader import (
    AgentConfigSchema,
    ConfigToolLoader,
    HttpTool,
    ToolConfig,
    export_config_schema,
    export_tool_schema,
)
from bu_agent_sdk.tools.decorator import Tool, ToolContent, tool
from bu_agent_sdk.tools.depends import DependencyOverrides, Depends
from bu_agent_sdk.tools.mcp_adapter import (
    MCPClient,
    MCPError,
    MCPServerConfig,
    MCPServiceLoader,
    MCPToolAdapter,
    MCPTransportType,
    load_mcp_config,
)

__all__ = [
    # Decorator-based tools
    "tool",
    "Tool",
    "ToolContent",
    # Dependency injection
    "Depends",
    "DependencyOverrides",
    # Configuration-driven tools
    "ConfigToolLoader",
    "HttpTool",
    "ToolConfig",
    "AgentConfigSchema",
    "export_config_schema",
    "export_tool_schema",
    # MCP integration
    "MCPClient",
    "MCPServiceLoader",
    "MCPToolAdapter",
    "MCPServerConfig",
    "MCPTransportType",
    "MCPError",
    "load_mcp_config",
]
