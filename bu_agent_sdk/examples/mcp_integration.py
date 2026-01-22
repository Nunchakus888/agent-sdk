"""
MCP (Model Context Protocol) integration example.

This example demonstrates how to connect to third-party MCP servers
and use their tools seamlessly with BU Agent SDK.

Run with:
    python -m bu_agent_sdk.examples.mcp_integration

Prerequisites:
    - MCP server running (or use the stdio example with npm packages)
    - httpx package installed
"""

import asyncio
from pathlib import Path

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import FinalResponseEvent, ToolCallEvent, ToolResultEvent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools.mcp_adapter import (
    MCPClient,
    MCPServiceLoader,
    MCPTransportType,
    load_mcp_config,
)


async def example_single_server():
    """Example: Connect to a single MCP server."""
    print("=" * 60)
    print("Example 1: Single MCP Server Connection")
    print("=" * 60)

    # Connect to HTTP-based MCP server
    async with MCPClient.from_url("http://localhost:8080") as client:
        # List available tools
        print("\nAvailable MCP Tools:")
        mcp_tools = await client.list_tools()
        for tool in mcp_tools:
            print(f"  - {tool.name}: {tool.description}")

        # List available resources
        print("\nAvailable MCP Resources:")
        resources = await client.list_resources()
        for resource in resources:
            print(f"  - {resource.uri}: {resource.description}")

        # Get tools as BU Agent SDK compatible tools
        tools = await client.get_tools()

        # Create agent with MCP tools
        agent = Agent(
            llm=ChatOpenAI(model="gpt-4o"),
            tools=tools,
            system_prompt="You have access to MCP tools. Use them to help the user.",
        )

        print("\nAgent created with MCP tools!")


async def example_multiple_servers():
    """Example: Connect to multiple MCP servers."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple MCP Servers")
    print("=" * 60)

    loader = MCPServiceLoader()

    # Add HTTP server
    loader.add_server(
        name="api",
        url="http://localhost:8080",
        transport_type=MCPTransportType.HTTP,
    )

    # Add stdio server (GitHub MCP from npm)
    # loader.add_stdio_server(
    #     name="github",
    #     command="npx",
    #     args=["-y", "@anthropic/mcp-server-github"],
    #     env={"GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
    # )

    async with loader:
        # Get all tools from all servers
        all_tools = await loader.get_all_tools()

        print("\nAll MCP Tools:")
        for tool in all_tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")

        # Create agent with all tools
        agent = Agent(
            llm=ChatOpenAI(model="gpt-4o"),
            tools=all_tools,
            system_prompt="You have access to multiple MCP services.",
        )

        print(f"\nAgent created with {len(all_tools)} tools!")


async def example_mixed_tools():
    """Example: Mix MCP tools with native tools."""
    print("\n" + "=" * 60)
    print("Example 3: Mixed MCP and Native Tools")
    print("=" * 60)

    from bu_agent_sdk.tools import tool

    # Native tool
    @tool("Calculate the sum of two numbers")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    # MCP tools (when server is available)
    loader = MCPServiceLoader()
    loader.add_server("api", "http://localhost:8080")

    try:
        async with loader:
            mcp_tools = await loader.get_all_tools()

            # Combine native and MCP tools
            all_tools = [add] + mcp_tools

            agent = Agent(
                llm=ChatOpenAI(model="gpt-4o"),
                tools=all_tools,
                system_prompt="You can calculate numbers and use MCP tools.",
            )

            print(f"\nAgent created with {len(all_tools)} tools (1 native + {len(mcp_tools)} MCP)")

    except Exception as e:
        print(f"\nMCP server not available, using native tools only: {e}")

        agent = Agent(
            llm=ChatOpenAI(model="gpt-4o"),
            tools=[add],
            system_prompt="You can calculate numbers.",
        )

        print("\nAgent created with native tools only")


async def example_config_file():
    """Example: Load MCP servers from JSON configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Configuration File")
    print("=" * 60)

    # Create example config
    config_path = Path(__file__).parent / "config" / "mcp_servers.json"

    print(f"\nExample configuration structure:")
    print("""
    {
      "mcp_servers": [
        {
          "name": "browser",
          "transport": "http",
          "url": "http://localhost:3000",
          "timeout": 30,
          "enabled": true
        },
        {
          "name": "github",
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@anthropic/mcp-server-github"],
          "env": {"GITHUB_TOKEN": "your-token"},
          "enabled": false
        }
      ]
    }
    """)

    # Load from config (if file exists)
    if config_path.exists():
        loader = load_mcp_config(str(config_path))
        async with loader:
            tools = await loader.get_all_tools()
            print(f"\nLoaded {len(tools)} tools from config")
    else:
        print(f"\nConfig file not found at {config_path}")
        print("Create it to test configuration-based loading")


async def example_with_conversation():
    """Example: Full conversation with MCP tools."""
    print("\n" + "=" * 60)
    print("Example 5: Full Conversation (Simulated)")
    print("=" * 60)

    # This example shows the expected flow without a real MCP server

    print("""
    # Expected flow with MCP server:

    1. Connect to MCP server
       async with MCPClient.from_url("http://localhost:8080") as client:

    2. Get tools
       tools = await client.get_tools()

    3. Create agent
       agent = Agent(llm=llm, tools=tools)

    4. Run conversation
       async for event in agent.query_stream("Search for Python tutorials"):
           match event:
               case ToolCallEvent(tool=name, args=args):
                   # MCP tool is being called
                   print(f"Calling MCP tool: {name}")
               case ToolResultEvent(result=result):
                   # MCP tool returned result
                   print(f"Result: {result[:100]}...")
               case FinalResponseEvent(content=text):
                   print(f"Agent: {text}")
    """)


def show_architecture():
    """Show the MCP integration architecture."""
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MCP Integration Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BU Agent SDK                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           Agent                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                      Tools List                              │    │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │   │
│  │  │  │ Native Tool  │  │ MCP Tool 1   │  │ MCP Tool 2   │       │    │   │
│  │  │  │ (@tool)      │  │ (Adapter)    │  │ (Adapter)    │       │    │   │
│  │  │  └──────────────┘  └──────┬───────┘  └──────┬───────┘       │    │   │
│  │  └────────────────────────────┼─────────────────┼───────────────┘    │   │
│  └───────────────────────────────┼─────────────────┼────────────────────┘   │
│                                  │                 │                        │
│                                  ▼                 ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MCPServiceLoader                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                        MCP Clients                           │    │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │   │
│  │  │  │ HTTP Client  │  │ SSE Client   │  │ Stdio Client │       │    │   │
│  │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │    │   │
│  │  └──────────┼────────────────┼────────────────┼─────────────────┘    │   │
│  └─────────────┼────────────────┼────────────────┼──────────────────────┘   │
│                │                │                │                          │
│                ▼                ▼                ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    External MCP Servers                              │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ HTTP Server  │  │ SSE Server   │  │ Local Process│               │   │
│  │  │ (REST API)   │  │ (Streaming)  │  │ (npm/pip)    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │                                                                      │   │
│  │  Examples:                                                           │   │
│  │  • @anthropic/mcp-server-github                                      │   │
│  │  • @anthropic/mcp-server-filesystem                                  │   │
│  │  • Custom business API servers                                       │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    """)


async def main():
    """Run all examples."""
    show_architecture()

    print("\nNote: These examples require MCP servers to be running.")
    print("Without servers, they will show the expected API usage.\n")

    # Run examples that don't require actual servers
    await example_config_file()
    await example_with_conversation()

    print("\n" + "=" * 60)
    print("To test with real MCP servers:")
    print("=" * 60)
    print("""
    1. Start an MCP server:
       npx -y @anthropic/mcp-server-filesystem /path/to/dir

    2. Or use a custom HTTP MCP server

    3. Update the URL in the examples and run again
    """)


if __name__ == "__main__":
    asyncio.run(main())
