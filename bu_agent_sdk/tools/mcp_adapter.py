"""
MCP (Model Context Protocol) adapter for integrating third-party MCP services.

This module provides seamless integration with MCP servers, allowing you to use
remote tools and resources as if they were native BU Agent SDK tools.

MCP is a protocol by Anthropic for standardizing AI model interactions with
external data sources and tools.

Usage:
    from bu_agent_sdk.tools.mcp_adapter import MCPClient, MCPToolAdapter

    # Connect to MCP server
    async with MCPClient("http://localhost:8080") as client:
        # Get tools as BU Agent SDK compatible tools
        tools = await client.get_tools()

        # Use in agent
        agent = Agent(llm=llm, tools=tools)

    # Or use the high-level loader
    from bu_agent_sdk.tools.mcp_adapter import MCPServiceLoader

    loader = MCPServiceLoader()
    loader.add_server("browser", "http://localhost:3000")
    loader.add_server("github", "npx", ["-y", "@anthropic/mcp-server-github"])

    tools = await loader.get_all_tools()
"""

import asyncio
import json
import subprocess
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Literal

import httpx
from pydantic import BaseModel, Field

from bu_agent_sdk.llm.base import ToolDefinition
from bu_agent_sdk.tools.decorator import ToolContent


# =============================================================================
# 1. MCP Protocol Types
# =============================================================================


class MCPTransportType(str, Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"      # Local process via stdin/stdout
    SSE = "sse"          # HTTP Server-Sent Events
    HTTP = "http"        # Simple HTTP (request/response)
    WEBSOCKET = "ws"     # WebSocket (bidirectional)


class MCPToolSchema(BaseModel):
    """MCP tool definition from server."""
    name: str
    description: str | None = None
    inputSchema: dict[str, Any] = Field(default_factory=dict)


class MCPResource(BaseModel):
    """MCP resource definition."""
    uri: str
    name: str
    description: str | None = None
    mimeType: str | None = None


class MCPPrompt(BaseModel):
    """MCP prompt template."""
    name: str
    description: str | None = None
    arguments: list[dict[str, Any]] = Field(default_factory=list)


class MCPServerCapabilities(BaseModel):
    """MCP server capabilities."""
    tools: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    prompts: dict[str, Any] | None = None
    sampling: dict[str, Any] | None = None


class MCPServerInfo(BaseModel):
    """MCP server information from initialization."""
    name: str
    version: str
    capabilities: MCPServerCapabilities = Field(default_factory=MCPServerCapabilities)


# =============================================================================
# 2. MCP Transport Layer (Abstract)
# =============================================================================


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        ...

    @abstractmethod
    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send JSON-RPC request and wait for response."""
        ...

    @abstractmethod
    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send JSON-RPC notification (no response expected)."""
        ...


class HTTPTransport(MCPTransport):
    """HTTP-based MCP transport (simple request/response)."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_id = 0

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Transport not connected")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
        }
        if params:
            request["params"] = params

        response = await self._client.post(
            f"{self.base_url}/mcp",
            json=request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()
        if "error" in result:
            raise MCPError(
                result["error"].get("code", -1),
                result["error"].get("message", "Unknown error"),
            )

        return result.get("result", {})

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        if not self._client:
            raise RuntimeError("Transport not connected")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        await self._client.post(
            f"{self.base_url}/mcp",
            json=notification,
            headers={"Content-Type": "application/json"},
        )


class SSETransport(MCPTransport):
    """Server-Sent Events transport for MCP."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._sse_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.timeout)
        # Start SSE listener in background
        self._sse_task = asyncio.create_task(self._listen_sse())

    async def disconnect(self) -> None:
        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _listen_sse(self) -> None:
        """Background task to listen for SSE events."""
        if not self._client:
            return

        try:
            async with self._client.stream(
                "GET", f"{self.base_url}/sse"
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        request_id = data.get("id")
                        if request_id in self._pending_requests:
                            future = self._pending_requests.pop(request_id)
                            if "error" in data:
                                future.set_exception(
                                    MCPError(
                                        data["error"].get("code", -1),
                                        data["error"].get("message", "Unknown"),
                                    )
                                )
                            else:
                                future.set_result(data.get("result", {}))
        except asyncio.CancelledError:
            pass

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Transport not connected")

        request_id = self._next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        # Send request
        await self._client.post(
            f"{self.base_url}/message",
            json=request,
            headers={"Content-Type": "application/json"},
        )

        # Wait for response via SSE
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPError(-1, f"Request timeout after {self.timeout}s")

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        if not self._client:
            raise RuntimeError("Transport not connected")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        await self._client.post(
            f"{self.base_url}/message",
            json=notification,
            headers={"Content-Type": "application/json"},
        )


class StdioTransport(MCPTransport):
    """Stdio-based transport for local MCP servers."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._read_lock = asyncio.Lock()

    async def connect(self) -> None:
        import os

        env = os.environ.copy()
        if self.env:
            env.update(self.env)

        self._process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    async def disconnect(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("Transport not connected")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
        }
        if params:
            request["params"] = params

        # Send request
        request_line = json.dumps(request) + "\n"
        self._process.stdin.write(request_line.encode())
        self._process.stdin.flush()

        # Read response (run in executor to avoid blocking)
        async with self._read_lock:
            loop = asyncio.get_event_loop()
            response_line = await loop.run_in_executor(
                None, self._process.stdout.readline
            )

        if not response_line:
            raise MCPError(-1, "No response from MCP server")

        result = json.loads(response_line.decode())
        if "error" in result:
            raise MCPError(
                result["error"].get("code", -1),
                result["error"].get("message", "Unknown error"),
            )

        return result.get("result", {})

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Transport not connected")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        notification_line = json.dumps(notification) + "\n"
        self._process.stdin.write(notification_line.encode())
        self._process.stdin.flush()


# =============================================================================
# 3. MCP Errors
# =============================================================================


class MCPError(Exception):
    """MCP protocol error."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"MCP Error {code}: {message}")


# =============================================================================
# 4. MCP Client
# =============================================================================


class MCPClient:
    """
    Client for connecting to MCP servers.

    Supports multiple transport types:
    - HTTP: Simple request/response
    - SSE: Server-Sent Events for streaming
    - Stdio: Local process communication

    Example:
        ```python
        # HTTP connection
        async with MCPClient("http://localhost:8080") as client:
            tools = await client.list_tools()
            result = await client.call_tool("search", {"query": "hello"})

        # Stdio connection (local MCP server)
        async with MCPClient.from_stdio("npx", ["-y", "@anthropic/mcp-server-github"]) as client:
            tools = await client.list_tools()
        ```
    """

    def __init__(
        self,
        transport: MCPTransport,
        client_name: str = "bu-agent-sdk",
        client_version: str = "1.0.0",
    ):
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self.server_info: MCPServerInfo | None = None
        self._initialized = False

    @classmethod
    def from_url(
        cls,
        url: str,
        transport_type: MCPTransportType = MCPTransportType.HTTP,
        timeout: float = 30.0,
    ) -> "MCPClient":
        """Create client from URL."""
        if transport_type == MCPTransportType.HTTP:
            transport = HTTPTransport(url, timeout)
        elif transport_type == MCPTransportType.SSE:
            transport = SSETransport(url, timeout)
        else:
            raise ValueError(f"URL-based transport not supported for {transport_type}")

        return cls(transport)

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "MCPClient":
        """Create client for stdio-based MCP server."""
        transport = StdioTransport(command, args, env)
        return cls(transport)

    async def connect(self) -> None:
        """Connect and initialize MCP session."""
        await self.transport.connect()

        # Initialize MCP session
        result = await self.transport.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        )

        self.server_info = MCPServerInfo(
            name=result.get("serverInfo", {}).get("name", "unknown"),
            version=result.get("serverInfo", {}).get("version", "unknown"),
            capabilities=MCPServerCapabilities(**result.get("capabilities", {})),
        )

        # Send initialized notification
        await self.transport.send_notification("notifications/initialized")
        self._initialized = True

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        await self.transport.disconnect()
        self._initialized = False

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    # -------------------------------------------------------------------------
    # Tool Operations
    # -------------------------------------------------------------------------

    async def list_tools(self) -> list[MCPToolSchema]:
        """List available tools from MCP server."""
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        result = await self.transport.send_request("tools/list")
        tools = result.get("tools", [])
        return [MCPToolSchema(**t) for t in tools]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Call a tool on the MCP server.

        Returns:
            List of content blocks (text, image, etc.)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        result = await self.transport.send_request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )

        return result.get("content", [])

    # -------------------------------------------------------------------------
    # Resource Operations
    # -------------------------------------------------------------------------

    async def list_resources(self) -> list[MCPResource]:
        """List available resources from MCP server."""
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        result = await self.transport.send_request("resources/list")
        resources = result.get("resources", [])
        return [MCPResource(**r) for r in resources]

    async def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """Read a resource by URI.

        Returns:
            List of content blocks
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        result = await self.transport.send_request(
            "resources/read", {"uri": uri}
        )
        return result.get("contents", [])

    # -------------------------------------------------------------------------
    # Prompt Operations
    # -------------------------------------------------------------------------

    async def list_prompts(self) -> list[MCPPrompt]:
        """List available prompts from MCP server."""
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        result = await self.transport.send_request("prompts/list")
        prompts = result.get("prompts", [])
        return [MCPPrompt(**p) for p in prompts]

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Get a prompt with filled arguments."""
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        return await self.transport.send_request(
            "prompts/get",
            {"name": name, "arguments": arguments or {}},
        )

    # -------------------------------------------------------------------------
    # High-level: Get as BU Agent SDK Tools
    # -------------------------------------------------------------------------

    async def get_tools(self) -> list["MCPToolAdapter"]:
        """Get all MCP tools as BU Agent SDK compatible tools.

        Returns:
            List of MCPToolAdapter instances that can be used with Agent
        """
        mcp_tools = await self.list_tools()
        return [
            MCPToolAdapter(
                client=self,
                mcp_tool=tool,
                server_name=self.server_info.name if self.server_info else "unknown",
            )
            for tool in mcp_tools
        ]


# =============================================================================
# 5. MCP Tool Adapter (Bridge to BU Agent SDK)
# =============================================================================


@dataclass
class MCPToolAdapter:
    """
    Adapter that wraps an MCP tool as a BU Agent SDK compatible tool.

    This allows MCP tools to be used seamlessly with the Agent class.
    """

    client: MCPClient
    mcp_tool: MCPToolSchema
    server_name: str = "mcp"
    _definition: ToolDefinition | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        """Tool name (prefixed with server name to avoid conflicts)."""
        return f"{self.server_name}__{self.mcp_tool.name}"

    @property
    def description(self) -> str:
        return self.mcp_tool.description or f"MCP tool: {self.mcp_tool.name}"

    @property
    def ephemeral(self) -> bool:
        return False

    @property
    def definition(self) -> ToolDefinition:
        """Generate ToolDefinition from MCP tool schema."""
        if self._definition is not None:
            return self._definition

        # Convert MCP inputSchema to our format
        schema = self.mcp_tool.inputSchema.copy()

        # Ensure required fields for OpenAI strict mode
        if "type" not in schema:
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        self._definition = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
            strict=True,
        )
        return self._definition

    async def execute(
        self, _overrides: dict | None = None, **kwargs: Any
    ) -> ToolContent:
        """Execute the MCP tool.

        Args:
            _overrides: Ignored (for compatibility)
            **kwargs: Tool arguments

        Returns:
            Tool result as string or content parts
        """
        try:
            content_blocks = await self.client.call_tool(
                self.mcp_tool.name, kwargs
            )

            # Convert MCP content blocks to string
            result_parts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    result_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    # Could return as ContentPartImageParam
                    result_parts.append(f"[Image: {block.get('mimeType', 'image')}]")
                elif block.get("type") == "resource":
                    result_parts.append(
                        f"[Resource: {block.get('resource', {}).get('uri', 'unknown')}]"
                    )

            return "\n".join(result_parts) if result_parts else "Tool executed successfully"

        except MCPError as e:
            return f"Error: {e.message}"
        except Exception as e:
            return f"Error executing MCP tool: {str(e)}"


# =============================================================================
# 6. MCP Service Loader (High-level API)
# =============================================================================


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    """Unique identifier for this server."""

    transport_type: MCPTransportType
    """Transport type to use."""

    # For HTTP/SSE
    url: str | None = None

    # For Stdio
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # Common
    timeout: float = 30.0
    enabled: bool = True


class MCPServiceLoader:
    """
    High-level loader for managing multiple MCP servers.

    Example:
        ```python
        loader = MCPServiceLoader()

        # Add HTTP server
        loader.add_server("api", "http://localhost:8080")

        # Add stdio server (npm package)
        loader.add_stdio_server(
            "github",
            command="npx",
            args=["-y", "@anthropic/mcp-server-github"],
            env={"GITHUB_TOKEN": "..."}
        )

        # Connect and get all tools
        async with loader:
            tools = await loader.get_all_tools()
            agent = Agent(llm=llm, tools=tools)
        ```
    """

    def __init__(self):
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}

    def add_server(
        self,
        name: str,
        url: str,
        transport_type: MCPTransportType = MCPTransportType.HTTP,
        timeout: float = 30.0,
    ) -> "MCPServiceLoader":
        """Add an HTTP/SSE MCP server."""
        self._configs[name] = MCPServerConfig(
            name=name,
            transport_type=transport_type,
            url=url,
            timeout=timeout,
        )
        return self

    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "MCPServiceLoader":
        """Add a stdio-based MCP server."""
        self._configs[name] = MCPServerConfig(
            name=name,
            transport_type=MCPTransportType.STDIO,
            command=command,
            args=args or [],
            env=env or {},
        )
        return self

    def add_from_dict(self, config: dict[str, Any]) -> "MCPServiceLoader":
        """Add server from dictionary configuration."""
        cfg = MCPServerConfig(
            name=config["name"],
            transport_type=MCPTransportType(config.get("transport", "http")),
            url=config.get("url"),
            command=config.get("command"),
            args=config.get("args", []),
            env=config.get("env", {}),
            timeout=config.get("timeout", 30.0),
            enabled=config.get("enabled", True),
        )
        self._configs[cfg.name] = cfg
        return self

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for name, config in self._configs.items():
            if not config.enabled:
                continue

            try:
                if config.transport_type == MCPTransportType.STDIO:
                    client = MCPClient.from_stdio(
                        config.command or "",
                        config.args,
                        config.env or None,
                    )
                else:
                    client = MCPClient.from_url(
                        config.url or "",
                        config.transport_type,
                        config.timeout,
                    )

                await client.connect()
                self._clients[name] = client

            except Exception as e:
                # Log error but continue with other servers
                print(f"Warning: Failed to connect to MCP server '{name}': {e}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception:
                pass
        self._clients.clear()

    async def __aenter__(self) -> "MCPServiceLoader":
        await self.connect_all()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect_all()

    async def get_all_tools(self) -> list[MCPToolAdapter]:
        """Get tools from all connected MCP servers."""
        all_tools: list[MCPToolAdapter] = []

        for name, client in self._clients.items():
            try:
                tools = await client.get_tools()
                all_tools.extend(tools)
            except Exception as e:
                print(f"Warning: Failed to get tools from '{name}': {e}")

        return all_tools

    async def get_tools_from(self, server_name: str) -> list[MCPToolAdapter]:
        """Get tools from a specific MCP server."""
        client = self._clients.get(server_name)
        if not client:
            raise ValueError(f"Server '{server_name}' not connected")

        return await client.get_tools()

    def get_client(self, server_name: str) -> MCPClient | None:
        """Get a specific client instance."""
        return self._clients.get(server_name)


# =============================================================================
# 7. JSON Configuration Support
# =============================================================================


class MCPConfigSchema(BaseModel):
    """JSON schema for MCP configuration.

    Example config:
    ```json
    {
      "mcp_servers": [
        {
          "name": "browser",
          "transport": "http",
          "url": "http://localhost:3000",
          "timeout": 30
        },
        {
          "name": "github",
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@anthropic/mcp-server-github"],
          "env": {"GITHUB_TOKEN": "..."}
        }
      ]
    }
    ```
    """

    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)


def load_mcp_config(config_path: str) -> MCPServiceLoader:
    """Load MCP configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configured MCPServiceLoader instance
    """
    from pathlib import Path

    with open(Path(config_path), encoding="utf-8") as f:
        data = json.load(f)

    config = MCPConfigSchema.model_validate(data)
    loader = MCPServiceLoader()

    for server_config in config.mcp_servers:
        loader.add_from_dict(server_config)

    return loader
