# bu-agent-sdk

_An agent is just a for-loop._

![Agent Loop](./static/agent-loop.png)

The simplest possible agent framework. No abstractions. No magic. Just a for-loop of tool calls. The framework powering [BU.app](https://bu.app).

## Install

```bash
uv sync
```

or

```bash
uv add bu-agent-sdk
```

## Quick Start

```python
import asyncio
from bu_agent_sdk import Agent, tool, TaskComplete
from bu_agent_sdk.llm import ChatAnthropic

@tool("Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b

@tool("Signal task completion")
async def done(message: str) -> str:
    raise TaskComplete(message)

agent = Agent(
    llm=ChatAnthropic(model="claude-sonnet-4-20250514"),
    tools=[add, done],
)

async def main():
    result = await agent.query("What is 2 + 3?")
    print(result)

asyncio.run(main())
```

## Philosophy

**The Bitter Lesson:** All the value is in the RL'd model, not your 10,000 lines of abstractions.

Agent frameworks fail not because models are weak, but because their action spaces are incomplete. Give the LLM as much freedom as possible, then vibe-restrict based on evals.

## Features

### Done Tool Pattern

The naive "stop when no tool calls" approach fails. Agents finish prematurely. Force explicit completion:

```python
@tool("Signal completion")
async def done(message: str) -> str:
    raise TaskComplete(message)

agent = Agent(
    llm=llm,
    tools=[..., done],
    require_done_tool=True,  # Autonomous mode
)
```

### Ephemeral Messages

Large tool outputs (browser state, screenshots) blow up context. Keep only the last N:

```python
@tool("Get browser state", ephemeral=3)  # Keep last 3 only
async def get_state() -> str:
    return massive_dom_and_screenshot
```

### Simple LLM Primitives

~300 lines per provider. Same interface. Full control:

```python
from bu_agent_sdk.llm import ChatAnthropic, ChatOpenAI, ChatGoogle

# All implement BaseChatModel
agent = Agent(llm=ChatAnthropic(model="claude-sonnet-4-20250514"), tools=tools)
agent = Agent(llm=ChatOpenAI(model="gpt-4o"), tools=tools)
agent = Agent(llm=ChatGoogle(model="gemini-2.0-flash"), tools=tools)
```

### Context Compaction

Auto-summarize when approaching context limits:

```python
from bu_agent_sdk.agent import CompactionConfig

agent = Agent(
    llm=llm,
    tools=tools,
    compaction=CompactionConfig(threshold_ratio=0.80),
)
```

### Dependency Injection

FastAPI-style, type-safe:

```python
from typing import Annotated
from bu_agent_sdk import Depends

def get_db():
    return Database()

@tool("Query users")
async def get_user(id: int, db: Annotated[Database, Depends(get_db)]) -> str:
    return await db.find(id)
```

### Streaming Events

```python
from bu_agent_sdk.agent import ToolCallEvent, ToolResultEvent, FinalResponseEvent

async for event in agent.query_stream("do something"):
    match event:
        case ToolCallEvent(tool=name, args=args):
            print(f"Calling {name}")
        case ToolResultEvent(tool=name, result=result):
            print(f"{name} -> {result[:50]}")
        case FinalResponseEvent(content=text):
            print(f"Done: {text}")
```

## Claude Code in 100 Lines

A sandboxed coding assistant with dependency injection:

```python
import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatAnthropic
from bu_agent_sdk.tools import Depends, tool


@dataclass
class SandboxContext:
    """All file operations restricted to root_dir."""
    root_dir: Path
    working_dir: Path

    def resolve_path(self, path: str) -> Path:
        resolved = (self.working_dir / path).resolve()
        resolved.relative_to(self.root_dir)  # Raises if escapes
        return resolved


def get_sandbox() -> SandboxContext:
    raise RuntimeError("Override via dependency_overrides")


@tool("Execute shell command")
async def bash(command: str, ctx: Annotated[SandboxContext, Depends(get_sandbox)]) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=ctx.working_dir)
    return result.stdout + result.stderr or "(no output)"


@tool("Read file contents")
async def read(path: str, ctx: Annotated[SandboxContext, Depends(get_sandbox)]) -> str:
    return ctx.resolve_path(path).read_text()


@tool("Write file contents")
async def write(path: str, content: str, ctx: Annotated[SandboxContext, Depends(get_sandbox)]) -> str:
    ctx.resolve_path(path).write_text(content)
    return f"Wrote {len(content)} bytes"


@tool("Find files by glob pattern")
async def glob(pattern: str, ctx: Annotated[SandboxContext, Depends(get_sandbox)]) -> str:
    files = [str(f.relative_to(ctx.root_dir)) for f in ctx.working_dir.glob(pattern)]
    return "\n".join(files) or "No matches"


@tool("Signal task completion")
async def done(message: str) -> str:
    from bu_agent_sdk.agent import TaskComplete
    raise TaskComplete(message)


async def main():
    # Create sandbox
    root = Path("./sandbox")
    root.mkdir(exist_ok=True)
    ctx = SandboxContext(root_dir=root.resolve(), working_dir=root.resolve())

    agent = Agent(
        llm=ChatAnthropic(model="claude-sonnet-4-20250514"),
        tools=[bash, read, write, glob, done],
        system_prompt=f"Coding assistant. Working dir: {ctx.working_dir}",
        dependency_overrides={get_sandbox: lambda: ctx},
    )

    print("Agent ready. Ctrl+C to exit.")
    while True:
        task = input("\n> ")
        async for event in agent.query_stream(task):
            if hasattr(event, "tool"):
                print(f"  → {event.tool}")
            elif hasattr(event, "content") and event.content:
                print(f"\n{event.content}")


asyncio.run(main())
```

See [`bu_agent_sdk/examples/claude_code.py`](./bu_agent_sdk/examples/claude_code.py) for the full version with grep, edit, and todo tools.

## Examples

See [`bu_agent_sdk/examples/`](./bu_agent_sdk/examples/) for more:

- `claude_code.py` - Full Claude Code clone with sandboxed filesystem
- `dependency_injection.py` - FastAPI-style dependency injection

## The Bitter Truth

Every abstraction is a liability. Every "helper" is a failure point.

The models got good. Really good. They were RL'd on computer use, coding, browsing. They don't need your guardrails. They need:

- A complete action space
- A for-loop
- An explicit exit
- Context management

**The bitter lesson: The less you build, the more it works.**

## License

MIT

## Credits

Built by [Browser Use](https://browser-use.com). Inspired by reverse-engineering Claude Code and Gemini CLI.
