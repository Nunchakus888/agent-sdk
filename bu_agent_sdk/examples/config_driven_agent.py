"""
Configuration-driven agent example.

This example demonstrates how to create an agent from JSON configuration,
enabling SaaS-style deployments where tools are defined declaratively.

Run with:
    python -m bu_agent_sdk.examples.config_driven_agent

Configuration file format: see config/lead-acquistion.json
"""

import asyncio
from pathlib import Path

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import FinalResponseEvent, ToolCallEvent, ToolResultEvent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools.config_loader import ConfigToolLoader, export_config_schema


async def main():
    # ==========================================================================
    # 1. Load configuration from JSON file
    # ==========================================================================
    config_path = Path(__file__).parent / "config" / "lead-acquistion.json"
    config = ConfigToolLoader.load_from_file(config_path)

    print("=" * 60)
    print("Agent Configuration Loaded")
    print("=" * 60)
    print(f"Name: {config.basic_settings.name}")
    print(f"Language: {config.basic_settings.language}")
    print(f"Tone: {config.basic_settings.tone}")
    print(f"Tools: {[t.name for t in config.config.tools]}")
    print(f"Action Rules: {len(config.action_books)}")
    print()

    # ==========================================================================
    # 2. Get tools with context variables
    # ==========================================================================
    # Context variables are used to auto-fill placeholders like "todo_autofill_by_system"
    context_vars = {
        "dialogId": "dialog_12345",
        "tenantId": "tenant_67890",
        "chatbotId": config.basic_settings.chatbot_id,
    }

    tools = config.get_tools(context_vars=context_vars)

    print("Tool Definitions:")
    print("-" * 40)
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")
        print(f"    Parameters: {list(tool.config.parameters.properties.keys())}")
        print(f"    Endpoint: {tool.config.endpoint.method} {tool.config.endpoint.url}")
        print()

    # ==========================================================================
    # 3. Build system prompt from configuration
    # ==========================================================================
    system_prompt = config.build_system_prompt()

    print("System Prompt Preview:")
    print("-" * 40)
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    print()

    # ==========================================================================
    # 4. Create and run agent
    # ==========================================================================
    agent = Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=tools,
        system_prompt=system_prompt,
    )

    # Simulate a customer conversation
    print("=" * 60)
    print("Starting conversation simulation...")
    print("=" * 60)

    # Note: In production, actual HTTP calls would be made to the configured endpoints
    # For this demo, we'll just show the tool calls that would be made

    messages = [
        "Hi, I'm interested in your WhatsApp business services",
        "Can I schedule a demo? My email is john@example.com and I'm John from the US",
    ]

    for user_msg in messages:
        print(f"\n👤 User: {user_msg}")
        print("-" * 40)

        try:
            async for event in agent.query_stream(user_msg):
                match event:
                    case ToolCallEvent(tool=name, args=args):
                        print(f"🔧 Tool Call: {name}")
                        print(f"   Args: {args}")
                    case ToolResultEvent(tool=name, result=result):
                        preview = result[:100] + "..." if len(result) > 100 else result
                        print(f"   Result: {preview}")
                    case FinalResponseEvent(content=text):
                        print(f"\n🤖 Agent: {text}")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("   (This is expected if the configured endpoints are not accessible)")


def show_schema():
    """Print the JSON Schema for the configuration format."""
    import json

    print("=" * 60)
    print("Configuration JSON Schema")
    print("=" * 60)
    schema = export_config_schema()
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    import sys

    if "--schema" in sys.argv:
        show_schema()
    else:
        asyncio.run(main())
