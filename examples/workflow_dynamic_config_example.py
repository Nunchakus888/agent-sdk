"""
Fully Dynamic Configuration Example - Using sop.json

This example demonstrates:
1. Loading configuration from sop.json (NO hardcoded tools)
2. All tools dynamically loaded from system_tools array
3. Skills mapped to tools via tool names
4. Flow and KB URLs automatically handled
"""

import asyncio
import json
from pathlib import Path

from bu_agent_sdk.workflow import (
    WorkflowOrchestrator,
    load_workflow_config_from_file,
    get_tool_names_from_config,
    validate_skill_tools,
)


async def demo_dynamic_loading():
    """Demonstrate fully dynamic tool loading from sop.json."""
    print("=" * 80)
    print("Dynamic Configuration Loading from sop.json")
    print("=" * 80)

    # Load configuration from sop.json
    config_path = Path(__file__).parent.parent / "docs" / "configs" / "sop.json"
    print(f"\n📁 Loading configuration from: {config_path}")

    config = load_workflow_config_from_file(str(config_path))

    # Display configuration summary
    print(f"\n✅ Configuration loaded successfully!")
    print(f"\n📋 Basic Settings:")
    print(f"  - Name: {config.basic_settings.get('name')}")
    print(f"  - Language: {config.basic_settings.get('language')}")
    print(f"  - Tone: {config.basic_settings.get('tone')}")

    print(f"\n🔗 URLs:")
    print(f"  - Flow URL: {config.flow_url or 'Not configured'}")
    print(f"  - KB URL: {config.retrieve_knowledge_url or 'Not configured'}")

    print(f"\n🎯 Skills: {len(config.skills)}")
    for i, skill in enumerate(config.skills, 1):
        print(f"  {i}. Condition: {skill.condition[:60]}...")
        print(f"     Tools: {', '.join(skill.tools)}")

    print(f"\n🔧 System Tools: {len(config.system_tools)}")
    for tool in config.system_tools:
        print(f"  - {tool.get('name')}: {tool.get('description', 'No description')[:50]}...")

    # Get all tool names
    tool_names = get_tool_names_from_config(config)
    print(f"\n📝 All Available Tools ({len(tool_names)}):")
    for name in tool_names:
        print(f"  - {name}")

    # Validate skill tools
    missing_tools = validate_skill_tools(config)
    if missing_tools:
        print(f"\n⚠️  Warning: Some skills reference missing tools:")
        for condition, missing in missing_tools.items():
            print(f"  - Skill: {condition[:50]}...")
            print(f"    Missing: {', '.join(missing)}")
    else:
        print(f"\n✅ All skill tools are valid!")


async def demo_orchestrator_creation():
    """Demonstrate orchestrator creation with dynamic tools."""
    print("\n" + "=" * 80)
    print("Orchestrator Creation with Dynamic Tools")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent.parent / "docs" / "configs" / "sop.json"
    config = load_workflow_config_from_file(str(config_path))

    # Set up context variables
    context_vars = {
        "dialogId": "dialog_12345",
        "tenantId": "tenant_67890",
        "chatbotId": config.basic_settings.get("chatbot_id"),
        "phoneNumber": "+1234567890",
    }

    print(f"\n🔧 Context Variables:")
    for key, value in context_vars.items():
        print(f"  - {key}: {value}")

    # Create orchestrator (all tools loaded dynamically)
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars=context_vars,
    )

    # Get all dynamically loaded tools
    tools = orchestrator.get_tools()

    print(f"\n✅ Orchestrator created with {len(tools)} dynamically loaded tools:")
    for tool in tools:
        print(f"\n  📦 {tool.name}")
        print(f"     Description: {tool.description}")
        print(f"     Endpoint: {tool.config.endpoint.url}")
        print(f"     Method: {tool.config.endpoint.method}")

        # Show parameters
        params = tool.config.parameters
        if params.properties:
            print(f"     Parameters:")
            for param_name, param_def in params.properties.items():
                required = " (required)" if param_name in params.required else ""
                print(f"       - {param_name}: {param_def.type}{required}")


async def demo_skill_tool_mapping():
    """Demonstrate skill to tool mapping."""
    print("\n" + "=" * 80)
    print("Skill to Tool Mapping")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent.parent / "docs" / "configs" / "sop.json"
    config = load_workflow_config_from_file(str(config_path))

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars={"dialogId": "123", "tenantId": "456"},
    )

    # Show skill to tool mappings
    print(f"\n🎯 Skill to Tool Mappings:")
    for i, skill in enumerate(config.skills, 1):
        print(f"\n  Skill {i}:")
        print(f"  Condition: {skill.condition[:70]}...")
        print(f"  Action: {skill.action[:70]}...")

        # Get tools for this skill
        skill_tools = orchestrator.get_skill_tools(skill.condition)
        print(f"  Tools ({len(skill_tools)}):")
        for tool in skill_tools:
            print(f"    - {tool.name}: {tool.description[:50]}...")


async def demo_system_prompt_generation():
    """Demonstrate system prompt generation."""
    print("\n" + "=" * 80)
    print("System Prompt Generation")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent.parent / "docs" / "configs" / "sop.json"
    config = load_workflow_config_from_file(str(config_path))

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars={},
    )

    # Build system prompt
    system_prompt = orchestrator.build_system_prompt(include_skills=True)

    print(f"\n📝 Generated System Prompt ({len(system_prompt)} characters):")
    print("\n" + "-" * 80)
    print(system_prompt)
    print("-" * 80)


async def demo_tool_lookup():
    """Demonstrate tool lookup by name."""
    print("\n" + "=" * 80)
    print("Tool Lookup by Name")
    print("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent.parent / "docs" / "configs" / "sop.json"
    config = load_workflow_config_from_file(str(config_path))

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars={"dialogId": "123", "tenantId": "456"},
    )

    # Lookup specific tools
    tool_names_to_lookup = [
        "save_customer_information",
        "handoff_to",
        "retrieve_knowledge",
        "close_conversation",
    ]

    print(f"\n🔍 Looking up tools by name:")
    for tool_name in tool_names_to_lookup:
        tool = orchestrator.get_tool_by_name(tool_name)
        if tool:
            print(f"\n  ✅ Found: {tool_name}")
            print(f"     Description: {tool.description}")
            print(f"     Endpoint: {tool.config.endpoint.url}")
            print(f"     Parameters: {len(tool.config.parameters.properties)} params")
        else:
            print(f"\n  ❌ Not found: {tool_name}")


async def demo_arbitrary_config():
    """Demonstrate that ANY tool configuration works (no hardcoding)."""
    print("\n" + "=" * 80)
    print("Arbitrary Tool Configuration (No Hardcoding)")
    print("=" * 80)

    # Create a completely custom configuration
    custom_config = {
        "basic_settings": {
            "name": "Custom Agent",
            "description": "Demonstrates arbitrary tool support",
            "language": "English",
        },
        "flow_url": "http://custom-flow.example.com/api",
        "retrieve_knowledge_url": "http://custom-kb.example.com/search",
        "skills": [
            {
                "condition": "User wants to do something custom",
                "action": "Use custom tools",
                "tools": ["custom_tool_1", "custom_tool_2"]
            }
        ],
        "system_tools": [
            {
                "name": "custom_tool_1",
                "description": "A completely custom tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Custom input"
                        }
                    },
                    "required": ["input"]
                },
                "endpoint": {
                    "url": "http://custom-api.example.com/tool1",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"data": "{input}"}
                }
            },
            {
                "name": "custom_tool_2",
                "description": "Another custom tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"}
                    },
                    "required": ["param1"]
                },
                "endpoint": {
                    "url": "http://custom-api.example.com/tool2",
                    "method": "PUT",
                    "headers": {"Authorization": "Bearer {token}"},
                    "body": {"p1": "{param1}", "p2": "{param2}"}
                }
            }
        ]
    }

    print(f"\n📝 Custom Configuration:")
    print(json.dumps(custom_config, indent=2))

    # Load custom configuration
    from bu_agent_sdk.workflow import load_workflow_config

    config = load_workflow_config(custom_config)

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars={"token": "secret123"},
    )

    # Get tools
    tools = orchestrator.get_tools()

    print(f"\n✅ Successfully loaded {len(tools)} custom tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print(f"\n🎉 This demonstrates that ANY tool configuration works!")
    print(f"   NO hardcoding required - fully dynamic!")


if __name__ == "__main__":
    print("=" * 80)
    print("Fully Dynamic Configuration Examples")
    print("=" * 80)

    # Run all demos
    asyncio.run(demo_dynamic_loading())
    asyncio.run(demo_orchestrator_creation())
    asyncio.run(demo_skill_tool_mapping())
    asyncio.run(demo_system_prompt_generation())
    asyncio.run(demo_tool_lookup())
    asyncio.run(demo_arbitrary_config())

    print("\n" + "=" * 80)
    print("Summary:")
    print("✅ All tools loaded dynamically from configuration")
    print("✅ NO hardcoded tools in the code")
    print("✅ Supports arbitrary tool configurations")
    print("✅ Skills mapped to tools via tool names")
    print("✅ Flow and KB URLs automatically handled")
    print("=" * 80)
