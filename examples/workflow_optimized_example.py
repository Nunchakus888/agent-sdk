"""
Optimized Workflow Orchestrator Example

This example demonstrates the tool strategy:
- LLM-visible: Skills (function), Tools, KB → Tool module
- LLM-invisible: Flows, System actions → Manual execution
"""

import asyncio
import json
from pathlib import Path

from bu_agent_sdk.agent import Agent
from bu_agent_sdk.llm.anthropic import AnthropicChatModel
from bu_agent_sdk.workflow import (
    WorkflowOrchestrator,
    load_workflow_config,
)


# Example configuration following the optimized strategy
EXAMPLE_CONFIG = {
    "basic_settings": {
        "name": "YCloud Customer Service",
        "description": "Help customers with WhatsApp business services",
        "background": "YCloud is a leading WhatsApp business service provider",
        "language": "English",
        "tone": "Friendly and professional",
        "chatbot_id": "67adb3abaa26c063de0f4bd9"
    },

    # Skills (function mode) - Registered as HttpTool (LLM-visible)
    "skills": [
        {
            "skill_id": "sentiment_analysis",
            "name": "Sentiment Analysis",
            "description": "Analyze text sentiment and emotion",
            "execution_mode": "function",
            "endpoint": {
                "url": "http://api.example.com/sentiment",
                "method": "POST",
                "body": {"text": "{text}"}
            },
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    }
                },
                "required": ["text"]
            },
            "output_parser": "json"
        }
    ],

    # Regular tools - Registered as HttpTool (LLM-visible)
    "tools": [
        {
            "name": "save_customer_information",
            "description": "Save customer contact information",
            "parameters": {
                "type": "object",
                "properties": {
                    "nickName": {
                        "type": "string",
                        "description": "Customer name"
                    },
                    "email": {
                        "type": "string",
                        "description": "Customer email"
                    },
                    "dynamic_field_1": {
                        "type": "string",
                        "description": "Customer country",
                        "default": None
                    }
                },
                "required": ["email", "nickName"]
            },
            "endpoint": {
                "url": "http://172.16.92.21:18080/contactsService/add",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "nickName": "{nickName}",
                    "phoneNumber": "todo_autofill_by_system",
                    "email": "{email}",
                    "customAttrs": [
                        {
                            "key": "dynamic_field_1",
                            "value": {"dynamic_field_1": "{dynamic_field_1}"}
                        }
                    ]
                }
            }
        }
    ],

    # Knowledge retrieval - Registered as HttpTool (LLM-visible)
    "retrieve_knowledge_url": "http://121.43.165.245:18080/chatbot/ai-inner/retrieve-knowledge",

    # Flows - NOT registered as Tool (LLM-invisible, manual execution)
    "flows": [
        {
            "flow_id": "leave_request",
            "name": "Leave Request",
            "description": "Employee leave request process",
            "trigger_patterns": ["我要请假", "申请.*假"],
            "endpoint": {
                "url": "http://api.example.com/leave/submit",
                "method": "POST",
                "body": {
                    "user_id": "{user_id}",
                    "message": "{user_message}"
                }
            },
            "response_template": "✅ Leave request submitted\n\n{result}",
            "silent": False
        }
    ],

    # System actions - NOT registered as Tool (LLM-invisible, manual execution)
    "system_actions": [
        {
            "action_id": "transfer_human",
            "name": "Transfer to Human",
            "handler": "handoff",
            "silent": False,
            "response_template": "Transferring to human agent...",
            "endpoint": {
                "url": "http://172.16.80.52:8080/inboxConversationService/handoff",
                "method": "POST",
                "body": {
                    "assigneeId": "{assigneeId}",
                    "type": "{type}",
                    "dialogId": "todo_autofill_by_system"
                }
            }
        },
        {
            "action_id": "update_profile",
            "name": "Update Profile",
            "handler": "update_profile",
            "silent": True,
            "endpoint": {
                "url": "http://api.example.com/profile/update",
                "method": "POST",
                "body": {
                    "user_id": "{user_id}",
                    "data": "{data}"
                }
            }
        }
    ]
}


async def demo_tool_registration():
    """Demonstrate tool registration strategy."""
    print("=" * 80)
    print("Tool Registration Strategy Demo")
    print("=" * 80)

    # Load configuration
    config = load_workflow_config(EXAMPLE_CONFIG)

    # Set up context
    context_vars = {
        "dialogId": "dialog_12345",
        "tenantId": "tenant_67890",
        "chatbotId": config.basic_settings.get("chatbot_id"),
        "phoneNumber": "+1234567890",
    }

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars=context_vars,
    )

    # Get LLM-visible tools
    tools = orchestrator.get_tools()

    print(f"\n✅ LLM-Visible Tools (registered as Tool): {len(tools)}")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print(f"\n❌ LLM-Invisible Operations (manual execution):")
    print(f"  - Flows: {len(config.flows)}")
    for flow in config.flows:
        print(f"    - {flow.flow_id}: {flow.name} (silent={flow.silent})")

    print(f"  - System Actions: {len(config.system_actions)}")
    for action in config.system_actions:
        print(f"    - {action.action_id}: {action.name} (silent={action.silent})")

    # Build system prompt
    system_prompt = orchestrator.build_system_prompt()
    print(f"\n📝 System Prompt Preview:")
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)


async def demo_agent_with_tools():
    """Demonstrate agent with registered tools."""
    print("\n" + "=" * 80)
    print("Agent with Tools Demo")
    print("=" * 80)

    # Load configuration
    config = load_workflow_config(EXAMPLE_CONFIG)

    # Context variables
    context_vars = {
        "dialogId": "dialog_12345",
        "tenantId": "tenant_67890",
        "chatbotId": config.basic_settings.get("chatbot_id"),
    }

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars=context_vars,
    )

    # Get tools and system prompt
    tools = orchestrator.get_tools()
    system_prompt = orchestrator.build_system_prompt()

    print(f"\n✅ Created agent with {len(tools)} tools")
    print(f"✅ System prompt: {len(system_prompt)} characters")

    # Note: Actual agent creation requires API key
    # llm = AnthropicChatModel(model="claude-sonnet-4-5-20250929")
    # agent = Agent(llm=llm, tools=tools, system_prompt=system_prompt)


async def demo_manual_execution():
    """Demonstrate manual execution of flows and system actions."""
    print("\n" + "=" * 80)
    print("Manual Execution Demo (Flows & System Actions)")
    print("=" * 80)

    # Load configuration
    config = load_workflow_config(EXAMPLE_CONFIG)

    # Context variables
    context_vars = {
        "dialogId": "dialog_12345",
        "tenantId": "tenant_67890",
        "user_id": "user_123",
    }

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        config=config,
        context_vars=context_vars,
    )

    # Example 1: Execute flow (LLM-invisible)
    print("\n1. Execute Flow (leave_request):")
    print("   - NOT visible to LLM")
    print("   - Triggered by pattern matching or explicit routing")
    print("   - Returns response directly to user")

    # Note: Actual execution requires valid endpoint
    # result = await orchestrator.execute_flow(
    #     flow_id="leave_request",
    #     user_message="我要请假3天",
    #     parameters={"user_id": "user_123"}
    # )
    # print(f"   Result: {result}")

    # Example 2: Execute system action (LLM-invisible)
    print("\n2. Execute System Action (transfer_human):")
    print("   - NOT visible to LLM")
    print("   - Triggered by explicit routing")
    print("   - Can be silent (returns None)")

    # Note: Actual execution requires valid endpoint
    # result = await orchestrator.execute_system_action(
    #     action_id="transfer_human",
    #     parameters={"assigneeId": "agent_456", "type": "agent"}
    # )
    # print(f"   Result: {result}")

    # Example 3: Silent system action
    print("\n3. Execute Silent System Action (update_profile):")
    print("   - NOT visible to LLM")
    print("   - Returns None (silent)")
    print("   - Exits iteration loop immediately")

    # Note: Actual execution requires valid endpoint
    # result = await orchestrator.execute_system_action(
    #     action_id="update_profile",
    #     parameters={"user_id": "user_123", "data": {"country": "US"}}
    # )
    # print(f"   Result: {result}")  # Should be None


async def demo_comparison():
    """Compare Tool module vs Manual implementation."""
    print("\n" + "=" * 80)
    print("Tool Module vs Manual Implementation Comparison")
    print("=" * 80)

    print("\n✅ Use Tool Module (HttpTool) for:")
    print("  1. Skills (function mode)")
    print("     - LLM needs to see and decide when to call")
    print("     - Standard input/output")
    print("     - Result enters Agent context")
    print()
    print("  2. Regular Tools")
    print("     - LLM needs to see and decide when to call")
    print("     - Standard tool call pattern")
    print("     - Result enters Agent context")
    print()
    print("  3. Knowledge Retrieval")
    print("     - LLM needs to see and decide when to query")
    print("     - Standard tool call pattern")
    print("     - Result enters Agent context")

    print("\n❌ Use Manual Implementation for:")
    print("  1. Flows (Fixed Business Processes)")
    print("     - LLM should NOT see (triggered by patterns)")
    print("     - May need silent mode")
    print("     - Result goes directly to user, NOT Agent context")
    print("     - May need to exit iteration loop")
    print()
    print("  2. System Actions (handoff, close, update)")
    print("     - LLM should NOT see (triggered by routing)")
    print("     - Often need silent mode")
    print("     - Special business logic")
    print("     - May need to exit iteration loop")

    print("\n📊 Decision Matrix:")
    print("  LLM Visible + Standard Response → Tool Module")
    print("  LLM Invisible OR Silent Mode → Manual Implementation")


if __name__ == "__main__":
    print("=" * 80)
    print("Optimized Workflow Orchestrator Examples")
    print("=" * 80)

    # Run demos
    asyncio.run(demo_tool_registration())
    asyncio.run(demo_agent_with_tools())
    asyncio.run(demo_manual_execution())
    asyncio.run(demo_comparison())

    print("\n" + "=" * 80)
    print("Summary:")
    print("- Skills (function), Tools, KB → HttpTool (LLM-visible)")
    print("- Flows, System Actions → Manual execution (LLM-invisible)")
    print("- Silent actions return None to exit iteration")
    print("=" * 80)
