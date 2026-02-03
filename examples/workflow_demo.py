"""
Workflow Agent Demo - Basic usage example.

Based on workflow-agent-v9.md design.
"""

import asyncio
import json
from pathlib import Path

from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.tools.actions import WorkflowConfigSchema
from bu_agent_sdk.llm import ChatOpenAI


async def main():
    """Demo workflow agent usage."""

    # 1. Load configuration
    config_path = Path(__file__).parent.parent / "docs" / "configs" / "workflow_config.json"

    # For demo, create a minimal config
    config_data = {
        "basic_settings": {
            "name": "Demo Assistant",
            "description": "A demo workflow assistant",
            "language": "English",
            "tone": "Friendly and professional"
        },
        "greeting": "Hello! I'm your demo assistant. How can I help you today?",
        "tools": [
            {
                "name": "search_weather",
                "description": "Search weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["city"]
                },
                "endpoint": {
                    "url": "http://api.example.com/weather",
                    "method": "GET",
                    "query_params": {
                        "city": "{city}"
                    }
                }
            }
        ],
        "skills": [],
        "flows": [],
        "system_actions": [],
        "action_books": [],
        "sop": "1. Understand user needs\n2. Match appropriate action\n3. Execute and provide feedback",
        "max_iterations": 5,
        "iteration_strategy": "sop_driven"
    }

    config = WorkflowConfigSchema(**config_data)

    # 2. Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # 3. Create WorkflowAgent
    agent = WorkflowAgent(
        config=config,
        llm=llm,
    )

    # 4. Interactive dialogue
    session_id = "demo_session_001"

    print("=" * 60)
    print("Workflow Agent Demo")
    print("=" * 60)
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = await agent.query(
                message=user_input,
                session_id=session_id
            )
            print(f"\n🤖 Assistant: {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
