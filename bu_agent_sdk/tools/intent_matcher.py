"""
Intent matcher - hybrid strategy (rule-based + LLM).

Based on workflow-agent-v9.md design.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from bu_agent_sdk.llm.base import BaseChatModel
from bu_agent_sdk.llm.messages import SystemMessage, UserMessage
from bu_agent_sdk.tools.actions import (
    ActionType,
    FlowDefinition,
    WorkflowConfigSchema,
)


@dataclass
class IntentMatchResult:
    """Intent match result."""
    matched: bool
    action_type: ActionType | None
    action_target: str | None
    parameters: dict
    confidence: float
    reasoning: str | None = None


class IntentMatcher:
    """
    Intent matcher - hybrid strategy.

    Strategy:
    1. Rule matching (Flow regex patterns) - fast, precise
    2. LLM matching (Skills/Tools/Message) - flexible, intelligent
    """

    def __init__(self, config: WorkflowConfigSchema, llm: BaseChatModel):
        self.config = config
        self.llm = llm
        self._compile_rules()

    def _compile_rules(self):
        """Pre-compile regex rules."""
        self.flow_patterns: list[tuple[re.Pattern, FlowDefinition]] = []

        for flow in self.config.flows:
            for pattern in flow.trigger_patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    self.flow_patterns.append((compiled, flow))
                except re.error as e:
                    # Log error but don't interrupt
                    print(f"⚠️  Invalid regex pattern '{pattern}': {e}")

    async def match(
        self,
        user_message: str,
        context: list[Any] | None = None
    ) -> IntentMatchResult:
        """
        Match user intent.

        Flow:
        1. Rule matching Flows (high priority, deterministic)
        2. LLM matching Skills/Tools (flexibility)
        3. Default to Message (fallback)
        """

        # Phase 1: Rule matching Flows
        for pattern, flow in self.flow_patterns:
            if pattern.search(user_message):
                return IntentMatchResult(
                    matched=True,
                    action_type=ActionType.FLOW,
                    action_target=flow.flow_id,
                    parameters={},
                    confidence=1.0,
                    reasoning=f"Pattern matched: {pattern.pattern}"
                )

        # Phase 2: LLM matching
        return await self._llm_match(user_message, context or [])

    async def _llm_match(
        self,
        message: str,
        context: list[Any]
    ) -> IntentMatchResult:
        """Use LLM for intent matching."""

        # Build prompt
        system_prompt = self._build_matching_prompt()

        # Build messages (include history context)
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history (recent N turns)
        if context:
            recent_context = context[-5:]  # Keep only recent 5 turns
            messages.extend(recent_context)

        messages.append(UserMessage(content=message))

        # Call LLM (use JSON mode)
        response = await self.llm.ainvoke(
            messages=messages,
            tools=None,
            response_format={"type": "json_object"}
        )

        # Parse result
        try:
            result = json.loads(response.content or "{}")

            return IntentMatchResult(
                matched=result.get("matched", True),
                action_type=ActionType(result.get("action_type", "tool")),
                action_target=result.get("action_target", ""),
                parameters=result.get("parameters", {}),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning")
            )
        except Exception as e:
            # Parse failed, return default
            return IntentMatchResult(
                matched=False,
                action_type=None,
                action_target=None,
                parameters={},
                confidence=0.0,
                reasoning=f"Parse error: {e}"
            )

    def _build_matching_prompt(self) -> str:
        """Build intent matching system prompt."""

        # List available Skills
        skills_list = "\n".join([
            f"  - {s.skill_id}: {s.description}"
            for s in self.config.skills
        ])

        # List available Tools
        tools_list = "\n".join([
            f"  - {t['name']}: {t.get('description', '')}"
            for t in self.config.tools
        ])

        prompt = f"""You are an intent matcher for a workflow agent.

## Available Resources

### Skills (Complex multi-step tasks)
{skills_list or "  (none)"}

### Tools (Single function calls)
{tools_list or "  (none)"}

## Your Task

Analyze the user's message and determine:
1. Which action type is most appropriate: skill, tool, or message
2. Which specific skill/tool to call (if applicable)
3. What parameters to pass

## Output Format (JSON)

{{
  "matched": true,
  "action_type": "skill" | "tool" | "message",
  "action_target": "skill_id or tool_name (empty if message)",
  "parameters": {{}},
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}}

## Rules

- If user needs a complex multi-step task → use "skill"
- If user needs a simple single function → use "tool"
- If user is just chatting or asking questions → use "message" (action_target="")
- Always output valid JSON
"""
        return prompt
