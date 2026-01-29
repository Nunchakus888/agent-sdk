"""
Workflow executors for handling different types of actions.

Based on workflow-agent-v9.md design.

Design Philosophy:
- SkillExecutor: Agent mode (sub-agent) + Function mode (HTTP call)
- FlowExecutor: Direct API call, black box service
- SystemExecutor: System actions with silent mode support
- TimerScheduler: Asyncio-based timer scheduling
- KBEnhancer: KB parallel query optimization
"""

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel

from bu_agent_sdk import Agent
from bu_agent_sdk.agent import TaskComplete
from bu_agent_sdk.tools import tool
from bu_agent_sdk.llm.base import BaseChatModel
from bu_agent_sdk.tools.actions import (
    SkillDefinition,
    FlowDefinition,
    SystemAction,
    TimerConfig,
    WorkflowConfigSchema,
)


# =============================================================================
# Skill Executor
# =============================================================================


class SkillExecutor:
    """
    Skill executor - supports multiple execution modes.

    Modes:
    1. Agent mode - Create sub-agent, supports multi-turn dialogue
    2. Function mode - Direct HTTP call to external service (single I/O)
    """

    def __init__(self, config: WorkflowConfigSchema, llm: BaseChatModel):
        self.config = config
        self.llm = llm
        self._skill_agents: dict[str, Agent] = {}
        self._http_client: httpx.AsyncClient | None = None

    async def execute(
        self,
        skill_id: str,
        user_request: str,
        parameters: dict,
    ) -> str:
        """Execute skill."""

        # Find skill definition
        skill = next(
            (s for s in self.config.skills if s.skill_id == skill_id),
            None
        )
        if not skill:
            return f"❌ Skill not found: {skill_id}"

        # Dispatch by execution mode
        if skill.execution_mode == "agent":
            return await self._execute_agent_mode(skill, user_request, parameters)
        elif skill.execution_mode == "function":
            return await self._execute_function_mode(skill, user_request, parameters)
        else:
            return f"❌ Unknown execution mode: {skill.execution_mode}"

    async def _execute_agent_mode(
        self,
        skill: SkillDefinition,
        user_request: str,
        parameters: dict,
    ) -> str:
        """Agent mode - create sub-agent execution."""

        if not skill.system_prompt:
            return f"❌ Agent mode missing system_prompt config"

        # Get or create sub-agent
        agent = self._get_or_create_agent(skill)

        # Execute
        try:
            result = await agent.query(user_request)
            return f"✅ [{skill.name}] Completed\n\n{result}"
        except TaskComplete as e:
            return f"✅ [{skill.name}] Completed\n\n{e.message}"
        except Exception as e:
            return f"❌ [{skill.name}] Execution failed: {e}"

    async def _execute_function_mode(
        self,
        skill: SkillDefinition,
        user_request: str,
        parameters: dict,
    ) -> str:
        """Function mode - direct HTTP call."""

        if not skill.endpoint:
            return f"❌ Function mode missing endpoint config"

        try:
            # Prepare request
            endpoint = skill.endpoint
            url = endpoint.get("url", "")
            method = endpoint.get("method", "POST")
            headers = endpoint.get("headers", {"Content-Type": "application/json"})

            # Prepare request body (parameter substitution)
            body_template = endpoint.get("body", {})
            body = self._substitute_parameters(
                body_template,
                {"input": user_request, **parameters}
            )

            # Send HTTP request
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=30.0)

            response = await self._http_client.request(
                method=method,
                url=url,
                headers=headers,
                json=body,
            )

            # Parse response
            if response.is_success:
                result = self._parse_response(response, skill.output_parser or "text")
                return f"✅ [{skill.name}] Completed\n\n{result}"
            else:
                return f"❌ [{skill.name}] Call failed: HTTP {response.status_code}"

        except Exception as e:
            return f"❌ [{skill.name}] Execution failed: {e}"

    def _get_or_create_agent(self, skill: SkillDefinition) -> Agent:
        """Get or create skill agent (cached reuse)."""

        if skill.skill_id in self._skill_agents:
            return self._skill_agents[skill.skill_id]

        # Create done tool
        @tool("Mark task as complete")
        async def done(message: str) -> str:
            raise TaskComplete(message)

        # TODO: Load skill's tool list
        skill_tools = [done]

        # Create agent
        agent = Agent(
            llm=self.llm,
            tools=skill_tools,
            system_prompt=skill.system_prompt or "",
            max_iterations=skill.max_iterations,
            require_done_tool=skill.require_done_tool,
        )

        self._skill_agents[skill.skill_id] = agent
        return agent

    def _substitute_parameters(self, template: Any, params: dict) -> Any:
        """Substitute parameter placeholders."""
        if isinstance(template, str):
            # Replace {param_name} placeholders
            for key, value in params.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
        elif isinstance(template, dict):
            return {k: self._substitute_parameters(v, params) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_parameters(item, params) for item in template]
        else:
            return template

    def _parse_response(self, response: Any, parser: str) -> str:
        """Parse response."""
        if parser == "json":
            try:
                data = response.json()
                return json.dumps(data, ensure_ascii=False, indent=2)
            except Exception:
                return response.text
        else:
            return response.text


# =============================================================================
# Flow Executor
# =============================================================================


class FlowExecutor:
    """
    Flow executor - direct API call, don't care about internal logic.

    Design philosophy:
    - Flow is a black box service, only care about input and output
    - Don't maintain state machine, don't manage steps
    - Suitable for existing business process APIs
    """

    def __init__(self, config: WorkflowConfigSchema):
        self.config = config
        self._http_client: httpx.AsyncClient | None = None

    async def execute(
        self,
        flow_id: str,
        user_message: str,
        parameters: dict,
        session: Any,
    ) -> str:
        """Execute flow."""

        flow = self._get_flow(flow_id)
        if not flow:
            return f"❌ Flow not found: {flow_id}"

        try:
            # Prepare request
            endpoint = flow.endpoint
            url = endpoint.get("url", "")
            method = endpoint.get("method", "POST")
            headers = endpoint.get("headers", {"Content-Type": "application/json"})

            # Build request body
            body_template = endpoint.get("body", {})
            body = self._substitute_parameters(
                body_template,
                {
                    "user_message": user_message,
                    "session_id": session.session_id,
                    **parameters,
                }
            )

            # Send request
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=30.0)

            response = await self._http_client.request(
                method=method,
                url=url,
                headers=headers,
                json=body,
            )

            # Handle response
            flow_name = flow.get_identifier()
            if response.is_success:
                result_text = self._parse_response(response)

                # Use response template (if configured)
                if flow.response_template:
                    return flow.response_template.replace("{result}", result_text)
                else:
                    return f"✅ [{flow_name}] Execution completed\n\n{result_text}"
            else:
                return f"❌ [{flow_name}] Execution failed: HTTP {response.status_code}\n{response.text}"

        except Exception as e:
            flow_name = flow.get_identifier()
            return f"❌ [{flow_name}] Execution failed: {e}"

    def _get_flow(self, flow_id: str) -> FlowDefinition | None:
        """Find flow definition by flow_id or name."""
        return next(
            (f for f in self.config.flows if f.flow_id == flow_id or f.name == flow_id),
            None
        )

    def _substitute_parameters(self, template: Any, params: dict) -> Any:
        """Substitute parameter placeholders."""
        if isinstance(template, str):
            for key, value in params.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
        elif isinstance(template, dict):
            return {k: self._substitute_parameters(v, params) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_parameters(item, params) for item in template]
        else:
            return template

    def _parse_response(self, response: Any) -> str:
        """Parse response."""
        try:
            data = response.json()
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return response.text


# =============================================================================
# System Executor
# =============================================================================


class SystemExecutor:
    """
    System action executor.

    Handles: handoff, close conversation, update info, etc.
    """

    def __init__(self, config: WorkflowConfigSchema):
        self.config = config

    async def execute(self, action_id: str, parameters: dict) -> str | None:
        """Execute system action."""

        action = next(
            (a for a in self.config.system_actions if a.action_id == action_id),
            None
        )
        if not action:
            return f"❌ System action not found: {action_id}"

        # Execute by handler type
        if action.handler == "handoff":
            result = await self._handoff(action, parameters)
        elif action.handler == "close":
            result = await self._close_conversation(action, parameters)
        elif action.handler == "update_profile":
            result = await self._update_profile(action, parameters)
        else:
            return f"❌ Unknown system action type: {action.handler}"

        # Silent mode: return None
        if action.silent:
            return None

        # Non-silent: return response
        return result

    async def _handoff(self, action: SystemAction, parameters: dict) -> str:
        """Handoff to human."""
        response = action.response_template or "Transferring to human service..."
        # TODO: Actual handoff logic
        return response

    async def _close_conversation(self, action: SystemAction, parameters: dict) -> str:
        """Close conversation."""
        return "Conversation closed, thank you!"

    async def _update_profile(self, action: SystemAction, parameters: dict) -> str:
        """Update user info."""
        # TODO: Actual update logic
        return "Information updated"


# =============================================================================
# Timer Scheduler
# =============================================================================


class TimerScheduler:
    """
    Timer scheduler - based on asyncio.

    Features:
    - Manage session-level timer tasks
    - Auto-trigger Action on expiration
    - Support cancel and reschedule
    """

    def __init__(self, workflow_agent: Any):
        self.workflow_agent = workflow_agent
        self._tasks: dict[str, asyncio.Task] = {}

    async def schedule(self, session_id: str, timers: list[TimerConfig]) -> None:
        """Register timers for session."""
        # Cancel existing timers
        await self.cancel_session_timers(session_id)

        # Register new timers
        for timer in timers:
            task_key = f"{session_id}:{timer.timer_id}"
            task = asyncio.create_task(
                self._delayed_trigger(session_id, timer)
            )
            self._tasks[task_key] = task

    async def _delayed_trigger(self, session_id: str, timer: TimerConfig) -> None:
        """Delayed trigger timer."""
        try:
            await asyncio.sleep(timer.delay_seconds)

            # Trigger Action
            await self.workflow_agent.query(
                message=timer.message or f"[Timer:{timer.timer_id}]",
                session_id=session_id
            )
        except asyncio.CancelledError:
            # Timer cancelled
            pass
        except Exception as e:
            print(f"⚠️  Timer execution failed: {e}")

    async def cancel_session_timers(self, session_id: str) -> None:
        """Cancel all timers for session."""
        keys_to_remove = [k for k in self._tasks if k.startswith(f"{session_id}:")]
        for key in keys_to_remove:
            task = self._tasks.pop(key)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# =============================================================================
# KB Enhancer
# =============================================================================


class KBEnhancer:
    """
    Knowledge base enhancer.

    Features:
    - Parallel pre-query KB (non-blocking main flow)
    - Cache query results, use directly when needed
    - Improve answer accuracy, shorten response time
    """

    def __init__(self, config: WorkflowConfigSchema, kb_tool: Any | None = None):
        self.config = config
        self.kb_tool = kb_tool
        self.enabled = False

        if config.kb_config:
            self.enabled = config.kb_config.get("enabled", False)
            self.auto_enhance = config.kb_config.get("auto_enhance", True)
            self.enhance_conditions = config.kb_config.get("enhance_conditions", [])

    async def query_kb(self, user_message: str) -> str:
        """
        Parallel query KB.

        Optimization: Called at iteration start, doesn't block LLM decision.
        """
        if not self.enabled or not self.kb_tool:
            return ""

        try:
            kb_result = await self.kb_tool.execute(query=user_message)
            return kb_result
        except Exception as e:
            # KB query failed, return empty result
            print(f"⚠️  KB query failed: {e}")
            return ""

    async def enhance(
        self,
        user_message: str,
        action_type: str,
        execution_result: str
    ) -> str:
        """Enhance execution result (backward compatible method)."""
        if not self.enabled or not self.kb_tool:
            return execution_result

        # Check if enhancement needed
        if not self.auto_enhance:
            return execution_result

        if self.enhance_conditions and action_type not in self.enhance_conditions:
            return execution_result

        # Query KB
        try:
            kb_result = await self.kb_tool.execute(query=user_message)

            # Merge results
            enhanced = f"{execution_result}\n\n📚 Related knowledge:\n{kb_result}"
            return enhanced
        except Exception as e:
            # KB query failed, return original result
            print(f"⚠️  KB enhancement failed: {e}")
            return execution_result
