"""
Workflow Agent - Configuration-driven workflow orchestrator.

Based on workflow-agent-v9.md design.

Core responsibilities:
1. Load and parse configuration
2. Intent matching and routing
3. Dynamically create sub-agents (Skills)
4. State management (greeting, hash detection, flow state)
5. Executor coordination (Skill/Flow/System)
"""

import asyncio
import json
from typing import Any

from bu_agent_sdk.llm.base import BaseChatModel
from bu_agent_sdk.llm.messages import SystemMessage, UserMessage
from bu_agent_sdk.tools.action_books import (
    IterationDecision,
    WorkflowConfigSchema,
)
from bu_agent_sdk.tools.config_loader import ConfigToolLoader, HttpTool, ToolConfig, AgentConfigSchema
from bu_agent_sdk.tools.intent_matcher import IntentMatcher
from bu_agent_sdk.agent.workflow_state import WorkflowState, Session
from bu_agent_sdk.workflow.cache import PlanCache, MemoryPlanCache, compute_config_hash
from bu_agent_sdk.workflow.executors import (
    SkillExecutor,
    FlowExecutor,
    SystemExecutor,
    TimerScheduler,
    KBEnhancer,
)


class WorkflowAgent:
    """
    Configuration-driven workflow agent.

    Usage example:
        ```python
        config = WorkflowConfigSchema.parse_file("config.json")
        agent = WorkflowAgent(
            config=config,
            llm=ChatOpenAI(model="gpt-4o"),
        )

        response = await agent.query(
            message="Help me write a blog",
            session_id="session_123"
        )
        ```
    """

    def __init__(
        self,
        config: WorkflowConfigSchema,
        llm: BaseChatModel,
        plan_cache: PlanCache | None = None,
        session_store: Any | None = None,
    ):
        self.config = config
        self.llm = llm
        self.plan_cache = plan_cache or MemoryPlanCache()
        self.session_store = session_store  # Optional persistent storage

        # Compute config hash
        self.config_hash = compute_config_hash(config)

        # Create components
        self.intent_matcher = IntentMatcher(config, llm)
        self.skill_executor = SkillExecutor(config, llm)
        self.flow_executor = FlowExecutor(config)
        self.system_executor = SystemExecutor(config)

        # Memory session cache (optional: connect to database)
        self._sessions: dict[str, Session] = {}

        # Base tools (loaded from config)
        self._base_tools = self._load_base_tools()

        # KB enhancer
        kb_tool = self._get_kb_tool()
        self.kb_enhancer = KBEnhancer(config, kb_tool)

        # Timer scheduler
        self.timer_scheduler = TimerScheduler(self)

    def _load_base_tools(self) -> list:
        """Load base tools (HTTP API tools)."""
        if not self.config.tools:
            return []

        # Use ConfigToolLoader to load HTTP tools
        tools = []
        for tool_dict in self.config.tools:
            tool_config = ToolConfig(**tool_dict)
            http_tool = HttpTool(config=tool_config)
            tools.append(http_tool)

        return tools

    def _get_kb_tool(self) -> Any | None:
        """Get KB tool."""
        if not self.config.kb_config:
            return None

        tool_name = self.config.kb_config.get("tool_name", "search_kb")
        return next((t for t in self._base_tools if t.name == tool_name), None)

    async def _get_or_create_session(self, session_id: str) -> Session:
        """Get or create session."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Create new session
        session = Session(
            session_id=session_id,
            agent_id=f"workflow_{self.config_hash[:8]}",
            workflow_state=WorkflowState(),
            messages=[],
        )

        self._sessions[session_id] = session
        return session

    async def _check_config_change(self, session: Session) -> bool:
        """Check if config changed."""
        if session.workflow_state.config_hash != self.config_hash:
            # Config changed, reset state
            session.workflow_state = WorkflowState(config_hash=self.config_hash)
            return True
        return False

    async def _send_greeting(self, session: Session) -> str | None:
        """Send greeting message."""
        if session.workflow_state.need_greeting and self.config.greeting:
            session.workflow_state.need_greeting = False
            return self.config.greeting
        return None

    async def query(
        self,
        message: str,
        session_id: str,
    ) -> str:
        """
        Process user message - SOP-driven multi-step execution.

        Flow:
        1. Load session state
        2. Check config change
        3. Send greeting (first time)
        4. SOP-driven iteration execution
           - LLM decides next step
           - Execute Action
           - Update context
           - Determine whether to continue
        5. KB enhancement (optional)
        6. Register Timer (optional)
        7. Return result
        """

        # 1. Get session
        session = await self._get_or_create_session(session_id)

        # 2. Check config change
        config_changed = await self._check_config_change(session)
        if config_changed:
            # Config changed, clear old timers
            await self.timer_scheduler.cancel_session_timers(session_id)

        # 3. Send greeting
        greeting = await self._send_greeting(session)
        if greeting:
            # Greeting returned directly, no intent matching
            return greeting

        # Clear execution history for new query
        session.clear_execution_history()

        # 4. SOP-driven iteration execution
        if self.config.iteration_strategy == "single_shot":
            # Single execution mode (backward compatible)
            response = await self._single_shot_execution(message, session)
        else:
            # SOP-driven multi-step execution
            response = await self._sop_driven_execution(message, session)

        # 5. Register Timer (optional)
        if self.config.timers:
            await self.timer_scheduler.schedule(session_id, self.config.timers)

        return response

    async def _sop_driven_execution(
        self,
        message: str,
        session: Session
    ) -> str:
        """
        SOP-driven multi-step execution.

        Core logic:
        1. Parallel pre-query KB at iteration start (if enabled)
        2. LLM decides next step
        3. Execute Action and update context
        4. Silent Action (Flow/System) exits iteration directly
        5. Determine whether to continue or respond
        """
        max_iterations = self.config.max_iterations
        kb_cache = {}  # KB query result cache

        for iteration in range(max_iterations):
            # 1. KB pre-query (parallel execution, non-blocking decision)
            kb_task = None
            if self.kb_enhancer.enabled and iteration == 0:
                # Start KB query task on first iteration
                kb_task = asyncio.create_task(
                    self.kb_enhancer.query_kb(message)
                )

            # 2. LLM decision: what to do next
            decision = await self._llm_decide(
                session=session,
                user_message=message,
                iteration=iteration
            )

            # Record decision
            session.add_decision(iteration, decision)

            # 3. Determine if should respond
            if decision.should_respond:
                # Wait for KB query to complete (if any)
                if kb_task:
                    kb_cache['result'] = await kb_task

                # Generate final response (use KB cache)
                response = await self._generate_response(
                    session=session,
                    decision=decision,
                    kb_cache=kb_cache
                )
                return response

            # 4. Execute Action
            if decision.next_action:
                action_type = decision.next_action.get("type")

                result = await self._execute_action_from_decision(
                    action_dict=decision.next_action,
                    session=session
                )

                # 5. Determine if Silent Action
                is_silent = await self._is_silent_action(
                    action_type=action_type,
                    action_target=decision.next_action.get("target")
                )

                if is_silent:
                    # Silent Action: exit iteration directly, don't participate in context
                    # Wait for KB query to complete (if any)
                    if kb_task:
                        kb_cache['result'] = await kb_task

                    response = await self._generate_response(
                        session=session,
                        kb_cache=kb_cache
                    )
                    return response
                else:
                    # Non-silent: update context, continue iteration
                    session.add_execution_result(
                        iteration=iteration,
                        action=decision.next_action,
                        result=result,
                        reasoning=decision.reasoning
                    )

            # 6. Determine whether to continue
            if not decision.should_continue:
                # Wait for KB query to complete (if any)
                if kb_task:
                    kb_cache['result'] = await kb_task

                response = await self._generate_response(
                    session=session,
                    kb_cache=kb_cache
                )
                return response

        # Reached max iterations, generate response
        if kb_task:
            kb_cache['result'] = await kb_task

        return await self._generate_response(session, kb_cache=kb_cache)

    async def _single_shot_execution(
        self,
        message: str,
        session: Session
    ) -> str:
        """Single execution mode (backward compatible)."""
        # Intent matching
        intent = await self.intent_matcher.match(message, session.messages)

        # Route execution
        if not intent.matched or intent.action_type is None:
            return self._generate_fallback_response(message)

        result = await self._dispatch(
            action_type=intent.action_type,
            action_target=intent.action_target or "",
            parameters=intent.parameters,
            user_message=message,
            session=session,
        )

        # Handle silent operation
        if result is None:
            return ""

        return result

    async def _llm_decide(
        self,
        session: Session,
        user_message: str,
        iteration: int
    ) -> IterationDecision:
        """
        LLM decision: determine next action.

        Key: Re-evaluate current state on each iteration.
        """
        # Build decision prompt
        system_prompt = f"""You are following this SOP:

{self.config.sop or "No specific SOP provided"}

## Current Situation
- User message: {user_message}
- Iteration: {iteration + 1}/{self.config.max_iterations}
- Execution history: {session.get_execution_summary()}

## Available Actions
{self._format_available_actions()}

## Your Task
Decide the next action based on the SOP and current situation.

## Output Format (JSON)
{{
  "should_continue": true/false,
  "should_respond": true/false,
  "next_action": {{"type": "skill|tool|flow|system", "target": "action_id", "params": {{}}}},
  "reasoning": "Why this decision"
}}

## Decision Rules
1. If user's need is satisfied → should_respond=true, should_continue=false
2. If need more actions → should_respond=false, should_continue=true, provide next_action
3. If uncertain or need clarification → should_respond=true (ask user)
4. If reached max iterations → should_respond=true
5. Follow the SOP step by step

## Constraints
{self.config.constraints or "No specific constraints"}
"""

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=f"Current user message: {user_message}")
        ]

        # Call LLM
        response = await self.llm.ainvoke(
            messages=messages,
            response_format={"type": "json_object"}
        )

        # Parse result
        try:
            result = json.loads(response.content or "{}")

            return IterationDecision(
                should_continue=result.get("should_continue", False),
                should_respond=result.get("should_respond", True),
                next_action=result.get("next_action"),
                reasoning=result.get("reasoning", "No reasoning provided")
            )
        except Exception as e:
            # Parse failed, default to respond
            return IterationDecision(
                should_continue=False,
                should_respond=True,
                next_action=None,
                reasoning=f"Parse error: {e}"
            )

    def _format_available_actions(self) -> str:
        """Format available Actions list."""
        lines = []

        # Skills
        if self.config.skills:
            lines.append("### Skills")
            for skill in self.config.skills:
                lines.append(f"  - {skill.skill_id}: {skill.description}")

        # Tools
        if self.config.tools:
            lines.append("\n### Tools")
            for tool in self.config.tools:
                lines.append(f"  - {tool['name']}: {tool.get('description', '')}")

        # Flows
        if self.config.flows:
            lines.append("\n### Flows")
            for flow in self.config.flows:
                lines.append(f"  - {flow.flow_id}: {flow.description}")

        # System Actions
        if self.config.system_actions:
            lines.append("\n### System Actions")
            for action in self.config.system_actions:
                lines.append(f"  - {action.action_id}: {action.name}")

        return "\n".join(lines) if lines else "No actions available"

    async def _execute_action_from_decision(
        self,
        action_dict: dict,
        session: Session
    ) -> str:
        """Execute Action from decision result."""
        action_type = action_dict.get("type")
        action_target = action_dict.get("target")
        params = action_dict.get("params", {})

        return await self._dispatch(
            action_type=action_type,
            action_target=action_target,
            parameters=params,
            user_message="",  # Already in context
            session=session
        )

    async def _is_silent_action(
        self,
        action_type: str,
        action_target: str
    ) -> bool:
        """
        Determine if Silent Action.

        Silent Action definition:
        - Flow: All Flows are silent (business process API, don't participate in context)
        - System: Determined by silent field in config
        """
        if action_type == "flow":
            # All Flows are silent
            return True

        if action_type == "system":
            # Find System Action config
            action = next(
                (a for a in self.config.system_actions if a.action_id == action_target),
                None
            )
            if action:
                return action.silent

        return False

    async def _generate_response(
        self,
        session: Session,
        decision: IterationDecision | None = None,
        kb_cache: dict | None = None
    ) -> str:
        """
        Generate final response.

        Optimization: Use KB cache (if available), avoid repeated queries.
        """
        # If decision contains response content, use directly
        if decision and hasattr(decision, 'response_content'):
            return decision.response_content

        # Build context (include KB results)
        context_parts = [f"## Execution History\n{session.get_execution_summary()}"]

        # Add KB results (if cached)
        if kb_cache and 'result' in kb_cache:
            context_parts.append(f"\n## Knowledge Base Results\n{kb_cache['result']}")

        # Otherwise generate response based on execution history
        system_prompt = f"""Generate a response to the user based on the execution history and knowledge base.

{chr(10).join(context_parts)}

## Guidelines
- Be concise and helpful
- Summarize what was done
- Use knowledge base information when relevant
- Ask follow-up questions if needed
- Follow the tone: {self.config.basic_settings.get('tone', 'professional')}
"""

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content="Generate response")
        ]

        response = await self.llm.ainvoke(messages=messages)
        return response.content or "I've processed your request."

    async def _dispatch(
        self,
        action_type: str,
        action_target: str,
        parameters: dict,
        user_message: str,
        session: Session,
    ) -> str:
        """Dispatch to specific executor."""

        if action_type == "skill":
            return await self.skill_executor.execute(
                skill_id=action_target,
                user_request=user_message,
                parameters=parameters,
            )

        elif action_type == "tool":
            return await self._execute_tool(action_target, parameters)

        elif action_type == "flow":
            # Execute flow directly (API call)
            return await self.flow_executor.execute(
                flow_id=action_target,
                user_message=user_message,
                parameters=parameters,
                session=session,
            )

        elif action_type == "system":
            return await self.system_executor.execute(
                action_id=action_target,
                parameters=parameters,
            )

        else:
            return self._generate_fallback_response(user_message)

    async def _execute_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute tool."""
        # Find tool
        tool = next((t for t in self._base_tools if t.name == tool_name), None)
        if not tool:
            return f"❌ Tool not found: {tool_name}"

        try:
            result = await tool.execute(**parameters)
            return f"✅ {tool_name} executed successfully\n\n{result}"
        except Exception as e:
            return f"❌ Tool execution failed: {e}"

    def _generate_fallback_response(self, message: str) -> str:
        """Generate fallback response (when no match)."""
        return f"Received your message: {message}\n\nSorry, I can't understand your request at the moment. You can:\n- Query weather\n- Write blog\n- Apply for leave"
