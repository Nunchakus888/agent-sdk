"""
WorkflowAgent V2 - based on bu_agent_sdk.Agent's unified LLM interaction architecture.

Core design:
1. Use Agent as internal engine, unified LLM context management
2. Convert all actions (skills, flows, tools, system) to Tool instances
3. Reuse Agent's compaction, token tracking, retry capabilities
4. SOP as system prompt injected

Advantages:
- Unified context: LLM always sees the complete execution history
- Efficient reuse: compaction, token tracking, retry capabilities ready-to-use
- Simple architecture: single Agent loop, cleaner code
- Best practices: follow bu_agent_sdk's design patterns
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from bu_agent_sdk.agent.compaction import CompactionConfig
from bu_agent_sdk.agent.service import Agent, TaskComplete
from bu_agent_sdk.llm.base import BaseChatModel, LLMProvider
from bu_agent_sdk.llm.messages import AssistantMessage, UserMessage
from bu_agent_sdk.prompts.builder import SystemPromptBuilder
from bu_agent_sdk.schemas import WorkflowConfigSchema
from bu_agent_sdk.tools.config_loader import HttpTool, ToolConfig
from bu_agent_sdk.tools.decorator import Tool
from bu_agent_sdk.tokens import UsageSummary

logger = logging.getLogger("agent_sdk.workflow_agent")


def _create_http_tool_wrapper(http_tool: HttpTool) -> Tool:
    """Wrap HttpTool as Tool instance."""
    defn = http_tool.definition

    async def execute_http(**kwargs: Any) -> str:
        result = await http_tool.execute(**kwargs)
        return str(result) if result else "Success"

    return Tool(
        func=execute_http,
        description=defn.description,
        name=defn.name,
        _definition=defn,  # Use HttpTool's definition directly
    )


def _create_skill_tool(skill_id: str, description: str, executor_fn) -> Tool:
    """Create Skill Tool."""
    async def execute_skill(request: str = "") -> str:
        if executor_fn:
            result = await executor_fn(skill_id=skill_id, user_request=request, parameters={})
            return str(result)
        return f"Skill {skill_id} executed with request: {request}"

    return Tool(
        func=execute_skill,
        description=description or f"Execute skill: {skill_id}",
        name=f"skill__{skill_id}",
    )


def _create_flow_tool(flow_id: str, description: str, executor_fn) -> Tool:
    """Create Flow Tool."""
    async def execute_flow() -> str:
        if executor_fn:
            result = await executor_fn(flow_id=flow_id, user_message="", parameters={}, session=None)
            return str(result)
        return f"Flow {flow_id} executed"

    return Tool(
        func=execute_flow,
        description=description or f"Execute flow: {flow_id}",
        name=f"flow__{flow_id}",
    )


def _create_system_tool(action_id: str, action_name: str, silent: bool = False) -> Tool:
    """Create System Action Tool."""
    async def execute_system(reason: str = "") -> str:
        # Special handling: handoff, transfer, etc. to terminate dialog
        if action_id in ("handoff", "transfer"):
            raise TaskComplete(f"Transferred to human agent. Reason: {reason}")
        if silent:
            raise TaskComplete(f"System action {action_id} completed silently.")
        return f"System action {action_id} executed. Reason: {reason}"

    return Tool(
        func=execute_system,
        description=f"System action: {action_name}",
        name=f"system__{action_id}",
    )


@dataclass
class WorkflowAgentV2:
    """
    Workflow Agent based on bu_agent_sdk.Agent.

    Use unified LLM context management, all actions as tools injected into Agent.

    Usage:
        ```python
        agent = WorkflowAgentV2(config=config, llm=llm)

        # Single query
        response = await agent.query(
            message="Help me",
            session_id="sess_123",
        )

        # Multi-turn conversation (context automatically maintained)
        response2 = await agent.query(
            message="Tell me more",
            session_id="sess_123",
        )

        # Get token usage statistics
        usage = await agent.get_usage()
        print(f"Total tokens: {usage.total_tokens}")
        ```
    """

    config: WorkflowConfigSchema
    llm: BaseChatModel | LLMProvider
    compaction: CompactionConfig | None = None
    include_cost: bool = False

    # Internal state
    _llm: BaseChatModel = field(default=None, repr=False)
    _agent: Agent = field(default=None, repr=False)
    _workflow_tools: list[Tool] = field(default_factory=list, repr=False)
    _system_prompt: str = field(default="", repr=False)

    def __post_init__(self):
        # LLM parsing
        if isinstance(self.llm, LLMProvider):
            self._llm = self.llm.get_response_llm()
        else:
            self._llm = self.llm

        # Build tools
        self._workflow_tools = self._build_workflow_tools()

        # Build system prompt
        self._system_prompt = self._build_system_prompt()

        # Create internal Agent
        self._agent = Agent(
            llm=self._llm,
            tools=self._workflow_tools,
            system_prompt=self._system_prompt,
            max_iterations=self.config.max_iterations * 2,  # Leave some room
            tool_choice="auto",
            compaction=self.compaction or CompactionConfig(
                threshold_ratio=0.75,  # 75% of context window
            ),
            include_cost=self.include_cost,
            require_done_tool=False,  # LLM can directly respond
        )

    def _build_workflow_tools(self) -> list[Tool]:
        """Convert all workflow actions to Tool instances."""
        tools: list[Tool] = []

        # 1. HTTP Tools - Use HttpTool directly
        if self.config.tools:
            for tool_config in self.config.tools:
                http_tool = HttpTool(config=ToolConfig(**tool_config))
                tools.append(_create_http_tool_wrapper(http_tool))

        # 2. Skills - Convert to Tool
        if self.config.skills:
            for skill in self.config.skills:
                skill_id = skill.get("skill_id", "unknown") if isinstance(skill, dict) else skill.skill_id
                description = skill.get("description", "") if isinstance(skill, dict) else skill.description
                tools.append(_create_skill_tool(skill_id, description, None))

        # 3. Flows - Convert to Tool
        if self.config.flows:
            for flow in self.config.flows:
                flow_id = flow.flow_id or flow.name or "unknown"
                description = flow.description or f"Execute flow {flow_id}"
                tools.append(_create_flow_tool(flow_id, description, None))

        # 4. System Actions - Convert to Tool
        # if self.config.system_actions:
        #     for action in self.config.system_actions:
        #         if isinstance(action, str):
        #             action_id = action
        #             action_name = action
        #             silent = False
        #         else:
        #             action_id = action.action_id
        #             action_name = action.name
        #             silent = action.silent
        #         tools.append(_create_system_tool(action_id, action_name, silent))

        return tools

    def _build_system_prompt(self) -> str:
        """Build System Prompt using unified builder."""
        builder = SystemPromptBuilder(config=self.config)
        return builder.build()

    async def query(
        self,
        message: str,
        context: list[dict] | None = None,
    ) -> str:
        """
        Process user message.

        Args:
            message: User message
            context: Injected history messages (optional)

        Returns:
            Agent response
        """
        # If there is injected context, load it into Agent
        if context:
            messages = []
            for msg in context:
                if msg["role"] == "user":
                    messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AssistantMessage(content=msg["content"]))
            self._agent.load_history(messages)

        # Log prompt context for debugging
        self._log_prompt_context(message)

        # Call Agent
        try:
            response = await self._agent.query(message)
            return response
        except TaskComplete as e:
            return e.message

    def _log_prompt_context(self, user_message: str):
        """Log prompt context for debugging."""
        history_count = len(self._agent.messages)
        system_prompt_len = len(self._system_prompt)
        tools_count = len(self._workflow_tools)

        logger.debug(
            f"[Prompt Context] "
            f"history_msgs={history_count}, "
            f"system_prompt_len={system_prompt_len}, "
            f"tools={tools_count}, "
            f"user_msg_len={len(user_message)}"
        )

        # Log detailed context at trace level (if enabled)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[System Prompt]\n{self._system_prompt[:500]}...")
            logger.debug(f"[User Message] {user_message[:200]}...")

            # Log history summary
            if self._agent.messages:
                history_summary = []
                for i, msg in enumerate(self._agent.messages[-5:]):  # Last 5 messages
                    role = type(msg).__name__
                    content_preview = str(msg.content)[:50] if msg.content else ""
                    history_summary.append(f"  [{i}] {role}: {content_preview}...")
                logger.debug(f"[History (last 5)]\n" + "\n".join(history_summary))

    async def query_stream(self, message: str):
        """
        Stream process user message.

        Yields:
            AgentEvent instance
        """
        async for event in self._agent.query_stream(message):
            yield event

    async def get_usage(self) -> UsageSummary:
        """Get token usage statistics."""
        return await self._agent.get_usage()

    def clear_history(self):
        """Clear conversation history."""
        self._agent.clear_history()

    @property
    def messages(self):
        """Get current message history."""
        return self._agent.messages

    @property
    def tool_definitions(self):
        """Get all tool definitions."""
        return self._agent.tool_definitions
