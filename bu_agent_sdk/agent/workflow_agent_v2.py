"""
WorkflowAgent V2 - based on bu_agent_sdk.Agent's unified LLM interaction architecture.

Core design:
1. Use Agent as internal engine, unified LLM context management
2. Convert all actions (skills, tools, system) to Tool instances
3. Reuse Agent's compaction, token tracking, retry capabilities
4. SOP as system prompt injected
5. Flow matching: keyword (code-based) + intent (LLM via flow_executor tool)

Flow execution:
- Keyword flows: matched by code, executed via flow_executor tool directly
- Intent flows: LLM calls flow_executor tool with flow_id
- Unified execution through flow_executor tool defined in config
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from bu_agent_sdk.agent.compaction import CompactionConfig
from bu_agent_sdk.agent.flow_matcher import FlowMatcher, FlowMatchResult
from bu_agent_sdk.agent.service import Agent, TaskComplete
from bu_agent_sdk.llm.base import BaseChatModel, LLMProvider
from bu_agent_sdk.llm.messages import AssistantMessage, UserMessage
from bu_agent_sdk.prompts.builder import SystemPromptBuilder
from bu_agent_sdk.schemas import WorkflowConfigSchema
from bu_agent_sdk.tools.config_loader import HttpTool, ToolConfig
from bu_agent_sdk.tools.decorator import Tool
from bu_agent_sdk.tokens import UsageSummary

logger = logging.getLogger("agent_sdk.workflow_agent")

TRIGGER_FLOW_TOOL_NAME = "flow_executor"


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
        _definition=defn,
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


@dataclass
class WorkflowAgentV2:
    """
    Workflow Agent based on bu_agent_sdk.Agent.

    Flow execution is unified through flow_executor tool:
    - Keyword match: code matches, then calls flow_executor directly
    - Intent match: LLM calls flow_executor with flow_id
    - Both use the same flow_executor tool defined in config
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
    _flow_matcher: FlowMatcher = field(default=None, repr=False)

    def __post_init__(self):
        if isinstance(self.llm, LLMProvider):
            self._llm = self.llm.get_response_llm()
        else:
            self._llm = self.llm

        self._flow_matcher = FlowMatcher(flows=self.config.flows)
        self._workflow_tools = self._build_workflow_tools()
        self._system_prompt = self._build_system_prompt()

        self._agent = Agent(
            llm=self._llm,
            tools=self._workflow_tools,
            system_prompt=self._system_prompt,
            max_iterations=self.config.max_iterations * 2,
            tool_choice="auto",
            compaction=self.compaction or CompactionConfig(threshold_ratio=0.75),
            include_cost=self.include_cost,
            require_done_tool=False,
        )

    def _build_workflow_tools(self) -> list[Tool]:
        """Build workflow tools from config."""
        tools: list[Tool] = []

        # HTTP Tools (includes flow_executor)
        if self.config.tools:
            for tool_config in self.config.tools:
                http_tool = HttpTool(config=ToolConfig(**tool_config))
                tools.append(_create_http_tool_wrapper(http_tool))

        # Skills
        if self.config.skills:
            for skill in self.config.skills:
                skill_id = skill.get("skill_id", "unknown") if isinstance(skill, dict) else skill.skill_id
                description = skill.get("description", "") if isinstance(skill, dict) else skill.description
                tools.append(_create_skill_tool(skill_id, description, None))

        # action_books
        if self.config.action_books:
            # action_books are converted to tools, just pick the condition to match the intent
            pass

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

        Flow:
        1. Try keyword matching (fast, deterministic)
        2. If matched, call flow_executor tool directly
        3. If not matched, enter Agent loop (LLM may call flow_executor)
        """
        # Step 1: Keyword matching
        if self._flow_matcher.has_keyword_flows:
            match_result = self._flow_matcher.match_keyword(message)
            if match_result.matched:
                return await self._execute_keyword_flow(match_result)

        # Step 2: Load context if provided
        if context:
            messages = []
            for msg in context:
                if msg["role"] == "user":
                    messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AssistantMessage(content=msg["content"]))
            self._agent.load_history(messages)

        # Step 3: Agent loop (LLM may call flow_executor for intent flows)
        self._log_prompt_context(message)

        try:
            return await self._agent.query(message)
        except TaskComplete as e:
            return e.message

    async def _execute_keyword_flow(self, match_result: FlowMatchResult) -> str:
        """
        Execute keyword-matched flow via flow_executor tool.

        Skip LLM, call flow_executor directly through Agent's tool_map.
        """
        flow = match_result.flow
        flow_id = flow.flow_id or flow.name

        logger.info(f"Keyword matched, executing flow: {flow_id}")

        # Get flow_executor tool from Agent's tool_map
        trigger_tool = self._agent._tool_map.get(TRIGGER_FLOW_TOOL_NAME)
        if not trigger_tool:
            logger.warning("flow_executor tool not configured")
            return f"Flow {flow_id} matched but flow_executor tool not configured"

        # Execute directly (same tool that LLM would call)
        result = await trigger_tool.execute(flow_id=flow_id)
        return str(result) if result else f"Flow {flow_id} executed"

    def _log_prompt_context(self, user_message: str):
        """Log prompt context for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        logger.debug(
            f"[Prompt Context] "
            f"history={len(self._agent.messages)}, "
            f"tools={len(self._workflow_tools)}, "
            f"msg_len={len(user_message)}"
        )

    async def query_stream(self, message: str):
        """Stream process user message."""
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
