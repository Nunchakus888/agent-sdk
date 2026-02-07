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

ActionBook execution:
- actionbook_executor is a built-in tool, auto-registered when action_books configured
- Handles response extraction (data.message) and token accumulation (data.total_tokens)
- Implementation in actionbook_executor.py module
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from bu_agent_sdk.agent.actionbook_executor import (
    ACTIONBOOK_EXECUTOR_TOOL_NAME,
    create_actionbook_executor_tool,
)
from bu_agent_sdk.agent.compaction import CompactionConfig
from bu_agent_sdk.agent.flow_matcher import FlowMatcher, FlowMatchResult
from bu_agent_sdk.agent.knowledge_retriever import KnowledgeRetriever
from bu_agent_sdk.agent.service import Agent, TaskComplete
from bu_agent_sdk.agent.system_actions import SystemActionExecutor
from bu_agent_sdk.llm.base import BaseChatModel, LLMProvider
from bu_agent_sdk.llm.messages import AssistantMessage, UserMessage
from bu_agent_sdk.prompts.builder import SystemPromptBuilder
from bu_agent_sdk.schemas import WorkflowConfigSchema
from bu_agent_sdk.tools.config_loader import HttpTool, ToolConfig
from bu_agent_sdk.tools.decorator import Tool
from bu_agent_sdk.tokens import UsageSummary

logger = logging.getLogger("agent_sdk.workflow_agent")

TRIGGER_FLOW_TOOL_NAME = "flow_executor"
ACTIONBOOK_EXECUTOR_TOOL_NAME = "actionbook_executor"


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

    Context variables (context_vars) are used to inject system parameters
    into HTTP tools at runtime. These are NOT exposed to LLM, but are used
    to fill placeholders like {chatbotId}, {tenantId}, {sessionId} in
    tool endpoint configurations.
    """

    config: WorkflowConfigSchema
    llm: BaseChatModel | LLMProvider
    context_vars: dict[str, Any] = field(default_factory=dict)
    compaction: CompactionConfig | None = None
    include_cost: bool = False

    # Internal state
    _llm: BaseChatModel = field(default=None, repr=False)
    _agent: Agent = field(default=None, repr=False)
    _workflow_tools: list[Tool] = field(default_factory=list, repr=False)
    _system_action_tools: dict[str, HttpTool] = field(default_factory=dict, repr=False)
    _system_prompt: str = field(default="", repr=False)
    _flow_matcher: FlowMatcher = field(default=None, repr=False)
    _knowledge_retriever: KnowledgeRetriever | None = field(default=None, repr=False)

    def __post_init__(self):
        if isinstance(self.llm, LLMProvider):
            self._llm = self.llm.get_response_llm()
        else:
            self._llm = self.llm

        self._flow_matcher = FlowMatcher(flows=self.config.flows)
        self._system_action_tools = self._build_system_action_tools()
        self._workflow_tools = self._build_workflow_tools()
        self._system_prompt = self._build_system_prompt()

        # Initialize knowledge retriever if configured
        if self.config.retrieve_knowledge_url:
            self._knowledge_retriever = KnowledgeRetriever(
                url=self.config.retrieve_knowledge_url,
                context_vars=self.context_vars,
                timeout=10.0,
            )

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

    def _build_system_action_tools(self) -> dict[str, HttpTool]:
        """Build system action tools (NOT registered with LLM).

        These tools are executed in parallel with the main flow,
        and their results are used for variable substitution.
        """
        if not self.config.system_actions or not self.config.tools:
            return {}

        system_action_names = set(self.config.system_actions)
        tools: dict[str, HttpTool] = {}

        for tool_config in self.config.tools:
            tool_name = tool_config.get("name", "")
            if tool_name in system_action_names:
                tools[tool_name] = HttpTool(
                    config=ToolConfig(**tool_config),
                    context_vars=self.context_vars,
                )

        if tools:
            logger.info(
                f"System action tools built: {list(tools.keys())}"
            )

        return tools

    def _build_workflow_tools(self) -> list[Tool]:
        """Build workflow tools from config.

        HTTP tools receive context_vars for system parameter injection.
        Placeholders like {chatbotId}, {tenantId} in endpoint config
        will be filled from context_vars at execution time.

        ActionBook executor is automatically added when action_books are configured.

        Excludes:
        - system_actions (executed separately, not for LLM)
        - actionbook_executor from config (uses built-in instead)
        """
        tools: list[Tool] = []
        system_action_names = set(self.config.system_actions or [])

        # ActionBook Executor - built-in tool, auto-registered when action_books configured
        if self.config.action_books:
            tools.append(
                create_actionbook_executor_tool(
                    context_vars=self.context_vars,
                    token_accumulator=self._accumulate_external_tokens,
                )
            )

        # HTTP Tools (skip actionbook_executor if in config, use built-in instead)
        # Also skip system_actions (executed separately, not for LLM)
        if self.config.tools:
            for tool_config in self.config.tools:
                tool_name = tool_config.get("name", "")
                if tool_name == ACTIONBOOK_EXECUTOR_TOOL_NAME:
                    logger.debug("Skipping actionbook_executor from config, using built-in")
                    continue
                if tool_name in system_action_names:
                    logger.debug(f"Skipping system action from LLM tools: {tool_name}")
                    continue

                http_tool = HttpTool(
                    config=ToolConfig(**tool_config),
                    context_vars=self.context_vars,
                )
                tools.append(_create_http_tool_wrapper(http_tool))

        # Skills
        if self.config.skills:
            for skill in self.config.skills:
                skill_id = skill.get("skill_id", "unknown") if isinstance(skill, dict) else skill.skill_id
                description = skill.get("description", "") if isinstance(skill, dict) else skill.description
                tools.append(_create_skill_tool(skill_id, description, None))

        return tools

    def _accumulate_external_tokens(self, tokens: int) -> None:
        """Accumulate external token usage to agent's token cost."""
        if hasattr(self, '_agent') and self._agent:
            self._agent._token_cost.add_external_tokens(tokens)

    def _build_system_prompt(self, kb_content: str | None = None) -> str:
        """Build System Prompt using unified builder.

        Args:
            kb_content: Optional knowledge base content to inject
        """
        builder = SystemPromptBuilder(config=self.config)
        return builder.build(kb_content=kb_content)

    async def _retrieve_knowledge(
        self,
        message: str,
        session_id: str | None = None,
    ) -> str | None:
        """
        Retrieve knowledge from knowledge base.

        Returns formatted knowledge content or None if:
        - No retriever configured
        - Retrieval fails
        - No relevant results
        """
        if not self._knowledge_retriever:
            return None

        return await self._knowledge_retriever.retrieve(
            query=message,
            session_id=session_id,
        )

    async def query(
        self,
        message: str,
        context: list[dict] | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Process user message.

        Flow:
        1. Start system actions in parallel (non-blocking)
        2. Retrieve knowledge (if configured)
        3. Rebuild system prompt with KB content
        4. Try keyword matching (fast, deterministic)
        5. If matched, call flow_executor tool directly
        6. If not matched, enter Agent loop (LLM may call flow_executor)
        7. Apply system action variable substitution to response
        """
        # Step 0: Start system actions in parallel
        system_executor = self._create_system_executor()
        system_task = system_executor.start()

        # Step 1: Knowledge retrieval (non-blocking on failure)
        kb_content = await self._retrieve_knowledge(message, session_id)

        # Step 2: Rebuild system prompt if KB content retrieved
        if kb_content:
            new_system_prompt = self._build_system_prompt(kb_content=kb_content)
            self._agent.system_prompt = new_system_prompt
            logger.debug(f"System prompt updated with KB content ({len(kb_content)} chars)")

        # Step 3: Keyword matching
        if self._flow_matcher.has_keyword_flows:
            match_result = self._flow_matcher.match_keyword(message)
            if match_result.matched:
                result = await self._execute_keyword_flow(match_result)
                return await system_executor.apply(result, system_task)

        # Step 4: Load context if provided
        if context:
            messages = []
            for msg in context:
                if msg["role"] == "user":
                    messages.append(UserMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AssistantMessage(content=msg["content"]))
            self._agent.load_history(messages)

        # Step 5: Agent loop (LLM may call flow_executor for intent flows)
        self._log_prompt_context(message)

        try:
            result = await self._agent.query(message)
        except TaskComplete as e:
            result = e.message

        # Step 6: Apply system action variable substitution
        return await system_executor.apply(result, system_task)

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
        """Stream process user message with system action support.

        System actions run in parallel. Variable substitution is applied
        to the FinalResponseEvent content before yielding.
        """
        from bu_agent_sdk.agent.events import FinalResponseEvent

        system_executor = self._create_system_executor()
        system_task = system_executor.start()

        async for event in self._agent.query_stream(message):
            if isinstance(event, FinalResponseEvent) and system_task:
                substituted = await system_executor.apply(
                    event.content, system_task
                )
                yield FinalResponseEvent(content=substituted)
            else:
                yield event

    def _create_system_executor(self) -> SystemActionExecutor:
        """Create a SystemActionExecutor for the current request."""
        return SystemActionExecutor(action_tools=self._system_action_tools)

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
