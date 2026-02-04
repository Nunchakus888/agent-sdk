"""
System Prompt Builder - 统一的 prompt 构建框架

设计原则：
1. 模板方法模式 - build() 定义骨架，子类可覆盖细节
2. 条件性包含 - 只包含有内容的部分
3. 7 层结构 - Identity, Context, Instructions, Capabilities, Action Rules, Constraints, Language & Response Guidelines
"""

from dataclasses import dataclass
from typing import Any

from bu_agent_sdk.prompts.templates import KNOWLEDGE_BASE_SECTION


# 语言要求模板 - 最高优先级
LANGUAGE_REQUIREMENTS_TEMPLATE = """## Language & Response Guidelines (HIGHEST PRIORITY)

### Language Detection
1. Primary: Match user's MOST RECENT message language
2. Fallback priority (when unclear - mixed languages, emojis only, ambiguous):
   - First: Extract from conversation history
   - Last: Use {fallback_language}

### Language Switch
If user explicitly requests (e.g., "use Spanish", "用中文"), switch immediately.

### Tone
Maintain a {tone} tone.

### Response Style
- Follow SOP step by step
- Use tools when needed
- Be concise and professional
- Ask for clarification if uncertain
- Explain tool failures and suggest alternatives"""


@dataclass
class SystemPromptBuilder:
    """
    System Prompt 构建器

    统一的 prompt 构建框架，支持 WorkflowConfigSchema 和 AgentConfigSchema。

    结构（7 层）：
    1. Identity - 身份定义（名称、角色、语言、语气）
    2. Context - 上下文背景（业务背景、知识库）
    3. Instructions - 工作指令（SOP、问候语）
    4. Capabilities - 能力定义（工具、技能、流程、系统动作）
    5. Action Rules - 条件-动作规则
    6. Constraints - 约束边界 + 错误处理
    7. Response Guidelines - 响应指南（多语言）

    Usage:
        ```python
        from bu_agent_sdk.prompts.builder import SystemPromptBuilder
        from bu_agent_sdk.schemas import WorkflowConfigSchema

        config = WorkflowConfigSchema(**config_dict)
        builder = SystemPromptBuilder(config=config)
        prompt = builder.build()

        # With KB content injection
        prompt = builder.build(kb_content="Retrieved knowledge here...")
        ```
    """

    config: Any  # WorkflowConfigSchema 或 AgentConfigSchema

    def build(self, kb_content: str | None = None) -> str:
        """构建完整的 system prompt

        Args:
            kb_content: 可选的知识库检索结果，运行时动态注入
        """
        sections = []

        # 1. Identity
        if identity := self._build_identity():
            sections.append(identity)

        # 2. Context (including KB content if provided)
        if context := self._build_context(kb_content=kb_content):
            sections.append(context)

        # 3. Instructions
        if instructions := self._build_instructions():
            sections.append(instructions)

        # 4. Capabilities
        if capabilities := self._build_capabilities():
            sections.append(capabilities)

        # 5. Action Rules
        if action_rules := self._build_action_rules():
            sections.append(action_rules)

        # 6. Constraints
        if constraints := self._build_constraints():
            sections.append(constraints)

        # 7. Response Guidelines
        sections.append(self._build_response_guidelines())

        return "\n\n".join(sections)

    def _build_identity(self) -> str:
        """构建身份定义部分

        注意：Language 和 Tone 已移至 Language & Response Guidelines 部分，
        避免重复定义。
        """
        basic = self._get_basic_settings()
        if not basic:
            return ""

        name = basic.get('name', 'Assistant')
        description = basic.get('description', '')

        return f"""## Agent Profile
You are {name}.

- **Role**: {description}"""

    def _build_context(self, kb_content: str | None = None) -> str:
        """构建上下文背景部分

        Args:
            kb_content: 可选的知识库检索结果
        """
        parts = []
        basic = self._get_basic_settings()

        # Background
        background = basic.get('background', '') if basic else ''
        if background:
            parts.append(f"""## Background
{background}""")

        # Knowledge Base - 动态注入 retrieval 结果
        if kb_content:
            parts.append(KNOWLEDGE_BASE_SECTION.format(kb_content=kb_content))

        return "\n\n".join(parts)

    def _build_instructions(self) -> str:
        """构建工作指令部分"""
        parts = []

        # SOP Instructions - 支持两种配置方式
        # 1. 顶层 instructions 字段
        # 2. basic_settings.instruction 字段
        instructions = getattr(self.config, 'instructions', None)
        if not instructions:
            basic = self._get_basic_settings()
            instructions = basic.get('instruction', '') or basic.get('instructions', '')

        if instructions:
            parts.append(f"""## SOP Instructions
{instructions}""")

        # Greeting
        need_greeting = getattr(self.config, 'need_greeting', '')
        if need_greeting:
            parts.append(f"""## Greeting
When starting a new conversation, greet the user with:
"{need_greeting}" """)

        return "\n\n".join(parts)

    def _build_capabilities(self) -> str:
        """构建能力定义部分

        注意：Tools 不在此处列出，因为已通过原生 tools API 传递给 LLM。
        这里只列出高级抽象概念（Skills、Flows、action_books）。
        """
        sections = []

        # Skills - 高级抽象，需要在 prompt 中说明
        skills = getattr(self.config, 'skills', [])
        if skills:
            skill_lines = []
            for s in skills:
                if isinstance(s, dict):
                    sid = s.get('skill_id', 'unknown')
                    desc = s.get('description', '')
                else:
                    sid = getattr(s, 'skill_id', 'unknown')
                    desc = getattr(s, 'description', '')
                if sid and desc:
                    skill_lines.append(f"- **{sid}**: {desc}")
            if skill_lines:
                sections.append("### Skills\n" + "\n".join(skill_lines))

        # Flows - 只描述 intent 类型的 flows（keyword 类型由代码匹配，无需 LLM 参与）
        flows = getattr(self.config, 'flows', [])
        if flows:
            intent_flows = []
            for f in flows:
                # 只包含 intent 类型的 flows
                flow_type = getattr(f, 'type', None)
                if flow_type and str(flow_type).lower() == 'keyword':
                    continue  # 跳过 keyword 类型

                fid = getattr(f, 'flow_id', None) or getattr(f, 'name', None) or 'unknown'
                desc = getattr(f, 'description', None) or f'Execute flow {fid}'
                intent_flows.append(f"- `{fid}`: {desc}")

            if intent_flows:
                sections.append(
                    "### Intent Flows\n"
                    "When user intent matches one of the following, call `trigger_flow` tool with the corresponding flow_id:\n"
                    + "\n".join(intent_flows)
                )

        # System Actions - 特殊系统行为
        system_actions = getattr(self.config, 'system_actions', [])
        if system_actions:
            action_lines = [f"- {a}" for a in system_actions]
            sections.append("### System Actions\n" + "\n".join(action_lines))

        if sections:
            return "## Available Capabilities\n\n" + "\n\n".join(sections)
        return ""

    def _build_action_rules(self) -> str:
        """构建条件-动作规则部分"""
        # 从 skills 中提取 action_books 风格的规则
        skills = getattr(self.config, 'skills', [])
        if not skills:
            return ""

        rules = []
        for i, skill in enumerate(skills, 1):
            if isinstance(skill, dict):
                condition = skill.get('condition', '')
                action = skill.get('action', '')
                skill_tools = skill.get('tools', [])

                if condition and action:
                    rule = f"""### Rule {i}
**When**: {condition}
**Action**: {action}"""
                    if skill_tools:
                        rule += f"\n**Tools**: {', '.join(skill_tools)}"
                    rules.append(rule)

        if rules:
            return "## Action Rules\n\n" + "\n\n".join(rules)
        return ""

    def _build_constraints(self) -> str:
        """构建约束边界部分"""
        # 支持两种配置方式
        # 1. 顶层 constraints 字段
        # 2. basic_settings.constraints 字段
        constraints = getattr(self.config, 'constraints', None)
        if not constraints:
            basic = self._get_basic_settings()
            constraints = basic.get('constraints', '')

        if not constraints:
            return ""

        return f"""## Constraints
{constraints}

### Error Handling
- If a tool call fails, explain the issue to the user
- Suggest alternative approaches when possible
- Escalate to human support when unable to resolve"""

    def _build_response_guidelines(self) -> str:
        """构建语言和响应指南部分

        整合语言检测、语气和响应风格指南。
        fallback_language 和 tone 从配置中获取。
        """
        basic = self._get_basic_settings()
        fallback_language = basic.get('language', 'English') if basic else 'English'
        tone = basic.get('tone', 'professional') if basic else 'professional'

        return LANGUAGE_REQUIREMENTS_TEMPLATE.format(
            fallback_language=fallback_language,
            tone=tone
        )

    def _get_basic_settings(self) -> dict:
        """获取基础设置"""
        basic = getattr(self.config, 'basic_settings', None)
        if isinstance(basic, dict):
            return basic
        return {}
