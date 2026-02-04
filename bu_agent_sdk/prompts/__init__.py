"""
Prompts 模块 - System Prompt 构建框架

提供统一的 prompt 构建器，支持 WorkflowAgentV2 和 ConfigToolLoader。
"""

from bu_agent_sdk.prompts.builder import SystemPromptBuilder
from bu_agent_sdk.prompts.templates import (
    INSTRUCTIONS_TEMPLATE,
    KNOWLEDGE_BASE_SECTION,
)

__all__ = [
    "SystemPromptBuilder",
    "INSTRUCTIONS_TEMPLATE",
    "KNOWLEDGE_BASE_SECTION",
]
