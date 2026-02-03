"""Debug utilities for LLM request/response logging."""

import json
import logging
import os
from typing import Any

from bu_agent_sdk.llm.base import ToolDefinition

# Environment variable to enable debug logging
LLM_DEBUG_ENV = "DEBUG"


def is_debug_enabled() -> bool:
    """Check if LLM debug logging is enabled."""
    return bool(os.getenv(LLM_DEBUG_ENV))


def log_llm_request(
    logger: logging.Logger,
    model: str,
    messages: list[Any],
    tools: list[ToolDefinition] | None = None,
    tool_choice: Any = None,
    system_prompt: str | None = None,
    params: dict[str, Any] | None = None,
    max_content_length: int = 2000,
) -> None:
    """
    Log LLM request details for debugging.
    
    Args:
        logger: Logger instance to use
        model: Model name/identifier
        messages: Serialized messages (provider-specific format)
        tools: Optional tool definitions
        tool_choice: Optional tool choice setting
        system_prompt: Optional system prompt (for providers that separate it)
        params: Optional model parameters
        max_content_length: Max chars before truncating content (default 2000)
    """
    if not is_debug_enabled():
        return

    def truncate(text: str) -> str:
        if len(text) > max_content_length:
            return f"{text[:max_content_length]}... [truncated, total {len(text)} chars]"
        return text

    def to_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)

    logger.debug("=" * 80)
    logger.debug(f"\n🔍 LLM Request to {model}")
    logger.debug("=" * 80)

    # Log system prompt (explicit or extracted from first message)
    if system_prompt:
        logger.debug(f"\n📋 System: {truncate(system_prompt)}")
    elif messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        logger.debug(f"\n📋 System: {truncate(str(messages[0].get('content', '')))}")

    # Log messages
    logger.debug("\n📝 Messages:")
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_str = to_json(content) if isinstance(content, (list, dict)) else str(content)
            logger.debug(f"  [{i}] {role}: {truncate(content_str)}")
            if "tool_calls" in msg:
                logger.debug(f"\n      tool_calls: {to_json(msg['tool_calls'])}")
        else:
            logger.debug(f"  [{i}]: {truncate(to_json(msg))}")

    # Log tools
    if tools:
        logger.debug(f"\n🔧 Tools ({len(tools)}):")
        for tool in tools:
            desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
            logger.debug(f"\n  - {tool.name}: {desc}")
            logger.debug(f"\n    params: {to_json(tool.parameters)}")

    # Log tool_choice
    if tool_choice:
        logger.debug(f"\n🎯 Tool Choice: {tool_choice}")

    # Log params (excluding tools)
    if params:
        params_to_log = {k: v for k, v in params.items() if k not in ["tools"]}
        if params_to_log:
            logger.debug(f"\n⚙️ Params: {to_json(params_to_log)}")

    logger.debug("=" * 80)
