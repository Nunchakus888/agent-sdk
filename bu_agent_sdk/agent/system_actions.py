"""
System Actions - Parallel execution of system-only tools with response variable substitution.

Design:
- System actions are HTTP tools executed in parallel with the main LLM flow
- Results are used for variable substitution in LLM responses (e.g., {{$contact.nickname}})
- System action tools are NOT registered with the LLM -- they are system-only

Usage:
    executor = SystemActionExecutor(
        action_tools={"get_contact_info": http_tool},
    )

    # Fire system actions (non-blocking)
    task = executor.start()

    # ... main LLM flow ...

    # After LLM response, substitute variables
    final_response = await executor.apply(llm_response, task)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bu_agent_sdk.tools.config_loader import HttpTool

logger = logging.getLogger("agent_sdk.system_actions")

# Pattern: {{$contact.fieldName}} or {{$contact.field_name}}
CONTACT_VAR_PATTERN = re.compile(r"\{\{\$contact\.(\w+)\}\}")


def _flatten_contact_response(data: Any) -> dict[str, str]:
    """Flatten a contact API response into a simple key-value dict.

    Handles nested structures like {"data": {"nickname": "John"}}
    and flat structures like {"nickname": "John"}.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {}

    if not isinstance(data, dict):
        return {}

    # Try common response wrappers
    for key in ("data", "result", "contact"):
        if key in data and isinstance(data[key], dict):
            data = data[key]
            break

    # Flatten to string values, skip None
    result: dict[str, str] = {}
    for k, v in data.items():
        if v is not None:
            result[k] = str(v) if not isinstance(v, str) else v
    return result


def substitute_contact_variables(text: str, contact_data: dict[str, str]) -> str:
    """Replace {{$contact.xxx}} variables in text with actual values.

    Parses field names from template, matches directly against contact_data keys.
    Unmatched variables are left as-is.
    """
    if not contact_data or not CONTACT_VAR_PATTERN.search(text):
        return text

    def replacer(match: re.Match) -> str:
        field_name = match.group(1)
        value = contact_data.get(field_name)
        return value if value is not None else match.group(0)

    return CONTACT_VAR_PATTERN.sub(replacer, text)


@dataclass
class SystemActionExecutor:
    """Execute system actions in parallel and apply variable substitution.

    Attributes:
        action_tools: Map of action_name -> HttpTool instance
    """

    action_tools: dict[str, HttpTool] = field(default_factory=dict)
    _results: dict[str, dict[str, str]] = field(
        default_factory=dict, repr=False
    )

    def has_actions(self) -> bool:
        """Check if there are any system actions to execute."""
        return bool(self.action_tools)

    def start(self) -> asyncio.Task | None:
        """Fire all system actions in parallel. Returns a Task to await later.

        Returns None if no system actions are configured.
        """
        if not self.action_tools:
            return None
        return asyncio.create_task(self._execute_all())

    async def _execute_all(self) -> None:
        """Execute all system action tools concurrently."""
        tasks = {
            name: asyncio.create_task(self._execute_one(name, tool))
            for name, tool in self.action_tools.items()
        }

        for name, task in tasks.items():
            try:
                result = await task
                self._results[name] = result
                logger.info(
                    f"System action completed: {name}, fields={len(result)}"
                )
            except Exception as e:
                logger.warning(f"System action failed: {name}, error={e}")
                self._results[name] = {}

    async def _execute_one(
        self, name: str, tool: HttpTool
    ) -> dict[str, str]:
        """Execute a single system action tool and parse the result."""
        try:
            raw_result = await tool.execute()
            return _flatten_contact_response(raw_result)
        except Exception as e:
            logger.warning(f"System action execution error: {name}, {e}")
            return {}

    async def apply(
        self,
        response: str,
        task: asyncio.Task | None,
    ) -> str:
        """Wait for system actions and substitute variables in response.

        Args:
            response: The LLM response text
            task: The task returned by start(), or None

        Returns:
            Response with contact variables substituted
        """
        if task is None or not response:
            return response

        # Wait for system actions if not yet complete
        if not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "System actions timed out, skipping substitution"
                )
                return response
            except Exception as e:
                logger.warning(f"System actions error: {e}")
                return response

        # Check if response contains any contact variables
        if not CONTACT_VAR_PATTERN.search(response):
            return response

        # Merge all system action results
        merged_data: dict[str, str] = {}
        for action_result in self._results.values():
            merged_data.update(action_result)

        if not merged_data:
            return response

        substituted = substitute_contact_variables(response, merged_data)
        if substituted != response:
            logger.info("Contact variables substituted in response")

        return substituted
