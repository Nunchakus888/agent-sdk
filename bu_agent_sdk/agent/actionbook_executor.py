"""
ActionBook Executor - Built-in tool for executing action books.

Design:
- Endpoint URL: ACTIONBOOK_ENDPOINT_URL env var
- Timeout: ACTIONBOOK_TIMEOUT env var (default: 60)
- Request format: reuses QueryRequest from api.models.schemas
- Response: extracts data.message, accumulates data.total_tokens
"""

import json
import logging
import os
from typing import Any, Callable

import httpx

from api.models.schemas import QueryRequest
from bu_agent_sdk.tools.decorator import Tool

logger = logging.getLogger("agent_sdk.actionbook_executor")

ACTIONBOOK_EXECUTOR_TOOL_NAME = "actionbook_executor"

# camelCase -> snake_case mapping (only for keys that differ)
_CAMEL_TO_SNAKE = {
    "chatbotId": "chatbot_id",
    "tenantId": "tenant_id",
    "sessionId": "session_id",
    "customerId": "customer_id",
}


def get_actionbook_endpoint_url() -> str:
    """Get actionbook endpoint URL from environment variable."""
    if url := os.getenv("ACTIONBOOK_ENDPOINT_URL"):
        return url
    raise ValueError("ACTIONBOOK_ENDPOINT_URL environment variable not set")


def get_actionbook_timeout() -> int:
    """Get actionbook timeout from environment variable (default: 60)."""
    return int(os.getenv("ACTIONBOOK_TIMEOUT", "60"))


def _normalize_context_vars(context_vars: dict[str, Any]) -> dict[str, Any]:
    """Normalize context_vars: camelCase -> snake_case, filter to QueryRequest fields."""
    # Convert camelCase to snake_case
    normalized = {_CAMEL_TO_SNAKE.get(k, k): v for k, v in context_vars.items()}
    # Filter to only QueryRequest fields (DRY: use model_fields)
    return {k: v for k, v in normalized.items() if k in QueryRequest.model_fields}


def _build_request_body(message: str, context_vars: dict[str, Any]) -> dict[str, Any]:
    """Build request body using QueryRequest model (DRY: reuse model definition)."""
    normalized = _normalize_context_vars(context_vars)

    # Set defaults for required fields and source
    defaults = {
        "session_id": "",
        "chatbot_id": "",
        "tenant_id": "",
        "source": "agent",
    }

    # Build request: message + defaults + context_vars (context_vars override defaults)
    request = QueryRequest(message=message, **{**defaults, **normalized})

    body = request.model_dump()
    body["timeout"] = get_actionbook_timeout()
    return body


def create_actionbook_executor_tool(
    context_vars: dict[str, Any],
    token_accumulator: Callable[[int], None] | None = None,
) -> Tool:
    """Create actionbook_executor tool."""

    async def execute_actionbook(message: str) -> str:
        """Execute action book based on user message."""
        body = _build_request_body(message, context_vars)
        timeout = get_actionbook_timeout()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    get_actionbook_endpoint_url(), json=body, timeout=timeout
                )
            except httpx.TimeoutException:
                return f"Error: Request timeout after {timeout}s"
            except httpx.RequestError as e:
                return f"Error: Request failed - {e}"

        if not response.is_success:
            return f"Error: HTTP {response.status_code}"

        try:
            data = response.json()
        except json.JSONDecodeError:
            return response.text

        result_data = data.get("data", {})
        if (tokens := result_data.get("total_tokens")) and token_accumulator:
            token_accumulator(int(tokens))

        return result_data.get("message") or json.dumps(data, ensure_ascii=False)

    return Tool(
        func=execute_actionbook,
        name=ACTIONBOOK_EXECUTOR_TOOL_NAME,
        description="Execute an action book when user intent matches configured conditions.",
    )
