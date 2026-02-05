"""
Knowledge Base Retrieval Service

Provides async knowledge retrieval capability for WorkflowAgentV2.
Uses AsyncHttpClient from config/http_config.py for HTTP requests.

Design:
- Atomic: Standalone, reusable component
- Graceful degradation: Failures return None, never block main flow
- Minimal: Uses existing HTTP utilities
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from api.utils.config.http_config import AsyncHttpClient, HttpRequestError

logger = logging.getLogger("agent_sdk.knowledge_retriever")


@dataclass
class KnowledgeRetriever:
    """
    Knowledge Base Retrieval Service.

    Retrieves relevant knowledge based on user query.
    Designed for minimal invasion into WorkflowAgentV2.

    Usage:
        retriever = KnowledgeRetriever(
            url="http://example.com/retrieve-knowledge",
            context_vars={"chatbotId": "xxx", "tenantId": "yyy"}
        )
        kb_content = await retriever.retrieve(query="user question")
    """

    url: str
    context_vars: dict[str, Any] = field(default_factory=dict)
    timeout: float = 10.0
    max_results: int = 5

    _http_client: AsyncHttpClient = field(default=None, repr=False)

    def __post_init__(self):
        self._http_client = AsyncHttpClient(logger, timeout=self.timeout)

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
    ) -> str | None:
        """
        Retrieve knowledge from knowledge base.

        Args:
            query: User's current message
            session_id: Session identifier for context

        Returns:
            Formatted knowledge content string, or None if retrieval fails/empty
        """
        if not self.url:
            return None

        try:
            payload = self._build_payload(query, session_id)
            response = await self._http_client.post_json(
                url=self.url,
                payload=payload,
                timeout=self.timeout,
            )
            return self._parse_response(response)

        except HttpRequestError as e:
            logger.warning(f"KB retrieval failed: {e}")
            return None
        except Exception as e:
            logger.error(f"KB retrieval unexpected error: {e}")
            return None

    def _build_payload(
        self,
        query: str,
        session_id: str | None,
    ) -> dict:
        """Build retrieval request payload."""
        payload = {
            "query": query,
            "maxResults": self.max_results,
        }

        # Add context vars (chatbotId, tenantId, etc.)
        for key in ["chatbotId", "tenantId"]:
            if key in self.context_vars:
                payload[key] = self.context_vars[key]

        if session_id:
            payload["sessionId"] = session_id

        return payload

    def _parse_response(self, response: dict) -> str | None:
        """
        Parse retrieval response and format as knowledge content.

        Expected response format:
        {
            "code": 0,
            "data": {
                "results": [
                    {"content": "...", "score": 0.95, "source": "..."},
                    ...
                ]
            }
        }
        """
        if response.get("code") != 0:
            logger.warning(f"KB retrieval error code: {response.get('code')}")
            return None

        data = response.get("data", {})
        results = data.get("results", [])

        if not results:
            logger.debug("KB retrieval returned empty results")
            return None

        # Format results as knowledge content
        formatted_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            source = result.get("source", "")

            if content:
                part = f"[{i}] {content}"
                if source:
                    part += f"\n   Source: {source}"
                formatted_parts.append(part)

        if not formatted_parts:
            return None

        return "\n\n".join(formatted_parts)
