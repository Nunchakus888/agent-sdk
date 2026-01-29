"""
Plan cache for workflow configuration optimization.

Based on workflow-agent-v9.md design.
"""

import hashlib
import json
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

from bu_agent_sdk.tools.actions import WorkflowConfigSchema


class CachedPlan(BaseModel):
    """Cached execution plan."""

    workflow_id: str
    config_hash: str

    # Optimized config (optional)
    optimized_sop: str | None = None
    optimized_constraints: str | None = None

    # Index data (speed up lookup)
    tool_names: list[str]
    skill_ids: list[str]
    flow_ids: list[str]

    # Metadata
    created_at: datetime
    expires_at: datetime | None = None


class PlanCache(Protocol):
    """Plan cache interface."""

    async def get(self, workflow_id: str, config_hash: str) -> CachedPlan | None:
        """Get cached plan."""
        ...

    async def set(self, plan: CachedPlan) -> None:
        """Save plan to cache."""
        ...

    async def delete(self, workflow_id: str, config_hash: str) -> None:
        """Delete cache."""
        ...

    async def clear_all(self, workflow_id: str) -> None:
        """Clear all caches for a workflow."""
        ...


class MemoryPlanCache:
    """Memory implementation (for development and testing)."""

    def __init__(self):
        self._cache: dict[str, CachedPlan] = {}

    def _make_key(self, workflow_id: str, config_hash: str) -> str:
        return f"{workflow_id}:{config_hash}"

    async def get(self, workflow_id: str, config_hash: str) -> CachedPlan | None:
        key = self._make_key(workflow_id, config_hash)
        plan = self._cache.get(key)

        # Check expiration
        if plan and plan.expires_at and plan.expires_at < datetime.utcnow():
            await self.delete(workflow_id, config_hash)
            return None

        return plan

    async def set(self, plan: CachedPlan) -> None:
        key = self._make_key(plan.workflow_id, plan.config_hash)
        self._cache[key] = plan

    async def delete(self, workflow_id: str, config_hash: str) -> None:
        key = self._make_key(workflow_id, config_hash)
        self._cache.pop(key, None)

    async def clear_all(self, workflow_id: str) -> None:
        keys_to_delete = [k for k in self._cache if k.startswith(f"{workflow_id}:")]
        for key in keys_to_delete:
            self._cache.pop(key)


def compute_config_hash(config: WorkflowConfigSchema) -> str:
    """Compute configuration hash."""
    # Serialize config to stable JSON (sorted keys)
    import json
    config_dict = config.model_dump(exclude_none=True)
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_json.encode()).hexdigest()
