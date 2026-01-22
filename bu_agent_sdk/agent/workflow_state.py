"""
Workflow state management for session-level data.

Based on workflow-agent-v9.md design.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class WorkflowState:
    """Session-level workflow state."""

    # Config version control
    config_hash: str = ""
    """Configuration MD5 hash for change detection"""

    # Lifecycle state
    need_greeting: bool = True
    """Whether to send greeting message"""

    status: str = "ready"
    """
    Current status:
    - ready: Ready, waiting for user input
    - processing: Processing
    - typing: Generating response
    - transferred: Transferred to human
    - closed: Session closed
    """

    # Extension fields
    metadata: dict = field(default_factory=dict)
    """Custom extension fields"""

    last_updated: datetime = field(default_factory=datetime.utcnow)
    """Last update time"""


@dataclass
class Session:
    """Session data."""
    session_id: str
    agent_id: str
    workflow_state: WorkflowState
    messages: list = field(default_factory=list)
    """Message history"""

    # Execution tracking
    _execution_history: list[dict] = field(default_factory=list)
    """Execution history for current query"""

    _decisions: list[dict] = field(default_factory=list)
    """Decision history for current query"""

    def add_execution_result(
        self,
        iteration: int,
        action: dict,
        result: str,
        reasoning: str
    ) -> None:
        """Add execution result to history."""
        self._execution_history.append({
            "iteration": iteration,
            "action": action,
            "result": result,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat()
        })

    def add_decision(self, iteration: int, decision: Any) -> None:
        """Add decision to history."""
        self._decisions.append({
            "iteration": iteration,
            "decision": decision,
            "timestamp": datetime.utcnow().isoformat()
        })

    def get_execution_summary(self) -> str:
        """Get execution summary for LLM context."""
        if not self._execution_history:
            return "No actions executed yet."

        lines = []
        for entry in self._execution_history:
            action = entry["action"]
            lines.append(
                f"Iteration {entry['iteration']}: "
                f"{action.get('type')} -> {action.get('target')} "
                f"| Result: {entry['result'][:100]}..."
            )
        return "\n".join(lines)

    def clear_execution_history(self) -> None:
        """Clear execution history (called at start of new query)."""
        self._execution_history.clear()
        self._decisions.clear()
