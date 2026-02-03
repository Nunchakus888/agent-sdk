"""
Workflow state management.

ExecutionContext: 请求级执行上下文（单次 query 生命周期）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExecutionContext:
    """
    Request-level execution context.

    Lifecycle: Single query call, destroyed after execution.
    """
    session_id: str
    messages: list[dict] = field(default_factory=list)
    """Injected history messages [{"role": "user", "content": "..."}]"""

    _execution_history: list[dict] = field(default_factory=list)
    _decisions: list[dict] = field(default_factory=list)

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
        })

    def add_decision(self, iteration: int, decision: Any) -> None:
        """Add decision to history."""
        self._decisions.append({"iteration": iteration, "decision": decision})

    def get_execution_summary(self) -> str:
        """Get execution summary for LLM context."""
        if not self._execution_history:
            return "No actions executed yet."

        lines = []
        for entry in self._execution_history:
            action = entry["action"]
            result = entry["result"] if entry["result"] else ""
            lines.append(
                f"Iteration {entry['iteration']}: "
                f"{action.get('type')} -> {action.get('target')} | {result}"
            )
        return "\n".join(lines)
