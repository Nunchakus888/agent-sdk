"""
会话上下文

封装单个会话的所有状态：
- WorkflowAgentV2 实例
- Timer 配置
- 会话元数据
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2
from api.utils.datetime import utc_now
logger = logging.getLogger(__name__)


@dataclass
class SessionTimer:
    """
    Session-level Timer

    Manage the timeout logic for a single session:
    - Trigger reminder when user is unresponsive
    - Support multiple triggers (with a limit)
    - Reset when user is active
    """
    session_id: str
    timeout_seconds: int = 300
    message: str = "Hello, are you still there?"
    max_triggers: int = 3
    trigger_count: int = 0
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    # Timer chain (optional)
    next_timer_name: Optional[str] = None
    action: str = "send_message"  # send_message | close_conversation

    def is_exhausted(self) -> bool:
        """Whether the trigger count has been exhausted"""
        return self.trigger_count >= self.max_triggers

    def increment(self):
        """Increment the trigger count"""
        self.trigger_count += 1

    def reset(self):
        """Reset the trigger count"""
        self.trigger_count = 0

    def cancel(self):
        """Cancel the Timer task"""
        if self.task and not self.task.done():
            self.task.cancel()
            self.task = None


@dataclass
class SessionContext:
    """
    Session context

    Wrap all states of a single session, with the same lifecycle as the session.

    Attributes:
        session_id: Session ID
        tenant_id: Tenant ID
        chatbot_id: Chatbot ID
        agent: WorkflowAgentV2 instance
        timer: Session-level Timer
    """
    session_id: str
    tenant_id: str
    chatbot_id: str

    # Agent instance (session-level)
    agent: WorkflowAgentV2 = field(default=None, repr=False)

    # Timer (session-level)
    timer: Optional[SessionTimer] = None

    # Metadata
    created_at: datetime = field(default_factory=utc_now)
    last_active_at: datetime = field(default_factory=utc_now)

    # Statistics
    query_count: int = 0

    def touch(self):
        """Update the active time"""
        self.last_active_at = utc_now()

    def increment_query(self):
        """Increment the query count"""
        self.query_count += 1
        self.touch()

    @property
    def idle_seconds(self) -> float:
        """Idle time (seconds)"""
        return (utc_now() - self.last_active_at).total_seconds()

    @property
    def age_seconds(self) -> float:
        """Session存活时间 (seconds)"""
        return (utc_now() - self.created_at).total_seconds()

    def cleanup(self):
        """Clean up resources"""
        if self.timer:
            self.timer.cancel()
        if self.agent:
            self.agent.clear_history()
        logger.debug(f"SessionContext cleaned up: {self.session_id}")

    def to_dict(self) -> dict:
        """Convert to dictionary (for debugging/monitoring)"""
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "chatbot_id": self.chatbot_id,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "idle_seconds": round(self.idle_seconds, 1),
            "query_count": self.query_count,
            "has_timer": self.timer is not None,
            "timer_trigger_count": self.timer.trigger_count if self.timer else 0,
        }
