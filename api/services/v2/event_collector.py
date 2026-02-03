"""
统一事件收集器

职责：
- 收集 Agent 流式事件（流式/非流式通用）
- 转换为统一的 QueryResult
- 支持增量收集
"""

from dataclasses import dataclass, field
from typing import Any
import time

from bu_agent_sdk.agent.events import (
    AgentEvent,
    ToolCallEvent,
    ToolResultEvent,
    FinalResponseEvent,
    TextEvent,
)
from bu_agent_sdk.tokens import UsageSummary


@dataclass
class ToolCallRecord:
    """Tool 调用记录"""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str = ""
    is_error: bool = False
    started_at: float = 0
    duration_ms: float = 0


@dataclass
class QueryResult:
    """Query 执行结果（统一数据结构）"""

    response: str
    usage: UsageSummary | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_duration_ms: float = 0


@dataclass
class EventCollector:
    """
    统一事件收集器

    Usage（非流式）:
        collector = EventCollector(correlation_id="xxx", session_id="yyy")
        async for event in agent.query_stream(message):
            collector.collect(event)  # 仅收集

        result = collector.to_result(usage)

    Usage（流式）:
        collector = EventCollector(correlation_id="xxx", session_id="yyy")
        async for event in agent.query_stream(message):
            collector.collect(event)
            yield format_sse_event(event)  # 收集 + 透传

        result = collector.to_result(usage)
    """

    correlation_id: str
    session_id: str
    user_message: str = ""

    # 收集的数据
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_response: str = ""
    text_chunks: list[str] = field(default_factory=list)

    # 内部状态
    _pending_calls: dict[str, ToolCallRecord] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.time)

    def collect(self, event: AgentEvent) -> None:
        """收集单个事件"""
        match event:
            case ToolCallEvent(tool=name, args=args, tool_call_id=call_id):
                record = ToolCallRecord(
                    tool_call_id=call_id,
                    tool_name=name,
                    arguments=args,
                    started_at=time.time(),
                )
                self._pending_calls[call_id] = record

            case ToolResultEvent(
                result=result, is_error=is_error, tool_call_id=call_id
            ):
                if call_id in self._pending_calls:
                    record = self._pending_calls.pop(call_id)
                    record.result = result
                    record.is_error = is_error
                    record.duration_ms = (time.time() - record.started_at) * 1000
                    self.tool_calls.append(record)

            case TextEvent(content=content):
                self.text_chunks.append(content)

            case FinalResponseEvent(content=content):
                self.final_response = content

    def to_result(self, usage: UsageSummary | None = None) -> QueryResult:
        """转换为统一的 QueryResult"""
        return QueryResult(
            response=self.final_response,
            usage=usage,
            tool_calls=list(self.tool_calls),
            total_duration_ms=(time.time() - self._start_time) * 1000,
        )

    def get_event_records(self) -> list[dict]:
        """转换为 events 表记录"""
        return [
            {
                "correlation_id": self.correlation_id,
                "session_id": self.session_id,
                "event_type": "tool_call",
                "tool_name": tc.tool_name,
                "tool_call_id": tc.tool_call_id,
                "arguments": tc.arguments,
                "result": tc.result,
                "is_error": tc.is_error,
                "duration_ms": tc.duration_ms,
            }
            for tc in self.tool_calls
        ]
