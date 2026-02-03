"""
服务层

核心组件：
- SessionContext: 会话上下文
- SessionManager: 会话管理器（内部处理配置加载，DB → HTTP）
- EventCollector: 统一事件收集器
- QueryRecorder: 统一 DB 写入
"""

from api.services.v2.session_context import SessionContext, SessionTimer
from api.services.v2.session_manager import SessionManager
from api.services.v2.event_collector import EventCollector, QueryResult, ToolCallRecord
from api.services.v2.query_recorder import QueryRecorder

__all__ = [
    "SessionContext",
    "SessionTimer",
    "SessionManager",
    "EventCollector",
    "QueryResult",
    "ToolCallRecord",
    "QueryRecorder",
]
