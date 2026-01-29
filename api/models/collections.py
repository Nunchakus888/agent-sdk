"""
集合名称定义

定义 MongoDB 集合名称常量
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Collections:
    """集合名称定义"""

    # 配置存储
    PARSED_CONFIGS = "parsed_configs"      # 解析后的配置

    # 会话存储
    SESSIONS = "sessions"                  # 会话数据
    SESSION_MESSAGES = "session_messages"  # 会话消息历史

    # Agent 相关
    AGENT_STATES = "agent_states"          # Agent 状态快照

    # 日志/审计
    AUDIT_LOGS = "audit_logs"              # 审计日志


# 全局实例
COLLECTIONS = Collections()
