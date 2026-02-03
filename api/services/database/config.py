"""
数据库配置和索引定义 V2

定义数据库名称、集合名称和索引配置
6表设计：configs, sessions, messages, tool_calls, usages, timers
"""

from .collections import COLLECTIONS

# 数据库名称
DB_NAME = "workflow_agent"


# 索引定义：(集合名, 索引名, 索引字段, 是否唯一, 其他选项)
INDEX_DEFINITIONS = [
    # ==========================================================================
    # configs 索引
    # ==========================================================================
    # 按 tenant_id + chatbot_id 查询最新配置
    (COLLECTIONS.CONFIGS, "idx_tenant_chatbot", [("tenant_id", 1), ("chatbot_id", 1), ("updated_at", -1)], False, {}),

    # ==========================================================================
    # sessions 索引
    # ==========================================================================
    # 按租户列出会话
    (COLLECTIONS.SESSIONS, "idx_tenant_chatbot_created", [("tenant_id", 1), ("chatbot_id", 1), ("created_at", -1)], False, {}),
    # 按客户查询
    (COLLECTIONS.SESSIONS, "idx_customer", [("customer_id", 1)], False, {"sparse": True}),

    # ==========================================================================
    # messages 索引
    # ==========================================================================
    # 按会话列出消息（时间排序）
    (COLLECTIONS.MESSAGES, "idx_session_created", [("session_id", 1), ("created_at", 1)], False, {}),
    # 按 correlation_id 查询
    (COLLECTIONS.MESSAGES, "idx_correlation", [("correlation_id", 1)], False, {"sparse": True}),

    # ==========================================================================
    # tool_calls 索引
    # ==========================================================================
    # 按会话列出工具调用（offset 排序）
    (COLLECTIONS.TOOL_CALLS, "idx_session_offset", [("session_id", 1), ("offset", 1)], False, {}),
    # 按 correlation_id 查询（全链路追踪）
    (COLLECTIONS.TOOL_CALLS, "idx_correlation", [("correlation_id", 1)], False, {}),

    # ==========================================================================
    # usages 索引
    # ==========================================================================
    # 按 correlation_id 唯一索引
    (COLLECTIONS.USAGES, "idx_correlation", [("correlation_id", 1)], True, {}),
    # 按会话统计
    (COLLECTIONS.USAGES, "idx_session_created", [("session_id", 1), ("created_at", -1)], False, {}),

    # ==========================================================================
    # timers 索引
    # ==========================================================================
    # 按会话 + timer_id 查询
    (COLLECTIONS.TIMERS, "idx_session_timer", [("session_id", 1), ("timer_id", 1)], False, {}),
    # 关键：超时查询（status=pending, next_trigger_at < now）
    (COLLECTIONS.TIMERS, "idx_status_trigger", [("status", 1), ("next_trigger_at", 1)], False, {}),
    # 按 next_trigger_at 查询（稀疏索引）
    (COLLECTIONS.TIMERS, "idx_next_trigger", [("next_trigger_at", 1)], False, {"sparse": True}),
]
