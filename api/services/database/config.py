"""
数据库配置和索引定义 v3

定义数据库名称、集合名称和索引配置
5表设计：configs, sessions, messages, events, usages
"""

from api.models.collections import COLLECTIONS

# 数据库名称
DB_NAME = "workflow_agent"


# 索引定义：(集合名, 索引名, 索引字段, 是否唯一, 其他选项)
INDEX_DEFINITIONS = [
    # ==========================================================================
    # configs 索引
    # ==========================================================================
    (COLLECTIONS.CONFIGS, "idx_tenant_chatbot", [("tenant_id", 1), ("chatbot_id", 1)], False, {}),
    (COLLECTIONS.CONFIGS, "idx_created_at", [("created_at", -1)], False, {}),

    # ==========================================================================
    # sessions 索引
    # ==========================================================================
    (COLLECTIONS.SESSIONS, "idx_tenant_chatbot_status", [("tenant_id", 1), ("chatbot_id", 1), ("status", 1)], False, {}),
    (COLLECTIONS.SESSIONS, "idx_tenant_created", [("tenant_id", 1), ("created_at", -1)], False, {}),
    (COLLECTIONS.SESSIONS, "idx_customer", [("customer_id", 1)], False, {}),
    # TTL 索引：自动清理已关闭会话（30天）
    # (COLLECTIONS.SESSIONS, "idx_ttl_closed", [("closed_at", 1)], False, {"expireAfterSeconds": 30 * 24 * 3600}),

    # ==========================================================================
    # messages 索引
    # ==========================================================================
    (COLLECTIONS.MESSAGES, "idx_session_created", [("session_id", 1), ("created_at", 1)], False, {}),
    (COLLECTIONS.MESSAGES, "idx_correlation", [("correlation_id", 1)], False, {}),

    # ==========================================================================
    # events 索引
    # ==========================================================================
    (COLLECTIONS.EVENTS, "idx_session_offset", [("session_id", 1), ("offset", 1)], True, {}),  # 唯一索引
    (COLLECTIONS.EVENTS, "idx_session_type", [("session_id", 1), ("event_type", 1)], False, {}),
    (COLLECTIONS.EVENTS, "idx_correlation", [("correlation_id", 1)], False, {}),
    # TTL 索引：自动清理事件日志（30天）
    # (COLLECTIONS.EVENTS, "idx_ttl", [("created_at", 1)], False, {"expireAfterSeconds": 30 * 24 * 3600}),

    # ==========================================================================
    # usages 索引
    # ==========================================================================
    (COLLECTIONS.USAGES, "idx_correlation", [("correlation_id", 1)], True, {}),  # 唯一索引
    (COLLECTIONS.USAGES, "idx_session", [("session_id", 1)], False, {}),
    (COLLECTIONS.USAGES, "idx_session_finalized", [("session_id", 1), ("is_finalized", 1)], False, {}),
]
