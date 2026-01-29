"""
数据库配置和索引定义

定义数据库名称、集合名称和索引配置
"""

from api.models.collections import COLLECTIONS

# 数据库名称
DB_NAME = "workflow_agent"


# 索引定义：(集合名, 索引名, 索引字段, 是否唯一, 其他选项)
INDEX_DEFINITIONS = [
    # parsed_configs 索引
    (COLLECTIONS.PARSED_CONFIGS, "idx_tenant_chatbot", [("tenant_id", 1), ("chatbot_id", 1)], False, {}),
    (COLLECTIONS.PARSED_CONFIGS, "idx_created_at", [("created_at", -1)], False, {}),

    # sessions 索引
    (COLLECTIONS.SESSIONS, "idx_tenant_chatbot", [("tenant_id", 1), ("chatbot_id", 1)], False, {}),
    (COLLECTIONS.SESSIONS, "idx_customer", [("tenant_id", 1), ("customer_id", 1)], False, {}),
    (COLLECTIONS.SESSIONS, "idx_status", [("status", 1), ("updated_at", -1)], False, {}),
    (COLLECTIONS.SESSIONS, "idx_created_at", [("created_at", -1)], False, {}),
    # TTL 索引：自动清理过期会话（30天）
    (COLLECTIONS.SESSIONS, "idx_ttl_closed", [("closed_at", 1)], False, {"expireAfterSeconds": 30 * 24 * 3600}),

    # session_messages 索引
    (COLLECTIONS.SESSION_MESSAGES, "idx_session", [("session_id", 1), ("created_at", 1)], False, {}),
    (COLLECTIONS.SESSION_MESSAGES, "idx_tenant_session", [("tenant_id", 1), ("session_id", 1)], False, {}),
    (COLLECTIONS.SESSION_MESSAGES, "idx_correlation", [("correlation_id", 1)], False, {}),
    (COLLECTIONS.SESSION_MESSAGES, "idx_created_at", [("created_at", -1)], False, {}),

    # agent_states 索引
    (COLLECTIONS.AGENT_STATES, "idx_tenant_chatbot", [("tenant_id", 1), ("chatbot_id", 1)], True, {}),
    (COLLECTIONS.AGENT_STATES, "idx_status", [("status", 1)], False, {}),
    (COLLECTIONS.AGENT_STATES, "idx_last_active", [("last_active_at", -1)], False, {}),

    # audit_logs 索引
    (COLLECTIONS.AUDIT_LOGS, "idx_tenant_action", [("tenant_id", 1), ("action", 1), ("created_at", -1)], False, {}),
    (COLLECTIONS.AUDIT_LOGS, "idx_session", [("session_id", 1), ("created_at", -1)], False, {}),
    (COLLECTIONS.AUDIT_LOGS, "idx_agent", [("agent_id", 1), ("created_at", -1)], False, {}),
    (COLLECTIONS.AUDIT_LOGS, "idx_correlation", [("correlation_id", 1)], False, {}),
    (COLLECTIONS.AUDIT_LOGS, "idx_created_at", [("created_at", -1)], False, {}),
    # TTL 索引：自动清理过期日志（90天）
    (COLLECTIONS.AUDIT_LOGS, "idx_ttl", [("created_at", 1)], False, {"expireAfterSeconds": 90 * 24 * 3600}),
]
