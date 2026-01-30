"""
集合名称定义 v3

定义 MongoDB 集合名称常量
5表设计：configs, sessions, messages, events, usages
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Collections:
    """
    集合名称定义 (v3)

    基于 data_model_design.md v3 设计:
    - configs: 配置缓存
    - sessions: 会话 (唯一存储 tenant_id/chatbot_id)
    - messages: 消息 + 嵌入状态
    - events: 事件日志 (纯事件，增加 offset)
    - usages: Token 消耗 (明细 + 汇总)
    """

    # v3 核心表
    CONFIGS = "configs"                    # 配置缓存
    SESSIONS = "sessions"                  # 会话
    MESSAGES = "messages"                  # 消息 + 状态
    EVENTS = "events"                      # 事件日志
    USAGES = "usages"                      # Token 消耗 (v3 新增)


# 全局实例
COLLECTIONS = Collections()
