"""
集合名称定义

定义 MongoDB 集合名称常量
6表设计：configs, sessions, messages, tool_calls, usages, timers
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Collections:
    """
    集合名称定义 (V2)

    基于 data_model_design_v2.md 设计:
    - configs: 配置缓存 (DB 级，避免重复 LLM 解析)
    - sessions: 会话元数据
    - messages: 用户/助手消息
    - tool_calls: 工具调用记录 (替代 events)
    - usages: Token 消耗 (扁平结构)
    - timers: 会话级定时器 (独立表)
    """

    CONFIGS = "configs"          # 配置缓存
    SESSIONS = "sessions"        # 会话元数据
    MESSAGES = "messages"        # 消息
    TOOL_CALLS = "tool_calls"    # 工具调用 (替代 events)
    USAGES = "usages"            # Token 消耗
    TIMERS = "timers"            # 定时器


COLLECTIONS = Collections()
