"""
数据库配置模块

统一管理数据库名称和集合定义
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 数据库配置常量
# =============================================================================

# 数据库名称
DB_NAME = "workflow_agent"


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


# =============================================================================
# 数据库管理器
# =============================================================================


class Database:
    """
    数据库管理器
    
    提供统一的数据库访问入口
    
    Usage:
        db = Database(mongo_client)
        configs = db.collection(COLLECTIONS.PARSED_CONFIGS)
        await configs.find_one({"_id": "hash123"})
    """
    
    def __init__(self, mongo_client: Any | None, db_name: str = DB_NAME):
        self._client = mongo_client
        self._db_name = db_name
        self._db = mongo_client[db_name] if mongo_client else None
        
        if self._db:
            logger.info(f"Database connected: {db_name}")
    
    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._db is not None
    
    @property
    def name(self) -> str:
        """数据库名称"""
        return self._db_name
    
    def collection(self, name: str) -> Any:
        """
        获取集合
        
        Args:
            name: 集合名称（建议使用 COLLECTIONS 常量）
        
        Returns:
            MongoDB Collection 对象
        
        Raises:
            RuntimeError: 数据库未连接
        """
        if not self._db:
            raise RuntimeError("Database not connected")
        return self._db[name]
    
    @property
    def parsed_configs(self) -> Any:
        """配置集合（快捷访问）"""
        return self.collection(COLLECTIONS.PARSED_CONFIGS)
    
    @property
    def sessions(self) -> Any:
        """会话集合（快捷访问）"""
        return self.collection(COLLECTIONS.SESSIONS)
    
    @property
    def session_messages(self) -> Any:
        """会话消息集合（快捷访问）"""
        return self.collection(COLLECTIONS.SESSION_MESSAGES)


# =============================================================================
# 工厂函数
# =============================================================================


def get_database(mongo_client: Any | None, db_name: str = DB_NAME) -> Database | None:
    """
    获取数据库实例
    
    Args:
        mongo_client: MongoDB 客户端
        db_name: 数据库名称
    
    Returns:
        Database 实例，未连接时返回 None
    """
    if mongo_client is None:
        return None
    return Database(mongo_client, db_name)
