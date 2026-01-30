"""
配置存储抽象层

设计原则：
- 简单直接，无复杂缓存逻辑
- Memory 模式：简单 dict，用于开发/单实例
- MongoDB 模式：直接访问，MongoDB 自带缓存
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from api.services.database import COLLECTIONS

logger = logging.getLogger(__name__)


# =============================================================================
# 数据模型
# =============================================================================


@dataclass
class StoredConfig:
    """持久化配置数据"""
    config_hash: str          # 主键
    tenant_id: str
    chatbot_id: str
    raw_config: dict          # 原始配置
    parsed_config: dict       # 解析后配置（可序列化）
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """转为 MongoDB 文档"""
        return {
            "_id": self.config_hash,
            "tenant_id": self.tenant_id,
            "chatbot_id": self.chatbot_id,
            "raw_config": self.raw_config,
            "parsed_config": self.parsed_config,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StoredConfig":
        """从 MongoDB 文档创建"""
        return cls(
            config_hash=data.get("_id") or data.get("config_hash"),
            tenant_id=data["tenant_id"],
            chatbot_id=data["chatbot_id"],
            raw_config=data["raw_config"],
            parsed_config=data["parsed_config"],
            created_at=data.get("created_at", datetime.utcnow()),
        )


# =============================================================================
# 存储接口
# =============================================================================


class ConfigStore(ABC):
    """配置存储抽象基类"""
    
    @abstractmethod
    async def get(self, config_hash: str) -> StoredConfig | None:
        """获取配置"""
        ...
    
    @abstractmethod
    async def save(self, config: StoredConfig) -> None:
        """保存配置"""
        ...
    
    @property
    @abstractmethod
    def store_type(self) -> str:
        """存储类型"""
        ...


# =============================================================================
# 内存存储（开发/单实例）
# =============================================================================


class MemoryConfigStore(ConfigStore):
    """
    内存配置存储
    
    特点：
    - 简单 dict，无大小限制
    - 进程重启后清空
    - 适用于开发和单实例部署
    """
    
    def __init__(self):
        self._data: dict[str, StoredConfig] = {}
    
    async def get(self, config_hash: str) -> StoredConfig | None:
        return self._data.get(config_hash)
    
    async def save(self, config: StoredConfig) -> None:
        self._data[config.config_hash] = config
    
    @property
    def store_type(self) -> str:
        return "memory"
    
    def __len__(self) -> int:
        return len(self._data)


# =============================================================================
# MongoDB 存储（生产/多实例）
# =============================================================================


class MongoConfigStore(ConfigStore):
    """
    MongoDB 配置存储
    
    特点：
    - 直接访问 MongoDB，无额外缓存层
    - MongoDB 自带查询缓存
    - 适用于生产环境和多实例部署
    """
    
    def __init__(self, db: Any, collection_name: str = COLLECTIONS.CONFIGS):
        self._collection = db[collection_name]
        self._indexes_created = False
    
    async def _ensure_indexes(self) -> None:
        """惰性创建索引"""
        if self._indexes_created:
            return
        try:
            await self._collection.create_index(
                [("tenant_id", 1), ("chatbot_id", 1)],
                name="idx_tenant_chatbot",
                background=True,
            )
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            # 索引可能已存在，忽略错误
            pass
        self._indexes_created = True
    
    async def get(self, config_hash: str) -> StoredConfig | None:
        await self._ensure_indexes()
        doc = await self._collection.find_one({"_id": config_hash})
        return StoredConfig.from_dict(doc) if doc else None
    
    async def save(self, config: StoredConfig) -> None:
        await self._ensure_indexes()
        await self._collection.replace_one(
            {"_id": config.config_hash},
            config.to_dict(),
            upsert=True,
        )
    
    @property
    def store_type(self) -> str:
        return "mongodb"


# =============================================================================
# 工厂函数
# =============================================================================


def create_config_store(mongo_db: Any | None = None) -> ConfigStore:
    """
    创建配置存储
    
    Args:
        mongo_db: MongoDB 数据库实例，None 则使用内存存储
    
    Returns:
        ConfigStore 实例
    """
    if mongo_db is not None:
        logger.info("ConfigStore: mongodb")
        return MongoConfigStore(mongo_db)
    else:
        logger.info("ConfigStore: memory")
        return MemoryConfigStore()
