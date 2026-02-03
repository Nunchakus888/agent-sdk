"""
数据库管理器 V2

提供统一的数据库访问入口
6表设计：configs, sessions, messages, tool_calls, usages, timers
"""

import logging
from typing import Any

from .collections import COLLECTIONS
from api.services.database.config import DB_NAME, INDEX_DEFINITIONS

logger = logging.getLogger(__name__)


class Database:
    """
    数据库管理器 (V2)

    提供统一的数据库访问入口
    6表设计：configs, sessions, messages, tool_calls, usages, timers

    Usage:
        db = Database(mongo_client)
        configs = db.configs
        await configs.find_one({"_id": "hash123"})
    """

    def __init__(self, mongo_client: Any | None, db_name: str = DB_NAME):
        self._client = mongo_client
        self._db_name = db_name
        self._db = mongo_client[db_name] if mongo_client else None
        self._indexes_created = False

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._db is not None

    @property
    def name(self) -> str:
        """数据库名称"""
        return self._db_name

    @property
    def client(self) -> Any:
        """获取 MongoDB 客户端"""
        return self._client

    @property
    def db(self) -> Any:
        """获取 MongoDB 数据库实例"""
        return self._db

    def collection(self, name: str) -> Any:
        """获取集合"""
        if self._db is None:
            raise RuntimeError("Database not connected")
        return self._db[name]

    # =========================================================================
    # V2 核心集合
    # =========================================================================

    @property
    def configs(self) -> Any:
        """配置集合"""
        return self.collection(COLLECTIONS.CONFIGS)

    @property
    def sessions(self) -> Any:
        """会话集合"""
        return self.collection(COLLECTIONS.SESSIONS)

    @property
    def messages(self) -> Any:
        """消息集合"""
        return self.collection(COLLECTIONS.MESSAGES)

    @property
    def tool_calls(self) -> Any:
        """工具调用集合"""
        return self.collection(COLLECTIONS.TOOL_CALLS)

    @property
    def usages(self) -> Any:
        """Token 消耗集合"""
        return self.collection(COLLECTIONS.USAGES)

    @property
    def timers(self) -> Any:
        """定时器集合"""
        return self.collection(COLLECTIONS.TIMERS)

    async def ensure_indexes(self) -> None:
        """确保所有索引已创建（幂等操作）"""
        if self._db is None:
            logger.warning("Database not connected, skipping index creation")
            return

        if self._indexes_created:
            return

        logger.info(f"Creating database indexes: {self._db_name}")

        created = existing = errors = 0

        for collection_name, index_name, fields, unique, options in INDEX_DEFINITIONS:
            try:
                collection = self._db[collection_name]
                index_options = {
                    "name": index_name,
                    "unique": unique,
                    "background": True,
                    **options,
                }
                await collection.create_index(fields, **index_options)
                logger.debug(f"Index created: {collection_name}.{index_name}")
                created += 1
            except Exception as e:
                error_msg = str(e).lower()
                if "already exists" in error_msg or "indexoptionsconflict" in error_msg:
                    existing += 1
                else:
                    logger.warning(f"Failed to create index {collection_name}.{index_name}: {e}")
                    errors += 1

        self._indexes_created = True
        logger.info(f"Indexes: created={created}, existing={existing}, errors={errors}")

    async def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        if self._db is None:
            return {}

        stats = {}
        for name in [
            COLLECTIONS.CONFIGS,
            COLLECTIONS.SESSIONS,
            COLLECTIONS.MESSAGES,
            COLLECTIONS.TOOL_CALLS,
            COLLECTIONS.USAGES,
            COLLECTIONS.TIMERS,
        ]:
            try:
                count = await self._db[name].count_documents({})
                stats[name] = {"count": count}
            except Exception as e:
                stats[name] = {"error": str(e)}

        return stats


def get_database(mongo_client: Any | None, db_name: str = DB_NAME) -> Database | None:
    """获取数据库实例"""
    if mongo_client is None:
        return None
    return Database(mongo_client, db_name)
