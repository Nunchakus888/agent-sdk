"""
数据库管理器 v3

提供统一的数据库访问入口
5表设计：configs, sessions, messages, events, usages
"""

import logging
from typing import Any

from api.models.collections import COLLECTIONS
from api.services.database.config import DB_NAME, INDEX_DEFINITIONS

logger = logging.getLogger(__name__)


class Database:
    """
    数据库管理器 (v3)

    提供统一的数据库访问入口
    5表设计：configs, sessions, messages, events, usages

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

        if self._db is not None:
            logger.info(f"Database connected: {db_name}")

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
        if self._db is None:
            raise RuntimeError("Database not connected")
        return self._db[name]

    # =========================================================================
    # v3 核心集合
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
    def events(self) -> Any:
        """事件集合"""
        return self.collection(COLLECTIONS.EVENTS)

    @property
    def usages(self) -> Any:
        """Token 消耗集合"""
        return self.collection(COLLECTIONS.USAGES)

    async def ensure_indexes(self) -> None:
        """
        确保所有索引已创建

        在服务启动时调用，幂等操作
        """
        if self._db is None:
            logger.warning("Database not connected, skipping index creation")
            return
            
        if self._indexes_created:
            logger.debug("Indexes already created, skipping")
            return

        logger.info(f"Creating database indexes for database: {self._db_name}")

        created_count = 0
        existing_count = 0
        error_count = 0

        for collection_name, index_name, fields, unique, options in INDEX_DEFINITIONS:
            try:
                collection = self._db[collection_name]

                # 构建索引选项
                index_options = {
                    "name": index_name,
                    "unique": unique,
                    "background": True,  # 后台创建，不阻塞
                    **options,
                }

                await collection.create_index(fields, **index_options)
                logger.info(f"✓ Index created: {collection_name}.{index_name}")
                created_count += 1

            except Exception as e:
                # 索引已存在时会抛出异常，忽略
                error_msg = str(e).lower()
                if "already exists" in error_msg or "duplicate key" in error_msg or "indexoptionsconflict" in error_msg:
                    logger.debug(f"○ Index already exists: {collection_name}.{index_name}")
                    existing_count += 1
                else:
                    logger.warning(f"✗ Failed to create index {collection_name}.{index_name}: {e}")
                    error_count += 1

        self._indexes_created = True
        logger.info(
            f"Database indexes creation completed - "
            f"created: {created_count}, existing: {existing_count}, errors: {error_count}"
        )

    async def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        if self._db is None:
            return {}

        stats = {}
        for collection_name in [
            COLLECTIONS.CONFIGS,
            COLLECTIONS.SESSIONS,
            COLLECTIONS.MESSAGES,
            COLLECTIONS.EVENTS,
            COLLECTIONS.USAGES,
        ]:
            try:
                count = await self._db[collection_name].count_documents({})
                stats[collection_name] = {"count": count}
            except Exception as e:
                stats[collection_name] = {"error": str(e)}

        return stats


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
