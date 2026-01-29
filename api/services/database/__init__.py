"""
数据库服务模块

提供数据库连接和管理功能
"""

from api.services.database.database import Database, get_database
from api.services.database.config import DB_NAME, INDEX_DEFINITIONS
from api.models.collections import COLLECTIONS

__all__ = [
    "DB_NAME",
    "COLLECTIONS",
    "Database",
    "get_database",
    "INDEX_DEFINITIONS",
]
