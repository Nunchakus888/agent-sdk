"""
服务模块单元测试

测试：
- Database 管理器
- ConfigDocumentV2 数据模型
- ConfigRepository
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from api.services.database import (
    DB_NAME,
    COLLECTIONS,
    Database,
    get_database,
)
from api.models.documents_v2 import ConfigDocumentV2
from api.services.repositories import ConfigRepository


# =============================================================================
# ConfigDocumentV2 测试
# =============================================================================


class TestConfigDocumentV2:
    """ConfigDocumentV2 数据模型测试"""

    def test_create(self):
        """测试创建"""
        config = ConfigDocumentV2(
            chatbot_id="bot-1",
            tenant_id="tenant-1",
            config_hash="abc123",
            raw_config={"key": "value"},
            parsed_config={"parsed": True},
        )
        assert config.chatbot_id == "bot-1"
        assert config.config_hash == "abc123"
        assert config.access_count == 0

    def test_to_dict(self):
        """测试 to_dict - _id = chatbot_id"""
        config = ConfigDocumentV2(
            chatbot_id="bot-1",
            tenant_id="tenant-1",
            config_hash="abc123",
            raw_config={},
            parsed_config={},
        )
        data = config.to_dict()
        assert data["_id"] == "bot-1"
        assert "chatbot_id" not in data  # 不重复存储

    def test_from_dict(self):
        """测试 from_dict"""
        data = {
            "_id": "bot-1",
            "tenant_id": "tenant-1",
            "config_hash": "abc123",
            "raw_config": {},
            "parsed_config": {},
            "access_count": 5,
        }
        config = ConfigDocumentV2.from_dict(data)
        assert config.chatbot_id == "bot-1"
        assert config.access_count == 5


# =============================================================================
# ConfigRepository 测试
# =============================================================================


class TestConfigRepository:
    """ConfigRepository 测试"""

    @pytest.mark.asyncio
    async def test_upsert(self):
        """测试 upsert"""
        repo = ConfigRepository(db=None)
        config = await repo.upsert(
            chatbot_id="bot-1",
            tenant_id="tenant-1",
            config_hash="hash-1",
            raw_config={},
            parsed_config={},
        )
        assert config.chatbot_id == "bot-1"
        assert config.access_count == 1

    @pytest.mark.asyncio
    async def test_get_with_access_count(self):
        """测试 get 自动 access_count+1"""
        repo = ConfigRepository(db=None)
        await repo.upsert(
            chatbot_id="bot-1",
            tenant_id="tenant-1",
            config_hash="hash-1",
            raw_config={},
            parsed_config={},
        )
        config = await repo.get("bot-1")
        assert config.access_count == 2

    @pytest.mark.asyncio
    async def test_get_with_hash_check(self):
        """测试 get 带 hash 检查"""
        repo = ConfigRepository(db=None)
        await repo.upsert(
            chatbot_id="bot-1",
            tenant_id="tenant-1",
            config_hash="hash-1",
            raw_config={},
            parsed_config={},
        )
        assert await repo.get("bot-1", expected_hash="hash-1") is not None
        assert await repo.get("bot-1", expected_hash="wrong") is None

    @pytest.mark.asyncio
    async def test_invalidate(self):
        """测试 invalidate"""
        repo = ConfigRepository(db=None)
        await repo.upsert(
            chatbot_id="bot-1",
            tenant_id="tenant-1",
            config_hash="hash-1",
            raw_config={},
            parsed_config={},
        )
        assert await repo.invalidate("bot-1") is True
        assert await repo.invalidate("bot-1") is False


# =============================================================================
# Database 常量测试
# =============================================================================


class TestDatabaseConstants:
    """数据库常量测试"""

    def test_db_name(self):
        """测试数据库名称"""
        assert DB_NAME == "workflow_agent"

    def test_collections(self):
        """测试集合名称"""
        assert COLLECTIONS.CONFIGS == "configs"
        assert COLLECTIONS.SESSIONS == "sessions"
        assert COLLECTIONS.MESSAGES == "messages"
        assert COLLECTIONS.EVENTS == "events"
        assert COLLECTIONS.USAGES == "usages"


# =============================================================================
# Database 管理器测试
# =============================================================================


class TestDatabase:
    """数据库管理器测试"""

    def test_not_connected(self):
        """测试未连接状态"""
        db = Database(mongo_client=None)
        assert db.is_connected is False

    def test_connected(self):
        """测试已连接状态"""
        mock_client = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=MagicMock())

        db = Database(mongo_client=mock_client)
        assert db.is_connected is True
        assert db.name == DB_NAME

    def test_collection_not_connected(self):
        """测试未连接时获取集合"""
        db = Database(mongo_client=None)

        with pytest.raises(RuntimeError, match="Database not connected"):
            db.collection(COLLECTIONS.CONFIGS)

    def test_collection_connected(self):
        """测试已连接时获取集合"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)

        db = Database(mongo_client=mock_client)
        collection = db.collection(COLLECTIONS.CONFIGS)

        mock_db.__getitem__.assert_called_with(COLLECTIONS.CONFIGS)

    def test_shortcut_properties(self):
        """测试快捷属性"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.__getitem__ = MagicMock(return_value=MagicMock())

        db = Database(mongo_client=mock_client)

        # 访问快捷属性
        _ = db.configs
        _ = db.sessions
        _ = db.messages
        _ = db.events
        _ = db.usages

        # 验证调用
        calls = mock_db.__getitem__.call_args_list
        collection_names = [call[0][0] for call in calls]

        assert COLLECTIONS.CONFIGS in collection_names
        assert COLLECTIONS.SESSIONS in collection_names
        assert COLLECTIONS.MESSAGES in collection_names
        assert COLLECTIONS.EVENTS in collection_names
        assert COLLECTIONS.USAGES in collection_names


# =============================================================================
# get_database 工厂函数测试
# =============================================================================


class TestGetDatabase:
    """get_database 工厂函数测试"""

    def test_no_client(self):
        """测试无客户端"""
        result = get_database(mongo_client=None)
        assert result is None

    def test_with_client(self):
        """测试有客户端"""
        mock_client = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=MagicMock())

        result = get_database(mongo_client=mock_client)

        assert result is not None
        assert isinstance(result, Database)
        assert result.is_connected is True
