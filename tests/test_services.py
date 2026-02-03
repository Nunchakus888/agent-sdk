"""
服务模块单元测试

测试：
- Database 管理器
- ConfigLoader
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
from api.services.v2.config_loader import StoredConfig, ConfigLoader


# =============================================================================
# StoredConfig 测试
# =============================================================================


class TestStoredConfig:
    """StoredConfig 数据模型测试"""

    def test_create_stored_config(self):
        """测试创建 StoredConfig"""
        config = StoredConfig(
            config_hash="abc123",
            tenant_id="tenant-1",
            chatbot_id="bot-1",
            raw_config={"key": "value"},
            parsed_config={"parsed": True},
        )

        assert config.config_hash == "abc123"
        assert config.tenant_id == "tenant-1"
        assert config.chatbot_id == "bot-1"
        assert config.raw_config == {"key": "value"}
        assert config.parsed_config == {"parsed": True}
        assert isinstance(config.created_at, datetime)

    def test_to_dict(self):
        """测试转换为字典"""
        config = StoredConfig(
            config_hash="abc123",
            tenant_id="tenant-1",
            chatbot_id="bot-1",
            raw_config={"key": "value"},
            parsed_config={"parsed": True},
        )

        data = config.to_dict()

        assert data["_id"] == "abc123"
        assert data["tenant_id"] == "tenant-1"
        assert data["chatbot_id"] == "bot-1"
        assert data["raw_config"] == {"key": "value"}
        assert data["parsed_config"] == {"parsed": True}

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "_id": "abc123",
            "tenant_id": "tenant-1",
            "chatbot_id": "bot-1",
            "raw_config": {"key": "value"},
            "parsed_config": {"parsed": True},
            "created_at": datetime.utcnow(),
        }

        config = StoredConfig.from_dict(data)

        assert config.config_hash == "abc123"
        assert config.tenant_id == "tenant-1"
        assert config.chatbot_id == "bot-1"


# =============================================================================
# ConfigLoader 测试
# =============================================================================


class TestConfigLoader:
    """ConfigLoader 测试"""

    def test_create_without_database(self):
        """测试创建 ConfigLoader（无数据库）"""
        loader = ConfigLoader(database=None)
        stats = loader.get_stats()

        assert stats["l2_enabled"] is False
        assert stats["llm_parsing"] is False

    def test_create_with_database(self):
        """测试创建 ConfigLoader（有数据库）"""
        mock_db = MagicMock()
        loader = ConfigLoader(database=mock_db)
        stats = loader.get_stats()

        assert stats["l2_enabled"] is True

    def test_invalidate(self):
        """测试使配置失效"""
        loader = ConfigLoader(database=None)
        # 先设置一个配置到 L1
        loader._l1.set("hash-1", MagicMock())

        # 使其失效
        result = loader.invalidate("hash-1")
        assert result is True

        # 再次失效应返回 False
        result = loader.invalidate("hash-1")
        assert result is False


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
