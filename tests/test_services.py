"""
服务模块单元测试

测试：
- ConfigStore (Memory/MongoDB)
- Database 管理器
- AgentManager 核心功能
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from api.services.config_store import (
    StoredConfig,
    MemoryConfigStore,
    MongoConfigStore,
    create_config_store,
)
from api.services.database import (
    DB_NAME,
    COLLECTIONS,
    Database,
    get_database,
)


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
# MemoryConfigStore 测试
# =============================================================================


class TestMemoryConfigStore:
    """内存配置存储测试"""
    
    @pytest_asyncio.fixture
    async def store(self):
        """创建内存存储实例"""
        return MemoryConfigStore()
    
    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        """测试保存和获取"""
        config = StoredConfig(
            config_hash="hash-1",
            tenant_id="tenant-1",
            chatbot_id="bot-1",
            raw_config={"test": True},
            parsed_config={"parsed": True},
        )
        
        # 保存
        await store.save(config)
        
        # 获取
        result = await store.get("hash-1")
        
        assert result is not None
        assert result.config_hash == "hash-1"
        assert result.tenant_id == "tenant-1"
    
    @pytest.mark.asyncio
    async def test_get_not_found(self, store):
        """测试获取不存在的配置"""
        result = await store.get("non-existent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_store_type(self, store):
        """测试存储类型"""
        assert store.store_type == "memory"
    
    @pytest.mark.asyncio
    async def test_multiple_configs(self, store):
        """测试多个配置"""
        for i in range(3):
            config = StoredConfig(
                config_hash=f"hash-{i}",
                tenant_id="tenant-1",
                chatbot_id=f"bot-{i}",
                raw_config={},
                parsed_config={},
            )
            await store.save(config)
        
        assert len(store) == 3
        
        # 验证每个都能获取
        for i in range(3):
            result = await store.get(f"hash-{i}")
            assert result is not None
            assert result.chatbot_id == f"bot-{i}"


# =============================================================================
# MongoConfigStore 测试
# =============================================================================


class TestMongoConfigStore:
    """MongoDB 配置存储测试"""
    
    @pytest_asyncio.fixture
    async def mock_db(self):
        """创建 Mock MongoDB"""
        db = MagicMock()
        collection = MagicMock()
        db.__getitem__ = MagicMock(return_value=collection)
        
        # Mock async methods
        collection.find_one = AsyncMock(return_value=None)
        collection.replace_one = AsyncMock()
        collection.create_index = AsyncMock()
        
        return db, collection
    
    @pytest.mark.asyncio
    async def test_store_type(self, mock_db):
        """测试存储类型"""
        db, _ = mock_db
        store = MongoConfigStore(db)
        assert store.store_type == "mongodb"
    
    @pytest.mark.asyncio
    async def test_save(self, mock_db):
        """测试保存到 MongoDB"""
        db, collection = mock_db
        store = MongoConfigStore(db)
        
        config = StoredConfig(
            config_hash="hash-1",
            tenant_id="tenant-1",
            chatbot_id="bot-1",
            raw_config={},
            parsed_config={},
        )
        
        await store.save(config)
        
        # 验证调用了 replace_one
        collection.replace_one.assert_called_once()
        call_args = collection.replace_one.call_args
        assert call_args[0][0] == {"_id": "hash-1"}
        assert call_args[1]["upsert"] is True
    
    @pytest.mark.asyncio
    async def test_get_found(self, mock_db):
        """测试从 MongoDB 获取（存在）"""
        db, collection = mock_db
        
        # Mock 返回数据
        collection.find_one = AsyncMock(return_value={
            "_id": "hash-1",
            "tenant_id": "tenant-1",
            "chatbot_id": "bot-1",
            "raw_config": {},
            "parsed_config": {},
            "created_at": datetime.utcnow(),
        })
        
        store = MongoConfigStore(db)
        result = await store.get("hash-1")
        
        assert result is not None
        assert result.config_hash == "hash-1"
    
    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_db):
        """测试从 MongoDB 获取（不存在）"""
        db, collection = mock_db
        collection.find_one = AsyncMock(return_value=None)
        
        store = MongoConfigStore(db)
        result = await store.get("non-existent")
        
        assert result is None


# =============================================================================
# create_config_store 工厂函数测试
# =============================================================================


class TestCreateConfigStore:
    """配置存储工厂函数测试"""
    
    def test_create_memory_store(self):
        """测试创建内存存储"""
        store = create_config_store(mongo_db=None)
        assert isinstance(store, MemoryConfigStore)
        assert store.store_type == "memory"
    
    def test_create_mongo_store(self):
        """测试创建 MongoDB 存储"""
        mock_db = MagicMock()
        store = create_config_store(mongo_db=mock_db)
        assert isinstance(store, MongoConfigStore)
        assert store.store_type == "mongodb"


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
        assert COLLECTIONS.PARSED_CONFIGS == "parsed_configs"
        assert COLLECTIONS.SESSIONS == "sessions"
        assert COLLECTIONS.SESSION_MESSAGES == "session_messages"
        assert COLLECTIONS.AGENT_STATES == "agent_states"
        assert COLLECTIONS.AUDIT_LOGS == "audit_logs"


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
            db.collection(COLLECTIONS.PARSED_CONFIGS)
    
    def test_collection_connected(self):
        """测试已连接时获取集合"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        
        db = Database(mongo_client=mock_client)
        collection = db.collection(COLLECTIONS.PARSED_CONFIGS)
        
        mock_db.__getitem__.assert_called_with(COLLECTIONS.PARSED_CONFIGS)
    
    def test_shortcut_properties(self):
        """测试快捷属性"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__ = MagicMock(return_value=mock_db)
        mock_db.__getitem__ = MagicMock(return_value=MagicMock())
        
        db = Database(mongo_client=mock_client)
        
        # 访问快捷属性
        _ = db.parsed_configs
        _ = db.sessions
        _ = db.session_messages
        
        # 验证调用
        calls = mock_db.__getitem__.call_args_list
        collection_names = [call[0][0] for call in calls]
        
        assert COLLECTIONS.PARSED_CONFIGS in collection_names
        assert COLLECTIONS.SESSIONS in collection_names
        assert COLLECTIONS.SESSION_MESSAGES in collection_names


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
