"""
V2 Services 单元测试

测试：
- ConfigCache: 配置缓存
- SessionContext: 会话上下文
- SessionManager: 会话管理器
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from api.services.v2.config_cache import ConfigCache, CachedConfig
from api.services.v2.session_context import SessionContext, SessionTimer
from api.services.v2.session_manager import SessionManager


# =============================================================================
# ConfigCache Tests
# =============================================================================


class TestConfigCache:
    """ConfigCache 单元测试"""

    def test_init(self):
        """测试初始化"""
        cache = ConfigCache(max_size=50, ttl=1800)
        assert cache._max_size == 50
        assert cache._ttl == 1800
        assert cache.size == 0

    def test_set_and_get(self):
        """测试设置和获取"""
        cache = ConfigCache()
        mock_config = MagicMock()

        cache.set("hash123", mock_config)
        assert cache.size == 1

        result = cache.get("hash123")
        assert result == mock_config

    def test_get_nonexistent(self):
        """测试获取不存在的配置"""
        cache = ConfigCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self):
        """测试 TTL 过期"""
        cache = ConfigCache(ttl=0)  # 立即过期
        mock_config = MagicMock()

        cache.set("hash123", mock_config)
        # 由于 TTL=0，应该立即过期
        result = cache.get("hash123")
        assert result is None

    def test_lru_eviction(self):
        """测试 LRU 淘汰"""
        cache = ConfigCache(max_size=2)

        cache.set("hash1", MagicMock())
        cache.set("hash2", MagicMock())

        # 访问 hash1，使其更新
        cache.get("hash1")

        # 添加 hash3，应该淘汰 hash2（最少访问）
        cache.set("hash3", MagicMock())

        assert cache.size == 2
        assert cache.get("hash1") is not None
        assert cache.get("hash2") is None
        assert cache.get("hash3") is not None

    def test_invalidate(self):
        """测试使配置失效"""
        cache = ConfigCache()
        mock_config = MagicMock()

        cache.set("hash123", mock_config)
        assert cache.size == 1

        result = cache.invalidate("hash123")
        assert result is True
        assert cache.size == 0

        # 再次失效应返回 False
        result = cache.invalidate("hash123")
        assert result is False

    def test_clear(self):
        """测试清空缓存"""
        cache = ConfigCache()

        cache.set("hash1", MagicMock())
        cache.set("hash2", MagicMock())
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0

    def test_get_stats(self):
        """测试获取统计"""
        cache = ConfigCache(max_size=100, ttl=3600)
        cache.set("hash123", MagicMock())

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["ttl"] == 3600
        assert len(stats["entries"]) == 1


# =============================================================================
# SessionTimer Tests
# =============================================================================


class TestSessionTimer:
    """SessionTimer 单元测试"""

    def test_init(self):
        """测试初始化"""
        timer = SessionTimer(
            session_id="sess_123",
            timeout_seconds=300,
            message="Hello",
            max_triggers=3,
        )
        assert timer.session_id == "sess_123"
        assert timer.timeout_seconds == 300
        assert timer.message == "Hello"
        assert timer.max_triggers == 3
        assert timer.trigger_count == 0

    def test_is_exhausted(self):
        """测试是否耗尽"""
        timer = SessionTimer(session_id="sess_123", max_triggers=2)

        assert timer.is_exhausted() is False

        timer.increment()
        assert timer.is_exhausted() is False

        timer.increment()
        assert timer.is_exhausted() is True

    def test_reset(self):
        """测试重置"""
        timer = SessionTimer(session_id="sess_123", max_triggers=2)

        timer.increment()
        timer.increment()
        assert timer.is_exhausted() is True

        timer.reset()
        assert timer.trigger_count == 0
        assert timer.is_exhausted() is False

    def test_cancel(self):
        """测试取消"""
        timer = SessionTimer(session_id="sess_123")

        # 模拟任务
        mock_task = MagicMock()
        mock_task.done.return_value = False
        timer.task = mock_task

        timer.cancel()
        mock_task.cancel.assert_called_once()
        assert timer.task is None


# =============================================================================
# SessionContext Tests
# =============================================================================


class TestSessionContext:
    """SessionContext 单元测试"""

    def test_init(self):
        """测试初始化"""
        ctx = SessionContext(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash123",
        )
        assert ctx.session_id == "sess_123"
        assert ctx.tenant_id == "tenant_1"
        assert ctx.chatbot_id == "bot_1"
        assert ctx.config_hash == "hash123"
        assert ctx.query_count == 0

    def test_touch(self):
        """测试更新活跃时间"""
        ctx = SessionContext(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash123",
        )
        old_time = ctx.last_active_at

        # 等待一小段时间
        import time
        time.sleep(0.01)

        ctx.touch()
        assert ctx.last_active_at > old_time

    def test_increment_query(self):
        """测试增加查询计数"""
        ctx = SessionContext(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash123",
        )
        assert ctx.query_count == 0

        ctx.increment_query()
        assert ctx.query_count == 1

        ctx.increment_query()
        assert ctx.query_count == 2

    def test_idle_seconds(self):
        """测试空闲时间"""
        ctx = SessionContext(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash123",
        )
        # 刚创建，空闲时间应该很小
        assert ctx.idle_seconds < 1

    def test_cleanup(self):
        """测试清理"""
        ctx = SessionContext(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash123",
        )

        # 添加 Timer
        ctx.timer = SessionTimer(session_id="sess_123")
        mock_task = MagicMock()
        mock_task.done.return_value = False
        ctx.timer.task = mock_task

        # 添加 Agent
        ctx.agent = MagicMock()

        ctx.cleanup()

        mock_task.cancel.assert_called_once()
        ctx.agent.clear_history.assert_called_once()

    def test_to_dict(self):
        """测试转换为字典"""
        ctx = SessionContext(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash123456789",
        )
        result = ctx.to_dict()

        assert result["session_id"] == "sess_123"
        assert result["tenant_id"] == "tenant_1"
        assert result["chatbot_id"] == "bot_1"
        assert result["config_hash"] == "hash12345678"  # 截断到 12 字符
        assert result["query_count"] == 0
        assert result["has_timer"] is False


# =============================================================================
# SessionManager Tests
# =============================================================================


class TestSessionManager:
    """SessionManager 单元测试"""

    @pytest.fixture
    def config_cache(self):
        """配置缓存 fixture"""
        return ConfigCache()

    @pytest.fixture
    def mock_repos(self):
        """Mock Repository fixture"""
        repos = MagicMock()
        repos.messages = AsyncMock()
        repos.messages.list_by_session = AsyncMock(return_value=[])
        repos.messages.create = AsyncMock()
        repos.sessions = AsyncMock()
        repos.sessions.allocate_event_offset = AsyncMock(return_value=0)
        repos.events = AsyncMock()
        repos.events.create = AsyncMock()
        return repos

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM fixture"""
        llm = MagicMock()
        return llm

    @pytest.fixture
    def session_manager(self, config_cache, mock_repos, mock_llm):
        """SessionManager fixture"""
        return SessionManager(
            config_cache=config_cache,
            repos=mock_repos,
            llm_provider=mock_llm,
            idle_timeout=60,
            cleanup_interval=10,
        )

    def test_init(self, session_manager):
        """测试初始化"""
        assert session_manager.session_count == 0
        assert session_manager._running is False

    @pytest.mark.asyncio
    async def test_start_stop(self, session_manager):
        """测试启动和停止"""
        await session_manager.start()
        assert session_manager._running is True

        await session_manager.stop()
        assert session_manager._running is False

    @pytest.mark.asyncio
    async def test_get_or_create_new_session(self, session_manager, config_cache):
        """测试创建新会话"""
        # 准备配置
        mock_config = MagicMock()
        mock_config.timers = None
        config_cache.set("hash123", mock_config)

        # Mock Agent 创建
        with patch.object(session_manager, '_create_agent', new_callable=AsyncMock) as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            ctx = await session_manager.get_or_create(
                session_id="sess_123",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash123",
            )

            assert ctx.session_id == "sess_123"
            assert ctx.tenant_id == "tenant_1"
            assert ctx.chatbot_id == "bot_1"
            assert ctx.config_hash == "hash123"
            assert session_manager.session_count == 1

    @pytest.mark.asyncio
    async def test_get_or_create_existing_session(self, session_manager, config_cache):
        """测试获取现有会话"""
        mock_config = MagicMock()
        mock_config.timers = None
        config_cache.set("hash123", mock_config)

        with patch.object(session_manager, '_create_agent', new_callable=AsyncMock) as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            # 第一次创建
            ctx1 = await session_manager.get_or_create(
                session_id="sess_123",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash123",
            )

            # 第二次获取（应该返回同一个）
            ctx2 = await session_manager.get_or_create(
                session_id="sess_123",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash123",
            )

            assert ctx1 is ctx2
            assert session_manager.session_count == 1
            # Agent 只创建一次
            assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_config_change_recreates_session(self, session_manager, config_cache):
        """测试配置变更时重建会话"""
        mock_config1 = MagicMock()
        mock_config1.timers = None
        mock_config2 = MagicMock()
        mock_config2.timers = None

        config_cache.set("hash123", mock_config1)
        config_cache.set("hash456", mock_config2)

        with patch.object(session_manager, '_create_agent', new_callable=AsyncMock) as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            # 第一次创建
            ctx1 = await session_manager.get_or_create(
                session_id="sess_123",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash123",
            )

            # 配置变更
            ctx2 = await session_manager.get_or_create(
                session_id="sess_123",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash456",
            )

            assert ctx1 is not ctx2
            assert ctx2.config_hash == "hash456"
            # Agent 创建两次
            assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_destroy_session(self, session_manager, config_cache):
        """测试销毁会话"""
        mock_config = MagicMock()
        mock_config.timers = None
        config_cache.set("hash123", mock_config)

        with patch.object(session_manager, '_create_agent', new_callable=AsyncMock) as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            await session_manager.get_or_create(
                session_id="sess_123",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash123",
            )
            assert session_manager.session_count == 1

            await session_manager.destroy("sess_123")
            assert session_manager.session_count == 0

    def test_exists(self, session_manager):
        """测试检查会话是否存在"""
        assert session_manager.exists("sess_123") is False

    def test_get_nonexistent(self, session_manager):
        """测试获取不存在的会话"""
        result = session_manager.get("nonexistent")
        assert result is None

    def test_reset_timer_nonexistent(self, session_manager):
        """测试重置不存在会话的 Timer"""
        # 不应该抛出异常
        session_manager.reset_timer("nonexistent")

    def test_get_stats(self, session_manager):
        """测试获取统计"""
        stats = session_manager.get_stats()
        assert stats["session_count"] == 0
        assert "max_sessions" in stats
        assert "idle_timeout" in stats

    def test_list_sessions_empty(self, session_manager):
        """测试列出空会话列表"""
        sessions = session_manager.list_sessions()
        assert sessions == []

    def test_get_session_info_nonexistent(self, session_manager):
        """测试获取不存在会话的信息"""
        info = session_manager.get_session_info("nonexistent")
        assert info is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSessionManagerIntegration:
    """SessionManager 集成测试"""

    @pytest.mark.asyncio
    async def test_timer_trigger(self):
        """测试 Timer 触发"""
        config_cache = ConfigCache()
        mock_repos = MagicMock()
        mock_repos.messages = AsyncMock()
        mock_repos.messages.list_by_session = AsyncMock(return_value=[])
        mock_repos.messages.create = AsyncMock()

        session_manager = SessionManager(
            config_cache=config_cache,
            repos=mock_repos,
            llm_provider=MagicMock(),
        )

        # 配置带 Timer
        mock_config = MagicMock()
        mock_config.timers = [
            {
                "delay_seconds": 0.1,  # 100ms
                "message": "Test message",
                "max_triggers": 2,
            }
        ]
        config_cache.set("hash123", mock_config)

        # 记录发送的消息
        sent_messages = []

        async def mock_send(session_id, message):
            sent_messages.append((session_id, message))

        session_manager.set_send_message_callback(mock_send)

        with patch.object(session_manager, '_create_agent', new_callable=AsyncMock) as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            ctx = await session_manager.get_or_create(
                session_id="sess_123",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash123",
            )

            # 等待 Timer 触发
            await asyncio.sleep(0.3)

            # 应该触发了消息
            assert len(sent_messages) >= 1
            assert sent_messages[0] == ("sess_123", "Test message")

            # 清理
            await session_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
