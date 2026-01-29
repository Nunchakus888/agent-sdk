"""
Unit tests for unified WorkflowEngine.

Tests:
- In-memory mode
- Agent caching
- Session management
- Config loading
- Cleanup task
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from bu_agent_sdk.workflow.engine import (
    WorkflowEngine,
    InMemoryConfigStore,
    InMemorySessionStore,
    CachedAgent,
    compute_config_hash,
)
from bu_agent_sdk.agent.workflow_state import Session, WorkflowState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
    return llm


@pytest.fixture
def llm_factory(mock_llm):
    """Create LLM factory."""
    return lambda: mock_llm


@pytest.fixture
def sample_config():
    """Sample workflow config."""
    return {
        "instructions": "You are a helpful assistant.",
        "tools": [],
        "flows": [],
    }


@pytest_asyncio.fixture
async def engine(llm_factory):
    """Create WorkflowEngine instance."""
    engine = WorkflowEngine(
        llm_factory=llm_factory,
        max_agents=10,
        agent_ttl=60,
        idle_timeout=30,
        cleanup_interval=5,
    )
    # Don't start cleanup for tests
    yield engine
    await engine.shutdown()


# =============================================================================
# Utility Tests
# =============================================================================


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_same_config_same_hash(self, sample_config):
        """Same config should produce same hash."""
        hash1 = compute_config_hash(sample_config)
        hash2 = compute_config_hash(sample_config)
        assert hash1 == hash2

    def test_different_config_different_hash(self, sample_config):
        """Different config should produce different hash."""
        hash1 = compute_config_hash(sample_config)
        modified = {**sample_config, "instructions": "Different"}
        hash2 = compute_config_hash(modified)
        assert hash1 != hash2

    def test_key_order_independent(self):
        """Hash should be independent of key order."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        assert compute_config_hash(config1) == compute_config_hash(config2)


# =============================================================================
# InMemoryConfigStore Tests
# =============================================================================


class TestInMemoryConfigStore:
    """Tests for InMemoryConfigStore."""

    @pytest.mark.asyncio
    async def test_save_and_get_by_hash(self, sample_config):
        """Test save and get_by_hash."""
        store = InMemoryConfigStore()
        config_hash = await store.save(sample_config, "tenant1", "bot1")

        result = await store.get_by_hash(config_hash)
        assert result == sample_config

    @pytest.mark.asyncio
    async def test_get_by_chatbot(self, sample_config):
        """Test get_by_chatbot."""
        store = InMemoryConfigStore()
        await store.save(sample_config, "tenant1", "bot1")

        result = await store.get_by_chatbot("bot1")
        assert result == sample_config

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test get nonexistent config."""
        store = InMemoryConfigStore()
        result = await store.get_by_hash("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sample_config):
        """Test delete config."""
        store = InMemoryConfigStore()
        config_hash = await store.save(sample_config, "tenant1", "bot1")

        await store.delete(config_hash)
        result = await store.get_by_hash(config_hash)
        assert result is None


# =============================================================================
# InMemorySessionStore Tests
# =============================================================================


class TestInMemorySessionStore:
    """Tests for InMemorySessionStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        """Test save and get session."""
        store = InMemorySessionStore()
        session = Session(
            session_id="sess1",
            agent_id="agent1",
            workflow_state=WorkflowState(config_hash="hash1"),
            messages=[],
        )

        await store.save(session)
        result = await store.get("sess1")

        assert result is not None
        assert result.session_id == "sess1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test get nonexistent session."""
        store = InMemorySessionStore()
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete session."""
        store = InMemorySessionStore()
        session = Session(
            session_id="sess1",
            agent_id="agent1",
            workflow_state=WorkflowState(config_hash="hash1"),
            messages=[],
        )

        await store.save(session)
        await store.delete("sess1")
        result = await store.get("sess1")
        assert result is None


# =============================================================================
# CachedAgent Tests
# =============================================================================


class TestCachedAgent:
    """Tests for CachedAgent."""

    def test_touch_updates_last_access(self, mock_llm):
        """Test touch updates last_access."""
        agent = MagicMock()
        cached = CachedAgent(agent, "hash1")
        initial_time = cached.last_access

        import time
        time.sleep(0.01)
        cached.touch()

        assert cached.last_access > initial_time

    def test_is_idle_no_sessions(self, mock_llm):
        """Test is_idle with no sessions."""
        agent = MagicMock()
        cached = CachedAgent(agent, "hash1")
        assert cached.is_idle is True

    def test_is_idle_with_sessions(self, mock_llm):
        """Test is_idle with sessions."""
        agent = MagicMock()
        cached = CachedAgent(agent, "hash1")
        cached.session_ids.add("sess1")
        assert cached.is_idle is False


# =============================================================================
# WorkflowEngine Tests
# =============================================================================


class TestWorkflowEngine:
    """Tests for WorkflowEngine."""

    @pytest.mark.asyncio
    async def test_register_config(self, engine, sample_config):
        """Test register_config."""
        config_hash = await engine.register_config(
            sample_config, "tenant1", "bot1"
        )
        assert config_hash is not None
        assert len(config_hash) == 32  # MD5 hash length

    @pytest.mark.asyncio
    async def test_get_stats(self, engine):
        """Test get_stats."""
        stats = engine.get_stats()

        assert "mode" in stats
        assert stats["mode"] == "memory"
        assert "cached_agents" in stats
        assert "max_agents" in stats
        assert "uptime_seconds" in stats

    @pytest.mark.asyncio
    async def test_session_management(self, engine):
        """Test session management."""
        # Create session
        session = Session(
            session_id="sess1",
            agent_id="agent1",
            workflow_state=WorkflowState(config_hash="hash1"),
            messages=[],
        )
        await engine.session_store.save(session)

        # Get session
        result = await engine.get_session("sess1")
        assert result is not None
        assert result.session_id == "sess1"

        # Delete session
        await engine.delete_session("sess1")
        result = await engine.get_session("sess1")
        assert result is None

    @pytest.mark.asyncio
    async def test_release_session(self, engine, sample_config):
        """Test release_session."""
        config_hash = await engine.register_config(
            sample_config, "tenant1", "bot1"
        )

        # Manually add cached agent
        agent = MagicMock()
        cached = CachedAgent(agent, config_hash)
        cached.session_ids.add("sess1")
        engine._agents[config_hash] = cached

        # Release session
        await engine.release_session("sess1", config_hash)

        assert "sess1" not in engine._agents[config_hash].session_ids

    @pytest.mark.asyncio
    async def test_evict_agents_ttl(self, llm_factory):
        """Test agent eviction by TTL."""
        engine = WorkflowEngine(
            llm_factory=llm_factory,
            max_agents=10,
            agent_ttl=0,  # Immediate expiration
        )

        # Add agent
        agent = MagicMock()
        cached = CachedAgent(agent, "hash1")
        engine._agents["hash1"] = cached

        # Evict
        await engine._evict_agents()

        assert "hash1" not in engine._agents
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_evict_agents_lru(self, llm_factory):
        """Test agent eviction by LRU."""
        engine = WorkflowEngine(
            llm_factory=llm_factory,
            max_agents=2,
            agent_ttl=3600,
        )

        # Add 3 agents (over limit)
        for i in range(3):
            agent = MagicMock()
            cached = CachedAgent(agent, f"hash{i}")
            engine._agents[f"hash{i}"] = cached
            await asyncio.sleep(0.01)  # Ensure different timestamps

        # Evict
        await engine._evict_agents()

        # Should have max_agents - 1 = 1 agent left
        assert len(engine._agents) <= 2
        await engine.shutdown()


# =============================================================================
# Query API Tests (with mocked WorkflowAgent)
# =============================================================================


class TestQueryAPI:
    """Tests for query API."""

    @pytest.mark.asyncio
    async def test_query_no_config(self, engine):
        """Test query with no config returns error."""
        response = await engine.query(
            session_id="sess1",
            chatbot_id="nonexistent",
            user_message="Hello",
        )
        assert "Error" in response

    @pytest.mark.asyncio
    async def test_query_with_inline_config(self, engine, sample_config):
        """Test query with inline config."""
        with patch(
            "bu_agent_sdk.workflow.engine.WorkflowAgent"
        ) as MockAgent:
            mock_agent = MagicMock()
            mock_agent.query = AsyncMock(return_value="Hello!")
            MockAgent.return_value = mock_agent

            response = await engine.query(
                session_id="sess1",
                chatbot_id="bot1",
                user_message="Hello",
                config=sample_config,
            )

            assert response == "Hello!"
            mock_agent.query.assert_called_once_with("Hello", "sess1")

    @pytest.mark.asyncio
    async def test_query_reuses_cached_agent(self, engine, sample_config):
        """Test query reuses cached agent."""
        with patch(
            "bu_agent_sdk.workflow.engine.WorkflowAgent"
        ) as MockAgent:
            mock_agent = MagicMock()
            mock_agent.query = AsyncMock(return_value="Response")
            MockAgent.return_value = mock_agent

            # First query
            await engine.query(
                session_id="sess1",
                chatbot_id="bot1",
                user_message="Hello",
                config=sample_config,
            )

            # Second query with same config
            await engine.query(
                session_id="sess2",
                chatbot_id="bot1",
                user_message="Hi",
                config=sample_config,
            )

            # Agent should be created only once
            assert MockAgent.call_count == 1

    @pytest.mark.asyncio
    async def test_query_tracks_sessions(self, engine, sample_config):
        """Test query tracks sessions in cached agent."""
        with patch(
            "bu_agent_sdk.workflow.engine.WorkflowAgent"
        ) as MockAgent:
            mock_agent = MagicMock()
            mock_agent.query = AsyncMock(return_value="Response")
            MockAgent.return_value = mock_agent

            await engine.query(
                session_id="sess1",
                chatbot_id="bot1",
                user_message="Hello",
                config=sample_config,
            )

            config_hash = compute_config_hash(sample_config)
            cached = engine._agents.get(config_hash)

            assert cached is not None
            assert "sess1" in cached.session_ids
