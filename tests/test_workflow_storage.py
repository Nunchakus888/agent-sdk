"""
Unit tests for workflow storage implementations.

Tests cover:
- MongoDBConfigStore
- MongoDBSessionStore
- PostgreSQLSessionStore
- RedisPlanCache
- ExecutionHistoryStore
- WorkflowEngine (complete flow)
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from bu_agent_sdk.workflow.storage import (
    MongoDBConfigStore,
    MongoDBSessionStore,
    PostgreSQLSessionStore,
    RedisPlanCache,
    ExecutionHistoryStore,
    WorkflowEngine,
)
from bu_agent_sdk.agent.workflow_state import Session, WorkflowState
from bu_agent_sdk.workflow.cache import CachedPlan


# =============================================================================
# Test MongoDBConfigStore
# =============================================================================


@pytest.fixture
def mock_mongo_client_for_config():
    """Create mock MongoDB client for config store."""
    client = Mock()
    db = Mock()
    collection = AsyncMock()

    client.__getitem__ = Mock(return_value=db)
    db.configs = collection

    return client


@pytest.mark.asyncio
async def test_config_store_save(mock_mongo_client_for_config):
    """Test config store save."""
    store = MongoDBConfigStore(mock_mongo_client_for_config)

    # Mock find_one to return None (config doesn't exist)
    store.collection.find_one.return_value = None

    config = {"basic_settings": {"name": "Test Bot"}, "skills": []}
    config_hash = await store.save(config, "tenant_001", "bot_001")

    assert config_hash is not None
    assert len(config_hash) == 32  # MD5 hash length
    store.collection.insert_one.assert_called_once()


@pytest.mark.asyncio
async def test_config_store_save_existing(mock_mongo_client_for_config):
    """Test config store save when config already exists."""
    store = MongoDBConfigStore(mock_mongo_client_for_config)

    # Mock find_one to return existing config
    store.collection.find_one.return_value = {"config_hash": "abc123"}

    config = {"basic_settings": {"name": "Test Bot"}}
    config_hash = await store.save(config, "tenant_001", "bot_001")

    # Should not insert if already exists
    store.collection.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_config_store_get_by_hash(mock_mongo_client_for_config):
    """Test config store get by hash."""
    store = MongoDBConfigStore(mock_mongo_client_for_config)

    store.collection.find_one.return_value = {
        "config_hash": "abc123",
        "config_data": {"basic_settings": {"name": "Test Bot"}}
    }

    config = await store.get_by_hash("abc123")

    assert config is not None
    assert config["basic_settings"]["name"] == "Test Bot"
    store.collection.find_one.assert_called_once_with({"config_hash": "abc123"})


@pytest.mark.asyncio
async def test_config_store_get_by_chatbot(mock_mongo_client_for_config):
    """Test config store get by chatbot."""
    store = MongoDBConfigStore(mock_mongo_client_for_config)

    store.collection.find_one.return_value = {
        "chatbot_id": "bot_001",
        "config_data": {"basic_settings": {"name": "Test Bot"}}
    }

    config = await store.get_by_chatbot("bot_001")

    assert config is not None
    store.collection.find_one.assert_called_once()


# =============================================================================
# Test MongoDBSessionStore
# =============================================================================


@pytest.fixture
def mock_mongo_client():
    """Create mock MongoDB client."""
    client = Mock()
    db = Mock()
    collection = AsyncMock()

    # Configure __getitem__ to return db
    client.__getitem__ = Mock(return_value=db)
    db.sessions = collection

    return client


@pytest.fixture
def test_session():
    """Create test session."""
    return Session(
        session_id="test_session_123",
        agent_id="agent_456",
        workflow_state=WorkflowState(
            config_hash="abc123",
            need_greeting=False,
            status="active",
            metadata={"key": "value"}
        ),
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
    )


@pytest.mark.asyncio
async def test_mongodb_save_session(mock_mongo_client, test_session):
    """Test MongoDB save session."""
    store = MongoDBSessionStore(mock_mongo_client)

    await store.save(test_session)

    # Verify update_one was called
    store.collection.update_one.assert_called_once()
    call_args = store.collection.update_one.call_args

    # Check filter
    assert call_args[0][0] == {"session_id": "test_session_123"}

    # Check upsert
    assert call_args[1]["upsert"] is True


@pytest.mark.asyncio
async def test_mongodb_get_session(mock_mongo_client, test_session):
    """Test MongoDB get session."""
    store = MongoDBSessionStore(mock_mongo_client)

    # Mock find_one response
    store.collection.find_one.return_value = {
        "session_id": "test_session_123",
        "agent_id": "agent_456",
        "workflow_state": {
            "config_hash": "abc123",
            "need_greeting": False,
            "status": "active",
            "metadata": {"key": "value"},
            "last_updated": datetime.utcnow().isoformat()
        },
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }

    session = await store.get("test_session_123")

    assert session is not None
    assert session.session_id == "test_session_123"
    assert session.agent_id == "agent_456"
    assert session.workflow_state.config_hash == "abc123"
    store.collection.find_one.assert_called_once_with({"session_id": "test_session_123"})


@pytest.mark.asyncio
async def test_mongodb_get_session_not_found(mock_mongo_client):
    """Test MongoDB get session when not found."""
    store = MongoDBSessionStore(mock_mongo_client)
    store.collection.find_one.return_value = None

    session = await store.get("non_existent")

    assert session is None


@pytest.mark.asyncio
async def test_mongodb_delete_session(mock_mongo_client):
    """Test MongoDB delete session."""
    store = MongoDBSessionStore(mock_mongo_client)

    await store.delete("test_session_123")

    store.collection.delete_one.assert_called_once_with({"session_id": "test_session_123"})


@pytest.mark.asyncio
async def test_mongodb_list_by_agent(mock_mongo_client):
    """Test MongoDB list sessions by agent."""
    store = MongoDBSessionStore(mock_mongo_client)

    # Mock cursor
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = [
        {
            "session_id": "session_1",
            "agent_id": "agent_456",
            "workflow_state": {
                "config_hash": "abc",
                "need_greeting": True,
                "status": "ready",
                "metadata": {},
                "last_updated": datetime.utcnow().isoformat()
            },
            "messages": []
        },
        {
            "session_id": "session_2",
            "agent_id": "agent_456",
            "workflow_state": {
                "config_hash": "def",
                "need_greeting": False,
                "status": "active",
                "metadata": {},
                "last_updated": datetime.utcnow().isoformat()
            },
            "messages": []
        }
    ]

    # Mock find().limit() chain
    mock_cursor.limit.return_value = mock_cursor
    store.collection.find.return_value = mock_cursor

    sessions = await store.list_by_agent("agent_456", limit=10)

    assert len(sessions) == 2
    assert sessions[0].session_id == "session_1"
    assert sessions[1].session_id == "session_2"
    store.collection.find.assert_called_once_with({"agent_id": "agent_456"})


# =============================================================================
# Test PostgreSQLSessionStore
# =============================================================================


@pytest.fixture
def mock_pg_pool():
    """Create mock PostgreSQL pool."""
    pool = AsyncMock()
    conn = AsyncMock()

    # Configure async context manager
    async_context = AsyncMock()
    async_context.__aenter__ = AsyncMock(return_value=conn)
    async_context.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = async_context

    return pool


@pytest.mark.asyncio
async def test_postgresql_init_schema(mock_pg_pool):
    """Test PostgreSQL schema initialization."""
    store = PostgreSQLSessionStore(mock_pg_pool)

    await store.init_schema()

    # Verify execute was called for table creation and index
    async_context = mock_pg_pool.acquire.return_value
    conn = await async_context.__aenter__()
    assert conn.execute.call_count >= 2


@pytest.mark.asyncio
async def test_postgresql_save_session(mock_pg_pool, test_session):
    """Test PostgreSQL save session."""
    store = PostgreSQLSessionStore(mock_pg_pool)

    await store.save(test_session)

    # Verify execute was called with INSERT ... ON CONFLICT
    async_context = mock_pg_pool.acquire.return_value
    conn = await async_context.__aenter__()
    conn.execute.assert_called_once()

    call_args = conn.execute.call_args[0]
    assert "INSERT INTO sessions" in call_args[0]
    assert "ON CONFLICT" in call_args[0]
    assert call_args[1] == "test_session_123"
    assert call_args[2] == "agent_456"


@pytest.mark.asyncio
async def test_postgresql_get_session(mock_pg_pool):
    """Test PostgreSQL get session."""
    store = PostgreSQLSessionStore(mock_pg_pool)

    # Mock fetchrow response
    async_context = mock_pg_pool.acquire.return_value
    conn = await async_context.__aenter__()
    conn.fetchrow.return_value = {
        "session_id": "test_session_123",
        "agent_id": "agent_456",
        "config_hash": "abc123",
        "need_greeting": False,
        "status": "active",
        "metadata": {"key": "value"},
        "messages": [{"role": "user", "content": "Hello"}],
        "updated_at": datetime.utcnow()
    }

    session = await store.get("test_session_123")

    assert session is not None
    assert session.session_id == "test_session_123"
    assert session.agent_id == "agent_456"
    conn.fetchrow.assert_called_once()


@pytest.mark.asyncio
async def test_postgresql_get_session_not_found(mock_pg_pool):
    """Test PostgreSQL get session when not found."""
    store = PostgreSQLSessionStore(mock_pg_pool)

    async_context = mock_pg_pool.acquire.return_value
    conn = await async_context.__aenter__()
    conn.fetchrow.return_value = None

    session = await store.get("non_existent")

    assert session is None


@pytest.mark.asyncio
async def test_postgresql_delete_session(mock_pg_pool):
    """Test PostgreSQL delete session."""
    store = PostgreSQLSessionStore(mock_pg_pool)

    await store.delete("test_session_123")

    async_context = mock_pg_pool.acquire.return_value
    conn = await async_context.__aenter__()
    conn.execute.assert_called_once()

    call_args = conn.execute.call_args[0]
    assert "DELETE FROM sessions" in call_args[0]
    assert call_args[1] == "test_session_123"


@pytest.mark.asyncio
async def test_postgresql_list_by_agent(mock_pg_pool):
    """Test PostgreSQL list sessions by agent."""
    store = PostgreSQLSessionStore(mock_pg_pool)

    # Mock fetch response
    async_context = mock_pg_pool.acquire.return_value
    conn = await async_context.__aenter__()
    conn.fetch.return_value = [
        {
            "session_id": "session_1",
            "agent_id": "agent_456",
            "config_hash": "abc",
            "need_greeting": True,
            "status": "ready",
            "metadata": {},
            "messages": [],
            "updated_at": datetime.utcnow()
        },
        {
            "session_id": "session_2",
            "agent_id": "agent_456",
            "config_hash": "def",
            "need_greeting": False,
            "status": "active",
            "metadata": {},
            "messages": [],
            "updated_at": datetime.utcnow()
        }
    ]

    sessions = await store.list_by_agent("agent_456", limit=10)

    assert len(sessions) == 2
    assert sessions[0].session_id == "session_1"
    assert sessions[1].session_id == "session_2"
    conn.fetch.assert_called_once()


# =============================================================================
# Test RedisPlanCache
# =============================================================================


@pytest.fixture
def mock_redis_client():
    """Create mock Redis client."""
    return AsyncMock()


@pytest.fixture
def test_cached_plan():
    """Create test cached plan."""
    return CachedPlan(
        workflow_id="workflow_123",
        config_hash="abc123",
        tool_names=["tool1", "tool2"],
        skill_ids=["skill1"],
        flow_ids=["flow1"],
        created_at=datetime.utcnow()
    )


@pytest.mark.asyncio
async def test_redis_set_plan(mock_redis_client, test_cached_plan):
    """Test Redis set plan."""
    cache = RedisPlanCache(mock_redis_client, ttl=3600)

    await cache.set(test_cached_plan)

    # Verify set was called with correct key and TTL
    mock_redis_client.set.assert_called_once()
    call_args = mock_redis_client.set.call_args

    assert call_args[0][0] == "workflow:plan:workflow_123:abc123"
    assert call_args[1]["ex"] == 3600

    # Verify the data is JSON serializable
    import json
    data = call_args[0][1]
    parsed = json.loads(data)
    assert parsed["workflow_id"] == "workflow_123"


@pytest.mark.asyncio
async def test_redis_get_plan(mock_redis_client, test_cached_plan):
    """Test Redis get plan."""
    cache = RedisPlanCache(mock_redis_client)

    # Mock get response
    import json
    mock_redis_client.get.return_value = json.dumps({
        "workflow_id": "workflow_123",
        "config_hash": "abc123",
        "tool_names": ["tool1", "tool2"],
        "skill_ids": ["skill1"],
        "flow_ids": ["flow1"],
        "created_at": datetime.utcnow().isoformat()
    })

    plan = await cache.get("workflow_123", "abc123")

    assert plan is not None
    assert plan.workflow_id == "workflow_123"
    assert plan.config_hash == "abc123"
    assert len(plan.tool_names) == 2
    mock_redis_client.get.assert_called_once_with("workflow:plan:workflow_123:abc123")


@pytest.mark.asyncio
async def test_redis_get_plan_not_found(mock_redis_client):
    """Test Redis get plan when not found."""
    cache = RedisPlanCache(mock_redis_client)
    mock_redis_client.get.return_value = None

    plan = await cache.get("workflow_123", "abc123")

    assert plan is None


@pytest.mark.asyncio
async def test_redis_delete_plan(mock_redis_client):
    """Test Redis delete plan."""
    cache = RedisPlanCache(mock_redis_client)

    await cache.delete("workflow_123", "abc123")

    mock_redis_client.delete.assert_called_once_with("workflow:plan:workflow_123:abc123")


@pytest.mark.asyncio
async def test_redis_clear_all(mock_redis_client):
    """Test Redis clear all plans for workflow."""
    cache = RedisPlanCache(mock_redis_client)

    # Mock scan responses (simulate pagination)
    mock_redis_client.scan.side_effect = [
        (10, [b"workflow:plan:workflow_123:hash1", b"workflow:plan:workflow_123:hash2"]),
        (0, [b"workflow:plan:workflow_123:hash3"])  # cursor=0 means done
    ]

    await cache.clear_all("workflow_123")

    # Verify scan was called with correct pattern
    assert mock_redis_client.scan.call_count == 2

    # Verify delete was called for all keys
    assert mock_redis_client.delete.call_count == 2


# =============================================================================
# Test ExecutionHistoryStore
# =============================================================================


@pytest.mark.asyncio
async def test_execution_history_log(mock_mongo_client):
    """Test execution history logging."""
    store = ExecutionHistoryStore(mock_mongo_client)

    # Mock decision object
    decision = Mock()
    decision.should_continue = True
    decision.should_respond = False
    decision.next_action = {"type": "skill", "target": "test_skill"}
    decision.reasoning = "User needs help"

    await store.log_execution(
        session_id="session_123",
        agent_id="agent_456",
        user_message="Help me",
        decision=decision,
        result="Success",
        metadata={"duration": 1.5}
    )

    # Verify insert_one was called
    store.collection.insert_one.assert_called_once()

    call_args = store.collection.insert_one.call_args[0][0]
    assert call_args["session_id"] == "session_123"
    assert call_args["agent_id"] == "agent_456"
    assert call_args["user_message"] == "Help me"
    assert call_args["result"] == "Success"
    assert call_args["metadata"]["duration"] == 1.5


@pytest.mark.asyncio
async def test_execution_history_get_session_history(mock_mongo_client):
    """Test get session history."""
    store = ExecutionHistoryStore(mock_mongo_client)

    # Mock cursor
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = [
        {
            "session_id": "session_123",
            "user_message": "Message 1",
            "result": "Success",
            "timestamp": datetime.utcnow()
        },
        {
            "session_id": "session_123",
            "user_message": "Message 2",
            "result": "Success",
            "timestamp": datetime.utcnow()
        }
    ]

    store.collection.find.return_value = mock_cursor
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor

    history = await store.get_session_history("session_123", limit=10)

    assert len(history) == 2
    assert history[0]["user_message"] == "Message 1"
    store.collection.find.assert_called_once_with({"session_id": "session_123"})


@pytest.mark.asyncio
async def test_execution_history_get_agent_stats(mock_mongo_client):
    """Test get agent statistics."""
    store = ExecutionHistoryStore(mock_mongo_client)

    # Mock aggregate response
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = [
        {
            "_id": None,
            "total_executions": 100,
            "avg_iterations": 2.5,
            "success_rate": 0.95
        }
    ]

    store.collection.aggregate.return_value = mock_cursor

    stats = await store.get_agent_stats("agent_456")

    assert stats["total_executions"] == 100
    assert stats["avg_iterations"] == 2.5
    assert stats["success_rate"] == 0.95
    store.collection.aggregate.assert_called_once()


@pytest.mark.asyncio
async def test_execution_history_get_agent_stats_no_data(mock_mongo_client):
    """Test get agent statistics when no data."""
    store = ExecutionHistoryStore(mock_mongo_client)

    # Mock empty aggregate response
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = []

    store.collection.aggregate.return_value = mock_cursor

    stats = await store.get_agent_stats("agent_456")

    assert stats == {}


# =============================================================================
# Test WorkflowEngine (Complete Flow)
# =============================================================================


@pytest.fixture
def mock_mongo_client_for_engine():
    """Create mock MongoDB client for WorkflowEngine."""
    client = Mock()
    db = Mock()

    # Config collection
    configs_collection = AsyncMock()
    # Session collection
    sessions_collection = AsyncMock()
    # History collection
    history_collection = AsyncMock()

    client.__getitem__ = Mock(return_value=db)
    db.configs = configs_collection
    db.sessions = sessions_collection
    db.execution_history = history_collection

    return client


@pytest.fixture
def mock_llm_factory():
    """Create mock LLM factory."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = Mock(content='{"should_respond": true}')
    return lambda: mock_llm


@pytest.mark.asyncio
async def test_workflow_engine_init(mock_mongo_client_for_engine, mock_llm_factory):
    """Test WorkflowEngine initialization."""
    engine = WorkflowEngine(
        mongo_client=mock_mongo_client_for_engine,
        llm_factory=mock_llm_factory,
        max_agents=50,
        agent_ttl_seconds=3600
    )

    assert engine._max_agents == 50
    assert engine._agent_ttl == 3600
    assert len(engine._agent_cache) == 0


@pytest.mark.asyncio
async def test_workflow_engine_register_config(mock_mongo_client_for_engine, mock_llm_factory):
    """Test WorkflowEngine register config."""
    engine = WorkflowEngine(
        mongo_client=mock_mongo_client_for_engine,
        llm_factory=mock_llm_factory
    )

    # Mock config store
    engine.config_store.collection.find_one.return_value = None

    config = {"basic_settings": {"name": "Test Bot"}, "skills": []}
    config_hash = await engine.register_config(
        config=config,
        tenant_id="tenant_001",
        chatbot_id="bot_001"
    )

    assert config_hash is not None
    assert len(config_hash) == 32


@pytest.mark.asyncio
async def test_workflow_engine_get_cache_stats(mock_mongo_client_for_engine, mock_llm_factory):
    """Test WorkflowEngine cache stats."""
    engine = WorkflowEngine(
        mongo_client=mock_mongo_client_for_engine,
        llm_factory=mock_llm_factory,
        max_agents=100,
        agent_ttl_seconds=86400
    )

    stats = engine.get_cache_stats()

    assert stats["cached_agents"] == 0
    assert stats["max_agents"] == 100
    assert stats["agent_ttl_seconds"] == 86400


@pytest.mark.asyncio
async def test_workflow_engine_agent_cache_eviction(mock_mongo_client_for_engine, mock_llm_factory):
    """Test WorkflowEngine agent cache eviction."""
    engine = WorkflowEngine(
        mongo_client=mock_mongo_client_for_engine,
        llm_factory=mock_llm_factory,
        max_agents=2,
        agent_ttl_seconds=86400
    )

    # Manually add agents to cache
    from datetime import timedelta
    now = datetime.utcnow()

    engine._agent_cache["hash1"] = (Mock(), now - timedelta(hours=2))
    engine._agent_cache["hash2"] = (Mock(), now - timedelta(hours=1))
    engine._agent_cache["hash3"] = (Mock(), now)

    # Trigger eviction
    await engine._evict_agents()

    # Should have evicted oldest agent (hash1)
    assert len(engine._agent_cache) <= 2


@pytest.mark.asyncio
async def test_workflow_engine_agent_cache_ttl_eviction(mock_mongo_client_for_engine, mock_llm_factory):
    """Test WorkflowEngine agent cache TTL eviction."""
    engine = WorkflowEngine(
        mongo_client=mock_mongo_client_for_engine,
        llm_factory=mock_llm_factory,
        max_agents=100,
        agent_ttl_seconds=3600  # 1 hour
    )

    # Manually add expired agent to cache
    from datetime import timedelta
    now = datetime.utcnow()

    engine._agent_cache["expired_hash"] = (Mock(), now - timedelta(hours=2))
    engine._agent_cache["valid_hash"] = (Mock(), now)

    # Trigger eviction
    await engine._evict_agents()

    # Expired agent should be removed
    assert "expired_hash" not in engine._agent_cache
    assert "valid_hash" in engine._agent_cache


@pytest.mark.asyncio
async def test_mongodb_session_store_list_by_chatbot(mock_mongo_client, test_session):
    """Test MongoDB list sessions by chatbot."""
    store = MongoDBSessionStore(mock_mongo_client)

    # Mock cursor
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = [
        {
            "session_id": "session_1",
            "agent_id": "agent_456",
            "chatbot_id": "bot_001",
            "workflow_state": {
                "config_hash": "abc",
                "need_greeting": True,
                "status": "ready",
                "metadata": {},
                "last_updated": datetime.utcnow().isoformat()
            },
            "messages": []
        }
    ]

    store.collection.find.return_value = mock_cursor
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor

    sessions = await store.list_by_chatbot("bot_001", limit=10)

    assert len(sessions) == 1
    assert sessions[0].session_id == "session_1"


# =============================================================================
# Integration Tests (Real MongoDB)
# =============================================================================
# Run with: MONGODB_URI=mongodb://localhost:27017 pytest tests/test_workflow_storage.py -k "integration" -v


import os
import pytest_asyncio

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
INTEGRATION_DB = "workflow_agent_test"


@pytest_asyncio.fixture
async def real_mongo_client():
    """Create real MongoDB client for integration tests."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
        # Test connection
        await client.admin.command("ping")
        yield client
        # Cleanup: drop test database
        await client.drop_database(INTEGRATION_DB)
        client.close()
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_config_store_full_flow(real_mongo_client):
    """Integration test: ConfigStore full CRUD flow."""
    store = MongoDBConfigStore(real_mongo_client, db_name=INTEGRATION_DB)

    # Ensure indexes
    await store.ensure_indexes()

    # Save config
    config = {
        "basic_settings": {"name": "Integration Test Bot", "language": "zh"},
        "skills": [{"skill_id": "s1", "name": "Test Skill"}],
        "tools": []
    }
    config_hash = await store.save(config, "tenant_test", "bot_test")

    assert config_hash is not None
    assert len(config_hash) == 32

    # Get by hash
    retrieved = await store.get_by_hash(config_hash)
    assert retrieved is not None
    assert retrieved["basic_settings"]["name"] == "Integration Test Bot"

    # Get by chatbot
    retrieved2 = await store.get_by_chatbot("bot_test")
    assert retrieved2 is not None
    assert retrieved2["basic_settings"]["name"] == "Integration Test Bot"

    # Save same config again (should return same hash)
    config_hash2 = await store.save(config, "tenant_test", "bot_test")
    assert config_hash2 == config_hash

    # Save different config (should create new version)
    config_v2 = {**config, "basic_settings": {"name": "Updated Bot"}}
    config_hash_v2 = await store.save(config_v2, "tenant_test", "bot_test")
    assert config_hash_v2 != config_hash

    # Get latest should return v2
    latest = await store.get_by_chatbot("bot_test")
    assert latest["basic_settings"]["name"] == "Updated Bot"

    # List by tenant
    configs = await store.list_by_tenant("tenant_test")
    assert len(configs) >= 2

    # Delete
    await store.delete(config_hash)
    deleted = await store.get_by_hash(config_hash)
    assert deleted is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_session_store_full_flow(real_mongo_client):
    """Integration test: SessionStore full CRUD flow."""
    store = MongoDBSessionStore(real_mongo_client, db_name=INTEGRATION_DB)

    # Ensure indexes
    await store.ensure_indexes()

    # Create session
    session = Session(
        session_id="int_session_001",
        agent_id="int_agent_001",
        workflow_state=WorkflowState(
            config_hash="int_hash_001",
            need_greeting=True,
            status="ready",
            metadata={"test": True}
        ),
        messages=[]
    )
    session.chatbot_id = "int_bot_001"

    # Save
    await store.save(session)

    # Get
    retrieved = await store.get("int_session_001")
    assert retrieved is not None
    assert retrieved.session_id == "int_session_001"
    assert retrieved.workflow_state.config_hash == "int_hash_001"
    assert retrieved.chatbot_id == "int_bot_001"

    # Update
    session.workflow_state.status = "processing"
    session.messages.append({"role": "user", "content": "Hello"})
    await store.save(session)

    updated = await store.get("int_session_001")
    assert updated.workflow_state.status == "processing"
    assert len(updated.messages) == 1

    # Create more sessions for list test
    for i in range(3):
        s = Session(
            session_id=f"int_session_00{i+2}",
            agent_id="int_agent_001",
            workflow_state=WorkflowState(config_hash="int_hash_001"),
            messages=[]
        )
        s.chatbot_id = "int_bot_001"
        await store.save(s)

    # List by chatbot
    sessions = await store.list_by_chatbot("int_bot_001", limit=10)
    assert len(sessions) >= 4

    # List by agent
    sessions2 = await store.list_by_agent("int_agent_001", limit=10)
    assert len(sessions2) >= 4

    # Delete
    await store.delete("int_session_001")
    deleted = await store.get("int_session_001")
    assert deleted is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_workflow_engine_full_flow(real_mongo_client):
    """Integration test: WorkflowEngine complete flow."""
    # Create mock LLM factory
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = Mock(content='{"should_respond": true}')
    llm_factory = lambda: mock_llm

    engine = WorkflowEngine(
        mongo_client=real_mongo_client,
        llm_factory=llm_factory,
        db_name=INTEGRATION_DB,
        max_agents=10,
        agent_ttl_seconds=3600
    )

    # Initialize
    await engine.init()

    # Register config
    config = {
        "basic_settings": {"name": "Engine Test Bot"},
        "instructions": "You are a helpful assistant.",
        "skills": [],
        "tools": [],
        "flows": []
    }
    config_hash = await engine.register_config(
        config=config,
        tenant_id="engine_tenant",
        chatbot_id="engine_bot"
    )
    assert config_hash is not None

    # Check cache stats
    stats = engine.get_cache_stats()
    assert stats["cached_agents"] == 0
    assert stats["max_agents"] == 10

    # Get session (should be None initially)
    session = await engine.get_session("engine_session_001")
    assert session is None

    # Delete session (should not error even if not exists)
    await engine.delete_session("engine_session_001")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_execution_history_store(real_mongo_client):
    """Integration test: ExecutionHistoryStore."""
    store = ExecutionHistoryStore(real_mongo_client, db_name=INTEGRATION_DB)

    # Create mock decision
    decision = Mock()
    decision.should_continue = True
    decision.should_respond = False
    decision.next_action = {"type": "skill", "target": "test_skill"}
    decision.reasoning = "Integration test"

    # Log execution
    await store.log_execution(
        session_id="hist_session_001",
        agent_id="hist_agent_001",
        user_message="Test message",
        decision=decision,
        result="Success",
        metadata={"test": True}
    )

    # Log more executions
    for i in range(5):
        await store.log_execution(
            session_id="hist_session_001",
            agent_id="hist_agent_001",
            user_message=f"Message {i}",
            decision=decision,
            result="Success"
        )

    # Get session history
    history = await store.get_session_history("hist_session_001", limit=10)
    assert len(history) >= 6

    # Get agent stats
    stats = await store.get_agent_stats("hist_agent_001")
    assert stats.get("total_executions", 0) >= 6


