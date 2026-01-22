"""
Unit tests for workflow storage implementations.

Tests cover:
- MongoDBSessionStore
- PostgreSQLSessionStore
- RedisPlanCache
- ExecutionHistoryStore
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from bu_agent_sdk.workflow.storage import (
    MongoDBSessionStore,
    PostgreSQLSessionStore,
    RedisPlanCache,
    ExecutionHistoryStore,
)
from bu_agent_sdk.agent.workflow_state import Session, WorkflowState
from bu_agent_sdk.workflow.cache import CachedPlan


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
