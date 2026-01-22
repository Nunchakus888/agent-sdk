"""
Database storage implementations for workflow agent.

Supports:
- Session storage (MongoDB, PostgreSQL, Redis)
- Plan cache storage
- Execution history tracking
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Protocol

from bu_agent_sdk.agent.workflow_state import Session, WorkflowState
from bu_agent_sdk.workflow.cache import CachedPlan


# =============================================================================
# Session Store Interface
# =============================================================================


class SessionStore(Protocol):
    """Session storage interface."""

    async def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        ...

    async def save(self, session: Session) -> None:
        """Save session."""
        ...

    async def delete(self, session_id: str) -> None:
        """Delete session."""
        ...

    async def list_by_agent(self, agent_id: str, limit: int = 100) -> list[Session]:
        """List sessions by agent ID."""
        ...


# =============================================================================
# MongoDB Implementation
# =============================================================================


class MongoDBSessionStore:
    """
    MongoDB session storage implementation.

    Usage:
        ```python
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient("mongodb://localhost:27017")
        store = MongoDBSessionStore(client, db_name="workflow_agent")

        # Use with WorkflowAgent
        agent = WorkflowAgent(config, llm, session_store=store)
        ```
    """

    def __init__(self, client: Any, db_name: str = "workflow_agent"):
        """
        Initialize MongoDB session store.

        Args:
            client: Motor AsyncIOMotorClient instance
            db_name: Database name
        """
        self.db = client[db_name]
        self.collection = self.db.sessions

    async def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        doc = await self.collection.find_one({"session_id": session_id})
        if not doc:
            return None

        return self._doc_to_session(doc)

    async def save(self, session: Session) -> None:
        """Save session."""
        doc = self._session_to_doc(session)

        await self.collection.update_one(
            {"session_id": session.session_id},
            {"$set": doc},
            upsert=True
        )

    async def delete(self, session_id: str) -> None:
        """Delete session."""
        await self.collection.delete_one({"session_id": session_id})

    async def list_by_agent(self, agent_id: str, limit: int = 100) -> list[Session]:
        """List sessions by agent ID."""
        cursor = self.collection.find({"agent_id": agent_id}).limit(limit)
        docs = await cursor.to_list(length=limit)

        return [self._doc_to_session(doc) for doc in docs]

    def _session_to_doc(self, session: Session) -> dict:
        """Convert Session to MongoDB document."""
        return {
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "workflow_state": {
                "config_hash": session.workflow_state.config_hash,
                "need_greeting": session.workflow_state.need_greeting,
                "status": session.workflow_state.status,
                "metadata": session.workflow_state.metadata,
                "last_updated": session.workflow_state.last_updated.isoformat()
            },
            "messages": session.messages,
            "updated_at": datetime.utcnow()
        }

    def _doc_to_session(self, doc: dict) -> Session:
        """Convert MongoDB document to Session."""
        workflow_state = WorkflowState(
            config_hash=doc["workflow_state"]["config_hash"],
            need_greeting=doc["workflow_state"]["need_greeting"],
            status=doc["workflow_state"]["status"],
            metadata=doc["workflow_state"]["metadata"],
            last_updated=datetime.fromisoformat(doc["workflow_state"]["last_updated"])
        )

        return Session(
            session_id=doc["session_id"],
            agent_id=doc["agent_id"],
            workflow_state=workflow_state,
            messages=doc.get("messages", [])
        )


# =============================================================================
# PostgreSQL Implementation
# =============================================================================


class PostgreSQLSessionStore:
    """
    PostgreSQL session storage implementation.

    Usage:
        ```python
        import asyncpg

        pool = await asyncpg.create_pool("postgresql://localhost/workflow_agent")
        store = PostgreSQLSessionStore(pool)

        # Use with WorkflowAgent
        agent = WorkflowAgent(config, llm, session_store=store)
        ```
    """

    def __init__(self, pool: Any):
        """
        Initialize PostgreSQL session store.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def init_schema(self):
        """Initialize database schema."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    agent_id VARCHAR(255) NOT NULL,
                    config_hash VARCHAR(64),
                    need_greeting BOOLEAN DEFAULT TRUE,
                    status VARCHAR(50) DEFAULT 'ready',
                    metadata JSONB DEFAULT '{}',
                    messages JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            ''')

            # Create index on agent_id
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_sessions_agent_id
                ON sessions(agent_id)
            ''')

    async def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                'SELECT * FROM sessions WHERE session_id = $1',
                session_id
            )

            if not row:
                return None

            return self._row_to_session(row)

    async def save(self, session: Session) -> None:
        """Save session."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO sessions (
                    session_id, agent_id, config_hash, need_greeting,
                    status, metadata, messages, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (session_id)
                DO UPDATE SET
                    agent_id = $2,
                    config_hash = $3,
                    need_greeting = $4,
                    status = $5,
                    metadata = $6,
                    messages = $7,
                    updated_at = NOW()
            ''',
                session.session_id,
                session.agent_id,
                session.workflow_state.config_hash,
                session.workflow_state.need_greeting,
                session.workflow_state.status,
                session.workflow_state.metadata,
                session.messages
            )

    async def delete(self, session_id: str) -> None:
        """Delete session."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                'DELETE FROM sessions WHERE session_id = $1',
                session_id
            )

    async def list_by_agent(self, agent_id: str, limit: int = 100) -> list[Session]:
        """List sessions by agent ID."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                'SELECT * FROM sessions WHERE agent_id = $1 ORDER BY updated_at DESC LIMIT $2',
                agent_id, limit
            )

            return [self._row_to_session(row) for row in rows]

    def _row_to_session(self, row: Any) -> Session:
        """Convert database row to Session."""
        workflow_state = WorkflowState(
            config_hash=row['config_hash'] or "",
            need_greeting=row['need_greeting'],
            status=row['status'],
            metadata=row['metadata'] or {},
            last_updated=row['updated_at']
        )

        return Session(
            session_id=row['session_id'],
            agent_id=row['agent_id'],
            workflow_state=workflow_state,
            messages=row['messages'] or []
        )


# =============================================================================
# Redis Implementation (for caching)
# =============================================================================


class RedisPlanCache:
    """
    Redis plan cache implementation.

    Usage:
        ```python
        import redis.asyncio as redis

        client = redis.from_url("redis://localhost")
        cache = RedisPlanCache(client)

        # Use with WorkflowAgent
        agent = WorkflowAgent(config, llm, plan_cache=cache)
        ```
    """

    def __init__(self, client: Any, ttl: int = 3600):
        """
        Initialize Redis plan cache.

        Args:
            client: redis.asyncio.Redis client
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.client = client
        self.ttl = ttl

    def _make_key(self, workflow_id: str, config_hash: str) -> str:
        """Make Redis key."""
        return f"workflow:plan:{workflow_id}:{config_hash}"

    async def get(self, workflow_id: str, config_hash: str) -> CachedPlan | None:
        """Get cached plan."""
        key = self._make_key(workflow_id, config_hash)
        data = await self.client.get(key)

        if not data:
            return None

        import json
        plan_dict = json.loads(data)
        return CachedPlan(**plan_dict)

    async def set(self, plan: CachedPlan) -> None:
        """Save plan to cache."""
        key = self._make_key(plan.workflow_id, plan.config_hash)

        import json
        data = json.dumps(plan.model_dump())

        await self.client.set(key, data, ex=self.ttl)

    async def delete(self, workflow_id: str, config_hash: str) -> None:
        """Delete cache."""
        key = self._make_key(workflow_id, config_hash)
        await self.client.delete(key)

    async def clear_all(self, workflow_id: str) -> None:
        """Clear all caches for a workflow."""
        pattern = f"workflow:plan:{workflow_id}:*"
        cursor = 0

        while True:
            cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break


# =============================================================================
# Execution History Store
# =============================================================================


class ExecutionHistoryStore:
    """
    Store execution history for analytics and debugging.

    Usage:
        ```python
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient("mongodb://localhost:27017")
        history_store = ExecutionHistoryStore(client)

        # Log execution
        await history_store.log_execution(
            session_id="session_123",
            agent_id="agent_456",
            user_message="Help me",
            decision=decision_obj,
            result="Success"
        )
        ```
    """

    def __init__(self, client: Any, db_name: str = "workflow_agent"):
        """
        Initialize execution history store.

        Args:
            client: Motor AsyncIOMotorClient instance
            db_name: Database name
        """
        self.db = client[db_name]
        self.collection = self.db.execution_history

    async def log_execution(
        self,
        session_id: str,
        agent_id: str,
        user_message: str,
        decision: Any,
        result: str,
        metadata: dict | None = None
    ) -> None:
        """Log execution record."""
        doc = {
            "session_id": session_id,
            "agent_id": agent_id,
            "user_message": user_message,
            "decision": {
                "should_continue": decision.should_continue,
                "should_respond": decision.should_respond,
                "next_action": decision.next_action,
                "reasoning": decision.reasoning
            },
            "result": result,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }

        await self.collection.insert_one(doc)

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 100
    ) -> list[dict]:
        """Get execution history for a session."""
        cursor = self.collection.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit)

        return await cursor.to_list(length=limit)

    async def get_agent_stats(self, agent_id: str) -> dict:
        """Get statistics for an agent."""
        pipeline = [
            {"$match": {"agent_id": agent_id}},
            {"$group": {
                "_id": None,
                "total_executions": {"$sum": 1},
                "avg_iterations": {"$avg": "$decision.iteration"},
                "success_rate": {
                    "$avg": {
                        "$cond": [
                            {"$regexMatch": {"input": "$result", "regex": "success", "options": "i"}},
                            1,
                            0
                        ]
                    }
                }
            }}
        ]

        result = await self.collection.aggregate(pipeline).to_list(length=1)
        return result[0] if result else {}
