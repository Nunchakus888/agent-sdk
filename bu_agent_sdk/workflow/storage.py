"""
Database storage implementations for workflow agent.

Supports:
- Config storage (MongoDB) - tenant_id, chatbot_id, config_hash indexes
- Session storage (MongoDB, PostgreSQL, Redis) - chatbot_id, session_id indexes
- Plan cache storage (Redis)
- Execution history tracking (MongoDB)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Protocol

from bu_agent_sdk.agent.workflow_state import Session, WorkflowState
from bu_agent_sdk.workflow.cache import CachedPlan


# =============================================================================
# Config Store Interface & Implementation
# =============================================================================


class ConfigStore(Protocol):
    """Config storage interface."""

    async def get_by_hash(self, config_hash: str) -> dict | None:
        """Get config by hash."""
        ...

    async def get_by_chatbot(self, chatbot_id: str) -> dict | None:
        """Get latest config by chatbot_id."""
        ...

    async def save(self, config: dict, tenant_id: str, chatbot_id: str) -> str:
        """Save config, returns config_hash."""
        ...

    async def delete(self, config_hash: str) -> None:
        """Delete config by hash."""
        ...


class MongoDBConfigStore:
    """
    MongoDB config storage implementation.

    Indexes:
    - tenant_id (for multi-tenant queries)
    - chatbot_id (for chatbot-level queries)
    - config_hash (unique, for version control)

    Usage:
        ```python
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient("mongodb://localhost:27017")
        store = MongoDBConfigStore(client, db_name="workflow_agent")

        # Save config
        config_hash = await store.save(config_dict, "tenant_001", "bot_001")

        # Get config
        config = await store.get_by_hash(config_hash)
        ```
    """

    def __init__(self, client: Any, db_name: str = "workflow_agent"):
        self.db = client[db_name]
        self.collection = self.db.configs

    async def ensure_indexes(self) -> None:
        """Create indexes for optimal query performance."""
        await self.collection.create_index("tenant_id")
        await self.collection.create_index("chatbot_id")
        await self.collection.create_index("config_hash", unique=True)
        await self.collection.create_index([("chatbot_id", 1), ("version", -1)])

    async def get_by_hash(self, config_hash: str) -> dict | None:
        """Get config by hash."""
        doc = await self.collection.find_one({"config_hash": config_hash})
        return doc.get("config_data") if doc else None

    async def get_by_chatbot(self, chatbot_id: str) -> dict | None:
        """Get latest config by chatbot_id."""
        doc = await self.collection.find_one(
            {"chatbot_id": chatbot_id},
            sort=[("version", -1)]
        )
        return doc.get("config_data") if doc else None

    async def save(self, config: dict, tenant_id: str, chatbot_id: str) -> str:
        """Save config, returns config_hash."""
        import hashlib
        import json

        # Compute config hash
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Check if already exists
        existing = await self.collection.find_one({"config_hash": config_hash})
        if existing:
            return config_hash

        # Get next version for this chatbot
        latest = await self.collection.find_one(
            {"chatbot_id": chatbot_id},
            sort=[("version", -1)]
        )
        version = (latest.get("version", 0) + 1) if latest else 1

        doc = {
            "tenant_id": tenant_id,
            "chatbot_id": chatbot_id,
            "config_hash": config_hash,
            "config_data": config,
            "version": version,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        await self.collection.insert_one(doc)
        return config_hash

    async def delete(self, config_hash: str) -> None:
        """Delete config by hash."""
        await self.collection.delete_one({"config_hash": config_hash})

    async def list_by_tenant(self, tenant_id: str, limit: int = 100) -> list[dict]:
        """List configs by tenant_id."""
        cursor = self.collection.find(
            {"tenant_id": tenant_id}
        ).sort("updated_at", -1).limit(limit)
        return await cursor.to_list(length=limit)


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

    Indexes:
    - session_id (unique, primary key)
    - chatbot_id (for chatbot-level queries)
    - config_hash (for config-level queries)
    - updated_at (for TTL expiration)

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
        self.db = client[db_name]
        self.collection = self.db.sessions

    async def ensure_indexes(self) -> None:
        """Create indexes for optimal query performance."""
        await self.collection.create_index("session_id", unique=True)
        await self.collection.create_index("chatbot_id")
        await self.collection.create_index("config_hash")
        await self.collection.create_index("updated_at", expireAfterSeconds=86400)  # 24h TTL

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

    async def list_by_chatbot(self, chatbot_id: str, limit: int = 100) -> list[Session]:
        """List sessions by chatbot_id."""
        cursor = self.collection.find(
            {"chatbot_id": chatbot_id}
        ).sort("updated_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self._doc_to_session(doc) for doc in docs]

    async def list_by_agent(self, agent_id: str, limit: int = 100) -> list[Session]:
        """List sessions by agent ID (backward compatible)."""
        cursor = self.collection.find({"agent_id": agent_id}).limit(limit)
        docs = await cursor.to_list(length=limit)

        return [self._doc_to_session(doc) for doc in docs]

    def _session_to_doc(self, session: Session) -> dict:
        """Convert Session to MongoDB document."""
        return {
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "chatbot_id": getattr(session, 'chatbot_id', ''),
            "config_hash": session.workflow_state.config_hash,
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

        session = Session(
            session_id=doc["session_id"],
            agent_id=doc["agent_id"],
            workflow_state=workflow_state,
            messages=doc.get("messages", [])
        )
        # Add chatbot_id if present
        if "chatbot_id" in doc:
            session.chatbot_id = doc["chatbot_id"]
        return session


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


# =============================================================================
# Workflow Engine (Service-level Instance Management)
# =============================================================================


class WorkflowEngine:
    """
    Service-level workflow engine with persistent storage.

    Features:
    - Agent cache pool (by config_hash, with LRU eviction)
    - Session management (MongoDB persistent storage)
    - Config management (MongoDB persistent storage)
    - Memory leak prevention (TTL + max size limits)

    Usage:
        ```python
        from motor.motor_asyncio import AsyncIOMotorClient

        client = AsyncIOMotorClient("mongodb://localhost:27017")
        engine = WorkflowEngine(
            mongo_client=client,
            llm_factory=lambda: ChatOpenAI(model="gpt-4o")
        )

        # Initialize indexes
        await engine.init()

        # Process message
        response = await engine.process_message(
            session_id="sess_001",
            chatbot_id="bot_001",
            user_message="Hello"
        )
        ```
    """

    def __init__(
        self,
        mongo_client: Any,
        llm_factory: Any,
        db_name: str = "workflow_agent",
        max_agents: int = 100,
        agent_ttl_seconds: int = 86400,  # 24h
    ):
        self.db_name = db_name
        self.llm_factory = llm_factory

        # Storage components
        self.config_store = MongoDBConfigStore(mongo_client, db_name)
        self.session_store = MongoDBSessionStore(mongo_client, db_name)
        self.history_store = ExecutionHistoryStore(mongo_client, db_name)

        # Agent cache pool (config_hash -> (agent, last_access_time))
        self._agent_cache: dict[str, tuple[Any, datetime]] = {}
        self._max_agents = max_agents
        self._agent_ttl = agent_ttl_seconds

    async def init(self) -> None:
        """Initialize storage indexes."""
        await self.config_store.ensure_indexes()
        await self.session_store.ensure_indexes()

    async def process_message(
        self,
        session_id: str,
        chatbot_id: str,
        user_message: str,
        tenant_id: str = "default"
    ) -> str:
        """
        Process user message through workflow agent.

        Flow:
        1. Load/create session
        2. Get config (by chatbot_id or session's config_hash)
        3. Get/create agent (by config_hash, cached)
        4. Execute query
        5. Save session (async)
        """
        # 1. Load or create session
        session = await self.session_store.get(session_id)

        if not session:
            # Get config for this chatbot
            config_data = await self.config_store.get_by_chatbot(chatbot_id)
            if not config_data:
                return "Error: No config found for chatbot"

            # Compute config hash
            import hashlib
            import json
            config_str = json.dumps(config_data, sort_keys=True, ensure_ascii=False)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()

            # Create new session
            session = Session(
                session_id=session_id,
                agent_id=f"workflow_{config_hash[:8]}",
                workflow_state=WorkflowState(config_hash=config_hash),
                messages=[]
            )
            session.chatbot_id = chatbot_id
        else:
            # Load config by session's config_hash
            config_data = await self.config_store.get_by_hash(
                session.workflow_state.config_hash
            )
            if not config_data:
                return "Error: Config not found"

        # 2. Get or create agent
        agent = await self._get_or_create_agent(
            session.workflow_state.config_hash,
            config_data
        )

        # 3. Execute query
        response = await agent.query(user_message, session_id)

        # 4. Save session (async)
        import asyncio
        asyncio.create_task(self.session_store.save(session))

        return response

    async def _get_or_create_agent(
        self,
        config_hash: str,
        config_data: dict
    ) -> Any:
        """Get agent from cache or create new one."""
        # Check cache
        if config_hash in self._agent_cache:
            agent, _ = self._agent_cache[config_hash]
            self._agent_cache[config_hash] = (agent, datetime.utcnow())
            return agent

        # Evict old agents if needed
        await self._evict_agents()

        # Create new agent
        from bu_agent_sdk.tools.actions import WorkflowConfigSchema
        from bu_agent_sdk.agent.workflow_agent import WorkflowAgent

        config = WorkflowConfigSchema(**config_data)
        llm = self.llm_factory()

        agent = WorkflowAgent(
            config=config,
            llm=llm,
            session_store=self.session_store
        )

        self._agent_cache[config_hash] = (agent, datetime.utcnow())
        return agent

    async def _evict_agents(self) -> None:
        """Evict old agents from cache (LRU + TTL)."""
        now = datetime.utcnow()

        # Remove expired agents
        expired = [
            h for h, (_, t) in self._agent_cache.items()
            if (now - t).total_seconds() > self._agent_ttl
        ]
        for h in expired:
            del self._agent_cache[h]

        # LRU eviction if still over limit
        if len(self._agent_cache) >= self._max_agents:
            # Sort by last access time, remove oldest
            sorted_items = sorted(
                self._agent_cache.items(),
                key=lambda x: x[1][1]
            )
            to_remove = len(self._agent_cache) - self._max_agents + 1
            for h, _ in sorted_items[:to_remove]:
                del self._agent_cache[h]

    async def register_config(
        self,
        config: dict,
        tenant_id: str,
        chatbot_id: str
    ) -> str:
        """Register a new config, returns config_hash."""
        return await self.config_store.save(config, tenant_id, chatbot_id)

    async def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return await self.session_store.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        """Delete session."""
        await self.session_store.delete(session_id)

    def get_cache_stats(self) -> dict:
        """Get agent cache statistics."""
        return {
            "cached_agents": len(self._agent_cache),
            "max_agents": self._max_agents,
            "agent_ttl_seconds": self._agent_ttl
        }
