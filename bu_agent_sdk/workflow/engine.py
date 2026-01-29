"""
Unified Workflow Engine

统一的工作流引擎，合并 AgentManager 和 WorkflowEngine 功能：
- 支持内存模式和 MongoDB 持久化模式
- 基于 config_hash 的 Agent 缓存
- 可选的 LLM 配置解析
- LRU + TTL 缓存淘汰策略
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.agent.workflow_state import Session, WorkflowState
from bu_agent_sdk.tools.actions import WorkflowConfigSchema

logger = logging.getLogger(__name__)


# =============================================================================
# Storage Protocols
# =============================================================================


class ConfigStore(Protocol):
    """Config storage interface."""

    async def get_by_hash(self, config_hash: str) -> dict | None: ...
    async def get_by_chatbot(self, chatbot_id: str) -> dict | None: ...
    async def save(self, config: dict, tenant_id: str, chatbot_id: str) -> str: ...
    async def delete(self, config_hash: str) -> None: ...


class SessionStore(Protocol):
    """Session storage interface."""

    async def get(self, session_id: str) -> Session | None: ...
    async def save(self, session: Session) -> None: ...
    async def delete(self, session_id: str) -> None: ...


# =============================================================================
# In-Memory Storage (Default)
# =============================================================================


class InMemoryConfigStore:
    """In-memory config storage."""

    def __init__(self):
        self._configs: dict[str, dict] = {}  # config_hash -> config
        self._chatbot_index: dict[str, str] = {}  # chatbot_id -> config_hash

    async def get_by_hash(self, config_hash: str) -> dict | None:
        return self._configs.get(config_hash)

    async def get_by_chatbot(self, chatbot_id: str) -> dict | None:
        config_hash = self._chatbot_index.get(chatbot_id)
        return self._configs.get(config_hash) if config_hash else None

    async def save(self, config: dict, tenant_id: str, chatbot_id: str) -> str:
        config_hash = compute_config_hash(config)
        self._configs[config_hash] = config
        self._chatbot_index[chatbot_id] = config_hash
        return config_hash

    async def delete(self, config_hash: str) -> None:
        self._configs.pop(config_hash, None)


class InMemorySessionStore:
    """In-memory session storage."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# =============================================================================
# Utility Functions
# =============================================================================


def compute_config_hash(config: dict) -> str:
    """Compute MD5 hash of config dict."""
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(config_str.encode()).hexdigest()


# =============================================================================
# Cached Agent Entry
# =============================================================================


class CachedAgent:
    """Agent cache entry with metadata."""

    __slots__ = ("agent", "config_hash", "created_at", "last_access", "session_ids")

    def __init__(self, agent: WorkflowAgent, config_hash: str):
        self.agent = agent
        self.config_hash = config_hash
        self.created_at = time.time()
        self.last_access = time.time()
        self.session_ids: set[str] = set()

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = time.time()

    @property
    def idle_seconds(self) -> float:
        """Seconds since last access."""
        return time.time() - self.last_access

    @property
    def is_idle(self) -> bool:
        """True if no active sessions."""
        return len(self.session_ids) == 0


# =============================================================================
# Unified Workflow Engine
# =============================================================================


class WorkflowEngine:
    """
    Unified workflow engine.

    Features:
    - Dual mode: in-memory (default) or MongoDB persistent
    - Agent caching by config_hash with LRU + TTL eviction
    - Optional LLM config parsing
    - File-based config loading for development

    Usage:
        ```python
        # In-memory mode (default)
        engine = WorkflowEngine(llm_factory=lambda: AnthropicChat())
        await engine.init()

        # MongoDB mode
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        engine = WorkflowEngine(
            llm_factory=lambda: AnthropicChat(),
            mongo_client=client
        )
        await engine.init()

        # Process message
        response = await engine.query(
            session_id="sess_001",
            chatbot_id="bot_001",
            user_message="Hello"
        )
        ```
    """

    def __init__(
        self,
        llm_factory: Callable[[], Any],
        mongo_client: Any | None = None,
        db_name: str = "workflow_agent",
        config_dir: str | None = None,
        max_agents: int = 100,
        agent_ttl: int = 3600,  # 1 hour
        idle_timeout: int = 300,  # 5 minutes
        cleanup_interval: int = 60,  # 1 minute
    ):
        """
        Initialize workflow engine.

        Args:
            llm_factory: Factory function to create LLM instances
            mongo_client: MongoDB client (optional, enables persistent mode)
            db_name: MongoDB database name
            config_dir: Directory for file-based configs (development mode)
            max_agents: Maximum cached agents
            agent_ttl: Agent TTL in seconds
            idle_timeout: Idle agent timeout in seconds
            cleanup_interval: Cleanup check interval in seconds
        """
        self.llm_factory = llm_factory
        self.db_name = db_name
        self.config_dir = Path(config_dir) if config_dir else None
        self.max_agents = max_agents
        self.agent_ttl = agent_ttl
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval

        # Storage (in-memory or MongoDB)
        self._mongo_client = mongo_client
        self._init_storage(mongo_client, db_name)

        # Agent cache: config_hash -> CachedAgent
        self._agents: dict[str, CachedAgent] = {}

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._start_time = time.time()

        logger.info(
            f"WorkflowEngine initialized: "
            f"mode={'mongodb' if mongo_client else 'memory'}, "
            f"max_agents={max_agents}, agent_ttl={agent_ttl}s"
        )

    def _init_storage(self, mongo_client: Any | None, db_name: str) -> None:
        """Initialize storage backends."""
        if mongo_client:
            from bu_agent_sdk.workflow.storage import (
                MongoDBConfigStore,
                MongoDBSessionStore,
            )
            self.config_store = MongoDBConfigStore(mongo_client, db_name)
            self.session_store = MongoDBSessionStore(mongo_client, db_name)
        else:
            self.config_store = InMemoryConfigStore()
            self.session_store = InMemorySessionStore()

    async def init(self) -> None:
        """Initialize engine (create indexes, start cleanup)."""
        if self._mongo_client:
            await self.config_store.ensure_indexes()
            await self.session_store.ensure_indexes()
        self.start_cleanup()
        logger.info("WorkflowEngine initialized")

    async def shutdown(self) -> None:
        """Shutdown engine."""
        await self.stop_cleanup()
        self._agents.clear()
        logger.info("WorkflowEngine shutdown complete")

    # =========================================================================
    # Core API
    # =========================================================================

    async def query(
        self,
        session_id: str,
        chatbot_id: str,
        user_message: str,
        tenant_id: str = "default",
        config: dict | None = None,
    ) -> str:
        """
        Process user message through workflow agent.

        Args:
            session_id: Session ID
            chatbot_id: Chatbot ID
            user_message: User message
            tenant_id: Tenant ID (for multi-tenant)
            config: Optional config dict (if not in store)

        Returns:
            Agent response string
        """
        # 1. Get or create session
        session = await self.session_store.get(session_id)

        if session:
            # Existing session - load config by hash
            config_data = await self.config_store.get_by_hash(
                session.workflow_state.config_hash
            )
            if not config_data:
                return "Error: Config not found for session"
        else:
            # New session - get config
            config_data = config or await self._load_config(chatbot_id, tenant_id)
            if not config_data:
                return "Error: No config found"

            config_hash = compute_config_hash(config_data)
            session = Session(
                session_id=session_id,
                agent_id=f"workflow_{config_hash[:8]}",
                workflow_state=WorkflowState(config_hash=config_hash),
                messages=[],
            )
            session.chatbot_id = chatbot_id

        # 2. Get or create agent
        agent = await self._get_or_create_agent(
            session.workflow_state.config_hash, config_data
        )

        # 3. Track session
        cached = self._agents.get(session.workflow_state.config_hash)
        if cached:
            cached.session_ids.add(session_id)
            cached.touch()

        # 4. Execute query
        response = await agent.query(user_message, session_id)

        # 5. Save session (async)
        asyncio.create_task(self.session_store.save(session))

        return response

    async def register_config(
        self, config: dict, tenant_id: str, chatbot_id: str
    ) -> str:
        """Register config, returns config_hash."""
        return await self.config_store.save(config, tenant_id, chatbot_id)

    async def release_session(self, session_id: str, config_hash: str) -> None:
        """Release session from agent tracking."""
        cached = self._agents.get(config_hash)
        if cached:
            cached.session_ids.discard(session_id)
            cached.touch()
            logger.debug(f"Session {session_id} released from {config_hash[:8]}")

    async def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return await self.session_store.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        """Delete session."""
        await self.session_store.delete(session_id)

    # =========================================================================
    # Agent Management
    # =========================================================================

    async def _load_config(self, chatbot_id: str, tenant_id: str) -> dict | None:
        """Load config from store or file."""
        # Try store first
        config = await self.config_store.get_by_chatbot(chatbot_id)
        if config:
            return config

        # Try file-based config
        if self.config_dir:
            return self._load_config_file(chatbot_id, tenant_id)

        return None

    def _load_config_file(self, chatbot_id: str, tenant_id: str) -> dict | None:
        """Load config from file system."""
        if not self.config_dir:
            return None

        # Try: tenant_id/chatbot_id.json -> chatbot_id.json -> default.json
        paths = [
            self.config_dir / tenant_id / f"{chatbot_id}.json",
            self.config_dir / f"{chatbot_id}.json",
            self.config_dir / "workflow_config.json",
        ]

        for path in paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    logger.debug(f"Loaded config from {path}")
                    return json.load(f)

        return None

    async def _get_or_create_agent(
        self, config_hash: str, config_data: dict
    ) -> WorkflowAgent:
        """Get agent from cache or create new one."""
        # Check cache
        if config_hash in self._agents:
            cached = self._agents[config_hash]
            cached.touch()
            return cached.agent

        # Evict if needed
        await self._evict_agents()

        # Create new agent
        config = WorkflowConfigSchema(**config_data)
        llm = self.llm_factory()

        agent = WorkflowAgent(
            config=config,
            llm=llm,
            session_store=self.session_store,
        )

        self._agents[config_hash] = CachedAgent(agent, config_hash)
        logger.info(f"Created agent for config {config_hash[:8]}")

        return agent

    async def _evict_agents(self) -> None:
        """Evict agents (TTL + LRU)."""
        now = time.time()

        # Remove expired
        expired = [
            h for h, c in self._agents.items()
            if now - c.last_access > self.agent_ttl
        ]
        for h in expired:
            del self._agents[h]
            logger.debug(f"Evicted expired agent {h[:8]}")

        # LRU eviction if over limit
        if len(self._agents) >= self.max_agents:
            sorted_agents = sorted(
                self._agents.items(), key=lambda x: x[1].last_access
            )
            to_remove = len(self._agents) - self.max_agents + 1
            for h, _ in sorted_agents[:to_remove]:
                del self._agents[h]
                logger.debug(f"Evicted LRU agent {h[:8]}")

    # =========================================================================
    # Cleanup Task
    # =========================================================================

    def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cleanup task started")

    async def stop_cleanup(self) -> None:
        """Stop cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_idle_agents(self) -> None:
        """Remove idle agents."""
        to_remove = [
            h for h, c in self._agents.items()
            if c.is_idle and c.idle_seconds > self.idle_timeout
        ]
        for h in to_remove:
            del self._agents[h]
            logger.info(f"Cleaned up idle agent {h[:8]}")

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> dict:
        """Get engine statistics."""
        total_sessions = sum(len(c.session_ids) for c in self._agents.values())
        idle_count = sum(1 for c in self._agents.values() if c.is_idle)

        return {
            "mode": "mongodb" if self._mongo_client else "memory",
            "cached_agents": len(self._agents),
            "max_agents": self.max_agents,
            "idle_agents": idle_count,
            "active_sessions": total_sessions,
            "uptime_seconds": time.time() - self._start_time,
        }
