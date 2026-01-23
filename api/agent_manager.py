"""
Agent 管理器

负责 Agent 的创建、缓存、生命周期管理和自动回收
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.config import load_config, get_llm_decision_llm
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema

logger = logging.getLogger(__name__)


class AgentInfo:
    """Agent 信息"""

    def __init__(
        self,
        agent: WorkflowAgent,
        chatbot_id: str,
        tenant_id: str,
        config_hash: str,
    ):
        self.agent = agent
        self.chatbot_id = chatbot_id
        self.tenant_id = tenant_id
        self.config_hash = config_hash
        self.session_ids: set[str] = set()
        self.created_at = time.time()
        self.last_active_at = time.time()

    def add_session(self, session_id: str):
        """添加会话"""
        self.session_ids.add(session_id)
        self.last_active_at = time.time()

    def remove_session(self, session_id: str):
        """移除会话"""
        self.session_ids.discard(session_id)
        self.last_active_at = time.time()

    @property
    def session_count(self) -> int:
        """会话数量"""
        return len(self.session_ids)

    @property
    def is_idle(self) -> bool:
        """是否空闲（无会话）"""
        return self.session_count == 0

    @property
    def idle_time(self) -> float:
        """空闲时间（秒）"""
        return time.time() - self.last_active_at


class AgentManager:
    """
    Agent 管理器

    职责：
    1. 根据 chatbot_id + tenant_id 创建和缓存 Agent
    2. 管理 Agent 的生命周期
    3. 自动回收空闲 Agent
    4. 配置文件变更检测和热重载
    """

    def __init__(
        self,
        config_dir: str = "config",
        idle_timeout: int = 300,  # 5分钟无会话自动回收
        cleanup_interval: int = 60,  # 每分钟检查一次
    ):
        self.config_dir = Path(config_dir)
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval

        # Agent 缓存: {agent_key: AgentInfo}
        self._agents: Dict[str, AgentInfo] = {}

        # 应用配置（LLM、存储等）
        self._app_config = load_config()

        # 启动时间
        self._start_time = time.time()

        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None

    @staticmethod
    def _get_agent_key(chatbot_id: str, tenant_id: str) -> str:
        """生成 Agent 缓存键"""
        return f"{tenant_id}:{chatbot_id}"

    @staticmethod
    def _compute_config_hash(config: WorkflowConfigSchema) -> str:
        """计算配置哈希"""
        config_str = json.dumps(config.model_dump(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_config_path(self, chatbot_id: str, tenant_id: str) -> Path:
        """获取配置文件路径"""
        # 支持多种配置文件命名方式
        # 1. tenant_id/chatbot_id.json
        # 2. chatbot_id.json
        # 3. workflow_config.json (默认)

        tenant_config = self.config_dir / tenant_id / f"{chatbot_id}.json"
        if tenant_config.exists():
            return tenant_config

        chatbot_config = self.config_dir / f"{chatbot_id}.json"
        if chatbot_config.exists():
            return chatbot_config

        default_config = self.config_dir / "workflow_config.json"
        if default_config.exists():
            return default_config

        raise FileNotFoundError(
            f"Configuration file not found for chatbot_id={chatbot_id}, "
            f"tenant_id={tenant_id}"
        )

    async def _load_workflow_config(
        self, chatbot_id: str, tenant_id: str
    ) -> WorkflowConfigSchema:
        """加载工作流配置"""
        config_path = self._get_config_path(chatbot_id, tenant_id)

        logger.info(f"Loading workflow config from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return WorkflowConfigSchema(**config_data)

    async def _create_agent(
        self, chatbot_id: str, tenant_id: str
    ) -> AgentInfo:
        """创建新的 Agent"""
        logger.info(f"Creating new agent for chatbot_id={chatbot_id}, tenant_id={tenant_id}")

        # 加载工作流配置
        workflow_config = await self._load_workflow_config(chatbot_id, tenant_id)

        # 计算配置哈希
        config_hash = self._compute_config_hash(workflow_config)

        # 创建 LLM
        llm = get_llm_decision_llm(self._app_config)

        # 创建存储组件（使用内存存储）
        # 注意：如果需要持久化存储，可以在这里配置 MongoDB/Redis
        session_store = None
        plan_cache = None

        # 创建 WorkflowAgent
        agent = WorkflowAgent(
            config=workflow_config,
            llm=llm,
            session_store=session_store,
            plan_cache=plan_cache,
        )

        # 创建 AgentInfo
        agent_info = AgentInfo(
            agent=agent,
            chatbot_id=chatbot_id,
            tenant_id=tenant_id,
            config_hash=config_hash,
        )

        logger.info(
            f"Agent created successfully: agent_key={self._get_agent_key(chatbot_id, tenant_id)}, "
            f"config_hash={config_hash}"
        )

        return agent_info

    async def get_or_create_agent(
        self,
        chatbot_id: str,
        tenant_id: str,
        session_id: str,
        md5_checksum: Optional[str] = None,
    ) -> WorkflowAgent:
        """
        获取或创建 Agent

        Args:
            chatbot_id: Chatbot ID
            tenant_id: 租户 ID
            session_id: 会话 ID
            md5_checksum: 配置文件 MD5 校验和（用于检测配置变更）

        Returns:
            WorkflowAgent 实例
        """
        agent_key = self._get_agent_key(chatbot_id, tenant_id)

        # 检查是否已存在
        if agent_key in self._agents:
            agent_info = self._agents[agent_key]

            # 检查配置是否变更
            if md5_checksum and md5_checksum != agent_info.config_hash:
                logger.info(
                    f"Configuration changed for {agent_key}, "
                    f"old_hash={agent_info.config_hash}, new_hash={md5_checksum}"
                )
                # 配置变更，重新创建 Agent
                await self.remove_agent(chatbot_id, tenant_id)
            else:
                # 配置未变更，复用现有 Agent
                agent_info.add_session(session_id)
                logger.debug(
                    f"Reusing existing agent: {agent_key}, "
                    f"session_count={agent_info.session_count}"
                )
                return agent_info.agent

        # 创建新 Agent
        agent_info = await self._create_agent(chatbot_id, tenant_id)
        agent_info.add_session(session_id)
        self._agents[agent_key] = agent_info

        return agent_info.agent

    async def release_session(
        self, chatbot_id: str, tenant_id: str, session_id: str
    ):
        """
        释放会话

        当会话结束或超时时调用，减少 Agent 的会话计数
        """
        agent_key = self._get_agent_key(chatbot_id, tenant_id)

        if agent_key in self._agents:
            agent_info = self._agents[agent_key]
            agent_info.remove_session(session_id)

            logger.debug(
                f"Session released: {agent_key}, session_id={session_id}, "
                f"remaining_sessions={agent_info.session_count}"
            )

            # 如果没有会话了，标记为空闲
            if agent_info.is_idle:
                logger.info(f"Agent {agent_key} is now idle")

    async def remove_agent(self, chatbot_id: str, tenant_id: str):
        """移除 Agent"""
        agent_key = self._get_agent_key(chatbot_id, tenant_id)

        if agent_key in self._agents:
            agent_info = self._agents[agent_key]
            logger.info(
                f"Removing agent: {agent_key}, "
                f"session_count={agent_info.session_count}"
            )
            del self._agents[agent_key]

    async def _cleanup_idle_agents(self):
        """清理空闲 Agent"""
        now = time.time()
        to_remove = []

        for agent_key, agent_info in self._agents.items():
            if agent_info.is_idle and agent_info.idle_time > self.idle_timeout:
                to_remove.append(agent_key)
                logger.info(
                    f"Agent {agent_key} idle for {agent_info.idle_time:.1f}s, "
                    f"removing..."
                )

        for agent_key in to_remove:
            chatbot_id, tenant_id = agent_key.split(":", 1)
            await self.remove_agent(chatbot_id, tenant_id)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} idle agents")

    async def _cleanup_loop(self):
        """清理循环"""
        logger.info(
            f"Agent cleanup loop started: "
            f"idle_timeout={self.idle_timeout}s, "
            f"cleanup_interval={self.cleanup_interval}s"
        )

        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle_agents()
            except asyncio.CancelledError:
                logger.info("Agent cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    def start_cleanup(self):
        """启动清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Agent cleanup task started")

    async def stop_cleanup(self):
        """停止清理任务"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Agent cleanup task stopped")

    def get_stats(self) -> dict:
        """获取统计信息"""
        total_sessions = sum(info.session_count for info in self._agents.values())
        idle_agents = sum(1 for info in self._agents.values() if info.is_idle)

        return {
            "active_agents": len(self._agents),
            "idle_agents": idle_agents,
            "active_sessions": total_sessions,
            "uptime": time.time() - self._start_time,
        }

    def get_agent_info(self, chatbot_id: str, tenant_id: str) -> Optional[Dict]:
        """获取 Agent 信息"""
        agent_key = self._get_agent_key(chatbot_id, tenant_id)

        if agent_key not in self._agents:
            return None

        agent_info = self._agents[agent_key]

        return {
            "agent_id": agent_key,
            "chatbot_id": agent_info.chatbot_id,
            "tenant_id": agent_info.tenant_id,
            "config_hash": agent_info.config_hash,
            "session_count": agent_info.session_count,
            "created_at": datetime.fromtimestamp(agent_info.created_at).isoformat(),
            "last_active_at": datetime.fromtimestamp(agent_info.last_active_at).isoformat(),
            "is_idle": agent_info.is_idle,
            "idle_time": agent_info.idle_time if agent_info.is_idle else 0,
        }
