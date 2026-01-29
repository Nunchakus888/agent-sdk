"""
Repository 层

提供各数据表的 CRUD 操作，封装数据库访问逻辑
支持 MongoDB 和内存存储两种模式
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional

from api.services.database import (
    Database,
    COLLECTIONS,
)
from api.models import (
    SessionDocument,
    SessionStatus,
    MessageDocument,
    MessageRole,
    AgentStateDocument,
    AgentStatus,
    AuditLogDocument,
    AuditAction,
)

logger = logging.getLogger(__name__)


def generate_id() -> str:
    """生成唯一 ID"""
    return uuid.uuid4().hex


# =============================================================================
# 基础 Repository 接口
# =============================================================================


class BaseRepository(ABC):
    """Repository 基类"""

    def __init__(self, db: Database | None):
        self._db = db

    @property
    def is_persistent(self) -> bool:
        """是否持久化存储"""
        return self._db is not None and self._db.is_connected


# =============================================================================
# Session Repository
# =============================================================================


class SessionRepository(BaseRepository):
    """
    会话 Repository

    管理会话的创建、查询、更新和关闭
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, SessionDocument] = {}

    async def create(
        self,
        session_id: str,
        tenant_id: str,
        chatbot_id: str,
        customer_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        config_hash: Optional[str] = None,
        title: Optional[str] = None,
        source: str = "api",
        is_preview: bool = False,
        metadata: Optional[dict] = None,
    ) -> SessionDocument:
        """创建会话"""
        now = datetime.utcnow()
        session = SessionDocument(
            session_id=session_id,
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            customer_id=customer_id,
            status=SessionStatus.ACTIVE,
            agent_id=agent_id,
            config_hash=config_hash,
            message_count=0,
            title=title,
            source=source,
            is_preview=is_preview,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        if self.is_persistent:
            await self._db.sessions.insert_one(session.to_dict())
            logger.debug(f"Session created in DB: {session_id}")
        else:
            self._memory_store[session_id] = session
            logger.debug(f"Session created in memory: {session_id}")

        return session

    async def get(self, session_id: str) -> Optional[SessionDocument]:
        """获取会话"""
        if self.is_persistent:
            doc = await self._db.sessions.find_one({"_id": session_id})
            return SessionDocument.from_dict(doc) if doc else None
        else:
            return self._memory_store.get(session_id)

    async def get_or_create(
        self,
        session_id: str,
        tenant_id: str,
        chatbot_id: str,
        **kwargs,
    ) -> tuple[SessionDocument, bool]:
        """
        获取或创建会话

        Returns:
            (session, created): 会话对象和是否新创建
        """
        session = await self.get(session_id)
        if session:
            return session, False

        session = await self.create(
            session_id=session_id,
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            **kwargs,
        )
        return session, True

    async def update(
        self,
        session_id: str,
        **updates,
    ) -> Optional[SessionDocument]:
        """更新会话"""
        updates["updated_at"] = datetime.utcnow()

        if self.is_persistent:
            result = await self._db.sessions.find_one_and_update(
                {"_id": session_id},
                {"$set": updates},
                return_document=True,
            )
            return SessionDocument.from_dict(result) if result else None
        else:
            session = self._memory_store.get(session_id)
            if session:
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                return session
            return None

    async def increment_message_count(
        self,
        session_id: str,
        increment: int = 1,
    ) -> None:
        """增加消息计数"""
        now = datetime.utcnow()

        if self.is_persistent:
            await self._db.sessions.update_one(
                {"_id": session_id},
                {
                    "$inc": {"message_count": increment},
                    "$set": {
                        "updated_at": now,
                        "last_message_at": now,
                    },
                },
            )
        else:
            session = self._memory_store.get(session_id)
            if session:
                session.message_count += increment
                session.updated_at = now
                session.last_message_at = now

    async def close(self, session_id: str) -> Optional[SessionDocument]:
        """关闭会话"""
        now = datetime.utcnow()
        return await self.update(
            session_id,
            status=SessionStatus.CLOSED.value,
            closed_at=now,
        )

    async def list_by_tenant(
        self,
        tenant_id: str,
        chatbot_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionDocument]:
        """按租户列出会话"""
        query = {"tenant_id": tenant_id}
        if chatbot_id:
            query["chatbot_id"] = chatbot_id
        if status:
            query["status"] = status.value

        if self.is_persistent:
            cursor = self._db.sessions.find(query).sort("created_at", -1).skip(offset).limit(limit)
            docs = await cursor.to_list(length=limit)
            return [SessionDocument.from_dict(doc) for doc in docs]
        else:
            sessions = [
                s for s in self._memory_store.values()
                if s.tenant_id == tenant_id
                and (chatbot_id is None or s.chatbot_id == chatbot_id)
                and (status is None or s.status == status)
            ]
            sessions.sort(key=lambda x: x.created_at, reverse=True)
            return sessions[offset:offset + limit]


# =============================================================================
# Message Repository
# =============================================================================


class MessageRepository(BaseRepository):
    """
    消息 Repository

    管理会话消息的存储和查询
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, MessageDocument] = {}
        self._session_index: dict[str, List[str]] = {}  # session_id -> [message_ids]

    async def create(
        self,
        session_id: str,
        tenant_id: str,
        role: MessageRole,
        content: str,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        tool_calls: Optional[List[dict]] = None,
        tool_call_id: Optional[str] = None,
        token_count: Optional[int] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> MessageDocument:
        """创建消息"""
        msg_id = message_id or generate_id()
        message = MessageDocument(
            message_id=msg_id,
            session_id=session_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            correlation_id=correlation_id,
            parent_message_id=parent_message_id,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            token_count=token_count,
            latency_ms=latency_ms,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )

        if self.is_persistent:
            await self._db.session_messages.insert_one(message.to_dict())
            logger.debug(f"Message created in DB: {msg_id}")
        else:
            self._memory_store[msg_id] = message
            if session_id not in self._session_index:
                self._session_index[session_id] = []
            self._session_index[session_id].append(msg_id)
            logger.debug(f"Message created in memory: {msg_id}")

        return message

    async def get(self, message_id: str) -> Optional[MessageDocument]:
        """获取消息"""
        if self.is_persistent:
            doc = await self._db.session_messages.find_one({"_id": message_id})
            return MessageDocument.from_dict(doc) if doc else None
        else:
            return self._memory_store.get(message_id)

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
        order: str = "asc",
    ) -> List[MessageDocument]:
        """按会话列出消息"""
        sort_order = 1 if order == "asc" else -1

        if self.is_persistent:
            cursor = (
                self._db.session_messages
                .find({"session_id": session_id})
                .sort("created_at", sort_order)
                .skip(offset)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [MessageDocument.from_dict(doc) for doc in docs]
        else:
            msg_ids = self._session_index.get(session_id, [])
            messages = [self._memory_store[mid] for mid in msg_ids if mid in self._memory_store]
            messages.sort(key=lambda x: x.created_at, reverse=(order == "desc"))
            return messages[offset:offset + limit]

    async def count_by_session(self, session_id: str) -> int:
        """统计会话消息数"""
        if self.is_persistent:
            return await self._db.session_messages.count_documents({"session_id": session_id})
        else:
            return len(self._session_index.get(session_id, []))

    async def delete_by_session(self, session_id: str) -> int:
        """删除会话的所有消息"""
        if self.is_persistent:
            result = await self._db.session_messages.delete_many({"session_id": session_id})
            return result.deleted_count
        else:
            msg_ids = self._session_index.pop(session_id, [])
            for mid in msg_ids:
                self._memory_store.pop(mid, None)
            return len(msg_ids)


# =============================================================================
# Agent State Repository
# =============================================================================


class AgentStateRepository(BaseRepository):
    """
    Agent 状态 Repository

    管理 Agent 运行时状态的持久化
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, AgentStateDocument] = {}

    @staticmethod
    def make_agent_id(tenant_id: str, chatbot_id: str) -> str:
        """生成 Agent ID"""
        return f"{tenant_id}:{chatbot_id}"

    async def create_or_update(
        self,
        tenant_id: str,
        chatbot_id: str,
        status: AgentStatus = AgentStatus.READY,
        config_hash: str = "",
        **kwargs,
    ) -> AgentStateDocument:
        """创建或更新 Agent 状态"""
        agent_id = self.make_agent_id(tenant_id, chatbot_id)
        now = datetime.utcnow()

        existing = await self.get(agent_id)
        if existing:
            # 更新
            updates = {
                "status": status.value,
                "config_hash": config_hash,
                "updated_at": now,
                "last_active_at": now,
                **kwargs,
            }
            return await self.update(agent_id, **updates)
        else:
            # 创建
            state = AgentStateDocument(
                agent_id=agent_id,
                tenant_id=tenant_id,
                chatbot_id=chatbot_id,
                status=status,
                config_hash=config_hash,
                created_at=now,
                updated_at=now,
                last_active_at=now,
            )

            if self.is_persistent:
                await self._db.agent_states.insert_one(state.to_dict())
            else:
                self._memory_store[agent_id] = state

            return state

    async def get(self, agent_id: str) -> Optional[AgentStateDocument]:
        """获取 Agent 状态"""
        if self.is_persistent:
            doc = await self._db.agent_states.find_one({"_id": agent_id})
            return AgentStateDocument.from_dict(doc) if doc else None
        else:
            return self._memory_store.get(agent_id)

    async def get_by_tenant_chatbot(
        self,
        tenant_id: str,
        chatbot_id: str,
    ) -> Optional[AgentStateDocument]:
        """按租户和 Chatbot 获取 Agent 状态"""
        agent_id = self.make_agent_id(tenant_id, chatbot_id)
        return await self.get(agent_id)

    async def update(
        self,
        agent_id: str,
        **updates,
    ) -> Optional[AgentStateDocument]:
        """更新 Agent 状态"""
        updates["updated_at"] = datetime.utcnow()

        # 处理枚举值
        if "status" in updates and isinstance(updates["status"], AgentStatus):
            updates["status"] = updates["status"].value

        if self.is_persistent:
            result = await self._db.agent_states.find_one_and_update(
                {"_id": agent_id},
                {"$set": updates},
                return_document=True,
            )
            return AgentStateDocument.from_dict(result) if result else None
        else:
            state = self._memory_store.get(agent_id)
            if state:
                for key, value in updates.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
                return state
            return None

    async def add_session(
        self,
        agent_id: str,
        session_id: str,
    ) -> None:
        """添加会话到 Agent"""
        now = datetime.utcnow()

        if self.is_persistent:
            await self._db.agent_states.update_one(
                {"_id": agent_id},
                {
                    "$addToSet": {"active_sessions": session_id},
                    "$inc": {"total_sessions": 1},
                    "$set": {
                        "updated_at": now,
                        "last_active_at": now,
                        "status": AgentStatus.PROCESSING.value,
                    },
                },
            )
        else:
            state = self._memory_store.get(agent_id)
            if state:
                if session_id not in state.active_sessions:
                    state.active_sessions.append(session_id)
                    state.total_sessions += 1
                state.updated_at = now
                state.last_active_at = now
                state.status = AgentStatus.PROCESSING

    async def remove_session(
        self,
        agent_id: str,
        session_id: str,
    ) -> None:
        """从 Agent 移除会话"""
        now = datetime.utcnow()

        if self.is_persistent:
            # 先移除会话
            await self._db.agent_states.update_one(
                {"_id": agent_id},
                {
                    "$pull": {"active_sessions": session_id},
                    "$set": {"updated_at": now, "last_active_at": now},
                },
            )
            # 检查是否还有活跃会话，更新状态
            state = await self.get(agent_id)
            if state and len(state.active_sessions) == 0:
                await self.update(agent_id, status=AgentStatus.IDLE.value)
        else:
            state = self._memory_store.get(agent_id)
            if state:
                if session_id in state.active_sessions:
                    state.active_sessions.remove(session_id)
                state.updated_at = now
                state.last_active_at = now
                if len(state.active_sessions) == 0:
                    state.status = AgentStatus.IDLE

    async def increment_message_count(
        self,
        agent_id: str,
        increment: int = 1,
    ) -> None:
        """增加消息计数"""
        now = datetime.utcnow()

        if self.is_persistent:
            await self._db.agent_states.update_one(
                {"_id": agent_id},
                {
                    "$inc": {"total_messages": increment},
                    "$set": {"updated_at": now, "last_active_at": now},
                },
            )
        else:
            state = self._memory_store.get(agent_id)
            if state:
                state.total_messages += increment
                state.updated_at = now
                state.last_active_at = now

    async def delete(self, agent_id: str) -> bool:
        """删除 Agent 状态"""
        if self.is_persistent:
            result = await self._db.agent_states.delete_one({"_id": agent_id})
            return result.deleted_count > 0
        else:
            return self._memory_store.pop(agent_id, None) is not None

    async def list_by_status(
        self,
        status: AgentStatus,
        limit: int = 100,
    ) -> List[AgentStateDocument]:
        """按状态列出 Agent"""
        if self.is_persistent:
            cursor = (
                self._db.agent_states
                .find({"status": status.value})
                .sort("last_active_at", -1)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [AgentStateDocument.from_dict(doc) for doc in docs]
        else:
            states = [s for s in self._memory_store.values() if s.status == status]
            states.sort(key=lambda x: x.last_active_at, reverse=True)
            return states[:limit]


# =============================================================================
# Audit Log Repository
# =============================================================================


class AuditLogRepository(BaseRepository):
    """
    审计日志 Repository

    记录所有重要操作的审计轨迹
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: List[AuditLogDocument] = []

    async def log(
        self,
        tenant_id: str,
        action: AuditAction,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        chatbot_id: Optional[str] = None,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[dict] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        source: str = "api",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditLogDocument:
        """记录审计日志"""
        log_id = generate_id()
        audit_log = AuditLogDocument(
            log_id=log_id,
            tenant_id=tenant_id,
            action=action,
            session_id=session_id,
            agent_id=agent_id,
            chatbot_id=chatbot_id,
            message_id=message_id,
            correlation_id=correlation_id,
            request_id=request_id,
            details=details or {},
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
            source=source,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=datetime.utcnow(),
        )

        if self.is_persistent:
            await self._db.audit_logs.insert_one(audit_log.to_dict())
            logger.debug(f"Audit log created: {action.value}")
        else:
            self._memory_store.append(audit_log)
            # 内存模式下限制日志数量
            if len(self._memory_store) > 10000:
                self._memory_store = self._memory_store[-5000:]

        return audit_log

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogDocument]:
        """按会话列出审计日志"""
        if self.is_persistent:
            cursor = (
                self._db.audit_logs
                .find({"session_id": session_id})
                .sort("created_at", -1)
                .skip(offset)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [AuditLogDocument.from_dict(doc) for doc in docs]
        else:
            logs = [log for log in self._memory_store if log.session_id == session_id]
            logs.sort(key=lambda x: x.created_at, reverse=True)
            return logs[offset:offset + limit]

    async def list_by_tenant(
        self,
        tenant_id: str,
        action: Optional[AuditAction] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogDocument]:
        """按租户列出审计日志"""
        query = {"tenant_id": tenant_id}
        if action:
            query["action"] = action.value

        if self.is_persistent:
            cursor = (
                self._db.audit_logs
                .find(query)
                .sort("created_at", -1)
                .skip(offset)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [AuditLogDocument.from_dict(doc) for doc in docs]
        else:
            logs = [
                log for log in self._memory_store
                if log.tenant_id == tenant_id
                and (action is None or log.action == action)
            ]
            logs.sort(key=lambda x: x.created_at, reverse=True)
            return logs[offset:offset + limit]

    async def list_by_correlation(
        self,
        correlation_id: str,
        limit: int = 100,
    ) -> List[AuditLogDocument]:
        """按关联 ID 列出审计日志"""
        if self.is_persistent:
            cursor = (
                self._db.audit_logs
                .find({"correlation_id": correlation_id})
                .sort("created_at", 1)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [AuditLogDocument.from_dict(doc) for doc in docs]
        else:
            logs = [log for log in self._memory_store if log.correlation_id == correlation_id]
            logs.sort(key=lambda x: x.created_at)
            return logs[:limit]


# =============================================================================
# Repository Manager
# =============================================================================


class RepositoryManager:
    """
    Repository 管理器

    统一管理所有 Repository 实例
    """

    def __init__(self, db: Database | None):
        self._db = db
        self._sessions = SessionRepository(db)
        self._messages = MessageRepository(db)
        self._agent_states = AgentStateRepository(db)
        self._audit_logs = AuditLogRepository(db)

        logger.info(f"RepositoryManager initialized: persistent={db is not None and db.is_connected}")

    @property
    def sessions(self) -> SessionRepository:
        """会话 Repository"""
        return self._sessions

    @property
    def messages(self) -> MessageRepository:
        """消息 Repository"""
        return self._messages

    @property
    def agent_states(self) -> AgentStateRepository:
        """Agent 状态 Repository"""
        return self._agent_states

    @property
    def audit_logs(self) -> AuditLogRepository:
        """审计日志 Repository"""
        return self._audit_logs

    @property
    def is_persistent(self) -> bool:
        """是否持久化存储"""
        return self._db is not None and self._db.is_connected


def create_repository_manager(db: Database | None) -> RepositoryManager:
    """创建 Repository 管理器"""
    return RepositoryManager(db)
