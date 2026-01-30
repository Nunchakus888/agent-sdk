"""
Repository 层 v3

提供各数据表的 CRUD 操作，封装数据库访问逻辑
5表设计：configs, sessions, messages, events, usages
"""

import logging
import uuid
from abc import ABC
from datetime import datetime
from typing import List, Optional

from api.services.database import Database
from api.models import (
    # v3 核心模型
    SessionDocument,
    MessageDocument,
    MessageState,
    EventDocument,
    EventType,
    EventStatus,
    TokenDocument,
    TokenDetail,
    # 枚举
    MessageRole,
    AgentPhase,
)

logger = logging.getLogger(__name__)


def generate_id() -> str:
    """生成唯一 ID"""
    return uuid.uuid4().hex


# =============================================================================
# 基础 Repository
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
# Session Repository (v3)
# =============================================================================


class SessionRepository(BaseRepository):
    """
    会话 Repository (v3)

    管理会话的创建、查询、更新和关闭
    v3: 字段平铺设计，统计字段在会话关闭时计算
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
        config_hash: Optional[str] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SessionDocument:
        """创建会话"""
        now = datetime.utcnow()
        session = SessionDocument(
            session_id=session_id,
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            customer_id=customer_id,
            config_hash=config_hash,
            title=title,
            source=source,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        if self.is_persistent:
            await self._db.sessions.insert_one(session.to_dict())
        else:
            self._memory_store[session_id] = session

        return session

    async def get(self, session_id: str) -> Optional[SessionDocument]:
        """获取会话"""
        if self.is_persistent:
            doc = await self._db.sessions.find_one({"_id": session_id})
            return SessionDocument.from_dict(doc) if doc else None
        return self._memory_store.get(session_id)

    async def get_or_create(
        self,
        session_id: str,
        tenant_id: str,
        chatbot_id: str,
        **kwargs,
    ) -> tuple[SessionDocument, bool]:
        """获取或创建会话，返回 (session, created)"""
        session = await self.get(session_id)
        if session:
            return session, False
        session = await self.create(session_id, tenant_id, chatbot_id, **kwargs)
        return session, True

    async def update(self, session_id: str, **updates) -> Optional[SessionDocument]:
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

    async def allocate_event_offset(self, session_id: str) -> int:
        """原子操作分配 event offset"""
        if self.is_persistent:
            result = await self._db.sessions.find_one_and_update(
                {"_id": session_id},
                {"$inc": {"event_count": 1}},
                return_document=True,
            )
            return result["event_count"] if result else 0
        else:
            session = self._memory_store.get(session_id)
            if session:
                # 内存模式下模拟 event_count
                count = session.metadata.get("_event_count", 0) + 1
                session.metadata["_event_count"] = count
                return count
            return 0

    async def close(
        self,
        session_id: str,
        message_count: int = 0,
        event_count: int = 0,
        total_tokens: int = 0,
        total_cost: float = 0.0,
    ) -> Optional[SessionDocument]:
        """关闭会话并更新统计"""
        return await self.update(
            session_id,
            closed_at=datetime.utcnow(),
            message_count=message_count,
            event_count=event_count,
            total_tokens=total_tokens,
            total_cost=total_cost,
        )

    async def list_by_tenant(
        self,
        tenant_id: str,
        chatbot_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionDocument]:
        """按租户列出会话"""
        query = {"tenant_id": tenant_id}
        if chatbot_id:
            query["chatbot_id"] = chatbot_id

        if self.is_persistent:
            cursor = self._db.sessions.find(query).sort("created_at", -1).skip(offset).limit(limit)
            docs = await cursor.to_list(length=limit)
            return [SessionDocument.from_dict(doc) for doc in docs]
        else:
            sessions = [
                s for s in self._memory_store.values()
                if s.tenant_id == tenant_id
                and (chatbot_id is None or s.chatbot_id == chatbot_id)
            ]
            sessions.sort(key=lambda x: x.created_at, reverse=True)
            return sessions[offset:offset + limit]


# =============================================================================
# Message Repository (v3)
# =============================================================================


class MessageRepository(BaseRepository):
    """
    消息 Repository (v3)

    v3 变更：移除 tool_calls, tool_call_id, parent_message_id, latency_ms
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, MessageDocument] = {}
        self._session_index: dict[str, List[str]] = {}

    async def create(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        message_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        state: Optional[MessageState] = None,
        metadata: Optional[dict] = None,
    ) -> MessageDocument:
        """创建消息"""
        msg_id = message_id or generate_id()
        message = MessageDocument(
            message_id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            correlation_id=correlation_id,
            state=state,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )

        if self.is_persistent:
            await self._db.messages.insert_one(message.to_dict())
        else:
            self._memory_store[msg_id] = message
            if session_id not in self._session_index:
                self._session_index[session_id] = []
            self._session_index[session_id].append(msg_id)

        return message

    async def get(self, message_id: str) -> Optional[MessageDocument]:
        """获取消息"""
        if self.is_persistent:
            doc = await self._db.messages.find_one({"_id": message_id})
            return MessageDocument.from_dict(doc) if doc else None
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
                self._db.messages
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
            return await self._db.messages.count_documents({"session_id": session_id})
        return len(self._session_index.get(session_id, []))

    async def update_state(
        self,
        message_id: str,
        phase: AgentPhase,
        sop_step: Optional[str] = None,
        context: Optional[dict] = None,
        decision: Optional[dict] = None,
    ) -> bool:
        """更新消息的嵌入状态"""
        state = MessageState(
            phase=phase,
            sop_step=sop_step,
            context=context or {},
            decision=decision,
        )

        if self.is_persistent:
            result = await self._db.messages.update_one(
                {"_id": message_id},
                {"$set": {"state": state.to_dict()}}
            )
            return result.modified_count > 0
        else:
            message = self._memory_store.get(message_id)
            if message:
                message.state = state
                return True
            return False


# =============================================================================
# Event Repository (v3)
# =============================================================================


class EventRepository(BaseRepository):
    """
    事件 Repository (v3)

    v3 变更：增加 offset, action；移除 tokens（独立到 usages 表）
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, EventDocument] = {}
        self._correlation_index: dict[str, List[str]] = {}

    async def create(
        self,
        session_id: str,
        correlation_id: str,
        event_type: EventType,
        offset: int,
        event_id: Optional[str] = None,
        action: Optional[str] = None,
        message_id: Optional[str] = None,
        status: EventStatus = EventStatus.STARTED,
        duration_ms: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> EventDocument:
        """创建事件"""
        eid = event_id or generate_id()
        event = EventDocument(
            event_id=eid,
            session_id=session_id,
            correlation_id=correlation_id,
            offset=offset,
            message_id=message_id,
            event_type=event_type,
            action=action,
            status=status,
            duration_ms=duration_ms,
            start_time=start_time or datetime.utcnow(),
            end_time=end_time,
            input_data=input_data,
            output_data=output_data,
            error=error,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )

        if self.is_persistent:
            await self._db.events.insert_one(event.to_dict())
        else:
            self._memory_store[eid] = event
            if correlation_id not in self._correlation_index:
                self._correlation_index[correlation_id] = []
            self._correlation_index[correlation_id].append(eid)

        return event

    async def get(self, event_id: str) -> Optional[EventDocument]:
        """获取事件"""
        if self.is_persistent:
            doc = await self._db.events.find_one({"_id": event_id})
            return EventDocument.from_dict(doc) if doc else None
        return self._memory_store.get(event_id)

    async def complete(
        self,
        event_id: str,
        output_data: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> Optional[EventDocument]:
        """完成事件"""
        now = datetime.utcnow()
        event = await self.get(event_id)
        if not event:
            return None

        duration_ms = None
        if event.start_time:
            duration_ms = int((now - event.start_time).total_seconds() * 1000)

        updates = {
            "status": EventStatus.FAILED.value if error else EventStatus.COMPLETED.value,
            "end_time": now,
            "duration_ms": duration_ms,
        }
        if output_data:
            updates["output_data"] = output_data
        if error:
            updates["error"] = error

        if self.is_persistent:
            result = await self._db.events.find_one_and_update(
                {"_id": event_id},
                {"$set": updates},
                return_document=True,
            )
            return EventDocument.from_dict(result) if result else None
        else:
            for key, value in updates.items():
                if hasattr(event, key):
                    setattr(event, key, value)
            return event

    async def list_by_session(
        self,
        session_id: str,
        event_type: Optional[EventType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[EventDocument]:
        """按会话列出事件（按 offset 排序）"""
        query = {"session_id": session_id}
        if event_type:
            query["event_type"] = event_type.value

        if self.is_persistent:
            cursor = (
                self._db.events
                .find(query)
                .sort("offset", 1)
                .skip(offset)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [EventDocument.from_dict(doc) for doc in docs]
        else:
            events = [
                e for e in self._memory_store.values()
                if e.session_id == session_id
                and (event_type is None or e.event_type == event_type)
            ]
            events.sort(key=lambda x: x.offset)
            return events[offset:offset + limit]

    async def list_by_correlation(
        self,
        correlation_id: str,
        limit: int = 100,
    ) -> List[EventDocument]:
        """按 correlation_id 列出事件（全链路追踪）"""
        if self.is_persistent:
            cursor = (
                self._db.events
                .find({"correlation_id": correlation_id})
                .sort("offset", 1)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [EventDocument.from_dict(doc) for doc in docs]
        else:
            event_ids = self._correlation_index.get(correlation_id, [])
            events = [self._memory_store[eid] for eid in event_ids if eid in self._memory_store]
            events.sort(key=lambda x: x.offset)
            return events[:limit]

    async def count_by_session(self, session_id: str) -> int:
        """统计会话事件数"""
        if self.is_persistent:
            return await self._db.events.count_documents({"session_id": session_id})
        return len([e for e in self._memory_store.values() if e.session_id == session_id])


# =============================================================================
# Usage Repository (v3 新增)
# =============================================================================


class UsageRepository(BaseRepository):
    """
    Token 消耗 Repository (v3)

    独立存储 Token 消耗，支持明细和汇总
    一个 correlation_id 对应一条记录
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, TokenDocument] = {}
        self._correlation_index: dict[str, str] = {}

    async def create(
        self,
        session_id: str,
        correlation_id: str,
        token_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> TokenDocument:
        """创建 Token 记录"""
        tid = token_id or generate_id()
        token_doc = TokenDocument(
            token_id=tid,
            session_id=session_id,
            correlation_id=correlation_id,
            message_id=message_id,
            details=[],
            summary=None,
            is_finalized=False,
            created_at=datetime.utcnow(),
        )

        if self.is_persistent:
            await self._db.usages.insert_one(token_doc.to_dict())
        else:
            self._memory_store[tid] = token_doc
            self._correlation_index[correlation_id] = tid

        return token_doc

    async def get(self, token_id: str) -> Optional[TokenDocument]:
        """获取 Token 记录"""
        if self.is_persistent:
            doc = await self._db.usages.find_one({"_id": token_id})
            return TokenDocument.from_dict(doc) if doc else None
        return self._memory_store.get(token_id)

    async def get_by_correlation(self, correlation_id: str) -> Optional[TokenDocument]:
        """按 correlation_id 获取 Token 记录"""
        if self.is_persistent:
            doc = await self._db.usages.find_one({"correlation_id": correlation_id})
            return TokenDocument.from_dict(doc) if doc else None
        else:
            tid = self._correlation_index.get(correlation_id)
            return self._memory_store.get(tid) if tid else None

    async def add_detail(self, token_id: str, detail: TokenDetail) -> bool:
        """添加 Token 明细"""
        if self.is_persistent:
            result = await self._db.usages.update_one(
                {"_id": token_id},
                {"$push": {"details": detail.to_dict()}}
            )
            return result.modified_count > 0
        else:
            token_doc = self._memory_store.get(token_id)
            if token_doc:
                token_doc.add_detail(detail)
                return True
            return False

    async def finalize(self, token_id: str) -> Optional[TokenDocument]:
        """计算汇总并标记完成"""
        token_doc = await self.get(token_id)
        if not token_doc:
            return None

        # 在内存中计算汇总
        token_doc.finalize()

        if self.is_persistent:
            await self._db.usages.update_one(
                {"_id": token_id},
                {"$set": {
                    "summary": token_doc.summary.to_dict() if token_doc.summary else None,
                    "is_finalized": True,
                    "finalized_at": token_doc.finalized_at,
                }}
            )

        return token_doc

    async def get_session_stats(self, session_id: str) -> dict:
        """获取会话的 Token 统计"""
        if self.is_persistent:
            pipeline = [
                {"$match": {"session_id": session_id, "is_finalized": True}},
                {"$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$summary.total_tokens"},
                    "total_cost": {"$sum": "$summary.total_cost"},
                    "request_count": {"$sum": 1},
                }}
            ]
            result = await self._db.usages.aggregate(pipeline).to_list(1)
            if result:
                return {
                    "total_tokens": result[0].get("total_tokens", 0),
                    "total_cost": result[0].get("total_cost", 0.0),
                    "request_count": result[0].get("request_count", 0),
                }
        else:
            tokens = [
                t for t in self._memory_store.values()
                if t.session_id == session_id and t.is_finalized and t.summary
            ]
            return {
                "total_tokens": sum(t.summary.total_tokens for t in tokens),
                "total_cost": sum(t.summary.total_cost for t in tokens),
                "request_count": len(tokens),
            }
        return {"total_tokens": 0, "total_cost": 0.0, "request_count": 0}


# =============================================================================
# Repository Manager (v3)
# =============================================================================


class RepositoryManager:
    """
    Repository 管理器 (v3)

    统一管理所有 Repository 实例
    5表设计：configs, sessions, messages, events, usages
    """

    def __init__(self, db: Database | None):
        self._db = db
        self._sessions = SessionRepository(db)
        self._messages = MessageRepository(db)
        self._events = EventRepository(db)
        self._usages = UsageRepository(db)

        logger.info(f"RepositoryManager v3 initialized: persistent={db is not None and db.is_connected}")

    @property
    def sessions(self) -> SessionRepository:
        """会话 Repository"""
        return self._sessions

    @property
    def messages(self) -> MessageRepository:
        """消息 Repository"""
        return self._messages

    @property
    def events(self) -> EventRepository:
        """事件 Repository"""
        return self._events

    @property
    def usages(self) -> UsageRepository:
        """Token 消耗 Repository"""
        return self._usages

    @property
    def is_persistent(self) -> bool:
        """是否持久化存储"""
        return self._db is not None and self._db.is_connected


def create_repository_manager(db: Database | None) -> RepositoryManager:
    """创建 Repository 管理器"""
    return RepositoryManager(db)
