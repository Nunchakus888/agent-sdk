"""
Repository 层 V2

提供各数据表的 CRUD 操作，封装数据库访问逻辑
6表设计：configs, sessions, messages, tool_calls, usages, timers
"""

import logging
import uuid
from abc import ABC
from datetime import timedelta
from typing import List, Optional, TYPE_CHECKING

from api.services.database import Database
from api.utils.datetime import utc_now

if TYPE_CHECKING:
    from api.models import (
        ConfigDocumentV2,
        SessionDocumentV2,
        MessageDocumentV2,
        ToolCallDocumentV2,
        UsageDocumentV2,
        TimerDocumentV2,
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
# Config Repository (V2)
# =============================================================================


class ConfigRepository(BaseRepository):
    """配置缓存 Repository - _id=chatbot_id, config_hash 用于失效检测, 查询自动 access_count+1"""

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, "ConfigDocumentV2"] = {}

    async def get(
        self,
        chatbot_id: str,
        tenant_id: str,
        expected_hash: Optional[str] = None,
    ) -> Optional["ConfigDocumentV2"]:
        """获取配置（自动 access_count+1），expected_hash 不匹配时返回 None"""
        from api.models import ConfigDocumentV2

        if self.is_persistent:
            doc = await self._db.configs.find_one_and_update(
                {"_id": chatbot_id, "tenant_id": tenant_id},
                {"$inc": {"access_count": 1}, "$set": {"updated_at": utc_now()}},
                return_document=True,
            )
            if not doc:
                return None
            config = ConfigDocumentV2.from_dict(doc)
            if expected_hash and config.config_hash != expected_hash:
                return None
            return config
        else:
            config = self._memory_store.get(chatbot_id)
            if config:
                if expected_hash and config.config_hash != expected_hash:
                    return None
                config.access_count += 1
                config.updated_at = utc_now()
            return config

    async def upsert(
        self,
        chatbot_id: str,
        tenant_id: str,
        config_hash: str,
        raw_config: dict,
        parsed_config: dict,
    ) -> "ConfigDocumentV2":
        """保存或更新配置（保留 created_at, 累加 access_count）"""
        from api.models import ConfigDocumentV2

        now = utc_now()

        if self.is_persistent:
            await self._db.configs.update_one(
                {"_id": chatbot_id},
                {
                    "$set": {
                        "tenant_id": tenant_id,
                        "config_hash": config_hash,
                        "raw_config": raw_config,
                        "parsed_config": parsed_config,
                        "updated_at": now,
                    },
                    "$inc": {"access_count": 1},
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            logger.debug(f"Config upserted: {chatbot_id}, hash={config_hash[:12]}")

        doc = ConfigDocumentV2(
            chatbot_id=chatbot_id,
            tenant_id=tenant_id,
            config_hash=config_hash,
            raw_config=raw_config,
            parsed_config=parsed_config,
            created_at=now,
            updated_at=now,
            access_count=1,
        )

        if not self.is_persistent:
            existing = self._memory_store.get(chatbot_id)
            if existing:
                doc.created_at = existing.created_at
                doc.access_count = existing.access_count + 1
            self._memory_store[chatbot_id] = doc

        return doc

    async def invalidate(self, chatbot_id: str) -> bool:
        """删除配置缓存"""
        if self.is_persistent:
            result = await self._db.configs.delete_one({"_id": chatbot_id})
            return result.deleted_count > 0
        return self._memory_store.pop(chatbot_id, None) is not None

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List["ConfigDocumentV2"]:
        """列出所有配置缓存"""
        from api.models import ConfigDocumentV2

        if self.is_persistent:
            cursor = (
                self._db.configs
                .find()
                .sort("updated_at", -1)
                .skip(offset)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [ConfigDocumentV2.from_dict(doc) for doc in docs]
        else:
            configs = sorted(
                self._memory_store.values(),
                key=lambda x: x.updated_at,
                reverse=True,
            )
            return configs[offset:offset + limit]

    async def count(self) -> int:
        """统计配置缓存数"""
        if self.is_persistent:
            return await self._db.configs.count_documents({})
        return len(self._memory_store)

    async def get_by_chatbot_id(self, chatbot_id: str) -> Optional["ConfigDocumentV2"]:
        """按 chatbot_id 获取配置（不校验 tenant_id 和 hash）"""
        from api.models import ConfigDocumentV2

        if self.is_persistent:
            doc = await self._db.configs.find_one({"_id": chatbot_id})
            return ConfigDocumentV2.from_dict(doc) if doc else None
        return self._memory_store.get(chatbot_id)


# =============================================================================
# Session Repository (V2)
# =============================================================================


class SessionRepository(BaseRepository):
    """
    会话 Repository (V2)

    管理会话的创建、查询、更新和关闭
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, "SessionDocumentV2"] = {}

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
    ) -> "SessionDocumentV2":
        """创建会话"""
        from api.models import SessionDocumentV2

        now = utc_now()
        session = SessionDocumentV2(
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

    async def get(self, session_id: str) -> Optional["SessionDocumentV2"]:
        """获取会话"""
        from api.models import SessionDocumentV2

        if self.is_persistent:
            doc = await self._db.sessions.find_one({"_id": session_id})
            return SessionDocumentV2.from_dict(doc) if doc else None
        return self._memory_store.get(session_id)

    async def get_or_create(
        self,
        session_id: str,
        tenant_id: str,
        chatbot_id: str,
        **kwargs,
    ) -> "SessionDocumentV2":
        """获取或创建会话"""
        session = await self.get(session_id)
        if session:
            logger.info(f"Session already exists: {session_id}")
            # update updated_at
            await self.update(session_id, updated_at=utc_now())
            return session
        return await self.create(session_id, tenant_id, chatbot_id, **kwargs)

    async def update(self, session_id: str, **updates) -> Optional["SessionDocumentV2"]:
        """更新会话"""
        from api.models import SessionDocumentV2

        updates["updated_at"] = utc_now()

        if self.is_persistent:
            result = await self._db.sessions.find_one_and_update(
                {"_id": session_id},
                {"$set": updates},
                return_document=True,
            )
            return SessionDocumentV2.from_dict(result) if result else None
        else:
            session = self._memory_store.get(session_id)
            if session:
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                return session
            return None

    async def close(self, session_id: str) -> Optional["SessionDocumentV2"]:
        """关闭会话"""
        return await self.update(session_id, closed_at=utc_now())

    async def allocate_event_offset(self, session_id: str) -> int:
        """
        原子性分配事件 offset
        
        通过 $inc 原子操作递增 event_count，返回新值作为 offset
        """
        if self.is_persistent:
            from pymongo import ReturnDocument
            result = await self._db.sessions.find_one_and_update(
                {"_id": session_id},
                {"$inc": {"event_count": 1}},
                return_document=ReturnDocument.AFTER,
            )
            return result["event_count"] if result else 0
        else:
            session = self._memory_store.get(session_id)
            if session:
                if not hasattr(session, "event_count"):
                    session.event_count = 0
                session.event_count += 1
                return session.event_count
            return 0

    async def count(
        self,
        tenant_id: Optional[str] = None,
        chatbot_id: Optional[str] = None,
    ) -> int:
        """统计会话数"""
        if self.is_persistent:
            query = {}
            if tenant_id:
                query["tenant_id"] = tenant_id
            if chatbot_id:
                query["chatbot_id"] = chatbot_id
            return await self._db.sessions.count_documents(query)
        else:
            sessions = list(self._memory_store.values())
            if tenant_id:
                sessions = [s for s in sessions if s.tenant_id == tenant_id]
            if chatbot_id:
                sessions = [s for s in sessions if s.chatbot_id == chatbot_id]
            return len(sessions)

    async def delete(self, session_id: str) -> bool:
        """删除会话"""
        if self.is_persistent:
            result = await self._db.sessions.delete_one({"_id": session_id})
            return result.deleted_count > 0
        return self._memory_store.pop(session_id, None) is not None

    async def list_by_tenant(
        self,
        tenant_id: str,
        chatbot_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List["SessionDocumentV2"]:
        """按租户列出会话"""
        from api.models import SessionDocumentV2

        if self.is_persistent:
            query = {}
            if tenant_id:
                query["tenant_id"] = tenant_id
            if chatbot_id:
                query["chatbot_id"] = chatbot_id
            cursor = self._db.sessions.find(query).sort("created_at", -1).skip(offset).limit(limit)
            docs = await cursor.to_list(length=limit)
            return [SessionDocumentV2.from_dict(doc) for doc in docs]
        else:
            if tenant_id:
                sessions = [
                    s for s in self._memory_store.values()
                    if s.tenant_id == tenant_id
                    and (chatbot_id is None or s.chatbot_id == chatbot_id)
                ]
            else:
                sessions = [
                    s for s in self._memory_store.values()
                    if (chatbot_id is None or s.chatbot_id == chatbot_id)
                ]
            sessions.sort(key=lambda x: x.created_at, reverse=True)
            return sessions[offset:offset + limit]


# =============================================================================
# Message Repository (V2)
# =============================================================================


class MessageRepository(BaseRepository):
    """
    消息 Repository (V2)

    简化设计：移除 MessageState, metadata
    每条消息自动分配会话内单调递增的 offset
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, "MessageDocumentV2"] = {}
        self._session_index: dict[str, list[str]] = {}
        self._session_offsets: dict[str, int] = {}  # memory 模式的 offset 计数器

    async def _next_offset(self, session_id: str) -> int:
        """分配会话内下一个 offset（原子递增）"""
        if self.is_persistent:
            from pymongo import ReturnDocument
            result = await self._db.sessions.find_one_and_update(
                {"_id": session_id},
                {"$inc": {"event_count": 1}},
                return_document=ReturnDocument.AFTER,
            )
            return (result["event_count"] - 1) if result else 0
        else:
            current = self._session_offsets.get(session_id, 0)
            self._session_offsets[session_id] = current + 1
            return current

    async def create(
        self,
        session_id: str,
        role: str,
        content: str,
        correlation_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> "MessageDocumentV2":
        """创建消息（自动分配 offset）"""
        from api.models import MessageDocumentV2

        mid = message_id or generate_id()
        offset = await self._next_offset(session_id)
        doc = MessageDocumentV2(
            message_id=mid,
            session_id=session_id,
            role=role,
            content=content,
            correlation_id=correlation_id,
            offset=offset,
            created_at=utc_now(),
        )

        if self.is_persistent:
            logger.info(f"MessageRepository.create: inserting to DB, session={session_id}, role={role}, offset={offset}")
            await self._db.messages.insert_one(doc.to_dict())
            logger.info(f"MessageRepository.create: inserted, message_id={mid}")
        else:
            logger.info(f"MessageRepository.create: memory mode, session={session_id}, role={role}, offset={offset}")
            self._memory_store[mid] = doc
            if session_id not in self._session_index:
                self._session_index[session_id] = []
            self._session_index[session_id].append(mid)

        return doc

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        order: str = "asc",
        min_offset: int = 0,
    ) -> list["MessageDocumentV2"]:
        """按会话列出消息，支持 offset 过滤"""
        from api.models import MessageDocumentV2

        sort_order = 1 if order == "asc" else -1

        if self.is_persistent:
            query: dict = {"session_id": session_id}
            if min_offset > 0:
                query["offset"] = {"$gte": min_offset}
            cursor = (
                self._db.messages
                .find(query)
                .sort("offset", sort_order)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [MessageDocumentV2.from_dict(doc) for doc in docs]
        else:
            ids = self._session_index.get(session_id, [])
            msgs = [self._memory_store[mid] for mid in ids if mid in self._memory_store]
            if min_offset > 0:
                msgs = [m for m in msgs if m.offset >= min_offset]
            msgs.sort(key=lambda x: x.offset, reverse=(order == "desc"))
            return msgs[:limit]

    async def count_by_session(self, session_id: str) -> int:
        """统计会话消息数"""
        if self.is_persistent:
            return await self._db.messages.count_documents({"session_id": session_id})
        return len(self._session_index.get(session_id, []))

    async def delete_from_offset(self, session_id: str, min_offset: int) -> int:
        """删除会话中 offset >= min_offset 的消息"""
        if self.is_persistent:
            result = await self._db.messages.delete_many({
                "session_id": session_id,
                "offset": {"$gte": min_offset},
            })
            return result.deleted_count
        else:
            ids = self._session_index.get(session_id, [])
            msgs = [self._memory_store[mid] for mid in ids if mid in self._memory_store]
            to_delete = [m for m in msgs if m.offset >= min_offset]
            for msg in to_delete:
                self._memory_store.pop(msg.message_id, None)
            self._session_index[session_id] = [
                mid for mid in ids if mid in self._memory_store
            ]
            return len(to_delete)


# =============================================================================
# ToolCall Repository (V2)
# =============================================================================


class ToolCallRepository(BaseRepository):
    """
    工具调用 Repository (V2)

    简化设计：只记录工具调用（替代 events 表）
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, "ToolCallDocumentV2"] = {}
        self._session_index: dict[str, list[str]] = {}

    async def create(
        self,
        tool_call_id: str,
        session_id: str,
        correlation_id: str,
        offset: int,
        tool_name: str,
        arguments: dict,
        result: Optional[str] = None,
        is_error: bool = False,
        duration_ms: int = 0,
    ) -> "ToolCallDocumentV2":
        """创建工具调用记录"""
        from api.models import ToolCallDocumentV2

        doc = ToolCallDocumentV2(
            tool_call_id=tool_call_id,
            session_id=session_id,
            correlation_id=correlation_id,
            offset=offset,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            is_error=is_error,
            duration_ms=duration_ms,
            created_at=utc_now(),
        )

        if self.is_persistent:
            await self._db.tool_calls.insert_one(doc.to_dict())
        else:
            self._memory_store[tool_call_id] = doc
            if session_id not in self._session_index:
                self._session_index[session_id] = []
            self._session_index[session_id].append(tool_call_id)

        return doc

    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list["ToolCallDocumentV2"]:
        """按会话列出工具调用"""
        from api.models import ToolCallDocumentV2

        if self.is_persistent:
            cursor = (
                self._db.tool_calls
                .find({"session_id": session_id})
                .sort("offset", 1)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [ToolCallDocumentV2.from_dict(doc) for doc in docs]
        else:
            ids = self._session_index.get(session_id, [])
            calls = [self._memory_store[tid] for tid in ids if tid in self._memory_store]
            calls.sort(key=lambda x: x.offset)
            return calls[:limit]

    async def list_by_correlation(
        self,
        correlation_id: str,
        limit: int = 100,
    ) -> list["ToolCallDocumentV2"]:
        """按 correlation_id 列出工具调用"""
        from api.models import ToolCallDocumentV2

        if self.is_persistent:
            cursor = (
                self._db.tool_calls
                .find({"correlation_id": correlation_id})
                .sort("offset", 1)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [ToolCallDocumentV2.from_dict(doc) for doc in docs]
        else:
            calls = [
                tc for tc in self._memory_store.values()
                if tc.correlation_id == correlation_id
            ]
            calls.sort(key=lambda x: x.offset)
            return calls[:limit]


# =============================================================================
# Usage Repository (V2)
# =============================================================================


class UsageRepository(BaseRepository):
    """
    Token 消耗 Repository (V2)

    简化设计：扁平化结构，单次写入
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, "UsageDocumentV2"] = {}
        self._correlation_index: dict[str, str] = {}

    async def create(
        self,
        session_id: str,
        correlation_id: str,
        total_input_tokens: int = 0,
        cached_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_tokens: int = 0,
        total_cost: float = 0.0,
        by_model: Optional[dict] = None,
        usage_id: Optional[str] = None,
    ) -> "UsageDocumentV2":
        """创建 Token 消耗记录"""
        from api.models import UsageDocumentV2

        uid = usage_id or generate_id()
        doc = UsageDocumentV2(
            usage_id=uid,
            session_id=session_id,
            correlation_id=correlation_id,
            total_input_tokens=total_input_tokens,
            cached_input_tokens=cached_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            total_cost=total_cost,
            by_model=by_model or {},
            created_at=utc_now(),
        )

        if self.is_persistent:
            logger.info(f"UsageRepository.create: inserting to DB, session={session_id}, tokens={total_tokens}")
            await self._db.usages.insert_one(doc.to_dict())
            logger.info(f"UsageRepository.create: inserted, usage_id={uid}")
        else:
            logger.info(f"UsageRepository.create: memory mode, session={session_id}, tokens={total_tokens}")
            self._memory_store[uid] = doc
            self._correlation_index[correlation_id] = uid

        return doc

    async def get_by_correlation(self, correlation_id: str) -> Optional["UsageDocumentV2"]:
        """按 correlation_id 获取记录"""
        from api.models import UsageDocumentV2

        if self.is_persistent:
            doc = await self._db.usages.find_one({"correlation_id": correlation_id})
            return UsageDocumentV2.from_dict(doc) if doc else None
        else:
            uid = self._correlation_index.get(correlation_id)
            return self._memory_store.get(uid) if uid else None

    async def get_session_stats(self, session_id: str) -> dict:
        """获取会话的 Token 统计"""
        if self.is_persistent:
            pipeline = [
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$total_cost"},
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
            usages = [u for u in self._memory_store.values() if u.session_id == session_id]
            return {
                "total_tokens": sum(u.total_tokens for u in usages),
                "total_cost": sum(u.total_cost for u in usages),
                "request_count": len(usages),
            }
        return {"total_tokens": 0, "total_cost": 0.0, "request_count": 0}


# =============================================================================
# Timer Repository (V2)
# =============================================================================


class TimerRepository(BaseRepository):
    """
    定时器 Repository (V2)

    独立表设计，支持每个 Session 多个 Timer
    """

    def __init__(self, db: Database | None):
        super().__init__(db)
        self._memory_store: dict[str, "TimerDocumentV2"] = {}
        self._session_index: dict[str, list[str]] = {}

    async def create(
        self,
        session_id: str,
        timer_id: str,
        delay_seconds: int = 300,
        max_triggers: int = 1,
        tool_name: str = "",
        tool_params: Optional[dict] = None,
        message: Optional[str] = None,
        timer_instance_id: Optional[str] = None,
    ) -> "TimerDocumentV2":
        """创建定时器"""
        from api.models import TimerDocumentV2

        tid = timer_instance_id or generate_id()
        now = utc_now()
        doc = TimerDocumentV2(
            timer_instance_id=tid,
            session_id=session_id,
            timer_id=timer_id,
            status="pending",
            trigger_count=0,
            created_at=now,
            next_trigger_at=now + timedelta(seconds=delay_seconds),
            delay_seconds=delay_seconds,
            max_triggers=max_triggers,
            tool_name=tool_name,
            tool_params=tool_params or {},
            message=message,
        )

        if self.is_persistent:
            await self._db.timers.insert_one(doc.to_dict())
        else:
            self._memory_store[tid] = doc
            if session_id not in self._session_index:
                self._session_index[session_id] = []
            self._session_index[session_id].append(tid)

        return doc

    async def get(self, timer_instance_id: str) -> Optional["TimerDocumentV2"]:
        """获取定时器"""
        from api.models import TimerDocumentV2

        if self.is_persistent:
            doc = await self._db.timers.find_one({"_id": timer_instance_id})
            return TimerDocumentV2.from_dict(doc) if doc else None
        return self._memory_store.get(timer_instance_id)

    async def update(self, timer_instance_id: str, **updates) -> Optional["TimerDocumentV2"]:
        """更新定时器"""
        from api.models import TimerDocumentV2

        if self.is_persistent:
            result = await self._db.timers.find_one_and_update(
                {"_id": timer_instance_id},
                {"$set": updates},
                return_document=True,
            )
            return TimerDocumentV2.from_dict(result) if result else None
        else:
            timer = self._memory_store.get(timer_instance_id)
            if timer:
                for key, value in updates.items():
                    if hasattr(timer, key):
                        setattr(timer, key, value)
                return timer
            return None

    async def find_pending(self, limit: int = 100) -> list["TimerDocumentV2"]:
        """查找待触发的定时器"""
        from api.models import TimerDocumentV2

        now = utc_now()

        if self.is_persistent:
            cursor = (
                self._db.timers
                .find({"status": "pending", "next_trigger_at": {"$lt": now}})
                .sort("next_trigger_at", 1)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
            return [TimerDocumentV2.from_dict(doc) for doc in docs]
        else:
            timers = [
                t for t in self._memory_store.values()
                if t.status == "pending" and t.next_trigger_at and t.next_trigger_at < now
            ]
            timers.sort(key=lambda x: x.next_trigger_at)
            return timers[:limit]

    async def reset_by_session(self, session_id: str) -> int:
        """重置会话的所有定时器"""
        now = utc_now()

        if self.is_persistent:
            # 获取会话的所有 pending 定时器
            cursor = self._db.timers.find({"session_id": session_id, "status": "pending"})
            timers = await cursor.to_list(length=100)

            count = 0
            for timer in timers:
                delay = timer.get("delay_seconds", 300)
                await self._db.timers.update_one(
                    {"_id": timer["_id"]},
                    {"$set": {"next_trigger_at": now + timedelta(seconds=delay)}}
                )
                count += 1
            return count
        else:
            ids = self._session_index.get(session_id, [])
            count = 0
            for tid in ids:
                timer = self._memory_store.get(tid)
                if timer and timer.status == "pending":
                    timer.next_trigger_at = now + timedelta(seconds=timer.delay_seconds)
                    count += 1
            return count

    async def cancel_by_session(self, session_id: str) -> int:
        """取消会话的所有定时器"""
        if self.is_persistent:
            result = await self._db.timers.update_many(
                {"session_id": session_id, "status": "pending"},
                {"$set": {"status": "cancelled"}}
            )
            return result.modified_count
        else:
            ids = self._session_index.get(session_id, [])
            count = 0
            for tid in ids:
                timer = self._memory_store.get(tid)
                if timer and timer.status == "pending":
                    timer.status = "cancelled"
                    count += 1
            return count


# =============================================================================
# Repository Manager (V2)
# =============================================================================


class RepositoryManager:
    """
    Repository 管理器 (V2)

    6表设计：configs, sessions, messages, tool_calls, usages, timers
    """

    def __init__(self, db: Database | None):
        self._db = db
        self._configs = ConfigRepository(db)
        self._sessions = SessionRepository(db)
        self._messages = MessageRepository(db)
        self._tool_calls = ToolCallRepository(db)
        self._usages = UsageRepository(db)
        self._timers = TimerRepository(db)

        logger.info(f"RepositoryManager V2 initialized: persistent={db is not None and db.is_connected}")

    @property
    def configs(self) -> ConfigRepository:
        """配置 Repository"""
        return self._configs

    @property
    def sessions(self) -> SessionRepository:
        """会话 Repository"""
        return self._sessions

    @property
    def messages(self) -> MessageRepository:
        """消息 Repository"""
        return self._messages

    @property
    def tool_calls(self) -> ToolCallRepository:
        """工具调用 Repository"""
        return self._tool_calls

    @property
    def usages(self) -> UsageRepository:
        """Token 消耗 Repository"""
        return self._usages

    @property
    def timers(self) -> TimerRepository:
        """定时器 Repository"""
        return self._timers

    @property
    def is_persistent(self) -> bool:
        """是否持久化存储"""
        return self._db is not None and self._db.is_connected


def create_repository_manager(db: Database | None) -> RepositoryManager:
    """创建 Repository 管理器"""
    return RepositoryManager(db)
