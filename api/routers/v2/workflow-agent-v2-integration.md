# WorkflowAgentV2 集成实现说明

## 0. 文件结构

```
api/
├── services/
│   └── v2/                         # V2 服务层
│       ├── __init__.py             # 模块导出
│       ├── config_cache.py         # 服务级配置缓存
│       ├── session_context.py      # 会话上下文
│       ├── session_manager.py      # 会话管理器
│       ├── event_collector.py      # 统一事件收集器 ✨
│       └── query_recorder.py       # 统一 DB 写入 ✨
├── routers/
│   └── v2/                         # V2 路由层
│       ├── __init__.py             # 模块导出
│       └── query.py                # V2 查询路由（使用统一模式）
├── container.py                    # 依赖注入
├── main.py                         # 应用入口
tests/
├── test_v2_services.py             # V2 服务单元测试
└── test_v2_query_api.py            # V2 Query API 集成测试（含 EventCollector 测试）
bu_agent_sdk/
└── agent/
    └── workflow_agent_v2.py        # WorkflowAgentV2
```



### 核心特性
- **会话级 Agent**: 每个会话维护独立的 Agent 实例
- **配置缓存**: 服务级配置缓存，支持多 Agent 复用
- **会话级 Timer**: 支持空闲提醒和会话关闭
- **自动回收**: 空闲会话自动清理
- **统一事件收集**: EventCollector 统一处理流式/非流式场景 ✨
- **统一 DB 写入**: QueryRecorder 统一记录 messages/events/usages ✨


## 🏗️ 架构设计


## 0.1 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v2/query` | V2 查询接口 |
| GET | `/api/v2/sessions` | 列出所有会话 |
| GET | `/api/v2/sessions/{id}` | 获取会话信息 |
| DELETE | `/api/v2/sessions/{id}` | 销毁会话 |
| GET | `/api/v2/config-cache/stats` | 配置缓存统计 |

## 0.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `V2_SESSION_IDLE_TIMEOUT` | 1800 | 会话空闲超时（秒） |
| `V2_MAX_SESSIONS` | 10000 | 最大会话数 |

## 1. 架构概述

### 1.1 核心问题

WorkflowAgentV2 的生命周期维度是什么？**答案：会话级别 (Session-scoped)**

原因：
- WorkflowAgentV2 内部的 `Agent` 维护对话历史 (`messages`)
- 不同会话的对话历史必须隔离
- 每个会话需要独立的上下文管理

### 1.2 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Query API   │  │ Session API │  │ Agent API   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
┌─────────▼────────────────▼────────────────▼─────────────────────┐
│                     Service Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   SessionManager                             ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          ││
│  │  │ Session A   │  │ Session B   │  │ Session C   │          ││
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │          ││
│  │  │ │AgentV2  │ │  │ │AgentV2  │ │  │ │AgentV2  │ │          ││
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │          ││
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │          ││
│  │  │ │ Timer   │ │  │ │ Timer   │ │  │ │ Timer   │ │          ││
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │          ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘          ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   ConfigCache (服务级)                       ││
│  │  config_hash → ParsedConfig (多 Agent 复用)                  ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   LLMService (服务级)                        ││
│  │  复用 LLM 连接                                               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────┐
│                    Repository Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Sessions │ │ Messages │ │ Events   │ │ Usages   │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 生命周期维度

| 组件 | 生命周期 | 说明 |
|------|----------|------|
| ConfigCache | 服务级 | 按 config_hash 缓存，多 Agent 复用 |
| LLMService | 服务级 | 复用 LLM 连接 |
| SessionManager | 服务级 | 管理所有会话 |
| WorkflowAgentV2 | **会话级** | 每个会话独立实例 |
| SessionTimer | 会话级 | 每个会话独立 Timer |

---

## 2. 核心组件设计

### 2.1 ConfigCache（服务级配置缓存）

```python
# api/services/config_cache.py

from dataclasses import dataclass
from typing import Optional
import time

from bu_agent_sdk.tools.actions import WorkflowConfigSchema


@dataclass
class CachedConfig:
    """缓存的配置"""
    config: WorkflowConfigSchema
    config_hash: str
    created_at: float
    access_count: int = 0


class ConfigCache:
    """
    服务级配置缓存

    职责：
    - 按 config_hash 缓存解析后的配置
    - 多个 Agent 实例复用同一配置
    - LRU 淘汰策略
    """

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self._cache: dict[str, CachedConfig] = {}
        self._max_size = max_size
        self._ttl = ttl

    def get(self, config_hash: str) -> Optional[WorkflowConfigSchema]:
        """获取缓存的配置"""
        cached = self._cache.get(config_hash)
        if cached:
            # 检查 TTL
            if time.time() - cached.created_at > self._ttl:
                del self._cache[config_hash]
                return None
            cached.access_count += 1
            return cached.config
        return None

    def set(self, config_hash: str, config: WorkflowConfigSchema):
        """缓存配置"""
        # LRU 淘汰
        if len(self._cache) >= self._max_size:
            self._evict_lru()

        self._cache[config_hash] = CachedConfig(
            config=config,
            config_hash=config_hash,
            created_at=time.time(),
        )

    def _evict_lru(self):
        """淘汰最少使用的配置"""
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k].access_count)
        del self._cache[lru_key]
```

### 2.2 SessionContext（会话上下文）

```python
# api/services/session_context.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import asyncio

from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2
from bu_agent_sdk.tools.actions import WorkflowConfigSchema


@dataclass
class SessionTimer:
    """会话级 Timer"""
    session_id: str
    timeout_seconds: int
    message: str
    max_triggers: int = 3
    trigger_count: int = 0
    task: Optional[asyncio.Task] = None

    def is_exhausted(self) -> bool:
        return self.trigger_count >= self.max_triggers


@dataclass
class SessionContext:
    """
    会话上下文

    封装单个会话的所有状态：
    - WorkflowAgentV2 实例
    - Timer 配置
    - 会话元数据
    """
    session_id: str
    tenant_id: str
    chatbot_id: str
    config_hash: str

    # Agent 实例（会话级）
    agent: WorkflowAgentV2 = field(default=None, repr=False)

    # Timer（会话级）
    timer: Optional[SessionTimer] = None

    # 元数据
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime = field(default_factory=datetime.utcnow)

    def touch(self):
        """更新活跃时间"""
        self.last_active_at = datetime.utcnow()

    @property
    def idle_seconds(self) -> float:
        """空闲时间（秒）"""
        return (datetime.utcnow() - self.last_active_at).total_seconds()
```

### 2.3 SessionManager（会话管理器）

```python
# api/services/session_manager.py

import asyncio
import logging
from typing import Optional, Callable, Awaitable

from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2
from bu_agent_sdk.tools.actions import WorkflowConfigSchema

from api.services.config_cache import ConfigCache
from api.services.session_context import SessionContext, SessionTimer
from api.services.llm_service import LLMService
from api.services.repositories import RepositoryManager

logger = logging.getLogger(__name__)


class SessionManager:
    """
    会话管理器

    职责：
    1. 会话生命周期管理（创建、获取、销毁）
    2. Agent 实例管理（会话级）
    3. Timer 管理（会话级）
    4. 空闲会话回收
    """

    def __init__(
        self,
        config_cache: ConfigCache,
        repos: RepositoryManager,
        idle_timeout: int = 1800,  # 30 分钟
        cleanup_interval: int = 60,
    ):
        self._config_cache = config_cache
        self._repos = repos
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval

        # 会话池：session_id -> SessionContext
        self._sessions: dict[str, SessionContext] = {}

        # 消息发送回调
        self._send_message: Optional[Callable[[str, str], Awaitable[None]]] = None

        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None

    # -------------------------------------------------------------------------
    # 会话管理
    # -------------------------------------------------------------------------

    async def get_or_create(
        self,
        session_id: str,
        tenant_id: str,
        chatbot_id: str,
        config_hash: str,
        config: Optional[WorkflowConfigSchema] = None,
    ) -> SessionContext:
        """
        获取或创建会话上下文

        Args:
            session_id: 会话 ID
            tenant_id: 租户 ID
            chatbot_id: Chatbot ID
            config_hash: 配置哈希
            config: 解析后的配置（可选，未提供时从缓存获取）

        Returns:
            SessionContext 实例
        """
        # 1. 检查现有会话
        if session_id in self._sessions:
            ctx = self._sessions[session_id]
            # 配置变更检测
            if ctx.config_hash != config_hash:
                logger.info(f"Config changed, recreating session: {session_id}")
                await self.destroy(session_id)
            else:
                ctx.touch()
                return ctx

        # 2. 获取配置
        if config is None:
            config = self._config_cache.get(config_hash)
            if config is None:
                raise ValueError(f"Config not found: {config_hash}")

        # 3. 创建 Agent
        llm = LLMService.get_instance().get_decision_llm()
        agent = WorkflowAgentV2(config=config, llm=llm)

        # 4. 加载历史消息
        history = await self._repos.messages.list_by_session(
            session_id=session_id,
            limit=50,
            order="asc",
        )
        if history:
            context = [{"role": m.role.value, "content": m.content} for m in history]
            # 注入历史到 Agent
            await agent.query("", session_id=session_id, context=context)

        # 5. 创建会话上下文
        ctx = SessionContext(
            session_id=session_id,
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            config_hash=config_hash,
            agent=agent,
        )

        # 6. 初始化 Timer（如果配置中有）
        if config.timers:
            timer_config = config.timers[0]  # 使用第一个 Timer
            ctx.timer = SessionTimer(
                session_id=session_id,
                timeout_seconds=timer_config.get("delay_seconds", 300),
                message=timer_config.get("message", "您好，请问还在吗？"),
                max_triggers=timer_config.get("max_triggers", 3),
            )
            self._start_timer(ctx)

        self._sessions[session_id] = ctx
        logger.info(f"Session created: {session_id}")

        return ctx

    async def destroy(self, session_id: str):
        """销毁会话"""
        ctx = self._sessions.pop(session_id, None)
        if ctx:
            # 取消 Timer
            if ctx.timer and ctx.timer.task:
                ctx.timer.task.cancel()
            logger.info(f"Session destroyed: {session_id}")

    # -------------------------------------------------------------------------
    # Timer 管理
    # -------------------------------------------------------------------------

    def _start_timer(self, ctx: SessionContext):
        """启动会话 Timer"""
        if not ctx.timer:
            return

        # 取消现有 Timer
        if ctx.timer.task and not ctx.timer.task.done():
            ctx.timer.task.cancel()

        async def timer_callback():
            try:
                await asyncio.sleep(ctx.timer.timeout_seconds)
                await self._trigger_timer(ctx)
            except asyncio.CancelledError:
                pass

        ctx.timer.task = asyncio.create_task(timer_callback())

    def reset_timer(self, session_id: str):
        """重置会话 Timer（用户活动时调用）"""
        ctx = self._sessions.get(session_id)
        if ctx:
            ctx.touch()
            self._start_timer(ctx)

    async def _trigger_timer(self, ctx: SessionContext):
        """触发 Timer"""
        if not ctx.timer or ctx.timer.is_exhausted():
            return

        ctx.timer.trigger_count += 1
        message = ctx.timer.message

        # 存储消息
        from api.models import MessageRole
        await self._repos.messages.create(
            session_id=ctx.session_id,
            role=MessageRole.ASSISTANT,
            content=message,
        )

        # 发送消息
        if self._send_message:
            await self._send_message(ctx.session_id, message)

        logger.info(f"Timer triggered: {ctx.session_id} ({ctx.timer.trigger_count}/{ctx.timer.max_triggers})")

        # 继续下一轮 Timer（如果未耗尽）
        if not ctx.timer.is_exhausted():
            self._start_timer(ctx)

    # -------------------------------------------------------------------------
    # 清理
    # -------------------------------------------------------------------------

    def start_cleanup(self):
        """启动清理任务"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self):
        """停止清理任务"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._evict_idle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _evict_idle(self):
        """回收空闲会话"""
        to_remove = [
            sid for sid, ctx in self._sessions.items()
            if ctx.idle_seconds > self._idle_timeout
        ]
        for sid in to_remove:
            await self.destroy(sid)
        if to_remove:
            logger.info(f"Evicted {len(to_remove)} idle sessions")
```

---

## 3. Query API 集成

### 3.1 优化后的 Query 流程

```python
# api/routers/query.py

async def query(
    request: QueryRequest,
    session_mgr: SessionManagerDep,
    config_cache: ConfigCacheDep,
    repos: RepositoryManagerDep,
):
    start_time = time.time()
    correlation_id = get_correlation_id()

    # ─────────────────────────────────────────────────────────
    # Phase 1: 准备
    # ─────────────────────────────────────────────────────────

    # 1.1 获取/解析配置（服务级缓存）
    config = config_cache.get(request.config_hash)
    if not config:
        config = await load_and_parse_config(request)
        config_cache.set(request.config_hash, config)

    # 1.2 获取/创建会话上下文（会话级）
    ctx = await session_mgr.get_or_create(
        session_id=request.session_id,
        tenant_id=request.tenant_id,
        chatbot_id=request.chatbot_id,
        config_hash=request.config_hash,
        config=config,
    )

    # ─────────────────────────────────────────────────────────
    # Phase 2: 执行
    # ─────────────────────────────────────────────────────────

    query_start = time.time()
    result = await ctx.agent.query(
        message=request.message,
        session_id=request.session_id,
    )
    query_latency_ms = int((time.time() - query_start) * 1000)

    # ─────────────────────────────────────────────────────────
    # Phase 3: 后处理
    # ─────────────────────────────────────────────────────────

    # 3.1 重置 Timer
    session_mgr.reset_timer(request.session_id)

    # 3.2 后台记录（fire & forget）
    asyncio.create_task(
        record_query(
            repos=repos,
            session_id=request.session_id,
            correlation_id=correlation_id,
            user_message=request.message,
            assistant_message=result,
            query_latency_ms=query_latency_ms,
        )
    )

    return QueryResponse(
        session_id=request.session_id,
        message=result,
        status="success",
    )
```

### 3.2 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Request                             │
│  session_id, tenant_id, chatbot_id, config_hash, message        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ConfigCache (服务级)                         │
│  config_hash → WorkflowConfigSchema                             │
│  缓存命中：直接返回                                               │
│  缓存未命中：加载 → 解析 → 缓存                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SessionManager (服务级)                        │
│  session_id → SessionContext                                    │
│  会话存在：返回现有上下文                                         │
│  会话不存在：创建 Agent → 加载历史 → 初始化 Timer                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 WorkflowAgentV2 (会话级)                         │
│  agent.query(message, session_id)                               │
│  内部 Agent 维护对话历史                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     后处理 (异步)                                │
│  1. 重置 Timer                                                  │
│  2. 存储消息 (messages)                                         │
│  3. 记录事件 (events)                                           │
│  4. 记录用量 (usages)                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. DB 交互设计

### 4.1 表结构

| 表 | 主键 | 说明 |
|---|------|------|
| sessions | session_id | 会话元数据，Timer 配置 |
| messages | message_id | 消息记录 |
| events | event_id | 事件日志（LLM 调用、Tool 执行等） |
| usages | token_id | Token 消耗明细和汇总 |

### 4.2 关联关系

```
sessions (1) ──────< messages (N)
    │                    │
    │                    │ correlation_id
    │                    ▼
    └──────────────< events (N)
                         │
                         │ correlation_id
                         ▼
                    usages (1)
```

### 4.3 Timer 字段（sessions 表）

```python
# SessionDocument 扩展字段
timer_config: Optional[dict] = None      # Timer 配置
timer_status: str = "pending"            # pending | triggered | disabled | cancelled
timer_trigger_count: int = 0             # 触发次数
last_active_at: datetime                 # 最后活跃时间
```

---

## 5. 最佳实践

### 5.1 配置复用

```python
# 多个会话复用同一配置
config = config_cache.get(config_hash)  # 服务级缓存

# 每个会话独立 Agent 实例
agent = WorkflowAgentV2(config=config, llm=llm)  # 会话级实例
```

### 5.2 历史加载策略

```python
# 首次创建会话时加载历史
history = await repos.messages.list_by_session(session_id, limit=50)
if history:
    context = [{"role": m.role.value, "content": m.content} for m in history]
    # 注入到 Agent
    agent._agent.load_history(convert_to_messages(context))
```

### 5.3 Timer 重置

```python
# 每次用户消息后重置 Timer
session_mgr.reset_timer(session_id)

# Timer 触发后继续下一轮（如果未耗尽）
if not timer.is_exhausted():
    start_timer(ctx)
```

### 5.4 资源回收

```python
# 空闲会话自动回收
if ctx.idle_seconds > idle_timeout:
    await session_mgr.destroy(session_id)

# 会话销毁时取消 Timer
if ctx.timer and ctx.timer.task:
    ctx.timer.task.cancel()
```

---

## 6. 实现清单

### 6.1 新增文件

| 文件 | 说明 |
|------|------|
| `api/services/config_cache.py` | 服务级配置缓存 |
| `api/services/session_context.py` | 会话上下文 |
| `api/services/session_manager.py` | 会话管理器 |

### 6.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `api/routers/query.py` | 集成 SessionManager |
| `api/container.py` | 注册新依赖 |
| `api/models/documents.py` | 添加 Timer 字段 |

### 6.3 废弃文件

| 文件 | 说明 |
|------|------|
| `api/services/agent_manager.py` | 被 SessionManager 替代 |
| `api/services/timer_service.py` | Timer 逻辑合并到 SessionManager |

---

## 7. 关键设计决策

### 7.1 为什么 Agent 是会话级？

- WorkflowAgentV2 内部的 `Agent` 维护对话历史
- 不同会话的历史必须隔离
- 会话级实例确保上下文独立

### 7.2 为什么配置是服务级缓存？

- 同一 chatbot 的多个会话使用相同配置
- 配置解析成本高（可能涉及 LLM 调用）
- 按 config_hash 缓存，配置变更自动失效

### 7.3 为什么 Timer 是会话级？

- 每个会话有独立的超时逻辑
- 用户活动重置当前会话的 Timer
- Timer 触发次数按会话计数

### 7.4 历史加载 vs 实时查询

**选择：首次加载 + 内存维护**

- 首次创建会话时从 DB 加载历史
- 后续消息在内存中维护
- 会话销毁时历史已持久化

**原因：**
- 减少 DB 查询
- Agent 内部已有历史管理
- 消息实时持久化确保不丢失

---

## 8. Agent 层与应用层数据交互

### 8.1 问题分析

**核心挑战**：
- Agent 层（SDK）产生 tool calls 和 usages 数据
- 应用层（API）需要将这些数据持久化到 DB
- Agent 层不应直接依赖应用层的 DB
- 需要保持层级分离的同时实现数据流通

**数据来源**：
| 数据类型 | 来源 | 目标表 |
|----------|------|--------|
| Token Usage | `Agent._token_cost` → `UsageSummary` | usages |
| Tool Calls | `query_stream()` → `ToolCallEvent/ToolResultEvent` | events |
| Messages | `Agent._messages` | messages |

### 8.2 解决方案：统一事件收集器

采用 **统一事件收集器** 设计，无论流式还是非流式都使用相同的收集逻辑：

```
┌─────────────────────────────────────────────────────────────────┐
│                     应用层 (API)                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   QueryService                               ││
│  │                                                              ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │              EventCollector (统一)                       │││
│  │  │  - 流式：透传事件 + 收集                                 │││
│  │  │  - 非流式：仅收集                                        │││
│  │  └─────────────────────────────────────────────────────────┘││
│  │                          │                                   ││
│  │                          ▼                                   ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │              QueryRecorder (统一写入)                    │││
│  │  │  - messages / events / usages                           │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Agent 层 (SDK)                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 WorkflowAgentV2                              ││
│  │  query_stream() → AgentEvent  (唯一数据源)                  ││
│  │  get_usage() → UsageSummary                                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**核心思想**：
- `query_stream()` 是唯一的数据源
- `EventCollector` 统一收集所有事件
- 流式/非流式的区别仅在于是否透传事件给客户端

### 8.3 EventCollector（统一事件收集器）

```python
# api/services/v2/event_collector.py

from dataclasses import dataclass, field
from typing import Any
import time

from bu_agent_sdk.agent.events import (
    AgentEvent, ToolCallEvent, ToolResultEvent,
    FinalResponseEvent, StepStartEvent, StepCompleteEvent,
    TextEvent,
)
from bu_agent_sdk.tokens import UsageSummary


@dataclass
class ToolCallRecord:
    """Tool 调用记录"""
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str = ""
    is_error: bool = False
    started_at: float = 0
    duration_ms: float = 0


@dataclass
class QueryResult:
    """Query 执行结果（统一数据结构）"""
    response: str
    usage: UsageSummary | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_duration_ms: float = 0


@dataclass
class EventCollector:
    """
    统一事件收集器

    职责：
    - 收集 Agent 流式事件（流式/非流式通用）
    - 转换为统一的 QueryResult
    - 支持增量收集

    Usage（非流式）:
        collector = EventCollector(correlation_id="xxx", session_id="yyy")
        async for event in agent.query_stream(message):
            collector.collect(event)  # 仅收集

        result = collector.to_result(usage)

    Usage（流式）:
        collector = EventCollector(correlation_id="xxx", session_id="yyy")
        async for event in agent.query_stream(message):
            collector.collect(event)
            yield format_sse_event(event)  # 收集 + 透传

        result = collector.to_result(usage)
    """
    correlation_id: str
    session_id: str
    user_message: str = ""

    # 收集的数据
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_response: str = ""
    text_chunks: list[str] = field(default_factory=list)

    # 内部状态
    _pending_calls: dict[str, ToolCallRecord] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.time)

    def collect(self, event: AgentEvent) -> None:
        """收集单个事件"""
        match event:
            case ToolCallEvent(tool=name, args=args, tool_call_id=call_id):
                record = ToolCallRecord(
                    tool_call_id=call_id,
                    tool_name=name,
                    arguments=args,
                    started_at=time.time(),
                )
                self._pending_calls[call_id] = record

            case ToolResultEvent(result=result, is_error=is_error, tool_call_id=call_id):
                if call_id in self._pending_calls:
                    record = self._pending_calls.pop(call_id)
                    record.result = result
                    record.is_error = is_error
                    record.duration_ms = (time.time() - record.started_at) * 1000
                    self.tool_calls.append(record)

            case TextEvent(content=content):
                self.text_chunks.append(content)

            case FinalResponseEvent(content=content):
                self.final_response = content

    def to_result(self, usage: UsageSummary | None = None) -> QueryResult:
        """转换为统一的 QueryResult"""
        return QueryResult(
            response=self.final_response,
            usage=usage,
            tool_calls=list(self.tool_calls),
            total_duration_ms=(time.time() - self._start_time) * 1000,
        )

    def get_event_records(self) -> list[dict]:
        """转换为 events 表记录"""
        return [
            {
                "correlation_id": self.correlation_id,
                "session_id": self.session_id,
                "event_type": "tool_call",
                "tool_name": tc.tool_name,
                "tool_call_id": tc.tool_call_id,
                "arguments": tc.arguments,
                "result": tc.result,
                "is_error": tc.is_error,
                "duration_ms": tc.duration_ms,
            }
            for tc in self.tool_calls
        ]
```

### 8.4 QueryRecorder（统一 DB 写入）

```python
# api/services/v2/query_recorder.py

import asyncio
import logging
from typing import TYPE_CHECKING

from bu_agent_sdk.tokens import UsageSummary

from api.models import MessageRole
from api.services.v2.event_collector import EventCollector

if TYPE_CHECKING:
    from api.services.repositories import RepositoryManager

logger = logging.getLogger(__name__)


class QueryRecorder:
    """
    统一 DB 写入

    职责：
    - 接收 EventCollector 收集的数据
    - 统一写入 messages / events / usages 表
    - 支持异步 fire-and-forget 模式
    """

    def __init__(self, repos: "RepositoryManager"):
        self._repos = repos

    async def record(
        self,
        collector: EventCollector,
        usage: UsageSummary | None = None,
    ) -> None:
        """
        统一记录逻辑

        Args:
            collector: 事件收集器
            usage: Token 使用统计
        """
        try:
            await asyncio.gather(
                self._record_messages(collector),
                self._record_events(collector),
                self._record_usage(collector, usage),
            )
        except Exception as e:
            logger.error(f"Failed to record query: {e}")

    async def _record_messages(self, collector: EventCollector) -> None:
        """记录消息"""
        # 用户消息
        if collector.user_message:
            await self._repos.messages.create(
                session_id=collector.session_id,
                role=MessageRole.USER,
                content=collector.user_message,
                correlation_id=collector.correlation_id,
            )

        # 助手消息
        if collector.final_response:
            await self._repos.messages.create(
                session_id=collector.session_id,
                role=MessageRole.ASSISTANT,
                content=collector.final_response,
                correlation_id=collector.correlation_id,
            )

    async def _record_events(self, collector: EventCollector) -> None:
        """记录事件"""
        event_records = collector.get_event_records()
        if event_records:
            await self._repos.events.batch_create(event_records)

    async def _record_usage(
        self,
        collector: EventCollector,
        usage: UsageSummary | None,
    ) -> None:
        """记录 usage"""
        if not usage:
            return

        await self._repos.usages.create(
            correlation_id=collector.correlation_id,
            session_id=collector.session_id,
            prompt_tokens=usage.total_prompt_tokens,
            completion_tokens=usage.total_completion_tokens,
            total_tokens=usage.total_tokens,
            total_cost=usage.total_cost,
        )

    def record_async(
        self,
        collector: EventCollector,
        usage: UsageSummary | None = None,
    ) -> asyncio.Task:
        """
        异步记录（Fire & Forget）

        Returns:
            asyncio.Task 用于可选的等待或取消
        """
        return asyncio.create_task(self.record(collector, usage))
```

### 8.5 应用层集成（统一模式）

```python
# api/routers/v2/query.py

from api.services.v2.event_collector import EventCollector
from api.services.v2.query_recorder import QueryRecorder


# ─────────────────────────────────────────────────────────
# 非流式：仅收集，不透传
# ─────────────────────────────────────────────────────────

async def query(
    request: QueryRequest,
    session_mgr: SessionManagerDep,
    repos: RepositoryManagerDep,
):
    correlation_id = get_correlation_id()
    ctx = await session_mgr.get_or_create(...)

    # 创建收集器
    collector = EventCollector(
        correlation_id=correlation_id,
        session_id=request.session_id,
        user_message=request.message,
    )

    # 执行 query，收集所有事件
    async for event in ctx.agent.query_stream(request.message):
        collector.collect(event)  # 仅收集，不透传

    # 获取 usage
    usage = await ctx.agent.get_usage()

    # 异步记录（Fire & Forget）
    recorder = QueryRecorder(repos)
    recorder.record_async(collector, usage)

    # 返回响应
    return QueryResponse(
        session_id=request.session_id,
        message=collector.final_response,
        status="success",
    )


# ─────────────────────────────────────────────────────────
# 流式：收集 + 透传
# ─────────────────────────────────────────────────────────

async def query_stream(
    request: QueryRequest,
    session_mgr: SessionManagerDep,
    repos: RepositoryManagerDep,
):
    correlation_id = get_correlation_id()
    ctx = await session_mgr.get_or_create(...)

    # 创建收集器
    collector = EventCollector(
        correlation_id=correlation_id,
        session_id=request.session_id,
        user_message=request.message,
    )

    async def generate():
        # 执行 query，收集 + 透传
        async for event in ctx.agent.query_stream(request.message):
            collector.collect(event)
            yield format_sse_event(event)  # 透传给客户端

        # 流结束后异步记录
        usage = await ctx.agent.get_usage()
        recorder = QueryRecorder(repos)
        recorder.record_async(collector, usage)

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 8.6 数据流总结

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query 请求                                │
│                   (流式 / 非流式)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     WorkflowAgentV2                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ query_stream() (唯一数据源)                                 ││
│  │   ├── LLM 调用 → _token_cost 累积 usage                     ││
│  │   ├── Tool 执行 → ToolCallEvent / ToolResultEvent           ││
│  │   └── 最终响应 → FinalResponseEvent                         ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 EventCollector (统一收集)                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ collect(event)                                              ││
│  │   ├── 非流式：仅收集                                        ││
│  │   └── 流式：收集 + 透传                                     ││
│  │                                                             ││
│  │ to_result(usage) → QueryResult                              ││
│  │ get_event_records() → list[dict]                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 QueryRecorder (统一写入)                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ record_async(collector, usage)  # Fire & Forget             ││
│  │   ├── _record_messages()                                    ││
│  │   ├── _record_events()                                      ││
│  │   └── _record_usage()                                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          DB                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │ messages │ │ events   │ │ usages   │                        │
│  └──────────┘ └──────────┘ └──────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 8.7 最佳实践

#### 8.7.1 统一模式的优势

| 方面 | 双模式设计 | 统一设计 |
|------|-----------|----------|
| 收集逻辑 | 两套（QueryResult + EventCollector） | 一套（EventCollector） |
| DB 写入 | 两个函数 | 一个类（QueryRecorder） |
| 维护成本 | 高（需同步两套逻辑） | 低（单一逻辑） |
| 数据一致性 | 可能不一致 | 保证一致 |
| 代码复用 | 低 | 高 |

#### 8.7.2 流式 vs 非流式

```python
# 唯一区别：是否透传事件

# 非流式
async for event in agent.query_stream(message):
    collector.collect(event)  # ✅ 仅收集

# 流式
async for event in agent.query_stream(message):
    collector.collect(event)  # ✅ 收集
    yield format_sse_event(event)  # ✅ + 透传
```

#### 8.7.3 异步写入

```python
# ✅ 推荐：Fire & Forget
recorder.record_async(collector, usage)

# ❌ 避免：同步等待
await recorder.record(collector, usage)  # 增加响应延迟
```

#### 8.7.4 错误处理

```python
class QueryRecorder:
    async def record(self, collector, usage):
        try:
            await asyncio.gather(
                self._record_messages(collector),
                self._record_events(collector),
                self._record_usage(collector, usage),
            )
        except Exception as e:
            # 记录错误但不影响主流程
            logger.error(f"Failed to record: {e}")
            # 可选：发送到错误队列重试
```

#### 8.7.5 批量写入优化

```python
async def _record_events(self, collector: EventCollector) -> None:
    event_records = collector.get_event_records()
    if event_records:
        # ✅ 批量插入而非逐条插入
        await self._repos.events.batch_create(event_records)
```

### 8.8 关键设计决策

#### Q1: 为什么使用统一的 EventCollector？

**答案**：收敛逻辑，降低维护成本
- 流式和非流式使用相同的收集逻辑
- 数据结构统一，保证一致性
- 修改一处，两种模式同时生效

#### Q2: 为什么 query_stream() 是唯一数据源？

**答案**：简化架构
- `query_stream()` 已经产生所有需要的事件
- 非流式只是不透传事件，收集逻辑相同
- 避免在 SDK 层维护两套逻辑

#### Q3: 为什么不让 Agent 直接写 DB？

**答案**：违反分层原则
- Agent 是 SDK 层，应该是通用的、可复用的
- 直接依赖 DB 会导致 SDK 与特定存储耦合
- 不同应用可能使用不同的存储方案

#### Q4: 为什么使用 QueryRecorder 而非直接写 DB？

**答案**：封装写入逻辑
- 统一的写入入口，便于添加日志、监控
- 支持 `record_async()` 异步模式
- 便于单元测试（可 mock）

#### Q5: Usage 数据何时获取？

**答案**：Query 结束后
- `get_usage()` 返回累积的 usage
- 在 `query_stream()` 完成后调用
- 确保获取完整的 token 统计
