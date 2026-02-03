# V2 数据模型设计文档

## 设计原则

1. **简化优先**：移除 V2 架构不使用的字段
2. **对齐架构**：数据模型匹配 EventCollector/QueryRecorder 模式
3. **DB 级缓存**：configs 表持久化配置，避免重复 LLM 解析
4. **扁平结构**：避免嵌套，简化查询

---

## 1. 架构对比

### V3 vs V2

| 维度 | V3 (5表) | V2 (6表) |
|------|----------|----------|
| 配置存储 | configs 表 | configs 表 (简化) + 内存 ConfigCache |
| 会话状态 | sessions 表 (含 timer) | sessions 表 (简化) + 独立 timers 表 |
| 消息状态 | messages.state | 无 (内存管理) |
| 事件类型 | 多种 EventType | 仅 tool_calls |
| Token 结构 | 嵌套 details/summary | 扁平 by_model |
| Timer 存储 | sessions 表内嵌 | 独立 timers 表 (支持多 Timer/Session) |

### V2 数据流

```
Query Request
    ↓
ConfigCache.get(config_hash)     → 内存缓存 (L1)
    ↓ (miss)
ConfigRepository.get()           → DB 缓存 (L2)
    ↓ (miss)
HttpConfigLoader.load()          → 远程加载 + LLM 解析
    ↓
ConfigRepository.set()           → 持久化到 DB
    ↓
SessionManager.get_or_create()   → 内存 SessionContext
    ↓
WorkflowAgentV2.query_stream()   → 产生 AgentEvents
    ↓
EventCollector.collect()         → 收集: tool_calls, final_response
    ↓
QueryRecorder.record()           → 写入 DB: messages, tool_calls, usages
```

---

## 2. 6表设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        V2 6表架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐                                                    │
│  │ configs  │ ◄─────── config_hash (DB 级配置缓存)               │
│  └──────────┘                                                    │
│       │                                                          │
│       │ config_hash                                              │
│       ▼                                                          │
│  ┌──────────┐      1:N       ┌──────────┐                       │
│  │ sessions │ ──────────────►│ messages │                       │
│  │          │                │          │                       │
│  │ tenant_id│                │ role     │                       │
│  │chatbot_id│                │ content  │                       │
│  └──────────┘                └──────────┘                       │
│       │                           │                              │
│       │ session_id                │ correlation_id               │
│       ▼                           ▼                              │
│  ┌──────────┐               ┌──────────┐                        │
│  │tool_calls│               │  usages  │                        │
│  │          │               │          │                        │
│  │tool_name │               │ by_model │                        │
│  │arguments │               │total_cost│                        │
│  └──────────┘               └──────────┘                        │
│       │                                                          │
│       │ session_id                                               │
│       ▼                                                          │
│  ┌──────────┐                                                    │
│  │  timers  │ ◄─────── 独立表，支持多 Timer/Session              │
│  │          │                                                    │
│  │ timer_id │                                                    │
│  │ status   │                                                    │
│  └──────────┘                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

表职责：
┌────────────┬────────────────────────────────────────────────────┐
│ 表名        │ 职责                                               │
├────────────┼────────────────────────────────────────────────────┤
│ configs    │ DB 级配置缓存，避免重复 LLM 解析                     │
│ sessions   │ 会话元数据，唯一存储 tenant_id/chatbot_id           │
│ messages   │ 用户/助手消息，构建对话上下文                        │
│ tool_calls │ 工具调用记录，追踪 Agent 行为                        │
│ usages     │ Token 消耗，支持计费和统计                          │
│ timers     │ 会话级定时器，支持多 Timer/Session，独立生命周期     │
└────────────┴────────────────────────────────────────────────────┘
```

---

## 3. 表结构设计

### 3.1 configs (配置缓存表)

```python
@dataclass
class ConfigDocumentV2:
    config_hash: str             # _id (MD5 哈希)
    tenant_id: str               # 租户ID
    chatbot_id: str              # Chatbot ID
    raw_config: dict             # 原始配置 JSON
    parsed_config: dict          # 解析后的配置 (LLM 解析结果)
    version: Optional[str]       # 配置版本
    created_at: datetime
    updated_at: datetime
    access_count: int            # 访问次数
```

**索引**:
```javascript
db.configs.createIndex({ "tenant_id": 1, "chatbot_id": 1 })
db.configs.createIndex({ "updated_at": -1 })
```

**设计说明**:
- 按 config_hash 索引，相同配置只存储一份
- 相同 chatbot/tenant 下的不同 sessions 复用
- access_count 用于 LRU 淘汰策略

### 3.2 sessions (会话表)

```python
@dataclass
class SessionDocumentV2:
    session_id: str              # _id
    tenant_id: str               # 租户ID (唯一存储位置)
    chatbot_id: str              # Chatbot ID (唯一存储位置)
    customer_id: Optional[str]   # 客户ID
    config_hash: Optional[str]   # 配置哈希 (变更检测)
    title: Optional[str]         # 会话标题
    source: Optional[str]        # 来源 (web | app | api)
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]
    metadata: dict               # 可选: tags, language
```

**索引**:
```javascript
db.sessions.createIndex({ "tenant_id": 1, "chatbot_id": 1, "created_at": -1 })
db.sessions.createIndex({ "customer_id": 1 }, { sparse: true })
```

**与 V3 对比**:
- 移除: timer_status, timer_config, last_active_at (内存管理)
- 移除: message_count, event_count, total_tokens, total_cost (按需计算)

---

### 3.2 messages (消息表)

```python
@dataclass
class MessageDocumentV2:
    message_id: str              # _id (UUID)
    session_id: str              # 外键
    role: str                    # "user" | "assistant"
    content: str                 # 消息内容
    correlation_id: Optional[str]
    created_at: datetime
```

**索引**:
```javascript
db.messages.createIndex({ "session_id": 1, "created_at": 1 })
db.messages.createIndex({ "correlation_id": 1 }, { sparse: true })
```

**与 V3 对比**:
- 移除: state (MessageState) - V2 不使用状态追踪
- 移除: metadata - V2 不使用
- 简化: role 为字符串 (只有 "user" | "assistant")

---

### 3.3 tool_calls (工具调用表)

```python
@dataclass
class ToolCallDocumentV2:
    tool_call_id: str            # _id (来自 LLM)
    session_id: str              # 外键
    correlation_id: str          # 请求关联
    offset: int                  # 会话内序号
    tool_name: str               # 工具名称
    arguments: dict              # 工具输入
    result: Optional[str]        # 工具输出 (截断)
    is_error: bool               # 是否错误
    duration_ms: int             # 耗时
    created_at: datetime
```

**索引**:
```javascript
db.tool_calls.createIndex({ "session_id": 1, "offset": 1 })
db.tool_calls.createIndex({ "correlation_id": 1 })
```

**与 V3 events 对比**:
- 重命名: events → tool_calls (语义清晰)
- 移除: event_type (始终为 TOOL_CALL)
- 移除: status (用 is_error 布尔值)
- 移除: message_id, start_time, end_time, metadata
- 重命名: action → tool_name, input_data → arguments, output_data → result

---

### 3.4 usages (Token 消耗表)

```python
@dataclass
class UsageDocumentV2:
    usage_id: str                # _id (UUID)
    session_id: str              # 外键
    correlation_id: str          # 请求关联 (唯一)

    # 输入统计
    total_input_tokens: int      # 总输入 tokens
    cached_input_tokens: int     # 缓存命中的输入 tokens

    # 输出统计
    total_output_tokens: int     # 总输出 tokens

    # 汇总统计
    total_tokens: int            # 总 tokens (input + output)
    total_cost: float            # 总费用 (USD)

    # 按模型统计
    by_model: dict               # {"gpt-4": {"input": 100, "cached": 20, "output": 50, "cost": 0.01}}
    created_at: datetime
```

**索引**:
```javascript
db.usages.createIndex({ "session_id": 1, "created_at": -1 })
db.usages.createIndex({ "correlation_id": 1 }, { unique: true })
```

**设计说明**:
- `cached_input_tokens`: 支持 OpenAI/Anthropic 的 prompt caching 统计
- `by_model`: 每个模型的详细统计，包含 cached 字段
- 扁平结构，单次写入，无 finalize 步骤

**与 V3 对比**:
- 移除: details 数组 (扁平化到 by_model)
- 移除: summary 对象 (内联到顶层)
- 移除: is_finalized, finalized_at (写入即完成)
- 新增: cached_input_tokens (支持缓存统计)

---

### 3.6 timers (定时器表)

```python
@dataclass
class TimerDocumentV2:
    timer_instance_id: str       # _id (UUID)
    session_id: str              # 外键
    timer_id: str                # 关联 TimerConfig.timer_id

    # 状态
    status: str                  # "pending" | "triggered" | "disabled" | "cancelled"
    trigger_count: int           # 已触发次数

    # 时间
    created_at: datetime
    next_trigger_at: datetime    # 下次触发时间（索引字段）
    last_triggered_at: Optional[datetime]

    # 配置快照（创建时复制）
    delay_seconds: int           # 延迟秒数
    max_triggers: int            # 最大触发次数，0=无限
    tool_name: str               # 工具名称
    tool_params: dict            # 工具参数
    message: Optional[str]       # 消息内容（generate_response 专用）
```

**索引**:
```javascript
db.timers.createIndex({ "session_id": 1, "timer_id": 1 })
db.timers.createIndex({ "status": 1, "next_trigger_at": 1 })  // 关键：超时查询
db.timers.createIndex({ "next_trigger_at": 1 }, { sparse: true })
```

**设计说明**:
- 独立表设计，支持每个 Session 多个 Timer
- `next_trigger_at` 索引便于超时查询，避免运行时计算
- 配置快照确保 Agent 配置更新不影响已有 Timer
- 状态机: pending → triggered → disabled/cancelled

---

## 4. 数据流设计

### 4.1 请求处理流程

```
User Request (correlation_id=xxx)
       │
       ├──► 1. EventCollector 收集事件
       │         ├── user_message
       │         ├── tool_calls[]
       │         └── final_response
       │
       └──► 2. QueryRecorder 写入 DB (并行)
                 ├── messages: USER + ASSISTANT
                 ├── tool_calls: 工具调用记录
                 └── usages: Token 消耗
```

### 4.2 QueryRecorder 实现

```python
class QueryRecorderV2:
    async def record(self, collector: EventCollector, usage: UsageSummary | None):
        await asyncio.gather(
            self._record_messages(collector),
            self._record_tool_calls(collector),
            self._record_usage(collector, usage),
        )

    async def _record_messages(self, collector: EventCollector):
        # 简单: session_id, role, content, correlation_id
        if collector.user_message:
            await self._repos.messages.create(
                session_id=collector.session_id,
                role="user",
                content=collector.user_message,
                correlation_id=collector.correlation_id,
            )
        if collector.final_response:
            await self._repos.messages.create(
                session_id=collector.session_id,
                role="assistant",
                content=collector.final_response,
                correlation_id=collector.correlation_id,
            )

    async def _record_tool_calls(self, collector: EventCollector):
        # 直接映射 ToolCallRecord
        for i, tc in enumerate(collector.tool_calls):
            await self._repos.tool_calls.create(
                tool_call_id=tc.tool_call_id,
                session_id=collector.session_id,
                correlation_id=collector.correlation_id,
                offset=i,
                tool_name=tc.tool_name,
                arguments=tc.arguments,
                result=tc.result[:1000] if tc.result else None,
                is_error=tc.is_error,
                duration_ms=int(tc.duration_ms),
            )

    async def _record_usage(self, collector: EventCollector, usage: UsageSummary | None):
        if not usage:
            return
        # 单次写入，扁平结构
        by_model = {
            model: {"input": s.prompt_tokens, "output": s.completion_tokens, "cost": s.cost}
            for model, s in usage.by_model.items()
        }
        await self._repos.usages.create(
            session_id=collector.session_id,
            correlation_id=collector.correlation_id,
            total_input_tokens=usage.total_prompt_tokens,
            total_output_tokens=usage.total_completion_tokens,
            total_tokens=usage.total_tokens,
            total_cost=usage.total_cost,
            by_model=by_model,
        )
```

---

## 5. 查询示例

### 5.1 获取会话消息

```python
messages = await db.messages.find(
    {"session_id": session_id}
).sort("created_at", 1).to_list()
```

### 5.2 获取请求的工具调用

```python
tool_calls = await db.tool_calls.find(
    {"correlation_id": correlation_id}
).sort("offset", 1).to_list()
```

### 5.3 获取会话 Token 消耗

```python
usages = await db.usages.find(
    {"session_id": session_id}
).to_list()

total = sum(u["total_tokens"] for u in usages)
```

### 5.4 全链路追踪

```python
correlation_id = "xxx"

# 1. 获取消息对
messages = await db.messages.find({"correlation_id": correlation_id}).to_list()

# 2. 获取工具调用
tool_calls = await db.tool_calls.find(
    {"correlation_id": correlation_id}
).sort("offset", 1).to_list()

# 3. 获取 Token 消耗
usage = await db.usages.find_one({"correlation_id": correlation_id})
```

---

## 6. 设计优势

| 维度 | 说明 |
|-----|------|
| **DB 缓存** | configs 表持久化配置，避免重复 LLM 解析 |
| **简化** | 字段数减少 40%+，移除冗余字段 |
| **对齐** | 数据模型匹配 EventCollector/QueryRecorder |
| **高效** | 扁平结构，单次写入，无嵌套更新 |
| **清晰** | tool_calls 语义明确，移除冗余 event_type |
| **灵活** | by_model 字典支持任意模型组合 |

---

## 7. 配置缓存策略

### 7.1 两级缓存

```
┌─────────────────────────────────────────────────────────────────┐
│                     配置缓存策略                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │ ConfigCache  │ ◄─── L1: 内存缓存 (LRU, TTL)                  │
│  │   (内存)     │      - 快速访问                                │
│  └──────────────┘      - 服务重启后失效                          │
│         │                                                        │
│         │ miss                                                   │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ ConfigRepo   │ ◄─── L2: DB 缓存                              │
│  │   (MongoDB)  │      - 持久化存储                              │
│  └──────────────┘      - 跨服务共享                              │
│         │                                                        │
│         │ miss                                                   │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ ConfigLoader │ ◄─── 远程加载 + LLM 解析                       │
│  │   (HTTP)     │      - 耗时操作                                │
│  └──────────────┘      - 解析后写入 L2                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 缓存流程

```python
async def get_config(config_hash: str, tenant_id: str, chatbot_id: str):
    # L1: 内存缓存
    config = config_cache.get(config_hash)
    if config:
        return config

    # L2: DB 缓存
    config_doc = await repos.configs.get(config_hash)
    if config_doc:
        config = WorkflowConfigSchema(**config_doc.parsed_config)
        config_cache.set(config_hash, config)  # 回填 L1
        return config

    # 远程加载 + LLM 解析
    raw_config = await config_loader.load(tenant_id, chatbot_id)
    parsed_config = await llm_parser.parse(raw_config)

    # 写入 L2
    await repos.configs.set(
        config_hash=config_hash,
        tenant_id=tenant_id,
        chatbot_id=chatbot_id,
        raw_config=raw_config,
        parsed_config=parsed_config.model_dump(),
    )

    # 写入 L1
    config_cache.set(config_hash, parsed_config)

    return parsed_config
```

### 7.3 配置复用

相同 chatbot/tenant 下的不同 sessions 复用同一份配置：

```python
# Session A 首次请求
config = await get_config("abc123", "tenant-1", "chatbot-1")
# → L1 miss → L2 miss → 远程加载 + LLM 解析 → 写入 L2 → 写入 L1

# Session B 请求（同一 chatbot）
config = await get_config("abc123", "tenant-1", "chatbot-1")
# → L1 hit → 直接返回（无 DB 查询，无 LLM 解析）

# 服务重启后，Session C 请求
config = await get_config("abc123", "tenant-1", "chatbot-1")
# → L1 miss → L2 hit → 回填 L1 → 返回（无 LLM 解析）
```

---

## 8. Timer 生命周期与最佳实践

### 8.1 Timer 生命周期

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Timer 完整生命周期                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │ 应用启动      │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │TimerService  │◄─────────────────────────────────────────────┐            │
│  │   .start()   │                                              │            │
│  └──────┬───────┘                                              │            │
│         │                                                      │            │
│         │ 后台循环 (每 30s)                                     │            │
│         ▼                                                      │            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │            │
│  │ 查询 timers  │───►│ 执行 Tool    │───►│ 更新状态     │──────┘            │
│  │ next_trigger │    │ (统一调用)   │    │              │                   │
│  │ < now        │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │ 用户首次消息  │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ 创建 Session │───►│ 读取 Agent   │───►│ 创建 Timer   │                   │
│  │              │    │ .timers 配置 │    │ 实例到 DB    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                           │
│  │ 用户后续消息  │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐    ┌──────────────┐                                       │
│  │ 更新 Session │───►│ 重置 Timer   │                                       │
│  │ last_active  │    │ next_trigger │                                       │
│  └──────────────┘    └──────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Timer 状态机

```
┌─────────────────────────────────────────────────────────────────┐
│                      Timer 状态机                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   用户消息   ┌──────────┐   超时触发   ┌──────────┐│
│  │ pending  │◄────────────│ triggered│◄────────────│ pending  ││
│  └────┬─────┘             └────┬─────┘             └──────────┘│
│       │                        │                               │
│       │ 达到 max_triggers      │ Session 关闭                   │
│       ▼                        ▼                               │
│  ┌──────────┐             ┌──────────┐                         │
│  │ disabled │             │ cancelled│                         │
│  └──────────┘             └──────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Timer 配置示例

```json
{
  "timers": [
    {
      "timer_id": "idle_reminder",
      "delay_seconds": 300,
      "max_triggers": 3,
      "tool_name": "generate_response",
      "message": "您好，请问还有什么可以帮您的吗？"
    },
    {
      "timer_id": "session_timeout",
      "delay_seconds": 1800,
      "max_triggers": 1,
      "tool_name": "close_conversation",
      "tool_params": {}
    },
    {
      "timer_id": "auto_handoff",
      "delay_seconds": 600,
      "max_triggers": 1,
      "tool_name": "handoff_to",
      "tool_params": {"type": "unassigned"}
    }
  ]
}
```



### 8.5 Timer 资源控制

```python
class TimerConfig:
    """Timer 全局配置"""
    MAX_TIMERS_PER_SESSION = 10       # 每个 Session 最大 Timer 数
    MAX_CONCURRENT_TRIGGERS = 50      # 单次扫描最大触发数
    SCAN_INTERVAL = 30                # 扫描间隔（秒）
    BATCH_SIZE = 100                  # 批量查询大小
    TIMER_TTL_DAYS = 7                # Timer 过期天数
```

### 8.6 Timer 清理策略

```python
class TimerCleanupPolicy:
    """Timer 清理策略"""

    async def cleanup_expired_timers(self, repos):
        """清理过期 Timer"""
        cutoff = datetime.utcnow() - timedelta(days=TIMER_TTL_DAYS)
        await repos.timers.delete_before(cutoff)

    async def cleanup_orphan_timers(self, repos):
        """清理孤儿 Timer（Session 已删除）"""
        await repos.timers.delete_orphans()

    async def cancel_session_timers(self, session_id: str, repos):
        """Session 关闭时取消所有 Timer"""
        await repos.timers.update_by_session(
            session_id,
            status="cancelled",
        )
```

### 8.7 Timer 重置时机

**关键点**：Timer 重置在响应返回后执行

```
用户消息 → Agent 处理 → 返回响应 → [异步] 重置 Timer
                                      │
                                      ▼
                              计时从此刻开始
```

```python
# query.py
async def query(...):
    # Phase 1: 并行准备
    # Phase 2: 核心执行
    result = await agent.query(...)

    # 返回响应（不等待 Timer 重置）
    response = QueryResponse(...)

    # Phase 3: 后台任务（fire & forget）
    asyncio.create_task(_record_and_reset_timer(...))

    return response  # 立即返回
```

---

## 9. 迁移说明

### 9.1 兼容性

V2 模型与 V3 模型并存于 `api/models/documents.py`:
- V3: SessionDocument, MessageDocument, EventDocument, TokenDocument
- V2: ConfigDocumentV2, SessionDocumentV2, MessageDocumentV2, ToolCallDocumentV2, UsageDocumentV2, TimerDocumentV2

### 9.2 迁移策略

1. **新会话**: 使用 V2 模型
2. **旧会话**: 继续使用 V3 模型
3. **渐进迁移**: 按需迁移历史数据

### 9.3 集合名称

```python
# V2 集合
CONFIGS = "configs"        # 配置缓存
SESSIONS = "sessions"      # 复用
MESSAGES = "messages"      # 复用
TOOL_CALLS = "tool_calls"  # 新集合 (替代 events)
USAGES = "usages"          # 复用
TIMERS = "timers"          # 新集合 (独立 Timer 表)
```
