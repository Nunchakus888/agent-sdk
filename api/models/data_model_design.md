# 数据模型设计文档 v3

## 设计原则

1. **避免冗余**：`tenant_id`、`chatbot_id` 只在 `sessions` 表存储
2. **职责单一**：每个表只做一件事
3. **字段平铺**：常用字段平铺，不常用字段放 metadata
4. **correlation_id 优先**：关联查询优先使用 `correlation_id`（请求级别）

---

## 1. 5表设计总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              5表设计架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐                                                                │
│  │ configs  │ ◄─────── config_hash (配置缓存)                                │
│  └──────────┘                                                                │
│       │                                                                      │
│       │ config_hash                                                          │
│       ▼                                                                      │
│  ┌──────────┐      1:N       ┌──────────────────┐                           │
│  │ sessions │ ──────────────►│ messages + state │                           │
│  │          │                │                  │                           │
│  │ tenant_id│                │ session_id       │                           │
│  │chatbot_id│                │ correlation_id   │                           │
│  └──────────┘                └──────────────────┘                           │
│       │                             │                                        │
│       │ session_id                  │ correlation_id                         │
│       ▼                             ▼                                        │
│  ┌──────────┐               ┌──────────┐                                    │
│  │  events  │               │  usages  │                                    │
│  │          │               │          │                                    │
│  │ offset   │◄──────────────│ corr_id  │                                    │
│  │ corr_id  │               │ summary  │                                    │
│  └──────────┘               │ details  │                                    │
│                             └──────────┘                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

表职责：
┌──────────┬────────────────────────────────────────────────────────────────┐
│ 表名      │ 职责                                                          │
├──────────┼────────────────────────────────────────────────────────────────┤
│ configs  │ 配置缓存，避免重复解析                                          │
│ sessions │ 会话元数据，唯一存储 tenant_id/chatbot_id                       │
│ messages │ 对话消息 + 嵌入状态，构建模型上下文                              │
│ events   │ 事件日志，追踪处理过程，支持 offset 顺序追溯                     │
│ tokens   │ Token 消耗，明细 + 汇总，支持计费和统计 (表名: usages)           │
└──────────┴────────────────────────────────────────────────────────────────┘
```

---

## 2. 表结构设计

### 2.1 configs (配置缓存表)

```python
configs = {
    "_id": "config_hash",           # 主键：MD5 哈希
    "tenant_id": str,               # 租户ID
    "chatbot_id": str,              # Chatbot ID
    "raw_config": dict,             # 原始配置 JSON
    "parsed_config": dict,          # 解析后的配置
    "version": str | None,          # 配置版本
    "created_at": datetime,
}
```

**索引**: `_id` (主键), `(tenant_id, chatbot_id)`

---

### 2.2 sessions (会话表) - 字段平铺设计

```python
sessions = {
    "_id": "session_id",            # 主键 (UUID)

    # === 核心字段 (平铺) ===
    "tenant_id": str,               # 租户ID (唯一存储位置)
    "chatbot_id": str,              # Chatbot ID (唯一存储位置)
    "customer_id": str | None,      # 客户ID
    "config_hash": str | None,      # 关联配置
    "title": str | None,            # 会话标题
    "source": str | None,           # 来源渠道 (web | app | api)

    # === 统计字段，低频读写 (平铺，便于会话关闭后统计汇总查询) ===
    "message_count": int,           # 消息数量
    "event_count": int,             # 事件数量 (用于 offset 分配)
    "total_tokens": int,            # Token 总消耗
    "total_cost": float,            # 总费用 (USD)
    "latency": float,               # 平均交互耗时

    # === 时间字段 ===
    "created_at": datetime,
    "updated_at": datetime,
    "closed_at": datetime | None,   # 关闭时间

    # === 扩展字段 (不常用) ===
    "metadata": {
        "tags": list[str],          # 标签
    },
}
```

**索引**:
- `_id` (主键)
- `(tenant_id, chatbot_id, status)` (复合索引)
- `(tenant_id, created_at)` (时间范围查询)
- `customer_id`

**设计说明**:
- 常用字段平铺：`title`, `source`, `message_count`, `total_tokens` 等
- 不常用字段放 `metadata`：`tags`, `user_agent`, `ip_address` 等
- `event_count` 用于分配 events 表的 offset

---

### 2.3 messages (消息表) - 嵌入状态

```python
messages = {
    "_id": "message_id",            # 主键 (UUID)
    "session_id": str,              # 外键：会话ID
    "correlation_id": str | None,   # 请求关联ID

    # === 消息内容 ===
    "role": str,                    # user | assistant | system
    "content": str,                 # 消息内容

    # === 嵌入状态 (仅 assistant 消息) ===
    "state": {
        "phase": str,               # planning | executing | responding | idle
        "sop_step": str | None,     # 当前 SOP 步骤
        "context": dict,            # 上下文快照 (精简)
        "decision": {
            "action": str,
            "confidence": float,
            "reasoning": str,
        } | None,
    } | None,

    "metadata": dict,
    "created_at": datetime,
}

```

**索引**:
- `_id` (主键)
- `(session_id, created_at)` (会话消息列表)
- `correlation_id` (请求追踪)

---

### 2.4 events (事件日志表) - 纯事件，增加 offset

```python
events = {
    "_id": "event_id",              # 主键 (UUID)
    "session_id": str,              # 外键：会话ID
    "correlation_id": str,          # 请求关联ID

    # === 事件顺序 ===
    "offset": int,                  # 会话内事件序号 (从 0 开始自增)

    # === 事件信息 ===
    "event_type": str,              # 事件类型
    "action": str | None,           # 具体动作 (便于分析，如 "get_weather", "to_human")

    # === 时间和性能 ===
    "duration_ms": int | None,      # 耗时 (性能分析核心指标)
    "start_time": datetime,
    "end_time": datetime | None,

    # === 输入输出 (调试追溯核心字段) ===
    "input": dict | None,           # 输入数据
    "output": dict | None,          # 输出数据 / 错误信息

    "metadata": dict,
    "created_at": datetime,
}
```

**event_type 枚举**:
```python
class EventType(str, Enum):
    # 配置相关
    CONFIG_LOAD = "config_load"
    CONFIG_PARSE = "config_parse"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"

    # LLM 相关
    LLM_DECISION = "llm_decision"
    LLM_RESPONSE = "llm_response"

    # 工具相关
    TOOL_CALL = "tool_call"
    KB_RETRIEVE = "kb_retrieve"

    # 流程相关
    TRANSFER = "transfer"
    API_CALL = "api_call"

    # 错误
    ERROR = "error"
```

**索引**:
- `_id` (主键)
- `(session_id, offset)` (唯一索引，事件顺序)
- `(session_id, event_type)` (事件类型查询)
- `correlation_id` (请求追踪)

**offset 设计说明**:
- 每个 session 内的 offset 从 1 开始自增
- 用于事件顺序追溯、断点续传、增量同步
- 通过 `sessions.event_count` 原子递增分配

---

### 2.5 usages (Token 消耗表) - 明细 + 汇总

> 详细字段说明见 [第8节 Tokens 表优化](#8-tokens-表优化-基于-openai-api-规范)

```python
usages = {
    "_id": "token_id",              # 主键 (UUID)
    "session_id": str,              # 外键：会话ID
    "correlation_id": str,          # 请求关联ID (索引)

    # === Token 明细 (按阶段，对齐 OpenAI API) ===
    "details": [
        {
            "phase": str,           # decision | tool_call | response
            "event_id": str,        # 关联事件ID
            "model": str,           # 模型名称
            # 基础统计
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            # 输入明细 (prompt_tokens_details)
            "cached_tokens": int,       # 缓存命中
            "audio_input_tokens": int,  # 音频输入
            # 输出明细 (completion_tokens_details)
            "reasoning_tokens": int,    # 推理 tokens (o1/o3)
            "audio_output_tokens": int, # 音频输出
            # 费用
            "cost_usd": float,
            "timestamp": datetime,
        }
    ],

    # === Token 汇总 (请求结束时计算) ===
    "summary": {
        "total_input": int,
        "total_output": int,
        "total_tokens": int,
        "total_cost": float,
        "total_cached": int,        # 缓存命中总计
        "total_reasoning": int,     # 推理 tokens 总计
        "model_breakdown": {        # 按模型统计
            "gpt-4o": {"input": int, "output": int, "cached": int, "cost": float},
            "o1": {"input": int, "output": int, "reasoning": int, "cost": float},
        },
    } | None,

    "is_finalized": bool,           # 是否已完成统计
    "created_at": datetime,
    "finalized_at": datetime | None,
}
```

**索引**:
- `_id` (主键)
- `correlation_id` (唯一索引，一个请求一条记录)
- `session_id` (会话统计)
- `(session_id, is_finalized)` (未完成统计查询)

---

## 3. 数据流设计

### 3.1 请求处理流程

```
User Request (correlation_id=xxx)
       │
       ├──► 1. 创建 usages 记录 (is_finalized=false)
       │
       ├──► 2. Event: LLM_DECISION
       │         ├──► events 表 (offset=1)
       │         └──► usages.details 追加
       │
       ├──► 3. Event: TOOL_CALL
       │         ├──► events 表 (offset=2)
       │         └──► usages.details 追加
       │
       ├──► 4. Event: LLM_RESPONSE
       │         ├──► events 表 (offset=3)
       │         └──► usages.details 追加
       │
       ├──► 5. 创建 assistant message
       │
       └──► 6. Finalize usages
                 ├──► 计算 summary
                 ├──► 更新 sessions.total_tokens
                 └──► is_finalized=true
```

### 3.2 offset 分配机制

```python
# 原子操作分配 offset
async def allocate_event_offset(session_id: str) -> int:
    result = await db.sessions.find_one_and_update(
        {"_id": session_id},
        {"$inc": {"event_count": 1}},
        return_document=ReturnDocument.AFTER
    )
    return result["event_count"]

# 创建事件时使用
offset = await allocate_event_offset(session_id)
event = EventDocument(
    session_id=session_id,
    correlation_id=correlation_id,
    offset=offset,
    event_type=event_type,
    ...
)
```

---

## 4. 查询示例

### 4.1 获取请求的 Token 消耗 (直接查询)

```python
# 直接读取，无需聚合
token_record = await db.usages.find_one({"correlation_id": correlation_id})
if token_record and token_record["is_finalized"]:
    summary = token_record["summary"]
    # {total_input, total_output, total_tokens, total_cost, model_breakdown}
```

### 4.2 获取会话的事件列表 (按 offset 排序)

```python
# 按 offset 顺序获取事件
events = await db.events.find(
    {"session_id": session_id}
).sort("offset", 1).to_list()

# 增量获取 (从某个 offset 开始)
new_events = await db.events.find(
    {"session_id": session_id, "offset": {"$gt": last_offset}}
).sort("offset", 1).to_list()
```

### 4.3 全链路追踪

```python
# 获取一次请求的所有数据
correlation_id = "xxx"

# 1. 获取消息
messages = await db.messages.find({"correlation_id": correlation_id}).to_list()

# 2. 获取事件 (按 offset 排序)
events = await db.events.find(
    {"correlation_id": correlation_id}
).sort("offset", 1).to_list()

# 3. 获取 Token 消耗
usage = await db.usages.find_one({"correlation_id": correlation_id})
```

---

## 5. 性能分析

### 5.1 查询性能对比

| 查询场景 | 旧设计 (events+tokens) | 新设计 (分离) |
|---------|----------------------|--------------|
| 获取请求 token 汇总 | 聚合查询 O(n) | **直接读取 O(1)** |
| 获取 token 明细 | 查询 events + 筛选 | 直接读取 usages.details |
| 事件追踪 | 查询 events | 查询 events (更小更快) |
| 事件顺序 | 依赖 created_at | **offset 精确排序** |
| 增量同步 | 时间戳比较 | **offset 比较 (更可靠)** |

### 5.2 写入开销

| 操作 | 旧设计 | 新设计 |
|-----|-------|-------|
| 记录事件 | 1次写入 | 1次写入 events |
| 记录 token | 嵌入 events | 1次更新 usages.details |
| 完成请求 | 无 | 1次更新 usages.summary |

### 5.3 存储开销

| 表 | 数据特点 | 生命周期 |
|---|---------|---------|
| events | 详细日志，可归档 | 短期保留 (30天) |
| usages | 计费数据，需保留 | 长期保留 (1年+) |

---

## 6. 设计优势总结

| 维度 | 说明 |
|-----|------|
| **职责单一** | events 专注事件日志，usages 专注消耗统计 |
| **查询优化** | Token 汇总直接读取，无需聚合 |
| **顺序可靠** | offset 保证事件顺序，支持增量同步 |
| **生命周期** | events 可归档，usages 长期保留 |
| **扩展性** | 各表独立扩展，互不影响 |

---

## 7. 字段必要性分析

### 7.1 messages 表字段分析

#### tool_calls / tool_call_id - 重新评估：不必要

**原始观点**：这些是 OpenAI/Anthropic API 标准字段，用于构建模型对话上下文。

**重新分析**：

1. **API 层面 vs 存储层面**
   - API 调用时需要 tool_calls，但这是**运行时**需求
   - 存储层面不一定需要持久化这些信息

2. **工具结果可能很大**
```python
# 知识库检索结果（可能 25KB+）
tool_result = {
    "docs": ["10KB文档1", "10KB文档2", "10KB文档3"],
    "scores": [0.9, 0.8, 0.7]
}

# API 调用结果（可能 10-100KB）
tool_result = {
    "order_details": {...},  # 大量数据
    "history": [...]
}
```

3. **存储策略对比**

| 策略 | messages 表大小 | 查询性能 | 调试能力 |
|-----|----------------|---------|---------|
| 存储完整 tool_calls | 膨胀严重 | 差 | 好 |
| 存储精简 tool_calls | 中等 | 中等 | 中等 |
| **不存储 tool_calls** | 最小 | **最优** | 通过 events 表 |

4. **推荐方案**

```python
# messages 表 - 只存储最终对话
messages = [
    {"role": "user", "content": "退货政策是什么？"},
    {"role": "assistant", "content": "根据退货政策，您可以在7天内申请退货..."}
]
# 不存储: tool_calls, tool 消息

# events 表 - 存储工具调用详情（精简）
events = [
    {
        "event_type": "TOOL_CALL",
        "action": "search_knowledge_base",
        "input": {"query": "退货政策", "top_k": 5},
        "output": {"doc_ids": ["doc_1", "doc_2"], "summary": "..."},  # 精简
        "duration_ms": 150
    }
]
```

5. **构建模型上下文**

```python
async def build_model_context(session_id: str):
    """构建模型上下文 - 只需要 user/assistant 消息"""
    messages = await repos.messages.list_by_session(session_id)

    context = []
    for msg in messages:
        # 只添加 user 和 assistant 消息
        if msg.role in ["user", "assistant"]:
            context.append({"role": msg.role, "content": msg.content})

    return context

    # 注意：当前请求的 tool_calls 在内存中处理
    # 不需要从数据库读取历史 tool_calls
```

**结论**：移除 tool_calls / tool_call_id，理由：
- 工具调用在单次请求内完成，不需要持久化
- 工具结果可能很大，存储会导致文档膨胀
- 调试追溯通过 events 表实现
- 模型上下文只需要 user/assistant 的最终消息


### 7.2 events 表字段分析

#### input_data / output_data - 必要

用于调试追溯和问题排查，不同事件类型的应用场景：

| event_type | input_data 示例 | output_data 示例 | 用途 |
|------------|-----------------|------------------|------|
| **LLM_DECISION** | `{"messages": [...], "tools": [...]}` | `{"decision": "call_tool", "tool_name": "get_weather"}` | 追溯模型决策依据 |
| **LLM_RESPONSE** | `{"messages": [...]}` | `{"content": "...", "finish_reason": "stop"}` | 追溯模型响应 |
| **TOOL_CALL** | `{"tool": "get_weather", "args": {"city": "北京"}}` | `{"result": {"weather": "晴", "temp": 25}}` | 追溯工具调用 |
| **KB_RETRIEVE** | `{"query": "退货政策", "top_k": 5}` | `{"docs": ["...", "..."], "scores": [0.9, 0.8]}` | 追溯知识检索 |
| **API_CALL** | `{"url": "...", "method": "POST", "body": {...}}` | `{"status": 200, "body": {...}}` | 追溯外部调用 |
| **ERROR** | `{"context": "..."}` | null | 追溯错误上下文 |

**存储策略建议**：
- 开发环境：完整存储，便于调试
- 生产环境：精简存储，只保留关键信息
- 可通过配置控制存储粒度

**结论**：保留，是调试和追溯的核心字段。

#### event_type 补充 - 建议增加 action 字段

当前 event_type 只表示事件类型，建议增加 action 字段表示具体动作：

```python
events = {
    ...
    "event_type": str,              # 事件类型 (LLM_DECISION, TOOL_CALL, ...)
    "action": str | None,           # 具体动作 (可选，便于分析)
    ...
}

# 示例
{"event_type": "TOOL_CALL", "action": "get_weather"}      # 调用天气工具
{"event_type": "TOOL_CALL", "action": "search_kb"}        # 调用知识库
{"event_type": "LLM_DECISION", "action": "call_tool"}     # 决定调用工具
{"event_type": "LLM_DECISION", "action": "direct_reply"}  # 决定直接回复
{"event_type": "FLOW_TRANSFER", "action": "to_human"}     # 转人工
```

**结论**：增加 action 字段，便于事件分析和统计。

### 7.3 优化后的表结构

#### messages 表 (优化后)

```python
messages = {
    "_id": "message_id",            # 主键 (UUID)
    "session_id": str,              # 外键：会话ID
    "correlation_id": str | None,   # 请求关联ID

    # === 消息内容 ===
    "role": str,                    # user | assistant | system
    "content": str,                 # 消息内容

    # === 嵌入状态 (仅 assistant 消息) ===
    "state": {...} | None,

    "metadata": dict,
    "created_at": datetime,
}

```

#### events 表 (优化后)

```python
events = {
    "_id": "event_id",              # 主键 (UUID)
    "session_id": str,              # 外键：会话ID
    "correlation_id": str,          # 请求关联ID
    "offset": int,                  # 会话内事件序号

    # === 事件信息 ===
    "event_type": str,              # 事件类型
    "action": str | None,           # 具体动作 (新增，便于分析)

    # === 时间和性能 ===
    "duration_ms": int | None,      # 耗时 (性能分析核心指标)
    "start_time": datetime,
    "end_time": datetime | None,

    # === 输入输出 (调试追溯核心字段) ===
    "input": dict | None,           # 输入数据
    "output": dict | None,          # 输出数据

    "metadata": dict,
    "created_at": datetime,
}
```

---

## 8. Tokens 表优化 (基于 OpenAI API 规范)

### 8.1 OpenAI API Token 结构分析

参考 OpenAI API 的 `CompletionUsage` 结构：

```python
# OpenAI API 返回的 usage 结构
{
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "prompt_tokens_details": {
        "cached_tokens": 20,        # 缓存命中的 tokens
        "audio_tokens": 0           # 音频输入 tokens
    },
    "completion_tokens_details": {
        "reasoning_tokens": 10,     # 推理 tokens (o1 模型)
        "audio_tokens": 0,          # 音频输出 tokens
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0
    }
}
```

### 8.2 优化后的 TokenDetail 结构

```python
@dataclass
class TokenDetail:
    """Token 消耗明细 (嵌入式) - 对齐 OpenAI API 规范"""
    phase: str                               # decision | tool_call | response
    event_id: str                            # 关联事件ID
    model: str                               # 模型名称

    # === 基础 Token 统计 ===
    input_tokens: int = 0                    # prompt_tokens
    output_tokens: int = 0                   # completion_tokens
    total_tokens: int = 0                    # 总计

    # === 输入 Token 明细 (OpenAI prompt_tokens_details) ===
    cached_tokens: int = 0                   # 缓存命中的 tokens
    audio_input_tokens: int = 0              # 音频输入 tokens

    # === 输出 Token 明细 (OpenAI completion_tokens_details) ===
    reasoning_tokens: int = 0                # 推理 tokens (o1/o3 模型)
    audio_output_tokens: int = 0             # 音频输出 tokens

    # === 费用 ===
    cost_usd: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### 8.3 优化后的 TokenSummary 结构

```python
@dataclass
class TokenSummary:
    """Token 消耗汇总 (嵌入式)"""
    # === 基础汇总 ===
    total_input: int = 0
    total_output: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # === 明细汇总 ===
    total_cached: int = 0                    # 缓存命中总计
    total_reasoning: int = 0                 # 推理 tokens 总计

    # === 按模型统计 ===
    model_breakdown: dict = field(default_factory=dict)
    # {
    #     "gpt-4o": {"input": 100, "output": 50, "cached": 20, "cost": 0.01},
    #     "o1": {"input": 200, "output": 100, "reasoning": 50, "cost": 0.05}
    # }
```

### 8.4 设计优势

| 维度 | 说明 |
|-----|------|
| **API 对齐** | 与 OpenAI API 结构一致，便于数据映射 |
| **模型兼容** | 支持 o1/o3 推理模型的 reasoning_tokens |
| **缓存分析** | 支持 prompt caching 效果分析 |
| **费用精确** | 缓存 tokens 通常有折扣，便于精确计费 |

---

## 9. 数据库自动计算分析

### 9.1 需求场景

1. **Token 汇总自动计算**：插入 TokenDetail 后自动更新 TokenSummary (usages 表)
2. **会话关闭自动汇总**：会话关闭时自动计算 sessions 表的统计字段

### 9.2 MongoDB 自动计算方案对比

| 方案 | 实现方式 | 优点 | 缺点 | 适用场景 |
|-----|---------|------|------|---------|
| **Change Streams** | 监听集合变更，触发计算 | 实时、解耦 | 需要额外进程、复杂度高 | 大规模系统 |
| **Atlas Triggers** | MongoDB Atlas 内置触发器 | 无需额外进程、托管 | 仅 Atlas 支持、有延迟 | Atlas 用户 |
| **应用层计算** | 代码中显式调用 | 简单、可控、事务支持 | 耦合度高 | 中小规模系统 |
| **聚合管道** | 查询时实时计算 | 无需存储汇总 | 查询性能差 | 低频查询 |

### 9.3 推荐方案：应用层计算 + 延迟汇总

**理由**：
1. 简单可控，无需额外基础设施
2. 支持事务，保证数据一致性
3. 可以批量处理，减少数据库压力
4. 便于调试和问题排查

### 9.4 Token 汇总实现

```python
# 方案 A：实时计算 (每次追加 detail 后更新 summary)
async def add_token_detail(token_id: str, detail: TokenDetail):
    """添加 Token 明细并更新汇总"""
    await db.usages.update_one(
        {"_id": token_id},
        {
            "$push": {"details": detail.to_dict()},
            "$inc": {
                "summary.total_input": detail.input_tokens,
                "summary.total_output": detail.output_tokens,
                "summary.total_tokens": detail.total_tokens,
                "summary.total_cost": detail.cost_usd,
                "summary.total_cached": detail.cached_tokens,
                "summary.total_reasoning": detail.reasoning_tokens,
            }
        }
    )

# 方案 B：延迟计算 (请求结束时一次性计算)
async def finalize_token_record(token_id: str):
    """请求结束时计算汇总"""
    token_doc = await db.usages.find_one({"_id": token_id})

    # 计算汇总
    summary = calculate_summary(token_doc["details"])

    await db.usages.update_one(
        {"_id": token_id},
        {
            "$set": {
                "summary": summary,
                "is_finalized": True,
                "finalized_at": datetime.utcnow()
            }
        }
    )
```

**推荐方案 B**：延迟计算
- 减少数据库写入次数
- 避免并发更新冲突
- 汇总计算在内存中完成，更高效

### 9.5 会话关闭自动汇总实现

```python
async def close_session(session_id: str):
    """关闭会话并计算统计"""

    # 1. 聚合计算会话统计
    pipeline = [
        {"$match": {"session_id": session_id, "is_finalized": True}},
        {"$group": {
            "_id": None,
            "total_tokens": {"$sum": "$summary.total_tokens"},
            "total_cost": {"$sum": "$summary.total_cost"},
            "request_count": {"$sum": 1}
        }}
    ]
    result = await db.usages.aggregate(pipeline).to_list(1)

    # 2. 获取消息数量
    message_count = await db.messages.count_documents({"session_id": session_id})

    # 3. 获取事件数量
    event_count = await db.events.count_documents({"session_id": session_id})

    # 4. 更新会话
    stats = result[0] if result else {}
    await db.sessions.update_one(
        {"_id": session_id},
        {
            "$set": {
                "closed_at": datetime.utcnow(),
                "message_count": message_count,
                "event_count": event_count,
                "total_tokens": stats.get("total_tokens", 0),
                "total_cost": stats.get("total_cost", 0.0),
                "updated_at": datetime.utcnow()
            }
        }
    )
```

### 9.6 自动化触发时机

| 触发点 | 操作 | 实现方式 |
|-------|------|---------|
| 请求结束 | 计算 TokenSummary | `finalize_usage_record()` |
| 会话关闭 | 计算会话统计 | `close_session()` |
| 定时任务 | 清理未完成记录 | Cron Job |


---

## 10. 完整数据流示例

### 10.1 请求处理流程 (含 Token 计算)

```
User Request (correlation_id=xxx)
       │
       ├──► 1. 创建 usages 记录
       │         {usage_id, session_id, correlation_id, details: [], is_finalized: false}
       │
       ├──► 2. LLM Decision
       │         ├──► events 表 (offset=1)
       │         └──► usages.details.push({phase: "decision", ...})
       │
       ├──► 3. Tool Call
       │         ├──► events 表 (offset=2)
       │         └──► usages.details.push({phase: "tool_call", ...})
       │
       ├──► 4. LLM Response
       │         ├──► events 表 (offset=3)
       │         └──► usages.details.push({phase: "response", ...})
       │
       ├──► 5. 创建 assistant message
       │
       └──► 6. Finalize usages (应用层计算)
                 ├──► 计算 summary (内存中)
                 ├──► 更新 usages 记录
                 └──► is_finalized=true
```

### 10.2 会话关闭流程

```
Session Close Request
       │
       ├──► 1. 检查未完成的 usages 记录
       │         └──► 强制 finalize 或标记异常
       │
       ├──► 2. 聚合计算会话统计
       │         ├──► 消息数量 (messages 表)
       │         ├──► 事件数量 (events 表)
       │         └──► Token 汇总 (usages 表聚合)
       │
       └──► 3. 更新 sessions 表
                 ├──► message_count
                 ├──► event_count
                 ├──► total_tokens
                 ├──► total_cost
                 └──► closed_at
```

