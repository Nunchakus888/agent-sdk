# 会话级 Timer 设计 v2

## 1. 需求分析

```
Agent (tenant:chatbot)
├── Session A ──► Timer 1 (5min 无响应提醒)
│              └► Timer 2 (30min 会话超时关闭)
├── Session B ──► Timer 1 (独立计时)
│              └► Timer 2 (独立计时)
└── Session C ──► Timer 1 (独立计时)
```

**关键约束**：
1. Timer 配置在 Agent 级别（config 决定）
2. Timer 执行在 Session 级别（互不干扰）
3. 同一 Session 支持多个 Timer（不同用途）
4. Agent 多会话复用（无状态）
5. 服务重启后 Timer 需恢复
6. **统一 Tool 模型**：所有动作都是 Tool，无需额外 ActionType

---

## 2. Tool 分类设计

### 2.1 核心理念：一切皆 Tool

```
┌─────────────────────────────────────────────────────────────────┐
│                         tools[] 配置                             │
│  所有工具统一定义在 tools 数组中                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ system_actions  │    │ agent_actions   │                     │
│  │ (工具名称列表)   │    │ (工具名称列表)   │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ 执行后不参与     │    │ 执行后参与       │                     │
│  │ 上下文建设       │    │ 上下文建设       │                     │
│  │ (静默执行)       │    │ (结果入上下文)   │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 配置结构（基于 sop.json）

```json
{
  "tools": [
    {"name": "save_customer_information", "description": "...", "endpoint": {...}},
    {"name": "handoff_to", "description": "...", "endpoint": {...}},
    {"name": "close_conversation", "description": "...", "endpoint": {...}},
    {"name": "read_customer_information", "description": "...", "endpoint": {...}},
    {"name": "flow_executor", "description": "...", "endpoint": {...}}
  ],
  "system_actions": [
    "save_customer_information",
    "handoff_to",
    "close_conversation"
  ],
  "agent_actions": [
    "read_customer_information",
    "flow_executor",
    "actionbook_executor"
  ]
}
```

### 2.3 Tool 执行分类

| 分类 | 列表 | 执行特点 | 上下文 |
|------|------|----------|--------|
| **system_actions** | `["handoff_to", "close_conversation", ...]` | 可并行执行 | 不参与上下文 |
| **agent_actions** | `["read_customer_information", ...]` | 串行执行 | 结果入上下文 |
| **未分类** | 不在任何列表中 | 默认 agent_actions | 结果入上下文 |

### 2.4 执行流程

```python
async def execute_tool_calls(tool_calls: list[dict], config: WorkflowConfigSchema):
    """执行工具调用，根据分类决定执行方式"""
    system_tools = []
    agent_tools = []

    for call in tool_calls:
        tool_name = call["name"]
        if tool_name in config.system_actions:
            system_tools.append(call)
        else:
            agent_tools.append(call)

    # 1. 并行执行 system_actions（静默，不入上下文）
    if system_tools:
        await asyncio.gather(*[execute_tool(t) for t in system_tools])

    # 2. 串行执行 agent_actions（结果入上下文）
    results = []
    for tool in agent_tools:
        result = await execute_tool(tool)
        results.append({"tool": tool["name"], "result": result})

    return results  # 返回给 LLM 继续处理
```

---

## 3. Timer 数据结构（精简版）

### 3.1 SDK 配置层 - TimerConfig

```python
class TimerConfig(BaseModel):
    """Timer 配置（Agent 级别定义）"""
    timer_id: str                             # 唯一标识
    delay_seconds: int = 300                  # 延迟秒数
    max_triggers: int = 1                     # 最大触发次数，0=无限

    # 触发动作 = Tool 调用
    tool_name: str                            # 工具名称
    tool_params: dict = Field(default_factory=dict)  # 工具参数
    message: str | None = None                # 消息内容（generate_response 专用）
```

**内置 Tool**：
- `generate_response`: 生成并发送消息（使用 `message` 参数）
- 其他 Tool: 直接调用配置中定义的工具

### 3.2 配置示例

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

### 3.3 存储层 - SessionTimer（独立表）

```python
@dataclass
class SessionTimer:
    """Session Timer 实例（存储层）"""

    # 标识
    timer_instance_id: str                    # 主键 (UUID)
    session_id: str                           # 外键
    timer_id: str                             # 关联 TimerConfig.timer_id

    # 状态
    status: str = "pending"                   # pending | triggered | disabled | cancelled
    trigger_count: int = 0                    # 已触发次数

    # 时间
    created_at: datetime
    next_trigger_at: datetime                 # 下次触发时间（索引字段）
    last_triggered_at: datetime | None

    # 配置快照（创建时复制）
    delay_seconds: int
    max_triggers: int
    tool_name: str                            # 工具名称
    tool_params: dict                         # 工具参数
    message: str | None                       # 消息内容
```

### 3.4 存储结构

```
sessions (会话表)
├── session_id (PK)
├── tenant_id
├── chatbot_id
├── last_active_at
└── ...

timers (Timer 表) ← 独立表
├── timer_instance_id (PK)
├── session_id (FK, 索引)
├── timer_id
├── status (索引)
├── trigger_count
├── next_trigger_at (索引) ← 关键：便于超时查询
├── delay_seconds
├── max_triggers
├── tool_name
├── tool_params (JSON)
├── message
├── created_at
└── last_triggered_at
```

---

## 4. ActionType 精简

### 4.1 移除 SYSTEM 类型

```python
class ActionType(str, Enum):
    """Action types for workflow routing."""
    SKILL = "skill"                # Complex multi-step tasks
    TOOL = "tool"                  # Single tool calls (包含原 SYSTEM)
    FLOW = "flow"                  # Fixed business process APIs
    GENERATE_RESPONSE = "generate_response"  # Response to user
    # SYSTEM = "system"            # ❌ 移除，统一为 TOOL
```

### 4.2 Tool 执行分类（代码层面）

```python
class ToolExecutor:
    """工具执行器"""

    def __init__(self, config: WorkflowConfigSchema):
        self.config = config
        # 构建工具索引
        self.tools_map = {t["name"]: t for t in config.tools}
        self.system_actions = set(config.system_actions or [])
        self.agent_actions = set(config.agent_actions or [])

    def is_system_action(self, tool_name: str) -> bool:
        """判断是否为系统动作"""
        return tool_name in self.system_actions

    def is_agent_action(self, tool_name: str) -> bool:
        """判断是否为 Agent 动作"""
        return tool_name in self.agent_actions or tool_name not in self.system_actions

    async def execute(self, tool_calls: list[dict]) -> list[dict]:
        """执行工具调用"""
        system_calls = []
        agent_calls = []

        for call in tool_calls:
            if self.is_system_action(call["name"]):
                system_calls.append(call)
            else:
                agent_calls.append(call)

        # 并行执行 system_actions
        if system_calls:
            await asyncio.gather(*[self._execute_one(c) for c in system_calls])

        # 串行执行 agent_actions，收集结果
        results = []
        for call in agent_calls:
            result = await self._execute_one(call)
            results.append({"tool": call["name"], "result": result})

        return results

    async def _execute_one(self, call: dict) -> Any:
        """执行单个工具"""
        tool_def = self.tools_map.get(call["name"])
        if not tool_def:
            raise ValueError(f"Tool not found: {call['name']}")

        # HTTP 调用
        endpoint = tool_def.get("endpoint", {})
        # ... 执行逻辑
```

---

## 5. 生命周期设计

### 5.1 Timer 创建时机

```
用户首次消息 → Session 创建 → 检查 Agent.timers 配置 → 创建 SessionTimer 实例
```

```python
async def _init_session_timers(session_id: str, agent: WorkflowAgent, repos):
    """初始化 Session 的 Timer 实例"""
    if not agent.config.timers:
        return

    now = datetime.utcnow()
    for timer_config in agent.config.timers:
        # 检查是否已存在
        existing = await repos.timers.get_by_session_and_timer(
            session_id=session_id,
            timer_id=timer_config.timer_id,
        )
        if existing:
            continue

        # 创建 Timer 实例
        await repos.timers.create(
            session_id=session_id,
            timer_id=timer_config.timer_id,
            status="pending",
            next_trigger_at=now + timedelta(seconds=timer_config.delay_seconds),
            delay_seconds=timer_config.delay_seconds,
            max_triggers=timer_config.max_triggers,
            tool_name=timer_config.tool_name,
            tool_params=timer_config.tool_params,
            message=timer_config.message,
        )
```

### 5.2 Timer 重置时机

```
用户发送消息 → 更新 Session.last_active_at → 重置所有 pending Timer 的 next_trigger_at
```

```python
async def _reset_session_timers(session_id: str, repos):
    """重置 Session 的所有 Timer"""
    now = datetime.utcnow()

    # 更新 Session 活跃时间
    await repos.sessions.update(session_id, last_active_at=now)

    # 重置所有 pending Timer 的触发时间
    await repos.timers.reset_by_session(session_id, base_time=now)
```

### 5.3 TimerService 启动时机

**推荐：FastAPI Lifespan**

```python
# main.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    timer_service = create_timer_service(repos, tool_executor)
    await timer_service.start()

    yield

    # Shutdown
    await timer_service.stop()

app = FastAPI(lifespan=lifespan)
```

### 5.4 Timer 触发流程（统一 Tool 调用）

```python
class TimerService:
    """Timer 服务 - 统一 Tool 调用模型"""

    def __init__(self, repos, tool_executor: ToolExecutor):
        self._repos = repos
        self._tool_executor = tool_executor

    async def _execute_timer(self, timer: SessionTimer):
        """执行单个 Timer - 统一为 Tool 调用"""
        tool_name = timer.tool_name

        if tool_name == "generate_response":
            # 内置：生成响应消息
            await self._generate_response(timer)
        else:
            # 调用配置中的 Tool
            await self._tool_executor.execute([{
                "name": tool_name,
                "params": timer.tool_params,
            }])

        # 更新状态
        await self._update_timer_status(timer)

    async def _generate_response(self, timer: SessionTimer):
        """生成响应消息"""
        session = await self._repos.sessions.get(timer.session_id)
        if not session:
            return

        # 存储消息
        await self._repos.messages.create(
            session_id=timer.session_id,
            role="assistant",
            content=timer.message or "您好，请问还在吗？",
        )

        # TODO: 通过 WebSocket/Webhook 推送给客户端
```

---

## 6. 完整流程图

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

---

## 7. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| ActionType | 移除 SYSTEM，统一为 TOOL | 简化模型，减少复杂度 |
| Tool 分类 | system_actions / agent_actions | 配置驱动，代码层面区分执行方式 |
| Timer 存储 | 独立 timers 表 | 支持多 Timer/Session，查询高效 |
| 触发时间 | 存储 `next_trigger_at` | 避免运行时计算，查询简单 |
| Timer 动作 | 统一为 Tool 调用 | 复用现有 Tool 执行逻辑 |
| 扫描间隔 | 30s | 平衡精度与性能 |

---

## 8. 实现优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 移除 ActionType.SYSTEM | 精简枚举，统一为 TOOL |
| P0 | 更新 TimerConfig | 使用 tool_name + tool_params |
| P0 | 创建 ToolExecutor | 统一工具执行，区分 system/agent |
| P1 | 创建 timers 表 | 独立存储，支持多 Timer |
| P1 | TimerService 重构 | 使用 ToolExecutor |
| P2 | Lifespan 集成 | 应用启动时启动 TimerService |
| P2 | 单元测试 | 覆盖各场景 |

---

## 9. 优势

1. **模型统一**：所有动作都是 Tool，无需区分 ActionType
2. **配置驱动**：system_actions/agent_actions 通过配置区分
3. **执行灵活**：system_actions 并行，agent_actions 串行
4. **多 Timer 支持**：同一 Session 可配置多个独立 Timer
5. **查询高效**：`next_trigger_at` 索引，避免运行时计算
6. **复用逻辑**：Timer 触发复用 ToolExecutor

---

## 10. Timer 管理机制

### 10.1 资源控制

```python
class TimerConfig:
    """Timer 全局配置"""
    MAX_TIMERS_PER_SESSION = 10       # 每个 Session 最大 Timer 数
    MAX_CONCURRENT_TRIGGERS = 50      # 单次扫描最大触发数
    SCAN_INTERVAL = 30                # 扫描间隔（秒）
    BATCH_SIZE = 100                  # 批量查询大小
    TIMER_TTL_DAYS = 7                # Timer 过期天数
```

### 10.2 Timer 生命周期管理

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

### 10.3 清理策略

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

### 10.4 性能优化

| 优化点 | 策略 | 说明 |
|--------|------|------|
| **查询优化** | `next_trigger_at` 索引 | 避免全表扫描 |
| **批量处理** | 分批查询 + 并发执行 | 控制内存和连接数 |
| **限流** | 单次最大触发数 | 避免瞬时压力 |
| **去重** | 乐观锁 / 状态检查 | 避免重复触发 |

### 10.5 监控指标

```python
class TimerMetrics:
    """Timer 监控指标"""
    pending_count: int          # 待触发数
    triggered_count: int        # 已触发数
    avg_trigger_latency_ms: int # 平均触发延迟
    error_count: int            # 错误数
    last_scan_duration_ms: int  # 上次扫描耗时
```

---

## 11. 重置时机说明

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

这确保：
1. 用户体验不受影响（响应立即返回）
2. Timer 计时从用户收到响应后开始
3. 即使重置失败，不影响主流程

---

## 12. AgentPool 与 Timer 关系分析

### 12.1 核心问题

```
问题：Timer 是否应该在 AgentPool 环节管理？
      Agent 回收时，Timer 是否该释放？
```

### 12.2 关键概念澄清

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Agent vs Session vs Timer 关系                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AgentPool (内存)                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │  Agent A    │  │  Agent B    │  │  Agent C    │  ← 无状态实例     │   │
│  │  │  (chatbot1) │  │  (chatbot2) │  │  (chatbot3) │    可随时回收     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │   │
│  └─────────┼────────────────┼────────────────┼─────────────────────────┘   │
│            │                │                │                             │
│            │ 服务           │ 服务           │ 服务                        │
│            ▼                ▼                ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Sessions (持久化)                            │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ Sess 1  │ │ Sess 2  │ │ Sess 3  │ │ Sess 4  │ │ Sess 5  │       │   │
│  │  │(Agent A)│ │(Agent A)│ │(Agent B)│ │(Agent B)│ │(Agent C)│       │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  └───────┼──────────┼──────────┼──────────┼──────────┼────────────────┘   │
│          │          │          │          │          │                     │
│          ▼          ▼          ▼          ▼          ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Timers (持久化)                              │   │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐       │   │
│  │  │Timer 1│ │Timer 2│ │Timer 3│ │Timer 4│ │Timer 5│ │Timer 6│       │   │
│  │  │(Sess1)│ │(Sess1)│ │(Sess2)│ │(Sess3)│ │(Sess4)│ │(Sess5)│       │   │
│  │  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.3 设计决策：Timer 不应与 AgentPool 耦合

| 维度 | Agent (AgentPool) | Timer |
|------|-------------------|-------|
| **存储位置** | 内存缓存 | 数据库持久化 |
| **生命周期** | 可随时回收/重建 | 跟随 Session |
| **状态** | 无状态 | 有状态（trigger_count, status） |
| **粒度** | 1 Agent : N Sessions | 1 Session : N Timers |
| **重启恢复** | 按需重建 | 从 DB 恢复 |

**结论**：Timer 与 AgentPool 是**正交**的，不应耦合。

### 12.4 为什么 Agent 回收不应影响 Timer？

```
场景分析：

1. Agent A 服务 100 个活跃 Session
2. 每个 Session 有 2 个 Timer（共 200 个 Timer）
3. 由于内存压力，AgentPool 回收 Agent A

如果 Timer 与 Agent 耦合：
  ❌ 200 个 Timer 全部失效
  ❌ 用户体验严重受损
  ❌ 违背 Timer 的业务语义

正确设计：
  ✅ Agent A 被回收，Timer 不受影响
  ✅ 下次请求时，AgentPool 重建 Agent A
  ✅ Timer 继续正常触发
```

### 12.5 正确的生命周期管理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           生命周期独立管理                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AgentPool 生命周期：                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 首次请求  │───►│ 创建实例  │───►│ 缓存复用  │───►│ LRU 回收  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                             │
│  Timer 生命周期：                                                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Session  │───►│ 创建 Timer│───►│ 用户消息  │───►│ Session  │              │
│  │ 创建     │    │ (持久化)  │    │ 重置 Timer│    │ 关闭     │              │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘              │
│                                                       │                     │
│                                                       ▼                     │
│                                                 ┌──────────┐               │
│                                                 │ 取消 Timer│               │
│                                                 │ (cancelled)│              │
│                                                 └──────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 12.6 统一管理方案

```python
# 推荐架构：独立服务，统一管理

class ApplicationServices:
    """应用级服务管理"""

    def __init__(self):
        self.agent_pool = AgentPool()      # Agent 缓存
        self.timer_service = TimerService() # Timer 服务
        self.repos = RepositoryManager()    # 数据访问

    async def startup(self):
        """应用启动"""
        await self.timer_service.start()

    async def shutdown(self):
        """应用关闭"""
        await self.timer_service.stop()
        await self.agent_pool.clear()


# Session 关闭时的处理
async def close_session(session_id: str, services: ApplicationServices):
    """关闭 Session"""
    # 1. 取消该 Session 的所有 Timer
    await services.timer_service.cancel_session_timers(session_id)

    # 2. 更新 Session 状态
    await services.repos.sessions.update(session_id, status="closed")

    # 注意：不需要处理 AgentPool，Agent 是无状态的
```

### 12.7 关键设计原则

| 原则 | 说明 |
|------|------|
| **关注点分离** | AgentPool 管理 Agent 实例，TimerService 管理 Timer |
| **持久化优先** | Timer 状态存储在 DB，不依赖内存 |
| **Session 粒度** | Timer 跟随 Session 生命周期，不跟随 Agent |
| **独立服务** | TimerService 独立运行，不依赖 AgentPool |
| **优雅降级** | Agent 回收不影响 Timer，Timer 失败不影响 Agent |

### 12.8 边界情况处理

```python
# 场景 1: Agent 被回收，Timer 触发时需要 Agent
async def execute_timer_with_agent(timer: SessionTimer, services):
    """Timer 触发时获取 Agent"""
    session = await services.repos.sessions.get(timer.session_id)

    # 从 AgentPool 获取（如果已回收，会自动重建）
    agent = await services.agent_pool.get_or_create(
        tenant_id=session.tenant_id,
        chatbot_id=session.chatbot_id,
    )

    # 执行 Timer 动作
    await execute_timer_action(timer, agent)


# 场景 2: Session 关闭，清理 Timer
async def on_session_close(session_id: str, services):
    """Session 关闭回调"""
    await services.timer_service.cancel_session_timers(session_id)


# 场景 3: Agent 配置更新，已有 Timer 不受影响
# Timer 创建时已复制配置快照，配置更新只影响新 Timer
```

### 12.9 总结

```
┌─────────────────────────────────────────────────────────────────┐
│                         最佳实践总结                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Timer 与 AgentPool 解耦                                      │
│     - Timer 持久化存储，不依赖 Agent 内存实例                      │
│     - Agent 回收不影响 Timer                                     │
│                                                                 │
│  2. Timer 跟随 Session 生命周期                                   │
│     - Session 创建 → 初始化 Timer                                │
│     - Session 关闭 → 取消 Timer                                  │
│                                                                 │
│  3. 独立服务管理                                                  │
│     - AgentPool: 管理 Agent 实例缓存                             │
│     - TimerService: 管理 Timer 扫描和触发                        │
│     - 两者独立运行，互不干扰                                      │
│                                                                 │
│  4. 配置快照                                                     │
│     - Timer 创建时复制配置                                       │
│     - Agent 配置更新不影响已有 Timer                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```