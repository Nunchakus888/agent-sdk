# BU Agent SDK 架构分析

> _"An agent is just a for-loop."_ —— 项目核心理念

## 1. 设计范式概述

### 1.1 核心理念：The Bitter Lesson

本项目遵循 Rich Sutton 提出的 **"The Bitter Lesson"** (苦涩的教训) 理念：

> "All the value is in the RL'd model, not your 10,000 lines of abstractions."
>
> 所有价值都在经过强化学习训练的模型中，而不是你的 10,000 行抽象代码。

这一理念导出了三个核心设计原则：

| 原则 | 描述 |
|------|------|
| **极简主义** | 越少构建，越能工作 (The less you build, the more it works) |
| **完整行动空间** | Agent 失败不是因为模型弱，而是因为行动空间不完整 |
| **模型自主决策** | 给 LLM 尽可能大的自由度，然后基于评估进行约束 |

### 1.2 Agent 执行范式：模型控制驱动 (Model-Controlled Agent)

本 SDK 采用的是 **模型控制驱动** 范式，而非代码控制驱动。

#### 三种主流 Agent 范式对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Agent 执行范式对比                                │
├─────────────────┬────────────────────┬──────────────────────────────────┤
│ 代码控制驱动     │ 模型控制驱动 ✓     │ 混合驱动                          │
│ (Code-Driven)   │ (Model-Driven)     │ (Hybrid)                         │
├─────────────────┼────────────────────┼──────────────────────────────────┤
│ 预定义工作流     │ 模型决定下一步     │ 框架控制主流程                    │
│ 状态机/DAG      │ 循环 + 工具调用     │ 模型控制子任务                    │
│ 确定性执行      │ 非确定性执行        │ 半确定性执行                      │
├─────────────────┼────────────────────┼──────────────────────────────────┤
│ LangGraph       │ BU Agent SDK ✓     │ AutoGPT, BabyAGI                 │
│ Prefect         │ Claude Code        │ MetaGPT                          │
│ Airflow         │ Cursor Agent       │                                  │
└─────────────────┴────────────────────┴──────────────────────────────────┘
```

**BU Agent SDK 的选择理由**：现代 LLM（Claude, GPT-4, Gemini）已经过大量的强化学习训练（computer use, coding, browsing），它们不需要复杂的框架逻辑来引导，只需要：

1. 完整的行动空间（工具集）
2. 一个执行循环
3. 显式的退出机制
4. 上下文管理

## 2. 核心执行流程

### 2.1 The Agent Loop（Agent 循环）

项目的核心就是一个简单的 `while` 循环：

```
┌─────────────────────────────────────────────────────────────────┐
│                    The Agent Loop                               │
│                                                                 │
│    while True:                                                  │
│        response = await llm.invoke(messages, tools)             │
│        if not response.tool_calls:                              │
│            break                          ← 模型决定停止         │
│        for tool_call in response.tool_calls:                    │
│            result = await execute(tool_call)                    │
│            messages.append(result)        ← 结果反馈给模型       │
│                                                                 │
│    That's the entire agent framework.                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 详细执行流程图

```
                           ┌──────────────┐
                           │  User Query  │
                           └──────┬───────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  Add to Message History     │
                    │  (System Prompt + User Msg) │
                    └─────────────┬───────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │                    AGENT LOOP                      │
        │  ┌─────────────────────────────────────────────┐  │
        │  │                                             │  │
        │  │    ┌─────────────────────────────────┐      │  │
        │  │    │  Destroy Ephemeral Messages     │      │  │
        │  │    └─────────────┬───────────────────┘      │  │
        │  │                  │                          │  │
        │  │                  ▼                          │  │
        │  │    ┌─────────────────────────────────┐      │  │
        │  │    │     Invoke LLM                  │      │  │
        │  │    │  (messages + tool definitions) │      │  │
        │  │    └─────────────┬───────────────────┘      │  │
        │  │                  │                          │  │
        │  │                  ▼                          │  │
        │  │    ┌─────────────────────────────────┐      │  │
        │  │    │  Has Tool Calls?                │      │  │
        │  │    └─────────────┬───────────────────┘      │  │
        │  │                  │                          │  │
        │  │         ┌────────┴────────┐                 │  │
        │  │         │                 │                 │  │
        │  │        YES               NO                 │  │
        │  │         │                 │                 │  │
        │  │         ▼                 │                 │  │
        │  │    ┌──────────────┐       │                 │  │
        │  │    │ Execute      │       │                 │  │
        │  │    │ Tool Calls   │       │                 │  │
        │  │    └──────┬───────┘       │                 │  │
        │  │           │               │                 │  │
        │  │           ▼               │                 │  │
        │  │    ┌──────────────┐       │                 │  │
        │  │    │TaskComplete? │       │                 │  │
        │  │    └──────┬───────┘       │                 │  │
        │  │           │               │                 │  │
        │  │     ┌─────┴─────┐         │                 │  │
        │  │    YES          NO        │                 │  │
        │  │     │           │         │                 │  │
        │  │     │           ▼         │                 │  │
        │  │     │    ┌────────────┐   │                 │  │
        │  │     │    │ Compaction │   │                 │  │
        │  │     │    │ Check      │   │                 │  │
        │  │     │    └────┬───────┘   │                 │  │
        │  │     │         │           │                 │  │
        │  │     │         └───────────┼──► CONTINUE     │  │
        │  │     │                     │                 │  │
        │  └─────┼─────────────────────┼─────────────────┘  │
        │        │                     │                    │
        └────────┼─────────────────────┼────────────────────┘
                 │                     │
                 ▼                     ▼
        ┌─────────────────┐   ┌─────────────────┐
        │  Task Complete  │   │ Final Response  │
        │     Message     │   │    (No Tools)   │
        └─────────────────┘   └─────────────────┘
```

### 2.3 两种终止模式

| 模式 | 触发条件 | 适用场景 |
|------|----------|----------|
| **CLI 模式** (`require_done_tool=False`) | LLM 不再返回工具调用 | 交互式对话 |
| **自主模式** (`require_done_tool=True`) | 调用 `done` 工具抛出 `TaskComplete` | 自动化任务 |

```python
# CLI 模式：LLM 决定何时停止
agent = Agent(llm=llm, tools=tools)

# 自主模式：必须显式调用 done 工具
agent = Agent(llm=llm, tools=[..., done], require_done_tool=True)
```

## 3. 核心组件架构

### 3.1 系统架构图

```
┌───────────────────────────────────────────────────────────────────────┐
│                         BU Agent SDK                                  │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                          Agent Layer                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │  │
│  │  │   Agent     │  │  Events     │  │    Compaction           │  │  │
│  │  │  Service    │  │  System     │  │    Service              │  │  │
│  │  │             │  │             │  │                         │  │  │
│  │  │ • query()   │  │ • ToolCall  │  │ • Token tracking        │  │  │
│  │  │ • stream()  │  │ • ToolResult│  │ • Auto-summarization    │  │  │
│  │  │ • history   │  │ • Text      │  │ • Threshold management  │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                  │
│                                    ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                          LLM Layer                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │  │
│  │  │ BaseChatModel│ │  Messages   │  │    Serializers          │  │  │
│  │  │  (Protocol) │  │             │  │                         │  │  │
│  │  │             │  │ • User      │  │ • AnthropicSerializer   │  │  │
│  │  │ • ainvoke() │  │ • Assistant │  │ • OpenAISerializer      │  │  │
│  │  │ • provider  │  │ • System    │  │ • GoogleSerializer      │  │  │
│  │  │ • model     │  │ • Tool      │  │                         │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │  │
│  │         │                                                       │  │
│  │    ┌────┴────┬──────────────┬──────────────┐                    │  │
│  │    ▼         ▼              ▼              ▼                    │  │
│  │ ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐               │  │
│  │ │Anthropic│ │ OpenAI │ │ Google   │ │OpenAI-like │               │  │
│  │ │ChatModel│ │ChatModel│ │ChatModel │ │ (Custom)  │               │  │
│  │ └────────┘ └────────┘ └──────────┘ └────────────┘               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                  │
│                                    ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                         Tools Layer                             │  │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────┐   │  │
│  │  │      @tool Decorator    │  │    Dependency Injection     │   │  │
│  │  │                         │  │                             │   │  │
│  │  │ • Auto JSON Schema      │  │ • Depends() marker          │   │  │
│  │  │ • Type inference        │  │ • Annotated[T, Depends(fn)] │   │  │
│  │  │ • Docstring extraction  │  │ • Override support          │   │  │
│  │  │ • Pydantic integration  │  │ • Async resolution          │   │  │
│  │  │ • Ephemeral support     │  │                             │   │  │
│  │  └─────────────────────────┘  └─────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心类关系

```
                    ┌─────────────────────┐
                    │    BaseChatModel    │ ◄─── Protocol
                    │      (Protocol)     │
                    └─────────┬───────────┘
                              │ implements
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ChatAnthropic│    │ ChatOpenAI  │    │ ChatGoogle  │
    └─────────────┘    └─────────────┘    └─────────────┘


    ┌─────────────┐         ┌─────────────┐
    │   Agent     │────────►│    Tool     │
    │  (Service)  │    N    │  (Wrapper)  │
    └──────┬──────┘         └──────┬──────┘
           │                       │
           │ uses                  │ wraps
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │BaseChatModel│         │  Function   │
    └─────────────┘         │  (async)    │
                            └─────────────┘

    ┌─────────────────────────────────────────┐
    │              Message Types              │
    ├─────────────┬─────────────┬─────────────┤
    │ UserMessage │ Assistant   │ToolMessage  │
    │             │   Message   │             │
    └─────────────┴─────────────┴─────────────┘
            │             │            │
            └─────────────┼────────────┘
                          ▼
                    ┌───────────┐
                    │BaseMessage│
                    └───────────┘
```

## 4. 关键设计模式

### 4.1 Done Tool Pattern（显式完成模式）

**问题**：朴素的 "无工具调用时停止" 方法会导致 Agent 过早结束。

**解决方案**：使用 `TaskComplete` 异常强制显式完成。

```python
class TaskComplete(Exception):
    """通过 done 工具抛出，表示任务完成"""
    def __init__(self, message: str):
        self.message = message

@tool("Signal task completion")
async def done(message: str) -> str:
    raise TaskComplete(message)
```

**执行流程**：

```
Tool Call: done("Task finished successfully")
           │
           ▼
     ┌───────────────┐
     │ TaskComplete  │
     │  Exception    │
     └───────┬───────┘
             │
             ▼ (Caught by Agent Loop)
     ┌───────────────┐
     │ Return        │
     │ e.message     │
     └───────────────┘
```

### 4.2 Ephemeral Messages（临时消息模式）

**问题**：大型工具输出（浏览器状态、截图）会撑爆上下文。

**解决方案**：只保留最近 N 条输出。

```python
@tool("Get browser state", ephemeral=3)  # 只保留最近 3 条
async def get_state() -> str:
    return massive_dom_and_screenshot
```

**实现原理**：

```
Messages: [... tool_result_1, tool_result_2, tool_result_3, tool_result_4]
                    │                │              │              │
                 OLDEST           OLDER          RECENT         NEWEST
                    │                │              │              │
                    ▼                ▼              ▼              ▼
             ┌───────────┐   ┌───────────┐  ┌───────────┐  ┌───────────┐
             │ destroyed │   │ destroyed │  │   kept    │  │   kept    │
             │ = True    │   │ = True    │  │           │  │           │
             └───────────┘   └───────────┘  └───────────┘  └───────────┘
                    │                │
                    ▼                ▼
          "<removed to save context>"
```

### 4.3 Context Compaction（上下文压缩）

**问题**：长时间运行的 Agent 会超出模型的上下文窗口。

**解决方案**：当 token 使用接近阈值时，自动总结对话历史。

```python
agent = Agent(
    llm=llm,
    tools=tools,
    compaction=CompactionConfig(
        threshold_ratio=0.80,  # 80% 上下文窗口时触发
        enabled=True,
    ),
)
```

**压缩流程**：

```
                 ┌─────────────────────────────────────┐
                 │        Token Usage Check            │
                 │  current_tokens >= threshold?       │
                 └─────────────────┬───────────────────┘
                                   │
                              YES  │  NO
                                   │   └──► Continue normally
                                   ▼
                 ┌─────────────────────────────────────┐
                 │     Prepare Messages for Summary    │
                 │  (Remove pending tool_calls)        │
                 └─────────────────┬───────────────────┘
                                   │
                                   ▼
                 ┌─────────────────────────────────────┐
                 │        LLM Summarization            │
                 │   messages + summary_prompt         │
                 └─────────────────┬───────────────────┘
                                   │
                                   ▼
                 ┌─────────────────────────────────────┐
                 │     Replace History with Summary    │
                 │   messages = [UserMessage(summary)] │
                 └─────────────────────────────────────┘
```

### 4.4 Dependency Injection（依赖注入）

采用 FastAPI 风格的类型安全依赖注入：

```python
from typing import Annotated
from bu_agent_sdk import Depends

def get_db() -> Database:
    return Database()

@tool("Query database")
async def query(
    sql: str,
    db: Annotated[Database, Depends(get_db)]  # 自动注入
) -> str:
    return await db.execute(sql)

# 运行时覆盖（用于测试或作用域上下文）
agent = Agent(
    llm=llm,
    tools=[query],
    dependency_overrides={get_db: lambda: mock_db}
)
```

**解析流程**：

```
┌────────────────────────────────────────────────────────────┐
│                 Tool Execution                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Parse function signature                               │
│     └─► Annotated[Database, Depends(get_db)]               │
│                                                            │
│  2. Check dependency overrides                             │
│     └─► overrides.get(get_db) → mock_db or get_db          │
│                                                            │
│  3. Resolve dependency (sync or async)                     │
│     └─► await depends.resolve(overrides)                   │
│                                                            │
│  4. Inject into function call                              │
│     └─► await func(sql=sql, db=resolved_db)                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 5. Tool 系统实现机制详解

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Tool 系统架构                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   @tool decorator                                                           │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         Tool 类                                     │  │
│   │  ┌─────────────┬──────────────────┬───────────────────────────┐    │  │
│   │  │ func        │ description      │ name                      │    │  │
│   │  │ (原函数)     │ (工具描述)        │ (工具名)                   │    │  │
│   │  └─────────────┴──────────────────┴───────────────────────────┘    │  │
│   │  ┌─────────────┬──────────────────┬───────────────────────────┐    │  │
│   │  │ _definition │ _dependencies    │ _param_types              │    │  │
│   │  │ (JSON Schema)│ (依赖注入)       │ (参数类型映射)              │    │  │
│   │  └─────────────┴──────────────────┴───────────────────────────┘    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│        │                                                                    │
│        ├────────────► definition 属性 ──► ToolDefinition ──► LLM API       │
│        │                                                                    │
│        └────────────► execute() 方法 ──► 执行函数 + 依赖注入                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 核心组件详解

#### 5.2.1 @tool 装饰器

```python
@tool("Search the web for information", ephemeral=2)
async def search(query: str, max_results: int = 10) -> str:
    return f"Results for: {query}"
```

**装饰器参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `description` | `str` | 工具描述，发送给 LLM 用于决策 |
| `name` | `str \| None` | 工具名，默认使用函数名 |
| `ephemeral` | `int \| bool` | 临时输出保留数量（见上下文管理） |

**约束**：

```python
# ❌ 同步函数不允许
@tool("sync function")
def sync_tool() -> str:  # TypeError!
    return "result"

# ✅ 必须是 async 函数
@tool("async function")
async def async_tool() -> str:
    return "result"
```

#### 5.2.2 Tool 类

```python
@dataclass
class Tool:
    func: Callable[..., Awaitable[Any]]  # 原始异步函数
    description: str                      # 工具描述
    name: str                             # 工具名称
    ephemeral: int | bool                 # 输出保留策略
    _definition: ToolDefinition | None    # 缓存的 JSON Schema
    _dependencies: dict[str, Depends]     # 依赖注入映射
    _param_types: dict[str, type]         # 参数类型映射
```

**关键方法**：

| 方法 | 功能 |
|------|------|
| `_analyze_signature()` | 分析函数签名，提取参数类型和依赖 |
| `definition` (property) | 生成/缓存 ToolDefinition (JSON Schema) |
| `execute(**kwargs)` | 执行函数：解析依赖 → 实例化 Pydantic → 调用函数 → 序列化结果 |
| `_serialize_result()` | 结果序列化：str/dict/list/BaseModel → string/content parts |

### 5.3 类型系统与 JSON Schema 生成

#### 5.3.1 支持的类型映射

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Python 类型 → JSON Schema 映射                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Python Type          │  JSON Schema                                      │
│   ─────────────────────┼─────────────────────────────────────────────────  │
│   str                  │  {"type": "string"}                               │
│   int                  │  {"type": "integer"}                              │
│   float                │  {"type": "number"}                               │
│   bool                 │  {"type": "boolean"}                              │
│   None                 │  {"type": "null"}                                 │
│   list[T]              │  {"type": "array", "items": {T schema}}           │
│   dict[K, V]           │  {"type": "object", "additionalProperties": {V}}  │
│   Literal["a", "b"]    │  {"type": "string", "enum": ["a", "b"]}           │
│   Optional[T]          │  {T schema} (nullable handled by LLM)             │
│   Union[A, B]          │  {"anyOf": [{A schema}, {B schema}]}              │
│   BaseModel            │  model.model_json_schema() (optimized)            │
│   TypedDict            │  {"type": "object", "properties": {...}}          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 5.3.2 Pydantic 模型支持

```python
class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, ge=1, le=100)
    filters: list[str] = Field(default_factory=list)

@tool("Search with complex params")
async def search(params: SearchParams) -> str:
    return f"Searching: {params.query}"
```

**处理流程**：

```
1. 检测到 Pydantic 模型参数
2. 调用 SchemaOptimizer.create_optimized_json_schema(model)
3. 生成扁平化 schema（消除 $ref/$defs）
4. 添加 additionalProperties: false（OpenAI strict mode）
5. 设置所有属性为 required（OpenAI strict mode）
```

#### 5.3.3 SchemaOptimizer 优化策略

```python
SchemaOptimizer.create_optimized_json_schema(model)
```

**优化操作**：

| 操作 | 目的 |
|------|------|
| 展开 `$ref`/`$defs` | 内联所有引用，消除间接层 |
| 移除 `title` | 减少 schema 体积 |
| 保留完整 `description` | 保持工具描述完整性 |
| 添加 `additionalProperties: false` | OpenAI strict mode 兼容 |
| 所有属性设为 `required` | OpenAI strict mode 兼容 |

### 5.4 依赖注入系统

#### 5.4.1 使用方式

```python
# 方式1: Annotated + Depends (推荐)
@tool("Query database")
async def query(sql: str, db: Annotated[Database, Depends(get_db)]) -> str:
    return await db.execute(sql)

# 方式2: 默认值 (FastAPI 风格)
@tool("Query database")
async def query(sql: str, db = Depends(get_db)) -> str:
    return await db.execute(sql)
```

#### 5.4.2 Depends 类实现

```python
class Depends(Generic[T]):
    __slots__ = ('dependency',)
    
    def __init__(self, dependency: Callable[[], T | Awaitable[T]]) -> None:
        self.dependency = dependency
    
    async def resolve(self, overrides: DependencyOverrides | None = None) -> T:
        # 1. 检查是否有覆盖
        func = self.dependency
        if overrides and func in overrides:
            func = overrides[func]
        
        # 2. 调用并处理 sync/async
        result = func()
        if asyncio.iscoroutine(result):
            return await result
        return result
```

#### 5.4.3 依赖覆盖 (用于测试)

```python
# 生产代码
def get_database() -> Database:
    return ProductionDatabase()

# 测试时覆盖
agent = Agent(
    llm=llm,
    tools=[query_tool],
    dependency_overrides={
        get_database: lambda: MockDatabase()
    }
)
```

### 5.5 工具执行流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Tool 执行流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LLM 返回 tool_calls                                                        │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ for tool_call in response.tool_calls:      ← 顺序执行 (非并行)       │   │
│  │     tool_result = await _execute_tool_call(tool_call)                │   │
│  │     _messages.append(tool_result)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  _execute_tool_call() 内部流程:                                             │
│       │                                                                     │
│       ├── 1. 查找工具: tool = _tool_map.get(tool_name)                      │
│       │       └── 未找到 → ToolMessage(is_error=True)                       │
│       │                                                                     │
│       ├── 2. 解析参数: args = json.loads(tool_call.function.arguments)      │
│       │       └── JSON 错误 → ToolMessage(is_error=True)                    │
│       │                                                                     │
│       ├── 3. 执行工具: result = await tool.execute(_overrides, **args)      │
│       │       │                                                             │
│       │       ├── 解析依赖: await depends.resolve(overrides)                │
│       │       ├── 实例化 Pydantic (如需要)                                   │
│       │       ├── 调用原函数: await func(**call_kwargs)                     │
│       │       └── 序列化结果: _serialize_result()                           │
│       │                                                                     │
│       ├── 4. TaskComplete 异常 → 终止 Agent 循环                            │
│       │                                                                     │
│       └── 5. 其他异常 → ToolMessage(is_error=True)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.6 并发模型分析

#### 5.6.1 当前实现：顺序执行

```python
# agent/service.py:679-684
for tool_call in response.tool_calls:
    tool_result = await self._execute_tool_call(tool_call)  # 顺序 await
    self._messages.append(tool_result)
```

**问题**：LLM 可能返回多个 tool_calls，但当前是顺序执行。

#### 5.6.2 并发执行方案 (可扩展)

```python
# 并发执行方案
async def _execute_tool_calls_concurrent(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
    tasks = [self._execute_tool_call(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)

# 使用
results = await self._execute_tool_calls_concurrent(response.tool_calls)
for result in results:
    if isinstance(result, Exception):
        # 处理异常
        pass
    else:
        self._messages.append(result)
```

#### 5.6.3 并发 vs 顺序的权衡

| 维度 | 顺序执行 (当前) | 并发执行 |
|------|----------------|----------|
| **实现复杂度** | 简单 | 中等 |
| **错误处理** | 直观（一个失败立即可见） | 需要 gather + 异常处理 |
| **上下文一致性** | 严格顺序 | 可能乱序 |
| **执行效率** | O(n) 串行 | O(1) 并行（受 I/O 限制） |
| **资源消耗** | 低（单线程） | 高（多协程） |
| **适用场景** | 有依赖关系的工具 | 独立工具（如多个搜索） |

### 5.7 返回值处理

#### 5.7.1 支持的返回类型

```python
# 1. 字符串 (最常见)
@tool("Simple tool")
async def simple() -> str:
    return "result"

# 2. Pydantic 模型 (自动 JSON 序列化)
@tool("Structured result")
async def structured() -> MyModel:
    return MyModel(...)  # → model.model_dump_json()

# 3. dict/list (自动 JSON 序列化)
@tool("JSON result")
async def json_result() -> dict:
    return {"key": "value"}  # → json.dumps()

# 4. 多模态内容 (图片 + 文本)
@tool("Screenshot tool")
async def screenshot() -> list[ContentPartTextParam | ContentPartImageParam]:
    return [
        ContentPartTextParam(text="Screenshot taken"),
        ContentPartImageParam(image_url=ImageURL(url="data:image/png;base64,..."))
    ]

# 5. None (返回空字符串)
@tool("Void action")
async def void_action() -> None:
    do_something()  # → ""
```

### 5.8 优缺点分析

#### 5.8.1 优点

| 优点 | 说明 |
|------|------|
| **声明式定义** | `@tool` 装饰器极简，一行即可定义工具 |
| **类型安全** | 完整利用 Python 类型系统，IDE 智能提示 |
| **自动 Schema** | 从类型自动生成 JSON Schema，无需手动维护 |
| **依赖注入** | FastAPI 风格 DI，易于测试和模块化 |
| **多模态支持** | 原生支持图片/文档返回 |
| **Pydantic 集成** | 复杂参数用 Pydantic 模型，自动验证 |
| **缓存 Schema** | `_definition` 缓存避免重复生成 |
| **统一序列化** | 自动处理各种返回类型 |

#### 5.8.2 缺点

| 缺点 | 说明 |
|------|------|
| **强制 async** | 必须是异步函数，同步函数需包装 |
| **顺序执行** | 多工具调用顺序执行，非并行 |
| **类型限制** | 不支持泛型、Protocol 等高级类型 |
| **无重试机制** | 工具失败没有内置重试 |
| **无超时控制** | 长时间运行的工具可能阻塞 |
| **无资源限制** | 无并发数限制、内存限制 |

### 5.9 扩展性分析

#### 5.9.1 易扩展点

```python
# 1. 添加工具重试
class RetryTool(Tool):
    max_retries: int = 3
    
    async def execute(self, **kwargs):
        for i in range(self.max_retries):
            try:
                return await super().execute(**kwargs)
            except Exception:
                if i == self.max_retries - 1:
                    raise

# 2. 添加工具超时
import asyncio

class TimeoutTool(Tool):
    timeout: float = 30.0
    
    async def execute(self, **kwargs):
        return await asyncio.wait_for(
            super().execute(**kwargs), 
            timeout=self.timeout
        )

# 3. 添加工具钩子
class HookedTool(Tool):
    async def execute(self, **kwargs):
        await self.on_before_execute(kwargs)
        result = await super().execute(**kwargs)
        await self.on_after_execute(result)
        return result
```

#### 5.9.2 架构扩展方向

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        潜在扩展方向                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 并发执行器                                                               │
│     └── asyncio.gather() 或 TaskGroup 实现多工具并发                         │
│                                                                             │
│  2. 工具中间件                                                               │
│     └── 类似 FastAPI middleware: logging, auth, rate-limit                  │
│                                                                             │
│  3. 工具组合                                                                 │
│     └── 将多个工具组合为 ToolChain / ToolGraph                               │
│                                                                             │
│  4. 远程工具                                                                 │
│     └── 支持 HTTP/gRPC 远程工具调用                                          │
│                                                                             │
│  5. 工具版本管理                                                             │
│     └── 支持工具版本、AB 测试                                                │
│                                                                             │
│  6. 工具沙箱                                                                 │
│     └── 在隔离环境中执行危险工具                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.10 最佳实践

#### 5.10.1 工具设计原则

```python
# ✅ 单一职责
@tool("Read a file from disk")
async def read_file(path: str) -> str: ...

@tool("Write content to a file")
async def write_file(path: str, content: str) -> str: ...

# ❌ 职责混杂
@tool("File operations")
async def file_op(action: str, path: str, content: str = "") -> str: ...
```

#### 5.10.2 描述清晰

```python
# ✅ 描述完整，帮助 LLM 决策
@tool("Search the web using Google. Returns top 10 results with titles and snippets.")
async def search(query: str) -> str: ...

# ❌ 描述模糊
@tool("Search")
async def search(query: str) -> str: ...
```

#### 5.10.3 错误处理

```python
# ✅ 返回有意义的错误信息
@tool("Read file")
async def read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"Error: File '{path}' not found"
    except PermissionError:
        return f"Error: No permission to read '{path}'"

# ❌ 抛出异常（会被转为 is_error=True）
@tool("Read file")
async def read_file(path: str) -> str:
    return Path(path).read_text()  # 异常被 Agent 捕获
```

#### 5.10.4 大输出使用 ephemeral

```python
# ✅ 大输出标记为 ephemeral
@tool("Get webpage DOM", ephemeral=2)
async def get_dom() -> str:
    return massive_html  # 只保留最近 2 次

# ❌ 大输出不标记
@tool("Get webpage DOM")
async def get_dom() -> str:
    return massive_html  # 会撑爆上下文
```

### 5.11 配置驱动工具 (SaaS 模式)

对于 SaaS 场景，工具可以通过 JSON 配置文件动态定义，而非代码硬编码。

#### 5.11.1 配置 JSON Schema

```json
{
  "basic_settings": {
    "name": "Agent Name",
    "description": "Agent role description",
    "background": "Company/context background",
    "language": "English",
    "tone": "Friendly and professional",
    "chatbot_id": "optional_id"
  },
  "action_books": [
    {
      "condition": "When customer wants to schedule demo",
      "action": "Save their info using save_customer_information",
      "tools": ["save_customer_information"]
    }
  ],
  "tools": [
    {
      "name": "save_customer_information",
      "description": "Save customer info to CRM",
      "parameters": {
        "type": "object",
        "properties": {
          "email": {
            "type": "string",
            "description": "Customer email"
          },
          "name": {
            "type": "string",
            "description": "Customer name"
          }
        },
        "required": ["email"]
      },
      "endpoint": {
        "url": "http://api.example.com/contacts/add",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": {
          "email": "{email}",
          "name": "{name}",
          "dialogId": "todo_autofill_by_system"
        }
      }
    }
  ]
}
```

#### 5.11.2 Tool 配置结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Tool 配置 JSON Schema                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ToolConfig                                                                 │
│  ├── name: string          # 工具唯一标识                                    │
│  ├── description: string   # LLM 决策用描述                                  │
│  ├── parameters            # JSON Schema 定义                               │
│  │   ├── type: "object"                                                     │
│  │   ├── properties: {...} # 参数定义                                       │
│  │   │   └── paramName                                                      │
│  │   │       ├── type: string|integer|number|boolean|array|object           │
│  │   │       ├── description: string                                        │
│  │   │       ├── default: any                                               │
│  │   │       ├── enum: [...]      # 枚举值                                  │
│  │   │       └── examples: [...]  # 示例值                                  │
│  │   └── required: [...]   # 必填参数列表                                    │
│  └── endpoint              # HTTP 调用配置                                   │
│      ├── url: string       # API URL                                        │
│      ├── method: GET|POST|PUT|PATCH|DELETE                                  │
│      ├── headers: {...}    # HTTP 头                                        │
│      ├── body: {...}       # 请求体模板 (支持 {param} 占位符)                 │
│      ├── query_params: {}  # URL 查询参数模板                                │
│      └── timeout: number   # 超时秒数 (默认 30)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.11.3 占位符替换机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        占位符替换流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  配置模板:                                                                   │
│  {                                                                          │
│    "email": "{email}",           ← LLM 提供的参数                            │
│    "dialogId": "todo_autofill_by_system"  ← 系统上下文变量                   │
│  }                                                                          │
│                                                                             │
│  LLM 调用参数: {"email": "john@example.com"}                                │
│  系统上下文:   {"dialogId": "12345", "tenantId": "67890"}                   │
│                                                                             │
│                              ↓ 替换                                         │
│                                                                             │
│  实际请求:                                                                   │
│  {                                                                          │
│    "email": "john@example.com",                                             │
│    "dialogId": "12345"                                                      │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.11.4 使用方式

```python
from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import ConfigToolLoader

# 1. 加载配置
config = ConfigToolLoader.load_from_file("config/agent.json")

# 2. 获取工具 (带上下文变量)
tools = config.get_tools(context_vars={
    "dialogId": "dialog_123",
    "tenantId": "tenant_456",
    "chatbotId": config.basic_settings.chatbot_id,
})

# 3. 构建系统提示词
system_prompt = config.build_system_prompt()

# 4. 创建 Agent
agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    system_prompt=system_prompt,
)

# 5. 运行
response = await agent.query("I want to schedule a demo, my email is john@example.com")
```

#### 5.11.5 架构对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    代码定义 vs 配置定义                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  代码定义 (@tool 装饰器)              配置定义 (JSON)                         │
│  ┌───────────────────────────┐      ┌───────────────────────────┐          │
│  │ @tool("Save customer")    │      │ {"name": "save_customer", │          │
│  │ async def save(           │      │  "description": "Save...",│          │
│  │     email: str,           │  vs  │  "parameters": {...},     │          │
│  │     db: Depends(get_db)   │      │  "endpoint": {...}}       │          │
│  │ ) -> str:                 │      │                           │          │
│  │     await db.save(email)  │      └───────────────────────────┘          │
│  └───────────────────────────┘                                             │
│                                                                             │
│  优点:                               优点:                                   │
│  • 完全控制                          • 无需部署代码                          │
│  • 类型安全                          • 运行时动态加载                         │
│  • 依赖注入                          • 非技术人员可配置                       │
│  • IDE 支持                          • 多租户/SaaS 友好                      │
│                                                                             │
│  缺点:                               缺点:                                   │
│  • 需要部署                          • 仅支持 HTTP 调用                      │
│  • 需要开发者                        • 无依赖注入                            │
│                                      • 复杂逻辑难实现                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.11.6 混合模式

可以同时使用代码工具和配置工具：

```python
# 代码定义的工具
@tool("Complex analysis requiring local processing")
async def analyze_data(data: str, analyzer: Annotated[Analyzer, Depends(get_analyzer)]) -> str:
    return await analyzer.analyze(data)

# 配置定义的工具
config = ConfigToolLoader.load_from_file("config/tools.json")
http_tools = config.get_tools()

# 混合使用
agent = Agent(
    llm=llm,
    tools=[analyze_data] + http_tools,  # 合并两种工具
    system_prompt=system_prompt,
)
```

### 5.12 MCP (Model Context Protocol) 集成

MCP 是 Anthropic 提出的标准协议，用于统一 AI 模型与外部数据源、工具之间的交互。

#### 5.12.1 MCP 集成架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MCP 集成架构                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BU Agent SDK                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           Agent                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                      Tools List                              │    │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │   │
│  │  │  │ Native Tool  │  │ MCP Tool 1   │  │ HTTP Tool    │       │    │   │
│  │  │  │ (@tool)      │  │ (Adapter)    │  │ (Config)     │       │    │   │
│  │  │  └──────────────┘  └──────┬───────┘  └──────────────┘       │    │   │
│  │  └────────────────────────────┼─────────────────────────────────┘    │   │
│  └───────────────────────────────┼──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     MCPToolAdapter                                   │   │
│  │  • MCP 工具 → BU Agent SDK Tool 接口适配                             │   │
│  │  • 自动 Schema 转换                                                  │   │
│  │  • 结果格式化                                                        │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MCPClient                                     │   │
│  │  • JSON-RPC 2.0 协议实现                                             │   │
│  │  • 生命周期管理 (connect/disconnect)                                  │   │
│  │  • tools/resources/prompts API                                       │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Transport Layer                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ HTTP         │  │ SSE          │  │ Stdio        │               │   │
│  │  │ (请求/响应)   │  │ (流式)       │  │ (本地进程)    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    External MCP Servers                              │   │
│  │  • @anthropic/mcp-server-github                                      │   │
│  │  • @anthropic/mcp-server-filesystem                                  │   │
│  │  • Custom business MCP servers                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.12.2 核心组件

| 组件 | 职责 |
|------|------|
| `MCPClient` | MCP 协议客户端，支持多种传输方式 |
| `MCPToolAdapter` | 将 MCP 工具适配为 BU Agent SDK Tool 接口 |
| `MCPServiceLoader` | 管理多个 MCP 服务器连接 |
| `MCPTransport` | 传输层抽象（HTTP/SSE/Stdio） |

#### 5.12.3 使用方式

**方式 1: 单个 MCP 服务器**

```python
from bu_agent_sdk import Agent
from bu_agent_sdk.tools import MCPClient

# 连接到 MCP 服务器
async with MCPClient.from_url("http://localhost:8080") as client:
    # 获取工具
    tools = await client.get_tools()

    # 创建 Agent
    agent = Agent(llm=llm, tools=tools)
    response = await agent.query("Search for Python tutorials")
```

**方式 2: 多个 MCP 服务器**

```python
from bu_agent_sdk.tools import MCPServiceLoader, MCPTransportType

loader = MCPServiceLoader()

# 添加 HTTP 服务器
loader.add_server("api", "http://localhost:8080")

# 添加 Stdio 服务器 (npm 包)
loader.add_stdio_server(
    "github",
    command="npx",
    args=["-y", "@anthropic/mcp-server-github"],
    env={"GITHUB_TOKEN": "..."}
)

async with loader:
    # 获取所有工具
    all_tools = await loader.get_all_tools()
    agent = Agent(llm=llm, tools=all_tools)
```

**方式 3: JSON 配置**

```json
{
  "mcp_servers": [
    {
      "name": "browser",
      "transport": "http",
      "url": "http://localhost:3000",
      "timeout": 30
    },
    {
      "name": "github",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-github"],
      "env": {"GITHUB_TOKEN": "your-token"},
      "enabled": true
    }
  ]
}
```

```python
from bu_agent_sdk.tools import load_mcp_config

loader = load_mcp_config("config/mcp_servers.json")
async with loader:
    tools = await loader.get_all_tools()
```

**方式 4: 混合使用（Native + MCP + HTTP）**

```python
from bu_agent_sdk.tools import tool, ConfigToolLoader, MCPServiceLoader

# Native tool
@tool("Calculate sum")
async def add(a: int, b: int) -> str:
    return str(a + b)

# HTTP tools from config
config = ConfigToolLoader.load_from_file("config/http_tools.json")
http_tools = config.get_tools()

# MCP tools
loader = MCPServiceLoader()
loader.add_server("external", "http://localhost:8080")

async with loader:
    mcp_tools = await loader.get_all_tools()

    # 合并所有工具
    all_tools = [add] + http_tools + mcp_tools

    agent = Agent(llm=llm, tools=all_tools)
```

#### 5.12.4 传输层支持

| 传输类型 | 适用场景 | 特点 |
|----------|----------|------|
| **HTTP** | 远程 REST API | 简单请求/响应，无状态 |
| **SSE** | 远程流式 API | 支持服务器推送，适合长时间任务 |
| **Stdio** | 本地 npm/pip 包 | 通过 stdin/stdout 通信 |
| **WebSocket** | 双向实时通信 | 全双工，低延迟 (计划支持) |

#### 5.12.5 MCP 协议支持

| 功能 | 方法 | 说明 |
|------|------|------|
| **Tools** | `list_tools()`, `call_tool()` | 工具发现和调用 |
| **Resources** | `list_resources()`, `read_resource()` | 资源访问 |
| **Prompts** | `list_prompts()`, `get_prompt()` | 提示词模板 |
| **Sampling** | - | 服务器请求模型生成 (计划支持) |

#### 5.12.6 错误处理与降级

```python
loader = MCPServiceLoader()
loader.add_server("primary", "http://primary:8080")
loader.add_server("backup", "http://backup:8080")

async with loader:
    try:
        tools = await loader.get_tools_from("primary")
    except MCPError:
        # 降级到备用服务器
        tools = await loader.get_tools_from("backup")
```

#### 5.12.7 工具命名规范

MCP 工具会自动添加服务器名前缀以避免冲突：

```
原始 MCP 工具名: search
适配后名称: github__search  (格式: {server_name}__{tool_name})
```

## 6. 事件驱动架构

### 6.1 事件类型体系

```python
AgentEvent = (
    TextEvent           # LLM 生成文本
    | ThinkingEvent     # LLM 思考内容（Claude extended thinking）
    | ToolCallEvent     # 工具调用开始
    | ToolResultEvent   # 工具执行结果
    | FinalResponseEvent # 最终响应
    | StepStartEvent    # 步骤开始
    | StepCompleteEvent # 步骤完成
    | HiddenUserMessageEvent  # 隐藏用户消息（如 todo 检查提示）
)
```

### 6.2 事件流示例

```
User: "List files and read hello.py"
          │
          ▼
    ┌─────────────┐
    │ StepStart   │ step_id=1, title="glob_search"
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ ToolCall    │ tool="glob_search", args={"pattern": "*.py"}
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ ToolResult  │ result="hello.py\nutils.py"
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ StepComplete│ step_id=1, status="completed"
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ StepStart   │ step_id=2, title="read"
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ ToolCall    │ tool="read", args={"file_path": "hello.py"}
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ ToolResult  │ result="print('Hello')"
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ StepComplete│ step_id=2, status="completed"
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ TextEvent   │ content="Here are the Python files..."
    └──────┬──────┘
           ▼
    ┌─────────────────┐
    │ FinalResponse   │ content="..."
    └─────────────────┘
```

## 7. 意图识别机制

### 7.1 设计理念：模型原生意图识别

BU Agent SDK **没有独立的意图识别模块**，而是采用 **"模型原生意图识别"** 范式——将意图识别完全委托给 LLM 本身。

这是 "The Bitter Lesson" 理念在意图识别层面的体现：

> 不要用复杂的规则/模型去做意图识别，让预训练好的 LLM 直接处理。
> 你只需要提供清晰的工具描述（意图空间），让模型自己决定。

### 7.2 意图识别流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          意图识别流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用户输入                                                                   │
│   "帮我搜索 Python 教程并保存到文件"                                          │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │              LLM 调用 (ainvoke)                          │               │
│   │   ┌───────────────────────────────────────────────────┐ │               │
│   │   │ messages: [UserMessage("帮我搜索...")]             │ │               │
│   │   │ tools: [search_tool, write_tool, done_tool, ...]  │ │ ← 意图空间    │
│   │   │ tool_choice: "auto"                               │ │ ← 选择策略    │
│   │   └───────────────────────────────────────────────────┘ │               │
│   └─────────────────────────────┬───────────────────────────┘               │
│                                 │                                           │
│                                 ▼                                           │
│              ┌─────────────────────────────────────┐                        │
│              │      LLM 内部处理（黑盒）            │                        │
│              │   • 语义理解                         │                        │
│              │   • 意图推理                         │                        │
│              │   • 工具匹配                         │                        │
│              │   • 参数提取                         │                        │
│              └─────────────────┬───────────────────┘                        │
│                                │                                            │
│                                ▼                                            │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │           ChatInvokeCompletion 响应                      │               │
│   │   ┌───────────────────────────────────────────────────┐ │               │
│   │   │ tool_calls: [                                     │ │               │
│   │   │   ToolCall(                                       │ │               │
│   │   │     id="call_123",                                │ │               │
│   │   │     function=Function(                            │ │ ← 识别出的意图│
│   │   │       name="search",          ← 意图类型          │ │               │
│   │   │       arguments='{"query":"Python教程"}' ← 意图槽 │ │               │
│   │   │     )                                             │ │               │
│   │   │   )                                               │ │               │
│   │   │ ]                                                 │ │               │
│   │   │ content: None                                     │ │               │
│   │   └───────────────────────────────────────────────────┘ │               │
│   └─────────────────────────────────────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 核心概念映射

#### 工具定义 = 意图空间定义

```python
# bu_agent_sdk/llm/base.py
class ToolDefinition(BaseModel):
    name: str           # 意图名称 (e.g., "search", "write_file")
    description: str    # 意图描述 - LLM 根据此判断何时触发
    parameters: dict    # 意图槽位 (slots) - JSON Schema
    strict: bool = True
```

当定义一个工具时，本质上是在定义一种"意图类型"：

```python
@tool("Search the web for information")  # description = 意图触发条件
async def search(query: str) -> str:     # query = 意图槽位
    ...
```

#### 意图选择策略 (tool_choice)

```python
# bu_agent_sdk/llm/base.py
ToolChoice = Literal["auto", "required", "none"] | str
```

| tool_choice | 行为 | 类比传统 NLU |
|-------------|------|--------------|
| `"auto"` | 模型决定是否有意图 | 带 fallback 的意图分类 |
| `"required"` | 必须识别出一个意图 | 强制意图分类 |
| `"none"` | 不进行意图识别 | 纯闲聊模式 |
| `"tool_name"` | 强制特定意图 | 固定意图路由 |

#### 意图识别结果

```python
# bu_agent_sdk/llm/views.py
class ChatInvokeCompletion(BaseModel):
    content: str | None = None      # 无意图时的纯文本响应
    tool_calls: list[ToolCall] = [] # 识别出的意图列表
    
    @property
    def has_tool_calls(self) -> bool:
        """是否识别出意图"""
        return len(self.tool_calls) > 0

# bu_agent_sdk/llm/messages.py
class Function(BaseModel):
    name: str       # 意图类型 (intent type)
    arguments: str  # 意图槽位值 (slot values) - JSON 格式

class ToolCall(BaseModel):
    id: str
    function: Function  # 包含意图类型和槽位
```

### 7.4 Agent Loop 中的意图分发

```python
# bu_agent_sdk/agent/service.py (简化)

# 检查是否识别出意图
if not response.has_tool_calls:    # ← 无意图识别
    if not self.require_done_tool:
        return response.content or ""
    continue

# 有意图，执行对应动作
for tool_call in response.tool_calls:
    tool_result = await self._execute_tool_call(tool_call)
    self._messages.append(tool_result)
```

### 7.5 与传统 NLU 意图识别对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    传统 NLU vs BU Agent SDK                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  传统 NLU 流水线:                                                        │
│  ┌────────┐   ┌─────────────┐   ┌────────────┐   ┌────────────┐        │
│  │ 分词   │ → │ 意图分类器   │ → │  槽位填充   │ → │ 动作执行   │        │
│  │(Tokenize)│ │(Classifier) │   │(Slot Filling)│  │(Dispatch)  │        │
│  └────────┘   └─────────────┘   └────────────┘   └────────────┘        │
│       ↓              ↓               ↓                                  │
│    词向量        softmax 概率     CRF/BERT        规则引擎              │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  BU Agent SDK:                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         LLM (黑盒)                               │   │
│  │   用户输入 ──────────────────────────────────────► tool_calls    │   │
│  │             (意图理解 + 槽位提取 + 动作选择)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 详细对比

| 维度 | 传统 NLU | BU Agent SDK |
|------|----------|--------------|
| **意图定义** | 训练数据 + 标注 | `ToolDefinition.description` |
| **意图分类** | 分类模型 (BERT/RNN) | LLM 原生能力 |
| **槽位提取** | CRF/BERT + 规则 | LLM + JSON Schema |
| **泛化能力** | 受限于训练数据 | 强泛化（预训练知识）|
| **新意图** | 需要重新训练 | 添加 `@tool` 装饰器即可 |
| **多意图** | 需要额外设计 | 自动支持（多个 tool_calls）|
| **可解释性** | 概率分数 | 黑盒 |
| **延迟** | 低 (ms 级) | 较高 (100ms-1s) |
| **成本** | 低 (本地推理) | 较高 (API 调用) |

### 7.6 优缺点分析

**优点**：

- ✅ **零样本意图识别** - 不需要训练数据
- ✅ **强泛化能力** - 理解同义表达、口语化表达
- ✅ **添加新意图极其简单** - 只需定义 `@tool`
- ✅ **自动处理多意图** - 多个 tool_calls
- ✅ **自动槽位提取** - JSON Schema 约束
- ✅ **上下文感知** - 基于对话历史理解

**缺点**：

- ❌ **意图识别过程不可解释** - 无法获取置信度
- ❌ **无法精确控制分类阈值** - 不能设置 fallback 阈值
- ❌ **依赖 LLM 的稳定性** - 可能有随机性
- ❌ **成本较高** - 每次都要调用 LLM API
- ❌ **延迟较高** - 相比本地分类模型

### 7.7 最佳实践

#### 1. 编写清晰的工具描述

```python
# ❌ 不好：描述模糊
@tool("Search")
async def search(query: str) -> str: ...

# ✅ 好：描述清晰、具体
@tool("Search the web for information using Google. Use when the user asks about facts, news, or needs to look up information.")
async def search(query: str) -> str: ...
```

#### 2. 使用 JSON Schema 约束槽位

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="The search query to look up")
    max_results: int = Field(default=10, description="Maximum number of results")
    
@tool("Search the web")
async def search(params: SearchParams) -> str: ...
```

#### 3. 利用 tool_choice 控制意图选择

```python
# 强制必须选择一个工具（适合任务型场景）
agent = Agent(llm=llm, tools=tools, tool_choice="required")

# 自动决定（适合混合场景）
agent = Agent(llm=llm, tools=tools, tool_choice="auto")

# 强制特定工具（适合单一任务）
response = await llm.ainvoke(messages, tools=tools, tool_choice="search")
```

### 7.8 多类型意图路由实现

当需要支持多种意图类型（Skills、Tools、Flows、Message）时，可以通过以下架构实现：

#### 架构设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         多类型意图路由架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用户输入: "帮我写一篇博客" / "北京天气" / "我要请假" / "你好"                 │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    预匹配层 (Pre-Match)                          │      │
│   │   • 正则匹配 Flows 触发词                                         │      │
│   │   • 命中 → 直接路由到 Flow Engine                                 │      │
│   │   • 未命中 → 进入 LLM 路由                                        │      │
│   └───────────────────────────┬─────────────────────────────────────┘      │
│                               │                                            │
│                               ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    Router Agent (LLM 路由)                       │      │
│   │                                                                 │      │
│   │   tools: [                                                      │      │
│   │     dispatch_to_skill,   # → Skill Agent (子Agent)              │      │
│   │     dispatch_to_tool,    # → Tool 直接执行                       │      │
│   │     dispatch_to_flow,    # → Flow Engine (状态机)                │      │
│   │     respond_message,     # → 直接文本回复                         │      │
│   │   ]                                                             │      │
│   └───────────────────────────┬─────────────────────────────────────┘      │
│                               │                                            │
│          ┌────────────────────┼────────────────────┐                       │
│          ▼                    ▼                    ▼                       │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │   Skills    │      │   Tools     │      │   Flows     │                │
│   │             │      │             │      │             │                │
│   │ blog_writer │      │ weather     │      │ leave_req   │                │
│   │ code_review │      │ calculate   │      │ reimburse   │                │
│   │ ...         │      │ translate   │      │ ...         │                │
│   │             │      │ ...         │                    │                │
│   │ (子Agent)   │      │ (直接执行)   │      │ (状态机)    │                │
│   └─────────────┘      └─────────────┘      └─────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 意图类型定义

| 类型 | 描述 | 实现方式 | 示例 |
|------|------|----------|------|
| **Skills** | 复杂多步骤能力 | 子 Agent | 写博客、代码审查 |
| **Tools** | 单一功能调用 | 直接执行 | 天气查询、计算 |
| **Flows** | 固定模式工作流 | 状态机 + 预匹配 | 请假、报销 |
| **Message** | 纯文本对话 | 直接回复 | 闲聊、问答 |

#### 核心实现

```python
# 1. 定义路由工具
@tool("调用复杂技能")
async def dispatch_to_skill(skill_name: str, request: str, ctx: ...) -> str:
    """当用户需要完成复杂任务时，启动子 Agent"""
    skill = ctx.skills[skill_name]
    skill_agent = Agent(llm=ctx.llm, tools=skill.tools, system_prompt=skill.prompt)
    return await skill_agent.query(request)

@tool("调用单一工具")
async def dispatch_to_tool(tool_name: str, args: dict, ctx: ...) -> str:
    """当用户需要执行简单功能时，直接调用工具"""
    return await ctx.tools[tool_name].execute(**args)

@tool("启动固定流程")
async def dispatch_to_flow(flow_name: str, ctx: ...) -> str:
    """当用户触发标准化流程时，启动状态机"""
    return ctx.flows[flow_name].start()

@tool("直接回复消息")
async def respond_message(content: str) -> str:
    """当用户只是闲聊时，直接回复"""
    raise TaskComplete(content)

# 2. 创建路由 Agent
router = Agent(
    llm=llm,
    tools=[dispatch_to_skill, dispatch_to_tool, dispatch_to_flow, respond_message],
    system_prompt=ROUTER_PROMPT,  # 包含路由规则
    dependency_overrides={get_context: lambda: ctx},
)

# 3. 带预匹配的路由器
class IntentRouter:
    async def route(self, user_input: str) -> str:
        # 预匹配 Flow（正则）
        if flow := match_flow(user_input):
            return flow.start()
        # LLM 路由
        return await self.router_agent.query(user_input)
```

#### 路由优先级

```
1. Flow 预匹配 (正则命中 → 100% 准确, 零成本)
   ↓ 未命中
2. LLM 路由决策
   ├── Tool (简单功能) → 直接执行
   ├── Skill (复杂任务) → 启动子 Agent
   └── Message (对话) → 直接回复
```

完整示例代码见：[`bu_agent_sdk/examples/intent_router.py`](../bu_agent_sdk/examples/intent_router.py)

## 8. 上下文管理与多轮对话

### 8.1 上下文构成要素

BU Agent SDK 的上下文由以下几部分构成：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM 上下文构成                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1. System Prompt (系统提示)                       │   │
│  │   • 定义 Agent 身份和行为准则                                         │   │
│  │   • 首次调用时添加，支持 Anthropic 缓存                               │   │
│  │   • 通过 agent.system_prompt 配置                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    2. Message History (消息历史)                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ UserMessage      │ 用户输入（文本/图片/文档）                  │   │   │
│  │   ├──────────────────┼──────────────────────────────────────────┤   │   │
│  │   │ AssistantMessage │ 模型回复 + tool_calls                     │   │   │
│  │   ├──────────────────┼──────────────────────────────────────────┤   │   │
│  │   │ ToolMessage      │ 工具执行结果（文本/图片/错误）              │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    3. Tool Definitions (工具定义)                    │   │
│  │   • 每次调用都完整发送所有工具定义                                     │   │
│  │   • 包含：name, description, parameters (JSON Schema)               │   │
│  │   • Anthropic: 最后 N 个工具支持缓存                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 消息类型详解

#### 消息结构

```python
# 基础消息类型
BaseMessage = Union[
    UserMessage,      # 用户消息
    SystemMessage,    # 系统提示
    AssistantMessage, # 助手回复
    ToolMessage,      # 工具结果
    DeveloperMessage, # 开发者指令 (OpenAI o1+)
]
```

#### 各消息类型字段

| 消息类型 | 核心字段 | 说明 |
|----------|----------|------|
| **UserMessage** | `content: str \| list[Text\|Image\|Document]` | 支持多模态输入 |
| **AssistantMessage** | `content`, `tool_calls`, `thinking` | 回复 + 工具调用 + 思考过程 |
| **ToolMessage** | `tool_call_id`, `tool_name`, `content`, `is_error`, `ephemeral`, `destroyed` | 工具执行结果 |
| **SystemMessage** | `content`, `cache` | 系统提示，支持缓存 |

#### 内容类型 (Content Parts)

```python
# 文本内容
ContentPartTextParam(text="Hello", type="text")

# 图片内容 (支持 URL 或 base64)
ContentPartImageParam(
    image_url=ImageURL(
        url="data:image/png;base64,...",  # 或 https://...
        detail="auto",  # auto/low/high
        media_type="image/png"
    ),
    type="image_url"
)

# 文档内容 (PDF)
ContentPartDocumentParam(
    source=DocumentSource(data="base64...", media_type="application/pdf"),
    type="document"
)

# 思考内容 (Claude extended thinking)
ContentPartThinkingParam(thinking="...", signature="...", type="thinking")
```

### 8.3 多轮对话流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         多轮对话上下文演进                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  第1轮: agent.query("帮我搜索 Python 教程")                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ _messages = [                                                       │   │
│  │   SystemMessage(content="You are a helpful assistant...", cache=T), │   │
│  │   UserMessage(content="帮我搜索 Python 教程"),                        │   │
│  │ ]                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓ LLM 调用                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ _messages = [                                                       │   │
│  │   SystemMessage(...),                                               │   │
│  │   UserMessage(content="帮我搜索 Python 教程"),                        │   │
│  │   AssistantMessage(                                                 │   │
│  │     content=None,                                                   │   │
│  │     tool_calls=[ToolCall(id="1", function=search(query="Python"))]  │   │
│  │   ),                                                                │   │
│  │ ]                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓ 工具执行                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ _messages = [                                                       │   │
│  │   SystemMessage(...),                                               │   │
│  │   UserMessage(content="帮我搜索 Python 教程"),                        │   │
│  │   AssistantMessage(tool_calls=[...]),                               │   │
│  │   ToolMessage(                                                      │   │
│  │     tool_call_id="1",                                               │   │
│  │     tool_name="search",                                             │   │
│  │     content="找到 10 个结果: 1. Python 官方教程..."                   │   │
│  │   ),                                                                │   │
│  │ ]                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓ LLM 第二次调用                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ _messages = [                                                       │   │
│  │   ...(同上),                                                        │   │
│  │   AssistantMessage(content="我找到了以下 Python 教程: ..."),          │   │
│  │ ]                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  第2轮: agent.query("第一个链接是什么")                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ _messages = [                                                       │   │
│  │   SystemMessage(...),                         ← 保留                 │   │
│  │   UserMessage("帮我搜索..."),                  ← 历史轮次              │   │
│  │   AssistantMessage(tool_calls=[search...]),   ← 历史轮次              │   │
│  │   ToolMessage(search result...),              ← 历史轮次 (上下文来源)  │   │
│  │   AssistantMessage("我找到了..."),             ← 历史轮次              │   │
│  │   UserMessage("第一个链接是什么"),             ← 新输入                │   │
│  │ ]                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LLM 可以从历史 ToolMessage 中获取上下文来回答问题                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 上下文管理机制

#### 7.4.1 Ephemeral Messages (临时消息)

大型工具输出（截图、DOM 状态）会撑爆上下文。Ephemeral 机制只保留最近 N 条：

```python
@tool("Get browser state", ephemeral=3)  # 只保留最近 3 条
async def get_state() -> str:
    return massive_dom_and_screenshot
```

**处理流程**：

```
调用 _destroy_ephemeral_messages() 时机: 每次 LLM 调用前

┌─────────────────────────────────────────────────────────────────────────────┐
│                     Ephemeral Message 处理流程                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Before:                                                                    │
│  _messages = [..., ToolMsg(get_state, ephemeral=True), ...                  │
│                   ToolMsg(get_state, ephemeral=True),                       │
│                   ToolMsg(get_state, ephemeral=True),                       │
│                   ToolMsg(get_state, ephemeral=True)]  ← 4 条               │
│                                                                             │
│  After (ephemeral=3):                                                       │
│  _messages = [..., ToolMsg(destroyed=True),  ← 内容替换为占位符              │
│                   ToolMsg(get_state, ...),    ← 保留                        │
│                   ToolMsg(get_state, ...),    ← 保留                        │
│                   ToolMsg(get_state, ...)]    ← 保留                        │
│                                                                             │
│  序列化时 destroyed=True 的消息:                                             │
│  content = "<removed to save context>"                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.4.2 Context Compaction (上下文压缩)

当 token 使用接近模型上下文窗口时，自动总结历史：

```python
agent = Agent(
    llm=llm,
    tools=tools,
    compaction=CompactionConfig(
        threshold_ratio=0.80,  # 80% 时触发
        summary_prompt="...",   # 总结提示词
    ),
)
```

**压缩后的消息结构**：

```
Before compaction:
_messages = [SystemMsg, UserMsg, AssistantMsg, ToolMsg, ..., UserMsg, AssistantMsg]
             ↑ 大量历史消息

After compaction:
_messages = [UserMessage(content="<summary>
  ## Task Overview
  用户请求帮助搜索 Python 教程并保存到文件...
  
  ## Current State
  已完成搜索，找到 10 个结果...
  
  ## Next Steps
  需要将结果保存到文件...
</summary>")]
```

#### 7.4.3 Prompt Caching (提示缓存)

针对 Anthropic 的缓存优化：

```python
# System prompt 缓存
SystemMessage(content="...", cache=True)

# 只有最后一个 cache=True 的消息会被缓存
# AnthropicMessageSerializer._clean_cache_messages() 自动处理
```

### 8.5 意图上下文在多轮对话中的体现

当前 SDK 采用 **隐式意图延续** 模式：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     意图上下文延续方式                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  方式1: 隐式延续 (当前 SDK 默认)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • LLM 从消息历史中推断当前意图上下文                                   │   │
│  │ • 优点: 简单，无需额外管理                                            │   │
│  │ • 缺点: 不可控，长对话可能丢失上下文                                   │   │
│  │                                                                     │   │
│  │ 示例:                                                               │   │
│  │ User: "帮我写一篇博客"        → 意图: blog_writer                     │   │
│  │ Assistant: [调用 outline 工具]                                       │   │
│  │ User: "加一个总结"            → LLM 从历史推断仍在 blog_writer 上下文  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  方式2: 显式状态注入 (需要扩展实现)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • 在每次 LLM 调用前注入结构化状态                                      │   │
│  │ • 可以通过 system_prompt 动态更新或隐藏消息实现                         │   │
│  │                                                                     │   │
│  │ 示例实现:                                                            │   │
│  │ class StatefulAgent(Agent):                                         │   │
│  │     def _build_context_message(self) -> str:                        │   │
│  │         return f'''## 当前执行状态                                    │   │
│  │         - 活跃意图: {self.current_intent}                            │   │
│  │         - 已完成步骤: {self.completed_steps}                         │   │
│  │         - 当前步骤: {self.current_step}                              │   │
│  │         - 待执行步骤: {self.pending_steps}'''                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  方式3: HiddenUserMessage 注入 (SDK 已支持)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • 使用 _get_incomplete_todos_prompt() 钩子注入状态                    │   │
│  │ • 通过 HiddenUserMessageEvent 通知但不显示给用户                       │   │
│  │                                                                     │   │
│  │ 代码位置: agent/service.py:664-671                                   │   │
│  │ if not incomplete_todos_prompted:                                   │   │
│  │     incomplete_prompt = await self._get_incomplete_todos_prompt()   │   │
│  │     if incomplete_prompt:                                           │   │
│  │         self._messages.append(UserMessage(content=incomplete_prompt))│   │
│  │         yield HiddenUserMessageEvent(content=incomplete_prompt)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.6 上下文生命周期

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        上下文生命周期管理                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  创建 Agent                                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  agent = Agent(llm, tools, system_prompt)                                   │
│      │                                                                      │
│      │ _messages = []  (空)                                                 │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        agent.query() 循环                           │   │
│  │                                                                     │   │
│  │  1. 首次调用: 添加 SystemMessage (if system_prompt)                  │   │
│  │  2. 添加 UserMessage                                                │   │
│  │  3. while iterations < max_iterations:                              │   │
│  │     ├── _destroy_ephemeral_messages()  ← 清理临时消息                 │   │
│  │     ├── _invoke_llm()                  ← 调用 LLM                    │   │
│  │     ├── 添加 AssistantMessage                                       │   │
│  │     ├── 执行 tool_calls → 添加 ToolMessage(s)                        │   │
│  │     └── _check_and_compact()           ← 检查是否需要压缩            │   │
│  │  4. 返回最终响应                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│      │                                                                      │
│      │ _messages 保留完整历史                                               │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     后续调用 agent.query()                           │   │
│  │                                                                     │   │
│  │  • _messages 继续累积                                                │   │
│  │  • LLM 可以访问完整历史上下文                                         │   │
│  │  • 直到 clear_history() 或 compaction 触发                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│      │                                                                      │
│      ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        上下文管理操作                                │   │
│  │                                                                     │   │
│  │  • agent.clear_history()  → 清空 _messages                          │   │
│  │  • agent.load_history()   → 加载历史消息 (恢复会话)                   │   │
│  │  • agent.messages         → 获取当前消息历史 (只读)                   │   │
│  │  • compaction 触发        → _messages 替换为总结                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.7 跨会话上下文持久化

```python
# 保存会话状态
messages = agent.messages  # 获取消息历史
# 序列化并存储到数据库...

# 恢复会话状态
agent = Agent(llm=llm, tools=tools, system_prompt=system_prompt)
agent.load_history(messages)  # 加载历史
response = await agent.query("继续之前的任务...")
```

## 9. 与其他框架对比

### 9.1 设计理念对比

| 特性 | BU Agent SDK | LangChain | AutoGPT |
|------|--------------|-----------|---------|
| **核心理念** | 极简主义，信任模型 | 模块化，可组合 | 自主代理 |
| **控制流** | 模型控制 | 代码/模型混合 | 模型控制 + 规划 |
| **抽象层级** | 最小化 | 高度抽象 | 中等抽象 |
| **工具定义** | 装饰器 + 类型推断 | 多种方式 | 插件系统 |
| **上下文管理** | Compaction | Memory 模块 | 向量存储 |
| **学习曲线** | 低 | 高 | 中 |

### 9.2 代码量对比

```
BU Agent SDK:
├── Agent Loop:    ~150 lines
├── LLM Wrapper:   ~300 lines/provider
├── Tool System:   ~400 lines
└── Total Core:    ~1,000 lines

vs.

LangChain:
└── Total Core:    ~50,000+ lines

AutoGPT:
└── Total Core:    ~20,000+ lines
```

## 10. 使用场景建议

### 10.1 适合使用 BU Agent SDK 的场景

✅ **Coding Assistants** - 文件操作、代码生成
✅ **Browser Automation** - 网页操作、数据抓取
✅ **CLI Tools** - 命令行交互式任务
✅ **Simple Workflows** - 需要工具调用的简单任务
✅ **快速原型** - 需要快速验证 Agent 想法

### 10.2 可能需要其他框架的场景

❓ **复杂工作流** - 需要 DAG、条件分支、人工审批
❓ **多 Agent 协作** - 需要 Agent 间通信
❓ **长期记忆** - 需要向量存储、知识图谱
❓ **企业级需求** - 需要复杂的监控、审计、回滚

## 11. 总结

### 11.1 设计范式定性

**BU Agent SDK 是一个典型的「模型控制驱动」(Model-Controlled) Agent 框架**，其特点是：

1. **控制权完全交给模型** - 代码只提供执行环境
2. **最小化抽象** - 核心就是一个 for-loop
3. **完整行动空间** - 通过丰富的工具集赋能模型
4. **显式边界** - 明确的任务完成机制

### 11.2 核心价值

```
┌─────────────────────────────────────────────────────────────┐
│                    核心价值主张                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    "Every abstraction is a liability.                       │
│     Every 'helper' is a failure point."                     │
│                                                             │
│     每一层抽象都是负债。                                      │
│     每一个"帮助函数"都是潜在的故障点。                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    模型已经足够强大。它们需要的是：                            │
│                                                             │
│    ✓ 完整的行动空间 (Complete action space)                  │
│    ✓ 一个执行循环 (A for-loop)                               │
│    ✓ 显式的退出机制 (An explicit exit)                       │
│    ✓ 上下文管理 (Context management)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

*文档版本: 1.5*
*最后更新: 2026-01-21*
*[基于 BU Agent SDK 源码分析](https://github.com/browser-use/agent-sdk)*
