# Workflow Executor 优化总结

## 一、核心设计决策

### 1.1 工具模块使用策略

基于对现有 `bu_agent_sdk/tools/` 模块的深入分析，我们采用以下策略：

| 场景 | 实现方式 | LLM可见性 | 原因 |
|------|---------|----------|------|
| **Skills (function模式)** | `HttpTool` | ✅ 可见 | LLM需要决策何时调用，结果进入上下文 |
| **普通Tools** | `HttpTool` | ✅ 可见 | LLM需要决策何时调用，结果进入上下文 |
| **Knowledge Retrieval** | `HttpTool` | ✅ 可见 | LLM需要决策何时查询，结果进入上下文 |
| **Flows (业务流程)** | 手动实现 | ❌ 不可见 | 需要silent模式，直接返回用户，跳出迭代 |
| **System Actions** | 手动实现 | ❌ 不可见 | 需要silent模式，特殊业务逻辑，跳出迭代 |

### 1.2 决策原则

```
需要LLM可见并参与决策？
├─ Yes → 使用 Tool 模块 (HttpTool)
│   ✅ 自动生成 JSON Schema
│   ✅ 统一的错误处理
│   ✅ 上下文变量注入
│   ✅ 与 Agent 无缝集成
│
└─ No → 手动实现 Executor
    ✅ 支持 silent 模式 (返回 None)
    ✅ 自定义响应处理
    ✅ 可以跳出迭代循环
    ✅ 特殊业务逻辑处理
```

## 二、实现架构

### 2.1 模块结构

```
bu_agent_sdk/workflow/
├── __init__.py
└── executors.py
    ├── FlowExecutor          # 手动实现 (LLM不可见)
    ├── SystemExecutor        # 手动实现 (LLM不可见)
    ├── SkillMatcher          # 意图匹配
    ├── WorkflowOrchestrator  # 统一编排
    └── Tool Registration Helpers
        ├── create_knowledge_retrieval_tool()  # 使用 HttpTool
        └── create_skill_function_tool()       # 使用 HttpTool
```

### 2.2 WorkflowOrchestrator 职责分离

```python
class WorkflowOrchestrator:
    """
    职责分离：
    1. Tool 模块：处理 LLM 可见的工具
    2. 手动实现：处理 LLM 不可见的操作
    """

    def __init__(self, config, context_vars, http_client):
        # LLM 不可见的执行器
        self.flow_executor = FlowExecutor(...)
        self.system_executor = SystemExecutor(...)

        # 意图匹配
        self.skill_matcher = SkillMatcher(...)

    def get_tools(self) -> list[Tool]:
        """获取所有 LLM 可见的工具"""
        tools = []

        # 1. Skills (function模式) - HttpTool
        for skill in self.config.skills:
            if skill.execution_mode == "function":
                tools.append(create_skill_function_tool(skill))

        # 2. 普通工具 - HttpTool
        tools.extend(load_tools_from_config())

        # 3. 知识库检索 - HttpTool
        if self.config.retrieve_knowledge_url:
            tools.append(create_knowledge_retrieval_tool())

        return tools

    async def execute_flow(self, flow_id, ...):
        """执行 Flow (LLM 不可见)"""
        return await self.flow_executor.execute(...)

    async def execute_system_action(self, action_id, ...):
        """执行 System Action (LLM 不可见)"""
        return await self.system_executor.execute(...)
```

## 三、关键特性

### 3.1 Silent 模式支持

```python
# Flow 或 System Action 可以返回 None
result = await orchestrator.execute_flow("leave_request", ...)

if result is None:
    # Silent 执行，跳出迭代循环
    break
else:
    # 返回响应给用户
    return result
```

### 3.2 上下文变量注入

```python
# 所有工具和执行器共享上下文变量
context_vars = {
    "dialogId": "dialog_12345",
    "tenantId": "tenant_67890",
    "chatbotId": "chatbot_123",
    "phoneNumber": "+1234567890",
}

orchestrator = WorkflowOrchestrator(
    config=config,
    context_vars=context_vars,
)

# 自动注入到所有 HTTP 请求中
# body: {"dialogId": "{dialogId}"} → {"dialogId": "dialog_12345"}
```

### 3.3 HTTP 客户端复用

```python
# 共享 HTTP 客户端，提升性能
http_client = httpx.AsyncClient(timeout=30.0)

orchestrator = WorkflowOrchestrator(
    config=config,
    http_client=http_client,
)

# 所有 HttpTool、FlowExecutor、SystemExecutor 共享同一个客户端
```

## 四、配置示例

### 4.1 完整配置

```json
{
  "basic_settings": {
    "name": "Customer Service Agent",
    "description": "Help customers with inquiries"
  },

  "skills": [
    {
      "skill_id": "sentiment_analysis",
      "execution_mode": "function",
      "description": "Analyze sentiment",
      "endpoint": {
        "url": "http://api.example.com/sentiment",
        "method": "POST"
      }
    }
  ],

  "tools": [
    {
      "name": "save_customer_information",
      "description": "Save customer info",
      "endpoint": {...}
    }
  ],

  "retrieve_knowledge_url": "http://kb-api.example.com/retrieve",

  "flows": [
    {
      "flow_id": "leave_request",
      "name": "Leave Request",
      "endpoint": {...},
      "silent": false
    }
  ],

  "system_actions": [
    {
      "action_id": "transfer_human",
      "handler": "handoff",
      "silent": false,
      "endpoint": {...}
    },
    {
      "action_id": "update_profile",
      "handler": "update_profile",
      "silent": true,
      "endpoint": {...}
    }
  ]
}
```

### 4.2 使用示例

```python
from bu_agent_sdk.workflow import WorkflowOrchestrator, load_workflow_config
from bu_agent_sdk.agent import Agent
from bu_agent_sdk.llm.anthropic import AnthropicChatModel

# 1. 加载配置
config = load_workflow_config(config_dict)

# 2. 创建编排器
orchestrator = WorkflowOrchestrator(
    config=config,
    context_vars={"dialogId": "123", "tenantId": "456"},
)

# 3. 获取 LLM 可见的工具
tools = orchestrator.get_tools()

# 4. 创建 Agent
agent = Agent(
    llm=AnthropicChatModel(model="claude-sonnet-4-5-20250929"),
    tools=tools,
    system_prompt=orchestrator.build_system_prompt(),
)

# 5. 使用 Agent (LLM 可见的工具会自动调用)
response = await agent.query("Analyze this text sentiment")

# 6. 手动执行 Flow (LLM 不可见)
result = await orchestrator.execute_flow(
    flow_id="leave_request",
    user_message="我要请假",
)

# 7. 手动执行 System Action (LLM 不可见)
result = await orchestrator.execute_system_action(
    action_id="transfer_human",
    parameters={"assigneeId": "agent_123"},
)
```

## 五、优势总结

### 5.1 复用现有架构

✅ **充分利用 Tool 模块**
- `HttpTool` 用于配置驱动的 HTTP 工具
- 自动生成 JSON Schema
- 统一的错误处理
- 上下文变量注入

✅ **与 Agent 无缝集成**
- 所有 LLM 可见的工具都是标准 Tool
- Agent 可以直接使用
- 不需要修改 Agent 代码

### 5.2 清晰的职责分离

✅ **LLM 可见 vs 不可见**
- 可见：Skills (function), Tools, KB → Tool 模块
- 不可见：Flows, System Actions → 手动实现

✅ **Silent 模式支持**
- Flow 和 System Action 可以返回 None
- 跳出迭代循环
- 不污染 Agent 上下文

### 5.3 灵活性和可扩展性

✅ **配置驱动**
- 所有工具和执行器都从配置加载
- 易于维护和扩展

✅ **性能优化**
- HTTP 客户端复用
- Tool 定义缓存
- 支持并行执行

## 六、与 v9 文档的对应关系

### 6.1 Skills 执行模式

| v9 文档 | 实现方式 |
|---------|---------|
| Agent 模式 | 动态创建子 Agent (未在本次实现) |
| Function 模式 | `create_skill_function_tool()` → HttpTool |

### 6.2 Flow 执行

| v9 文档 | 实现方式 |
|---------|---------|
| 直接 API 调用 | `FlowExecutor.execute()` |
| Silent 模式 | 返回 None |
| 跳出迭代 | 检查返回值是否为 None |

### 6.3 System Actions

| v9 文档 | 实现方式 |
|---------|---------|
| handoff | `SystemExecutor._handoff()` |
| close | `SystemExecutor._close_conversation()` |
| update_profile | `SystemExecutor._update_profile()` |
| Silent 模式 | 返回 None |

## 七、后续优化方向

### 7.1 Skills Agent 模式

```python
# TODO: 实现 Skills (agent 模式)
def create_skill_agent_tool(skill: SkillConfig) -> Tool:
    """创建 Skill Agent 工具"""

    @tool(skill.description)
    async def skill_agent(**kwargs) -> str:
        # 创建子 Agent
        sub_agent = Agent(
            llm=llm,
            tools=load_skill_tools(skill.tools),
            system_prompt=skill.system_prompt,
            max_iterations=skill.max_iterations,
        )

        # 执行子 Agent
        result = await sub_agent.query(...)
        return result

    return skill_agent
```

### 7.2 并行 KB 查询

```python
# TODO: 实现 KB 预查询优化
async def _iteration_step(self, user_message: str):
    # 并行执行 KB 查询和 LLM 决策
    kb_task = asyncio.create_task(self._query_kb(user_message))
    decision_task = asyncio.create_task(self._llm_decision(user_message))

    kb_result, decision = await asyncio.gather(kb_task, decision_task)
```

### 7.3 Timer 调度器

```python
# TODO: 实现 Timer 调度器
class TimerScheduler:
    """基于 asyncio 的定时器调度器"""

    async def schedule(self, session_id: str, timers: list[TimerConfig]):
        for timer in timers:
            task = asyncio.create_task(
                self._delayed_trigger(session_id, timer)
            )
            self._tasks[f"{session_id}:{timer.timer_id}"] = task
```

## 八、文件清单

### 8.1 核心实现

- ✅ `bu_agent_sdk/workflow/executors.py` - 核心执行器实现
- ✅ `bu_agent_sdk/workflow/__init__.py` - 模块导出

### 8.2 文档

- ✅ `docs/workflow-tool-strategy-analysis.md` - 工具策略分析
- ✅ `docs/workflow-executor-implementation.md` - 实现文档 (旧版)
- ✅ `docs/workflow-executor-optimization-summary.md` - 本文档

### 8.3 示例

- ✅ `examples/workflow_optimized_example.py` - 优化后的示例
- ✅ `examples/workflow_orchestrator_example.py` - 旧版示例

## 九、运行示例

```bash
# 运行优化后的示例
python examples/workflow_optimized_example.py

# 输出：
# - Tool 注册策略演示
# - Agent 集成演示
# - 手动执行演示
# - 对比分析
```

## 十、总结

本次优化的核心思想是：

1. **充分复用现有 Tool 模块** - 用于 LLM 可见的工具
2. **手动实现特殊场景** - 用于 LLM 不可见的操作
3. **清晰的职责分离** - 可见 vs 不可见，标准 vs 特殊
4. **保持架构一致性** - 与现有 Agent 架构无缝集成

这种设计既利用了现有基础设施的优势，又保持了足够的灵活性来处理特殊场景。
