# Workflow Agent 工具模块使用策略分析

## 一、核心问题

在 Workflow Agent 实现中，需要决定：
1. **哪些场景使用现有的 `Tool` 模块**
2. **哪些场景手动实现 HTTP 请求**
3. **如何平衡灵活性和复用性**

## 二、现有 Tool 模块能力分析

### 2.1 Tool 模块的核心优势

从 `bu_agent_sdk/tools/` 模块分析：

```python
# 1. @tool 装饰器 - 用于 Python 函数
@tool("Search the web")
async def search(query: str) -> str:
    return "results"

# 优势：
# ✅ 自动生成 JSON Schema
# ✅ 类型安全（基于 Python 类型提示）
# ✅ 依赖注入支持（Depends）
# ✅ Pydantic 模型支持
# ✅ 与 Agent 无缝集成
```

```python
# 2. HttpTool - 用于配置驱动的 HTTP API
from bu_agent_sdk.tools.config_loader import HttpTool

tool = HttpTool(
    config=ToolConfig(...),
    context_vars={"dialogId": "123"},
    http_client=client,
)

# 优势：
# ✅ 配置驱动，无需编写代码
# ✅ 模板替换（{param} 占位符）
# ✅ 上下文变量注入
# ✅ 统一的错误处理
# ✅ 与 Agent 兼容
```

### 2.2 Tool 模块的局限性

```python
# 局限性：
# ❌ 必须返回 ToolContent (str | list[ContentPart])
# ❌ 必须符合 Tool 接口（name, description, definition, execute）
# ❌ 执行结果会进入 Agent 的消息历史
# ❌ 不适合"静默执行"场景（silent actions）
# ❌ 不适合需要特殊响应处理的场景
```

## 三、Workflow 场景分类与策略

### 3.1 场景分类矩阵

| 场景类型 | LLM可见性 | 执行方式 | 响应处理 | 推荐方案 |
|---------|----------|---------|---------|---------|
| **Skills (Agent模式)** | ✅ 可见 | 子Agent | 标准 | 使用 Tool |
| **Skills (Function模式)** | ✅ 可见 | HTTP调用 | 标准 | 使用 HttpTool |
| **Tools (普通工具)** | ✅ 可见 | 函数/HTTP | 标准 | 使用 Tool/HttpTool |
| **Flow (业务流程)** | ❌ 不可见 | HTTP调用 | Silent | 手动实现 |
| **System (系统动作)** | ❌ 不可见 | 特殊处理 | Silent/自定义 | 手动实现 |
| **Knowledge Retrieval** | ✅ 可见 | HTTP调用 | 标准 | 使用 HttpTool |

### 3.2 详细策略分析

#### 策略 1: Skills - 使用 Tool 模块 ✅

**Agent 模式 (子Agent执行)**

```python
# ✅ 推荐：使用 @tool 装饰器
from bu_agent_sdk.tools import tool

@tool("博客写作助手 - 撰写完整的博客文章")
async def blog_writer_skill(
    topic: str,
    style: str = "professional"
) -> str:
    """
    创建子 Agent 执行博客写作任务
    """
    # 创建子 Agent
    sub_agent = Agent(
        llm=llm,
        tools=[search_kb, save_draft, done],
        system_prompt="你是专业博客写手...",
        max_iterations=20,
    )

    try:
        result = await sub_agent.query(f"写一篇关于{topic}的文章，风格：{style}")
        return result
    except TaskComplete as e:
        return e.message

# 优势：
# ✅ LLM 可以看到这个 skill 并决定何时调用
# ✅ 参数自动验证（topic, style）
# ✅ 结果自动进入上下文，供后续决策使用
# ✅ 与现有 Agent 架构完美集成
```

**Function 模式 (HTTP调用外部服务)**

```python
# ✅ 推荐：使用 HttpTool
from bu_agent_sdk.tools.config_loader import HttpTool, ToolConfig

# 从配置创建
skill_tool = HttpTool(
    config=ToolConfig(
        name="sentiment_analysis",
        description="分析文本情感倾向",
        parameters=ToolParameters(
            type="object",
            properties={
                "text": ToolParameterProperty(
                    type="string",
                    description="待分析文本"
                )
            },
            required=["text"]
        ),
        endpoint=EndpointConfig(
            url="http://api.example.com/sentiment",
            method="POST",
            body={"text": "{text}"}
        )
    ),
    context_vars={"tenantId": "123"},
)

# 优势：
# ✅ 配置驱动，易于维护
# ✅ LLM 可见，参与决策
# ✅ 统一的错误处理
# ✅ 支持上下文变量注入
```

#### 策略 2: Tools (普通工具) - 使用 Tool 模块 ✅

```python
# ✅ 推荐：使用 @tool 装饰器或 HttpTool

# 方式1：Python 函数工具
@tool("保存客户信息到数据库")
async def save_customer_info(
    name: str,
    email: str,
    country: str | None = None,
    db: Annotated[Database, Depends(get_db)] = None
) -> str:
    await db.insert("customers", {
        "name": name,
        "email": email,
        "country": country
    })
    return f"✅ 已保存客户信息：{name} ({email})"

# 方式2：HTTP API 工具（从配置加载）
from bu_agent_sdk.tools.config_loader import ConfigToolLoader

loader = ConfigToolLoader.load_from_file("config.json")
tools = loader.get_tools(context_vars={"dialogId": "123"})

# 优势：
# ✅ LLM 可以智能选择何时调用
# ✅ 参数自动验证
# ✅ 依赖注入支持（如数据库连接）
# ✅ 结果进入上下文
```

#### 策略 3: Flow (业务流程) - 手动实现 HTTP ⚠️

**为什么不用 Tool？**

```python
# ❌ 问题：Flow 是"静默"操作
# - Flow 执行后不应该返回给 LLM
# - Flow 执行后应该直接跳出迭代循环
# - Flow 的响应格式可能需要特殊处理
# - Flow 可能需要特殊的错误处理逻辑

# ✅ 推荐：手动实现 FlowExecutor
class FlowExecutor:
    """
    流程执行器 - 直接 API 调用

    特点：
    - 不注册为 Tool（LLM 不可见）
    - 通过规则匹配触发（trigger_patterns）
    - 执行后直接返回给用户，不进入 Agent 上下文
    - 支持 silent 模式
    """

    def __init__(self, config: WorkflowConfigSchema):
        self.config = config
        self._http_client = httpx.AsyncClient()

    async def execute(
        self,
        flow_id: str,
        user_message: str,
        parameters: dict,
    ) -> str | None:
        """
        执行流程

        Returns:
            str: 响应内容（非 silent）
            None: 静默执行（silent）
        """
        flow = self._get_flow(flow_id)

        # 构建请求
        endpoint = flow.endpoint
        body = self._substitute_parameters(
            endpoint["body"],
            {"user_message": user_message, **parameters}
        )

        # 发送请求
        response = await self._http_client.request(
            method=endpoint["method"],
            url=endpoint["url"],
            headers=endpoint.get("headers", {}),
            json=body,
        )

        # 处理响应
        if flow.silent:
            return None  # 静默执行，不返回内容

        if response.is_success:
            result = response.json()
            # 使用自定义模板
            if flow.response_template:
                return flow.response_template.format(result=result)
            return str(result)
        else:
            return f"❌ 流程执行失败: {response.status_code}"

# 优势：
# ✅ 完全控制执行流程
# ✅ 支持 silent 模式
# ✅ 自定义响应处理
# ✅ 不污染 Agent 上下文
# ✅ 可以直接跳出迭代循环
```

#### 策略 4: System Actions - 手动实现 ⚠️

```python
# ✅ 推荐：手动实现 SystemExecutor
class SystemExecutor:
    """
    系统动作执行器

    处理：
    - handoff (转人工)
    - close_conversation (关闭会话)
    - update_profile (更新用户信息)

    特点：
    - 不注册为 Tool
    - 支持 silent 模式
    - 特殊的业务逻辑处理
    """

    async def execute(
        self,
        action_id: str,
        parameters: dict,
    ) -> str | None:
        """
        执行系统动作

        Returns:
            str: 响应内容（非 silent）
            None: 静默执行（silent）
        """
        action = self._get_action(action_id)

        # 根据 handler 类型执行
        if action.handler == "handoff":
            result = await self._handoff(action, parameters)
        elif action.handler == "close":
            result = await self._close_conversation(action, parameters)
        elif action.handler == "update_profile":
            result = await self._update_profile(action, parameters)

        # 静默模式：返回 None
        if action.silent:
            return None

        # 非静默：返回响应
        return result

    async def _handoff(self, action: SystemAction, params: dict) -> str:
        """转人工"""
        # 调用转人工 API
        await self._call_handoff_api(params)

        # 返回自定义响应
        return action.response_template or "正在为您转接人工服务..."

    async def _update_profile(self, action: SystemAction, params: dict) -> str:
        """更新用户信息（通常是 silent）"""
        # 调用更新 API
        await self._call_update_api(params)

        # Silent 模式下返回 None
        return "信息已更新"

# 优势：
# ✅ 支持 silent 模式
# ✅ 特殊业务逻辑处理
# ✅ 不进入 Agent 上下文
# ✅ 可以触发副作用（如转人工、关闭会话）
```

#### 策略 5: Knowledge Retrieval - 使用 HttpTool ✅

```python
# ✅ 推荐：使用 HttpTool
knowledge_tool = HttpTool(
    config=ToolConfig(
        name="retrieve_knowledge",
        description="从知识库检索相关信息",
        parameters=ToolParameters(
            type="object",
            properties={
                "keywords": ToolParameterProperty(
                    type="string",
                    description="搜索关键词"
                )
            },
            required=["keywords"]
        ),
        endpoint=EndpointConfig(
            url="http://kb-api.example.com/retrieve",
            method="GET",
            body={
                "chatbotId": "{chatbotId}",
                "tenantId": "{tenantId}",
                "keywords": "{keywords}"
            }
        )
    ),
    context_vars={
        "chatbotId": "123",
        "tenantId": "456"
    }
)

# 优势：
# ✅ LLM 可以决定何时查询知识库
# ✅ 查询结果进入上下文
# ✅ 支持上下文变量注入
# ✅ 统一的错误处理
```

## 四、最佳实践总结

### 4.1 决策树

```
需要 LLM 可见并参与决策？
├─ Yes → 使用 Tool 模块
│   ├─ Python 函数？ → @tool 装饰器
│   ├─ HTTP API？ → HttpTool
│   └─ 复杂参数？ → Pydantic 模型 + @tool
│
└─ No → 手动实现
    ├─ 需要 silent 模式？ → 手动实现 Executor
    ├─ 需要特殊响应处理？ → 手动实现 Executor
    ├─ 需要跳出迭代循环？ → 手动实现 Executor
    └─ 需要触发副作用？ → 手动实现 Executor
```

### 4.2 推荐架构

```python
# bu_agent_sdk/workflow/executors.py

class WorkflowOrchestrator:
    """
    工作流编排器

    职责分离：
    1. Tool 模块：处理 LLM 可见的工具（Skills, Tools, KB）
    2. 手动实现：处理 LLM 不可见的操作（Flow, System）
    """

    def __init__(self, config: WorkflowConfigSchema):
        self.config = config

        # 1. 使用 Tool 模块的部分
        self.tools = self._register_tools()  # Skills + Tools + KB

        # 2. 手动实现的部分
        self.flow_executor = FlowExecutor(config)
        self.system_executor = SystemExecutor(config)

    def _register_tools(self) -> list[Tool]:
        """注册所有 LLM 可见的工具"""
        tools = []

        # 1. Skills (Agent 模式) - 使用 @tool
        for skill in self.config.skills:
            if skill.execution_mode == "agent":
                tools.append(self._create_skill_agent_tool(skill))

        # 2. Skills (Function 模式) - 使用 HttpTool
        for skill in self.config.skills:
            if skill.execution_mode == "function":
                tools.append(self._create_skill_function_tool(skill))

        # 3. Tools - 使用 HttpTool
        loader = ConfigToolLoader(self.config)
        tools.extend(loader.get_tools(context_vars=self.context_vars))

        # 4. Knowledge Retrieval - 使用 HttpTool
        if self.config.retrieve_knowledge_url:
            tools.append(self._create_knowledge_tool())

        return tools

    async def execute_action(
        self,
        action_type: ActionType,
        action_target: str,
        parameters: dict,
    ) -> tuple[str | None, bool]:
        """
        执行动作

        Returns:
            (result, should_exit_iteration)
        """
        if action_type == ActionType.SKILL:
            # Skills 通过 Agent 的 tool 调用机制执行
            # 这里不需要特殊处理
            return None, False

        elif action_type == ActionType.TOOL:
            # Tools 通过 Agent 的 tool 调用机制执行
            return None, False

        elif action_type == ActionType.FLOW:
            # Flow 手动执行
            result = await self.flow_executor.execute(
                flow_id=action_target,
                parameters=parameters,
            )
            # Silent flow 返回 None，跳出迭代
            return result, (result is None)

        elif action_type == ActionType.SYSTEM:
            # System 手动执行
            result = await self.system_executor.execute(
                action_id=action_target,
                parameters=parameters,
            )
            # Silent system action 返回 None，跳出迭代
            return result, (result is None)
```

### 4.3 配置示例

```json
{
  "skills": [
    {
      "skill_id": "blog_writer",
      "execution_mode": "agent",
      "system_prompt": "你是专业博客写手",
      "tools": ["search_kb", "save_draft"]
    },
    {
      "skill_id": "sentiment_analysis",
      "execution_mode": "function",
      "endpoint": {
        "url": "http://api.example.com/sentiment",
        "method": "POST"
      }
    }
  ],
  "tools": [
    {
      "name": "save_customer_information",
      "description": "保存客户信息",
      "endpoint": {...}
    }
  ],
  "flows": [
    {
      "flow_id": "leave_request",
      "trigger_patterns": ["我要请假"],
      "endpoint": {...},
      "silent": false
    }
  ],
  "system_actions": [
    {
      "action_id": "transfer_human",
      "handler": "handoff",
      "silent": false
    },
    {
      "action_id": "update_profile",
      "handler": "update_profile",
      "silent": true
    }
  ]
}
```

## 五、性能优化建议

### 5.1 HTTP 客户端复用

```python
class WorkflowOrchestrator:
    def __init__(self, config: WorkflowConfigSchema):
        # ✅ 共享 HTTP 客户端
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # 传递给所有需要的组件
        self.tools = self._register_tools(http_client=self._http_client)
        self.flow_executor = FlowExecutor(config, http_client=self._http_client)
        self.system_executor = SystemExecutor(config, http_client=self._http_client)
```

### 5.2 工具定义缓存

```python
# Tool 模块已经实现了定义缓存
@property
def definition(self) -> ToolDefinition:
    if self._definition is not None:
        return self._definition

    # 生成并缓存
    self._definition = ToolDefinition(...)
    return self._definition
```

### 5.3 并行执行

```python
# KB 预查询优化
async def _iteration_step(self, user_message: str):
    # 并行执行 KB 查询和 LLM 决策
    kb_task = asyncio.create_task(self._query_knowledge(user_message))
    decision_task = asyncio.create_task(self._llm_decision(user_message))

    kb_result, decision = await asyncio.gather(kb_task, decision_task)

    # 如果需要 KB 结果，直接使用缓存
    if decision.needs_kb:
        return kb_result
```

## 六、总结

### 使用 Tool 模块的场景 ✅
1. **Skills (Agent 模式)** - @tool 装饰器
2. **Skills (Function 模式)** - HttpTool
3. **普通工具** - @tool 或 HttpTool
4. **知识库检索** - HttpTool

### 手动实现 HTTP 的场景 ⚠️
1. **Flow (业务流程)** - FlowExecutor
2. **System Actions** - SystemExecutor

### 核心原则
- **LLM 可见 → Tool 模块**
- **LLM 不可见 → 手动实现**
- **需要 silent 模式 → 手动实现**
- **需要特殊处理 → 手动实现**
- **标准工具调用 → Tool 模块**
