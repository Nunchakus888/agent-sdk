# Knowledge Base 和 Flow 可见性设计分析

## 一、核心问题分析

### 1.1 Knowledge Base (KB) 可见性

**场景描述：**
- KB 是在每次迭代开始时**预先查询**（并行执行）
- 查询结果缓存，LLM 决策时可以直接使用
- 从 v9 文档看：`KB预查询 → 并行执行 → LLM决策 → 生成响应（使用KB缓存）`

**关键问题：**
1. KB 查询是**自动触发**的（每次迭代开始）
2. 还是 LLM **主动调用**的（作为工具）？

**两种设计对比：**

| 设计方案 | LLM可见性 | 触发方式 | 优点 | 缺点 |
|---------|----------|---------|------|------|
| **方案A：自动预查询** | ❌ 不可见 | 每次迭代自动 | 减少 LLM 调用，提升性能 | LLM 无法控制是否查询 |
| **方案B：工具调用** | ✅ 可见 | LLM 主动调用 | LLM 可以决定何时查询 | 增加 LLM 调用次数 |

### 1.2 Flow 可见性

**场景描述：**
- Flow 是固定业务流程 API
- 通过**意图匹配**触发（trigger_patterns）
- 从 v9 文档看：`EXEC → FLOW → 流程API (silent) → 跳出迭代`

**关键问题：**
1. Flow 是通过**规则匹配**触发（trigger_patterns）
2. 还是 LLM **决策**触发？

**两种设计对比：**

| 设计方案 | LLM可见性 | 触发方式 | 优点 | 缺点 |
|---------|----------|---------|------|------|
| **方案A：规则匹配** | ❌ 不可见 | 正则/关键词匹配 | 快速响应，确定性高 | 匹配规则可能不够灵活 |
| **方案B：LLM决策** | ✅ 可见 | LLM 判断意图 | 更智能，更灵活 | 增加 LLM 调用，可能误判 |

## 二、v9 文档的设计意图

### 2.1 KB 并行查询优化

从 v9 文档第 22-26 行：
```
3. **KB 并行查询优化**
   - ✅ 迭代开始时预先查询知识库
   - ✅ 不阻塞 LLM 决策，并行执行
   - ✅ 缓存查询结果，需要时直接使用
   - ✅ 显著缩短响应时间
```

**设计意图：自动预查询（LLM 不可见）**

```
ITER[迭代循环] --> KB_PRE[KB预查询 并行执行]
KB_PRE --> DEC[LLM决策]
DEC --> JUDGE{决策结果}
JUDGE -->|should_respond| RESP[生成响应 使用KB缓存]
```

**关键点：**
1. KB 查询在 LLM 决策**之前**自动触发
2. 并行执行，不阻塞 LLM
3. 结果缓存，LLM 生成响应时直接使用

**结论：KB 应该是 LLM 不可见的自动预查询**

### 2.2 Flow 触发机制

从 v9 文档第 266-316 行（FlowDefinition）：
```python
class FlowDefinition(BaseModel):
    # 触发规则
    trigger_patterns: list[str] = Field(
        default_factory=list,
        description="正则表达式列表，用于快速匹配"
    )
```

**设计意图：规则匹配（LLM 不可见）**

但是从架构图看：
```
DEC[LLM决策] --> JUDGE{决策结果}
JUDGE -->|should_continue| EXEC[执行Action]
EXEC -->|FLOW| FE[流程API silent]
```

**矛盾点：**
- 配置中有 `trigger_patterns`（规则匹配）
- 架构图中 Flow 是 LLM 决策后执行的 Action

**可能的设计：**
1. **两层路由**：先规则匹配（快速），匹配不到再 LLM 决策
2. **LLM 决策为主**：trigger_patterns 只是辅助信息

## 三、推荐设计方案

### 3.1 Knowledge Base：自动预查询（LLM 不可见）

**设计理由：**
1. ✅ 符合 v9 文档的"并行查询优化"设计
2. ✅ 减少 LLM 调用次数，提升性能
3. ✅ KB 查询是高频操作，自动化更合理
4. ✅ LLM 在生成响应时可以直接使用缓存的 KB 结果

**实现方式：**

```python
class WorkflowOrchestrator:
    """
    KB 作为自动预查询，不注册为 Tool
    """

    def __init__(self, config, context_vars, http_client):
        self.kb_url = config.retrieve_knowledge_url
        self._kb_cache = {}

    async def _pre_query_knowledge(self, user_message: str) -> str | None:
        """
        预查询知识库（并行执行，不阻塞 LLM）

        Returns:
            KB 查询结果（缓存）
        """
        if not self.kb_url:
            return None

        try:
            # 提取关键词
            keywords = self._extract_keywords(user_message)

            # 查询 KB
            response = await self.http_client.request(
                method="GET",
                url=self.kb_url,
                json={
                    "chatbotId": self.context_vars.get("chatbotId"),
                    "tenantId": self.context_vars.get("tenantId"),
                    "keywords": keywords,
                },
            )

            if response.is_success:
                result = response.json()
                # 缓存结果
                self._kb_cache[user_message] = result
                return result

        except Exception as e:
            print(f"KB query error: {e}")
            return None

    async def query(self, user_message: str) -> str:
        """
        处理用户消息（迭代循环）
        """
        # 1. 并行执行 KB 查询和 LLM 决策
        kb_task = asyncio.create_task(self._pre_query_knowledge(user_message))
        decision_task = asyncio.create_task(self._llm_decision(user_message))

        kb_result, decision = await asyncio.gather(kb_task, decision_task)

        # 2. 根据决策执行
        if decision.should_respond:
            # 生成响应时，将 KB 结果注入到上下文
            return await self._generate_response(
                user_message,
                kb_context=kb_result
            )
        elif decision.should_continue:
            # 执行 Action
            return await self._execute_action(decision.next_action)
```

**配置示例：**

```json
{
  "retrieve_knowledge_url": "http://kb-api.example.com/retrieve",
  // KB URL 不在 system_tools 中，不注册为 Tool
  "system_tools": [
    // 其他工具...
  ]
}
```

### 3.2 Flow：两层路由（规则匹配 + LLM 决策）

**设计理由：**
1. ✅ 规则匹配快速响应（确定性场景）
2. ✅ LLM 决策兜底（复杂场景）
3. ✅ 灵活性和性能兼顾

**实现方式：**

```python
class WorkflowOrchestrator:
    """
    Flow 使用两层路由：
    1. 先规则匹配（快速）
    2. 匹配不到再 LLM 决策
    """

    async def query(self, user_message: str) -> str:
        """
        处理用户消息
        """
        # 1. 第一层：规则匹配（快速路由）
        flow_match = self._match_flow_by_pattern(user_message)
        if flow_match:
            # 直接执行 Flow，不经过 LLM
            return await self.flow_executor.execute(
                user_message,
                parameters={}
            )

        # 2. 第二层：LLM 决策
        # 并行执行 KB 查询和 LLM 决策
        kb_task = asyncio.create_task(self._pre_query_knowledge(user_message))
        decision_task = asyncio.create_task(self._llm_decision(user_message))

        kb_result, decision = await asyncio.gather(kb_task, decision_task)

        # 3. 根据 LLM 决策执行
        if decision.action_type == "flow":
            # LLM 决策执行 Flow
            return await self.flow_executor.execute(
                user_message,
                parameters=decision.parameters
            )
        elif decision.should_respond:
            return await self._generate_response(user_message, kb_context=kb_result)
        # ... 其他 action 类型

    def _match_flow_by_pattern(self, user_message: str) -> FlowConfig | None:
        """
        规则匹配 Flow

        Returns:
            匹配的 Flow 配置，或 None
        """
        for flow in self.config.flows:
            for pattern in flow.trigger_patterns:
                if re.search(pattern, user_message):
                    return flow
        return None
```

**配置示例：**

```json
{
  "flow_url": "http://flow-api.example.com",
  "flows": [
    {
      "flow_id": "leave_request",
      "name": "请假申请",
      "trigger_patterns": ["我要请假", "申请.*假", "请.*天假"],
      "endpoint": {...}
    }
  ]
}
```

**Flow 对 LLM 的可见性：**

```python
# 方案A：Flow 完全不可见（仅规则匹配）
# - 优点：快速响应
# - 缺点：无法处理复杂意图

# 方案B：Flow 部分可见（两层路由）
# - 规则匹配优先（快速）
# - LLM 决策兜底（灵活）
# - 推荐方案！

# 方案C：Flow 完全可见（注册为 Tool）
# - 优点：最灵活
# - 缺点：增加 LLM 调用，可能误判
```

## 四、最终推荐架构

### 4.1 可见性矩阵（更新）

| 组件 | LLM可见性 | 触发方式 | 实现方式 | 原因 |
|------|----------|---------|---------|------|
| **KB (retrieve_knowledge_url)** | ❌ 不可见 | 自动预查询 | 并行执行，缓存结果 | 性能优化，高频操作 |
| **Flow (flow_url)** | ⚠️ 部分可见 | 规则匹配 + LLM决策 | 两层路由 | 快速响应 + 灵活性 |
| **System Tools** | ✅ 可见 | LLM 决策 | HttpTool | LLM 需要决策何时调用 |
| **Skills** | ✅ 可见 | LLM 决策 | 条件映射 | LLM 需要匹配条件 |

### 4.2 处理流程

```
用户消息
    │
    ▼
┌─────────────────┐
│ 1. 规则匹配     │
│   (Flow)        │
└─────────────────┘
    │
    ├─ 匹配成功 ──> 执行 Flow ──> 返回结果
    │
    └─ 匹配失败
        │
        ▼
    ┌─────────────────┐
    │ 2. 并行执行     │
    │   - KB 预查询   │
    │   - LLM 决策    │
    └─────────────────┘
        │
        ▼
    ┌─────────────────┐
    │ 3. LLM 决策结果 │
    └─────────────────┘
        │
        ├─ should_respond ──> 生成响应（使用 KB 缓存）
        │
        ├─ action: flow ──> 执行 Flow
        │
        ├─ action: skill ──> 执行 Skill
        │
        └─ action: tool ──> 执行 Tool
```

### 4.3 配置结构（更新）

```json
{
  "basic_settings": {...},

  // KB URL - 自动预查询（LLM 不可见）
  "retrieve_knowledge_url": "http://kb-api.example.com/retrieve",

  // Flow URL - 两层路由（规则匹配 + LLM 决策）
  "flow_url": "http://flow-api.example.com",
  "flows": [
    {
      "flow_id": "leave_request",
      "trigger_patterns": ["我要请假", "申请.*假"],  // 规则匹配
      "endpoint": {...}
    }
  ],

  // Skills - LLM 可见（条件映射）
  "skills": [
    {
      "condition": "Customer wants to schedule a demo",
      "action": "Save customer information",
      "tools": ["save_customer_information"]
    }
  ],

  // System Tools - LLM 可见（动态加载）
  "system_tools": [
    {
      "name": "save_customer_information",
      "description": "Save customer info",
      "parameters": {...},
      "endpoint": {...}
    }
  ]
}
```

## 五、实现建议

### 5.1 KB 自动预查询实现

```python
class KnowledgeBaseExecutor:
    """
    知识库执行器（自动预查询，LLM 不可见）
    """

    def __init__(self, kb_url: str | None, http_client, context_vars):
        self.kb_url = kb_url
        self.http_client = http_client
        self.context_vars = context_vars

    async def pre_query(self, user_message: str) -> dict | None:
        """
        预查询知识库（并行执行）

        Returns:
            KB 查询结果
        """
        if not self.kb_url:
            return None

        try:
            # 提取关键词（简单实现）
            keywords = user_message[:100]  # 或使用更复杂的关键词提取

            response = await self.http_client.request(
                method="GET",
                url=self.kb_url,
                json={
                    "chatbotId": self.context_vars.get("chatbotId"),
                    "tenantId": self.context_vars.get("tenantId"),
                    "keywords": keywords,
                },
                timeout=5.0,  # 短超时，不阻塞主流程
            )

            if response.is_success:
                return response.json()

        except Exception as e:
            # 静默失败，不影响主流程
            print(f"KB pre-query failed: {e}")

        return None
```

### 5.2 Flow 两层路由实现

```python
class FlowRouter:
    """
    Flow 路由器（两层路由）
    """

    def __init__(self, flows: list[FlowConfig]):
        self.flows = flows

    def match_by_pattern(self, user_message: str) -> FlowConfig | None:
        """
        第一层：规则匹配

        Returns:
            匹配的 Flow 配置，或 None
        """
        for flow in self.flows:
            for pattern in flow.trigger_patterns:
                if re.search(pattern, user_message, re.IGNORECASE):
                    return flow
        return None

    def build_flow_descriptions(self) -> str:
        """
        第二层：为 LLM 构建 Flow 描述

        Returns:
            Flow 描述文本（用于 system prompt）
        """
        if not self.flows:
            return ""

        parts = ["## Available Flows", ""]
        for flow in self.flows:
            parts.append(f"- **{flow.name}**: {flow.description}")
        parts.append("")

        return "\n".join(parts)
```

### 5.3 WorkflowOrchestrator 集成

```python
class WorkflowOrchestrator:
    """
    工作流编排器（集成 KB 和 Flow）
    """

    def __init__(self, config, context_vars, http_client):
        self.config = config
        self.context_vars = context_vars
        self.http_client = http_client

        # KB 执行器（自动预查询）
        self.kb_executor = KnowledgeBaseExecutor(
            config.retrieve_knowledge_url,
            http_client,
            context_vars
        )

        # Flow 路由器（两层路由）
        self.flow_router = FlowRouter(config.flows)
        self.flow_executor = FlowExecutor(
            config.flow_url,
            http_client,
            context_vars
        )

        # System 执行器
        self.system_executor = SystemExecutor(http_client, context_vars)

        # Tool 加载器（LLM 可见）
        self._tool_loader = self._create_tool_loader()

    def _create_tool_loader(self) -> ConfigToolLoader:
        """
        创建 Tool 加载器

        注意：
        - KB URL 不注册为 Tool（自动预查询）
        - Flow 不注册为 Tool（两层路由）
        - 只注册 system_tools（LLM 可见）
        """
        tools = []

        # 只加载 system_tools
        tools.extend(self.config.system_tools)

        # 不添加 KB 和 Flow

        return ConfigToolLoader(AgentConfigSchema(
            basic_settings=self.config.basic_settings,
            tools=[ToolConfig.model_validate(tool) for tool in tools]
        ))

    def build_system_prompt(self) -> str:
        """
        构建系统提示

        包含：
        - Basic settings
        - Skills（LLM 可见）
        - Flow descriptions（LLM 部分可见）
        - 不包含 KB（自动预查询）
        """
        parts = []

        # Basic settings
        # ...

        # Skills
        # ...

        # Flow descriptions（可选，用于 LLM 决策兜底）
        flow_desc = self.flow_router.build_flow_descriptions()
        if flow_desc:
            parts.append(flow_desc)

        return "\n".join(parts)

    async def query(self, user_message: str) -> str:
        """
        处理用户消息

        流程：
        1. 规则匹配 Flow（快速路由）
        2. 并行执行 KB 预查询和 LLM 决策
        3. 根据决策执行相应 Action
        """
        # 1. 第一层：规则匹配 Flow
        flow_match = self.flow_router.match_by_pattern(user_message)
        if flow_match:
            return await self.flow_executor.execute(
                flow_match.flow_id,
                user_message,
                {}
            )

        # 2. 第二层：并行执行 KB 和 LLM 决策
        kb_task = asyncio.create_task(self.kb_executor.pre_query(user_message))
        decision_task = asyncio.create_task(self._llm_decision(user_message))

        kb_result, decision = await asyncio.gather(kb_task, decision_task)

        # 3. 根据决策执行
        if decision.should_respond:
            return await self._generate_response(
                user_message,
                kb_context=kb_result
            )
        elif decision.action_type == "flow":
            return await self.flow_executor.execute(
                decision.action_target,
                user_message,
                decision.parameters
            )
        # ... 其他 action 类型
```

## 六、总结

### 6.1 设计决策

| 组件 | 可见性 | 触发方式 | 理由 |
|------|--------|---------|------|
| **KB** | ❌ 不可见 | 自动预查询 | 性能优化，高频操作，并行执行 |
| **Flow** | ⚠️ 部分可见 | 规则匹配 + LLM决策 | 快速响应 + 灵活性兼顾 |
| **System Tools** | ✅ 可见 | LLM 决策 | LLM 需要决策何时调用 |
| **Skills** | ✅ 可见 | LLM 决策 | LLM 需要匹配条件 |

### 6.2 优势

1. ✅ **KB 自动预查询** - 提升性能，减少 LLM 调用
2. ✅ **Flow 两层路由** - 快速响应 + 灵活性
3. ✅ **System Tools 动态加载** - 配置驱动，无硬编码
4. ✅ **Skills 条件映射** - LLM 智能匹配

### 6.3 实现要点

1. KB 不注册为 Tool，在迭代开始时自动预查询
2. Flow 使用两层路由：规则匹配优先，LLM 决策兜底
3. System Tools 全部动态加载，LLM 可见
4. Skills 通过条件映射到工具名称

这个设计既符合 v9 文档的性能优化目标，又保持了足够的灵活性！
