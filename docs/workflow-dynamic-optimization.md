# Workflow Executor 完全动态化优化总结（更新版）

## 一、核心优化：消除硬编码 + 正确的可见性设计

### 1.1 可见性设计矩阵（最终版）

| 组件 | LLM可见性 | 触发方式 | 实现方式 | 设计理由 |
|------|----------|---------|---------|---------|
| **KB (retrieve_knowledge_url)** | ❌ **不可见** | 自动预查询 | 并行执行，缓存结果 | 性能优化，高频操作，v9文档明确要求 |
| **Flow (flow_url)** | ⚠️ **部分可见** | 规则匹配 + LLM决策 | 两层路由 | 快速响应（规则）+ 灵活性（LLM） |
| **System Tools** | ✅ **可见** | LLM 决策 | HttpTool 动态加载 | LLM 需要决策何时调用 |
| **Skills** | ✅ **可见** | LLM 决策 | 条件映射到工具 | LLM 需要匹配条件 |

### 1.2 关键设计决策

#### 决策 1: KB 不应该对 LLM 可见

**原因：**
1. ✅ **v9 文档明确设计**：KB 预查询是性能优化手段
2. ✅ **并行执行**：在 LLM 决策的同时查询 KB，不阻塞
3. ✅ **自动化**：每次迭代自动触发，无需 LLM 决策
4. ✅ **缓存使用**：LLM 生成响应时直接使用缓存的 KB 结果

**v9 文档原文：**
```
3. **KB 并行查询优化**
   - ✅ 迭代开始时预先查询知识库
   - ✅ 不阻塞 LLM 决策，并行执行
   - ✅ 缓存查询结果，需要时直接使用
   - ✅ 显著缩短响应时间
```

**架构流程：**
```
ITER[迭代循环] --> KB_PRE[KB预查询 并行执行]
KB_PRE --> DEC[LLM决策]
DEC --> JUDGE{决策结果}
JUDGE -->|should_respond| RESP[生成响应 使用KB缓存]
```

#### 决策 2: Flow 使用两层路由

**原因：**
1. ✅ **快速响应**：规则匹配（trigger_patterns）快速处理确定性场景
2. ✅ **灵活兜底**：LLM 决策处理复杂意图
3. ✅ **性能优化**：规则匹配不需要 LLM 调用

**两层路由流程：**
```
用户消息
    │
    ▼
规则匹配 Flow?
    │
    ├─ Yes ──> 直接执行 Flow（无需 LLM）
    │
    └─ No ──> LLM 决策 ──> 可能执行 Flow
```

## 二、配置结构对应（更新）

### 2.1 sop.json 配置结构

```json
{
  "basic_settings": {...},

  // KB URL - 自动预查询（LLM 不可见）
  "retrieve_knowledge_url": "http://kb-api.example.com/retrieve",

  // Flow URL - 两层路由（规则匹配 + LLM 决策）
  "flow_url": "http://flow-api.example.com",

  // Skills - LLM 可见（条件映射到工具）
  "skills": [
    {
      "condition": "Customer wants to schedule a demo",
      "action": "Save customer information",
      "tools": ["save_customer_information"]  // 引用 system_tools 中的工具
    }
  ],

  // System Tools - LLM 可见（全部动态加载）
  "system_tools": [
    {
      "name": "save_customer_information",
      "description": "Save customer info",
      "parameters": {...},
      "endpoint": {...}
    },
    {
      "name": "handoff_to",
      "description": "Transfer to human agent",
      "parameters": {...},
      "endpoint": {...}
    }
    // 可以添加任意数量的工具，无需修改代码
  ]
}
```

### 2.2 配置到实现的映射

| 配置项 | 实现方式 | LLM可见性 | 说明 |
|--------|---------|----------|------|
| `retrieve_knowledge_url` | `KnowledgeBaseExecutor` | ❌ 不可见 | 自动预查询，并行执行 |
| `flow_url` | `FlowExecutor` + `FlowRouter` | ⚠️ 部分可见 | 两层路由：规则匹配 + LLM决策 |
| `skills[]` | `SkillMatcher` | ✅ 可见 | 条件映射到工具名称 |
| `system_tools[]` | `ConfigToolLoader` → `HttpTool` | ✅ 可见 | 全部动态加载 |

## 三、实现架构（更新）

### 3.1 WorkflowOrchestrator 结构

```python
class WorkflowOrchestrator:
    """
    工作流编排器

    职责分离：
    1. KB - 自动预查询（LLM 不可见）
    2. Flow - 两层路由（规则匹配 + LLM 决策）
    3. System Tools - 动态加载（LLM 可见）
    4. Skills - 条件映射（LLM 可见）
    """

    def __init__(self, config, context_vars, http_client):
        # 1. KB 执行器（自动预查询，LLM 不可见）
        self.kb_executor = KnowledgeBaseExecutor(
            config.retrieve_knowledge_url,
            http_client,
            context_vars
        )

        # 2. Flow 路由器（两层路由）
        self.flow_router = FlowRouter(config.flows)
        self.flow_executor = FlowExecutor(
            config.flow_url,
            http_client,
            context_vars
        )

        # 3. System 执行器
        self.system_executor = SystemExecutor(http_client, context_vars)

        # 4. Skill 匹配器
        self.skill_matcher = SkillMatcher(config.skills)

        # 5. Tool 加载器（只加载 system_tools，LLM 可见）
        self._tool_loader = self._create_tool_loader()

    def _create_tool_loader(self) -> ConfigToolLoader:
        """
        创建 Tool 加载器

        重要：
        - 只加载 system_tools（LLM 可见）
        - 不加载 KB（自动预查询）
        - 不加载 Flow（两层路由）
        """
        tools = []

        # 只加载 system_tools
        tools.extend(self.config.system_tools)

        # 不添加 KB 和 Flow

        return ConfigToolLoader(AgentConfigSchema(
            basic_settings=self.config.basic_settings,
            tools=[ToolConfig.model_validate(tool) for tool in tools]
        ))

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
                kb_context=kb_result  # 使用缓存的 KB 结果
            )
        elif decision.action_type == "flow":
            return await self.flow_executor.execute(
                decision.action_target,
                user_message,
                decision.parameters
            )
        # ... 其他 action 类型
```

### 3.2 KB 自动预查询实现

```python
class KnowledgeBaseExecutor:
    """
    知识库执行器（自动预查询，LLM 不可见）

    特点：
    - 在迭代开始时自动触发
    - 并行执行，不阻塞 LLM 决策
    - 结果缓存，供 LLM 生成响应时使用
    """

    def __init__(self, kb_url: str | None, http_client, context_vars):
        self.kb_url = kb_url
        self.http_client = http_client
        self.context_vars = context_vars

    async def pre_query(self, user_message: str) -> dict | None:
        """
        预查询知识库（并行执行）

        Args:
            user_message: 用户消息

        Returns:
            KB 查询结果（缓存）
        """
        if not self.kb_url:
            return None

        try:
            # 提取关键词（简单实现）
            keywords = user_message[:100]

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

### 3.3 Flow 两层路由实现

```python
class FlowRouter:
    """
    Flow 路由器（两层路由）

    第一层：规则匹配（快速）
    第二层：LLM 决策（灵活）
    """

    def __init__(self, flows: list[FlowConfig]):
        self.flows = flows

    def match_by_pattern(self, user_message: str) -> FlowConfig | None:
        """
        第一层：规则匹配

        Args:
            user_message: 用户消息

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

## 四、使用示例（更新）

### 4.1 加载配置并创建编排器

```python
from bu_agent_sdk.workflow import (
    WorkflowOrchestrator,
    load_workflow_config_from_file,
)

# 1. 加载配置
config = load_workflow_config_from_file("docs/configs/sop.json")

# 2. 创建编排器
orchestrator = WorkflowOrchestrator(
    config=config,
    context_vars={
        "dialogId": "123",
        "tenantId": "456",
        "chatbotId": config.basic_settings.get("chatbot_id"),
    }
)

# 3. 获取 LLM 可见的工具（只有 system_tools）
tools = orchestrator.get_tools()
print(f"LLM-visible tools: {len(tools)}")
# 输出：只包含 system_tools 中的工具
# 不包含 KB（自动预查询）
# 不包含 Flow（两层路由）

# 4. 构建系统提示
system_prompt = orchestrator.build_system_prompt()
# 包含：
# - Basic settings
# - Skills（条件映射）
# - Flow descriptions（可选，用于 LLM 决策兜底）
# 不包含：KB（自动预查询）

# 5. 创建 Agent
agent = Agent(
    llm=llm,
    tools=tools,
    system_prompt=system_prompt,
)
```

### 4.2 处理用户消息

```python
# 场景 1: 规则匹配 Flow（快速路由）
user_message = "我要请假3天"
response = await orchestrator.query(user_message)
# 流程：
# 1. 规则匹配 "我要请假" → 匹配成功
# 2. 直接执行 Flow，无需 LLM 决策
# 3. 返回结果

# 场景 2: LLM 决策 + KB 缓存
user_message = "YCloud 有哪些功能？"
response = await orchestrator.query(user_message)
# 流程：
# 1. 规则匹配 Flow → 匹配失败
# 2. 并行执行：
#    - KB 预查询（自动）
#    - LLM 决策
# 3. LLM 决策：should_respond
# 4. 生成响应，使用 KB 缓存

# 场景 3: LLM 决策调用工具
user_message = "我的邮箱是 john@example.com"
response = await agent.query(user_message)
# 流程：
# 1. 规则匹配 Flow → 匹配失败
# 2. 并行执行 KB 和 LLM 决策
# 3. LLM 决策：调用 save_customer_information 工具
# 4. 工具执行，返回结果
```

## 五、配置示例（完整）

### 5.1 sop.json 完整示例

```json
{
  "basic_settings": {
    "name": "YCloud Customer Service",
    "description": "Help customers with WhatsApp business services",
    "background": "YCloud is a leading WhatsApp business service provider",
    "language": "English",
    "tone": "Friendly and professional",
    "chatbot_id": "67adb3abaa26c063de0f4bd9"
  },

  // KB URL - 自动预查询（LLM 不可见）
  "retrieve_knowledge_url": "http://121.43.165.245:18080/chatbot/ai-inner/retrieve-knowledge",

  // Flow URL - 两层路由（规则匹配 + LLM 决策）
  "flow_url": "http://flow-examples.test.svc.cluster.local:8080",

  // Skills - LLM 可见（条件映射）
  "skills": [
    {
      "condition": "Customer wants to schedule a demo or requests detailed information",
      "action": "Persuade customer to provide contact info, then save using save_customer_information",
      "tools": ["save_customer_information"]
    },
    {
      "condition": "Customer explicitly requests to transfer to human agent",
      "action": "Transfer to human agent using handoff_to tool",
      "tools": ["handoff_to"]
    }
  ],

  // System Tools - LLM 可见（全部动态加载）
  "system_tools": [
    {
      "name": "save_customer_information",
      "description": "Save the customer's information",
      "parameters": {
        "type": "object",
        "properties": {
          "nickName": {"type": "string", "description": "Customer name"},
          "email": {"type": "string", "description": "Customer email"},
          "dynamic_field_1": {"type": "string", "description": "Country", "default": null}
        },
        "required": ["email", "nickName"]
      },
      "endpoint": {
        "url": "http://172.16.92.21:18080/contactsService/add",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": {
          "nickName": "{nickName}",
          "phoneNumber": "todo_autofill_by_system",
          "email": "{email}",
          "customAttrs": [
            {"key": "dynamic_field_1", "value": {"dynamic_field_1": "{dynamic_field_1}"}}
          ]
        }
      }
    },
    {
      "name": "handoff_to",
      "description": "Transfer conversation to human agent or team",
      "parameters": {
        "type": "object",
        "properties": {
          "assigneeId": {"type": "string", "description": "Assignee ID", "default": null},
          "type": {"type": "string", "description": "Assignee type", "default": "unassigned"}
        }
      },
      "endpoint": {
        "url": "http://172.16.80.52:8080/inboxConversationService/handoff",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": {
          "assigneeId": "{assigneeId}",
          "type": "{type}",
          "dialogId": "todo_autofill_by_system"
        }
      }
    },
    {
      "name": "close_conversation",
      "description": "Close the current conversation",
      "parameters": {"type": "object", "properties": {}},
      "endpoint": {
        "url": "http://www-dev.ycloud.com/inbox-api/inboxConversationService/close",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "body": {"dialogId": "todo_autofill_by_system"}
      }
    }
  ]
}
```

## 六、优势总结（更新）

### 6.1 正确的可见性设计

✅ **KB 自动预查询**
- 符合 v9 文档的性能优化设计
- 并行执行，不阻塞 LLM 决策
- 减少 LLM 调用次数
- 缓存结果，供响应生成使用

✅ **Flow 两层路由**
- 规则匹配快速响应（确定性场景）
- LLM 决策灵活兜底（复杂场景）
- 性能和灵活性兼顾

✅ **System Tools 动态加载**
- 完全配置驱动，无硬编码
- LLM 可见，智能决策何时调用
- 支持任意工具配置

✅ **Skills 条件映射**
- LLM 可见，智能匹配条件
- 通过工具名称引用，无需硬编码
- 灵活的意图处理

### 6.2 性能优化

1. ✅ **KB 并行查询** - 不阻塞 LLM 决策
2. ✅ **Flow 规则匹配** - 快速路由，无需 LLM
3. ✅ **HTTP 客户端复用** - 减少连接开销
4. ✅ **Tool 定义缓存** - 减少重复计算

### 6.3 灵活性

1. ✅ **完全配置驱动** - 修改配置即可添加功能
2. ✅ **支持任意工具** - 无需修改代码
3. ✅ **两层路由** - 规则 + LLM 双重保障
4. ✅ **类型安全** - Pydantic 验证

## 七、文件清单

### 核心实现
- ✅ `bu_agent_sdk/workflow/executors.py` - 完全动态化实现
- ✅ `bu_agent_sdk/workflow/__init__.py` - 模块导出

### 文档
- ✅ `docs/workflow-visibility-design-analysis.md` - 可见性设计分析
- ✅ `docs/workflow-dynamic-optimization.md` - 本文档（动态化优化总结）
- ✅ `docs/workflow-tool-strategy-analysis.md` - 工具策略分析

### 示例
- ✅ `examples/workflow_dynamic_config_example.py` - 动态配置示例

## 八、总结

### 核心改进

1. **正确的可见性设计**
   - KB：自动预查询（LLM 不可见）
   - Flow：两层路由（部分可见）
   - System Tools：动态加载（LLM 可见）
   - Skills：条件映射（LLM 可见）

2. **完全消除硬编码**
   - 所有工具从配置动态加载
   - 支持任意工具配置
   - 配置即代码

3. **性能优化**
   - KB 并行查询
   - Flow 规则匹配
   - HTTP 客户端复用

4. **符合 v9 文档设计**
   - KB 预查询优化
   - 两层路由机制
   - Silent action 支持

这个设计既符合 v9 文档的架构要求，又实现了完全的配置驱动和动态化！
