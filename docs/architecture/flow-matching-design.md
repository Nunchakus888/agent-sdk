# Flow 匹配与执行设计文档

## 概述

WorkflowAgentV2 支持两种 Flow 匹配模式，统一通过 `flow_executor` tool 执行：

| 模式 | 匹配方式 | 执行方式 | 特点 |
|------|---------|---------|------|
| **keyword** | 代码匹配 (exact/contains/regex) | 直接调用 `flow_executor` | 快速、确定性、零 LLM 成本 |
| **intent** | LLM 语义理解 | LLM 调用 `flow_executor` | 灵活、上下文感知 |

## 架构流程

```
用户消息
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                 FlowMatcher.match_keyword()                 │
│                                                             │
│   遍历 keyword flows:                                       │
│   ├── match_type: exact    → user_msg == pattern            │
│   ├── match_type: contains → pattern in user_msg            │
│   └── match_type: regex    → re.search(pattern, user_msg)   │
└─────────────────────────────────────────────────────────────┘
    │
    ├── matched=True ──────────────────────────────────────────┐
    │       │                                                  │
    │       ▼                                                  │
    │   _execute_keyword_flow()                                │
    │       │                                                  │
    │       ▼                                                  │
    │   _agent._tool_map["flow_executor"].execute(flow_id=xxx)  │
    │       │                      ↑                           │
    │       │               同一个 Tool 实例                   │
    │       │                      ↓                           │
    │       └──────────────────────────────────────────────────┤
    │                                                          │
    └── matched=False                                          │
            │                                                  │
            ▼                                                  │
        Agent.query()                                          │
            │                                                  │
            ▼                                                  │
        LLM 决定调用 flow_executor(flow_id=xxx) ────────────────┘
                                   │
                                   ▼
                          HTTP POST 请求
                          /chatbot/ai-inner/trigger-flow
```

## 配置结构

### Flow 定义 (flows)

```json
{
  "flows": [
    {
      "flow_id": "6982bf54b7fb8c0ff4bf43c5",
      "description": "Greeting flow",
      "type": "keyword",
      "trigger_patterns": ["hello", "hi", "你好"],
      "match_type": "exact"
    },
    {
      "flow_id": "product_recommendation",
      "description": "Recommend products based on customer needs",
      "type": "intent"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `flow_id` | string | Flow 唯一标识 |
| `description` | string | Flow 描述（用于 LLM 理解） |
| `type` | enum | `keyword` 或 `intent` |
| `trigger_patterns` | string[] | 触发模式（keyword 类型必填） |
| `match_type` | enum | `exact` / `contains` / `regex` |

### flow_executor Tool 定义 (tools)

`flow_executor` 是统一的 Flow 执行入口，定义在配置的 `tools` 数组中：

```json
{
  "name": "flow_executor",
  "description": "Trigger a predefined business flow. Use this tool when user intent matches an available flow from the Intent Flows list. Pass the corresponding flow_id to execute the matched flow.",
  "parameters": {
    "type": "object",
    "properties": {
      "flow_id": {
        "description": "The unique identifier of the flow to trigger. Must be one of the flow_id values from the Intent Flows list in system prompt.",
        "type": "string"
      }
    },
    "required": ["flow_id"]
  },
  "endpoint": {
    "url": "http://yunpian-attila-security:8080/chatbot/ai-inner/trigger-flow",
    "method": "POST",
    "headers": {
      "Content-Type": "application/json"
    },
    "body": {
      "chatbotId": "6982b54ab0156d11ca09e76a",
      "conversationId": "#conversationId#",
      "tenantId": "6336b6724011d05a0edbbe1a",
      "flowId": "{flow_id}",
      "customerPhoneNumber": "#phoneNumber#",
      "businessPhoneNumber": "#businessPhoneNumber#"
    }
  }
}
```

**description 优化说明**：

| 版本 | description | 问题 |
|------|-------------|------|
| 旧版 | `"Trigger a predefined flow execution based on user intent"` | 过于笼统，LLM 不知道何时使用、如何选择 flow_id |
| **新版** | `"Trigger a predefined business flow. Use this tool when user intent matches an available flow from the Intent Flows list. Pass the corresponding flow_id to execute the matched flow."` | 明确指导 LLM：何时使用、flow_id 来源 |

**关键字段说明**：

| 字段 | 说明 |
|------|------|
| `name` | 固定为 `flow_executor`，代码中通过此名称查找 tool |
| `description` | 增强语义，指导 LLM 关联 Intent Flows 列表 |
| `parameters.flow_id` | 明确说明必须来自 Intent Flows 列表 |
| `required: ["flow_id"]` | 标记为必填参数 |
| `endpoint.body.flowId` | 使用 `{flow_id}` 占位符，运行时替换 |
| `#conversationId#` 等 | 上下文变量，由系统自动填充 |

**LLM 决策流程**：

```
1. LLM 读取 system prompt 中的 Intent Flows 列表
2. 用户消息: "我想看看有什么适合我的产品"
3. LLM 语义匹配: 最接近 "product_recommendation" flow
4. LLM 调用: flow_executor(flow_id="product_recommendation")
```

## Intent Flow 匹配机制

### 设计决策

Intent flow 的匹配通过 **Prompt 描述 + 单一 flow_executor tool** 实现：

| 方案 | 描述 | 选择原因 |
|------|------|---------|
| ~~多 Tool~~ | 每个 intent flow 创建独立 tool | tools 列表膨胀，与 flow_executor 重复 |
| **Prompt 描述** | system prompt 中列出 flows，LLM 调用 flow_executor | 简洁、利用现有架构 |
| ~~两阶段~~ | 先 LLM 判断意图，再调用 tool | 多一次 LLM 调用，成本高 |

### 实现方式

1. **SystemPromptBuilder** 在构建 prompt 时，只包含 `type=intent` 的 flows
2. 生成的 prompt 片段示例：

```
### Intent Flows
When user intent matches one of the following, call `flow_executor` tool with the corresponding flow_id:
- `product_recommendation`: 根据客户需求推荐产品
- `complaint_handling`: 处理客户投诉和问题
```

3. LLM 根据用户消息语义，选择最匹配的 flow_id，调用 `flow_executor(flow_id="xxx")`

### SystemPromptBuilder (builder.py)

支持多种 flow 配置格式：

```python
def _build_capabilities(self) -> str:
    # Flows - 只描述 intent 类型的 flows（keyword 类型由代码匹配，无需 LLM 参与）
    flows = getattr(self.config, 'flows', [])
    if flows:
        intent_flows = []
        for f in flows:
            # 判断 flow 类型，支持多种格式：
            # - FlowType 枚举: FlowType.KEYWORD / FlowType.INTENT
            # - 字符串: "keyword" / "intent"
            # - dict: {"type": "keyword"} / {"type": "intent"}
            # - None: 默认为 intent
            if isinstance(f, dict):
                flow_type = f.get('type', 'intent')
                fid = f.get('flow_id') or f.get('name')
                desc = f.get('description')
            else:
                flow_type = getattr(f, 'type', None)
                fid = getattr(f, 'flow_id', None) or getattr(f, 'name', None)
                desc = getattr(f, 'description', None)

            # 跳过 keyword 类型（由代码匹配，无需 LLM）
            flow_type_str = str(flow_type).lower() if flow_type else 'intent'
            if flow_type_str == 'keyword' or flow_type_str == 'flowtype.keyword':
                continue

            intent_flows.append(f"- `{fid}`: {desc}")
```

**适配的配置格式**：

| 格式 | 示例 | 说明 |
|------|------|------|
| FlowType 枚举 | `type: FlowType.INTENT` | Pydantic model |
| 字符串 | `"type": "intent"` | JSON 配置 |
| dict | `{"type": "keyword", ...}` | 原始字典 |
| None/缺省 | 无 type 字段 | 默认为 intent |

## 核心代码

### FlowMatcher (flow_matcher.py)

```python
@dataclass
class FlowMatcher:
    """Flow 匹配器 - 只负责 keyword 匹配，不负责执行"""

    flows: list[FlowDefinition]

    def match_keyword(self, user_message: str) -> FlowMatchResult:
        """尝试 keyword 匹配"""
        for flow in self._keyword_flows:
            matched_pattern = KeywordMatcher.match(user_message, flow)
            if matched_pattern:
                return FlowMatchResult(matched=True, flow=flow)
        return FlowMatchResult(matched=False)
```

### WorkflowAgentV2 (workflow_agent_v2.py)

```python
async def query(self, message: str, context=None) -> str:
    # Step 1: Keyword 匹配（快速、确定性）
    if self._flow_matcher.has_keyword_flows:
        match_result = self._flow_matcher.match_keyword(message)
        if match_result.matched:
            return await self._execute_keyword_flow(match_result)

    # Step 2: Agent loop（LLM 可能调用 trigger_flow）
    return await self._agent.query(message)

async def _execute_keyword_flow(self, match_result: FlowMatchResult) -> str:
    """通过 flow_executor tool 执行 keyword 匹配的 flow"""
    flow_id = match_result.flow.flow_id

    # 从 Agent 的 tool_map 获取 trigger_flow（与 LLM 调用的是同一个实例）
    trigger_tool = self._agent._tool_map.get("flow_executor")

    # 直接执行
    result = await trigger_tool.execute(flow_id=flow_id)
    return str(result)
```

## Schema 定义 (schemas.py)

```python
class FlowType(str, Enum):
    KEYWORD = "keyword"  # 代码匹配
    INTENT = "intent"    # LLM 语义匹配

class MatchType(str, Enum):
    EXACT = "exact"      # 精确匹配
    CONTAINS = "contains"  # 包含匹配
    REGEX = "regex"      # 正则匹配

class FlowDefinition(BaseModel):
    flow_id: str | None
    description: str | None
    type: FlowType = FlowType.INTENT
    trigger_patterns: list[str] = []
    match_type: MatchType = MatchType.CONTAINS
```

## 设计优势

### 1. 统一执行入口

无论 keyword 还是 intent，都通过同一个 `flow_executor` tool 执行：
- 代码简洁，无冗余
- 执行逻辑一致
- 便于维护和调试

### 2. 职责分离

| 组件 | 职责 |
|------|------|
| `FlowMatcher` | 只负责 keyword 匹配 |
| `flow_executor` tool | 只负责执行 |
| `WorkflowAgentV2` | 协调匹配和执行 |

### 3. 性能对比

| 模式 | 响应时间 | LLM 成本 | 准确性 |
|------|---------|---------|--------|
| keyword | < 1ms + HTTP | 零 | 100% 确定 |
| intent | LLM 延迟 + HTTP | 消耗 tokens | 依赖 LLM |

## 使用场景

### Keyword 模式适用于

- 固定触发词：`hello`, `hi`, `你好`
- 命令式交互：`查订单`, `转人工`
- 高频简单意图

### Intent 模式适用于

- 复杂表达：`我想看看有什么适合我的产品`
- 模糊意图：`能帮我推荐一下吗`
- 上下文相关的请求

## 配置示例

```json
{
  "flows": [
    {
      "flow_id": "greeting",
      "description": "问候客户",
      "type": "keyword",
      "trigger_patterns": ["hello", "hi", "hey", "你好", "早上好"],
      "match_type": "exact"
    },
    {
      "flow_id": "order_status",
      "description": "查询订单状态",
      "type": "keyword",
      "trigger_patterns": ["订单", "查订单", "order status"],
      "match_type": "contains"
    },
    {
      "flow_id": "leave_request",
      "description": "请假申请流程",
      "type": "keyword",
      "trigger_patterns": ["我要请假", "申请.*假", "请.*天假"],
      "match_type": "regex"
    },
    {
      "flow_id": "product_recommendation",
      "description": "根据客户需求推荐产品",
      "type": "intent"
    },
    {
      "flow_id": "complaint_handling",
      "description": "处理客户投诉和问题",
      "type": "intent"
    }
  ]
}
```

## 相关文件

| 文件 | 说明 |
|------|------|
| [builder.py](../../bu_agent_sdk/prompts/builder.py) | SystemPromptBuilder |
| [schemas.py](../../bu_agent_sdk/schemas.py) | FlowType, MatchType, FlowDefinition |
| [flow_matcher.py](../../bu_agent_sdk/agent/flow_matcher.py) | FlowMatcher, KeywordMatcher |
| [workflow_agent_v2.py](../../bu_agent_sdk/agent/workflow_agent_v2.py) | WorkflowAgentV2 |
| [sopv3.json](../configs/sopv3.json) | 配置示例 |

## 版本历史

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-02-04 | v1.0 | 初始设计，支持 keyword/intent 双模式 |
| 2026-02-04 | v1.1 | 补充 intent flow 匹配机制说明 |
| 2026-02-04 | v1.2 | 优化 flow_executor description 语义增强 |
