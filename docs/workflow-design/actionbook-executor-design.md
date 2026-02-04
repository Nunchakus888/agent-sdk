# ActionBook Executor 设计文档

## 概述

`actionbook_executor` 是一个基于条件匹配的工具，用于执行预定义的 Action Book。当用户意图匹配任意一条配置的 condition 时，LLM 调用此工具并传入用户消息。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Config (sopv3.json)                         │
├─────────────────────────────────────────────────────────────────────┤
│  action_books:                                                      │
│    - condition: "Wants to schedule a demo..."                       │
│    - condition: "Customer wants to transfer to human..."            │
│                                                                     │
│  tools:                                                             │
│    - name: actionbook_executor                                      │
│      parameters.properties: { message }  ← 只定义 LLM 参数          │
│      parameters.required: ["message"]                               │
│      endpoint.body: {message, chatbotId, tenantId, ...}             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SystemPromptBuilder._build_capabilities()              │
├─────────────────────────────────────────────────────────────────────┤
│  ### Action Books                                                   │
│  When user intent matches ANY of the following conditions,          │
│  call `actionbook_executor` tool with the user's message:           │
│  - **Condition 1**: Wants to schedule a demo...                     │
│  - **Condition 2**: Customer wants to transfer to human...          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Tool Definition                          │
├─────────────────────────────────────────────────────────────────────┤
│  name: actionbook_executor                                          │
│  parameters.properties: { message }  ← 只暴露 LLM 参数              │
│  parameters.required: ["message"]                                   │
│                                                                     │
│  ✗ chatbotId, tenantId 等系统参数不在 properties 中，LLM 看不到     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     HttpTool.execute() 执行时                        │
├─────────────────────────────────────────────────────────────────────┤
│  LLM 提供:     { message: "I want a demo" }                         │
│  系统填充:     { chatbotId, tenantId, sessionId, ... }              │
│                     ↓                                               │
│  合并后请求:   { message, chatbotId, tenantId, sessionId, ... }     │
│                                                                     │
│  未提供的可选参数 → 返回 None → 从 body 中移除                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 参数来源分离设计

### 核心原则

| 参数类型 | 定义位置 | 暴露给 LLM | 来源 | 示例 |
|---------|---------|-----------|------|------|
| LLM 必填 | `properties` + `required` | ✓ | LLM 提供 | `message` |
| LLM 可选 | `properties` (不在 required) | ✓ | LLM 可选提供 | `note` |
| 系统参数 | 只在 `endpoint.body` | ✗ | `context_vars` | `chatbotId`, `tenantId` |

### JSON Schema 规范

JSON Schema 使用 `required` 数组指定必填字段，**不在 `required` 中的字段自动为可选**：

```json
{
  "type": "object",
  "properties": {
    "message": {"type": "string"},   // 必填 (在 required 中)
    "note": {"type": "string"}       // 可选 (不在 required 中)
  },
  "required": ["message"]            // 只列出必填字段
}
```

### 参数处理流程

```
1. LLM 必填参数
   - 在 properties 中定义
   - 在 required 数组中列出
   - LLM API (strict=true) 强制校验

2. LLM 可选参数
   - 在 properties 中定义
   - 不在 required 数组中
   - 未提供时返回 None，从 body 中移除

3. 系统参数
   - 不在 properties 中定义 (LLM 看不到)
   - 只在 endpoint body 中使用 {param}
   - 从 context_vars 填充
```

## 与 flow_executor 对比

| 特性 | flow_executor | actionbook_executor |
|------|---------------|---------------------|
| 匹配方式 | keyword (代码) + intent (LLM) | intent only (LLM) |
| LLM 参数 | `flow_id` (选择具体 flow) | `message` (用户原始消息) |
| 执行决策 | LLM 选择具体 flow | 后端根据 message 决定 |
| Prompt 内容 | 列出 flow_id 和描述 | 列出所有 conditions |

## 配置示例

### action_books 配置

```json
{
  "action_books": [
    {
      "condition": "1. Wants to schedule a demo 2. Requests product info...",
      "action": "Save customer information",
      "tools": ["save_customer_information"]
    },
    {
      "condition": "Customer wants to transfer to human agent",
      "action": "Transfer to human",
      "tools": ["handoff_to"]
    }
  ]
}
```

### actionbook_executor 工具配置

```json
{
  "tools": [
    {
      "name": "actionbook_executor",
      "description": "Execute an action book based on user intent",
      "parameters": {
        "type": "object",
        "properties": {
          "message": {
            "description": "The user message to process",
            "type": "string"
          }
        },
        "required": ["message"]
      },
      "endpoint": {
        "url": "http://service/actionbookService/execute",
        "method": "POST",
        "body": {
          "message": "{message}",
          "chatbotId": "{chatbotId}",
          "tenantId": "{tenantId}",
          "sessionId": "{sessionId}"
        }
      }
    }
  ]
}
```

**注意**: `chatbotId`, `tenantId`, `sessionId` 不在 `properties` 中定义，因此 LLM 看不到这些参数。

## 使用示例

```python
from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2
from bu_agent_sdk.schemas import WorkflowConfigSchema

# 加载配置
config = WorkflowConfigSchema(**config_dict)

# 创建 Agent，传入系统参数
agent = WorkflowAgentV2(
    config=config,
    llm=llm,
    context_vars={
        "chatbotId": "abc123",
        "tenantId": "xyz789",
        "sessionId": "session_001",
        "phoneNumber": "+1234567890",
    }
)

# 查询
response = await agent.query("I want to schedule a demo")
```

## 实现文件

| 文件 | 职责 |
|------|------|
| `bu_agent_sdk/prompts/builder.py` | `_build_capabilities()` 生成 Action Books prompt |
| `bu_agent_sdk/agent/workflow_agent_v2.py` | `context_vars` 参数传递给 HttpTool |
| `bu_agent_sdk/tools/config_loader.py` | `HttpTool` 执行时合并参数，未填充参数返回 None |

## 执行流程

```
1. 用户发送消息: "I want to schedule a demo"
                    │
2. LLM 看到 prompt: │
   "### Action Books │
    When user intent matches ANY conditions,
    call actionbook_executor with message..."
                    │
3. LLM 判断匹配 Condition 1
                    │
4. LLM 调用: actionbook_executor(message="I want to schedule a demo")
                    │
5. HttpTool.execute():
   - LLM 参数: {message: "I want to schedule a demo"}
   - context_vars: {chatbotId: "abc", tenantId: "xyz", ...}
   - 合并后发送 HTTP 请求
   - 未提供的可选参数从 body 中移除
                    │
6. 后端服务根据 message 内容决定执行哪个 action_book
```

## 占位符模式配置

### 支持的模式

| 模式 | 占位符 | 转义 | 用途 |
|------|--------|------|------|
| `SINGLE_BRACE` (默认) | `{param}` | `{{` → `{`, `}}` → `}` | 标准模式 |
| `DOUBLE_BRACE` | `{{param}}` | `{{{{` → `{{`, `}}}}` → `}}` | Jinja2 兼容 |

### 使用示例

```python
from bu_agent_sdk.tools.config_loader import HttpTool, PlaceholderStyle

# 单括号模式 (默认)
http_tool = HttpTool(
    config=tool_config,
    context_vars={"tenantId": "123"},
    placeholder_style=PlaceholderStyle.SINGLE_BRACE
)

# 双括号模式 (Jinja2 兼容)
http_tool = HttpTool(
    config=tool_config,
    context_vars={"tenantId": "123"},
    placeholder_style=PlaceholderStyle.DOUBLE_BRACE
)
```

### 转义示例

```python
# 单括号模式
"Use {{literal}} braces"  →  "Use {literal} braces"
"JSON: {{\"key\": \"{value}\"}}"  →  "JSON: {\"key\": \"actual_value\"}"

# 双括号模式
"Use {{{{literal}}}} braces"  →  "Use {{literal}} braces"
```

## 关键设计点

1. **参数来源分离**: LLM 参数在 `properties` 中定义，系统参数只在 `endpoint.body` 中使用
2. **LLM 只关注 `message` 参数**: 其他参数由系统自动填充
3. **条件匹配由 LLM 完成**: 基于 prompt 中的 conditions 列表
4. **具体执行由后端决定**: 后端根据 message 内容选择 action_book
5. **可选参数自动移除**: 未提供的可选参数返回 None，从 body 中过滤
