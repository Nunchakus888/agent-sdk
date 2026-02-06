# Chat Async API 实现文档

## 概述

`/api/v1/chat` 是一个异步聊天接口，立即返回 `correlation_id` 用于追踪，处理完成后通过 HTTP POST 回调通知结果。

## 核心特性

1. **异步处理**: 立即返回 202，后台处理
2. **问候消息**: 新会话自动发送问候（如果配置了 `need_greeting`）
3. **智能取消**: 同 session 新请求仅在旧请求 message 写入 DB 后才取消
4. **回调保证**: 每个请求保证收到回调
5. **Tokens 累加**: 被取消任务的 tokens 消耗会累加到后续请求
6. **详细日志**: Callback 请求/响应日志便于调试
7. **Context 传递**: `correlation_id` 自动传递给 flow_executor 等工具

## 消息流程

### 首次会话（有问候语配置）

```
客户端 → POST /api/v1/chat → 202 Accepted
                              ↓
                         后台处理
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              回调1: greeting      回调2: message
              (问候消息)           (AI响应)
```

客户端会收到 **2 条回调消息**：
1. `kind: "greeting"` - 问候消息
2. `kind: "message"` - AI 对用户消息的响应

### 后续会话

```
客户端 → POST /api/v1/chat → 202 Accepted
                              ↓
                         后台处理
                              ↓
                       回调: message
                       (AI响应)
```

客户端只收到 **1 条回调消息**（无问候）。

## API 端点

### POST /api/v1/chat

#### 请求参数

```json
{
  "message": "用户消息",
  "session_id": "会话ID",
  "chatbot_id": "Chatbot ID",
  "tenant_id": "租户ID",
  "customer_id": "客户ID (可选)",
  "md5_checksum": "配置MD5 (可选)",
  "timeout": 300
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| message | string | 是 | 用户消息，最小长度 1 |
| session_id | string | 是 | 会话ID |
| chatbot_id | string | 是 | Chatbot ID |
| tenant_id | string | 是 | 租户ID |
| customer_id | string | 否 | 客户ID |
| md5_checksum | string | 否 | 配置文件MD5校验和 |
| timeout | int | 否 | 超时时间(秒)，默认300，范围1-600 |

#### 立即响应 (202 Accepted)

```json
{
  "status": 202,
  "code": 0,
  "message": "PROCESSING",
  "correlation_id": "R1234567890::process",
  "session_id": "sess_123"
}
```

## 回调机制

### 回调地址

由环境变量 `CHAT_CALLBACK_HOST` + `/api/callback/agent/receive` 组成。

### 回调响应结构

```json
{
  "status": 200,
  "code": 0,
  "message": "SUCCESS",
  "duration": 2.345,
  "correlation_id": "corr_123xyz::process",
  "data": {
    "id": "evt_abc123",
    "source": "ai_agent",
    "kind": "message",
    "creation_utc": "2025-01-15T10:30:00Z",
    "correlation_id": "corr_123xyz::process",
    "total_tokens": 1234,
    "session_id": "sess_xyz789",
    "message": "AI响应消息"
  }
}
```

### 事件类型 (kind)

| kind | 说明 |
|------|------|
| `greeting` | 问候消息（首次会话） |
| `message` | AI 响应消息 |

## 业务状态码 (code)

| code | 含义 | 说明 |
|------|------|------|
| `0` | SUCCESS | 成功 |
| `1` | CANCELLED | 被同一 session 的新请求取消 |
| `-1` | PROCESSING_ERROR | 处理错误 |
| `-2` | TIMEOUT | 超时 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHAT_CALLBACK_HOST` | 回调服务地址 | 无（不配置则跳过回调） |

## 代码位置

| 模块 | 路径 |
|------|------|
| Chat 路由 | `api/routers/v1/chat.py` |
| 任务服务 | `api/services/cancellable_tasks.py` |
| Callback 服务 | `api/services/callback.py` |
| 测试 | `tests/test_v1_chat_api.py` |

## 使用示例

### Python 客户端

```python
import httpx

async def chat_async(message: str, session_id: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/api/v1/chat",
            json={
                "message": message,
                "session_id": session_id,
                "chatbot_id": "bot_123",
                "tenant_id": "tenant_456",
                "timeout": 60,
            },
        )
        return resp.json()

# 立即返回 correlation_id
result = await chat_async("Hello", "sess_123")
print(f"Tracking ID: {result['correlation_id']}")
```

### 回调接收端

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/callback/agent/receive")
async def receive_callback(payload: dict):
    code = payload.get("code")
    kind = payload.get("data", {}).get("kind") if payload.get("data") else None

    if code == 0:  # SUCCESS
        if kind == "greeting":
            print(f"问候消息: {payload['data']['message']}")
        else:
            print(f"AI响应: {payload['data']['message']}")
    elif code == 1:  # CANCELLED
        print(f"请求已取消: {payload['correlation_id']}")
    elif code == -1:  # PROCESSING_ERROR
        print(f"处理错误: {payload['message']}")
    elif code == -2:  # TIMEOUT
        print(f"请求超时: {payload['correlation_id']}")

    return {"received": True}
```

## Callback 服务 API

`CallbackService` 提供以下方法：

```python
from api.services.callback import CallbackService, get_callback_service

# 获取默认服务实例
service = get_callback_service()

# 发送成功回调
await service.send_success(
    correlation_id="corr_123",
    session_id="sess_123",
    message="Hello!",
    duration=1.0,
    total_tokens=100,
)

# 发送问候回调
await service.send_greeting(
    correlation_id="corr_123",
    session_id="sess_123",
    greeting_message="您好！有什么可以帮您的？",
    duration=0.1,
)

# 发送取消回调
await service.send_cancelled("corr_123", duration=0.5)

# 发送超时回调
await service.send_timeout("corr_123", duration=300.0)

# 发送错误回调
await service.send_error("corr_123", duration=1.0, error_message="Connection failed")
```

## 任务取消机制

### 简洁设计

任务取消由 `CancellableTaskService` 统一管理（详见 [cancellable-task-service.md](cancellable-task-service.md)）。新请求到来时，`restart()` 自动取消同 session 的旧任务，无需手动清理。

```python
# 任务流程
新请求 → restart(tag=session_id) 自动取消旧任务
       → Session 初始化
       → AI 交互（可取消阶段）
       → 发送回调
       → collect() 自动清理已完成任务
```

### Tokens 累加统计

被取消任务的 tokens 消耗会累加到后续请求的统计中：

```
请求1: 消耗 100 tokens → 被取消
请求2: 消耗 50 tokens → 被取消
请求3: 消耗 80 tokens → 成功
       ↓
回调返回 total_tokens = 100 + 50 + 80 = 230
```

## Greeting Tokens 统计

Greeting 消息的 tokens 来自配置解析（LLM 增强）：

- **首次解析**：调用 LLM 解析配置，生成 greeting 消息，消耗 tokens
- **缓存命中**：从 DB 读取已解析的配置，tokens = 0

```python
# 首次会话（配置未缓存）
greeting 回调: total_tokens = 配置解析消耗的 tokens

# 后续会话（配置已缓存）
greeting 回调: total_tokens = 0
```

## Callback 日志

Callback 服务会记录详细的请求/响应日志：

### 请求日志 (INFO)
```
Callback REQ: [corr_123xyz] code=0, msg=SUCCESS, dur=1.23s, kind=message, tokens=150
```

### 响应日志 (INFO/WARNING)
```
# 成功
Callback RESP: [corr_123xyz] status=200 OK

# 失败
Callback RESP: [corr_123xyz] status=500, body=Internal Server Error
```

### 错误日志 (ERROR)
```
Callback ERROR: [corr_123xyz] TimeoutError: Connection timeout
```

## Context 变量传递

`correlation_id` 会自动注入到 `context_vars`，供 HTTP 工具使用：

```python
# 自动注入的 context_vars
{
    "correlationId": "corr_123xyz::process",
    "correlation_id": "corr_123xyz::process",
    "tenantId": "tenant_456",
    "chatbotId": "bot_123",
    "sessionId": "sess_789",
    # ... 其他变量
}
```

在 flow_executor 等工具配置中可以使用：

```json
{
  "name": "flow_executor",
  "endpoint": {
    "url": "{flow_url}",
    "body": {
      "flowId": "{flowId}",
      "correlationId": "{{correlationId}}"
    }
  }
}
```
