# CancellableTaskService

## 概述

`CancellableTaskService` 是一个通用的可取消任务服务，实现 **one-task-per-tag** 语义：同一 tag 下只允许一个任务运行，新任务自动取消旧任务。

**核心特性：**
- 自动取消：`restart()` 自动取消同 tag 旧任务
- Token 累加：被取消任务的 token 消耗自动累加
- 自动清理：`collect()` GC 机制，无需手动 `finish()`
- 生命周期管理：`async with` 上下文管理器，退出时自动清理

**文件位置：** `api/services/cancellable_tasks.py`

## API

### 构造函数

```python
CancellableTaskService(*, gc_interval: float = 5.0)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| gc_interval | float | 5.0 | GC 间隔（秒），避免频繁扫描 |

### restart(coro, *, tag) -> RestartResult

启动新任务，自动取消同 tag 旧任务。

```python
result = await service.restart(my_coro(), tag="session:123")
result.was_cancelled  # bool: 是否取消了旧任务
result.task           # asyncio.Task: 新任务引用
```

### set_tokens(tag, tokens)

记录当前任务的 token 数（用于取消时累加）。

### total_tokens(tag) -> int

获取总 token 数 = 当前任务 + 所有被取消任务的累计。

### cancel(*, tag, reason) / cancel_all(*, reason)

显式取消指定任务或全部任务。

### collect(*, force=False)

GC 扫描：清理已完成任务及其 token 数据。由 `restart()` 自动调用，通常无需手动调用。

### get_active_tasks() -> list[str]

返回当前活跃任务的 tag 列表。

## 生命周期

```python
async with CancellableTaskService() as tasks:
    await tasks.restart(coro(), tag="sess:1")
    ...
# __aexit__ 自动: cancel_all() + collect(force=True)
```

`__aexit__` 行为：
- 异常退出时：先 `cancel_all()`，再 `collect(force=True)`
- 正常退出时：直接 `collect(force=True)` 等待所有任务完成

## 自动清理机制

借鉴 `BackgroundTaskService` 的 `collect()` 模式，无需手动调用 `finish()`：

```
restart("sess:1") → 触发 collect() → 扫描已完成任务 → 清理 task + tokens
                   → 取消旧任务（累加 tokens）→ 启动新任务
```

**Token 数据生命周期：**

```
restart(tag) #1 → task A 启动, _tokens[tag] = 0
  task A: set_tokens(tag, 50)
restart(tag) #2 → task A 被取消, _cancelled_tokens[tag] += 50, task B 启动
  task B: set_tokens(tag, 30)
  task B: total_tokens(tag) → 30 + 50 = 80  ← 正确累加
  task B 完成
collect() → task B 已完成 → 清理 _tasks[tag], _tokens[tag], _cancelled_tokens[tag]
```

`total_tokens()` 在任务内部调用（任务完成前），所以数据始终可用。任务完成后由 GC 自动清理。

## 使用示例

### V1 Chat API 中的使用

```python
from api.services.cancellable_tasks import CancellableTaskService

_tasks = CancellableTaskService()

async def chat_async(request):
    async def process_chat():
        # ... 业务逻辑 ...
        _tasks.set_tokens(request.session_id, current_tokens)
        total = _tasks.total_tokens(request.session_id)
        # 无需 finally: _tasks.finish()

    result = await _tasks.restart(process_chat(), tag=request.session_id)
    if result.was_cancelled:
        logger.info(f"Cancelled old task for {request.session_id}")
```

### Token 累加场景

```
请求1: 消耗 100 tokens → 被请求2取消
请求2: 消耗 50 tokens  → 被请求3取消
请求3: 消耗 80 tokens  → 成功完成
       ↓
total_tokens = 100 + 50 + 80 = 230
```

## 与 BackgroundTaskService 对比

| 特性 | CancellableTaskService | BackgroundTaskService |
|------|----------------------|---------------------|
| 语义 | one-task-per-tag（互斥） | 多任务并存 |
| 启动 | `restart()` 自动取消旧任务 | `start()` 抛异常 / `restart()` |
| Token 追踪 | 内置 set/total_tokens | 无 |
| GC | `collect()` 清理 task + tokens | `collect()` 仅清理 task |
| 生命周期 | `__aenter__` / `__aexit__` | `__aenter__` / `__aexit__` |
| 并发安全 | asyncio.Lock | asyncio.Lock |

## 代码位置

| 模块 | 路径 |
|------|------|
| 服务实现 | `api/services/cancellable_tasks.py` |
| 使用方 | `api/routers/v1/chat.py` |
| 单元测试 | `tests/test_v1_chat_api.py` (TestChatTaskManager) |

## 取消机制深度解析

### asyncio 取消原理

`asyncio.Task.cancel()` 不会立即终止协程，而是设置一个取消标志。`CancelledError` 在协程的**下一个 `await` 点**被抛出。

```python
async def process_chat():
    # --- 同步代码，无 await，不可能被取消 ---
    is_new = not session_manager.exists(session_id)

    # --- await 点：理论上可被取消，但执行极快（毫秒级） ---
    ctx = await session_manager.get_or_create(...)     # ~2ms
    await callback_service.send_greeting(...)           # ~5ms

    # --- await 点：长时间等待（秒级），取消几乎总是命中这里 ---
    async for event in ctx.agent.query_stream(message): # ~3-30s
        collector.collect(event)
```

### 为什么会话创建能完成，而 LLM 请求被取消？

**纯粹是时间窗口问题**，不是选择性保护：

```
T=0ms     task A 启动
T=2ms     await get_or_create()    ← 毫秒级完成
T=5ms     await send_greeting()    ← 毫秒级完成
T=10ms    async for query_stream() ← 开始等待 LLM 流式响应（秒级）
          ...
T=500ms   新请求到达 → cancel() 设置取消标志
          ↓
T=501ms   query_stream() 内部下一个 await → CancelledError 抛出
```

会话创建和问候发送在取消信号到达前就已完成（毫秒 vs 秒）。LLM 流式请求耗时数秒，取消信号几乎总是在这个阶段命中。

> **注意：** 代码中 Phase 1 的"不可取消"注释是意图描述，并非通过 `asyncio.shield()` 强制保护。极端情况下（两个请求间隔 < 5ms），取消可能命中 `get_or_create()`。

### 取消后 LLM 请求的行为

当 `CancelledError` 在 `query_stream()` 内部抛出时：

```
query_stream() 正在接收 LLM 流式响应
    ↓ cancel()
CancelledError 在下一个 await 点抛出
    ↓
HTTP 客户端（httpx/aiohttp）关闭 TCP 连接
    ↓
LLM 提供商检测到客户端断开 → 停止推理
    ↓
except CancelledError → send_cancelled() 回调
    ↓
客户端收到 code=1 (CANCELLED)
```

**关键行为：**

| 方面 | 行为 |
|------|------|
| HTTP 连接 | 立即关闭（TCP RST/FIN） |
| LLM 推理 | 提供商检测到断开后停止生成 |
| 已生成 tokens | 仍然计费（提供商已消耗算力） |
| 部分响应 | 丢弃（EventCollector 中的数据不会发送） |
| 客户端通知 | 收到 `CANCELLED` 回调（code=1），不会收到部分内容 |
| 异常处理 | 受控的优雅取消，非异常崩溃 |

这是一个**优雅的受控取消**，每个请求都保证收到回调通知（成功/取消/错误/超时）。
