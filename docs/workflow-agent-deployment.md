# Workflow Agent 部署与架构最佳实践

> Web API 集成与微服务化架构指南

---


## FastAPI Web API 集成

### 2.1 基础实现

```python
# api/main.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema
from bu_agent_sdk.llm import ChatOpenAI
import json

app = FastAPI(title="Workflow Agent API")

# ============================================================================
# 1. 数据模型
# ============================================================================

class QueryRequest(BaseModel):
    """查询请求"""
    message: str
    session_id: str
    user_id: str | None = None

class QueryResponse(BaseModel):
    """查询响应"""
    session_id: str
    message: str
    status: str

class StreamChunk(BaseModel):
    """流式响应块"""
    type: str  # "tool_call" | "tool_result" | "text" | "complete"
    data: dict

# ============================================================================
# 2. 全局状态（生产环境应使用依赖注入）
# ============================================================================

# 配置加载
with open("config/workflow_config.json", encoding="utf-8") as f:
    config_data = json.load(f)
config = WorkflowConfigSchema(**config_data)

# LLM 初始化
llm = ChatOpenAI(model="gpt-4o")

# WorkflowAgent 单例
workflow_agent = WorkflowAgent(config=config, llm=llm)

# ============================================================================
# 3. API 端点
# ============================================================================

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    同步查询接口

    使用场景：
    - 简单的请求-响应模式
    - 不需要实时反馈
    """
    try:
        result = await workflow_agent.query(
            message=request.message,
            session_id=request.session_id,
        )

        return QueryResponse(
            session_id=request.session_id,
            message=result,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式查询接口（未来扩展）

    使用场景：
    - 需要实时反馈工具调用
    - 长时间运行的任务
    - 改善用户体验
    """
    from fastapi.responses import StreamingResponse
    import asyncio

    async def event_generator():
        # TODO: 实现流式输出
        # 需要 WorkflowAgent 支持 query_stream()

        yield f"data: {json.dumps({'type': 'start', 'data': {}})}\n\n"

        # 模拟流式输出
        result = await workflow_agent.query(
            message=request.message,
            session_id=request.session_id,
        )

        yield f"data: {json.dumps({'type': 'complete', 'data': {'message': result}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/api/v1/session/{session_id}")
async def get_session(session_id: str):
    """
    获取会话状态

    返回：
    - 会话历史
    - 当前状态
    - 元数据
    """
    session = workflow_agent._sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "workflow_state": {
            "config_hash": session.workflow_state.config_hash,
            "need_greeting": session.workflow_state.need_greeting,
            "status": session.workflow_state.status,
        },
        "message_count": len(session.messages),
    }


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """清除会话"""
    if session_id in workflow_agent._sessions:
        del workflow_agent._sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "config_hash": workflow_agent.config_hash,
        "sessions_count": len(workflow_agent._sessions),
    }

# ============================================================================
# 4. 启动
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.2 生产级改进

#### A. 依赖注入模式

```python
# api/dependencies.py

from typing import Annotated
from fastapi import Depends
from redis import Redis
from sqlalchemy.orm import Session as DBSession

def get_db() -> DBSession:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis() -> Redis:
    """获取 Redis 连接"""
    return redis_client

def get_workflow_agent() -> WorkflowAgent:
    """获取 WorkflowAgent 实例"""
    return workflow_agent_singleton

# 使用
@app.post("/api/v1/query")
async def query(
    request: QueryRequest,
    agent: Annotated[WorkflowAgent, Depends(get_workflow_agent)],
    db: Annotated[DBSession, Depends(get_db)],
):
    # 持久化会话到数据库
    result = await agent.query(...)

    # 保存到数据库
    db.add(SessionRecord(...))
    db.commit()

    return result
```

#### B. 会话持久化

```python
# api/session_store.py

from typing import Protocol
from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SessionRecord(Base):
    """会话记录（数据库模型）"""
    __tablename__ = "workflow_sessions"

    session_id = Column(String, primary_key=True)
    agent_id = Column(String)
    workflow_state = Column(JSON)
    messages = Column(JSON)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class SessionStore(Protocol):
    """会话存储接口"""
    async def get(self, session_id: str) -> Session | None: ...
    async def save(self, session: Session) -> None: ...
    async def delete(self, session_id: str) -> None: ...

class DatabaseSessionStore:
    """数据库会话存储"""
    def __init__(self, db: DBSession):
        self.db = db

    async def get(self, session_id: str) -> Session | None:
        record = self.db.query(SessionRecord).filter_by(session_id=session_id).first()
        if not record:
            return None

        return Session(
            session_id=record.session_id,
            agent_id=record.agent_id,
            workflow_state=WorkflowState(**record.workflow_state),
            messages=record.messages,
        )

    async def save(self, session: Session) -> None:
        record = SessionRecord(
            session_id=session.session_id,
            agent_id=session.agent_id,
            workflow_state=session.workflow_state.__dict__,
            messages=[m.dict() for m in session.messages],
        )
        self.db.merge(record)
        self.db.commit()
```

#### C. 异步任务队列（长时间运行）

```python
# api/tasks.py

from celery import Celery

celery_app = Celery("workflow_agent", broker="redis://localhost:6379/0")

@celery_app.task
async def execute_workflow_async(session_id: str, message: str):
    """
    异步执行工作流

    使用场景：
    - 长时间运行的任务
    - 批量处理
    - 定时任务
    """
    result = await workflow_agent.query(message, session_id)

    # 发送通知（WebSocket / Webhook）
    await notify_user(session_id, result)

    return result

# API 端点
@app.post("/api/v1/query/async")
async def query_async(request: QueryRequest):
    """异步查询接口"""
    task = execute_workflow_async.delay(
        request.session_id,
        request.message
    )

    return {
        "task_id": task.id,
        "status": "pending",
        "session_id": request.session_id,
    }

@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None,
    }
```

---

## 三、微服务化架构分析

### 3.1 是否需要微服务化？

#### 决策矩阵

| 场景 | 单体应用 | 微服务 |
|------|---------|--------|
| **团队规模** | < 10 人 | > 10 人 |
| **请求量** | < 1000 QPS | > 1000 QPS |
| **业务复杂度** | 简单-中等 | 高 |
| **部署频率** | 低频（周/月） | 高频（日/小时） |
| **资源需求** | LLM 调用较少 | LLM 调用频繁 |
| **团队经验** | 微服务经验少 | 微服务经验丰富 |

**建议：**

- ✅ **初期（MVP）：单体应用** - 快速迭代，简单部署
- ⚠️ **成长期：考虑微服务** - 当遇到性能瓶颈或团队扩张
- ✅ **成熟期：微服务** - 大规模、高并发、多团队协作

---

### 3.2 微服务拆分方案

#### 方案 A: 按功能模块拆分（推荐）

```
┌─────────────────────────────────────────────────────┐
│                   API Gateway                        │
│            (Kong / Nginx / Traefik)                  │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌─────▼──────┐
│   Workflow   │  │   Admin    │
│   Service    │  │   Service  │
│              │  │            │
│ - 意图匹配   │  │ - 配置管理 │
│ - 路由决策   │  │ - 监控统计 │
└──────┬───────┘  └────────────┘
       │
   ┌───┴────────────────┐
   │                    │
┌──▼────────┐  ┌───────▼──────┐  ┌────────────┐
│  Skills    │  │    Flows     │  │   Tools    │
│  Service   │  │   Service    │  │  Service   │
│            │  │              │  │            │
│ - Agent 执行│  │ - API 调用   │  │ - HTTP调用 │
│ - 子任务   │  │ - 流程管理   │  │ - 工具执行 │
└──────┬─────┘  └──────┬───────┘  └─────┬──────┘
       │                │                │
       └────────┬───────┴────────────────┘
                │
       ┌────────▼─────────┐
       │  Shared Services │
       │                  │
       │ - LLM Gateway    │ ← 统一 LLM 调用入口
       │ - Session Store  │ ← Redis / PostgreSQL
       │ - Cache Service  │ ← PlanCache
       └──────────────────┘
```

**服务职责：**

1. **Workflow Service（核心服务）**
   - 意图匹配（IntentMatcher）
   - 路由决策
   - 会话管理

2. **Skills Service**
   - Agent 模式执行
   - Function 模式执行
   - 子任务管理

3. **Flows Service**
   - Flow API 调用
   - 响应模板处理

4. **Tools Service**
   - HTTP 工具执行
   - 工具注册管理

5. **LLM Gateway（共享服务）**
   - 统一 LLM 调用
   - 负载均衡
   - 限流控制
   - 成本追踪

---

#### 方案 B: 按负载特征拆分（高性能）

```
┌──────────────────────────────────────────┐
│         Load Balancer (Nginx)            │
└──────────────┬───────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼────────┐  ┌──▼─────────────┐
│  Lightweight  │  │    Heavy       │
│   Service     │  │   Service      │
│               │  │                │
│ - Flows      │  │ - Skills       │
│ - Tools      │  │   (Agent模式)  │
│ - 规则匹配   │  │                │
│               │  │ - LLM 调用     │
│ (无状态)      │  │ (有状态)       │
└───────────────┘  └────────────────┘
     │                    │
     │                    │
     └────────┬───────────┘
              │
     ┌────────▼─────────┐
     │  Storage Layer   │
     │                  │
     │ - Redis (Session)│
     │ - PostgreSQL     │
     └──────────────────┘
```

**优势：**
- Lightweight Service 快速响应（规则匹配、API 调用）
- Heavy Service 独立扩展（LLM 调用、复杂计算）
- 资源利用率高

---

### 3.3 单体 vs 微服务对比

#### 单体应用（推荐起步）

```python
# 单一进程，所有模块在一起

app = FastAPI()

# 全局单例
workflow_agent = WorkflowAgent(config, llm)

@app.post("/api/v1/query")
async def query(request: QueryRequest):
    return await workflow_agent.query(...)

# 优点：
# ✅ 简单：一次部署，一个进程
# ✅ 快速：无网络调用开销
# ✅ 易调试：日志集中
# ✅ 低成本：单机器运行

# 缺点：
# ❌ 扩展受限：只能垂直扩展
# ❌ 故障影响大：一处出错全挂
# ❌ 资源争抢：LLM 调用阻塞其他请求
```

**适用场景：**
- MVP 阶段
- < 1000 QPS
- 小团队（< 5 人）

---

#### 微服务（成熟阶段）

```python
# 多个独立服务

# Service 1: Workflow Service
@app.post("/api/v1/query")
async def query(request: QueryRequest):
    # 调用 Skills Service
    result = await http_client.post(
        "http://skills-service/execute",
        json={"skill_id": "xxx", "input": "..."}
    )
    return result

# Service 2: Skills Service
@app.post("/execute")
async def execute_skill(request: SkillRequest):
    # 执行 Agent
    agent = Agent(llm, tools)
    return await agent.query(request.input)

# 优点：
# ✅ 独立扩展：Skills Service 可多实例
# ✅ 故障隔离：Skills 挂了不影响 Flows
# ✅ 技术灵活：不同服务用不同语言/框架
# ✅ 团队独立：不同团队维护不同服务

# 缺点：
# ❌ 复杂度高：服务发现、配置管理、监控
# ❌ 网络开销：服务间 HTTP 调用延迟
# ❌ 调试困难：分布式 tracing 必需
# ❌ 运维成本：多个服务部署、监控
```

**适用场景：**
- 成熟产品
- > 1000 QPS
- 大团队（> 10 人）

---

### 3.4 推荐架构演进路径

#### Phase 1: 单体应用（0-6 个月）

```
┌─────────────────────────────┐
│      FastAPI App            │
│                             │
│  ┌───────────────────────┐  │
│  │   WorkflowAgent       │  │
│  │                       │  │
│  │  - IntentMatcher      │  │
│  │  - SkillExecutor      │  │
│  │  - FlowExecutor       │  │
│  │  - ToolExecutor       │  │
│  └───────────────────────┘  │
│                             │
│  ┌───────────────────────┐  │
│  │   LLM (OpenAI/Claude) │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
         │
         │
    ┌────▼──────┐
    │  Redis    │ ← Session 缓存
    └───────────┘
```

**特点：**
- 快速开发
- 简单部署
- 低成本

---

#### Phase 2: 单体 + 缓存/队列（6-12 个月）

```
┌──────────────────────┐
│   Load Balancer      │
└──────┬───────────────┘
       │
   ┌───┴────┐
   │        │
┌──▼──┐  ┌─▼───┐
│ App │  │ App │ ← 水平扩展（相同代码）
│  1  │  │  2  │
└──┬──┘  └──┬──┘
   │        │
   └───┬────┘
       │
   ┌───▼─────────┐
   │   Redis     │ ← Session 共享
   ├─────────────┤
   │  PostgreSQL │ ← 持久化
   ├─────────────┤
   │   Celery    │ ← 异步任务
   └─────────────┘
```

**改进：**
- 水平扩展
- 会话共享
- 异步处理

---

#### Phase 3: 微服务化（12+ 个月）

```
┌──────────────────────────────┐
│        API Gateway           │
│     (Kong / Traefik)         │
└──────────┬───────────────────┘
           │
    ┌──────┴────────┐
    │               │
┌───▼────────┐  ┌──▼──────────┐
│ Workflow   │  │  Skills     │
│ Service    │  │  Service    │
│ (Stateless)│  │  (Stateful) │
└───┬────────┘  └──┬──────────┘
    │              │
    │              │
┌───▼──────────────▼───┐
│   Shared Services    │
│                      │
│ - LLM Gateway        │
│ - Session Store      │
│ - Config Center      │
└──────────────────────┘
```

**优势：**
- 独立扩展
- 故障隔离
- 团队独立

---

## 四、最佳实践总结

###  Web API

**推荐做法：**

1. **使用 FastAPI**
   - 异步支持
   - 自动文档
   - 类型安全

2. **提供多种接口**
   - 同步接口：`/api/v1/query`
   - 流式接口：`/api/v1/query/stream`
   - 异步接口：`/api/v1/query/async`

3. **会话持久化**
   - Redis: Session 缓存
   - PostgreSQL: 持久化存储

---

### 4.3 微服务化

**推荐做法：**

| 阶段 | 架构 | 适用场景 |
|------|------|----------|
| **MVP** | 单体应用 | < 1000 QPS, 小团队 |
| **成长期** | 单体 + 缓存/队列 | 1000-5000 QPS, 中型团队 |
| **成熟期** | 微服务 | > 5000 QPS, 大团队 |

**拆分原则：**
- 按负载特征拆分（Lightweight vs Heavy）
- LLM Gateway 独立服务
- 会话存储共享

---

### 4.4 性能优化

1. **LLM 调用优化**
   ```python
   # 使用更快的模型
   llm = ChatOpenAI(model="gpt-4o-mini")  # 更快更便宜

   # 缓存 Prompt
   system_prompt = SystemMessage(content="...", cache=True)

   # 批量调用
   results = await asyncio.gather(*[
       agent.query(msg) for msg in messages
   ])
   ```

2. **会话管理优化**
   ```python
   # Redis 缓存 Session
   session = await redis.get(f"session:{session_id}")

   # 定期清理过期 Session
   await redis.expire(f"session:{session_id}", 3600)
   ```

3. **规则匹配优先**
   ```python
   # Flows 用正则匹配，避免 LLM 调用
   if self._match_flow_pattern(message):
       return await self.flow_executor.execute(...)
   ```

---

## 五、完整示例

### 5.1 生产级 FastAPI 应用

参见 `api/main.py`（第二章示例）

### 5.2 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=password
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### 5.3 Kubernetes 部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: workflow-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: workflow-agent
  template:
    metadata:
      labels:
        app: workflow-agent
    spec:
      containers:
      - name: api
        image: workflow-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: workflow-agent
spec:
  selector:
    app: workflow-agent
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 总结

1. **Web API**
   - FastAPI 实现
   - 同步/流式/异步接口
   - 会话持久化（Redis + PostgreSQL）

2. **微服务化**
   - MVP: 单体应用
   - 成长期: 单体 + 缓存/队列
   - 成熟期: 微服务拆分

3. **关键建议**
   - 从简单开始，逐步演进
   - 根据实际负载决定是否微服务化
   - LLM Gateway 是关键共享服务
   - 监控和日志从第一天开始

---

**文档版本：** v1.0
**最后更新：** 2026-01-22
**相关文档：** [workflow-agent-v9.md](./workflow-agent-v9.md)
