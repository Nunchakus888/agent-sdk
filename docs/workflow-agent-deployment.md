# Workflow Agent 部署与架构最佳实践

> 多轮迭代、Web API 集成与微服务化架构指南

---

## 一、多轮迭代能力分析

### 1.1 当前支持情况

**✅ 已支持多轮迭代的场景：**

#### Scenario 1: Skills (Agent 模式) - 完全支持
```python
# Agent 模式的 Skill 天然支持多轮迭代
skill = {
    "skill_id": "blog_writer",
    "execution_mode": "agent",
    "system_prompt": "你是博客写手...",
    "max_iterations": 20  # ← 支持最多 20 轮迭代
}

# 内部执行流程
skill_agent = Agent(
    llm=llm,
    tools=[search_kb, save_draft, done],
    max_iterations=20,  # Agent 自带迭代能力
)
result = await skill_agent.query("写一篇关于 AI 的博客")
# Agent 会自动进行多轮迭代：
# 1. 调用 search_kb 搜索资料
# 2. 生成大纲
# 3. 撰写正文
# 4. 调用 save_draft 保存
# 5. 调用 done 完成
```

**工作原理：**
- Agent 内部的 `while iterations < max_iterations` 循环
- 支持工具调用、推理、决策的完整循环
- 自动管理上下文和历史

---

#### Scenario 2: 会话级多轮对话 - 部分支持

**当前实现：**
```python
# 每次 query() 调用是独立的
workflow_agent = WorkflowAgent(config, llm)

# 第 1 轮
response1 = await workflow_agent.query("帮我写博客", session_id="sess_001")
# → 触发 blog_writer skill（Agent 模式）
# → Skill 内部多轮迭代完成博客

# 第 2 轮（新的对话）
response2 = await workflow_agent.query("帮我改一下标题", session_id="sess_001")
# → 意图匹配：无法识别上下文
# → 需要改进
```

**问题：**
- ❌ Session 保存了 `messages`，但未被 `IntentMatcher` 使用
- ❌ 每次 `query()` 都重新进行意图匹配，无上下文连续性
- ❌ 多轮对话需要依赖 Skill 内部的 Agent

---

### 1.2 改进方案：支持会话级多轮对话

#### 方案 A: 主路由 Agent 化（推荐）

**设计思路：** WorkflowAgent 本身也是一个 Agent

```python
class WorkflowAgent:
    """
    主路由 Agent - 支持多轮对话

    核心改变：
    1. WorkflowAgent 继承或包含一个 Agent 实例
    2. Skills/Flows/Tools 都作为主 Agent 的工具
    3. 自然支持多轮对话和上下文
    """

    def __init__(self, config, llm):
        self.config = config
        self.llm = llm

        # 创建主路由 Agent
        self.router_agent = Agent(
            llm=llm,
            tools=self._build_all_tools(),  # 包含 Skills/Flows/Tools
            system_prompt=self._build_system_prompt(),
        )

    def _build_all_tools(self) -> list[Tool]:
        """
        将 Skills、Flows、Tools 都转换为工具
        """
        tools = []

        # 1. 基础工具（HTTP Tools）
        tools.extend(self._base_tools)

        # 2. Skills 作为工具
        for skill in self.config.skills:
            tools.append(self._skill_to_tool(skill))

        # 3. Flows 作为工具
        for flow in self.config.flows:
            tools.append(self._flow_to_tool(flow))

        # 4. System Actions 作为工具
        for action in self.config.system_actions:
            tools.append(self._system_to_tool(action))

        return tools

    def _skill_to_tool(self, skill: SkillDefinition) -> Tool:
        """将 Skill 封装为工具"""
        @tool(skill.description)
        async def skill_tool(user_request: str) -> str:
            return await self.skill_executor.execute(
                skill_id=skill.skill_id,
                user_request=user_request,
                parameters={},
            )
        return skill_tool

    async def query(self, message: str, session_id: str) -> str:
        """
        多轮对话入口

        优势：
        - 自动保持上下文
        - 支持多轮对话
        - Agent 自动决策调用哪个工具
        """
        session = await self._get_or_create_session(session_id)

        # 加载历史消息到 Agent
        if session.messages:
            self.router_agent.load_history(session.messages)

        # Agent 自动处理（多轮迭代）
        result = await self.router_agent.query(message)

        # 保存历史
        session.messages = self.router_agent.messages

        return result
```

**优势：**
- ✅ 完全支持多轮对话
- ✅ 上下文自动管理
- ✅ 复用 Agent 的所有能力（compaction、retry、streaming）
- ✅ 代码更简洁

**劣势：**
- ❌ 需要重构现有代码
- ❌ 失去了规则匹配的性能优势（Flows 的正则匹配）

---

#### 方案 B: 混合模式（平衡方案）

**设计思路：** 保留规则匹配 + Agent 模式补充

```python
class WorkflowAgent:
    """
    混合模式：规则匹配 + Agent 多轮

    决策树：
    1. 首轮对话 → 规则匹配 Flows（快速）
    2. 无匹配 → IntentMatcher（LLM 匹配）
    3. 后续轮次 → 检查是否在"对话态"
       - 如果在对话态 → 进入 Agent 模式（多轮）
       - 否则 → 重新匹配
    """

    async def query(self, message: str, session_id: str) -> str:
        session = await self._get_or_create_session(session_id)

        # 检查是否在"多轮对话态"
        if session.workflow_state.status == "in_conversation":
            # 进入 Agent 模式，继续对话
            return await self._continue_conversation(session, message)

        # 首轮：规则匹配 + 意图匹配
        flow = self._match_flow_by_pattern(message)
        if flow:
            return await self.flow_executor.execute(...)

        intent = await self.intent_matcher.match(message, session.messages)

        if intent.action_type == "skill" and self._is_conversational_skill(intent):
            # 启动多轮对话态
            session.workflow_state.status = "in_conversation"
            session.workflow_state.metadata["current_skill"] = intent.action_target

            # 创建对话 Agent
            return await self._start_conversation(session, intent, message)
        else:
            # 单次执行
            return await self._dispatch(...)

    async def _continue_conversation(self, session: Session, message: str) -> str:
        """继续多轮对话"""
        skill_id = session.workflow_state.metadata.get("current_skill")

        # 获取或创建对话 Agent
        agent = self._get_conversation_agent(session, skill_id)

        # 继续对话
        result = await agent.query(message)

        # 检查是否结束
        if self._is_conversation_complete(result):
            session.workflow_state.status = "ready"
            session.workflow_state.metadata.pop("current_skill")

        return result
```

**优势：**
- ✅ 保留规则匹配的性能优势
- ✅ 支持 Skills 的多轮对话
- ✅ 灵活控制对话态

**劣势：**
- ⚠️ 需要手动管理对话态
- ⚠️ 复杂度增加

---

#### 方案 C: 最简方案（当前架构微调）

**设计思路：** 最小改动，仅在 Skills 内部支持多轮

```python
# 当前架构已经支持这种模式
# Skills (Agent 模式) 内部自带多轮迭代
# 外层 WorkflowAgent 只负责路由

# 使用示例
response = await workflow_agent.query(
    "帮我写一篇博客，主题是 AI 的未来。要求：\n"
    "1. 搜索最新资料\n"
    "2. 生成大纲\n"
    "3. 撰写 2000 字正文\n"
    "4. 保存草稿",
    session_id="sess_001"
)
# → 触发 blog_writer skill (Agent 模式)
# → Skill 内部自动多轮迭代完成所有步骤
```

**优势：**
- ✅ 无需修改现有架构
- ✅ 简单直接

**劣势：**
- ❌ 不支持跨 query 的多轮对话
- ❌ 每次 query 都是独立任务

---

### 1.3 推荐方案

**对于大多数场景：方案 C（当前架构）已足够**

- Skills (Agent 模式) 内部多轮迭代可以处理大部分复杂任务
- Flows 和 Tools 本身就是单次调用
- 如果需要跨 query 的对话，可以在应用层实现会话管理

**对于需要真正多轮对话的场景：方案 A（主路由 Agent 化）**

- 适用于聊天机器人、客服助手等场景
- 需要完整的上下文理解和对话连续性

---

## 二、FastAPI Web API 集成

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

### 4.1 多轮迭代

**推荐做法：**

1. **大多数场景：使用 Skills (Agent 模式)**
   ```python
   # Skill 内部自动多轮迭代
   skill = {
       "execution_mode": "agent",
       "max_iterations": 20,
   }
   ```

2. **需要跨 query 对话：考虑主路由 Agent 化**
   ```python
   # WorkflowAgent 本身也是 Agent
   router_agent = Agent(llm, tools=all_skills_and_flows)
   ```

3. **简单任务：Flows/Tools 单次调用即可**

---

### 4.2 Web API

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

1. **多轮迭代**
   - 当前架构：Skills (Agent 模式) 已支持内部多轮迭代
   - 需要跨 query 对话：主路由 Agent 化

2. **Web API**
   - FastAPI 实现
   - 同步/流式/异步接口
   - 会话持久化（Redis + PostgreSQL）

3. **微服务化**
   - MVP: 单体应用
   - 成长期: 单体 + 缓存/队列
   - 成熟期: 微服务拆分

4. **关键建议**
   - 从简单开始，逐步演进
   - 根据实际负载决定是否微服务化
   - LLM Gateway 是关键共享服务
   - 监控和日志从第一天开始

---

**文档版本：** v1.0
**最后更新：** 2026-01-22
**相关文档：** [workflow-agent-v9.md](./workflow-agent-v9.md)
