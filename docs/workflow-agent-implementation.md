# Workflow Agent 完整实施方案

> **方案 A：主路由 Agent 化** - 支持会话级多轮对话
>
> **方案 B：按负载特征拆分** - 高性能微服务架构

---

## 一、核心架构

### 1.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │    API Gateway      │
                │    (Port: 8000)     │
                └──────────┬──────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
┌──────────▼──────────┐       ┌───────────▼───────────┐
│  Lightweight Service │       │    Heavy Service      │
│    (Port: 8001)      │       │    (Port: 8002)       │
│                      │       │                       │
│  • Flow 规则匹配     │       │  • Router Agent       │
│  • Flow API 调用     │       │  • Skills 执行        │
│  • Tool 直接调用     │       │  • LLM 调用           │
│                      │       │                       │
│  特性：              │       │  特性：               │
│  - 无状态            │       │  - 有状态             │
│  - 快速响应 <100ms   │       │  - 会话级多轮对话     │
│  - 水平扩展 (N 实例) │       │  - 按需扩展           │
└──────────┬───────────┘       └───────────┬───────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
              ┌────────────▼────────────┐
              │    Shared Services      │
              │                         │
              │  • Redis (Session)      │
              │  • PostgreSQL (DB)      │
              └─────────────────────────┘
```

### 1.2 核心设计理念

**方案 A：主路由 Agent 化**

```python
# 传统方式：每次 query 独立匹配
response1 = await workflow_agent.query("写博客", session_id)  # 意图匹配 → 执行
response2 = await workflow_agent.query("改标题", session_id)  # 无上下文，无法理解

# 方案 A：Router Agent 管理上下文
response1 = await router_agent.query("写博客", session_id)  # Agent 执行
response2 = await router_agent.query("改标题", session_id)  # Agent 记得上下文，继续处理
```

**方案 B：按负载特征拆分**

```
Lightweight (快) ─────────────────── Heavy (强)
     │                                   │
     ├─ Flows (正则 + API)              ├─ Router Agent (LLM 推理)
     ├─ Tools (HTTP 调用)               ├─ Skills Agent 模式
     └─ 无需 LLM                        └─ Skills Function 模式
```

---

## 二、项目结构

```
agent-sdk/
├── bu_agent_sdk/
│   ├── agent/
│   │   ├── service.py                  # 现有 Agent
│   │   ├── workflow_router_agent.py    # NEW: 主路由 Agent
│   │   └── workflow_state.py           # 状态管理
│   ├── tools/
│   │   ├── action_books.py             # 配置 Schema
│   │   └── config_loader.py            # 工具加载
│   └── workflow/
│       ├── executors.py                # 执行器
│       └── cache.py                    # 缓存
│
├── services/
│   ├── gateway/
│   │   ├── main.py                     # API Gateway
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── lightweight/
│   │   ├── main.py                     # Lightweight Service
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── heavy/
│       ├── main.py                     # Heavy Service
│       ├── requirements.txt
│       └── Dockerfile
│
├── config/
│   └── workflow_config.json            # 配置文件
│
├── docker-compose.yml
├── docker-compose.prod.yml
└── k8s/
    ├── namespace.yaml
    ├── configmap.yaml
    ├── secrets.yaml
    ├── redis.yaml
    ├── postgres.yaml
    ├── lightweight.yaml
    ├── heavy.yaml
    └── gateway.yaml
```

---

## 三、核心实现

### 3.1 WorkflowRouterAgent（主路由 Agent）

```python
# bu_agent_sdk/agent/workflow_router_agent.py

from typing import Any, AsyncIterator
from dataclasses import dataclass, field
from bu_agent_sdk.agent.service import Agent, TaskComplete
from bu_agent_sdk.agent.events import AgentEvent, TextEvent, FinalResponseEvent
from bu_agent_sdk.llm.base import BaseChatModel
from bu_agent_sdk.tools.decorator import tool
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema, SkillDefinition
from bu_agent_sdk.agent.workflow_state import WorkflowState, Session
from bu_agent_sdk.workflow.executors import SkillExecutor, FlowExecutor, SystemExecutor
from bu_agent_sdk.workflow.cache import compute_config_hash
import httpx


class WorkflowRouterAgent:
    """
    主路由 Agent - 支持会话级多轮对话

    核心设计：
    1. 内部包含一个 Router Agent（每个 session 独立）
    2. Skills/Flows/Tools 都封装为 Agent 的工具
    3. Agent 自动管理上下文和多轮对话

    使用示例：
        router = WorkflowRouterAgent(config, llm)

        # 多轮对话
        r1 = await router.query("帮我写博客", session_id="s1")
        r2 = await router.query("改一下标题", session_id="s1")  # 记得上下文
    """

    def __init__(
        self,
        config: WorkflowConfigSchema,
        llm: BaseChatModel,
        redis_url: str | None = None,
    ):
        self.config = config
        self.llm = llm
        self.redis_url = redis_url
        self.config_hash = compute_config_hash(config)

        # 执行器
        self.skill_executor = SkillExecutor(config, llm)
        self.flow_executor = FlowExecutor(config)
        self.system_executor = SystemExecutor(config)

        # 会话管理
        self._sessions: dict[str, Session] = {}
        self._router_agents: dict[str, Agent] = {}

        # HTTP 工具
        self._base_tools = self._load_base_tools()

    def _load_base_tools(self) -> list:
        """加载 HTTP 工具"""
        from bu_agent_sdk.tools.config_loader import HttpTool, ToolConfig

        tools = []
        for tool_dict in self.config.tools:
            tool_config = ToolConfig(**tool_dict)
            tools.append(HttpTool(config=tool_config))
        return tools

    def _build_router_tools(self) -> list:
        """将 Skills/Flows/SystemActions 封装为工具"""
        tools = list(self._base_tools)

        # Skills → 工具
        for skill in self.config.skills:
            tools.append(self._create_skill_tool(skill))

        # Flows → 工具
        for flow in self.config.flows:
            tools.append(self._create_flow_tool(flow))

        # System Actions → 工具
        for action in self.config.system_actions:
            tools.append(self._create_system_tool(action))

        return tools

    def _create_skill_tool(self, skill: SkillDefinition):
        """Skill 封装为工具"""
        executor = self.skill_executor

        @tool(skill.description)
        async def skill_tool(user_request: str) -> str:
            return await executor.execute(
                skill_id=skill.skill_id,
                user_request=user_request,
                parameters={},
            )

        skill_tool.name = f"skill_{skill.skill_id}"
        return skill_tool

    def _create_flow_tool(self, flow):
        """Flow 封装为工具"""
        executor = self.flow_executor
        sessions = self._sessions

        @tool(flow.description)
        async def flow_tool(user_message: str, session_id: str = "") -> str:
            session = sessions.get(session_id) or Session(
                session_id=session_id,
                agent_id="",
                workflow_state=WorkflowState(),
            )
            return await executor.execute(
                flow_id=flow.flow_id,
                user_message=user_message,
                parameters={},
                session=session,
            )

        flow_tool.name = f"flow_{flow.flow_id}"
        return flow_tool

    def _create_system_tool(self, action):
        """System Action 封装为工具"""
        executor = self.system_executor

        @tool(action.name)
        async def system_tool() -> str:
            return await executor.execute(
                action_id=action.action_id,
                parameters={},
            )

        system_tool.name = f"system_{action.action_id}"
        return system_tool

    def _build_system_prompt(self) -> str:
        """构建 Router Agent 的 system prompt"""
        basic = self.config.basic_settings
        name = basic.get("name", "助手")
        description = basic.get("description", "")
        language = basic.get("language", "中文")
        tone = basic.get("tone", "友好、专业")

        skills_list = "\n".join([
            f"  - skill_{s.skill_id}: {s.description}"
            for s in self.config.skills
        ]) or "  （无）"

        flows_list = "\n".join([
            f"  - flow_{f.flow_id}: {f.description}"
            for f in self.config.flows
        ]) or "  （无）"

        tools_list = "\n".join([
            f"  - {t['name']}: {t.get('description', '')}"
            for t in self.config.tools
        ]) or "  （无）"

        sop = self.config.sop or "根据用户需求，选择合适的工具完成任务。"
        constraints = self.config.constraints or "保持专业和礼貌。"

        return f"""你是 {name}，{description}

## 可用能力

### 技能（复杂任务）
{skills_list}

### 流程（业务流程）
{flows_list}

### 工具（单一功能）
{tools_list}

## 工作流程
{sop}

## 约束条件
{constraints}

## 对话风格
- 语言：{language}
- 语气：{tone}

## 重要提示
- 通过调用工具完成任务
- 支持多轮对话，记住上下文
- 根据用户需求智能选择工具
"""

    def _get_or_create_router_agent(self, session_id: str) -> Agent:
        """获取或创建 Router Agent"""
        if session_id in self._router_agents:
            return self._router_agents[session_id]

        agent = Agent(
            llm=self.llm,
            tools=self._build_router_tools(),
            system_prompt=self._build_system_prompt(),
            max_iterations=50,
            tool_choice="auto",
        )

        self._router_agents[session_id] = agent
        return agent

    async def _get_or_create_session(self, session_id: str) -> Session:
        """获取或创建会话"""
        if session_id in self._sessions:
            return self._sessions[session_id]

        session = Session(
            session_id=session_id,
            agent_id=f"router_{self.config_hash[:8]}",
            workflow_state=WorkflowState(config_hash=self.config_hash),
            messages=[],
        )
        self._sessions[session_id] = session
        return session

    async def query(self, message: str, session_id: str) -> str:
        """
        处理用户消息（支持会话级多轮对话）
        """
        session = await self._get_or_create_session(session_id)

        # 配置变更检测
        if session.workflow_state.config_hash != self.config_hash:
            if session_id in self._router_agents:
                del self._router_agents[session_id]
            session.workflow_state.config_hash = self.config_hash

        router_agent = self._get_or_create_router_agent(session_id)

        # 问候语
        if session.workflow_state.need_greeting and self.config.greeting:
            session.workflow_state.need_greeting = False
            return self.config.greeting

        # Router Agent 处理
        try:
            result = await router_agent.query(message)
            session.messages = router_agent.messages
            return result
        except TaskComplete as e:
            return e.message
        except Exception as e:
            return f"❌ 处理失败: {e}"

    async def query_stream(
        self,
        message: str,
        session_id: str,
    ) -> AsyncIterator[AgentEvent]:
        """流式查询"""
        session = await self._get_or_create_session(session_id)

        if session.workflow_state.config_hash != self.config_hash:
            if session_id in self._router_agents:
                del self._router_agents[session_id]
            session.workflow_state.config_hash = self.config_hash

        router_agent = self._get_or_create_router_agent(session_id)

        # 问候语
        if session.workflow_state.need_greeting and self.config.greeting:
            session.workflow_state.need_greeting = False
            yield TextEvent(content=self.config.greeting)
            yield FinalResponseEvent(content=self.config.greeting)
            return

        async for event in router_agent.query_stream(message):
            yield event

        session.messages = router_agent.messages

    def clear_session(self, session_id: str):
        """清除会话"""
        self._sessions.pop(session_id, None)
        self._router_agents.pop(session_id, None)

    def get_session_count(self) -> int:
        """获取会话数量"""
        return len(self._sessions)
```

### 3.2 Lightweight Service

```python
# services/lightweight/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import re
import json
from pathlib import Path
from typing import Any

app = FastAPI(title="Lightweight Service", version="1.0.0")

# ============================================================================
# 配置加载
# ============================================================================

CONFIG_PATH = Path("/app/config/workflow_config.json")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {"flows": [], "tools": []}

# 预编译 Flow 正则
FLOW_PATTERNS: list[tuple[re.Pattern, dict]] = []
for flow in CONFIG.get("flows", []):
    for pattern in flow.get("trigger_patterns", []):
        try:
            FLOW_PATTERNS.append((re.compile(pattern, re.IGNORECASE), flow))
        except re.error:
            pass

# ============================================================================
# 数据模型
# ============================================================================

class MatchFlowRequest(BaseModel):
    message: str

class ExecuteFlowRequest(BaseModel):
    flow_id: str
    user_message: str
    session_id: str
    parameters: dict = {}

class ExecuteToolRequest(BaseModel):
    tool_name: str
    parameters: dict = {}

# ============================================================================
# API 端点
# ============================================================================

@app.post("/api/v1/match-flow")
async def match_flow(request: MatchFlowRequest):
    """规则匹配 Flow（快速，无 LLM）"""
    for pattern, flow in FLOW_PATTERNS:
        if pattern.search(request.message):
            return {
                "matched": True,
                "flow_id": flow["flow_id"],
                "flow_name": flow["name"],
            }
    return {"matched": False, "flow_id": None}


@app.post("/api/v1/execute-flow")
async def execute_flow(request: ExecuteFlowRequest):
    """执行 Flow（直接 API 调用）"""
    flow = next(
        (f for f in CONFIG["flows"] if f["flow_id"] == request.flow_id),
        None
    )
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    endpoint = flow.get("endpoint", {})
    url = endpoint.get("url", "")
    method = endpoint.get("method", "POST")
    headers = endpoint.get("headers", {"Content-Type": "application/json"})

    body = _substitute(
        endpoint.get("body", {}),
        {
            "user_message": request.user_message,
            "session_id": request.session_id,
            **request.parameters,
        }
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(method=method, url=url, headers=headers, json=body)

    if resp.is_success:
        result = resp.text
        template = flow.get("response_template", "✅ 执行完成\n\n{result}")
        return {
            "success": True,
            "flow_id": request.flow_id,
            "result": template.replace("{result}", result),
        }
    else:
        return {
            "success": False,
            "flow_id": request.flow_id,
            "error": f"HTTP {resp.status_code}",
        }


@app.post("/api/v1/execute-tool")
async def execute_tool(request: ExecuteToolRequest):
    """执行 Tool（HTTP 调用）"""
    tool_config = next(
        (t for t in CONFIG["tools"] if t["name"] == request.tool_name),
        None
    )
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool not found")

    endpoint = tool_config.get("endpoint", {})
    url = endpoint.get("url", "")
    method = endpoint.get("method", "POST")
    body = _substitute(endpoint.get("body", {}), request.parameters)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(method=method, url=url, json=body)

    return {
        "success": resp.is_success,
        "tool_name": request.tool_name,
        "result": resp.text if resp.is_success else f"HTTP {resp.status_code}",
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "service": "lightweight", "flows": len(FLOW_PATTERNS)}


# ============================================================================
# 工具函数
# ============================================================================

def _substitute(template: Any, params: dict) -> Any:
    """参数替换"""
    if isinstance(template, str):
        for k, v in params.items():
            template = template.replace(f"{{{k}}}", str(v))
        return template
    elif isinstance(template, dict):
        return {k: _substitute(v, params) for k, v in template.items()}
    elif isinstance(template, list):
        return [_substitute(item, params) for item in template]
    return template


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 3.3 Heavy Service

```python
# services/heavy/main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from pathlib import Path
import os

app = FastAPI(title="Heavy Service", version="1.0.0")

# ============================================================================
# 初始化
# ============================================================================

from bu_agent_sdk.agent.workflow_router_agent import WorkflowRouterAgent
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema
from bu_agent_sdk.llm import ChatOpenAI

CONFIG_PATH = Path("/app/config/workflow_config.json")
with open(CONFIG_PATH, encoding="utf-8") as f:
    config_data = json.load(f)

config = WorkflowConfigSchema(**config_data)
llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o"))
router_agent = WorkflowRouterAgent(config=config, llm=llm)

# ============================================================================
# 数据模型
# ============================================================================

class QueryRequest(BaseModel):
    message: str
    session_id: str
    user_id: str | None = None

class QueryResponse(BaseModel):
    session_id: str
    message: str
    status: str

# ============================================================================
# API 端点
# ============================================================================

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """主查询接口（支持会话级多轮对话）"""
    try:
        result = await router_agent.query(
            message=request.message,
            session_id=request.session_id,
        )
        return QueryResponse(
            session_id=request.session_id,
            message=result,
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/stream")
async def query_stream(request: QueryRequest):
    """流式查询接口"""
    async def generate():
        async for event in router_agent.query_stream(
            message=request.message,
            session_id=request.session_id,
        ):
            data = {
                "type": event.__class__.__name__,
                "data": getattr(event, "__dict__", {}),
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """清除会话"""
    router_agent.clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "heavy",
        "sessions": router_agent.get_session_count(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### 3.4 API Gateway

```python
# services/gateway/main.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import os

app = FastAPI(title="API Gateway", version="1.0.0")

LIGHTWEIGHT_URL = os.getenv("LIGHTWEIGHT_URL", "http://lightweight:8001")
HEAVY_URL = os.getenv("HEAVY_URL", "http://heavy:8002")

# ============================================================================
# API 端点
# ============================================================================

@app.post("/api/v1/query")
async def query(request: Request):
    """
    智能路由

    策略：
    1. 尝试规则匹配 Flow（Lightweight）
    2. 匹配成功 → 执行 Flow（Lightweight）
    3. 匹配失败 → Router Agent（Heavy）
    """
    body = await request.json()
    message = body.get("message", "")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. 尝试规则匹配
        match_resp = await client.post(
            f"{LIGHTWEIGHT_URL}/api/v1/match-flow",
            json={"message": message}
        )
        match_result = match_resp.json()

        # 2. 匹配成功 → Lightweight 执行
        if match_result.get("matched"):
            flow_resp = await client.post(
                f"{LIGHTWEIGHT_URL}/api/v1/execute-flow",
                json={
                    "flow_id": match_result["flow_id"],
                    "user_message": message,
                    "session_id": body.get("session_id", ""),
                    "parameters": body.get("parameters", {}),
                }
            )
            result = flow_resp.json()
            return {
                "session_id": body.get("session_id"),
                "message": result.get("result", result.get("error", "")),
                "status": "success" if result.get("success") else "error",
                "routed_to": "lightweight",
            }

        # 3. 匹配失败 → Heavy 执行
        heavy_resp = await client.post(
            f"{HEAVY_URL}/api/v1/query",
            json=body
        )
        result = heavy_resp.json()
        result["routed_to"] = "heavy"
        return result


@app.post("/api/v1/query/stream")
async def query_stream(request: Request):
    """流式查询（直接转发到 Heavy）"""
    body = await request.json()

    async def generate():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{HEAVY_URL}/api/v1/query/stream",
                json=body
            ) as resp:
                async for chunk in resp.aiter_text():
                    yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """清除会话"""
    async with httpx.AsyncClient() as client:
        resp = await client.delete(f"{HEAVY_URL}/api/v1/session/{session_id}")
        return resp.json()


@app.get("/health")
async def health():
    """健康检查（聚合）"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            lw = await client.get(f"{LIGHTWEIGHT_URL}/health")
            lw_status = lw.json()
        except Exception:
            lw_status = {"status": "unhealthy"}

        try:
            hv = await client.get(f"{HEAVY_URL}/health")
            hv_status = hv.json()
        except Exception:
            hv_status = {"status": "unhealthy"}

    return {
        "status": "healthy",
        "service": "gateway",
        "lightweight": lw_status,
        "heavy": hv_status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 四、部署配置

### 4.1 Docker Compose（开发环境）

```yaml
# docker-compose.yml

version: '3.8'

services:
  gateway:
    build: ./services/gateway
    ports:
      - "8000:8000"
    environment:
      - LIGHTWEIGHT_URL=http://lightweight:8001
      - HEAVY_URL=http://heavy:8002
    depends_on:
      - lightweight
      - heavy

  lightweight:
    build: ./services/lightweight
    ports:
      - "8001:8001"
    volumes:
      - ./config:/app/config:ro
    deploy:
      replicas: 2

  heavy:
    build: ./services/heavy
    ports:
      - "8002:8002"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_MODEL=gpt-4o
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./config:/app/config:ro
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### 4.2 Dockerfile

```dockerfile
# services/lightweight/Dockerfile

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

```dockerfile
# services/heavy/Dockerfile

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### 4.3 Kubernetes 部署

```yaml
# k8s/lightweight.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightweight
spec:
  replicas: 5
  selector:
    matchLabels:
      app: lightweight
  template:
    metadata:
      labels:
        app: lightweight
    spec:
      containers:
      - name: lightweight
        image: workflow-lightweight:latest
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: config
          mountPath: /app/config
      volumes:
      - name: config
        configMap:
          name: workflow-config
---
apiVersion: v1
kind: Service
metadata:
  name: lightweight
spec:
  selector:
    app: lightweight
  ports:
  - port: 8001
```

```yaml
# k8s/heavy.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: heavy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: heavy
  template:
    metadata:
      labels:
        app: heavy
    spec:
      containers:
      - name: heavy
        image: workflow-heavy:latest
        ports:
        - containerPort: 8002
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        - name: REDIS_URL
          value: "redis://redis:6379"
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: config
          mountPath: /app/config
      volumes:
      - name: config
        configMap:
          name: workflow-config
---
apiVersion: v1
kind: Service
metadata:
  name: heavy
spec:
  selector:
    app: heavy
  ports:
  - port: 8002
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: heavy-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: heavy
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 五、使用示例

### 5.1 启动服务

```bash
# 开发环境
docker-compose up -d

# 生产环境
kubectl apply -f k8s/
```

### 5.2 API 调用

```bash
# 多轮对话示例
# 第 1 轮
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我写一篇关于 AI 的博客", "session_id": "user_001"}'

# 第 2 轮（记得上下文）
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"message": "标题改得更吸引人一点", "session_id": "user_001"}'

# Flow 触发示例（自动路由到 Lightweight）
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"message": "我要请假", "session_id": "user_002"}'

# 流式查询
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "分析这段代码的问题", "session_id": "user_003"}'
```

### 5.3 健康检查

```bash
curl http://localhost:8000/health
```

---

## 六、实施路线图

### Phase 1: 核心实现（1 周）

- [ ] 实现 WorkflowRouterAgent
- [ ] Skills/Flows/Tools 封装为工具
- [ ] 会话管理和上下文保持
- [ ] 单元测试

### Phase 2: 微服务化（1 周）

- [ ] Lightweight Service 实现
- [ ] Heavy Service 实现
- [ ] API Gateway 实现
- [ ] Docker Compose 本地测试

### Phase 3: 生产部署（1 周）

- [ ] Kubernetes 部署配置
- [ ] HPA 自动扩展
- [ ] 监控和日志
- [ ] 性能测试

### Phase 4: 优化（持续）

- [ ] Redis 会话持久化
- [ ] 分布式追踪（Jaeger）
- [ ] LLM 调用优化
- [ ] 高可用方案

---

## 七、核心优势

| 特性 | 说明 |
|------|------|
| **会话级多轮对话** | Router Agent 自动管理上下文 |
| **智能路由** | Flow 规则匹配 → Lightweight，其他 → Heavy |
| **高性能** | Lightweight 无 LLM，快速响应 |
| **按需扩展** | Heavy 独立扩展，资源利用率高 |
| **简单部署** | Docker Compose / Kubernetes |

---

**文档版本：** v1.0
**最后更新：** 2026-01-22
