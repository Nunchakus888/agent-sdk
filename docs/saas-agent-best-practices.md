# SaaS Agent 服务最佳实践指南

> 基于 BU Agent SDK 构建面向多业务接入的 Agent 服务架构设计

---

## 目录

1. [架构总览](#1-架构总览)
2. [微服务架构设计](#2-微服务架构设计)
3. [高可用与稳定性](#3-高可用与稳定性)
4. [OAuth 认证体系](#4-oauth-认证体系)
5. [租户配置管理](#5-租户配置管理)
6. [知识库系统](#6-知识库系统)
7. [MCP 工具协议](#7-mcp-工具协议)
8. [三方 API 集成](#8-三方-api-集成)
9. [并发性设计](#9-并发性设计)
10. [上下文管理](#10-上下文管理)
11. [持久化方案](#11-持久化方案)
12. [工具系统设计](#12-工具系统设计)
13. [Prompt 管理](#13-prompt-管理)
14. [多租户隔离](#14-多租户隔离)
15. [可扩展性设计](#15-可扩展性设计)
16. [可维护性保障](#16-可维护性保障)
17. [安全性考量](#17-安全性考量)
18. [监控与可观测性](#18-监控与可观测性)
19. [参考实现](#19-参考实现)

---

## 1. 架构总览

### 1.1 SaaS Agent 服务分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                         │
│              (认证、限流、路由、负载均衡)                          │
├─────────────────────────────────────────────────────────────────┤
│                      Business Router Layer                       │
│         (业务路由、租户识别、意图分发、Flow 预匹配)                 │
├─────────────────────────────────────────────────────────────────┤
│                       Agent Service Layer                        │
│    ┌──────────────┬──────────────┬──────────────┐               │
│    │ Agent Pool   │ Session Mgr  │ Tool Registry│               │
│    │ (实例池管理)  │ (会话管理)    │ (工具注册)   │               │
│    └──────────────┴──────────────┴──────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                       Core Engine Layer                          │
│    ┌──────────────┬──────────────┬──────────────┐               │
│    │ BU Agent SDK │ LLM Adapter  │ Context Mgr  │               │
│    │ (核心循环)    │ (模型适配)    │ (上下文管理) │               │
│    └──────────────┴──────────────┴──────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure Layer                        │
│    ┌──────────────┬──────────────┬──────────────┐               │
│    │ Redis/Cache  │ PostgreSQL   │ Message Queue│               │
│    │ (缓存/锁)     │ (持久化)     │ (异步任务)   │               │
│    └──────────────┴──────────────┴──────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心设计原则

| 原则 | 说明 | 实现方式 |
|------|------|---------|
| **模型驱动** | 信任 LLM 的决策能力，最小化硬编码逻辑 | 使用 SDK 的 for-loop 模式 |
| **无状态服务** | Agent 实例可任意调度和销毁 | 状态外置到 Redis/DB |
| **租户隔离** | 数据和资源完全隔离 | 依赖注入 + 命名空间 |
| **可插拔扩展** | 工具、Prompt、模型均可动态配置 | 注册表模式 |
| **优雅降级** | 单点故障不影响整体服务 | 熔断器 + 降级策略 |

---

## 2. 微服务架构设计

### 2.1 服务拆分策略

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            微服务架构全景图                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web App   │  │  Mobile App │  │  Third-party│  │   Webhook   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│                          ┌────────▼────────┐                               │
│                          │   API Gateway   │  Kong / Traefik               │
│                          │  (认证/限流/路由) │                               │
│                          └────────┬────────┘                               │
│                                   │                                         │
│    ┌──────────────────────────────┼──────────────────────────────┐         │
│    │                              │                              │         │
│    ▼                              ▼                              ▼         │
│ ┌──────────────┐          ┌──────────────┐          ┌──────────────┐      │
│ │ Auth Service │          │ Agent Service│          │ Admin Service│      │
│ │  (认证授权)   │          │  (核心对话)   │          │  (管理后台)   │      │
│ └──────┬───────┘          └──────┬───────┘          └──────┬───────┘      │
│        │                         │                         │              │
│        │         ┌───────────────┼───────────────┐         │              │
│        │         │               │               │         │              │
│        ▼         ▼               ▼               ▼         ▼              │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│ │Tool Service│ │ KB Service │ │Config Svc  │ │Billing Svc │              │
│ │ (工具执行)  │ │ (知识库)    │ │ (配置中心)  │ │ (计费)     │              │
│ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘              │
│       │              │              │              │                      │
│       └──────────────┴──────────────┴──────────────┘                      │
│                              │                                             │
│              ┌───────────────┼───────────────┐                            │
│              │               │               │                            │
│              ▼               ▼               ▼                            │
│       ┌────────────┐  ┌────────────┐  ┌────────────┐                     │
│       │   Redis    │  │ PostgreSQL │  │  Qdrant    │                     │
│       │  (缓存/队列)│  │ (持久化)   │  │ (向量库)   │                     │
│       └────────────┘  └────────────┘  └────────────┘                     │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心服务定义

```python
# 服务职责划分
SERVICES = {
    "api-gateway": {
        "职责": "统一入口、认证、限流、路由",
        "技术": "Kong / Traefik / APISIX",
        "实例数": "2-4 (高可用)",
    },
    "auth-service": {
        "职责": "OAuth2.0、JWT、用户管理",
        "技术": "FastAPI + Redis",
        "实例数": "2+",
    },
    "agent-service": {
        "职责": "Agent 核心循环、对话处理",
        "技术": "FastAPI + BU Agent SDK",
        "实例数": "按负载弹性伸缩",
        "特点": "CPU/内存密集，需要弹性伸缩",
    },
    "tool-service": {
        "职责": "工具执行、MCP 代理",
        "技术": "FastAPI + 沙箱环境",
        "实例数": "按负载伸缩",
    },
    "kb-service": {
        "职责": "知识库管理、向量检索",
        "技术": "FastAPI + Qdrant/Milvus",
        "实例数": "2+",
    },
    "config-service": {
        "职责": "租户配置、Prompt 管理",
        "技术": "FastAPI + PostgreSQL",
        "实例数": "2+",
    },
}
```

### 2.3 服务间通信

```python
from enum import Enum
from pydantic import BaseModel
import httpx
import asyncio

class CommunicationType(Enum):
    SYNC_HTTP = "sync_http"      # 同步 HTTP (低延迟要求)
    ASYNC_QUEUE = "async_queue"  # 异步队列 (可延迟处理)
    GRPC = "grpc"                # gRPC (高性能内部调用)


class ServiceClient:
    """服务间调用客户端"""

    def __init__(self, service_name: str, base_url: str):
        self.service_name = service_name
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            limits=httpx.Limits(max_connections=100),
        )

    async def call(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict:
        """带重试和熔断的服务调用"""
        for attempt in range(3):
            try:
                resp = await self._client.request(method, path, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise
        raise ServiceUnavailable(self.service_name)


# 服务发现（Kubernetes 原生）
class ServiceRegistry:
    """基于 K8s DNS 的服务发现"""

    SERVICES = {
        "auth": "http://auth-service.agent.svc.cluster.local:8000",
        "tool": "http://tool-service.agent.svc.cluster.local:8000",
        "kb": "http://kb-service.agent.svc.cluster.local:8000",
        "config": "http://config-service.agent.svc.cluster.local:8000",
    }

    @classmethod
    def get_client(cls, service: str) -> ServiceClient:
        return ServiceClient(service, cls.SERVICES[service])
```

### 2.4 Kubernetes 部署配置

```yaml
# k8s/agent-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  namespace: agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
      - name: agent
        image: agent-service:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: redis-url
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 3. 高可用与稳定性

### 3.1 熔断器模式

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开（尝试恢复）


@dataclass
class CircuitConfig:
    failure_threshold: int = 5       # 失败阈值
    success_threshold: int = 3       # 恢复阈值
    timeout: float = 30.0            # 熔断超时
    half_open_max_calls: int = 3     # 半开状态最大尝试数


class CircuitBreaker:
    """熔断器"""

    def __init__(self, name: str, config: CircuitConfig = None):
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime = None
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """通过熔断器执行函数"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitOpenError(self.name)

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def _on_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time > timedelta(
            seconds=self.config.timeout
        )


# 使用示例
llm_circuit = CircuitBreaker("llm_api", CircuitConfig(failure_threshold=3))

async def safe_llm_call(messages):
    return await llm_circuit.call(llm.ainvoke, messages)
```

### 3.2 优雅降级策略

```python
from typing import Optional, Callable
from functools import wraps

class DegradationStrategy:
    """降级策略"""

    @staticmethod
    def fallback_response() -> str:
        """返回兜底响应"""
        return "抱歉，服务暂时繁忙，请稍后重试。"

    @staticmethod
    def cached_response(cache_key: str) -> Optional[str]:
        """返回缓存的响应"""
        return cache.get(f"fallback:{cache_key}")

    @staticmethod
    def simplified_mode(agent: Agent) -> Agent:
        """简化模式：减少工具，用小模型"""
        return Agent(
            llm=ChatAnthropic(model="claude-3-haiku-20240307"),
            tools=[],  # 无工具
            system_prompt="You are a helpful assistant. Keep responses brief.",
        )


def with_degradation(fallback_func: Callable):
    """降级装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except CircuitOpenError:
                return await fallback_func(*args, **kwargs)
            except RateLimitError:
                return await fallback_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Unexpected error, degrading: {e}")
                return await fallback_func(*args, **kwargs)
        return wrapper
    return decorator


# 使用
@with_degradation(fallback_func=lambda *a, **kw: DegradationStrategy.fallback_response())
async def chat(request: ChatRequest):
    return await agent.query(request.message)
```

### 3.3 负载均衡与限流

```python
from redis import asyncio as aioredis
from typing import Tuple

class AdaptiveRateLimiter:
    """自适应限流器

    根据系统负载动态调整限流阈值
    """

    def __init__(self, redis: aioredis.Redis):
        self.redis = redis

    async def check(
        self,
        key: str,
        base_limit: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, int]:
        """
        Returns: (allowed, current_count, adjusted_limit)
        """
        # 获取系统负载因子
        load_factor = await self._get_load_factor()

        # 动态调整限制
        adjusted_limit = int(base_limit * load_factor)

        # 滑动窗口计数
        pipe = self.redis.pipeline()
        now = int(time.time())
        window_key = f"ratelimit:{key}:{now // window_seconds}"

        pipe.incr(window_key)
        pipe.expire(window_key, window_seconds * 2)

        results = await pipe.execute()
        current = results[0]

        return current <= adjusted_limit, current, adjusted_limit

    async def _get_load_factor(self) -> float:
        """获取负载因子 (0.5 ~ 1.0)"""
        # 从监控系统获取当前负载
        cpu_usage = await self._get_metric("cpu_usage")
        memory_usage = await self._get_metric("memory_usage")
        queue_depth = await self._get_metric("queue_depth")

        # 综合计算负载因子
        if cpu_usage > 0.9 or memory_usage > 0.9:
            return 0.5  # 高负载，限流 50%
        elif cpu_usage > 0.7 or memory_usage > 0.7:
            return 0.7  # 中等负载
        else:
            return 1.0  # 正常


class LoadBalancer:
    """加权负载均衡"""

    def __init__(self):
        self.backends: dict[str, float] = {}  # backend -> weight

    def register(self, backend: str, weight: float = 1.0):
        self.backends[backend] = weight

    def select(self) -> str:
        """加权随机选择"""
        total = sum(self.backends.values())
        r = random.uniform(0, total)
        cumulative = 0
        for backend, weight in self.backends.items():
            cumulative += weight
            if r <= cumulative:
                return backend
        return list(self.backends.keys())[0]

    async def update_weights(self):
        """根据健康检查更新权重"""
        for backend in self.backends:
            health = await self._check_health(backend)
            if health["status"] == "healthy":
                latency = health.get("latency_ms", 100)
                # 延迟越低，权重越高
                self.backends[backend] = 1000 / max(latency, 1)
            else:
                self.backends[backend] = 0  # 不健康的节点权重为 0
```

### 3.4 故障恢复

```python
class RecoveryManager:
    """故障恢复管理器"""

    def __init__(self, session_store: SessionStore):
        self.session_store = session_store

    async def checkpoint(self, session_id: str, agent: Agent):
        """保存检查点"""
        await self.session_store.save(
            f"checkpoint:{session_id}",
            {
                "messages": agent.messages,
                "iteration": agent._current_iteration,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def recover(self, session_id: str) -> Optional[Agent]:
        """从检查点恢复"""
        checkpoint = await self.session_store.get(f"checkpoint:{session_id}")
        if not checkpoint:
            return None

        agent = await self._create_agent()
        agent.load_history(checkpoint["messages"])
        return agent

    async def handle_partial_failure(
        self,
        session_id: str,
        agent: Agent,
        error: Exception,
    ) -> str:
        """处理部分失败"""
        # 1. 保存当前状态
        await self.checkpoint(session_id, agent)

        # 2. 记录错误
        logger.error(f"Partial failure in session {session_id}: {error}")

        # 3. 尝试继续
        if isinstance(error, ToolExecutionError):
            # 工具失败，让模型知道并尝试其他方法
            return f"工具执行失败: {error.message}。正在尝试其他方法..."

        # 4. 无法恢复，返回友好消息
        return "遇到了一些问题，已保存进度。您可以稍后继续。"
```

---

## 4. OAuth 认证体系

### 4.1 OAuth 2.0 完整实现

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel
import secrets

# 配置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE = 30  # minutes
REFRESH_TOKEN_EXPIRE = 7  # days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: str          # user_id
    tenant_id: str
    scopes: list[str]
    exp: datetime
    iat: datetime
    jti: str          # JWT ID (用于撤销)


class OAuthService:
    """OAuth 2.0 服务"""

    def __init__(self, redis: aioredis.Redis, db: AsyncSession):
        self.redis = redis
        self.db = db

    # ==================== 密码模式 ====================
    async def authenticate_password(
        self,
        username: str,
        password: str,
        tenant_id: str,
    ) -> Optional[Token]:
        """密码模式认证"""
        user = await self._get_user(username, tenant_id)
        if not user or not pwd_context.verify(password, user.hashed_password):
            return None

        return await self._create_tokens(user)

    # ==================== 客户端凭证模式 ====================
    async def authenticate_client(
        self,
        client_id: str,
        client_secret: str,
    ) -> Optional[Token]:
        """客户端凭证模式（服务间调用）"""
        client = await self._get_client(client_id)
        if not client or not secrets.compare_digest(
            client.secret_hash,
            self._hash_secret(client_secret),
        ):
            return None

        return await self._create_tokens(client, is_client=True)

    # ==================== 授权码模式 ====================
    async def create_authorization_code(
        self,
        client_id: str,
        redirect_uri: str,
        scope: str,
        user_id: str,
        state: str,
    ) -> str:
        """创建授权码"""
        code = secrets.token_urlsafe(32)
        await self.redis.setex(
            f"auth_code:{code}",
            300,  # 5 分钟过期
            json.dumps({
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "user_id": user_id,
            }),
        )
        return code

    async def exchange_code(
        self,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> Optional[Token]:
        """授权码换取 Token"""
        data = await self.redis.get(f"auth_code:{code}")
        if not data:
            return None

        auth_data = json.loads(data)
        if (
            auth_data["client_id"] != client_id
            or auth_data["redirect_uri"] != redirect_uri
        ):
            return None

        # 验证客户端
        client = await self._get_client(client_id)
        if not client or client.secret != client_secret:
            return None

        # 删除已使用的授权码
        await self.redis.delete(f"auth_code:{code}")

        user = await self._get_user_by_id(auth_data["user_id"])
        return await self._create_tokens(user, scopes=auth_data["scope"].split())

    # ==================== Token 管理 ====================
    async def refresh_token(self, refresh_token: str) -> Optional[Token]:
        """刷新 Token"""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "refresh":
                return None

            # 检查是否已撤销
            if await self.redis.exists(f"revoked:{payload['jti']}"):
                return None

            user = await self._get_user_by_id(payload["sub"])
            if not user:
                return None

            return await self._create_tokens(user)

        except JWTError:
            return None

    async def revoke_token(self, token: str):
        """撤销 Token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            # 将 JTI 加入黑名单
            ttl = payload["exp"] - datetime.utcnow().timestamp()
            if ttl > 0:
                await self.redis.setex(f"revoked:{payload['jti']}", int(ttl), "1")
        except JWTError:
            pass

    async def _create_tokens(
        self,
        subject,
        scopes: list[str] = None,
        is_client: bool = False,
    ) -> Token:
        """创建 Token 对"""
        now = datetime.utcnow()
        jti = secrets.token_urlsafe(16)

        access_payload = {
            "sub": str(subject.id),
            "tenant_id": subject.tenant_id if hasattr(subject, "tenant_id") else None,
            "scopes": scopes or ["agent:chat"],
            "type": "access",
            "jti": jti,
            "iat": now,
            "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE),
        }

        refresh_payload = {
            "sub": str(subject.id),
            "type": "refresh",
            "jti": f"{jti}_refresh",
            "iat": now,
            "exp": now + timedelta(days=REFRESH_TOKEN_EXPIRE),
        }

        return Token(
            access_token=jwt.encode(access_payload, SECRET_KEY, algorithm=ALGORITHM),
            refresh_token=jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM),
            expires_in=ACCESS_TOKEN_EXPIRE * 60,
        )


# FastAPI 依赖
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """验证并获取当前用户"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        # 检查是否已撤销
        if await redis.exists(f"revoked:{payload['jti']}"):
            raise HTTPException(401, "Token revoked")

        user = await get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(401, "User not found")

        return user
    except JWTError:
        raise HTTPException(401, "Invalid token")
```

### 4.2 API Key 认证

```python
class APIKeyService:
    """API Key 管理服务"""

    def __init__(self, db: AsyncSession, redis: aioredis.Redis):
        self.db = db
        self.redis = redis

    async def create_key(
        self,
        tenant_id: str,
        name: str,
        scopes: list[str],
        expires_at: Optional[datetime] = None,
    ) -> Tuple[str, str]:
        """创建 API Key，返回 (key_id, secret)"""
        key_id = f"ak_{secrets.token_urlsafe(16)}"
        secret = secrets.token_urlsafe(32)
        secret_hash = pwd_context.hash(secret)

        api_key = APIKey(
            id=key_id,
            tenant_id=tenant_id,
            name=name,
            secret_hash=secret_hash,
            scopes=scopes,
            expires_at=expires_at,
        )
        self.db.add(api_key)
        await self.db.commit()

        return key_id, secret

    async def validate_key(self, key_id: str, secret: str) -> Optional[APIKey]:
        """验证 API Key"""
        # 先查缓存
        cached = await self.redis.get(f"apikey:{key_id}")
        if cached:
            api_key = APIKey.parse_raw(cached)
        else:
            api_key = await self.db.get(APIKey, key_id)
            if api_key:
                await self.redis.setex(
                    f"apikey:{key_id}",
                    3600,
                    api_key.json(),
                )

        if not api_key:
            return None

        if not pwd_context.verify(secret, api_key.secret_hash):
            return None

        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        return api_key

    async def revoke_key(self, key_id: str):
        """撤销 API Key"""
        await self.db.execute(
            update(APIKey).where(APIKey.id == key_id).values(revoked=True)
        )
        await self.db.commit()
        await self.redis.delete(f"apikey:{key_id}")
```

---

## 5. 租户配置管理

### 5.1 配置数据模型

```python
from sqlalchemy import Column, String, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB

class TenantModel(Base):
    """租户表"""
    __tablename__ = "tenants"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    tier = Column(SQLEnum(TenantTier), default=TenantTier.FREE)
    status = Column(String(32), default="active")

    # Agent 配置
    agent_config = Column(JSONB, default={})
    # 包含: model, max_tokens, temperature, max_iterations

    # 工具配置
    enabled_tools = Column(JSONB, default=[])
    tool_configs = Column(JSONB, default={})

    # Prompt 配置
    system_prompt = Column(Text, nullable=True)
    prompt_template_id = Column(String(64), nullable=True)

    # 知识库配置
    kb_namespaces = Column(JSONB, default=[])

    # 限制配置
    limits = Column(JSONB, default={})

    # OAuth 配置
    oauth_config = Column(JSONB, default={})

    # 元数据
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


class AgentConfigModel(Base):
    """Agent 配置表（支持多 Agent）"""
    __tablename__ = "agent_configs"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), ForeignKey("tenants.id"))
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # LLM 配置
    llm_provider = Column(String(32), default="anthropic")
    llm_model = Column(String(64), default="claude-3-5-sonnet-20241022")
    max_tokens = Column(Integer, default=4096)
    temperature = Column(Float, default=0.7)

    # 行为配置
    max_iterations = Column(Integer, default=20)
    require_done_tool = Column(Boolean, default=False)

    # Prompt
    system_prompt = Column(Text)

    # 工具列表
    tools = Column(JSONB, default=[])

    # 上下文配置
    compaction_config = Column(JSONB, default={})

    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 5.2 配置 CRUD API

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/tenants", tags=["tenant-config"])


class AgentConfigCreate(BaseModel):
    name: str
    description: str = ""
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    temperature: float = Field(0.7, ge=0, le=2)
    max_iterations: int = Field(20, ge=1, le=100)
    system_prompt: str = ""
    tools: list[str] = []


class AgentConfigUpdate(BaseModel):
    name: str = None
    llm_model: str = None
    max_tokens: int = None
    temperature: float = None
    system_prompt: str = None
    tools: list[str] = None


class TenantConfigService:
    """租户配置服务"""

    def __init__(self, db: AsyncSession, cache: aioredis.Redis):
        self.db = db
        self.cache = cache

    # ==================== Agent 配置 CRUD ====================
    async def create_agent_config(
        self,
        tenant_id: str,
        config: AgentConfigCreate,
    ) -> AgentConfigModel:
        """创建 Agent 配置"""
        agent_config = AgentConfigModel(
            id=f"ac_{secrets.token_urlsafe(8)}",
            tenant_id=tenant_id,
            **config.dict(),
        )
        self.db.add(agent_config)
        await self.db.commit()
        await self._invalidate_cache(tenant_id)
        return agent_config

    async def get_agent_config(
        self,
        tenant_id: str,
        config_id: str = None,
    ) -> Optional[AgentConfigModel]:
        """获取 Agent 配置"""
        cache_key = f"agent_config:{tenant_id}:{config_id or 'default'}"

        # 查缓存
        cached = await self.cache.get(cache_key)
        if cached:
            return AgentConfigModel.parse_raw(cached)

        # 查数据库
        if config_id:
            config = await self.db.get(AgentConfigModel, config_id)
        else:
            config = await self.db.execute(
                select(AgentConfigModel)
                .where(AgentConfigModel.tenant_id == tenant_id)
                .where(AgentConfigModel.is_default == True)
            )
            config = config.scalar_one_or_none()

        if config:
            await self.cache.setex(cache_key, 300, config.json())

        return config

    async def update_agent_config(
        self,
        tenant_id: str,
        config_id: str,
        updates: AgentConfigUpdate,
    ) -> AgentConfigModel:
        """更新 Agent 配置"""
        config = await self.db.get(AgentConfigModel, config_id)
        if not config or config.tenant_id != tenant_id:
            raise HTTPException(404, "Config not found")

        for key, value in updates.dict(exclude_unset=True).items():
            setattr(config, key, value)

        await self.db.commit()
        await self._invalidate_cache(tenant_id)
        return config

    async def delete_agent_config(self, tenant_id: str, config_id: str):
        """删除 Agent 配置"""
        config = await self.db.get(AgentConfigModel, config_id)
        if not config or config.tenant_id != tenant_id:
            raise HTTPException(404, "Config not found")

        await self.db.delete(config)
        await self.db.commit()
        await self._invalidate_cache(tenant_id)

    async def list_agent_configs(
        self,
        tenant_id: str,
        skip: int = 0,
        limit: int = 20,
    ) -> list[AgentConfigModel]:
        """列出所有 Agent 配置"""
        result = await self.db.execute(
            select(AgentConfigModel)
            .where(AgentConfigModel.tenant_id == tenant_id)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def _invalidate_cache(self, tenant_id: str):
        """清除缓存"""
        pattern = f"agent_config:{tenant_id}:*"
        keys = await self.cache.keys(pattern)
        if keys:
            await self.cache.delete(*keys)


# API 路由
@router.post("/{tenant_id}/agents")
async def create_agent(
    tenant_id: str,
    config: AgentConfigCreate,
    service: TenantConfigService = Depends(),
    user: User = Depends(require_admin),
):
    return await service.create_agent_config(tenant_id, config)


@router.get("/{tenant_id}/agents")
async def list_agents(
    tenant_id: str,
    service: TenantConfigService = Depends(),
    user: User = Depends(get_current_user),
):
    return await service.list_agent_configs(tenant_id)


@router.get("/{tenant_id}/agents/{config_id}")
async def get_agent(
    tenant_id: str,
    config_id: str,
    service: TenantConfigService = Depends(),
):
    config = await service.get_agent_config(tenant_id, config_id)
    if not config:
        raise HTTPException(404, "Not found")
    return config


@router.patch("/{tenant_id}/agents/{config_id}")
async def update_agent(
    tenant_id: str,
    config_id: str,
    updates: AgentConfigUpdate,
    service: TenantConfigService = Depends(),
):
    return await service.update_agent_config(tenant_id, config_id, updates)


@router.delete("/{tenant_id}/agents/{config_id}")
async def delete_agent(
    tenant_id: str,
    config_id: str,
    service: TenantConfigService = Depends(),
):
    await service.delete_agent_config(tenant_id, config_id)
    return {"status": "deleted"}
```

---

## 6. 知识库系统

### 6.1 向量知识库架构

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
)
import hashlib

class KnowledgeBaseService:
    """知识库服务"""

    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: EmbeddingModel,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.vector_size = 1536  # OpenAI ada-002

    # ==================== 命名空间管理 ====================
    async def create_namespace(
        self,
        tenant_id: str,
        namespace: str,
        description: str = "",
    ):
        """创建知识库命名空间"""
        collection_name = f"{tenant_id}_{namespace}"

        await self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

        # 创建索引
        await self.qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema="keyword",
        )

    async def delete_namespace(self, tenant_id: str, namespace: str):
        """删除命名空间"""
        await self.qdrant.delete_collection(f"{tenant_id}_{namespace}")

    # ==================== 文档管理 ====================
    async def upsert_documents(
        self,
        tenant_id: str,
        namespace: str,
        documents: list[Document],
    ) -> int:
        """添加/更新文档"""
        collection = f"{tenant_id}_{namespace}"
        points = []

        for doc in documents:
            # 分块
            chunks = self._chunk_document(doc)

            for i, chunk in enumerate(chunks):
                # 生成唯一 ID
                chunk_id = hashlib.md5(
                    f"{doc.id}:{i}".encode()
                ).hexdigest()

                # 获取向量
                vector = await self.embedder.embed(chunk.text)

                points.append(PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "doc_id": doc.id,
                        "chunk_index": i,
                        "text": chunk.text,
                        "source": doc.source,
                        "metadata": doc.metadata,
                    },
                ))

        # 批量写入
        await self.qdrant.upsert(
            collection_name=collection,
            points=points,
        )

        return len(points)

    async def delete_document(
        self,
        tenant_id: str,
        namespace: str,
        doc_id: str,
    ):
        """删除文档"""
        await self.qdrant.delete(
            collection_name=f"{tenant_id}_{namespace}",
            points_selector=Filter(
                must=[FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )]
            ),
        )

    # ==================== 检索 ====================
    async def search(
        self,
        tenant_id: str,
        namespace: str,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filters: dict = None,
    ) -> list[SearchResult]:
        """向量检索"""
        query_vector = await self.embedder.embed(query)

        # 构建过滤器
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = await self.qdrant.search(
            collection_name=f"{tenant_id}_{namespace}",
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            SearchResult(
                text=r.payload["text"],
                score=r.score,
                source=r.payload["source"],
                metadata=r.payload["metadata"],
            )
            for r in results
        ]

    def _chunk_document(
        self,
        doc: Document,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[Chunk]:
        """文档分块"""
        text = doc.content
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # 尝试在句子边界切分
            if end < len(text):
                last_period = chunk_text.rfind("。")
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
                    chunk_text = text[start:end]

            chunks.append(Chunk(text=chunk_text, start=start, end=end))
            start = end - overlap

        return chunks


# Agent 工具集成
@tool("Search knowledge base for relevant information")
async def kb_search(
    query: str,
    namespace: str = "default",
    top_k: int = 5,
    kb_service: Annotated[KnowledgeBaseService, Depends(get_kb_service)],
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> str:
    """知识库检索工具"""
    results = await kb_service.search(
        tenant_id=ctx.tenant_id,
        namespace=namespace,
        query=query,
        top_k=top_k,
    )

    if not results:
        return "No relevant information found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] (Score: {r.score:.2f})\n{r.text}\nSource: {r.source}")

    return "\n\n".join(formatted)
```

### 6.2 RAG 增强

```python
class RAGEnhancer:
    """RAG 增强器"""

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        reranker: Optional[RerankerModel] = None,
    ):
        self.kb = kb_service
        self.reranker = reranker

    async def enhance_query(
        self,
        tenant_id: str,
        namespaces: list[str],
        query: str,
        max_context_tokens: int = 2000,
    ) -> str:
        """增强查询，返回上下文"""
        # 1. 多命名空间检索
        all_results = []
        for ns in namespaces:
            results = await self.kb.search(tenant_id, ns, query, top_k=10)
            all_results.extend(results)

        if not all_results:
            return ""

        # 2. 重排序
        if self.reranker:
            all_results = await self.reranker.rerank(query, all_results)

        # 3. 截断到 token 预算
        context_parts = []
        current_tokens = 0

        for result in all_results:
            tokens = len(result.text) // 4  # 粗略估算
            if current_tokens + tokens > max_context_tokens:
                break
            context_parts.append(result.text)
            current_tokens += tokens

        return "\n\n---\n\n".join(context_parts)

    def inject_context(self, agent: Agent, context: str):
        """注入上下文到 Agent"""
        if context:
            agent._inject_hidden_message(f"""
<knowledge-context>
以下是从知识库检索到的相关信息，请在回答时参考：

{context}
</knowledge-context>
""")
```

---

## 7. MCP 工具协议

### 7.1 MCP 服务端实现

```python
from typing import Any
import json

class MCPServer:
    """Model Context Protocol 服务端

    支持将工具暴露给外部 Agent 调用
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._resources: dict[str, Resource] = {}

    def register_tool(self, tool: Tool):
        """注册工具"""
        self._tools[tool.name] = tool

    def register_resource(self, resource: Resource):
        """注册资源"""
        self._resources[resource.uri] = resource

    async def handle_request(self, request: dict) -> dict:
        """处理 MCP 请求"""
        method = request.get("method")

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
        }

        handler = handlers.get(method)
        if not handler:
            return {"error": {"code": -32601, "message": "Method not found"}}

        try:
            result = await handler(request.get("params", {}))
            return {"result": result}
        except Exception as e:
            return {"error": {"code": -32000, "message": str(e)}}

    async def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True, "listChanged": True},
            },
            "serverInfo": {
                "name": "agent-mcp-server",
                "version": "1.0.0",
            },
        }

    async def _handle_list_tools(self, params: dict) -> dict:
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.definition.parameters,
                }
                for tool in self._tools.values()
            ]
        }

    async def _handle_call_tool(self, params: dict) -> dict:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        result = await tool.execute(**arguments)

        return {
            "content": [{"type": "text", "text": str(result)}],
            "isError": False,
        }


# FastAPI 集成
from fastapi import WebSocket

@app.websocket("/mcp")
async def mcp_endpoint(
    websocket: WebSocket,
    tenant_id: str = Depends(get_tenant_from_ws),
):
    """MCP WebSocket 端点"""
    await websocket.accept()

    # 为租户创建 MCP 服务器
    mcp_server = MCPServer()

    # 注册租户的工具
    tools = await tool_registry.get_tools_for_tenant(tenant_id)
    for tool in tools:
        mcp_server.register_tool(tool)

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            response = await mcp_server.handle_request(request)
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        pass
```

### 7.2 MCP 客户端（调用外部 MCP 服务）

```python
import websockets

class MCPClient:
    """MCP 客户端

    调用外部 MCP 服务器的工具
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._ws = None
        self._request_id = 0

    async def connect(self):
        """连接到 MCP 服务器"""
        self._ws = await websockets.connect(self.server_url)

        # 初始化
        await self._send({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent-client", "version": "1.0.0"},
            },
        })

    async def list_tools(self) -> list[dict]:
        """获取可用工具列表"""
        response = await self._send({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
        })
        return response.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用工具"""
        response = await self._send({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })

        result = response.get("result", {})
        if result.get("isError"):
            raise ToolExecutionError(result.get("content", [{}])[0].get("text"))

        return result.get("content", [{}])[0].get("text", "")

    async def _send(self, request: dict) -> dict:
        """发送请求"""
        self._request_id += 1
        request["id"] = self._request_id

        await self._ws.send(json.dumps(request))
        response = await self._ws.recv()
        return json.loads(response)


class MCPToolProxy:
    """MCP 工具代理

    将外部 MCP 工具包装为本地工具
    """

    def __init__(self, client: MCPClient):
        self.client = client

    async def create_tools(self) -> list[Tool]:
        """从 MCP 服务器创建工具"""
        await self.client.connect()
        mcp_tools = await self.client.list_tools()

        tools = []
        for mcp_tool in mcp_tools:
            tool = self._create_tool_wrapper(mcp_tool)
            tools.append(tool)

        return tools

    def _create_tool_wrapper(self, mcp_tool: dict) -> Tool:
        """创建工具包装器"""

        async def tool_func(**kwargs) -> str:
            return await self.client.call_tool(mcp_tool["name"], kwargs)

        # 使用 @tool 装饰器创建工具
        return tool(mcp_tool["description"])(tool_func)
```

---

## 8. 三方 API 集成

### 8.1 统一 API 网关

```python
from abc import ABC, abstractmethod
from typing import Any
import httpx

class ThirdPartyAPIClient(ABC):
    """三方 API 客户端基类"""

    @abstractmethod
    async def call(self, endpoint: str, **kwargs) -> Any:
        pass


class APIGateway:
    """三方 API 统一网关

    功能：
    - 统一认证
    - 请求限流
    - 响应缓存
    - 错误处理
    - 指标收集
    """

    def __init__(self, redis: aioredis.Redis):
        self.redis = redis
        self._clients: dict[str, ThirdPartyAPIClient] = {}
        self._rate_limiters: dict[str, ToolRateLimiter] = {}

    def register(
        self,
        name: str,
        client: ThirdPartyAPIClient,
        rate_limit: int = 100,
    ):
        """注册三方 API"""
        self._clients[name] = client
        self._rate_limiters[name] = ToolRateLimiter(self.redis)

    async def call(
        self,
        api_name: str,
        tenant_id: str,
        endpoint: str,
        cache_ttl: int = 0,
        **kwargs,
    ) -> Any:
        """调用三方 API"""
        client = self._clients.get(api_name)
        if not client:
            raise ValueError(f"Unknown API: {api_name}")

        # 1. 限流检查
        limiter = self._rate_limiters[api_name]
        allowed, _ = await limiter.check_and_increment(
            tenant_id, api_name, 100
        )
        if not allowed:
            raise RateLimitExceeded(f"Rate limit exceeded for {api_name}")

        # 2. 缓存检查
        if cache_ttl > 0:
            cache_key = f"api_cache:{api_name}:{endpoint}:{hash(str(kwargs))}"
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        # 3. 调用 API
        start_time = time.time()
        try:
            result = await client.call(endpoint, **kwargs)

            # 4. 缓存结果
            if cache_ttl > 0:
                await self.redis.setex(cache_key, cache_ttl, json.dumps(result))

            # 5. 记录指标
            API_CALL_LATENCY.labels(api=api_name).observe(time.time() - start_time)
            API_CALL_COUNT.labels(api=api_name, status="success").inc()

            return result

        except Exception as e:
            API_CALL_COUNT.labels(api=api_name, status="error").inc()
            raise


# 具体实现示例
class WeatherAPIClient(ThirdPartyAPIClient):
    """天气 API 客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

    async def call(self, endpoint: str, **kwargs) -> Any:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/{endpoint}",
                params={"appid": self.api_key, **kwargs},
            )
            resp.raise_for_status()
            return resp.json()


class SearchAPIClient(ThirdPartyAPIClient):
    """搜索 API 客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def call(self, endpoint: str, **kwargs) -> Any:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.example.com/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params=kwargs,
            )
            resp.raise_for_status()
            return resp.json()


# 注册
api_gateway = APIGateway(redis)
api_gateway.register("weather", WeatherAPIClient(WEATHER_API_KEY), rate_limit=60)
api_gateway.register("search", SearchAPIClient(SEARCH_API_KEY), rate_limit=100)
```

### 8.2 OAuth 三方集成

```python
class OAuthIntegration:
    """三方 OAuth 集成"""

    PROVIDERS = {
        "google": {
            "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
            "scopes": ["openid", "email", "profile"],
        },
        "github": {
            "authorize_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "userinfo_url": "https://api.github.com/user",
            "scopes": ["user:email"],
        },
        "slack": {
            "authorize_url": "https://slack.com/oauth/v2/authorize",
            "token_url": "https://slack.com/api/oauth.v2.access",
            "scopes": ["users:read", "chat:write"],
        },
    }

    async def get_authorize_url(
        self,
        provider: str,
        client_id: str,
        redirect_uri: str,
        state: str,
    ) -> str:
        """获取授权 URL"""
        config = self.PROVIDERS[provider]
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(config["scopes"]),
            "state": state,
        }
        return f"{config['authorize_url']}?{urlencode(params)}"

    async def exchange_token(
        self,
        provider: str,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> dict:
        """交换 Token"""
        config = self.PROVIDERS[provider]
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                config["token_url"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )
            return resp.json()

    async def get_user_info(
        self,
        provider: str,
        access_token: str,
    ) -> dict:
        """获取用户信息"""
        config = self.PROVIDERS[provider]
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                config["userinfo_url"],
                headers={"Authorization": f"Bearer {access_token}"},
            )
            return resp.json()
```

### 8.3 Webhook 集成

```python
class WebhookManager:
    """Webhook 管理器"""

    def __init__(self, db: AsyncSession, queue: MessageQueue):
        self.db = db
        self.queue = queue

    async def register(
        self,
        tenant_id: str,
        url: str,
        events: list[str],
        secret: str = None,
    ) -> str:
        """注册 Webhook"""
        webhook_id = secrets.token_urlsafe(16)
        secret = secret or secrets.token_urlsafe(32)

        webhook = Webhook(
            id=webhook_id,
            tenant_id=tenant_id,
            url=url,
            events=events,
            secret_hash=pwd_context.hash(secret),
        )
        self.db.add(webhook)
        await self.db.commit()

        return webhook_id

    async def trigger(
        self,
        tenant_id: str,
        event: str,
        payload: dict,
    ):
        """触发 Webhook"""
        webhooks = await self._get_webhooks(tenant_id, event)

        for webhook in webhooks:
            # 异步发送
            await self.queue.enqueue(
                "webhook_delivery",
                {
                    "webhook_id": webhook.id,
                    "url": webhook.url,
                    "payload": payload,
                    "secret": webhook.secret_hash,
                },
            )

    async def deliver(self, job: dict):
        """投递 Webhook"""
        payload = job["payload"]
        signature = self._sign(payload, job["secret"])

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    job["url"],
                    json=payload,
                    headers={
                        "X-Webhook-Signature": signature,
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
            except Exception as e:
                # 重试逻辑
                await self._handle_delivery_failure(job, e)

    def _sign(self, payload: dict, secret: str) -> str:
        """签名 Webhook payload"""
        import hmac
        message = json.dumps(payload, sort_keys=True)
        return hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
```

---

## 9. 并发性设计

### 9.1 Agent 实例池管理

```python
from typing import Dict, Optional
from dataclasses import dataclass, field
from asyncio import Semaphore, Lock
from bu_agent_sdk import Agent
from bu_agent_sdk.llm.anthropic import ChatAnthropic

@dataclass
class AgentPoolConfig:
    """Agent 池配置"""
    max_concurrent_agents: int = 100        # 最大并发 Agent 数
    max_agents_per_tenant: int = 10         # 每租户最大并发数
    agent_idle_timeout: int = 300           # 空闲超时（秒）
    max_queue_size: int = 1000              # 等待队列大小


class AgentPool:
    """Agent 实例池管理器

    职责：
    1. 控制全局并发数
    2. 租户级别限流
    3. Agent 实例复用（可选）
    4. 资源清理
    """

    def __init__(self, config: AgentPoolConfig):
        self.config = config
        self._global_semaphore = Semaphore(config.max_concurrent_agents)
        self._tenant_semaphores: Dict[str, Semaphore] = {}
        self._tenant_lock = Lock()
        self._active_agents: Dict[str, Agent] = {}

    async def _get_tenant_semaphore(self, tenant_id: str) -> Semaphore:
        """获取租户级别的信号量"""
        async with self._tenant_lock:
            if tenant_id not in self._tenant_semaphores:
                self._tenant_semaphores[tenant_id] = Semaphore(
                    self.config.max_agents_per_tenant
                )
            return self._tenant_semaphores[tenant_id]

    async def acquire(self, tenant_id: str, session_id: str) -> Agent:
        """获取 Agent 实例

        双层限流：全局 + 租户级别
        """
        # 1. 全局并发控制
        await self._global_semaphore.acquire()

        try:
            # 2. 租户级别控制
            tenant_sem = await self._get_tenant_semaphore(tenant_id)
            await tenant_sem.acquire()

            # 3. 创建或复用 Agent
            agent = await self._create_agent(tenant_id, session_id)
            self._active_agents[session_id] = agent
            return agent

        except Exception:
            self._global_semaphore.release()
            raise

    async def release(self, session_id: str, tenant_id: str):
        """释放 Agent 实例"""
        if session_id in self._active_agents:
            del self._active_agents[session_id]

        tenant_sem = await self._get_tenant_semaphore(tenant_id)
        tenant_sem.release()
        self._global_semaphore.release()

    async def _create_agent(self, tenant_id: str, session_id: str) -> Agent:
        """创建 Agent 实例 - 按租户配置"""
        # 从配置中心获取租户配置
        tenant_config = await TenantConfigService.get(tenant_id)

        llm = ChatAnthropic(
            model=tenant_config.model,
            api_key=tenant_config.api_key,  # 租户自己的 API Key
            max_tokens=tenant_config.max_tokens,
        )

        tools = await ToolRegistry.get_tools_for_tenant(tenant_id)
        system_prompt = await PromptRegistry.get_system_prompt(tenant_id)

        return Agent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=tenant_config.max_iterations,
        )
```

### 2.2 请求级并发控制

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def agent_context(
    pool: AgentPool,
    tenant_id: str,
    session_id: str,
) -> AsyncGenerator[Agent, None]:
    """Agent 上下文管理器

    确保资源正确获取和释放
    """
    agent = None
    try:
        agent = await pool.acquire(tenant_id, session_id)
        yield agent
    finally:
        if agent:
            await pool.release(session_id, tenant_id)


# FastAPI 集成示例
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI()
agent_pool = AgentPool(AgentPoolConfig())

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/api/v1/chat")
async def chat(
    request: ChatRequest,
    tenant_id: str = Depends(get_tenant_from_token),  # 从 JWT 提取
):
    async with agent_context(agent_pool, tenant_id, request.session_id) as agent:
        # 恢复历史上下文
        history = await SessionStore.get(request.session_id)
        if history:
            agent.load_history(history)

        # 执行查询
        response = await agent.query(request.message)

        # 保存上下文
        await SessionStore.save(request.session_id, agent.messages)

        return {"response": response}
```

### 2.3 流式响应与背压控制

```python
from fastapi.responses import StreamingResponse
from bu_agent_sdk.agent.events import (
    AgentEvent, TextEvent, ToolCallEvent,
    ToolResultEvent, FinalResponseEvent
)

@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    tenant_id: str = Depends(get_tenant_from_token),
):
    async def event_generator():
        async with agent_context(agent_pool, tenant_id, request.session_id) as agent:
            history = await SessionStore.get(request.session_id)
            if history:
                agent.load_history(history)

            async for event in agent.query_stream(request.message):
                # 事件序列化
                event_data = serialize_event(event)
                yield f"data: {event_data}\n\n"

                # 背压控制：检查客户端是否还在消费
                # 这里可以添加速率限制逻辑

            # 保存上下文
            await SessionStore.save(request.session_id, agent.messages)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


def serialize_event(event: AgentEvent) -> str:
    """事件序列化为 SSE 格式"""
    import json

    match event:
        case TextEvent(content=text):
            return json.dumps({"type": "text", "content": text})
        case ToolCallEvent(tool=name, args=args):
            return json.dumps({"type": "tool_call", "tool": name, "args": args})
        case ToolResultEvent(tool=name, result=result):
            return json.dumps({"type": "tool_result", "tool": name, "result": result})
        case FinalResponseEvent(content=text):
            return json.dumps({"type": "final", "content": text})
        case _:
            return json.dumps({"type": "unknown"})
```

### 2.4 异步任务队列（长时任务）

```python
from celery import Celery
from bu_agent_sdk import Agent

celery_app = Celery('agent_tasks', broker='redis://localhost:6379/0')

@celery_app.task(bind=True, max_retries=3)
async def execute_long_running_task(
    self,
    tenant_id: str,
    session_id: str,
    task_description: str,
):
    """异步执行长时任务

    适用场景：
    - 复杂代码生成
    - 大规模数据处理
    - 多步骤工作流
    """
    try:
        agent = await create_agent_for_tenant(tenant_id)

        # 使用 done tool 确保任务完成
        result = await agent.query(task_description)

        # 存储结果
        await TaskResultStore.save(session_id, result)

        # 通知用户（WebSocket/回调）
        await notify_task_complete(session_id, result)

    except Exception as e:
        # 重试逻辑
        self.retry(exc=e, countdown=60)
```

---

## 3. 上下文管理

### 3.1 上下文生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                     上下文生命周期                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  请求开始                                                        │
│      │                                                          │
│      ▼                                                          │
│  ┌────────────────┐                                             │
│  │ 加载历史上下文  │ ◄─── SessionStore.get()                     │
│  └────────────────┘                                             │
│      │                                                          │
│      ▼                                                          │
│  ┌────────────────┐                                             │
│  │ Agent 执行循环 │                                              │
│  │   ├─ LLM 调用  │                                              │
│  │   ├─ 工具执行  │                                              │
│  │   └─ 消息累积  │                                              │
│  └────────────────┘                                             │
│      │                                                          │
│      ▼                                                          │
│  ┌────────────────┐     ┌─────────────────┐                     │
│  │ 检查上下文大小 │────►│ 触发压缩？       │                     │
│  └────────────────┘     └─────────────────┘                     │
│      │                       │                                   │
│      │                       ▼ (是)                              │
│      │               ┌─────────────────┐                         │
│      │               │ CompactionService│                        │
│      │               │ 生成摘要替换历史 │                         │
│      │               └─────────────────┘                         │
│      ▼                                                          │
│  ┌────────────────┐                                             │
│  │ 清理临时消息   │ ◄─── ephemeral messages 处理                 │
│  └────────────────┘                                             │
│      │                                                          │
│      ▼                                                          │
│  ┌────────────────┐                                             │
│  │ 持久化上下文   │ ◄─── SessionStore.save()                     │
│  └────────────────┘                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 上下文压缩配置

```python
from bu_agent_sdk.agent.compaction import CompactionConfig

# 不同场景的压缩策略
COMPACTION_CONFIGS = {
    # 客服场景：保留更多上下文
    "customer_service": CompactionConfig(
        threshold_ratio=0.85,  # 85% 时触发
        summary_prompt="""
        请总结这段客服对话，重点保留：
        1. 客户的核心问题和诉求
        2. 已提供的解决方案
        3. 客户的反馈和满意度
        4. 待跟进的事项
        """,
    ),

    # 编程助手：注重技术细节
    "coding_assistant": CompactionConfig(
        threshold_ratio=0.80,
        summary_prompt="""
        总结编程会话，保留：
        1. 项目结构和关键文件
        2. 已修改的代码和原因
        3. 遇到的错误和解决方法
        4. 待完成的任务
        """,
    ),

    # 数据分析：保留数据洞察
    "data_analysis": CompactionConfig(
        threshold_ratio=0.75,
        summary_prompt="""
        总结数据分析会话：
        1. 数据源和数据结构
        2. 关键发现和洞察
        3. 已执行的分析步骤
        4. 生成的可视化和报告
        """,
    ),
}
```

### 3.3 Ephemeral 消息管理

```python
from bu_agent_sdk import tool
from bu_agent_sdk.llm.messages import ContentPartImageParam

@tool(
    "获取网页截图",
    ephemeral=2  # 只保留最近 2 次截图，避免上下文膨胀
)
async def screenshot(url: str) -> list:
    """大型输出使用 ephemeral 标记"""
    image_data = await take_screenshot(url)
    return [
        ContentPartImageParam(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": image_data,
            }
        )
    ]


@tool(
    "获取页面 DOM 结构",
    ephemeral=3  # 保留最近 3 次 DOM
)
async def get_dom(url: str) -> str:
    """DOM 结构通常很大"""
    return await fetch_dom_tree(url)
```

### 3.4 上下文隔离策略

```python
from dataclasses import dataclass
from typing import List, Optional
from bu_agent_sdk.llm.messages import BaseMessage

@dataclass
class ContextConfig:
    """上下文配置"""
    max_messages: int = 100          # 最大消息数
    max_tokens: int = 100000         # 最大 token 数
    preserve_system: bool = True     # 始终保留系统消息
    preserve_last_n: int = 10        # 始终保留最后 N 条消息


class ContextManager:
    """上下文管理器

    职责：
    1. 消息数量控制
    2. Token 预算管理
    3. 重要消息保护
    """

    def __init__(self, config: ContextConfig):
        self.config = config

    def trim_context(
        self,
        messages: List[BaseMessage],
        token_count: int,
    ) -> List[BaseMessage]:
        """裁剪上下文到预算内"""
        if token_count <= self.config.max_tokens:
            return messages

        # 1. 分离系统消息
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        # 2. 保留最后 N 条
        protected = other_msgs[-self.config.preserve_last_n:]
        removable = other_msgs[:-self.config.preserve_last_n]

        # 3. 从中间移除消息直到满足预算
        while removable and self._estimate_tokens(
            system_msgs + removable + protected
        ) > self.config.max_tokens:
            removable.pop(0)  # 移除最早的

        return system_msgs + removable + protected

    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """估算 token 数量"""
        # 简化实现，实际应使用 tiktoken
        total_chars = sum(len(str(m.content)) for m in messages)
        return total_chars // 4  # 粗略估算
```

---

## 4. 持久化方案

### 4.1 会话存储设计

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime, timedelta
import json
import pickle

from bu_agent_sdk.llm.messages import BaseMessage


class SessionStore(ABC):
    """会话存储抽象接口"""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[List[BaseMessage]]:
        """获取会话历史"""
        pass

    @abstractmethod
    async def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        """保存会话历史"""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """删除会话"""
        pass

    @abstractmethod
    async def list_sessions(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 50,
    ) -> List[dict]:
        """列出用户的会话"""
        pass


class RedisSessionStore(SessionStore):
    """Redis 会话存储

    优点：
    - 高性能读写
    - 自动过期
    - 支持集群

    适用场景：短期会话、高并发
    """

    def __init__(self, redis_client, ttl_hours: int = 24):
        self.redis = redis_client
        self.ttl = timedelta(hours=ttl_hours)

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def get(self, session_id: str) -> Optional[List[BaseMessage]]:
        data = await self.redis.get(self._key(session_id))
        if data:
            return self._deserialize(data)
        return None

    async def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        data = self._serialize(messages)
        await self.redis.setex(
            self._key(session_id),
            self.ttl,
            data,
        )

    def _serialize(self, messages: List[BaseMessage]) -> bytes:
        """序列化消息列表"""
        # 使用 pickle 保留完整类型信息
        return pickle.dumps(messages)

    def _deserialize(self, data: bytes) -> List[BaseMessage]:
        """反序列化消息列表"""
        return pickle.loads(data)


class PostgresSessionStore(SessionStore):
    """PostgreSQL 会话存储

    优点：
    - 持久化可靠
    - 支持复杂查询
    - 事务保证

    适用场景：长期存档、审计需求
    """

    async def get(self, session_id: str) -> Optional[List[BaseMessage]]:
        query = """
            SELECT messages FROM sessions
            WHERE session_id = $1 AND deleted_at IS NULL
        """
        row = await self.pool.fetchrow(query, session_id)
        if row:
            return self._deserialize(row['messages'])
        return None

    async def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        query = """
            INSERT INTO sessions (session_id, messages, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (session_id)
            DO UPDATE SET messages = $2, updated_at = NOW()
        """
        await self.pool.execute(
            query,
            session_id,
            self._serialize(messages),
        )

    async def list_sessions(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 50,
    ) -> List[dict]:
        query = """
            SELECT session_id, title, created_at, updated_at
            FROM sessions
            WHERE tenant_id = $1 AND user_id = $2 AND deleted_at IS NULL
            ORDER BY updated_at DESC
            LIMIT $3
        """
        rows = await self.pool.fetch(query, tenant_id, user_id, limit)
        return [dict(row) for row in rows]
```

### 4.2 分层存储策略

```python
class TieredSessionStore(SessionStore):
    """分层存储：Redis（热） + PostgreSQL（冷）

    策略：
    - 活跃会话保存在 Redis
    - 超过 TTL 的会话归档到 PostgreSQL
    - 查询时先查 Redis，未命中再查 PostgreSQL
    """

    def __init__(
        self,
        redis_store: RedisSessionStore,
        postgres_store: PostgresSessionStore,
        hot_ttl_hours: int = 2,
    ):
        self.redis = redis_store
        self.postgres = postgres_store
        self.hot_ttl = timedelta(hours=hot_ttl_hours)

    async def get(self, session_id: str) -> Optional[List[BaseMessage]]:
        # 1. 尝试从 Redis 获取（热数据）
        messages = await self.redis.get(session_id)
        if messages:
            return messages

        # 2. 从 PostgreSQL 获取（冷数据）
        messages = await self.postgres.get(session_id)
        if messages:
            # 预热到 Redis
            await self.redis.save(session_id, messages)

        return messages

    async def save(self, session_id: str, messages: List[BaseMessage]) -> None:
        # 同时写入两层
        await asyncio.gather(
            self.redis.save(session_id, messages),
            self.postgres.save(session_id, messages),
        )
```

### 4.3 消息序列化

```python
from bu_agent_sdk.llm.messages import (
    BaseMessage, UserMessage, AssistantMessage,
    ToolMessage, SystemMessage
)
from pydantic import TypeAdapter
import json

class MessageSerializer:
    """消息序列化器

    支持完整的消息类型还原
    """

    # 使用 Pydantic TypeAdapter 处理 Union 类型
    _adapter = TypeAdapter(list[BaseMessage])

    @classmethod
    def serialize(cls, messages: list[BaseMessage]) -> str:
        """序列化为 JSON"""
        return cls._adapter.dump_json(messages).decode()

    @classmethod
    def deserialize(cls, data: str) -> list[BaseMessage]:
        """从 JSON 反序列化"""
        return cls._adapter.validate_json(data)


# 使用示例
messages = [
    SystemMessage(content="You are a helpful assistant"),
    UserMessage(content="Hello"),
    AssistantMessage(
        content="Hi there!",
        tool_calls=[...],
    ),
]

# 序列化
json_str = MessageSerializer.serialize(messages)

# 反序列化
restored = MessageSerializer.deserialize(json_str)
```

### 4.4 数据库 Schema

```sql
-- 会话表
CREATE TABLE sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    title VARCHAR(255),
    messages JSONB NOT NULL DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    -- 索引
    INDEX idx_sessions_tenant_user (tenant_id, user_id),
    INDEX idx_sessions_updated (updated_at DESC)
);

-- 工具调用日志（审计）
CREATE TABLE tool_call_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(session_id),
    tool_name VARCHAR(128) NOT NULL,
    arguments JSONB NOT NULL,
    result JSONB,
    is_error BOOLEAN DEFAULT FALSE,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_tool_logs_session (session_id),
    INDEX idx_tool_logs_tool (tool_name)
);

-- Token 使用统计
CREATE TABLE token_usage (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    session_id UUID NOT NULL,
    model VARCHAR(64) NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    INDEX idx_usage_tenant (tenant_id, created_at)
);
```

---

## 5. 工具系统设计

### 5.1 工具注册表

```python
from typing import Dict, List, Set, Callable
from dataclasses import dataclass, field
from bu_agent_sdk.tools import Tool, tool

@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    category: str
    description: str
    version: str = "1.0.0"
    requires_permission: bool = False
    rate_limit: int = 100  # 每分钟调用次数
    timeout: int = 30      # 超时秒数


class ToolRegistry:
    """工具注册表

    职责：
    1. 工具注册和发现
    2. 租户级别工具配置
    3. 权限控制
    4. 版本管理
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, Tool] = {}
            cls._instance._metadata: Dict[str, ToolMetadata] = {}
            cls._instance._tenant_tools: Dict[str, Set[str]] = {}
        return cls._instance

    def register(
        self,
        tool_instance: Tool,
        metadata: ToolMetadata,
    ) -> None:
        """注册工具"""
        self._tools[metadata.name] = tool_instance
        self._metadata[metadata.name] = metadata

    def enable_for_tenant(
        self,
        tenant_id: str,
        tool_names: List[str],
    ) -> None:
        """为租户启用工具"""
        self._tenant_tools[tenant_id] = set(tool_names)

    async def get_tools_for_tenant(
        self,
        tenant_id: str,
    ) -> List[Tool]:
        """获取租户可用的工具列表"""
        enabled = self._tenant_tools.get(tenant_id, set())
        return [
            self._tools[name]
            for name in enabled
            if name in self._tools
        ]

    def get_metadata(self, tool_name: str) -> ToolMetadata:
        """获取工具元数据"""
        return self._metadata.get(tool_name)


# 全局注册表实例
registry = ToolRegistry()
```

### 5.2 工具分类与组织

```python
# tools/search.py - 搜索类工具
from bu_agent_sdk import tool
from bu_agent_sdk.tools import Depends
from typing import Annotated

@tool("Search the web for information")
async def web_search(
    query: str,
    max_results: int = 10,
    search_client: Annotated[SearchClient, Depends(get_search_client)],
) -> str:
    results = await search_client.search(query, max_results)
    return format_search_results(results)


@tool("Search internal knowledge base")
async def kb_search(
    query: str,
    namespace: str,
    kb_client: Annotated[KBClient, Depends(get_kb_client)],
) -> str:
    results = await kb_client.search(query, namespace)
    return format_kb_results(results)


# tools/database.py - 数据库类工具
@tool("Execute read-only SQL query")
async def sql_query(
    query: str,
    database: str,
    db_pool: Annotated[Pool, Depends(get_db_pool)],
    user_context: Annotated[UserContext, Depends(get_user_context)],
) -> str:
    # 权限检查
    if not user_context.can_access_database(database):
        return "Error: Access denied to this database"

    # 只读检查
    if not is_read_only_query(query):
        return "Error: Only SELECT queries are allowed"

    results = await db_pool.fetch(query)
    return format_query_results(results)


# tools/file.py - 文件类工具
@tool("Read file content", ephemeral=5)
async def read_file(
    path: str,
    sandbox: Annotated[Sandbox, Depends(get_sandbox)],
) -> str:
    # 路径安全检查
    safe_path = sandbox.resolve_path(path)
    content = await sandbox.read_file(safe_path)
    return add_line_numbers(content)
```

### 5.3 工具权限控制

```python
from functools import wraps
from typing import Callable, Any

class ToolPermission:
    """工具权限定义"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


def requires_permission(*permissions: str):
    """权限检查装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从依赖注入获取用户上下文
            user_context = kwargs.get('user_context')
            if not user_context:
                raise PermissionError("User context not found")

            # 检查权限
            for perm in permissions:
                if not user_context.has_permission(perm):
                    raise PermissionError(f"Missing permission: {perm}")

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# 使用示例
@tool("Delete file")
@requires_permission(ToolPermission.WRITE)
async def delete_file(
    path: str,
    user_context: Annotated[UserContext, Depends(get_user_context)],
    sandbox: Annotated[Sandbox, Depends(get_sandbox)],
) -> str:
    safe_path = sandbox.resolve_path(path)
    await sandbox.delete_file(safe_path)
    return f"Deleted: {path}"
```

### 5.4 工具速率限制

```python
from datetime import datetime
from typing import Dict, Tuple
import asyncio

class ToolRateLimiter:
    """工具调用速率限制器"""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_and_increment(
        self,
        tenant_id: str,
        tool_name: str,
        limit: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int]:
        """检查并增加计数

        Returns:
            (allowed, remaining)
        """
        key = f"rate_limit:{tenant_id}:{tool_name}"

        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)

        count, ttl = await pipe.execute()

        # 首次调用，设置过期时间
        if ttl == -1:
            await self.redis.expire(key, window_seconds)

        allowed = count <= limit
        remaining = max(0, limit - count)

        return allowed, remaining


class RateLimitedToolWrapper:
    """带速率限制的工具包装器"""

    def __init__(
        self,
        tool: Tool,
        limiter: ToolRateLimiter,
        limit: int,
    ):
        self.tool = tool
        self.limiter = limiter
        self.limit = limit

    async def execute(
        self,
        tenant_id: str,
        **kwargs,
    ) -> Any:
        allowed, remaining = await self.limiter.check_and_increment(
            tenant_id,
            self.tool.name,
            self.limit,
        )

        if not allowed:
            raise RateLimitExceeded(
                f"Rate limit exceeded for {self.tool.name}. "
                f"Try again in 60 seconds."
            )

        return await self.tool.execute(**kwargs)
```

---

## 6. Prompt 管理

### 6.1 Prompt 模板系统

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
from jinja2 import Environment, BaseLoader, select_autoescape
from pydantic import BaseModel

@dataclass
class PromptTemplate:
    """Prompt 模板"""
    name: str
    version: str
    template: str
    variables: list[str]
    description: str = ""


class PromptRegistry:
    """Prompt 注册表

    职责：
    1. 模板管理和版本控制
    2. 租户级别定制
    3. A/B 测试支持
    """

    def __init__(self):
        self._templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self._tenant_overrides: Dict[str, Dict[str, str]] = {}
        self._jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml']),
        )

    def register(self, template: PromptTemplate) -> None:
        """注册模板"""
        if template.name not in self._templates:
            self._templates[template.name] = {}
        self._templates[template.name][template.version] = template

    def set_tenant_override(
        self,
        tenant_id: str,
        template_name: str,
        custom_template: str,
    ) -> None:
        """设置租户定制模板"""
        if tenant_id not in self._tenant_overrides:
            self._tenant_overrides[tenant_id] = {}
        self._tenant_overrides[tenant_id][template_name] = custom_template

    def render(
        self,
        template_name: str,
        tenant_id: str,
        variables: Dict[str, Any],
        version: str = "latest",
    ) -> str:
        """渲染模板"""
        # 1. 检查租户定制
        if tenant_id in self._tenant_overrides:
            if template_name in self._tenant_overrides[tenant_id]:
                template_str = self._tenant_overrides[tenant_id][template_name]
                return self._render(template_str, variables)

        # 2. 使用默认模板
        versions = self._templates.get(template_name, {})
        if version == "latest":
            version = max(versions.keys()) if versions else None

        if version and version in versions:
            return self._render(versions[version].template, variables)

        raise ValueError(f"Template not found: {template_name}@{version}")

    def _render(self, template_str: str, variables: Dict[str, Any]) -> str:
        """执行模板渲染"""
        template = self._jinja_env.from_string(template_str)
        return template.render(**variables)


# 全局实例
prompt_registry = PromptRegistry()
```

### 6.2 System Prompt 设计

```python
# prompts/base.py

BASE_SYSTEM_PROMPT = PromptTemplate(
    name="base_system",
    version="1.0.0",
    template="""
You are {{ agent_name }}, an AI assistant for {{ company_name }}.

## Core Capabilities
{{ capabilities }}

## Guidelines
- Always be helpful, accurate, and concise
- If unsure, ask clarifying questions
- Use tools when appropriate to complete tasks
- Respect user privacy and data security

## Available Tools
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}

## Current Context
- User: {{ user_name }}
- Timezone: {{ timezone }}
- Language: {{ language }}
{% if additional_context %}
- {{ additional_context }}
{% endif %}
""",
    variables=[
        "agent_name", "company_name", "capabilities",
        "tools", "user_name", "timezone", "language",
        "additional_context"
    ],
)

# 客服场景
CUSTOMER_SERVICE_PROMPT = PromptTemplate(
    name="customer_service",
    version="1.0.0",
    template="""
You are a customer service representative for {{ company_name }}.

## Your Role
- Help customers resolve issues quickly and professionally
- Escalate complex issues to human agents when necessary
- Collect relevant information before escalating

## Tone
- Professional but friendly
- Empathetic to customer frustrations
- Solution-oriented

## Product Knowledge
{{ product_info }}

## Common Issues and Solutions
{% for issue in common_issues %}
### {{ issue.title }}
{{ issue.solution }}
{% endfor %}

## Escalation Criteria
Escalate to human agent when:
- Customer explicitly requests human assistance
- Issue cannot be resolved with available tools
- Customer expresses strong dissatisfaction
""",
    variables=[
        "company_name", "product_info", "common_issues"
    ],
)

# 编程助手
CODING_ASSISTANT_PROMPT = PromptTemplate(
    name="coding_assistant",
    version="1.0.0",
    template="""
You are an expert software engineer assistant.

## Expertise
- Languages: {{ languages | join(', ') }}
- Frameworks: {{ frameworks | join(', ') }}
- Best Practices: Clean code, testing, documentation

## Working Directory
{{ working_directory }}

## Project Context
{{ project_context }}

## Guidelines
1. Read existing code before making changes
2. Follow the project's coding style
3. Write tests for new functionality
4. Keep changes minimal and focused
5. Explain your reasoning

## Safety
- Never execute destructive commands without confirmation
- Always backup before major changes
- Validate file paths to prevent escaping sandbox
""",
    variables=[
        "languages", "frameworks", "working_directory", "project_context"
    ],
)

# 注册模板
prompt_registry.register(BASE_SYSTEM_PROMPT)
prompt_registry.register(CUSTOMER_SERVICE_PROMPT)
prompt_registry.register(CODING_ASSISTANT_PROMPT)
```

### 6.3 动态 Prompt 注入

```python
from bu_agent_sdk.llm.messages import SystemMessage, DeveloperMessage

class DynamicPromptInjector:
    """动态 Prompt 注入器

    在运行时向对话注入上下文信息
    """

    @staticmethod
    def inject_context(
        agent: Agent,
        context_type: str,
        context_data: dict,
    ) -> None:
        """注入上下文消息"""

        if context_type == "todo_reminder":
            # TODO 列表提醒
            if context_data.get("pending_todos"):
                todos = context_data["pending_todos"]
                reminder = f"""
<system-reminder>
You have {len(todos)} pending tasks:
{chr(10).join(f'- {t}' for t in todos)}
Please address these before ending the conversation.
</system-reminder>
"""
                agent._inject_hidden_message(reminder)

        elif context_type == "rate_limit_warning":
            # 速率限制警告
            warning = f"""
<system-reminder>
Warning: You are approaching rate limits for some tools.
Remaining calls: {context_data['remaining']}
Please use tools judiciously.
</system-reminder>
"""
            agent._inject_hidden_message(warning)

        elif context_type == "user_preference":
            # 用户偏好
            prefs = context_data
            pref_msg = f"""
<user-preferences>
- Communication style: {prefs.get('style', 'professional')}
- Response length: {prefs.get('length', 'moderate')}
- Technical level: {prefs.get('level', 'intermediate')}
</user-preferences>
"""
            agent._inject_hidden_message(pref_msg)
```

### 6.4 Prompt 版本控制与 A/B 测试

```python
import random
from typing import Tuple

class PromptExperiment:
    """Prompt A/B 测试"""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def get_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Tuple[str, str]:
        """获取用户的实验变体

        Returns:
            (variant_id, template_version)
        """
        # 检查用户是否已分配变体
        key = f"experiment:{experiment_id}:user:{user_id}"
        variant = await self.redis.get(key)

        if variant:
            return variant.split(":")

        # 分配新变体
        config = await self.get_experiment_config(experiment_id)
        variant_id = self._select_variant(config["variants"])
        template_version = config["variants"][variant_id]

        # 持久化分配
        await self.redis.setex(
            key,
            timedelta(days=30),
            f"{variant_id}:{template_version}",
        )

        return variant_id, template_version

    def _select_variant(self, variants: dict) -> str:
        """根据权重选择变体"""
        total = sum(v.get("weight", 1) for v in variants.values())
        r = random.uniform(0, total)

        cumulative = 0
        for variant_id, config in variants.items():
            cumulative += config.get("weight", 1)
            if r <= cumulative:
                return variant_id

        return list(variants.keys())[0]

    async def record_outcome(
        self,
        experiment_id: str,
        user_id: str,
        variant_id: str,
        outcome: dict,
    ) -> None:
        """记录实验结果"""
        await self.redis.lpush(
            f"experiment:{experiment_id}:outcomes:{variant_id}",
            json.dumps({
                "user_id": user_id,
                "outcome": outcome,
                "timestamp": datetime.utcnow().isoformat(),
            }),
        )
```

---

## 7. 多租户隔离

### 7.1 租户配置模型

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

class TenantTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class TenantLimits(BaseModel):
    """租户配额限制"""
    max_sessions_per_day: int = 100
    max_messages_per_session: int = 50
    max_tokens_per_month: int = 1_000_000
    max_concurrent_sessions: int = 5
    max_tool_calls_per_minute: int = 60
    allowed_models: List[str] = ["claude-3-haiku"]


class TenantConfig(BaseModel):
    """租户配置"""
    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.FREE

    # API 配置
    api_key: Optional[str] = None  # 租户自己的 API Key
    use_platform_key: bool = True  # 使用平台统一 Key

    # 模型配置
    default_model: str = "claude-3-haiku-20240307"
    max_tokens: int = 4096
    temperature: float = 0.7

    # 功能开关
    enabled_tools: List[str] = []
    custom_system_prompt: Optional[str] = None

    # 限制
    limits: TenantLimits = Field(default_factory=TenantLimits)

    # 元数据
    metadata: Dict[str, str] = {}


# 不同层级的默认配置
TIER_DEFAULTS = {
    TenantTier.FREE: TenantLimits(
        max_sessions_per_day=50,
        max_messages_per_session=20,
        max_tokens_per_month=100_000,
        max_concurrent_sessions=2,
        allowed_models=["claude-3-haiku-20240307"],
    ),
    TenantTier.STARTER: TenantLimits(
        max_sessions_per_day=500,
        max_messages_per_session=50,
        max_tokens_per_month=1_000_000,
        max_concurrent_sessions=10,
        allowed_models=["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
    ),
    TenantTier.PRO: TenantLimits(
        max_sessions_per_day=5000,
        max_messages_per_session=100,
        max_tokens_per_month=10_000_000,
        max_concurrent_sessions=50,
        allowed_models=["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
    ),
    TenantTier.ENTERPRISE: TenantLimits(
        max_sessions_per_day=-1,  # 无限制
        max_messages_per_session=-1,
        max_tokens_per_month=-1,
        max_concurrent_sessions=-1,
        allowed_models=["*"],  # 所有模型
    ),
}
```

### 7.2 租户上下文注入

```python
from dataclasses import dataclass
from contextvars import ContextVar
from typing import Optional

# 请求级别的租户上下文
_tenant_context: ContextVar[Optional["TenantContext"]] = ContextVar(
    "tenant_context",
    default=None,
)

@dataclass
class TenantContext:
    """租户上下文"""
    tenant_id: str
    user_id: str
    session_id: str
    config: TenantConfig
    permissions: set[str]

    @classmethod
    def current(cls) -> Optional["TenantContext"]:
        return _tenant_context.get()

    @classmethod
    def set(cls, context: "TenantContext") -> None:
        _tenant_context.set(context)


# 依赖注入函数
def get_tenant_context() -> TenantContext:
    """获取当前租户上下文"""
    ctx = TenantContext.current()
    if not ctx:
        raise RuntimeError("Tenant context not set")
    return ctx


# 工具中使用租户上下文
@tool("Query tenant database")
async def query_database(
    query: str,
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> str:
    # 自动使用租户的数据库
    db = get_tenant_database(ctx.tenant_id)
    return await db.execute(query)
```

### 7.3 资源隔离

```python
class TenantResourceManager:
    """租户资源管理器

    确保租户间资源完全隔离
    """

    def __init__(self, config: TenantConfig):
        self.config = config
        self.tenant_id = config.tenant_id

    def get_redis_prefix(self) -> str:
        """Redis Key 前缀"""
        return f"tenant:{self.tenant_id}:"

    def get_db_schema(self) -> str:
        """数据库 Schema"""
        return f"tenant_{self.tenant_id}"

    def get_storage_bucket(self) -> str:
        """对象存储 Bucket"""
        return f"agent-files-{self.tenant_id}"

    def get_log_namespace(self) -> str:
        """日志命名空间"""
        return f"tenant.{self.tenant_id}"

    async def check_quota(self, resource: str, amount: int = 1) -> bool:
        """检查配额"""
        limits = self.config.limits

        if resource == "session":
            current = await self._get_daily_sessions()
            return limits.max_sessions_per_day < 0 or current + amount <= limits.max_sessions_per_day

        elif resource == "tokens":
            current = await self._get_monthly_tokens()
            return limits.max_tokens_per_month < 0 or current + amount <= limits.max_tokens_per_month

        return True

    async def consume_quota(self, resource: str, amount: int) -> None:
        """消费配额"""
        key = f"{self.get_redis_prefix()}quota:{resource}"
        await redis.incrby(key, amount)


# 中间件示例
async def tenant_middleware(request: Request, call_next):
    """租户中间件"""
    # 从请求中提取租户信息
    tenant_id = request.headers.get("X-Tenant-ID")
    if not tenant_id:
        raise HTTPException(401, "Missing tenant ID")

    # 加载租户配置
    config = await TenantConfigService.get(tenant_id)
    if not config:
        raise HTTPException(404, "Tenant not found")

    # 检查配额
    resource_mgr = TenantResourceManager(config)
    if not await resource_mgr.check_quota("session"):
        raise HTTPException(429, "Daily session quota exceeded")

    # 设置上下文
    ctx = TenantContext(
        tenant_id=tenant_id,
        user_id=request.user.id,
        session_id=request.headers.get("X-Session-ID"),
        config=config,
        permissions=await get_user_permissions(request.user),
    )
    TenantContext.set(ctx)

    response = await call_next(request)
    return response
```

---

## 8. 可扩展性设计

### 8.1 插件系统

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type

class AgentPlugin(ABC):
    """Agent 插件基类"""

    name: str
    version: str

    @abstractmethod
    async def on_load(self, agent: Agent) -> None:
        """插件加载时调用"""
        pass

    @abstractmethod
    async def on_unload(self, agent: Agent) -> None:
        """插件卸载时调用"""
        pass

    def get_tools(self) -> List[Tool]:
        """返回插件提供的工具"""
        return []

    def get_system_prompt_additions(self) -> str:
        """返回需要追加到系统提示的内容"""
        return ""


class PluginManager:
    """插件管理器"""

    def __init__(self):
        self._plugins: Dict[str, AgentPlugin] = {}
        self._loaded: Dict[str, bool] = {}

    def register(self, plugin_class: Type[AgentPlugin]) -> None:
        """注册插件"""
        plugin = plugin_class()
        self._plugins[plugin.name] = plugin

    async def load_plugin(
        self,
        plugin_name: str,
        agent: Agent,
    ) -> None:
        """加载插件到 Agent"""
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")

        plugin = self._plugins[plugin_name]

        # 添加工具
        for tool in plugin.get_tools():
            agent.add_tool(tool)

        # 追加系统提示
        additions = plugin.get_system_prompt_additions()
        if additions:
            agent.system_prompt += f"\n\n{additions}"

        # 调用加载钩子
        await plugin.on_load(agent)
        self._loaded[plugin_name] = True


# 示例插件
class WebBrowsingPlugin(AgentPlugin):
    name = "web_browsing"
    version = "1.0.0"

    async def on_load(self, agent: Agent) -> None:
        self.browser = await launch_browser()

    async def on_unload(self, agent: Agent) -> None:
        await self.browser.close()

    def get_tools(self) -> List[Tool]:
        return [
            self._create_navigate_tool(),
            self._create_click_tool(),
            self._create_screenshot_tool(),
        ]

    def get_system_prompt_additions(self) -> str:
        return """
## Web Browsing Capabilities
You can browse the web using the following tools:
- navigate: Open a URL in the browser
- click: Click on an element
- screenshot: Take a screenshot of the current page
"""
```

### 8.2 业务适配器

```python
from abc import ABC, abstractmethod

class BusinessAdapter(ABC):
    """业务适配器基类

    不同业务线继承此类实现定制逻辑
    """

    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """获取业务专属工具"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """获取业务专属系统提示"""
        pass

    @abstractmethod
    def pre_process(self, user_input: str) -> str:
        """预处理用户输入"""
        return user_input

    @abstractmethod
    def post_process(self, response: str) -> str:
        """后处理 Agent 响应"""
        return response

    def get_compaction_config(self) -> CompactionConfig:
        """获取上下文压缩配置"""
        return CompactionConfig()


class ECommerceAdapter(BusinessAdapter):
    """电商业务适配器"""

    def get_tools(self) -> List[Tool]:
        return [
            search_products,
            get_product_details,
            check_inventory,
            create_order,
            track_order,
        ]

    def get_system_prompt(self) -> str:
        return """
You are an e-commerce shopping assistant.
Help customers find products, answer questions, and complete purchases.
Always verify inventory before promising availability.
"""

    def pre_process(self, user_input: str) -> str:
        # 提取商品关键词，优化搜索
        return user_input

    def post_process(self, response: str) -> str:
        # 添加商品链接、图片等
        return response


class FinanceAdapter(BusinessAdapter):
    """金融业务适配器"""

    def get_tools(self) -> List[Tool]:
        return [
            check_balance,
            transfer_funds,
            get_transaction_history,
            analyze_spending,
        ]

    def get_system_prompt(self) -> str:
        return """
You are a financial assistant.
Help users manage their finances, track spending, and make transfers.
IMPORTANT: Always verify identity before sensitive operations.
Never display full account numbers or sensitive information.
"""


# 适配器工厂
class AdapterFactory:
    _adapters: Dict[str, Type[BusinessAdapter]] = {
        "ecommerce": ECommerceAdapter,
        "finance": FinanceAdapter,
    }

    @classmethod
    def create(cls, business_type: str) -> BusinessAdapter:
        adapter_class = cls._adapters.get(business_type)
        if not adapter_class:
            raise ValueError(f"Unknown business type: {business_type}")
        return adapter_class()
```

### 8.3 模型路由

```python
from bu_agent_sdk.llm.anthropic import ChatAnthropic
from bu_agent_sdk.llm.openai import ChatOpenAI

class ModelRouter:
    """模型路由器

    根据场景选择最优模型
    """

    def __init__(self):
        self._models = {}
        self._routing_rules = []

    def register_model(
        self,
        name: str,
        provider: str,
        config: dict,
    ) -> None:
        """注册模型"""
        if provider == "anthropic":
            self._models[name] = ChatAnthropic(**config)
        elif provider == "openai":
            self._models[name] = ChatOpenAI(**config)

    def add_routing_rule(
        self,
        condition: Callable[[dict], bool],
        model_name: str,
        priority: int = 0,
    ) -> None:
        """添加路由规则"""
        self._routing_rules.append({
            "condition": condition,
            "model": model_name,
            "priority": priority,
        })
        self._routing_rules.sort(key=lambda x: -x["priority"])

    def select_model(self, context: dict) -> BaseChatModel:
        """根据上下文选择模型"""
        for rule in self._routing_rules:
            if rule["condition"](context):
                return self._models[rule["model"]]

        # 默认模型
        return self._models.get("default")


# 使用示例
router = ModelRouter()

# 注册模型
router.register_model("haiku", "anthropic", {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 4096,
})
router.register_model("sonnet", "anthropic", {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 8192,
})
router.register_model("opus", "anthropic", {
    "model": "claude-3-opus-20240229",
    "max_tokens": 4096,
})

# 添加路由规则
router.add_routing_rule(
    condition=lambda ctx: ctx.get("task_complexity") == "high",
    model_name="opus",
    priority=10,
)
router.add_routing_rule(
    condition=lambda ctx: ctx.get("needs_code_generation"),
    model_name="sonnet",
    priority=5,
)
router.add_routing_rule(
    condition=lambda ctx: True,  # 默认
    model_name="haiku",
    priority=0,
)
```

---

## 9. 可维护性保障

### 9.1 错误处理与恢复

```python
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    LOW = "low"           # 可继续，记录日志
    MEDIUM = "medium"     # 需重试
    HIGH = "high"         # 需人工介入
    CRITICAL = "critical" # 服务降级


@dataclass
class AgentError:
    code: str
    message: str
    severity: ErrorSeverity
    recoverable: bool
    context: dict


class ErrorHandler:
    """错误处理器"""

    async def handle(
        self,
        error: Exception,
        agent: Agent,
        context: dict,
    ) -> Optional[str]:
        """处理错误，返回恢复消息或 None"""

        if isinstance(error, RateLimitError):
            # 速率限制：等待后重试
            await asyncio.sleep(error.retry_after)
            return "Rate limit hit, retrying..."

        elif isinstance(error, ToolExecutionError):
            # 工具执行失败：返回错误给模型
            return f"Tool error: {error.message}. Please try a different approach."

        elif isinstance(error, ContextOverflowError):
            # 上下文溢出：触发压缩
            await agent._force_compaction()
            return "Context compressed, continuing..."

        elif isinstance(error, ModelRefusalError):
            # 模型拒绝：记录并通知
            await self._log_refusal(error, context)
            return "I cannot help with that request."

        else:
            # 未知错误：记录并降级
            await self._log_error(error, context)
            raise error


class RetryPolicy:
    """重试策略"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """计算第 N 次重试的延迟"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """带重试执行函数"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except RetryableError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.get_delay(attempt)
                    await asyncio.sleep(delay)

        raise last_error
```

### 9.2 健康检查

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

class HealthStatus(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    components: Dict[str, dict]
    timestamp: str


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self._checks: Dict[str, Callable] = {}

    def register(self, name: str, check_func: Callable) -> None:
        self._checks[name] = check_func

    async def check_all(self) -> HealthStatus:
        results = {}
        overall_healthy = True

        for name, check_func in self._checks.items():
            try:
                result = await check_func()
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result,
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                overall_healthy = False

        return HealthStatus(
            status="healthy" if overall_healthy else "degraded",
            components=results,
            timestamp=datetime.utcnow().isoformat(),
        )


# 注册检查项
health_checker = HealthChecker()

async def check_redis():
    return await redis.ping()

async def check_database():
    return await db.execute("SELECT 1")

async def check_llm_api():
    # 简单测试调用
    return True

health_checker.register("redis", check_redis)
health_checker.register("database", check_database)
health_checker.register("llm_api", check_llm_api)

# FastAPI 端点
@app.get("/health")
async def health():
    return await health_checker.check_all()
```

### 9.3 配置热更新

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml

class ConfigManager:
    """配置管理器

    支持：
    - 配置热更新
    - 环境变量覆盖
    - 默认值
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config: dict = {}
        self._callbacks: List[Callable] = []
        self._load_config()
        self._start_watcher()

    def _load_config(self) -> None:
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        # 应用环境变量覆盖
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """环境变量覆盖配置"""
        prefix = "AGENT_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._set_nested(config_key, value)

    def _start_watcher(self) -> None:
        """启动配置文件监控"""
        class Handler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager

            def on_modified(self, event):
                if event.src_path == self.manager.config_path:
                    self.manager._reload()

        observer = Observer()
        observer.schedule(Handler(self), os.path.dirname(self.config_path))
        observer.start()

    def _reload(self) -> None:
        """重新加载配置"""
        old_config = self._config.copy()
        self._load_config()

        # 通知回调
        for callback in self._callbacks:
            callback(old_config, self._config)

    def on_change(self, callback: Callable) -> None:
        """注册配置变更回调"""
        self._callbacks.append(callback)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
```

---

## 10. 安全性考量

### 10.1 输入验证

```python
from pydantic import BaseModel, validator, Field
import re

class ChatInput(BaseModel):
    """用户输入验证"""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str = Field(..., regex=r"^[a-zA-Z0-9-]{36}$")

    @validator("message")
    def sanitize_message(cls, v):
        # 移除潜在的注入攻击
        # 注意：这只是基础防护，LLM 本身需要额外的 prompt injection 防护
        v = v.strip()

        # 检测常见的 prompt injection 模式
        injection_patterns = [
            r"ignore previous instructions",
            r"disregard all prior",
            r"system:\s*",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially malicious input detected")

        return v


class ToolInputValidator:
    """工具输入验证器"""

    @staticmethod
    def validate_file_path(path: str, sandbox_root: str) -> str:
        """验证文件路径安全"""
        # 规范化路径
        abs_path = os.path.abspath(os.path.join(sandbox_root, path))

        # 检查是否在沙箱内
        if not abs_path.startswith(os.path.abspath(sandbox_root)):
            raise ValueError("Path escape attempt detected")

        return abs_path

    @staticmethod
    def validate_sql(query: str) -> str:
        """验证 SQL 安全"""
        # 只允许 SELECT
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed")

        # 禁止危险操作
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        for keyword in forbidden:
            if keyword in query.upper():
                raise ValueError(f"Forbidden keyword: {keyword}")

        return query
```

### 10.2 敏感数据处理

```python
import re
from typing import List, Tuple

class SensitiveDataMasker:
    """敏感数据脱敏器"""

    PATTERNS: List[Tuple[str, str]] = [
        # 信用卡号
        (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD_NUMBER]"),
        # 身份证
        (r"\b\d{17}[\dXx]\b", "[ID_CARD]"),
        # 手机号
        (r"\b1[3-9]\d{9}\b", "[PHONE]"),
        # 邮箱
        (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]"),
        # API Key (常见格式)
        (r"\b(sk|pk|api)[_-][\w]{20,}\b", "[API_KEY]"),
    ]

    @classmethod
    def mask(cls, text: str) -> str:
        """脱敏文本"""
        for pattern, replacement in cls.PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text

    @classmethod
    def mask_messages(cls, messages: List[BaseMessage]) -> List[BaseMessage]:
        """脱敏消息列表（用于日志）"""
        masked = []
        for msg in messages:
            masked_content = cls.mask(str(msg.content))
            # 创建副本
            masked_msg = msg.model_copy()
            masked_msg.content = masked_content
            masked.append(masked_msg)
        return masked
```

### 10.3 审计日志

```python
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class AuditEvent:
    timestamp: datetime
    tenant_id: str
    user_id: str
    session_id: str
    event_type: str
    details: dict
    ip_address: str
    user_agent: str


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, storage):
        self.storage = storage

    async def log(self, event: AuditEvent) -> None:
        """记录审计事件"""
        await self.storage.append({
            "timestamp": event.timestamp.isoformat(),
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "event_type": event.event_type,
            "details": SensitiveDataMasker.mask(json.dumps(event.details)),
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
        })

    async def log_tool_call(
        self,
        context: TenantContext,
        tool_name: str,
        arguments: dict,
        result: str,
        duration_ms: int,
    ) -> None:
        """记录工具调用"""
        await self.log(AuditEvent(
            timestamp=datetime.utcnow(),
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            session_id=context.session_id,
            event_type="tool_call",
            details={
                "tool": tool_name,
                "arguments": arguments,
                "result_length": len(result),
                "duration_ms": duration_ms,
            },
            ip_address=context.ip_address,
            user_agent=context.user_agent,
        ))
```

---

## 11. 监控与可观测性

### 11.1 指标收集

```python
from prometheus_client import Counter, Histogram, Gauge

# 请求指标
REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total number of agent requests",
    ["tenant_id", "status"],
)

REQUEST_LATENCY = Histogram(
    "agent_request_duration_seconds",
    "Request duration in seconds",
    ["tenant_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Agent 循环指标
AGENT_ITERATIONS = Histogram(
    "agent_iterations_total",
    "Number of iterations per request",
    ["tenant_id"],
    buckets=[1, 2, 5, 10, 20, 50],
)

# 工具调用指标
TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Total tool calls",
    ["tenant_id", "tool_name", "status"],
)

TOOL_LATENCY = Histogram(
    "agent_tool_duration_seconds",
    "Tool call duration",
    ["tool_name"],
)

# Token 使用指标
TOKEN_USAGE = Counter(
    "agent_tokens_total",
    "Total tokens used",
    ["tenant_id", "model", "type"],  # type: input/output
)

# 活跃会话
ACTIVE_SESSIONS = Gauge(
    "agent_active_sessions",
    "Number of active sessions",
    ["tenant_id"],
)


class MetricsMiddleware:
    """指标收集中间件"""

    async def __call__(self, request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID", "unknown")

        start_time = time.time()
        try:
            response = await call_next(request)
            REQUEST_COUNT.labels(
                tenant_id=tenant_id,
                status="success",
            ).inc()
            return response
        except Exception as e:
            REQUEST_COUNT.labels(
                tenant_id=tenant_id,
                status="error",
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(tenant_id=tenant_id).observe(duration)
```

### 11.2 分布式追踪

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

class TracedAgent:
    """带追踪的 Agent 包装器"""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def query(self, user_input: str) -> str:
        with tracer.start_as_current_span("agent.query") as span:
            span.set_attribute("user_input_length", len(user_input))

            try:
                result = await self.agent.query(user_input)
                span.set_attribute("response_length", len(result))
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    async def _trace_tool_call(
        self,
        tool_name: str,
        args: dict,
    ) -> Any:
        with tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.args", json.dumps(args))

            result = await self.agent._execute_tool(tool_name, args)

            span.set_attribute("tool.result_length", len(str(result)))
            return result
```

### 11.3 日志结构化

```python
import structlog
from typing import Any

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

class AgentLogger:
    """Agent 专用日志器"""

    def __init__(self, tenant_id: str, session_id: str):
        self.log = logger.bind(
            tenant_id=tenant_id,
            session_id=session_id,
        )

    def query_start(self, user_input: str) -> None:
        self.log.info(
            "agent.query.start",
            input_length=len(user_input),
        )

    def query_complete(
        self,
        response: str,
        iterations: int,
        duration_ms: int,
    ) -> None:
        self.log.info(
            "agent.query.complete",
            response_length=len(response),
            iterations=iterations,
            duration_ms=duration_ms,
        )

    def tool_call(
        self,
        tool_name: str,
        args: dict,
        duration_ms: int,
        success: bool,
    ) -> None:
        self.log.info(
            "agent.tool.call",
            tool=tool_name,
            args=SensitiveDataMasker.mask(json.dumps(args)),
            duration_ms=duration_ms,
            success=success,
        )

    def error(self, error: Exception, context: dict) -> None:
        self.log.error(
            "agent.error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            exc_info=True,
        )
```

---

## 12. 参考实现

### 12.1 完整的 SaaS Agent 服务示例

```python
"""
SaaS Agent Service - 完整实现示例

基于 BU Agent SDK 构建的多租户 Agent 服务
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio

from bu_agent_sdk import Agent
from bu_agent_sdk.llm.anthropic import ChatAnthropic
from bu_agent_sdk.agent.compaction import CompactionConfig

# 初始化
app = FastAPI(title="SaaS Agent Service")
agent_pool = AgentPool(AgentPoolConfig())
session_store = TieredSessionStore(
    RedisSessionStore(redis_client),
    PostgresSessionStore(db_pool),
)
prompt_registry = PromptRegistry()
tool_registry = ToolRegistry()

# 请求模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    usage: dict

# 中间件
app.add_middleware(TenantMiddleware)
app.add_middleware(MetricsMiddleware)

# API 端点
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    tenant_ctx: TenantContext = Depends(get_tenant_context),
):
    """同步聊天接口"""
    session_id = request.session_id or str(uuid.uuid4())

    async with agent_context(agent_pool, tenant_ctx.tenant_id, session_id) as agent:
        # 加载历史
        history = await session_store.get(session_id)
        if history:
            agent.load_history(history)

        # 执行查询
        try:
            response = await agent.query(request.message)
        except Exception as e:
            await error_handler.handle(e, agent, {"request": request})
            raise HTTPException(500, str(e))

        # 保存上下文
        await session_store.save(session_id, agent.messages)

        # 记录用量
        usage = await agent.get_usage()
        await record_usage(tenant_ctx, usage)

        return ChatResponse(
            response=response,
            session_id=session_id,
            usage=usage.model_dump(),
        )


@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    tenant_ctx: TenantContext = Depends(get_tenant_context),
):
    """流式聊天接口"""
    session_id = request.session_id or str(uuid.uuid4())

    async def event_stream():
        async with agent_context(agent_pool, tenant_ctx.tenant_id, session_id) as agent:
            history = await session_store.get(session_id)
            if history:
                agent.load_history(history)

            async for event in agent.query_stream(request.message):
                yield f"data: {serialize_event(event)}\n\n"

            await session_store.save(session_id, agent.messages)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@app.get("/api/v1/sessions")
async def list_sessions(
    tenant_ctx: TenantContext = Depends(get_tenant_context),
    limit: int = 50,
):
    """列出用户会话"""
    sessions = await session_store.list_sessions(
        tenant_ctx.tenant_id,
        tenant_ctx.user_id,
        limit=limit,
    )
    return {"sessions": sessions}


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    tenant_ctx: TenantContext = Depends(get_tenant_context),
):
    """删除会话"""
    await session_store.delete(session_id)
    return {"status": "deleted"}


@app.get("/health")
async def health():
    """健康检查"""
    return await health_checker.check_all()


# 启动事件
@app.on_event("startup")
async def startup():
    # 加载配置
    await load_tenant_configs()

    # 注册默认工具
    register_default_tools()

    # 注册 Prompt 模板
    register_prompt_templates()

    # 启动后台任务
    asyncio.create_task(cleanup_expired_sessions())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 12.2 目录结构建议

```
saas-agent-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理
│   ├── dependencies.py         # 依赖注入
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py             # 聊天 API
│   │   ├── sessions.py         # 会话管理 API
│   │   ├── admin.py            # 管理 API
│   │   └── health.py           # 健康检查
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent_pool.py       # Agent 池管理
│   │   ├── session_store.py    # 会话存储
│   │   ├── tool_registry.py    # 工具注册表
│   │   ├── prompt_registry.py  # Prompt 注册表
│   │   └── context_manager.py  # 上下文管理
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search.py           # 搜索工具
│   │   ├── database.py         # 数据库工具
│   │   ├── file.py             # 文件工具
│   │   └── custom/             # 租户自定义工具
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── base.py             # 基础 Prompt
│   │   ├── customer_service.py # 客服 Prompt
│   │   └── coding.py           # 编程助手 Prompt
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── ecommerce.py        # 电商适配器
│   │   ├── finance.py          # 金融适配器
│   │   └── support.py          # 客服适配器
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── tenant.py           # 租户中间件
│   │   ├── auth.py             # 认证中间件
│   │   ├── rate_limit.py       # 限流中间件
│   │   └── metrics.py          # 指标中间件
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tenant.py           # 租户模型
│   │   ├── session.py          # 会话模型
│   │   └── usage.py            # 用量模型
│   │
│   └── utils/
│       ├── __init__.py
│       ├── security.py         # 安全工具
│       ├── serializers.py      # 序列化工具
│       └── validators.py       # 验证工具
│
├── tests/
│   ├── __init__.py
│   ├── test_chat.py
│   ├── test_tools.py
│   └── test_sessions.py
│
├── migrations/                  # 数据库迁移
│   └── versions/
│
├── config/
│   ├── default.yaml
│   ├── production.yaml
│   └── development.yaml
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── scripts/
│   ├── setup.sh
│   └── migrate.sh
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 总结

本文档涵盖了基于 BU Agent SDK 构建 SaaS Agent 服务的关键设计要点：

| 领域 | 核心策略 |
|------|---------|
| **微服务架构** | 服务拆分 + K8s 部署 + 服务发现 |
| **高可用** | 熔断器 + 优雅降级 + 自适应限流 |
| **OAuth 认证** | JWT + API Key + 三方 OAuth 集成 |
| **租户配置** | CRUD API + 缓存 + 配置热更新 |
| **知识库** | 向量检索 + RAG 增强 + 多命名空间 |
| **MCP 协议** | 服务端/客户端 + 工具代理 |
| **三方 API** | 统一网关 + 限流 + 缓存 + Webhook |
| **并发性** | Agent 池 + 双层限流 + 异步任务队列 |
| **上下文管理** | Ephemeral 消息 + 自动压缩 + 分层存储 |
| **持久化** | Redis（热）+ PostgreSQL（冷）分层 |
| **工具系统** | 注册表模式 + 依赖注入 + 权限控制 |
| **Prompt 管理** | 模板化 + 版本控制 + A/B 测试 |
| **多租户** | 资源隔离 + 配额管理 + 上下文注入 |
| **可扩展性** | 插件系统 + 业务适配器 + 模型路由 |
| **可维护性** | 错误恢复 + 健康检查 + 配置热更新 |
| **安全性** | 输入验证 + 数据脱敏 + 审计日志 |
| **可观测性** | Prometheus 指标 + OpenTelemetry + 结构化日志 |

遵循这些最佳实践，可以构建一个健壮、可扩展、易维护的 SaaS Agent 服务平台。
