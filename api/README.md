# FastAPI Web API 使用指南

## 快速开始

### 1. 本地运行

#### 安装依赖

推荐使用 `uv` 进行包管理（更快、更可靠）：

```bash
# 安装 uv（如果还没有安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖（包括 API 依赖）
uv pip install -e ".[api]"

# 或者使用传统 pip
pip install -e ".[api]"
```

#### 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写必要配置：

```env
OPENAI_API_KEY=sk-your-key-here
DEFAULT_MODEL=gpt-4o
INTENT_MATCHING_MODEL=gpt-4o-mini  # 可选：优化成本
```

#### 启动服务

```bash
python -m api.main
```

或使用 uvicorn：

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- ReDoc 文档：http://localhost:8000/redoc
- 健康检查：http://localhost:8000/api/v1/health

### 2. Docker 运行

#### 构建镜像

```bash
docker build -t workflow-agent-api .
```

#### 运行容器

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key-here \
  -v $(pwd)/config:/app/config \
  --name workflow-agent \
  workflow-agent-api
```

### 3. Docker Compose 运行（推荐）

```bash
# 启动所有服务（包括 MongoDB、Redis）
docker-compose up -d

# 查看日志
docker-compose logs -f workflow-agent

# 停止服务
docker-compose down
```

## API 接口说明

### 1. 查询接口

#### POST `/api/v1/query`

发送消息到 Workflow Agent 并获取响应。

**请求示例**：

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，帮我查询订单状态",
    "session_id": "user_123_session_001",
    "user_id": "user_123"
  }'
```

```python
import httpx

async def query_agent():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={
                "message": "你好，帮我查询订单状态",
                "session_id": "user_123_session_001",
                "user_id": "user_123",
            }
        )
        print(response.json())
```

**响应示例**：

```json
{
  "session_id": "user_123_session_001",
  "message": "您的订单正在处理中，预计明天送达。",
  "status": "success"
}
```

### 2. 获取会话信息

#### GET `/api/v1/session/{session_id}`

获取指定会话的状态和元数据。

**请求示例**：

```bash
curl -X GET "http://localhost:8000/api/v1/session/user_123_session_001"
```

**响应示例**：

```json
{
  "session_id": "user_123_session_001",
  "agent_id": "workflow_agent_001",
  "config_hash": "abc123def456",
  "need_greeting": false,
  "status": "active",
  "message_count": 10
}
```

### 3. 删除会话

#### DELETE `/api/v1/session/{session_id}`

清除指定会话的所有数据。

**请求示例**：

```bash
curl -X DELETE "http://localhost:8000/api/v1/session/user_123_session_001"
```

**响应示例**：

```json
{
  "status": "deleted",
  "session_id": "user_123_session_001"
}
```

### 4. 健康检查

#### GET `/api/v1/health`

检查 API 服务和 WorkflowAgent 的健康状态。

**请求示例**：

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**响应示例**：

```json
{
  "status": "healthy",
  "config_hash": "abc123def456",
  "sessions_count": 5,
  "version": "1.0.0"
}
```

## 完整示例

### Python 客户端

```python
import asyncio
import httpx


class WorkflowAgentClient:
    """Workflow Agent API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def query(self, message: str, session_id: str, user_id: str | None = None):
        """发送查询请求"""
        response = await self.client.post(
            "/api/v1/query",
            json={
                "message": message,
                "session_id": session_id,
                "user_id": user_id,
            }
        )
        response.raise_for_status()
        return response.json()

    async def get_session(self, session_id: str):
        """获取会话信息"""
        response = await self.client.get(f"/api/v1/session/{session_id}")
        response.raise_for_status()
        return response.json()

    async def delete_session(self, session_id: str):
        """删除会话"""
        response = await self.client.delete(f"/api/v1/session/{session_id}")
        response.raise_for_status()
        return response.json()

    async def health_check(self):
        """健康检查"""
        response = await self.client.get("/api/v1/health")
        response.raise_for_status()
        return response.json()

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()


async def main():
    """示例：使用客户端"""
    client = WorkflowAgentClient()

    try:
        # 1. 健康检查
        health = await client.health_check()
        print(f"API Status: {health['status']}")

        # 2. 发送查询
        session_id = "user_123_session_001"
        result = await client.query(
            message="你好，帮我查询订单状态",
            session_id=session_id,
            user_id="user_123"
        )
        print(f"Response: {result['message']}")

        # 3. 获取会话信息
        session = await client.get_session(session_id)
        print(f"Session: {session}")

        # 4. 删除会话
        await client.delete_session(session_id)
        print("Session deleted")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript/TypeScript 客户端

```typescript
interface QueryRequest {
  message: string;
  session_id: string;
  user_id?: string;
}

interface QueryResponse {
  session_id: string;
  message: string;
  status: string;
}

class WorkflowAgentClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async getSession(sessionId: string) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/session/${sessionId}`
    );
    return await response.json();
  }

  async deleteSession(sessionId: string) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/session/${sessionId}`,
      { method: "DELETE" }
    );
    return await response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/api/v1/health`);
    return await response.json();
  }
}

// 使用示例
const client = new WorkflowAgentClient();

const result = await client.query({
  message: "你好，帮我查询订单状态",
  session_id: "user_123_session_001",
  user_id: "user_123",
});

console.log(result.message);
```

## 错误处理

API 返回标准的 HTTP 状态码和错误信息：

### 错误响应格式

```json
{
  "error": "ValueError",
  "message": "Invalid session_id",
  "detail": "Session ID must be non-empty"
}
```

### 常见错误码

| 状态码 | 说明 | 处理方式 |
|-------|------|---------|
| 400 | 请求参数错误 | 检查请求参数格式 |
| 404 | 会话不存在 | 使用有效的 session_id |
| 500 | 服务器内部错误 | 查看服务器日志，联系管理员 |

### Python 错误处理示例

```python
import httpx

async def query_with_error_handling(message: str, session_id: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/query",
                json={"message": message, "session_id": session_id},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"Session not found: {session_id}")
        elif e.response.status_code == 500:
            print(f"Server error: {e.response.json()}")
        else:
            print(f"HTTP error: {e}")
        raise

    except httpx.TimeoutException:
        print("Request timeout")
        raise

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

## 性能优化

### 1. 连接复用

使用长连接减少握手开销：

```python
# 创建全局客户端实例
client = httpx.AsyncClient(
    base_url="http://localhost:8000",
    timeout=30.0,
    limits=httpx.Limits(max_keepalive_connections=10)
)

# 复用连接
for i in range(100):
    response = await client.post("/api/v1/query", json={...})
```

### 2. 并发请求

```python
import asyncio

async def batch_query(messages: list[str], session_id: str):
    """并发查询多条消息"""
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                "http://localhost:8000/api/v1/query",
                json={"message": msg, "session_id": f"{session_id}_{i}"}
            )
            for i, msg in enumerate(messages)
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

### 3. 超时配置

```python
# 设置合理的超时时间
client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=5.0,  # 连接超时
        read=30.0,    # 读取超时
        write=5.0,    # 写入超时
        pool=5.0      # 连接池超时
    )
)
```

## 监控和日志

### 查看日志

```bash
# Docker 日志
docker-compose logs -f workflow-agent

# 本地运行日志
tail -f logs/api.log
```

### 健康检查

```bash
# 使用健康检查端点
curl http://localhost:8000/api/v1/health

# Docker 健康检查
docker inspect --format='{{.State.Health.Status}}' workflow-agent
```

## 部署建议

### 生产环境配置

1. **使用环境变量管理敏感信息**
2. **配置 HTTPS**
3. **限制 CORS 来源**
4. **添加请求限流**
5. **启用日志轮转**
6. **配置健康检查**
7. **使用任务特定模型优化成本**

### 扩展性建议

1. **水平扩展**：运行多个 API 实例 + 负载均衡
2. **会话持久化**：使用 Redis/MongoDB 存储会话
3. **异步任务队列**：使用 Celery 处理长时间运行的任务
4. **微服务化**：参考 [workflow-agent-deployment.md](../docs/workflow-agent-deployment.md)

## 故障排查

### 问题：API 无法启动

```bash
# 检查端口占用
lsof -i :8000

# 检查配置文件
python -c "from api.dependencies import initialize_agent; import asyncio; asyncio.run(initialize_agent())"
```

### 问题：数据库连接失败

```bash
# 检查数据库服务
docker-compose ps

# 测试连接
redis-cli ping
mongo --eval "db.adminCommand('ping')"
```

### 问题：API 响应慢

1. 检查 LLM API 响应时间
2. 使用任务特定模型（如 gpt-4o-mini）
3. 启用缓存
4. 优化数据库查询

## 测试

### 运行单元测试

API 提供了完整的单元测试套件，覆盖所有端点和功能。

```bash
# 安装测试依赖（使用 uv，推荐）
uv pip install -e ".[test]"

# 或使用传统 pip
pip install -e ".[test]"

# 运行所有测试
pytest tests/test_api.py -v

# 运行特定测试
pytest tests/test_api.py::test_query_endpoint_success -v

# 查看测试覆盖率
pytest tests/test_api.py --cov=api --cov-report=html
```

### 测试覆盖范围

测试套件涵盖以下方面：

1. **查询接口测试**
   - ✅ 成功查询
   - ✅ 缺少必填字段
   - ✅ 空消息验证
   - ✅ 可选字段处理
   - ✅ Agent 错误处理

2. **会话管理测试**
   - ✅ 获取已存在会话
   - ✅ 获取不存在会话（404）
   - ✅ 删除会话
   - ✅ 会话隔离

3. **健康检查测试**
   - ✅ 基本健康检查
   - ✅ 会话计数

4. **错误处理测试**
   - ✅ 无效 JSON
   - ✅ 错误的 HTTP 方法
   - ✅ 不存在的端点

5. **集成测试**
   - ✅ 完整工作流（查询 → 获取会话 → 删除会话）
   - ✅ 并发请求

6. **API 文档测试**
   - ✅ OpenAPI schema
   - ✅ Swagger UI
   - ✅ ReDoc UI

7. **CORS 测试**
   - ✅ CORS 头部验证

8. **数据模型测试**
   - ✅ Pydantic 模型验证

### 手动测试

使用示例客户端进行手动测试：

```bash
# 运行客户端示例
python examples/api_client_demo.py
```

## 相关文档

- [Workflow Agent v9 架构文档](../docs/workflow-agent-v9.md)
- [配置管理指南](../docs/configuration-guide.md)
- [任务特定模型配置](../docs/task-specific-models.md)
- [部署最佳实践](../docs/workflow-agent-deployment.md)
- [单元测试源码](../tests/test_api.py)
