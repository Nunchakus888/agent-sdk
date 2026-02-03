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


### 常见错误码

| 状态码 | 说明 | 处理方式 |
|-------|------|---------|
| 400 | 请求参数错误 | 检查请求参数格式 |
| 404 | 会话不存在 | 使用有效的 session_id |
| 500 | 服务器内部错误 | 查看服务器日志，联系管理员 |



## 监控和日志



### 健康检查

```bash
# 使用健康检查端点
curl http://localhost:8000/api/v1/health

# Docker 健康检查
docker inspect --format='{{.State.Health.Status}}' workflow-agent
```

## 部署建议

### 生产环境配置


### 扩展性建议

1. **水平扩展**：运行多个 API 实例 + 负载均衡
2. **会话持久化**：使用 Redis/MongoDB 存储会话
3. **异步任务队列**：使用 Celery 处理长时间运行的任务
4. **微服务化**：参考 [workflow-agent-deployment.md](../docs/workflow-agent-deployment.md)

## 故障排查



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
