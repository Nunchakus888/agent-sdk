# 配置管理指南

本文档说明如何配置 Workflow Agent 项目的各项参数，包括 API Keys、数据库连接等。

## 目录

- [快速开始](#快速开始)
- [配置方式](#配置方式)
- [配置项说明](#配置项说明)
- [使用示例](#使用示例)
- [生产环境部署](#生产环境部署)
- [Docker 部署](#docker-部署)

## 快速开始

### 1. 复制环境变量模板

```bash
cp .env.example .env
```

### 2. 编辑 `.env` 文件

```bash
# 编辑配置文件
vim .env  # 或使用其他编辑器
```

### 3. 填写必要的配置

```env
# 至少需要配置一个 LLM API Key
OPENAI_API_KEY=sk-your-openai-key-here

# 数据库连接（使用默认值即可开始）
MONGODB_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
```

### 4. 在代码中使用

```python
from bu_agent_sdk.config import load_config, get_llm_from_config

# 加载配置
config = load_config()

# 创建 LLM
llm = get_llm_from_config(config)
```

## 配置方式

### 方式 1：环境变量文件（推荐）

创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-xxx
MONGODB_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
```

**优点：**
- 简单易用
- 不会提交到版本控制
- 适合本地开发

### 方式 2：系统环境变量

```bash
export OPENAI_API_KEY=sk-xxx
export MONGODB_URI=mongodb://localhost:27017
```

**优点：**
- 适合 CI/CD 环境
- 适合 Docker/Kubernetes 部署

### 方式 3：代码中直接配置

```python
from bu_agent_sdk.llm import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    api_key="sk-xxx"  # 不推荐硬编码
)
```

**缺点：**
- 不安全（可能泄露密钥）
- 不灵活（需要修改代码）

## 配置项说明

### LLM 配置

#### 基础配置

| 配置项 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - | 使用 OpenAI 时必填 |
| `OPENAI_BASE_URL` | OpenAI API 基础 URL | - | 可选 |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 | - | 使用 Claude 时必填 |
| `GOOGLE_API_KEY` | Google API 密钥 | - | 使用 Gemini 时必填 |
| `DEFAULT_MODEL` | 默认使用的模型 | `gpt-4o` | 否 |

**支持的模型：**
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- Google: `gemini-1.5-pro`, `gemini-1.5-flash`

#### 任务特定模型配置（可选）

为不同任务配置不同的模型，以优化性能和成本：

| 配置项 | 说明 | 推荐模型 | 默认值 |
|--------|------|----------|--------|
| `INTENT_MATCHING_MODEL` | 意图识别模型 | `gpt-4o-mini` | 使用 `DEFAULT_MODEL` |
| `CONTENT_GENERATION_MODEL` | 内容生成模型 | `gpt-4o` | 使用 `DEFAULT_MODEL` |
| `WORKFLOW_PLANNING_MODEL` | 工作流规划模型 | `gpt-4o` | 使用 `DEFAULT_MODEL` |
| `LLM_DECISION_MODEL` | LLM决策模型 | `gpt-4o` | 使用 `DEFAULT_MODEL` |
| `RESPONSE_GENERATION_MODEL` | 响应生成模型 | `gpt-4o` | 使用 `DEFAULT_MODEL` |

**任务特定模型使用场景：**

1. **意图识别模型** (`INTENT_MATCHING_MODEL`)
   - 用途：快速判断用户意图、规则匹配
   - 特点：需要快速响应、低成本
   - 推荐：`gpt-4o-mini`, `gpt-3.5-turbo`, `gemini-1.5-flash`

2. **内容生成模型** (`CONTENT_GENERATION_MODEL`)
   - 用途：生成用户可见的高质量内容
   - 特点：需要高质量输出、复杂推理
   - 推荐：`gpt-4o`, `claude-3-5-sonnet-20241022`, `gemini-1.5-pro`

3. **工作流规划模型** (`WORKFLOW_PLANNING_MODEL`)
   - 用途：理解复杂SOP、制定执行计划
   - 特点：需要复杂逻辑理解能力
   - 推荐：`gpt-4o`, `claude-3-opus-20240229`

4. **LLM决策模型** (`LLM_DECISION_MODEL`)
   - 用途：SOP驱动的迭代决策
   - 特点：平衡性能和成本
   - 推荐：`gpt-4o`, `claude-3-5-sonnet-20241022`

5. **响应生成模型** (`RESPONSE_GENERATION_MODEL`)
   - 用途：生成最终用户响应
   - 特点：需要高质量、自然的语言输出
   - 推荐：`gpt-4o`, `claude-3-5-sonnet-20241022`

**成本优化示例：**

```env
# 使用不同模型优化成本
DEFAULT_MODEL=gpt-4o
INTENT_MATCHING_MODEL=gpt-4o-mini          # 节省约90%成本
CONTENT_GENERATION_MODEL=gpt-4o            # 保证质量
WORKFLOW_PLANNING_MODEL=gpt-4o             # 保证准确性
LLM_DECISION_MODEL=gpt-4o                  # 平衡性能
RESPONSE_GENERATION_MODEL=gpt-4o           # 保证用户体验
```

**混合使用不同提供商：**

```env
# 利用各家模型的优势
DEFAULT_MODEL=gpt-4o
INTENT_MATCHING_MODEL=gpt-4o-mini          # OpenAI 快速模型
CONTENT_GENERATION_MODEL=claude-3-5-sonnet-20241022  # Anthropic 高质量
WORKFLOW_PLANNING_MODEL=gpt-4o             # OpenAI 逻辑理解
LLM_DECISION_MODEL=gpt-4o                  # OpenAI 决策
RESPONSE_GENERATION_MODEL=claude-3-5-sonnet-20241022  # Anthropic 自然语言
```

### 数据库配置

#### MongoDB

| 配置项 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `MONGODB_URI` | MongoDB 连接字符串 | `mongodb://localhost:27017` | 否 |
| `MONGODB_DB_NAME` | 数据库名称 | `workflow_agent` | 否 |

**连接字符串示例：**
```
# 本地
mongodb://localhost:27017

# 带认证
mongodb://username:password@localhost:27017

# MongoDB Atlas
mongodb+srv://username:password@cluster.mongodb.net/

# 副本集
mongodb://host1:27017,host2:27017,host3:27017/?replicaSet=myReplicaSet
```

#### PostgreSQL

| 配置项 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `POSTGRESQL_URI` | PostgreSQL 连接字符串 | - | 可选 |

**连接字符串示例：**
```
# 本地
postgresql://localhost/workflow_agent

# 带认证
postgresql://user:password@localhost:5432/workflow_agent

# 完整格式
postgresql://user:password@host:port/database?sslmode=require
```

#### Redis

| 配置项 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `REDIS_URL` | Redis 连接 URL | `redis://localhost:6379` | 否 |
| `REDIS_TTL` | 缓存过期时间（秒） | `3600` | 否 |

**连接 URL 示例：**
```
# 本地
redis://localhost:6379

# 带密码
redis://:password@localhost:6379

# 指定数据库
redis://localhost:6379/0

# Redis Sentinel
redis-sentinel://localhost:26379/mymaster

# Redis Cluster
redis-cluster://localhost:7000,localhost:7001,localhost:7002
```

### 应用配置

| 配置项 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `ENVIRONMENT` | 运行环境 | `development` | 否 |
| `LOG_LEVEL` | 日志级别 | `INFO` | 否 |
| `MAX_ITERATIONS` | 最大迭代次数 | `5` | 否 |
| `ITERATION_STRATEGY` | 迭代策略 | `sop_driven` | 否 |

**环境选项：**
- `development`: 开发环境
- `staging`: 预发布环境
- `production`: 生产环境

**日志级别：**
- `DEBUG`: 调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

## 使用示例

### 基础使用

```python
from bu_agent_sdk.config import load_config, get_llm_from_config

# 加载配置
config = load_config()

# 创建 LLM
llm = get_llm_from_config(config)

# 访问配置
print(f"Environment: {config.environment}")
print(f"MongoDB URI: {config.database.mongodb_uri}")
```

### 使用任务特定模型

```python
from bu_agent_sdk.config import (
    load_config,
    get_intent_matching_llm,
    get_content_generation_llm,
    get_workflow_planning_llm,
    get_llm_decision_llm,
    get_response_generation_llm,
)

# 加载配置
config = load_config()

# 创建任务特定的LLM实例
intent_llm = get_intent_matching_llm(config)          # 意图识别（快速、低成本）
content_llm = get_content_generation_llm(config)      # 内容生成（高质量）
planning_llm = get_workflow_planning_llm(config)      # 工作流规划（复杂逻辑）
decision_llm = get_llm_decision_llm(config)           # LLM决策（SOP驱动）
response_llm = get_response_generation_llm(config)    # 响应生成（用户响应）

# 使用不同模型执行不同任务
intent_result = await intent_llm.ainvoke([
    {"role": "user", "content": "帮我查询订单"}
])

content_result = await content_llm.ainvoke([
    {"role": "user", "content": "生成产品介绍"}
])
```

### 完整示例

```python
import asyncio
from bu_agent_sdk.config import (
    load_config,
    get_llm_from_config,
    get_session_store_from_config,
    get_plan_cache_from_config,
)
from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema

async def main():
    # 1. 加载配置
    config = load_config()

    # 2. 创建组件
    llm = get_llm_from_config(config)
    session_store = await get_session_store_from_config(config)
    plan_cache = await get_plan_cache_from_config(config)

    # 3. 加载 workflow 配置
    workflow_config = WorkflowConfigSchema.parse_file("config/workflow_config.json")

    # 4. 创建 agent
    agent = WorkflowAgent(
        config=workflow_config,
        llm=llm,
        session_store=session_store,
        plan_cache=plan_cache,
    )

    # 5. 使用 agent
    response = await agent.query(
        message="你好",
        session_id="demo_001"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### 指定配置文件

```python
from pathlib import Path
from bu_agent_sdk.config import load_config

# 使用自定义配置文件
config = load_config(env_file=Path("/path/to/.env.production"))
```

## 生产环境部署

### 1. 使用环境变量

```bash
# 设置环境变量
export ENVIRONMENT=production
export OPENAI_API_KEY=sk-prod-xxx
export MONGODB_URI=mongodb://prod-host:27017
export REDIS_URL=redis://prod-host:6379

# 运行应用
python app.py
```

### 2. 使用配置文件

```bash
# 创建生产环境配置
cp .env.example .env.production

# 编辑配置
vim .env.production

# 指定配置文件运行
ENV_FILE=.env.production python app.py
```

### 3. 安全建议

**✅ 推荐做法：**
- 使用环境变量或密钥管理服务（如 AWS Secrets Manager、HashiCorp Vault）
- 不要将 `.env` 文件提交到版本控制
- 定期轮换 API Keys
- 使用最小权限原则

**❌ 避免做法：**
- 在代码中硬编码密钥
- 将密钥提交到 Git
- 在日志中打印密钥
- 使用弱密码

## Docker 部署

### 1. 使用 docker-compose

```bash
# 创建 .env 文件
cat > .env << EOF
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
POSTGRES_PASSWORD=secure_password
EOF

# 启动服务
docker-compose up -d
```

### 2. 环境变量传递

```yaml
# docker-compose.yml
services:
  workflow-agent:
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      MONGODB_URI: mongodb://mongodb:27017
      REDIS_URL: redis://redis:6379
```

### 3. 使用 Docker Secrets（推荐）

```bash
# 创建 secret
echo "sk-xxx" | docker secret create openai_api_key -

# 在 docker-compose.yml 中使用
services:
  workflow-agent:
    secrets:
      - openai_api_key
    environment:
      OPENAI_API_KEY_FILE: /run/secrets/openai_api_key

secrets:
  openai_api_key:
    external: true
```

## Kubernetes 部署

### 使用 ConfigMap 和 Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: workflow-agent-config
data:
  ENVIRONMENT: production
  MONGODB_URI: mongodb://mongodb-service:27017
  REDIS_URL: redis://redis-service:6379
  MAX_ITERATIONS: "5"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: workflow-agent-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: sk-xxx
  ANTHROPIC_API_KEY: sk-ant-xxx

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: workflow-agent
spec:
  template:
    spec:
      containers:
      - name: workflow-agent
        image: workflow-agent:latest
        envFrom:
        - configMapRef:
            name: workflow-agent-config
        - secretRef:
            name: workflow-agent-secrets
```

## 故障排查

### 问题：找不到配置文件

```
FileNotFoundError: .env file not found
```

**解决方案：**
1. 确认 `.env` 文件存在
2. 检查文件路径
3. 使用绝对路径：`load_config(env_file="/absolute/path/.env")`

### 问题：API Key 无效

```
AuthenticationError: Invalid API key
```

**解决方案：**
1. 检查 API Key 是否正确
2. 确认环境变量已设置：`echo $OPENAI_API_KEY`
3. 检查 API Key 是否过期

### 问题：数据库连接失败

```
ConnectionError: Could not connect to MongoDB
```

**解决方案：**
1. 检查数据库服务是否运行
2. 验证连接字符串格式
3. 检查网络连接和防火墙设置
4. 确认认证信息正确

## 参考资料

- [环境变量最佳实践](https://12factor.net/config)
- [Docker Secrets](https://docs.docker.com/engine/swarm/secrets/)
- [Kubernetes ConfigMap](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [MongoDB 连接字符串](https://www.mongodb.com/docs/manual/reference/connection-string/)
- [PostgreSQL 连接字符串](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING)
