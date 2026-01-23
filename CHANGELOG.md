# 更新日志

## v9.5 (2026-01-23)

### 新增功能

#### Web API 集成 ⭐

实现完整的 FastAPI Web API，提供 RESTful 接口。

**核心特性**：
- FastAPI RESTful API 实现
- 同步查询接口（/api/v1/query）
- 会话管理接口（获取、删除会话）
- 健康检查接口（监控和状态查询）
- 模块化设计（依赖注入、清晰分层）
- Docker 支持（Dockerfile + docker-compose）
- 完整文档和客户端示例
- 方便后期微服务化拓展

**新增文件**：
- `api/__init__.py` - API 包初始化
- `api/main.py` - FastAPI 主应用
- `api/models.py` - 数据模型（Request/Response）
- `api/routes.py` - API 路由定义
- `api/dependencies.py` - 依赖注入和配置管理
- `api/requirements.txt` - API 依赖包
- `api/README.md` - API 使用指南
- `Dockerfile` - Docker 镜像构建
- `docker-compose.yml` - 更新，支持任务特定模型配置
- `examples/api_client_demo.py` - Python 客户端示例
- `tests/test_api.py` - API 单元测试套件
- `requirements-test.txt` - 测试依赖包
- `docs/api-testing-summary.md` - 测试实现总结文档

**API 端点**：
- `POST /api/v1/query` - 发送消息到 Workflow Agent
- `GET /api/v1/session/{session_id}` - 获取会话信息
- `DELETE /api/v1/session/{session_id}` - 删除会话
- `GET /api/v1/health` - 健康检查
- `GET /` - API 根路径
- `GET /docs` - Swagger API 文档
- `GET /redoc` - ReDoc API 文档

**架构特点**：
- 单体架构（当前）
- 模块化设计（方便微服务化）
- 依赖注入模式
- 全局异常处理
- 结构化日志
- CORS 支持
- 健康检查

**使用示例**：

```bash
# 本地运行
python -m api.main

# Docker 运行
docker-compose up -d

# API 调用
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "session_id": "session_001"}'
```

### 文档更新

- `docs/workflow-agent-v9.md` - 新增"六、Web API 集成"章节（v9.5）
- 包含架构设计、API 端点、快速开始、性能优化、测试等内容
- `api/README.md` - 新增测试章节，包含单元测试指南

### 测试改进

- 新增完整的 API 单元测试套件（`tests/test_api.py`）
- 覆盖所有 API 端点和功能
- 测试覆盖范围：
  - ✅ 查询接口测试（成功、失败、参数验证）
  - ✅ 会话管理测试（获取、删除、隔离）
  - ✅ 健康检查测试
  - ✅ 错误处理测试
  - ✅ 集成测试（完整工作流、并发请求）
  - ✅ API 文档测试（OpenAPI、Swagger、ReDoc）
  - ✅ CORS 测试
  - ✅ 数据模型验证测试

### 部署改进

- Docker 镜像支持健康检查
- docker-compose 支持任务特定模型配置
- 环境变量默认值配置
- **使用 uv 进行包管理，大幅提升安装速度**

### 包管理优化 ⭐

- 迁移到 `uv` 包管理工具（比 pip 快 10-100 倍）
- 更新 `pyproject.toml`，添加 API 和测试依赖组
- 更新 Dockerfile 使用 uv 安装依赖
- 所有文档更新为推荐使用 uv

**依赖组**：
- `[api]` - API 相关依赖（fastapi, uvicorn, redis, motor, asyncpg）
- `[test]` - 测试相关依赖（pytest, pytest-asyncio, pytest-cov）
- `[dev]` - 开发环境所有依赖

**安装示例**：
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 API 依赖
uv pip install -e ".[api]"

# 安装测试依赖
uv pip install -e ".[test]"

# 安装所有开发依赖
uv pip install -e ".[dev]"
```

---

## v9.4 (2026-01-23)

### 新增功能

#### 任务特定模型配置 ⭐

支持为不同任务配置不同的 LLM 模型，以优化性能和成本。

**新增配置项**：
- `INTENT_MATCHING_MODEL` - 意图识别模型
- `CONTENT_GENERATION_MODEL` - 内容生成模型
- `WORKFLOW_PLANNING_MODEL` - 工作流规划模型
- `LLM_DECISION_MODEL` - LLM决策模型
- `RESPONSE_GENERATION_MODEL` - 响应生成模型

**新增函数**：
- `get_intent_matching_llm(config)` - 获取意图识别LLM
- `get_content_generation_llm(config)` - 获取内容生成LLM
- `get_workflow_planning_llm(config)` - 获取工作流规划LLM
- `get_llm_decision_llm(config)` - 获取LLM决策模型
- `get_response_generation_llm(config)` - 获取响应生成LLM

**核心优势**：
- 成本优化：意图识别使用低成本模型可节省约 90% 成本
- 性能优化：快速模型提升响应速度 2-3 倍
- 灵活配置：支持混合使用不同提供商的模型
- 向后兼容：所有配置可选，不影响现有代码

### 文件变更

#### 核心代码
- `bu_agent_sdk/config.py` - 新增任务特定模型配置和辅助函数

#### 配置文件
- `.env.example` - 新增任务特定模型配置示例

#### 文档
- `docs/workflow-agent-v9.md` - 新增"五、任务特定模型配置"章节
- `docs/configuration-guide.md` - 新增任务特定模型配置说明
- `docs/task-specific-models.md` - 新增任务特定模型详细指南
- `docs/task-specific-models-implementation.md` - 新增实现总结文档
- `docs/task-specific-models-quickstart.md` - 新增快速参考指南

#### 示例代码
- `examples/task_specific_models_demo.py` - 新增任务特定模型使用示例

### 使用示例

```python
from bu_agent_sdk.config import (
    load_config,
    get_intent_matching_llm,
    get_content_generation_llm,
)

# 加载配置
config = load_config()

# 获取任务特定的LLM
intent_llm = get_intent_matching_llm(config)      # 使用快速、低成本模型
content_llm = get_content_generation_llm(config)  # 使用高质量模型

# 使用不同模型执行不同任务
intent_result = await intent_llm.ainvoke([
    {"role": "user", "content": "帮我查询订单"}
])

content_result = await content_llm.ainvoke([
    {"role": "user", "content": "生成产品介绍"}
])
```

### 推荐配置

**生产环境优化配置**：
```env
DEFAULT_MODEL=gpt-4o
INTENT_MATCHING_MODEL=gpt-4o-mini          # 节省约90%成本
CONTENT_GENERATION_MODEL=gpt-4o            # 保证质量
WORKFLOW_PLANNING_MODEL=gpt-4o             # 保证准确性
LLM_DECISION_MODEL=gpt-4o                  # 平衡性能
RESPONSE_GENERATION_MODEL=gpt-4o           # 保证用户体验
```

### 成本分析

假设每天处理 10,000 个请求：

- **全部使用 gpt-4o**: $125/天
- **优化配置**: $101.5/天（节省 19%）
- **激进优化**: $54.5/天（节省 56%）

### 向后兼容性

- ✅ 所有任务特定模型配置都是可选的
- ✅ 如果不配置，系统会使用 `DEFAULT_MODEL`
- ✅ 现有代码无需修改即可继续使用
- ✅ 可以逐步迁移到任务特定模型配置

---

## v9.3 (2026-01-22)

### 新增功能

#### 数据库存储支持

- 新增 MongoDB 会话存储实现
- 新增 PostgreSQL 会话存储实现
- 新增 Redis 计划缓存实现
- 新增执行历史存储

### 文件变更

- `bu_agent_sdk/workflow/storage.py` - 新增数据库存储实现
- `tests/test_workflow_storage.py` - 新增存储单元测试
- `docs/workflow-agent-v9.md` - 新增"四、数据库存储"章节

---

## v9.2 (2026-01-21)

### 核心特性

1. **SOP驱动的轻量级迭代**
   - 外层轻量级循环：支持多步骤Action序列执行
   - LLM智能决策：每步判断是否继续或响应
   - 可控迭代：限制最大次数，防止无限循环

2. **Silent Action 优化**
   - Flow/System silent 操作直接跳出迭代
   - 不参与上下文构建，避免干扰后续决策

3. **KB 并行查询优化**
   - 迭代开始时预先查询知识库
   - 不阻塞 LLM 决策，并行执行

4. **Skills 支持多种执行模式**
   - Agent 模式：创建子 Agent
   - Function 模式：直接 HTTP 调用

---

## v9.1 (2026-01-20)

### 初始版本

- 基础架构设计
- 配置驱动工作流引擎
- 意图匹配系统
- Action 执行器
