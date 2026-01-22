# 任务特定模型配置 - 快速参考

## 快速开始

### 1. 配置环境变量

编辑 `.env` 文件：

```env
# API Keys
OPENAI_API_KEY=sk-your-key-here

# 默认模型
DEFAULT_MODEL=gpt-4o

# 任务特定模型（可选）
INTENT_MATCHING_MODEL=gpt-4o-mini
CONTENT_GENERATION_MODEL=gpt-4o
WORKFLOW_PLANNING_MODEL=gpt-4o
LLM_DECISION_MODEL=gpt-4o
RESPONSE_GENERATION_MODEL=gpt-4o
```

### 2. 在代码中使用

```python
from bu_agent_sdk.config import (
    load_config,
    get_intent_matching_llm,
    get_content_generation_llm,
)

# 加载配置
config = load_config()

# 获取任务特定的LLM
intent_llm = get_intent_matching_llm(config)
content_llm = get_content_generation_llm(config)

# 使用
result = await intent_llm.ainvoke([
    {"role": "user", "content": "查询订单"}
])
```

## 可用的辅助函数

| 函数 | 用途 | 推荐模型 |
|------|------|----------|
| `get_intent_matching_llm(config)` | 意图识别 | `gpt-4o-mini` |
| `get_content_generation_llm(config)` | 内容生成 | `gpt-4o` |
| `get_workflow_planning_llm(config)` | 工作流规划 | `gpt-4o` |
| `get_llm_decision_llm(config)` | LLM决策 | `gpt-4o` |
| `get_response_generation_llm(config)` | 响应生成 | `gpt-4o` |

## 推荐配置

### 开发环境（低成本）

```env
DEFAULT_MODEL=gpt-4o-mini
INTENT_MATCHING_MODEL=gpt-4o-mini
CONTENT_GENERATION_MODEL=gpt-4o-mini
WORKFLOW_PLANNING_MODEL=gpt-4o-mini
LLM_DECISION_MODEL=gpt-4o-mini
RESPONSE_GENERATION_MODEL=gpt-4o-mini
```

### 生产环境（优化配置）

```env
DEFAULT_MODEL=gpt-4o
INTENT_MATCHING_MODEL=gpt-4o-mini          # 节省90%成本
CONTENT_GENERATION_MODEL=gpt-4o            # 保证质量
WORKFLOW_PLANNING_MODEL=gpt-4o             # 保证准确性
LLM_DECISION_MODEL=gpt-4o                  # 平衡性能
RESPONSE_GENERATION_MODEL=gpt-4o           # 保证体验
```

### 混合提供商（最佳实践）

```env
DEFAULT_MODEL=gpt-4o
INTENT_MATCHING_MODEL=gpt-4o-mini
CONTENT_GENERATION_MODEL=claude-3-5-sonnet-20241022
WORKFLOW_PLANNING_MODEL=gemini-1.5-pro
LLM_DECISION_MODEL=gpt-4o
RESPONSE_GENERATION_MODEL=claude-3-5-sonnet-20241022
```

## 模型选择指南

### OpenAI 模型

| 模型 | 特点 | 适用场景 | 成本 |
|------|------|----------|------|
| `gpt-4o-mini` | 快速、便宜 | 意图识别、简单任务 | $ |
| `gpt-4o` | 平衡 | 大多数任务 | $$ |
| `gpt-4-turbo` | 快速、高质量 | 复杂任务 | $$$ |

### Anthropic 模型

| 模型 | 特点 | 适用场景 | 成本 |
|------|------|----------|------|
| `claude-3-5-sonnet-20241022` | 高质量、自然 | 内容生成、响应生成 | $$ |
| `claude-3-opus-20240229` | 最强推理 | 复杂逻辑、规划 | $$$ |

### Google 模型

| 模型 | 特点 | 适用场景 | 成本 |
|------|------|----------|------|
| `gemini-1.5-flash` | 超快 | 意图识别 | $ |
| `gemini-1.5-pro` | 长上下文 | 工作流规划 | $$ |

## 成本估算

假设每天处理 10,000 个请求：

### 全部使用 gpt-4o
- 意图识别: 10,000 × $0.0025 = $25
- 内容生成: 10,000 × $0.0025 = $25
- 工作流规划: 10,000 × $0.0025 = $25
- LLM决策: 10,000 × $0.0025 = $25
- 响应生成: 10,000 × $0.0025 = $25
- **总计: $125/天**

### 优化配置
- 意图识别: 10,000 × $0.00015 = $1.5 (gpt-4o-mini)
- 内容生成: 10,000 × $0.0025 = $25 (gpt-4o)
- 工作流规划: 10,000 × $0.0025 = $25 (gpt-4o)
- LLM决策: 10,000 × $0.0025 = $25 (gpt-4o)
- 响应生成: 10,000 × $0.0025 = $25 (gpt-4o)
- **总计: $101.5/天**
- **节省: $23.5/天 (19%)**

### 激进优化
- 意图识别: 10,000 × $0.00015 = $1.5 (gpt-4o-mini)
- 内容生成: 10,000 × $0.0025 = $25 (gpt-4o)
- 工作流规划: 10,000 × $0.00015 = $1.5 (gpt-4o-mini)
- LLM决策: 10,000 × $0.00015 = $1.5 (gpt-4o-mini)
- 响应生成: 10,000 × $0.0025 = $25 (gpt-4o)
- **总计: $54.5/天**
- **节省: $70.5/天 (56%)**

## 常见问题

### Q: 不配置任务特定模型会怎样？
A: 系统会使用 `DEFAULT_MODEL` 作为所有任务的模型。

### Q: 可以只配置部分任务的模型吗？
A: 可以。未配置的任务会使用 `DEFAULT_MODEL`。

### Q: 如何验证配置是否生效？
A: 运行以下代码：
```python
from bu_agent_sdk.config import load_config, get_intent_matching_llm

config = load_config()
intent_llm = get_intent_matching_llm(config)
print(f"意图识别模型: {intent_llm.model_name}")
```

### Q: 可以在运行时切换模型吗？
A: 可以。使用 `get_llm_from_config(config, model_override)` 动态指定模型。

## 更多资源

- [完整配置指南](configuration-guide.md)
- [任务特定模型详细文档](task-specific-models.md)
- [实现总结](task-specific-models-implementation.md)
- [示例代码](../examples/task_specific_models_demo.py)
