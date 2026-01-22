# 任务特定模型配置指南

## 概述

Workflow Agent 支持为不同任务配置不同的 LLM 模型，以优化性能和成本。通过合理配置任务特定模型，可以：

- **降低成本**：对简单任务使用低成本模型，节省约 90% 的 API 调用费用
- **提升性能**：对复杂任务使用高性能模型，确保输出质量
- **优化响应速度**：对需要快速响应的任务使用轻量级模型
- **灵活组合**：混合使用不同提供商的模型，发挥各自优势

## 支持的任务类型

### 1. 意图识别 (Intent Matching)

**用途**：快速判断用户意图、规则匹配

**特点**：
- 需要快速响应
- 低成本
- 简单分类任务

**推荐模型**：
- `gpt-4o-mini` (OpenAI) - 性价比最高
- `gpt-3.5-turbo` (OpenAI) - 快速且便宜
- `gemini-1.5-flash` (Google) - 超快响应

**配置示例**：
```env
INTENT_MATCHING_MODEL=gpt-4o-mini
```

**代码示例**：
```python
from bu_agent_sdk.config import load_config, get_intent_matching_llm

config = load_config()
intent_llm = get_intent_matching_llm(config)

# 快速意图识别
result = await intent_llm.ainvoke([
    {"role": "user", "content": "帮我查询订单状态"}
])
```

### 2. 内容生成 (Content Generation)

**用途**：生成用户可见的高质量内容

**特点**：
- 需要高质量输出
- 复杂推理能力
- 自然语言生成

**推荐模型**：
- `gpt-4o` (OpenAI) - 平衡质量和速度
- `claude-3-5-sonnet-20241022` (Anthropic) - 最高质量
- `gemini-1.5-pro` (Google) - 长上下文支持

**配置示例**：
```env
CONTENT_GENERATION_MODEL=gpt-4o
```

**代码示例**：
```python
from bu_agent_sdk.config import load_config, get_content_generation_llm

config = load_config()
content_llm = get_content_generation_llm(config)

# 生成高质量内容
result = await content_llm.ainvoke([
    {"role": "user", "content": "写一篇关于AI技术发展的文章"}
])
```

### 3. 工作流规划 (Workflow Planning)

**用途**：理解复杂 SOP、制定执行计划

**特点**：
- 需要复杂逻辑理解
- 多步骤推理
- 结构化输出

**推荐模型**：
- `gpt-4o` (OpenAI) - 强大的逻辑推理
- `claude-3-opus-20240229` (Anthropic) - 最强推理能力
- `gemini-1.5-pro` (Google) - 长上下文理解

**配置示例**：
```env
WORKFLOW_PLANNING_MODEL=gpt-4o
```

**代码示例**：
```python
from bu_agent_sdk.config import load_config, get_workflow_planning_llm

config = load_config()
planning_llm = get_workflow_planning_llm(config)

# 制定工作流计划
result = await planning_llm.ainvoke([
    {"role": "user", "content": "分析用户请求并制定执行计划"}
])
```

### 4. LLM 决策 (LLM Decision)

**用途**：SOP 驱动的迭代决策

**特点**：
- 平衡性能和成本
- 多轮对话决策
- 实时响应

**推荐模型**：
- `gpt-4o` (OpenAI) - 平衡选择
- `claude-3-5-sonnet-20241022` (Anthropic) - 高质量决策
- `gpt-4-turbo` (OpenAI) - 快速决策

**配置示例**：
```env
LLM_DECISION_MODEL=gpt-4o
```

**代码示例**：
```python
from bu_agent_sdk.config import load_config, get_llm_decision_llm

config = load_config()
decision_llm = get_llm_decision_llm(config)

# 执行决策
result = await decision_llm.ainvoke([
    {"role": "system", "content": "根据SOP判断下一步操作"},
    {"role": "user", "content": "用户请求: 查询订单"}
])
```

### 5. 响应生成 (Response Generation)

**用途**：生成最终用户响应

**特点**：
- 需要高质量输出
- 自然语言表达
- 用户体验关键

**推荐模型**：
- `gpt-4o` (OpenAI) - 自然流畅
- `claude-3-5-sonnet-20241022` (Anthropic) - 最自然的语言
- `gemini-1.5-pro` (Google) - 多语言支持

**配置示例**：
```env
RESPONSE_GENERATION_MODEL=gpt-4o
```

**代码示例**：
```python
from bu_agent_sdk.config import load_config, get_response_generation_llm

config = load_config()
response_llm = get_response_generation_llm(config)

# 生成用户响应
result = await response_llm.ainvoke([
    {"role": "system", "content": "生成友好的用户响应"},
    {"role": "user", "content": "订单已发货"}
])
```

## 成本优化策略

### 策略 1: 分层模型配置

使用不同成本的模型处理不同复杂度的任务：

```env
# 低成本任务
INTENT_MATCHING_MODEL=gpt-4o-mini          # ~$0.15/1M tokens

# 中等成本任务
LLM_DECISION_MODEL=gpt-4o                  # ~$2.50/1M tokens
WORKFLOW_PLANNING_MODEL=gpt-4o             # ~$2.50/1M tokens

# 高质量任务
CONTENT_GENERATION_MODEL=gpt-4o            # ~$2.50/1M tokens
RESPONSE_GENERATION_MODEL=gpt-4o           # ~$2.50/1M tokens
```

**预期节省**：相比全部使用 `gpt-4o`，可节省约 40-60% 的成本。

### 策略 2: 混合提供商

利用不同提供商的优势：

```env
# OpenAI - 快速响应
INTENT_MATCHING_MODEL=gpt-4o-mini
LLM_DECISION_MODEL=gpt-4o

# Anthropic - 高质量内容
CONTENT_GENERATION_MODEL=claude-3-5-sonnet-20241022
RESPONSE_GENERATION_MODEL=claude-3-5-sonnet-20241022

# Google - 长上下文
WORKFLOW_PLANNING_MODEL=gemini-1.5-pro
```

**优势**：
- 避免单点依赖
- 发挥各家模型优势
- 提高系统可靠性

### 策略 3: 动态模型选择

根据任务复杂度动态选择模型：

```python
from bu_agent_sdk.config import load_config, get_llm_from_config

config = load_config()

def get_task_llm(task_complexity: str):
    """根据任务复杂度选择模型"""
    if task_complexity == "simple":
        return get_llm_from_config(config, "gpt-4o-mini")
    elif task_complexity == "medium":
        return get_llm_from_config(config, "gpt-4o")
    else:  # complex
        return get_llm_from_config(config, "claude-3-opus-20240229")

# 使用
simple_llm = get_task_llm("simple")
complex_llm = get_task_llm("complex")
```

## 性能优化策略

### 策略 1: 并行执行

对于独立的任务，使用并行执行提升性能：

```python
import asyncio
from bu_agent_sdk.config import (
    load_config,
    get_intent_matching_llm,
    get_content_generation_llm,
)

config = load_config()
intent_llm = get_intent_matching_llm(config)
content_llm = get_content_generation_llm(config)

# 并行执行意图识别和内容生成
intent_task = intent_llm.ainvoke([{"role": "user", "content": "查询订单"}])
content_task = content_llm.ainvoke([{"role": "user", "content": "生成介绍"}])

intent_result, content_result = await asyncio.gather(intent_task, content_task)
```

### 策略 2: 缓存策略

对于常见任务，使用缓存避免重复调用：

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_intent(user_message: str):
    """缓存常见意图识别结果"""
    # 实现缓存逻辑
    pass
```

### 策略 3: 流式响应

对于内容生成任务，使用流式输出提升用户体验：

```python
from bu_agent_sdk.config import load_config, get_response_generation_llm

config = load_config()
response_llm = get_response_generation_llm(config)

# 流式生成响应
async for chunk in response_llm.astream([
    {"role": "user", "content": "生成长文本"}
]):
    print(chunk.content, end="", flush=True)
```

## 最佳实践

### 1. 开发环境配置

开发环境使用统一的低成本模型：

```env
# .env.development
DEFAULT_MODEL=gpt-4o-mini
INTENT_MATCHING_MODEL=gpt-4o-mini
CONTENT_GENERATION_MODEL=gpt-4o-mini
WORKFLOW_PLANNING_MODEL=gpt-4o-mini
LLM_DECISION_MODEL=gpt-4o-mini
RESPONSE_GENERATION_MODEL=gpt-4o-mini
```

### 2. 生产环境配置

生产环境使用优化的模型组合：

```env
# .env.production
DEFAULT_MODEL=gpt-4o
INTENT_MATCHING_MODEL=gpt-4o-mini          # 节省成本
CONTENT_GENERATION_MODEL=gpt-4o            # 保证质量
WORKFLOW_PLANNING_MODEL=gpt-4o             # 保证准确性
LLM_DECISION_MODEL=gpt-4o                  # 平衡性能
RESPONSE_GENERATION_MODEL=gpt-4o           # 保证体验
```

### 3. 监控和调优

监控各任务的性能和成本：

```python
import time
from bu_agent_sdk.config import load_config, get_intent_matching_llm

config = load_config()
intent_llm = get_intent_matching_llm(config)

# 监控执行时间和成本
start_time = time.time()
result = await intent_llm.ainvoke([{"role": "user", "content": "查询"}])
duration = time.time() - start_time

print(f"任务: 意图识别")
print(f"模型: {intent_llm.model_name}")
print(f"耗时: {duration:.2f}s")
print(f"Token使用: {result.usage_metadata}")
```

## 常见问题

### Q1: 如何选择合适的模型？

**A**: 根据任务特点选择：
- 简单分类任务 → `gpt-4o-mini`
- 复杂推理任务 → `gpt-4o` 或 `claude-3-5-sonnet`
- 长上下文任务 → `gemini-1.5-pro`
- 成本敏感任务 → `gpt-4o-mini` 或 `gpt-3.5-turbo`

### Q2: 不配置任务特定模型会怎样？

**A**: 系统会使用 `DEFAULT_MODEL` 作为所有任务的模型。这样配置简单，但无法优化成本和性能。

### Q3: 可以动态切换模型吗？

**A**: 可以。使用 `get_llm_from_config(config, model_override)` 方法动态指定模型：

```python
from bu_agent_sdk.config import load_config, get_llm_from_config

config = load_config()

# 动态选择模型
llm = get_llm_from_config(config, "gpt-4o-mini")
```

### Q4: 如何测试不同模型的效果？

**A**: 使用 A/B 测试比较不同模型的效果：

```python
import asyncio
from bu_agent_sdk.config import load_config, get_llm_from_config

config = load_config()

async def compare_models(prompt: str):
    """比较不同模型的效果"""
    models = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"]

    tasks = [
        get_llm_from_config(config, model).ainvoke([{"role": "user", "content": prompt}])
        for model in models
    ]

    results = await asyncio.gather(*tasks)

    for model, result in zip(models, results):
        print(f"\n模型: {model}")
        print(f"响应: {result.content[:100]}...")
        print(f"Token: {result.usage_metadata}")

# 运行比较
await compare_models("帮我查询订单状态")
```

## 参考资料

- [配置管理指南](configuration-guide.md)
- [OpenAI 模型定价](https://openai.com/pricing)
- [Anthropic 模型定价](https://www.anthropic.com/pricing)
- [Google AI 模型定价](https://ai.google.dev/pricing)
- [示例代码](../examples/task_specific_models_demo.py)
