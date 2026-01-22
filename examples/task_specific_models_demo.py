"""
任务特定模型使用示例

演示如何为不同任务使用不同的LLM模型，以优化性能和成本。
"""

import asyncio
from bu_agent_sdk.config import (
    load_config,
    get_intent_matching_llm,
    get_content_generation_llm,
    get_workflow_planning_llm,
    get_llm_decision_llm,
    get_response_generation_llm,
)


async def main():
    """演示任务特定模型的使用"""

    # 1. 加载配置
    print("=== 加载配置 ===")
    config = load_config()
    print(f"默认模型: {config.llm.default_model}")
    print(f"意图识别模型: {config.llm.intent_matching_model or '使用默认模型'}")
    print(f"内容生成模型: {config.llm.content_generation_model or '使用默认模型'}")
    print(f"工作流规划模型: {config.llm.workflow_planning_model or '使用默认模型'}")
    print(f"LLM决策模型: {config.llm.llm_decision_model or '使用默认模型'}")
    print(f"响应生成模型: {config.llm.response_generation_model or '使用默认模型'}")
    print()

    # 2. 创建任务特定的LLM实例
    print("=== 创建任务特定LLM实例 ===")

    # 意图识别LLM（快速、低成本）
    intent_llm = get_intent_matching_llm(config)
    print(f"意图识别LLM: {intent_llm.model_name}")

    # 内容生成LLM（高质量）
    content_llm = get_content_generation_llm(config)
    print(f"内容生成LLM: {content_llm.model_name}")

    # 工作流规划LLM（复杂逻辑理解）
    planning_llm = get_workflow_planning_llm(config)
    print(f"工作流规划LLM: {planning_llm.model_name}")

    # LLM决策模型（SOP驱动迭代）
    decision_llm = get_llm_decision_llm(config)
    print(f"LLM决策模型: {decision_llm.model_name}")

    # 响应生成LLM（最终用户响应）
    response_llm = get_response_generation_llm(config)
    print(f"响应生成LLM: {response_llm.model_name}")
    print()

    # 3. 演示不同任务使用不同模型
    print("=== 任务执行示例 ===")

    # 任务1: 意图识别（使用快速模型）
    print("\n[任务1] 意图识别")
    intent_result = await intent_llm.ainvoke([
        {"role": "user", "content": "帮我查询订单状态"}
    ])
    print(f"识别结果: {intent_result.content[:100]}...")

    # 任务2: 内容生成（使用高质量模型）
    print("\n[任务2] 内容生成")
    content_result = await content_llm.ainvoke([
        {"role": "user", "content": "写一篇关于AI技术发展的文章摘要"}
    ])
    print(f"生成内容: {content_result.content[:100]}...")

    # 任务3: 工作流规划（使用复杂逻辑理解模型）
    print("\n[任务3] 工作流规划")
    planning_result = await planning_llm.ainvoke([
        {"role": "user", "content": "分析用户请求并制定执行计划"}
    ])
    print(f"规划结果: {planning_result.content[:100]}...")

    print("\n=== 完成 ===")


async def demonstrate_cost_optimization():
    """演示成本优化策略"""

    print("\n=== 成本优化策略 ===\n")

    config = load_config()

    # 策略1: 意图识别使用便宜的模型
    print("策略1: 意图识别使用 gpt-4o-mini（快速、低成本）")
    print("  - 适用场景: 快速判断用户意图、规则匹配")
    print("  - 成本节省: 相比 gpt-4o 节省约 90% 成本")
    print()

    # 策略2: 内容生成使用高质量模型
    print("策略2: 内容生成使用 gpt-4o（高质量）")
    print("  - 适用场景: 生成用户可见的内容、复杂推理")
    print("  - 质量保证: 确保最终输出质量")
    print()

    # 策略3: 工作流规划使用平衡模型
    print("策略3: 工作流规划使用 gpt-4o（复杂逻辑理解）")
    print("  - 适用场景: 理解复杂SOP、制定执行计划")
    print("  - 平衡考虑: 准确性和成本的平衡")
    print()

    # 策略4: 混合使用不同提供商
    print("策略4: 混合使用不同提供商")
    print("  - OpenAI: gpt-4o-mini (意图识别)")
    print("  - Anthropic: claude-3-5-sonnet (内容生成)")
    print("  - Google: gemini-1.5-flash (工作流规划)")
    print("  - 优势: 利用各家模型的优势，避免单点依赖")
    print()


async def demonstrate_performance_optimization():
    """演示性能优化策略"""

    print("\n=== 性能优化策略 ===\n")

    # 策略1: 并行执行
    print("策略1: 并行执行多个任务")
    print("  - 意图识别和KB查询可以并行执行")
    print("  - 减少总体响应时间")
    print()

    # 策略2: 缓存策略
    print("策略2: 缓存常见意图识别结果")
    print("  - 对于常见问题，直接返回缓存结果")
    print("  - 避免重复调用LLM")
    print()

    # 策略3: 流式响应
    print("策略3: 使用流式响应提升用户体验")
    print("  - 对于内容生成任务，使用流式输出")
    print("  - 用户可以更快看到响应")
    print()


if __name__ == "__main__":
    # 运行主示例
    asyncio.run(main())

    # 演示优化策略
    asyncio.run(demonstrate_cost_optimization())
    asyncio.run(demonstrate_performance_optimization())
