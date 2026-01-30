"""
LLM Service - 模型服务单例

提供统一的 LLM 访问接口，支持：
- 服务级别单例，复用 HTTP 客户端连接
- 任务特定模型选择（意图识别、内容生成、工作流规划等）
- 统一的配置管理
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bu_agent_sdk.llm.base import BaseChatModel
from bu_agent_sdk.llm.openai.chat import ChatOpenAI


class ModelTask(str, Enum):
    """模型任务类型"""
    DEFAULT = "default"
    INTENT_MATCHING = "intent_matching"
    CONTENT_GENERATION = "content_generation"
    WORKFLOW_PLANNING = "workflow_planning"
    LLM_DECISION = "llm_decision"
    RESPONSE_GENERATION = "response_generation"


@dataclass
class LLMConfig:
    """LLM 配置"""
    api_key: str | None = None
    base_url: str | None = None
    default_model: str = "gpt-4o"
    # 任务特定模型
    intent_matching_model: str | None = None
    content_generation_model: str | None = None
    workflow_planning_model: str | None = None
    llm_decision_model: str | None = None
    response_generation_model: str | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            default_model=os.getenv("DEFAULT_MODEL", "gpt-4o"),
            intent_matching_model=os.getenv("INTENT_MATCHING_MODEL"),
            content_generation_model=os.getenv("CONTENT_GENERATION_MODEL"),
            workflow_planning_model=os.getenv("WORKFLOW_PLANNING_MODEL"),
            llm_decision_model=os.getenv("LLM_DECISION_MODEL"),
            response_generation_model=os.getenv("RESPONSE_GENERATION_MODEL"),
        )


class LLMService:
    """
    LLM 服务单例

    特性：
    - 服务级别单例，应用生命周期内复用
    - 按任务类型缓存 LLM 实例，复用 HTTP 连接
    - 支持任务特定模型配置

    Usage:
        ```python
        # 初始化（应用启动时）
        service = LLMService.initialize()

        # 获取实例
        service = LLMService.get_instance()

        # 获取特定任务的 LLM
        llm = service.get_llm(ModelTask.INTENT_MATCHING)
        response = await llm.ainvoke(messages)
        ```
    """

    _instance: "LLMService | None" = None

    def __init__(self, config: LLMConfig):
        self._config = config
        self._llm_cache: dict[str, BaseChatModel] = {}

    @classmethod
    def initialize(cls, config: LLMConfig | None = None) -> "LLMService":
        """初始化服务（应用启动时调用）"""
        if cls._instance is not None:
            return cls._instance

        config = config or LLMConfig.from_env()
        cls._instance = cls(config)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LLMService":
        """获取服务实例"""
        if cls._instance is None:
            raise RuntimeError(
                "LLMService not initialized. Call LLMService.initialize() first."
            )
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """关闭服务"""
        if cls._instance is not None:
            cls._instance._llm_cache.clear()
            cls._instance = None

    def get_model_name(self, task: ModelTask = ModelTask.DEFAULT) -> str:
        """获取任务对应的模型名称"""
        task_model_map = {
            ModelTask.INTENT_MATCHING: self._config.intent_matching_model,
            ModelTask.CONTENT_GENERATION: self._config.content_generation_model,
            ModelTask.WORKFLOW_PLANNING: self._config.workflow_planning_model,
            ModelTask.LLM_DECISION: self._config.llm_decision_model,
            ModelTask.RESPONSE_GENERATION: self._config.response_generation_model,
        }
        return task_model_map.get(task) or self._config.default_model

    def get_llm(self, task: ModelTask = ModelTask.DEFAULT) -> BaseChatModel:
        """
        获取指定任务的 LLM 实例

        相同模型名称复用同一实例，避免重复创建 HTTP 客户端
        """
        model = self.get_model_name(task)

        # 缓存命中
        if model in self._llm_cache:
            return self._llm_cache[model]

        # 创建新实例
        llm = ChatOpenAI(
            model=model,
            api_key=self._config.api_key,
            base_url=self._config.base_url,
        )
        self._llm_cache[model] = llm
        return llm

    # 便捷方法
    def get_default_llm(self) -> BaseChatModel:
        """获取默认 LLM"""
        return self.get_llm(ModelTask.DEFAULT)

    def get_intent_llm(self) -> BaseChatModel:
        """获取意图识别 LLM"""
        return self.get_llm(ModelTask.INTENT_MATCHING)

    def get_decision_llm(self) -> BaseChatModel:
        """获取决策 LLM"""
        return self.get_llm(ModelTask.LLM_DECISION)

    def get_response_llm(self) -> BaseChatModel:
        """获取响应生成 LLM"""
        return self.get_llm(ModelTask.RESPONSE_GENERATION)

    @property
    def config(self) -> LLMConfig:
        """获取配置"""
        return self._config

    def get_stats(self) -> dict[str, Any]:
        """获取服务统计信息"""
        return {
            "cached_models": list(self._llm_cache.keys()),
            "default_model": self._config.default_model,
            "task_models": {
                "intent_matching": self._config.intent_matching_model,
                "content_generation": self._config.content_generation_model,
                "workflow_planning": self._config.workflow_planning_model,
                "llm_decision": self._config.llm_decision_model,
                "response_generation": self._config.response_generation_model,
            },
        }
