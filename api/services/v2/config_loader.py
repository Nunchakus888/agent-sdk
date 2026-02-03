"""
配置加载器

职责：配置的加载、LLM 增强解析、两级缓存管理
设计：L1 (内存) + L2 (DB) 两级缓存，HTTP 远程加载，可选 LLM 增强

时间处理：所有时间字段使用 UTC 时区
详见 docs/architecture/datetime-best-practices.md
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from bu_agent_sdk.tools.actions import WorkflowConfigSchema

from api.services.v2.config_cache import ConfigCache
from api.utils.datetime import utc_now, ensure_utc
from config.http_config import HttpConfigLoader, AgentConfigRequest

if TYPE_CHECKING:
    from api.services.database import Database

logger = logging.getLogger(__name__)


# =============================================================================
# 持久化数据模型
# =============================================================================


@dataclass
class StoredConfig:
    """持久化配置数据"""

    config_hash: str
    tenant_id: str
    chatbot_id: str
    raw_config: dict
    parsed_config: dict
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "_id": self.config_hash,
            "tenant_id": self.tenant_id,
            "chatbot_id": self.chatbot_id,
            "raw_config": self.raw_config,
            "parsed_config": self.parsed_config,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StoredConfig":
        return cls(
            config_hash=data.get("_id") or data.get("config_hash"),
            tenant_id=data["tenant_id"],
            chatbot_id=data["chatbot_id"],
            raw_config=data["raw_config"],
            parsed_config=data["parsed_config"],
            created_at=ensure_utc(data.get("created_at")) or utc_now(),
        )


# =============================================================================
# 配置加载器
# =============================================================================


class ConfigLoader:
    """
    配置加载器

    两级缓存架构：
    - L1: 内存缓存 (ConfigCache)，进程内快速访问
    - L2: Database.configs，跨进程/重启持久化

    加载流程：
    1. L1 命中 → 直接返回
    2. L2 命中 → 回填 L1，返回
    3. 都未命中 → HTTP 加载 → 解析 → 存储 L2 → 存储 L1 → 返回
    """

    def __init__(
        self,
        database: "Database | None" = None,
        cache_size: int = 100,
        cache_ttl: int = 3600,
        enable_llm_parsing: bool = False,
    ):
        """
        Args:
            database: Database 实例（None 则仅使用 L1 内存缓存）
            cache_size: L1 缓存大小
            cache_ttl: L1 缓存 TTL（秒）
            enable_llm_parsing: 是否启用 LLM 增强解析
        """
        self._db = database
        self._l1 = ConfigCache(max_size=cache_size, ttl=cache_ttl)
        self._http = HttpConfigLoader(logger)
        self._enable_llm_parsing = enable_llm_parsing

    async def load(
        self,
        config_hash: str,
        tenant_id: str,
        chatbot_id: str,
    ) -> WorkflowConfigSchema:
        """
        加载配置（两级缓存）

        Args:
            config_hash: 配置哈希（客户端提供）
            tenant_id: 租户 ID
            chatbot_id: 机器人 ID

        Returns:
            解析后的 WorkflowConfigSchema

        Raises:
            ValueError: 配置加载失败
        """
        # L1: 内存缓存
        config = self._l1.get(config_hash)
        if config:
            logger.debug(f"Config L1 HIT: {config_hash[:12]}")
            return config

        # L2: Database
        if self._db:
            doc = await self._db.configs.find_one({"_id": config_hash})
            if doc:
                logger.debug(f"Config L2 HIT: {config_hash[:12]}")
                stored = StoredConfig.from_dict(doc)
                config = WorkflowConfigSchema(**stored.parsed_config)
                self._l1.set(config_hash, config)
                return config

        # L1 + L2 都未命中，从 HTTP 加载
        logger.info(f"Config MISS, loading from HTTP: {config_hash[:12]}")
        return await self._load_from_http(config_hash, tenant_id, chatbot_id)

    async def _load_from_http(
        self,
        config_hash: str,
        tenant_id: str,
        chatbot_id: str,
    ) -> WorkflowConfigSchema:
        """从 HTTP 加载配置并存储到两级缓存"""
        raw_config = await self._http.load_config_from_http(
            AgentConfigRequest(tenant_id=tenant_id, chatbot_id=chatbot_id)
        )

        config = await self._parse_config(raw_config, config_hash)

        # 存储到 L2 (Database)
        if self._db:
            stored = StoredConfig(
                config_hash=config_hash,
                tenant_id=tenant_id,
                chatbot_id=chatbot_id,
                raw_config=raw_config,
                parsed_config=config.model_dump(),
            )
            await self._db.configs.replace_one(
                {"_id": config_hash},
                stored.to_dict(),
                upsert=True,
            )
            logger.debug(f"Config stored to L2: {config_hash[:12]}")

        # 存储到 L1
        self._l1.set(config_hash, config)

        return config

    async def _parse_config(
        self, raw_config: dict, config_hash: str
    ) -> WorkflowConfigSchema:
        """解析配置"""
        # KB 配置
        kb_config = raw_config.get("kb_config", {})
        if not kb_config and raw_config.get("retrieve_knowledge_url"):
            kb_config = {
                "enabled": True,
                "retrieve_url": raw_config["retrieve_knowledge_url"],
                "chatbot_id": raw_config.get("basic_settings", {}).get("chatbot_id", ""),
                "auto_retrieve": True,
            }

        # 环境变量覆盖
        max_iterations = int(
            os.getenv("MAX_ITERATIONS", raw_config.get("max_iterations", 5))
        )
        iteration_strategy = os.getenv(
            "ITERATION_STRATEGY", raw_config.get("iteration_strategy", "sop_driven")
        )

        # LLM 增强解析
        llm_parsed = await self._llm_enhance(raw_config)

        # 合并配置
        final_config = {
            "kb_config": kb_config,
            "max_iterations": max_iterations,
            "iteration_strategy": iteration_strategy,
            **llm_parsed,
            "system_actions": raw_config.get("system_actions"),
            "agent_actions": raw_config.get("agent_actions"),
        }

        logger.info(
            f"Config parsed: hash={config_hash[:12]}, "
            f"kb={bool(kb_config)}, llm={self._enable_llm_parsing}"
        )

        return WorkflowConfigSchema(**final_config)

    async def _llm_enhance(self, raw_config: dict) -> dict:
        """LLM 增强解析"""
        if not self._enable_llm_parsing:
            return raw_config

        try:
            from api.services.llm_service import LLMService
            from bu_agent_sdk.llm.messages import UserMessage

            llm = LLMService.get_instance().get_decision_llm()
            prompt = self._build_prompt(raw_config)

            logger.info("Calling LLM for config enhancement...")
            response = await llm.ainvoke(messages=[UserMessage(content=prompt)])
            response_text = response.content or ""

            return self._parse_llm_response(response_text, raw_config)

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}, using original config")
            return raw_config

    def _build_prompt(self, raw_config: dict) -> str:
        """构建 LLM 增强 prompt"""
        basic = raw_config.get("basic_settings", {})
        instruction = basic.get("instruction", "")

        input_config = {
            "basic_settings": {
                "name": basic.get("name", ""),
                "description": basic.get("description", ""),
                "background": basic.get("background", ""),
                "language": basic.get("language", ""),
                "tone": basic.get("tone", ""),
            },
            "skills": raw_config.get("action_books", []),
            "tools": raw_config.get("tools"),
            "flows": raw_config.get("flows", []),
        }

        return f"""You are an AI configuration optimizer. Analyze and extract implicit configurations.

## INPUT CONFIGURATION
```json
{json.dumps(input_config, indent=2, ensure_ascii=False)}
```

## INSTRUCTION
<instruction>
{instruction}
</instruction>

## TASKS

1. **instructions**: Combine basic_settings and instruction into structured markdown
2. **timers**: Extract timeout/follow-up logic (delay_seconds, trigger, action, message)
3. **need_greeting**: Extract initial greeting message (preserve template variables)
4. **constraints**: Extract boundaries and rules

## OUTPUT FORMAT
Return ONLY valid JSON:
```json
{{
  "instructions": "...",
  "tools": [...],
  "flows": [...],
  "timers": [...] or null,
  "need_greeting": "..." or "",
  "constraints": "..." or null
}}
```"""

    def _parse_llm_response(self, response: str, fallback: dict) -> dict:
        """解析 LLM 响应"""
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            json_str = match.group(1) if match else response

            data = json.loads(json_str)
            data = self._clean_xml_tags(data)

            logger.info("LLM response parsed successfully")
            return data

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return fallback

    def _clean_xml_tags(self, config: dict) -> dict:
        """清理配置中的 XML 标签"""

        def clean(value):
            if isinstance(value, str):
                return re.sub(r"<[^>]+>(.*?)</[^>]+>", r"\1", value, flags=re.DOTALL).strip()
            elif isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean(item) for item in value]
            return value

        return clean(config)

    def invalidate(self, config_hash: str) -> bool:
        """使配置失效（仅 L1）"""
        return self._l1.invalidate(config_hash)

    def get_stats(self) -> dict:
        """获取缓存统计"""
        return {
            "l1": self._l1.get_stats(),
            "l2_enabled": self._db is not None,
            "llm_parsing": self._enable_llm_parsing,
        }
