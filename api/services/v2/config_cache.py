"""
服务级配置缓存

职责：
- 按 config_hash 缓存解析后的配置
- 多个 Agent 实例复用同一配置
- LRU 淘汰策略
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from bu_agent_sdk.tools.actions import WorkflowConfigSchema

logger = logging.getLogger(__name__)


@dataclass
class CachedConfig:
    """缓存的配置"""
    config: WorkflowConfigSchema
    config_hash: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_access_at: float = field(default_factory=time.time)

    def touch(self):
        """更新访问时间和计数"""
        self.access_count += 1
        self.last_access_at = time.time()


class ConfigCache:
    """
    服务级配置缓存

    特性：
    - 按 config_hash 缓存解析后的 WorkflowConfigSchema
    - 多个 SessionContext 复用同一配置
    - LRU 淘汰策略
    - TTL 过期机制
    """

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Args:
            max_size: 最大缓存数量
            ttl: 缓存过期时间（秒）
        """
        self._cache: dict[str, CachedConfig] = {}
        self._max_size = max_size
        self._ttl = ttl
        logger.info(f"ConfigCache initialized: max_size={max_size}, ttl={ttl}s")

    def get(self, config_hash: str) -> Optional[WorkflowConfigSchema]:
        """
        获取缓存的配置

        Args:
            config_hash: 配置哈希

        Returns:
            WorkflowConfigSchema 或 None
        """
        cached = self._cache.get(config_hash)
        if cached is None:
            return None

        # 检查 TTL
        if time.time() - cached.created_at > self._ttl:
            del self._cache[config_hash]
            logger.debug(f"Config expired: {config_hash[:12]}")
            return None

        cached.touch()
        logger.debug(f"Config cache HIT: {config_hash[:12]}")
        return cached.config

    def set(self, config_hash: str, config: WorkflowConfigSchema):
        """
        缓存配置

        Args:
            config_hash: 配置哈希
            config: 解析后的配置
        """
        # LRU 淘汰
        if len(self._cache) >= self._max_size:
            self._evict_lru()

        self._cache[config_hash] = CachedConfig(
            config=config,
            config_hash=config_hash,
        )
        logger.debug(f"Config cached: {config_hash[:12]}")

    def invalidate(self, config_hash: str) -> bool:
        """
        使配置失效

        Args:
            config_hash: 配置哈希

        Returns:
            是否成功删除
        """
        if config_hash in self._cache:
            del self._cache[config_hash]
            logger.debug(f"Config invalidated: {config_hash[:12]}")
            return True
        return False

    def clear(self):
        """清空所有缓存"""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Config cache cleared: {count} entries")

    def _evict_lru(self):
        """淘汰最少使用的配置"""
        if not self._cache:
            return

        # 按 last_access_at 排序，淘汰最旧的
        lru_key = min(
            self._cache,
            key=lambda k: self._cache[k].last_access_at
        )
        del self._cache[lru_key]
        logger.debug(f"Config evicted (LRU): {lru_key[:12]}")

    @property
    def size(self) -> int:
        """当前缓存数量"""
        return len(self._cache)

    def get_stats(self) -> dict:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl": self._ttl,
            "entries": [
                {
                    "config_hash": c.config_hash[:12],
                    "access_count": c.access_count,
                    "age_seconds": int(time.time() - c.created_at),
                }
                for c in self._cache.values()
            ],
        }
