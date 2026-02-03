"""
配置数据模型

仅保留 StoredConfig 数据模型，用于 DB 持久化。

时间处理：所有时间字段使用 UTC 时区
详见 docs/architecture/datetime-best-practices.md
"""

from dataclasses import dataclass, field
from datetime import datetime

from api.utils.datetime import utc_now, ensure_utc


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
