"""
时间处理工具函数

最佳实践：
- 存储：始终使用 UTC
- 传输：使用 ISO 8601 格式
- 显示：转换为用户时区

详见：docs/architecture/datetime-best-practices.md
"""

from datetime import datetime, timezone, timedelta
from typing import Optional


def utc_now() -> datetime:
    """
    获取当前 UTC 时间（带时区）

    Returns:
        带 UTC 时区的 datetime 对象

    Example:
        >>> now = utc_now()
        >>> now.tzinfo
        datetime.timezone.utc
    """
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    确保 datetime 有 UTC 时区信息

    MongoDB 返回的 datetime 是 naive（无时区），需要添加 UTC 时区。

    Args:
        dt: datetime 对象，可能有或没有时区信息

    Returns:
        带 UTC 时区的 datetime，或 None

    Example:
        >>> naive_dt = datetime(2026, 1, 1, 12, 0, 0)
        >>> aware_dt = ensure_utc(naive_dt)
        >>> aware_dt.tzinfo
        datetime.timezone.utc
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    转换为 ISO 8601 格式字符串

    Args:
        dt: datetime 对象

    Returns:
        ISO 8601 格式字符串，如 "2026-02-03T10:30:00+00:00"

    Example:
        >>> dt = datetime(2026, 2, 3, 10, 30, 0, tzinfo=timezone.utc)
        >>> to_iso(dt)
        '2026-02-03T10:30:00+00:00'
    """
    if dt is None:
        return None
    return ensure_utc(dt).isoformat()


def from_iso(s: Optional[str]) -> Optional[datetime]:
    """
    从 ISO 8601 格式字符串解析

    支持 "Z" 后缀和 "+00:00" 格式。

    Args:
        s: ISO 8601 格式字符串

    Returns:
        带 UTC 时区的 datetime 对象

    Example:
        >>> from_iso("2026-02-03T10:30:00Z")
        datetime.datetime(2026, 2, 3, 10, 30, tzinfo=datetime.timezone.utc)
    """
    if s is None:
        return None
    # 处理 "Z" 后缀
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return ensure_utc(dt)


def is_expired(dt: datetime, ttl_seconds: int) -> bool:
    """
    检查时间是否已过期

    Args:
        dt: 创建时间
        ttl_seconds: 生存时间（秒）

    Returns:
        True 如果已过期

    Example:
        >>> old_time = utc_now() - timedelta(hours=2)
        >>> is_expired(old_time, ttl_seconds=3600)  # 1小时 TTL
        True
    """
    return (utc_now() - ensure_utc(dt)).total_seconds() > ttl_seconds


def format_duration(seconds: float) -> str:
    """
    格式化持续时间为人类可读格式

    Args:
        seconds: 秒数

    Returns:
        格式化字符串，如 "1h 30m 45s" 或 "500ms"

    Example:
        >>> format_duration(5432.5)
        '1h 30m 32s'
        >>> format_duration(0.5)
        '500ms'
    """
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"

    parts = []
    if seconds >= 3600:
        hours = int(seconds // 3600)
        seconds %= 3600
        parts.append(f"{hours}h")
    if seconds >= 60:
        minutes = int(seconds // 60)
        seconds %= 60
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{int(seconds)}s")

    return " ".join(parts)
