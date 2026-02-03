# 时间格式最佳实践

> 避免时区问题的统一时间处理规范

## 字段命名规范

### 时间字段命名约定

| 字段名 | 含义 | 示例 |
|--------|------|------|
| `created_at` | 记录创建时间 | 会话创建、消息创建 |
| `updated_at` | 记录更新时间 | 会话更新、配置更新 |
| `closed_at` | 关闭/结束时间 | 会话关闭 |
| `finalized_at` | 完成/定稿时间 | Token 统计完成 |
| `last_active_at` | 最后活跃时间 | 会话最后活动 |
| `next_trigger_at` | 下次触发时间 | Timer 下次触发 |
| `last_triggered_at` | 上次触发时间 | Timer 上次触发 |
| `start_time` | 开始时间 | 事件开始 |
| `end_time` | 结束时间 | 事件结束 |
| `timestamp` | 时间戳 | 通用时间点 |

### 命名规则

1. **使用 `_at` 后缀**：表示时间点的字段使用 `_at` 后缀
   - `created_at`, `updated_at`, `closed_at`

2. **使用 `_time` 后缀**：表示事件时间的字段使用 `_time` 后缀
   - `start_time`, `end_time`

3. **避免模糊命名**：
   - ❌ `time`, `date`, `datetime`
   - ✅ `created_at`, `start_time`, `timestamp`

4. **持续时间使用 `_ms` 或 `_seconds`**：
   - `duration_ms`: 毫秒
   - `idle_seconds`: 秒
   - `ttl_seconds`: 秒

---

## 核心原则

### 1. 存储：始终使用 UTC

```python
from datetime import datetime, timezone

# ✅ 正确：使用 UTC
created_at = datetime.now(timezone.utc)

# ❌ 错误：使用本地时间（有时区歧义）
created_at = datetime.now()

# ❌ 错误：使用 utcnow()（已废弃，返回 naive datetime）
created_at = datetime.utcnow()
```

### 2. 传输：使用 ISO 8601 格式

```python
# ✅ 正确：带时区的 ISO 8601
"2026-02-03T10:30:00+00:00"
"2026-02-03T10:30:00Z"

# ❌ 错误：无时区信息
"2026-02-03T10:30:00"
"2026-02-03 10:30:00"
```

### 3. 显示：转换为用户时区

```python
from datetime import timezone, timedelta

# 存储的 UTC 时间
utc_time = datetime(2026, 2, 3, 10, 30, 0, tzinfo=timezone.utc)

# 转换为用户时区（如 UTC+8）
user_tz = timezone(timedelta(hours=8))
local_time = utc_time.astimezone(user_tz)
# 结果：2026-02-03 18:30:00+08:00
```

---

## 实现规范

### MongoDB 存储

MongoDB 内部以 UTC 存储 `datetime`，但默认返回 naive datetime（无时区）。

**关键配置**：在创建 MongoDB 客户端时启用时区感知：

```python
from datetime import timezone
from motor.motor_asyncio import AsyncIOMotorClient

# ✅ 正确：配置时区感知
client = AsyncIOMotorClient(
    mongodb_uri,
    tz_aware=True,           # 返回带时区的 datetime
    tzinfo=timezone.utc,     # 使用 UTC 时区
)

# ❌ 错误：默认配置（返回 naive datetime）
client = AsyncIOMotorClient(mongodb_uri)
```

**存储时**：确保传入 UTC 时间

```python
from api.utils.datetime import utc_now

doc = {
    "created_at": utc_now(),  # 带 UTC 时区
}
await collection.insert_one(doc)
```

**读取时**：如果客户端配置了 `tz_aware=True`，返回的 datetime 自动带 UTC 时区

```python
doc = await collection.find_one({"_id": "xxx"})
# doc["created_at"] 已经是 timezone-aware datetime

# 如果未配置 tz_aware，需要手动添加时区
from api.utils.datetime import ensure_utc
created_at = ensure_utc(doc["created_at"])
```

**注意**：MongoDB BSON Date 类型只支持毫秒精度，微秒会被截断。

### Pydantic 模型

```python
from datetime import datetime, timezone
from pydantic import BaseModel, field_validator

class DocumentBase(BaseModel):
    created_at: datetime
    updated_at: datetime | None = None

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def ensure_timezone(cls, v):
        if v is None:
            return None
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
```

### FastAPI 响应

```python
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
import json

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            # 确保有时区，然后输出 ISO 格式
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()
        return super().default(obj)
```

---

## 常见场景

### 1. 创建记录

```python
from datetime import datetime, timezone

def create_document():
    return {
        "created_at": datetime.now(timezone.utc),
        "updated_at": None,
    }
```

### 2. 更新记录

```python
def update_document(doc: dict):
    doc["updated_at"] = datetime.now(timezone.utc)
    return doc
```

### 3. 时间比较

```python
from datetime import datetime, timezone, timedelta

def is_expired(created_at: datetime, ttl_seconds: int) -> bool:
    """检查是否过期"""
    # 确保 created_at 有时区
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return (now - created_at).total_seconds() > ttl_seconds
```

### 4. 时间范围查询

```python
from datetime import datetime, timezone, timedelta

def get_recent_records(hours: int = 24):
    """获取最近 N 小时的记录"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return collection.find({"created_at": {"$gte": cutoff}})
```

### 5. 前端显示

```javascript
// 前端接收 ISO 8601 格式
const isoString = "2026-02-03T10:30:00+00:00";

// 转换为本地时间显示
const date = new Date(isoString);
const localString = date.toLocaleString(); // 自动转换为用户时区
```

---

## 工具函数

```python
# api/utils/datetime.py

from datetime import datetime, timezone, timedelta
from typing import Optional

def utc_now() -> datetime:
    """获取当前 UTC 时间（带时区）"""
    return datetime.now(timezone.utc)

def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """确保 datetime 有 UTC 时区信息"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """转换为 ISO 8601 格式字符串"""
    if dt is None:
        return None
    return ensure_utc(dt).isoformat()

def from_iso(s: Optional[str]) -> Optional[datetime]:
    """从 ISO 8601 格式字符串解析"""
    if s is None:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return ensure_utc(dt)

def is_expired(dt: datetime, ttl_seconds: int) -> bool:
    """检查时间是否已过期"""
    return (utc_now() - ensure_utc(dt)).total_seconds() > ttl_seconds
```

---

## 检查清单

- [x] 所有 `datetime.now()` 调用都使用 `utc_now()` 工具函数
- [x] 不使用已废弃的 `datetime.utcnow()`
- [x] 从 DB 读取的 datetime 都使用 `ensure_utc()` 添加 UTC 时区
- [ ] API 响应使用 ISO 8601 格式
- [x] 时间比较前确保两边都有时区信息
- [ ] 前端显示时转换为用户本地时区
- [x] 时间字段命名遵循 `_at` / `_time` 后缀规范

---

## 参考

- [Python datetime 文档](https://docs.python.org/3/library/datetime.html)
- [ISO 8601 标准](https://en.wikipedia.org/wiki/ISO_8601)
- [MongoDB 日期处理](https://www.mongodb.com/docs/manual/reference/bson-types/#date)
