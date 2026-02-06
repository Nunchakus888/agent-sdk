"""
V1 Config 路由

提供配置缓存记录的查看和差异对比 API。
展示 configs 表中的缓存记录，支持对比原始配置与 LLM 增强后的配置差异。

端点：
- GET /config                       — 配置缓存列表
- GET /config/{chatbot_id}          — 配置详情
- GET /config/{chatbot_id}/diff     — 原始 vs 增强 差异对比
"""

import math

from fastapi import APIRouter, HTTPException, Query, status

from api.core.logging import get_logger

logger = get_logger(__name__)


def _compute_diff(raw: dict, parsed: dict, path: str = "") -> list[dict]:
    """
    递归对比两个 dict，返回差异列表。

    每个差异项: {path, type, raw_value, parsed_value}
    type: added | removed | changed
    """
    diffs = []
    all_keys = set(list(raw.keys()) + list(parsed.keys()))

    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key
        in_raw = key in raw
        in_parsed = key in parsed

        if in_raw and not in_parsed:
            diffs.append({
                "path": current_path,
                "type": "removed",
                "raw_value": raw[key],
                "parsed_value": None,
            })
        elif not in_raw and in_parsed:
            diffs.append({
                "path": current_path,
                "type": "added",
                "raw_value": None,
                "parsed_value": parsed[key],
            })
        else:
            raw_val = raw[key]
            parsed_val = parsed[key]
            if isinstance(raw_val, dict) and isinstance(parsed_val, dict):
                diffs.extend(_compute_diff(raw_val, parsed_val, current_path))
            elif raw_val != parsed_val:
                diffs.append({
                    "path": current_path,
                    "type": "changed",
                    "raw_value": raw_val,
                    "parsed_value": parsed_val,
                })

    return diffs


def create_router() -> APIRouter:
    """创建 V1 Config 路由"""
    router = APIRouter(prefix="/config", tags=["Config"])

    def get_deps():
        from api.container import get_repository_manager
        return get_repository_manager

    @router.get(
        "",
        summary="配置缓存列表",
        description="获取所有缓存的配置记录，包含访问次数和时间信息",
    )
    async def list_configs(
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    ):
        get_repository_manager = get_deps()
        repos = get_repository_manager()

        total = await repos.configs.count()
        offset = (page - 1) * page_size
        configs = await repos.configs.list_all(limit=page_size, offset=offset)

        items = []
        for c in configs:
            items.append({
                "chatbot_id": c.chatbot_id,
                "tenant_id": c.tenant_id,
                "config_hash": c.config_hash,
                "version": c.version,
                "access_count": c.access_count,
                "created_at": c.created_at.isoformat() if c.created_at else "",
                "updated_at": c.updated_at.isoformat() if c.updated_at else "",
                "has_raw_config": bool(c.raw_config),
                "has_parsed_config": bool(c.parsed_config),
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": math.ceil(total / page_size) if total > 0 else 1,
        }

    @router.get(
        "/{chatbot_id}",
        summary="配置详情",
        description="获取指定 chatbot 的完整配置（原始和解析后）",
    )
    async def get_config(chatbot_id: str):
        get_repository_manager = get_deps()
        repos = get_repository_manager()

        config = await repos.configs.get_by_chatbot_id(chatbot_id)

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "ConfigNotFound", "message": f"Config not found: {chatbot_id}"},
            )

        return {
            "chatbot_id": config.chatbot_id,
            "tenant_id": config.tenant_id,
            "config_hash": config.config_hash,
            "version": config.version,
            "access_count": config.access_count,
            "created_at": config.created_at.isoformat() if config.created_at else "",
            "updated_at": config.updated_at.isoformat() if config.updated_at else "",
            "raw_config": config.raw_config,
            "parsed_config": config.parsed_config,
        }

    @router.get(
        "/{chatbot_id}/diff",
        summary="配置差异对比",
        description="对比原始配置与 LLM 增强后的配置差异",
    )
    async def get_config_diff(chatbot_id: str):
        get_repository_manager = get_deps()
        repos = get_repository_manager()

        config = await repos.configs.get_by_chatbot_id(chatbot_id)

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "ConfigNotFound", "message": f"Config not found: {chatbot_id}"},
            )

        raw = config.raw_config or {}
        parsed = config.parsed_config or {}
        diffs = _compute_diff(raw, parsed)

        return {
            "chatbot_id": config.chatbot_id,
            "tenant_id": config.tenant_id,
            "config_hash": config.config_hash,
            "diff_count": len(diffs),
            "diffs": diffs,
            "raw_config": raw,
            "parsed_config": parsed,
        }

    return router
