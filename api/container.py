"""
应用上下文容器

设计原则：
- 使用 AppContext 类封装所有服务实例，避免全局变量散落
- 单例模式通过类变量实现，保证全局唯一
- 清晰的生命周期管理（create / shutdown）
- 与 FastAPI 依赖注入系统兼容
- 易于测试（可以创建独立的上下文实例）

架构简化：
- 移除 ConfigLoader 内存缓存层
- 配置加载由 SessionManager 内部处理（DB → HTTP）
- Session 存在直接返回，不存在才加载配置
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Annotated, Any, TYPE_CHECKING

from fastapi import Depends

from api.services.database import DB_NAME, Database, get_database as create_database
from api.services.repositories import RepositoryManager, create_repository_manager
from api.services.llm_service import LLMService

if TYPE_CHECKING:
    from api.services.v2 import SessionManager

logger = logging.getLogger(__name__)


# =============================================================================
# 配置
# =============================================================================


@dataclass
class AppConfig:
    """应用配置"""

    # Database
    mongodb_uri: str | None = None
    mongodb_db: str = DB_NAME

    # SessionManager
    idle_timeout: int = 1800
    cleanup_interval: int = 60
    max_sessions: int = 10000
    enable_llm_parsing: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量创建配置"""
        return cls(
            mongodb_uri=os.getenv("MONGODB_URI"),
            mongodb_db=os.getenv("MONGODB_DB", DB_NAME),
            idle_timeout=int(os.getenv("SESSION_IDLE_TIMEOUT", "1800")),
            cleanup_interval=int(os.getenv("SESSION_CLEANUP_INTERVAL", "60")),
            max_sessions=int(os.getenv("MAX_SESSIONS", "10000")),
            enable_llm_parsing=os.getenv("ENABLE_LLM_PARSING", "").lower() == "true",
        )


# =============================================================================
# 应用上下文
# =============================================================================


@dataclass
class AppContext:
    """
    应用上下文容器

    封装所有服务实例，提供统一的生命周期管理。

    Usage:
        ```python
        ctx = await AppContext.create()
        await ctx.session_manager.start()

        # 使用服务（配置加载由 SessionManager 内部处理）
        session = await ctx.session_manager.get_or_create(
            session_id="...",
            tenant_id="...",
            chatbot_id="...",
            config_hash="...",
        )

        # 关闭
        await ctx.shutdown()
        ```
    """

    # 基础设施
    mongo_client: Any = field(default=None, repr=False)
    database: Database | None = None
    repository_manager: RepositoryManager | None = None

    # 业务服务
    session_manager: "SessionManager | None" = None

    # 单例
    _instance: "AppContext | None" = field(default=None, init=False, repr=False)

    @classmethod
    async def create(cls, config: AppConfig | None = None) -> "AppContext":
        """创建并初始化应用上下文"""
        if cls._instance is not None:
            return cls._instance

        config = config or AppConfig.from_env()
        ctx = cls()

        # 初始化服务
        LLMService.initialize()
        await ctx._init_database(config)
        ctx._init_repository_manager()
        ctx._init_services(config)

        cls._instance = ctx
        logger.info("AppContext initialized")
        return ctx

    async def _init_database(self, config: AppConfig) -> None:
        """初始化数据库"""
        if not config.mongodb_uri:
            logger.info("Database: memory mode (set MONGODB_URI to enable MongoDB)")
            return

        try:
            from datetime import timezone
            from motor.motor_asyncio import AsyncIOMotorClient
            from bson.codec_options import CodecOptions

            logger.info(f"Connecting to MongoDB: {config.mongodb_uri}")

            # 配置时区感知：读取时返回带 UTC 时区的 datetime
            # 详见 docs/architecture/datetime-best-practices.md
            self.mongo_client = AsyncIOMotorClient(
                config.mongodb_uri,
                serverSelectionTimeoutMS=3000,  # 3秒超时，快速失败
                tz_aware=True,                  # 返回带时区的 datetime
                tzinfo=timezone.utc,            # 使用 UTC 时区
            )
            await self.mongo_client.admin.command("ping")

            self.database = create_database(self.mongo_client, config.mongodb_db)
            if self.database:
                await self.database.ensure_indexes()
                logger.info(f"Database: mongodb/{config.mongodb_db}")

        except Exception as e:
            logger.warning(
                f"MongoDB connection failed ({config.mongodb_uri}): {e}. "
                f"Falling back to memory mode. Set DISABLE_MONGODB=true to suppress this warning."
            )
            self.mongo_client = None
            self.database = None

    def _init_repository_manager(self) -> None:
        """初始化 RepositoryManager"""
        self.repository_manager = create_repository_manager(self.database)

    def _init_services(self, config: AppConfig) -> None:
        """初始化业务服务"""
        from api.services.v2 import SessionManager

        self.session_manager = SessionManager(
            repos=self.repository_manager,
            database=self.database,
            llm_provider=LLMService.get_instance(),
            idle_timeout=config.idle_timeout,
            cleanup_interval=config.cleanup_interval,
            max_sessions=config.max_sessions,
            enable_llm_parsing=config.enable_llm_parsing,
        )

    async def shutdown(self) -> None:
        """关闭所有服务"""
        if self.session_manager:
            await self.session_manager.stop()
            self.session_manager = None

        if self.mongo_client:
            self.mongo_client.close()
            self.mongo_client = None

        AppContext._instance = None
        logger.info("AppContext shutdown")

    @classmethod
    def get_instance(cls) -> "AppContext":
        """获取单例"""
        if cls._instance is None:
            raise RuntimeError("AppContext not initialized")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例（测试用）"""
        cls._instance = None


# =============================================================================
# FastAPI 依赖注入
# =============================================================================

from api.services.v2 import SessionManager


def get_app_context() -> AppContext:
    return AppContext.get_instance()


def get_database() -> Database | None:
    return AppContext.get_instance().database


def get_repository_manager() -> RepositoryManager:
    ctx = AppContext.get_instance()
    if ctx.repository_manager is None:
        raise RuntimeError("RepositoryManager not initialized")
    return ctx.repository_manager


def get_session_manager() -> SessionManager:
    ctx = AppContext.get_instance()
    if ctx.session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return ctx.session_manager


def get_llm_service() -> LLMService:
    return LLMService.get_instance()


# =============================================================================
# 依赖类型别名
# =============================================================================

AppContextDep = Annotated[AppContext, Depends(get_app_context)]
DatabaseDep = Annotated[Database | None, Depends(get_database)]
RepositoryManagerDep = Annotated[RepositoryManager, Depends(get_repository_manager)]
SessionManagerDep = Annotated[SessionManager, Depends(get_session_manager)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
