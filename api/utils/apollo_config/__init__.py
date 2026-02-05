"""
Apollo Config Loader - 轻量级 Apollo 配置中心客户端

从 Apollo 配置中心加载配置并设置为环境变量，支持 YAML 和 KEY=VALUE 格式。

Usage:
    from apollo_config_loader import load_config, load_config_from_env

    # 方式1: 直接传参
    config = await load_config(
        server_url="http://apollo.example.com:8080",
        app_id="your-app-id"
    )

    # 方式2: 从环境变量读取连接参数
    config = await load_config_from_env()
"""

from .loader import (
    ApolloConfigLoader,
    load_config,
    load_config_from_env,
    ApolloConfigError,
)

__version__ = "1.0.0"
__all__ = [
    "ApolloConfigLoader",
    "load_config",
    "load_config_from_env",
    "ApolloConfigError",
]
