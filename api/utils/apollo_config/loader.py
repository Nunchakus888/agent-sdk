"""
Apollo Config Loader - 核心实现

轻量级、异步的 Apollo 配置中心客户端。
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import aiohttp
import yaml


# Auto-load .env file for env config
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class ApolloConfigError(Exception):
    """Apollo 配置加载异常"""
    pass


@dataclass
class ApolloConfig:
    """Apollo 连接配置"""
    server_url: str
    app_id: str
    cluster: str = "default"
    namespace: str = "application.yaml"
    timeout: int = 30


class ApolloConfigLoader:
    """
    Apollo 配置加载器

    从 Apollo 配置中心获取配置并设置为环境变量。
    支持 YAML 和 KEY=VALUE 两种配置格式。
    """

    def __init__(self, config: ApolloConfig):
        self.config = config

    async def fetch(self) -> str:
        """
        从 Apollo API 获取原始配置内容

        Returns:
            配置内容字符串

        Raises:
            ApolloConfigError: 请求失败时抛出
        """
        url = (
            f"{self.config.server_url}/configs/"
            f"{self.config.app_id}/{self.config.cluster}/{self.config.namespace}"
        )
        logger.info(f"Fetching config from: {url}")

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise ApolloConfigError(
                            f"Apollo API returned {resp.status}"
                        )

                    data = await resp.json()
                    content = data.get("configurations", {}).get("content", "")

                    if not content:
                        raise ApolloConfigError("Empty config content")

                    return content

        except aiohttp.ClientError as e:
            raise ApolloConfigError(f"Network error: {e}") from e

    def parse(self, content: str) -> Dict[str, Any]:
        """
        解析配置内容

        支持两种格式:
        1. YAML 格式 (优先)
        2. KEY=VALUE 格式

        Args:
            content: 原始配置字符串

        Returns:
            解析后的配置字典
        """
        # 尝试 YAML
        try:
            config = yaml.safe_load(content)
            if isinstance(config, dict):
                logger.info("Parsed as YAML")
                return config
        except yaml.YAMLError:
            pass

        # 回退到 KEY=VALUE
        config = {}
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()

        logger.info("Parsed as KEY=VALUE")
        return config

    def apply_to_env(self, config: Dict[str, Any]) -> int:
        """
        将配置设置为环境变量

        Args:
            config: 配置字典

        Returns:
            设置的环境变量数量
        """
        count = 0
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ[key] = str(value)
                count += 1
                logger.debug(f"Set env: {key}")
        return count

    async def load(self, apply_env: bool = True) -> Dict[str, Any]:
        """
        加载配置的完整流程

        Args:
            apply_env: 是否设置为环境变量，默认 True

        Returns:
            配置字典
        """
        content = await self.fetch()
        config = self.parse(content)

        if apply_env:
            count = self.apply_to_env(config)
            logger.info(f"Applied {count} env variables")

        return config


async def load_config(
    server_url: str,
    app_id: str,
    cluster: str = "default",
    namespace: str = "application.yaml",
    timeout: int = 30,
    apply_env: bool = True,
) -> Dict[str, Any]:
    """
    便捷函数: 加载 Apollo 配置

    Args:
        server_url: Apollo 配置中心地址
        app_id: 应用 ID
        cluster: 集群名称，默认 "default"
        namespace: 命名空间，默认 "application.yaml"
        timeout: 超时时间（秒），默认 30
        apply_env: 是否设置为环境变量，默认 True

    Returns:
        配置字典

    Example:
        config = await load_config(
            server_url="http://apollo.example.com:8080",
            app_id="my-app"
        )
    """
    loader = ApolloConfigLoader(
        ApolloConfig(
            server_url=server_url,
            app_id=app_id,
            cluster=cluster,
            namespace=namespace,
            timeout=timeout,
        )
    )
    return await loader.load(apply_env=apply_env)


async def load_config_from_env(apply_env: bool = True) -> Optional[Dict[str, Any]]:
    """
    从环境变量读取 Apollo 连接参数并加载配置

    环境变量:
        APOLLO_CONFIG_SERVER_URL: Apollo 配置中心地址 (必需)
        APOLLO_APP_ID: 应用 ID (必需)
        APOLLO_CLUSTER: 集群名称 (可选，默认 "default")
        APOLLO_NAMESPACE: 命名空间 (可选，默认 "application.yaml")
        APOLLO_TIMEOUT: 超时时间秒 (可选，默认 30)

    Args:
        apply_env: 是否设置为环境变量，默认 True

    Returns:
        配置字典，未配置或加载失败时返回 None

    Example:
        # 先设置环境变量
        os.environ["APOLLO_CONFIG_SERVER_URL"] = "http://apollo.example.com:8080"
        os.environ["APOLLO_APP_ID"] = "my-app"

        # 然后加载
        config = await load_config_from_env()
    """
    server_url = os.environ.get("APOLLO_CONFIG_SERVER_URL")
    app_id = os.environ.get("APOLLO_APP_ID")

    # 未配置则跳过
    if not server_url or not app_id:
        raise ApolloConfigError(
            "Missing required env: APOLLO_CONFIG_SERVER_URL, APOLLO_APP_ID"
        )

    try:
        config = await load_config(
            server_url=server_url,
            app_id=app_id,
            cluster=os.environ.get("APOLLO_CLUSTER", "default"),
            namespace=os.environ.get("APOLLO_NAMESPACE", "application.yaml"),
            timeout=int(os.environ.get("APOLLO_TIMEOUT", "30")),
            apply_env=apply_env,
        )
        logger.info(f"⚙️ Apollo config: loaded {len(config)} items")
        return config

    except Exception as e:
        logger.warning(f"⚠️ Apollo config: failed ({e}), using env defaults")
        return None
