"""
Apollo配置中心集成模块

从Apollo配置中心获取YAML格式的配置，并设置为环境变量供程序使用。
支持模型配置、agent配置host等信息。
"""

import os
import yaml
import logging
import aiohttp
from typing import Dict, Any

# 配置logger
logger = logging.getLogger(__name__)
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)


class ApolloConfigManager:
    """Apollo配置管理器 - 简化版本"""
    
    def __init__(
        self,
        config_server_url: str,
        app_id: str,
        cluster: str = "default",
        env: str = "DEV",
        timeout: int = 30
    ):
        """
        初始化Apollo配置管理器
        
        Args:
            config_server_url: Apollo配置中心地址
            app_id: 应用ID
            cluster: 集群名称，默认为default
            env: 环境名称，默认为DEV
            timeout: 超时时间（秒）
        """
        self.config_server_url = config_server_url
        self.app_id = app_id
        self.cluster = cluster
        self.env = env
        self.timeout = timeout
    
    async def _get_config_from_api(self, namespace: str = "application.yaml") -> str:
        """
        直接从 Apollo API 获取配置
        
        Args:
            namespace: 命名空间名称
            
        Returns:
            配置内容字符串
        """
        try:
            # 构建 API URL
            url = f"{self.config_server_url}/configs/{self.app_id}/{self.cluster}/{namespace}"
            logger.info(f"请求 Apollo API: {url}")


            connector = aiohttp.TCPConnector()
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            ) as session:
                async with session.get(url) as response:
                    logger.info(f"API 响应状态码: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # 从 configurations.content 中提取配置
                        configurations = data.get("configurations", {})
                        content = configurations.get("content", "")
                        
                        if content:
                            return content
                        else:
                            logger.warning("API 响应中没有找到 configurations.content")
                            return ""
                    else:
                        logger.error(f"API 请求失败，状态码: {response.status}")
                        return ""
                
        except Exception as e:
            logger.error(f"请求 Apollo API 失败: {e}")
            return ""
    
    def _parse_config_content(self, content: str) -> Dict[str, Any]:
        """
        解析配置内容，支持多种格式
        
        Args:
            content: 配置内容字符串
            
        Returns:
            解析后的配置字典
        """
        try:
            # 首先尝试解析为 YAML
            try:
                config = yaml.safe_load(content)
                if isinstance(config, dict):
                    logger.info("成功解析为 YAML 格式")
                    return config
            except yaml.YAMLError:
                logger.info("不是有效的 YAML 格式，尝试解析为环境变量格式")
            
            # 如果不是 YAML，尝试解析为环境变量格式
            config = {}
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value:
                        config[key] = value
                    else:
                        config[key] = ""

            return config
            
        except Exception as e:
            logger.error(f"解析配置内容失败: {e}")
            return {}
    
    async def load_config(self, namespace: str = "application.yaml") -> Dict[str, Any]:
        """
        从Apollo加载配置
        
        Args:
            namespace: 命名空间名称，默认为"application.yaml"
            
        Returns:
            解析后的配置字典
        """
        try:
            logger.info(f"开始加载配置，namespace: {namespace}")
            
            # 尝试从 Apollo API 获取配置
            config_content = await self._get_config_from_api(namespace)
            
            # 如果获取失败，使用本地回退配置
            if not config_content:
                raise ValueError("从 Apollo API 获取配置失败")
                
            
            # 解析配置内容
            config = self._parse_config_content(config_content)
            
            if not isinstance(config, dict):
                logger.warning(f"配置格式错误，期望dict，实际: {type(config)}")
                return {}
                
            return config
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def set_env_variables(self, config: Dict[str, Any]) -> None:
        """
        将配置设置为环境变量 - 简化版本
        
        Args:
            config: 配置字典
        """
        # 直接设置环境变量，因为配置已经是平铺的键值对
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ[key] = str(value)
                logger.debug(f"设置环境变量: {key}={value}")


async def load_apollo_config(
    config_server_url: str,
    app_id: str,
    cluster: str = "default",
    env: str = "DEV",
    namespace: str = "application.yaml",
    timeout: int = 30
) -> Dict[str, Any]:
    """
    便捷函数：从Apollo加载配置并设置为环境变量
    
    Args:
        config_server_url: Apollo配置中心地址
        app_id: 应用ID
        cluster: 集群名称
        env: 环境名称
        namespace: 命名空间名称
        timeout: 超时时间
        
    Returns:
        加载的配置字典
    """
    manager = ApolloConfigManager(
        config_server_url=config_server_url,
        app_id=app_id,
        cluster=cluster,
        env=env,
        timeout=timeout
    )
    
    config = await manager.load_config(namespace)
    manager.set_env_variables(config)
    
    return config


async def load_apollo_config_from_env() -> Dict[str, Any]:
    """
    从环境变量读取Apollo连接参数并加载配置
    
    需要的环境变量：
    - APOLLO_CONFIG_SERVER_URL: Apollo配置中心地址
    - APOLLO_APP_ID: 应用ID
    - APOLLO_CLUSTER: 集群名称（可选，默认default）
    - APOLLO_ENV: 环境名称（可选，默认DEV）
    - APOLLO_NAMESPACE: 命名空间名称（可选，默认application.yaml）
    - APOLLO_TIMEOUT: 超时时间（可选，默认30秒）
    
    Returns:
        加载的配置字典
    """
    config_server_url = os.environ.get("APOLLO_CONFIG_SERVER_URL")
    app_id = os.environ.get("APOLLO_APP_ID")
    
    if not config_server_url or not app_id:
        raise ValueError(
            "缺少必需的Apollo环境变量: APOLLO_CONFIG_SERVER_URL, APOLLO_APP_ID"
        )
    
    return await load_apollo_config(
        config_server_url=config_server_url,
        app_id=app_id,
        cluster=os.environ.get("APOLLO_CLUSTER", "default"),
        env=os.environ.get("APOLLO_ENV", "DEV"),
        namespace=os.environ.get("APOLLO_NAMESPACE", "application.yaml"),
        timeout=int(os.environ.get("APOLLO_TIMEOUT", "30"))
    )