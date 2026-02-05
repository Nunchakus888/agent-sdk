# Apollo Config Loader

轻量级、异步的 Apollo 配置中心客户端。

## 特性

- 异步架构 (async/await)
- 支持 YAML 和 KEY=VALUE 配置格式
- 自动设置环境变量
- 零依赖（仅需 aiohttp + pyyaml）
- 简洁 API

## 安装

```bash
# 从本地安装
pip install -e packages/apollo_config_loader

# 或复制文件到项目
cp -r packages/apollo_config_loader your_project/
```

## 快速开始

### 方式 1: 直接传参

```python
import asyncio
from apollo_config_loader import load_config

async def main():
    config = await load_config(
        server_url="http://apollo.example.com:8080",
        app_id="your-app-id",
    )
    print(f"Loaded {len(config)} config items")

asyncio.run(main())
```

### 方式 2: 从环境变量

```python
import os
import asyncio
from apollo_config_loader import load_config_from_env

# 设置连接参数
os.environ["APOLLO_SERVER_URL"] = "http://apollo.example.com:8080"
os.environ["APOLLO_APP_ID"] = "your-app-id"

async def main():
    config = await load_config_from_env()
    print(f"Loaded {len(config)} config items")

asyncio.run(main())
```

### 方式 3: 使用类

```python
from apollo_config_loader import ApolloConfigLoader, ApolloConfig

loader = ApolloConfigLoader(
    ApolloConfig(
        server_url="http://apollo.example.com:8080",
        app_id="your-app-id",
        cluster="default",
        namespace="application.yaml",
        timeout=30,
    )
)

# 分步操作
content = await loader.fetch()      # 获取原始内容
config = loader.parse(content)      # 解析配置
loader.apply_to_env(config)         # 设置环境变量

# 或一步完成
config = await loader.load()
```

## API 参考

### `load_config()`

```python
async def load_config(
    server_url: str,           # Apollo 配置中心地址
    app_id: str,               # 应用 ID
    cluster: str = "default",  # 集群名称
    namespace: str = "application.yaml",  # 命名空间
    timeout: int = 30,         # 超时时间（秒）
    apply_env: bool = True,    # 是否设置环境变量
) -> Dict[str, Any]
```

### `load_config_from_env()`

```python
async def load_config_from_env(
    apply_env: bool = True,    # 是否设置环境变量
) -> Dict[str, Any]
```

**环境变量:**

| 变量名 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `APOLLO_SERVER_URL` | 是 | - | Apollo 配置中心地址 |
| `APOLLO_APP_ID` | 是 | - | 应用 ID |
| `APOLLO_CLUSTER` | 否 | `default` | 集群名称 |
| `APOLLO_NAMESPACE` | 否 | `application.yaml` | 命名空间 |
| `APOLLO_TIMEOUT` | 否 | `30` | 超时时间（秒） |

## 配置格式支持

### YAML 格式 (推荐)

```yaml
DATABASE_URL: postgresql://localhost/mydb
REDIS_HOST: localhost
REDIS_PORT: 6379
DEBUG: true
```

### KEY=VALUE 格式

```
DATABASE_URL=postgresql://localhost/mydb
REDIS_HOST=localhost
REDIS_PORT=6379
DEBUG=true
```

## 错误处理

```python
from apollo_config_loader import load_config, ApolloConfigError

try:
    config = await load_config(
        server_url="http://apollo.example.com:8080",
        app_id="your-app-id",
    )
except ApolloConfigError as e:
    print(f"配置加载失败: {e}")
    # 回退到本地配置
    from dotenv import load_dotenv
    load_dotenv()
```

## 服务启动集成示例

```python
import asyncio
from dotenv import load_dotenv
from apollo_config_loader import load_config_from_env, ApolloConfigError

async def init_config():
    """初始化配置，优先 Apollo，回退本地 .env"""
    try:
        print("Loading config from Apollo...")
        config = await load_config_from_env()
        print(f"Loaded {len(config)} items from Apollo")
        return config
    except ApolloConfigError as e:
        print(f"Apollo failed: {e}, falling back to .env")
        load_dotenv()
        return {}

async def main():
    await init_config()
    # 启动服务...

if __name__ == "__main__":
    asyncio.run(main())
```

## Apollo API 说明

本模块调用的 Apollo API:

```
GET {server_url}/configs/{app_id}/{cluster}/{namespace}
```

响应格式:

```json
{
  "appId": "your-app-id",
  "cluster": "default",
  "namespaceName": "application.yaml",
  "configurations": {
    "content": "DATABASE_URL: xxx\nREDIS_HOST: xxx"
  }
}
```

## License

MIT
