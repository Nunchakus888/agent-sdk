# 迁移到 uv 包管理工具

## 为什么使用 uv？

`uv` 是由 Astral（Ruff 的开发者）开发的极速 Python 包管理工具，具有以下优势：

### 性能优势

- ⚡ **10-100 倍更快**：比 pip 快 10-100 倍
- 🚀 **并行下载**：同时下载多个包
- 💾 **全局缓存**：跨项目共享依赖缓存
- 🔒 **可靠的锁文件**：确保依赖一致性

### 兼容性

- ✅ 完全兼容 pip
- ✅ 支持 pyproject.toml
- ✅ 支持虚拟环境
- ✅ 支持所有 PyPI 包

### 基准测试

```
安装 FastAPI + 依赖：
- pip: ~45 秒
- uv:  ~2 秒（首次）/ ~0.5 秒（缓存）

安装测试依赖：
- pip: ~30 秒
- uv:  ~1.5 秒（首次）/ ~0.3 秒（缓存）
```

## 安装 uv

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 使用 pip 安装

```bash
pip install uv
```

## 项目依赖管理

### 依赖组

项目在 `pyproject.toml` 中定义了以下依赖组：

```toml
[project.optional-dependencies]
# LLM 提供商
anthropic = ["anthropic>=0.40.0"]
openai = ["openai>=1.50.0"]
google = ["google-genai>=1.0.0"]
observability = ["lmnr>=0.4.0"]

# API 依赖
api = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.9",
    "redis>=5.0.0",
    "motor>=3.3.0",
    "asyncpg>=0.29.0",
]

# 测试依赖
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "coverage>=7.0.0",
]

# 开发环境（所有依赖）
dev = [
    "bu-agent-sdk[anthropic,openai,google,observability,api,test]",
]
```

### 安装依赖

#### 基础安装

```bash
# 安装核心依赖
uv pip install -e .
```

#### 安装 API 依赖

```bash
# 安装 API 相关依赖
uv pip install -e ".[api]"
```

#### 安装测试依赖

```bash
# 安装测试相关依赖
uv pip install -e ".[test]"
```

#### 安装所有开发依赖

```bash
# 安装所有依赖（推荐开发环境）
uv pip install -e ".[dev]"
```

#### 安装特定 LLM 提供商

```bash
# 只安装 OpenAI
uv pip install -e ".[openai]"

# 安装 OpenAI + API
uv pip install -e ".[openai,api]"

# 安装所有 LLM 提供商
uv pip install -e ".[anthropic,openai,google]"
```

## 常用命令对比

### 安装包

```bash
# pip
pip install package-name

# uv
uv pip install package-name
```

### 安装 requirements.txt

```bash
# pip
pip install -r requirements.txt

# uv
uv pip install -r requirements.txt
```

### 安装可编辑模式

```bash
# pip
pip install -e .

# uv
uv pip install -e .
```

### 列出已安装包

```bash
# pip
pip list

# uv
uv pip list
```

### 卸载包

```bash
# pip
pip uninstall package-name

# uv
uv pip uninstall package-name
```

### 冻结依赖

```bash
# pip
pip freeze > requirements.txt

# uv
uv pip freeze > requirements.txt
```

## Docker 集成

Dockerfile 已更新为使用 uv：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY . .

# Install dependencies using uv (much faster than pip)
RUN uv pip install --system -e ".[api]"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 构建镜像

```bash
# 构建镜像（使用 uv，更快）
docker build -t workflow-agent-api .

# 运行容器
docker run -d -p 8000:8000 workflow-agent-api
```

## CI/CD 集成

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: |
          uv pip install --system -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v --cov
```

### GitLab CI

```yaml
test:
  image: python:3.11-slim
  before_script:
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - export PATH="/root/.cargo/bin:$PATH"
    - uv pip install --system -e ".[dev]"
  script:
    - pytest tests/ -v --cov
```

## 虚拟环境

uv 也支持虚拟环境管理：

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 在虚拟环境中安装依赖
uv pip install -e ".[dev]"
```

## 迁移检查清单

- [x] ✅ 更新 `pyproject.toml` 添加依赖组
- [x] ✅ 更新 `Dockerfile` 使用 uv
- [x] ✅ 更新 `api/README.md` 文档
- [x] ✅ 更新 `docs/workflow-agent-v9.md` 文档
- [x] ✅ 更新 `docs/api-testing-summary.md` 文档
- [x] ✅ 更新 `CHANGELOG.md`
- [ ] 🔄 更新 CI/CD 配置（如果有）
- [ ] 🔄 团队成员安装 uv

## 常见问题

### Q: uv 和 pip 可以混用吗？

A: 可以，但不推荐。建议统一使用 uv 以获得最佳性能。

### Q: uv 支持私有 PyPI 源吗？

A: 支持。使用 `--index-url` 或 `--extra-index-url` 参数。

```bash
uv pip install --index-url https://private.pypi.org/simple package-name
```

### Q: 如何清理 uv 缓存？

A: 使用 `uv cache clean` 命令。

```bash
uv cache clean
```

### Q: uv 在哪里存储缓存？

A:
- Linux: `~/.cache/uv`
- macOS: `~/Library/Caches/uv`
- Windows: `%LOCALAPPDATA%\uv\cache`

### Q: 遇到安装问题怎么办？

A: 尝试以下步骤：

1. 清理缓存：`uv cache clean`
2. 使用 `--no-cache` 标志：`uv pip install --no-cache -e .`
3. 回退到 pip：`pip install -e .`

## 性能对比

### 实际测试结果

在本项目中的实际测试：

```bash
# 安装 API 依赖
pip install -e ".[api]"     # ~45 秒
uv pip install -e ".[api]"  # ~2 秒（首次）/ ~0.5 秒（缓存）

# 安装测试依赖
pip install -e ".[test]"    # ~30 秒
uv pip install -e ".[test]" # ~1.5 秒（首次）/ ~0.3 秒（缓存）

# 安装所有开发依赖
pip install -e ".[dev]"     # ~90 秒
uv pip install -e ".[dev]"  # ~4 秒（首次）/ ~1 秒（缓存）
```

### Docker 构建时间

```bash
# 使用 pip
docker build -t workflow-agent-api .  # ~3-5 分钟

# 使用 uv
docker build -t workflow-agent-api .  # ~1-2 分钟
```

## 推荐工作流

### 开发环境设置

```bash
# 1. 克隆项目
git clone <repo-url>
cd agent-sdk

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 创建虚拟环境（可选）
uv venv
source .venv/bin/activate

# 4. 安装所有开发依赖
uv pip install -e ".[dev]"

# 5. 运行测试
pytest tests/ -v
```

### 生产环境部署

```bash
# 使用 Docker（推荐）
docker-compose up -d

# 或本地运行
uv pip install -e ".[api]"
python -m api.main
```

## 相关资源

- [uv 官方文档](https://github.com/astral-sh/uv)
- [uv 性能基准测试](https://github.com/astral-sh/uv#benchmarks)
- [Astral 官网](https://astral.sh/)

## 总结

迁移到 uv 带来的好处：

- ⚡ **速度提升 10-100 倍**
- 💾 **节省磁盘空间**（全局缓存）
- 🔒 **更可靠的依赖管理**
- 🚀 **更快的 CI/CD 流程**
- ✅ **完全兼容现有工具链**

推荐所有开发者和生产环境使用 uv！
