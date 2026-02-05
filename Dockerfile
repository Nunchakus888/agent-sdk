# =============================================================================
# Python Application (Chat UI 预构建版本)
# =============================================================================
# 注意: 需要先在本地执行 cd api/chat && npm run build 生成 dist 目录
# =============================================================================
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install uv via pip (避免 apt-get 网络问题)
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies
RUN uv sync --frozen --no-dev --extra api

# Copy application code
COPY bu_agent_sdk/ ./bu_agent_sdk/
COPY api/ ./api/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Health check (使用 Python 代替 curl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=5)"

# Run the application
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
