"""
Tool Executor - 统一工具执行器

职责：
- 根据配置分类执行工具（system_actions / agent_actions）
- system_actions: 并行执行，不参与上下文
- agent_actions: 串行执行，结果入上下文
"""

import asyncio
import logging
from typing import Any

import httpx

from bu_agent_sdk.schemas import WorkflowConfigSchema

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    统一工具执行器。

    Usage:
        ```python
        executor = ToolExecutor(config)
        results = await executor.execute([
            {"name": "save_customer_information", "params": {...}},
            {"name": "read_customer_information", "params": {...}},
        ])
        ```
    """

    def __init__(self, config: WorkflowConfigSchema):
        self.config = config
        # 构建工具索引
        self.tools_map = {t.get("name"): t for t in config.tools}
        self.system_actions = set(config.system_actions or [])
        self.agent_actions = set(config.agent_actions or [])

    def is_system_action(self, tool_name: str) -> bool:
        """判断是否为系统动作（静默执行）"""
        return tool_name in self.system_actions

    def is_agent_action(self, tool_name: str) -> bool:
        """判断是否为 Agent 动作（结果入上下文）"""
        return tool_name in self.agent_actions or tool_name not in self.system_actions

    async def execute(self, tool_calls: list[dict]) -> list[dict]:
        """
        执行工具调用。

        Args:
            tool_calls: 工具调用列表 [{"name": "...", "params": {...}}]

        Returns:
            Agent 动作的执行结果列表
        """
        system_calls = []
        agent_calls = []

        for call in tool_calls:
            tool_name = call.get("name")
            if self.is_system_action(tool_name):
                system_calls.append(call)
            else:
                agent_calls.append(call)

        # 1. 并行执行 system_actions（静默，不入上下文）
        if system_calls:
            await asyncio.gather(
                *[self._execute_one(c) for c in system_calls],
                return_exceptions=True,
            )
            logger.debug(f"Executed {len(system_calls)} system actions")

        # 2. 串行执行 agent_actions（结果入上下文）
        results = []
        for call in agent_calls:
            try:
                result = await self._execute_one(call)
                results.append({
                    "tool": call.get("name"),
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                logger.error(f"Tool execution failed: {call.get('name')}, {e}")
                results.append({
                    "tool": call.get("name"),
                    "status": "error",
                    "error": str(e),
                })

        return results

    async def execute_single(self, tool_name: str, params: dict = None) -> Any:
        """执行单个工具调用。"""
        return await self._execute_one({"name": tool_name, "params": params or {}})

    async def _execute_one(self, call: dict) -> Any:
        """执行单个工具。"""
        tool_name = call.get("name")
        params = call.get("params", {})

        tool_def = self.tools_map.get(tool_name)
        if not tool_def:
            raise ValueError(f"Tool not found: {tool_name}")

        endpoint = tool_def.get("endpoint", {})
        if not endpoint:
            raise ValueError(f"Tool endpoint not configured: {tool_name}")

        return await self._http_call(endpoint, params, tool_def)

    async def _http_call(self, endpoint: dict, params: dict, tool_def: dict) -> Any:
        """执行 HTTP 调用。"""
        url = endpoint.get("url")
        method = endpoint.get("method", "POST").upper()
        headers = endpoint.get("headers", {})
        body_template = endpoint.get("body", {})

        # 参数替换
        body = self._render_body(body_template, params)

        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params)
            else:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body,
                )

            response.raise_for_status()

            # 解析响应
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text

    def _render_body(self, template: dict, params: dict) -> dict:
        """渲染请求体模板。"""
        if not template:
            return params

        result = {}
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # 占位符替换
                param_name = value[1:-1]
                result[key] = params.get(param_name, value)
            elif isinstance(value, dict):
                result[key] = self._render_body(value, params)
            else:
                result[key] = value

        return result

    def get_tool_definition(self, tool_name: str) -> dict | None:
        """获取工具定义。"""
        return self.tools_map.get(tool_name)

    def list_tools(self) -> list[str]:
        """列出所有工具名称。"""
        return list(self.tools_map.keys())

    def list_system_actions(self) -> list[str]:
        """列出系统动作。"""
        return list(self.system_actions)

    def list_agent_actions(self) -> list[str]:
        """列出 Agent 动作。"""
        return list(self.agent_actions)
