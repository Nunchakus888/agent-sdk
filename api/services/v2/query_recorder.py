"""
统一 DB 写入

职责：
- 接收 EventCollector 收集的数据
- 统一写入 messages / tool_calls / usages 表
- 支持异步 fire-and-forget 模式

V2 优化：
- 使用简化的 V2 数据模型
- 单次写入 usages（无 finalize 步骤）
- tool_calls 替代 events
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from bu_agent_sdk.tokens import UsageSummary

from api.services.v2.event_collector import EventCollector

if TYPE_CHECKING:
    from api.services.repositories import RepositoryManager

logger = logging.getLogger(__name__)


class QueryRecorder:
    """
    统一 DB 写入 (V2 优化版)

    使用简化的 V2 数据模型：
    - messages: 简化字段，无 state/metadata
    - tool_calls: 替代 events，直接映射 ToolCallRecord
    - usages: 扁平结构，单次写入

    Usage:
        recorder = QueryRecorder(repos)

        # 同步记录
        await recorder.record(collector, usage)

        # 异步记录（Fire & Forget）
        recorder.record_async(collector, usage)
    """

    def __init__(self, repos: "RepositoryManager"):
        self._repos = repos

    async def record(
        self,
        collector: EventCollector,
        usage: UsageSummary | None = None,
    ) -> None:
        """
        统一记录逻辑

        Args:
            collector: 事件收集器
            usage: Token 使用统计
        """
        logger.info(
            f"QueryRecorder.record: session={collector.session_id}, "
            f"correlation={collector.correlation_id}"
        )
        try:
            await asyncio.gather(
                self._record_messages(collector),
                self._record_tool_calls(collector),
                self._record_usage(collector, usage),
            )
            logger.info(f"QueryRecorder.record completed: {collector.session_id}")
        except Exception as e:
            logger.error(f"Failed to record query: {e}", exc_info=True)

    async def _record_messages(self, collector: EventCollector) -> None:
        """记录消息 (V2: 简化字段)"""
        # 用户消息
        if collector.user_message:
            logger.debug(f"Recording user message: {collector.session_id}")
            await self._repos.messages.create(
                session_id=collector.session_id,
                role="user",
                content=collector.user_message,
                correlation_id=collector.correlation_id,
            )
            logger.debug(f"User message recorded: {collector.session_id}")

        # 助手消息
        if collector.final_response:
            logger.debug(f"Recording assistant message: {collector.session_id}")
            await self._repos.messages.create(
                session_id=collector.session_id,
                role="assistant",
                content=collector.final_response,
                correlation_id=collector.correlation_id,
            )
            logger.debug(f"Assistant message recorded: {collector.session_id}")

    async def _record_tool_calls(self, collector: EventCollector) -> None:
        """记录工具调用 (V2: 直接映射 ToolCallRecord)"""
        for i, tc in enumerate(collector.tool_calls):
            await self._repos.tool_calls.create(
                tool_call_id=tc.tool_call_id,
                session_id=collector.session_id,
                correlation_id=collector.correlation_id,
                offset=i,  # 简单顺序 offset
                tool_name=tc.tool_name,
                arguments=tc.arguments,
                result=tc.result[:1000] if tc.result else None,
                is_error=tc.is_error,
                duration_ms=int(tc.duration_ms),
            )

    async def _record_usage(
        self,
        collector: EventCollector,
        usage: UsageSummary | None,
    ) -> None:
        """记录 usage (V2: 单次写入，扁平结构)"""
        if not usage:
            logger.debug(f"No usage to record: {collector.session_id}")
            return

        logger.debug(f"Recording usage: {collector.session_id}, tokens={usage.total_tokens}")

        # 构建 by_model 字典
        by_model = {
            model: {
                "input": stats.prompt_tokens,
                "output": stats.completion_tokens,
                "cost": stats.cost,
            }
            for model, stats in usage.by_model.items()
        }

        # 单次写入（无 finalize 步骤）
        await self._repos.usages.create(
            session_id=collector.session_id,
            correlation_id=collector.correlation_id,
            total_input_tokens=usage.total_prompt_tokens,
            total_output_tokens=usage.total_completion_tokens,
            total_tokens=usage.total_tokens,
            total_cost=usage.total_cost,
            by_model=by_model,
        )
        logger.debug(f"Usage recorded: {collector.session_id}")

    def record_async(
        self,
        collector: EventCollector,
        usage: UsageSummary | None = None,
    ) -> asyncio.Task:
        """
        异步记录（Fire & Forget）

        Returns:
            asyncio.Task 用于可选的等待或取消
        """
        return asyncio.create_task(self.record(collector, usage))
