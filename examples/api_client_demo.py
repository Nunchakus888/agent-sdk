"""
Workflow Agent API 客户端示例

演示如何使用 API 进行查询和会话管理
"""

import asyncio
import httpx
from typing import Optional


class WorkflowAgentClient:
    """Workflow Agent API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端

        Args:
            base_url: API 基础 URL
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=10)
        )

    async def query(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None
    ) -> dict:
        """
        发送查询请求

        Args:
            message: 用户消息
            session_id: 会话ID
            user_id: 用户ID（可选）

        Returns:
            响应字典
        """
        response = await self.client.post(
            "/api/v1/query",
            json={
                "message": message,
                "session_id": session_id,
                "user_id": user_id,
            }
        )
        response.raise_for_status()
        return response.json()

    async def get_session(self, session_id: str) -> dict:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典
        """
        response = await self.client.get(f"/api/v1/session/{session_id}")
        response.raise_for_status()
        return response.json()

    async def delete_session(self, session_id: str) -> dict:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            删除结果字典
        """
        response = await self.client.delete(f"/api/v1/session/{session_id}")
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> dict:
        """
        健康检查

        Returns:
            健康状态字典
        """
        response = await self.client.get("/api/v1/health")
        response.raise_for_status()
        return response.json()

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()


async def demo_basic_usage():
    """演示基础使用"""
    print("=" * 60)
    print("示例 1: 基础使用")
    print("=" * 60)

    client = WorkflowAgentClient()

    try:
        # 1. 健康检查
        print("\n1. 健康检查...")
        health = await client.health_check()
        print(f"   ✓ API 状态: {health['status']}")
        print(f"   ✓ 版本: {health['version']}")
        print(f"   ✓ 活跃会话数: {health['sessions_count']}")

        # 2. 发送第一条消息（触发问候语）
        print("\n2. 发送第一条消息...")
        session_id = "demo_session_001"
        result = await client.query(
            message="你好",
            session_id=session_id,
            user_id="demo_user"
        )
        print(f"   ✓ Agent 响应: {result['message']}")

        # 3. 发送后续消息
        print("\n3. 发送后续消息...")
        result = await client.query(
            message="帮我查询订单状态",
            session_id=session_id
        )
        print(f"   ✓ Agent 响应: {result['message'][:100]}...")

        # 4. 获取会话信息
        print("\n4. 获取会话信息...")
        session = await client.get_session(session_id)
        print(f"   ✓ 会话ID: {session['session_id']}")
        print(f"   ✓ 消息数量: {session['message_count']}")
        print(f"   ✓ 状态: {session['status']}")

        # 5. 删除会话
        print("\n5. 删除会话...")
        await client.delete_session(session_id)
        print(f"   ✓ 会话已删除")

    finally:
        await client.close()


async def demo_multi_session():
    """演示多会话管理"""
    print("\n" + "=" * 60)
    print("示例 2: 多会话管理")
    print("=" * 60)

    client = WorkflowAgentClient()

    try:
        # 创建多个会话
        sessions = ["session_001", "session_002", "session_003"]

        print("\n创建多个会话...")
        for session_id in sessions:
            result = await client.query(
                message=f"你好，我是会话 {session_id}",
                session_id=session_id
            )
            print(f"   ✓ {session_id}: {result['message'][:50]}...")

        # 检查健康状态
        health = await client.health_check()
        print(f"\n当前活跃会话数: {health['sessions_count']}")

        # 清理会话
        print("\n清理所有会话...")
        for session_id in sessions:
            await client.delete_session(session_id)
            print(f"   ✓ {session_id} 已删除")

    finally:
        await client.close()


async def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 60)
    print("示例 3: 错误处理")
    print("=" * 60)

    client = WorkflowAgentClient()

    try:
        # 1. 查询不存在的会话
        print("\n1. 查询不存在的会话...")
        try:
            await client.get_session("non_existent_session")
        except httpx.HTTPStatusError as e:
            print(f"   ✓ 捕获错误: {e.response.status_code} - {e.response.json()['message']}")

        # 2. 删除不存在的会话
        print("\n2. 删除不存在的会话...")
        try:
            await client.delete_session("non_existent_session")
        except httpx.HTTPStatusError as e:
            print(f"   ✓ 捕获错误: {e.response.status_code} - {e.response.json()['message']}")

    finally:
        await client.close()


async def demo_concurrent_requests():
    """演示并发请求"""
    print("\n" + "=" * 60)
    print("示例 4: 并发请求")
    print("=" * 60)

    client = WorkflowAgentClient()

    try:
        # 并发发送多个请求
        messages = [
            "查询订单状态",
            "查询天气",
            "写一篇博客",
            "帮我总结文档",
            "查询库存"
        ]

        print(f"\n并发发送 {len(messages)} 个请求...")

        import time
        start_time = time.time()

        # 创建并发任务
        tasks = [
            client.query(
                message=msg,
                session_id=f"concurrent_session_{i}"
            )
            for i, msg in enumerate(messages)
        ]

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed_time = time.time() - start_time

        # 打印结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"   ✗ 请求 {i+1} 失败: {result}")
            else:
                print(f"   ✓ 请求 {i+1} 成功: {result['message'][:50]}...")

        print(f"\n总耗时: {elapsed_time:.2f} 秒")
        print(f"平均耗时: {elapsed_time/len(messages):.2f} 秒/请求")

        # 清理会话
        print("\n清理会话...")
        for i in range(len(messages)):
            try:
                await client.delete_session(f"concurrent_session_{i}")
            except:
                pass

    finally:
        await client.close()


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Workflow Agent API 客户端演示")
    print("=" * 60)

    # 运行所有示例
    await demo_basic_usage()
    await demo_multi_session()
    await demo_error_handling()
    await demo_concurrent_requests()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
