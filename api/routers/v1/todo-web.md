
api/v1/chat 实现：

1. [x] 结合 api/chat 文件目录（前端代码）实现：
   1. [x] sessions 等API端点，用于表示 会话列表，v2已有实现，可以参考，优化，独立出来，注意DRY，尽量复用
      - 实现: `api/routers/v1/session.py` — GET/PATCH/DELETE /session, GET /session/{id}/events
      - 复用: `RepositoryManager` (SessionRepository, MessageRepository, UsageRepository)
   2. [x] 新增页面展示 configs 缓存记录，并支持对比 原始&llm增强内容的差异
      - 实现: `api/routers/v1/config.py` — GET /config, GET /config/{id}, GET /config/{id}/diff
      - diff 使用递归对比算法，返回 added/removed/changed 差异列表
   3. [x] 以及展示对应的会话记录，tokens消耗等信息
      - GET /session/{id} 返回 usage 统计和 message_count
      - GET /session/{id}/events 返回消息列表（匹配前端 EventInterface 格式）
2. [x] 做好单元测试，注意避免mock数据，使用真实客户端请求
   - 实现: `tests/test_v1_session_api.py` — 21 个测试用例，全部通过
   - 使用 httpx.AsyncClient + ASGI transport 进行真实 HTTP 请求
   - 内存模式运行，无需 MongoDB
3. [x] 会话列表，消息列表等可以直接进行API交互，不依赖模型请求，直接使用真实请求
   - Session/Agent/Config API 均不依赖 LLM，直接操作数据库
4. [x] 更新文档
   - 更新: `docs/configuration-guide.md` — 新增 V1 API 端点文档
5. [x] 注意优雅精简，最佳实践，OOP，DRY设计原则
   - 复用 RepositoryManager 数据访问层，不重复实现
   - 延迟导入避免循环依赖（与 v2 router 模式一致）
   - Repository 层新增方法遵循现有模式（persistent + memory 双模式）
