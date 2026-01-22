# Workflow Agent v9.2 更新日志

## 发布日期
2026-01-22

## 版本类型
**重大更新 (Major Update)**

## 核心变更

### 1. SOP 驱动的轻量级迭代机制

**问题背景：**
- v9.1 采用单次执行模式，无法处理多步骤业务场景
- 实际业务（如客服机器人）需要：查询知识库 → 保存客户信息 → 生成响应
- 需要在每个 Action 执行后，由 LLM 判断是否继续或响应

**解决方案：**
引入 SOP 驱动的轻量级外层迭代循环：
- 外层循环：控制 Action 序列执行
- LLM 决策：每步判断 should_continue 或 should_respond
- 可控迭代：max_iterations 限制，防止无限循环
- 清晰职责：外层控制流程，内层（Skill Agent）处理细节

### 2. 新增数据模型

#### IterationDecision
```python
@dataclass
class IterationDecision:
    """迭代决策结果 - LLM驱动的决策"""
    should_continue: bool      # 是否继续执行下一个Action
    should_respond: bool       # 是否生成最终响应给用户
    next_action: dict | None   # 下一个要执行的Action
    reasoning: str             # 决策理由
```

### 3. WorkflowConfigSchema 扩展

新增配置字段：
```python
max_iterations: int = 5                    # 最大迭代次数
iteration_strategy: str = "sop_driven"     # 迭代策略
```

支持两种策略：
- `sop_driven`：SOP 驱动的多步骤执行（默认）
- `single_shot`：单次执行（向后兼容 v9.1）

### 4. WorkflowAgent 核心方法重写

#### query 方法
```python
async def query(self, message: str, session_id: str) -> str:
    # 1. 加载会话状态
    # 2. 检查配置变更
    # 3. 发送问候语（首次）
    # 4. SOP驱动的迭代执行
    if self.config.iteration_strategy == "single_shot":
        response = await self._single_shot_execution(message, session)
    else:
        response = await self._sop_driven_execution(message, session)
    # 5. KB增强（可选）
    # 6. 注册Timer（可选）
    return response
```

#### _sop_driven_execution 方法（新增）
```python
async def _sop_driven_execution(self, message: str, session: Session) -> str:
    """SOP驱动的多步骤执行"""
    for iteration in range(max_iterations):
        # 1. LLM决策：下一步做什么
        decision = await self._llm_decide(session, message, iteration)

        # 2. 判断是否应该响应
        if decision.should_respond:
            return await self._generate_response(session, decision)

        # 3. 执行Action
        if decision.next_action:
            result = await self._execute_action_from_decision(...)
            session.add_execution_result(...)

        # 4. 判断是否继续
        if not decision.should_continue:
            return await self._generate_response(session)

    return await self._generate_response(session)
```

#### _llm_decide 方法（新增）
```python
async def _llm_decide(
    self,
    session: Session,
    user_message: str,
    iteration: int
) -> IterationDecision:
    """LLM决策：判断下一步动作"""
    # 构建决策 prompt，包含：
    # - SOP 流程
    # - 当前状态（迭代次数、执行历史）
    # - 可用 Actions（Skills/Tools/Flows/System）
    # - 决策规则

    # 调用 LLM 获取决策
    # 返回 IterationDecision
```

### 5. 辅助方法

新增三个辅助方法：
- `_format_available_actions()`：格式化可用 Actions 列表
- `_execute_action_from_decision()`：从决策结果执行 Action
- `_generate_response()`：生成最终响应

### 6. 架构图更新

更新了整体架构图（section 1.2），新增：
- 迭代循环节点（max_iterations）
- LLM 决策节点
- 决策分支（should_respond / should_continue）
- 上下文更新流程

### 7. 配置示例更新

在配置文件示例中添加：
```json
{
  "max_iterations": 5,
  "iteration_strategy": "sop_driven",
  "kb_config": {
    "enabled": true,
    "tool_name": "search_kb",
    "auto_enhance": true,
    "enhance_conditions": ["skill", "tool"]
  }
}
```

### 8. 对比说明更新

更新了与 v8.md 的对比表格，新增维度：
- **多轮迭代**：v8（外层while循环+状态机） vs v9.2（SOP驱动的轻量级循环）
- **决策机制**：v8（状态转换驱动） vs v9.2（LLM每步决策）
- **迭代控制**：v8（状态管理） vs v9.2（max_iterations配置）

## 设计理念变化

### v9.1 → v9.2 的关键转变

| 维度 | v9.1 | v9.2 |
|------|------|------|
| 执行模式 | 单次执行 | 多步骤序列执行 |
| 迭代位置 | 仅 Skill 内部 | 外层 + Skill 内部（分层） |
| 决策方式 | 意图匹配后直接执行 | LLM 每步决策 |
| 适用场景 | 简单单步任务 | 复杂多步骤流程 |

### 核心优势

1. **更实用**：符合实际业务场景（如客服机器人的多步骤流程）
2. **更清晰**：分层迭代，职责明确
   - 外层：控制 Action 序列
   - 内层：Skill Agent 处理细节
3. **更可控**：LLM 驱动决策 + 迭代次数限制
4. **更灵活**：支持单次执行和多步骤执行两种模式

## 向后兼容性

✅ **完全向后兼容**

通过 `iteration_strategy` 配置：
- 设置为 `single_shot`：保持 v9.1 行为
- 设置为 `sop_driven`（默认）：启用新的迭代机制

## 迁移指南

### 从 v9.1 迁移到 v9.2

**无需修改代码**，只需更新配置文件：

```json
{
  // 新增配置（可选）
  "max_iterations": 5,              // 默认值：5
  "iteration_strategy": "sop_driven" // 默认值："sop_driven"
}
```

**如果想保持 v9.1 行为：**
```json
{
  "iteration_strategy": "single_shot"
}
```

## 适用场景

### 推荐使用 v9.2（sop_driven）的场景

1. **客服机器人**：查询知识库 → 保存信息 → 响应
2. **工单处理**：验证信息 → 创建工单 → 通知用户
3. **审批流程**：收集信息 → 提交审批 → 更新状态
4. **数据处理**：获取数据 → 转换 → 存储 → 通知

### 继续使用 single_shot 的场景

1. 简单的单步查询（天气、汇率等）
2. 单一工具调用
3. 纯对话场景

## 性能影响

- **LLM 调用次数**：每次迭代增加 1 次 LLM 调用（决策）
- **响应延迟**：多步骤场景下，总延迟 = 决策时间 + 执行时间
- **成本**：根据 max_iterations 和实际迭代次数线性增长

**优化建议：**
- 合理设置 max_iterations（建议 3-5）
- 在 SOP 中明确步骤，减少不必要的迭代
- 使用更快的 LLM 模型进行决策（如 GPT-4o-mini）

## 测试建议

1. **单元测试**：
   - 测试 _llm_decide 的决策逻辑
   - 测试 _sop_driven_execution 的迭代控制
   - 测试 max_iterations 限制

2. **集成测试**：
   - 测试多步骤场景（Tool → System → Skill）
   - 测试迭代中断（should_respond=true）
   - 测试达到最大迭代次数的行为

3. **性能测试**：
   - 测试不同 max_iterations 下的响应时间
   - 测试 LLM 调用次数
   - 测试并发场景下的表现

## 已知限制

1. **迭代次数限制**：max_iterations 硬限制，达到后强制响应
2. **上下文累积**：每次迭代都会累积上下文，可能导致 token 增长
3. **决策准确性**：依赖 LLM 的决策能力，可能出现误判

## 未来计划

- [ ] 支持动态调整 max_iterations
- [ ] 添加迭代历史可视化
- [ ] 优化决策 prompt，提高准确率
- [ ] 支持条件分支（if-else）
- [ ] 支持并行执行多个 Action

## 贡献者

- Claude (Sonnet 4.5) - 设计与实现

## 参考文档

- [workflow-agent-v9.md](./workflow-agent-v9.md) - 完整技术方案
- [v8.md](./v8.md) - v8 版本参考
- [sop.json](./configs/sop.json) - 实际业务配置示例
- [procedure-customer-service.md](./configs/procedure-customer-service.md) - SOP 示例
