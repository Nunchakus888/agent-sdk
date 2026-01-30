优化当前数据模型，配置缓存结构比较确定，
WorkflowConfigSchema
还需要设计：
sessions
agentstate
events(日志）
messages
inspections (tokens消耗)
该怎样组织数据model，表结构设计？

关联关系是：
tenant_id:chatbot_id 1:N （一个租户可以有N个chatbot，租户id仅限查询使用，代码中之关注chatbot即可）
chatbot_id:agent 1:N （chatbot 配置供它下面所有agent共享，可以有N个agent）
agent:session 1:N 或者1:1 （一个agent可以有1个或多个会话，彼此之间无状态，完全隔离）
session:messages 1:N （一个会话可能有多轮message交互）
session:agentstate 1:N (一个会话可能有多个阶段的agent状态)
message:agentstate 1:1 （每条对话消息对应一个agent状态）
messages:events 1:N （一条消息可能经过不同阶段处理，见下一条）
messages:inspections 1:N(一条消息可能有不同阶段，llm解析配置，llm决策，tools执行、输出等多种tokens消耗）

inspections 需要统计不同阶段消耗，以及总计消耗
events 统计任务处理中所有阶段：配置解析，缓存读取，llm决策，tools调用，其他，最后生成内容等
session 会话详情，关联 chatbot_id, customer_id, session_id, agentState
configs 配置缓存 tenant_id, chatbot_id, md5checksum, **WorkflowConfigSchema
