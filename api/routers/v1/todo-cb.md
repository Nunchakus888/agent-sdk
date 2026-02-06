

api/v1/chat 实现：

1. callback 输出请求&响应日志，注意可读性
2. gretting消息回调通知，tokens消耗也需要统计并返回，检查是否已实现
3. 同一个sessionid的多个请求，需要确保前一个请求的message写db成功，该请求才可以被后面的新请求取消，不然会造成message丢失，注意优化一下当前实现
   1. 以及被取消任务发生的tokens消耗需要得到累加统计
4. flow_executor 执行加个参数，correlation_id 与 flow_id 同级
