# Future V2V 自适应匹配时机研究方案

## 研究定位

本项目研究未来车车互助补能平台中的动态批量匹配问题。平台不是每个时刻都立即派单，而是在订单等待、车辆供给、电量约束、取消风险、空间压力和 dispatch 成本之间权衡，决定当前继续等待、做有限批量匹配，还是做全量匹配。

主环境使用 TLC Manhattan 黄出租数据校准需求。出租车数据只提供真实城市时空需求、OD 热点、行程时间和价格强度；V2V 专属属性仍由业务模型生成，避免把出租车业务直接等同于 V2V 互助补能。

核心范式：

```text
强化学习决定何时、以多大批量匹配
运筹优化决定具体订单-车辆匹配边
```

## MDP 定义

每个决策 tick 为 3 分钟。环境观测是低维聚合状态，包括：

- 活跃订单数、总需求电量、平均等待比例、临近截止订单数。
- 可用车辆数、总可供电量、平均剩余在线时间、fleet 占比。
- 供需压力、热点压力、可行边密度、候选边利润统计。

动作空间：

```text
0 = WAIT
1 = MATCH_TOP_BATCH
2 = MATCH_FULL
```

`WAIT` 推进一个 tick，订单可能继续等待、取消或过期。
`MATCH_TOP_BATCH` 调用约束匹配器，但只执行当前批次中最高价值的一部分匹配。
`MATCH_FULL` 执行完整正收益匹配，主要作为 fixed/full diagnostic baseline。

## 算法路线

第一版采用 Double DQN 学习匹配时机与批量强度。DQN 不直接选择具体订单-车辆边，只选择三类动作；匹配结果由约束二分图匹配器产生。

训练采用并行环境采样、单进程 learner：

- 多个 rollout worker 独立运行 episode。
- worker 返回 transitions、episode metrics 和动作分布。
- 主进程维护 replay buffer、DQN 网络和 optimizer。
- checkpoint 按 validation seeds 上的 `future_v2v_score_mean` 选择，而不是默认取最后一轮。

teacher replay prefill 只输出同一动作空间下的 `WAIT / MATCH_TOP_BATCH / MATCH_FULL`，不输出匹配边，避免模仿学习接口和 RL 控制接口不一致。

## 对比基线

默认评估同一仿真器、同一候选图、同一约束匹配器下的 timing policy：

- `fixed_1_tick_full_match`
- `fixed_2_tick_full_match`
- `fixed_1_tick_top_batch`
- `queue_threshold_top_batch`
- `deadline_trigger_top_batch`
- `supply_demand_pressure_top_batch`
- `short_lookahead_top_batch`
- `dqn_adaptive_timing`

主比较对象不是某个规则的边级得分，而是不同匹配时机策略在约束利润、服务率、取消/过期和平均 batch interval 上的综合表现。

## 研究价值

V2V 和网约车派单不同：订单不是“人等车”，而是“能量需求等待移动供给”。除空间距离外，还存在需求电量、可供电量、保留电量、车主时间窗口、愿付价和报价等约束。

动态匹配时机能体现 V2V 的关键难点：等待可能形成更优批次并摊薄 dispatch 成本，但也可能导致取消、过期和车辆离池。三动作设计让 RL 不只学习“是否匹配”，还能学习“当前是否适合做有限批量匹配”。
