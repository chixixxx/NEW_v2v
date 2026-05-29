# Future V2V 自适应批量匹配研究方案

## 研究定位

本项目研究双边 V2V 平台中的自适应批量匹配时机：在电池健康、供需参与和城市时空压力约束下，由强化学习决定何时触发匹配，由约束优化器决定具体 CV-DV 匹配结果。

平台中的 CV 是需要获得有效充电量的需求车辆，DV 是愿意输出电量并获得补偿的供给车辆。订单需求、OD 热点、行程时间和价格强度由 TLC Manhattan 黄出租数据校准；V2V 专属属性，包括需求电量、等待窗口、SOC、保留电量、报价、在线时间和 fleet/private 差异，由业务模型生成。

核心范式：

```text
强化学习决定 WAIT / MATCH_TOP_BATCH / MATCH_FULL
约束优化器决定订单-车辆匹配边
电池健康和双边价格机制决定边是否可行、是否有正平台收益
```

## MDP 定义

每个决策 tick 为 3 分钟。环境观测是低维聚合状态，包括活跃订单、等待比例、取消风险、可用车辆、健康可供电量、供需压力、可行边密度、候选边收益和电池健康约束紧张度。

动作空间固定为：

```text
0 = WAIT
1 = MATCH_TOP_BATCH
2 = MATCH_FULL
```

- `WAIT`：推进一个 tick，不触发匹配；订单可能继续等待、取消或过期，车辆也可能离池。
- `MATCH_TOP_BATCH`：先求完整约束匹配，再只执行最高价值的一部分匹配，是主控制动作。
- `MATCH_FULL`：执行完整正收益匹配，主要作为 fixed/full 诊断基线，也允许 DQN 在极端状态下选择。

## V2V 价格与电池健康

`order.demand_kwh` 表示 CV 实际需要获得的有效电量。由于传输损耗，DV 输出电量为：

```text
donor_output_kwh = demand_kwh / transfer_efficiency
```

默认 `transfer_efficiency=0.90`，约 10% 能量以损耗形式消失。DV 可供电量由 reserve 和最低健康 SOC 共同约束：

```text
available_energy =
  current_soc_kwh - max(reserve_kwh, donor_min_soc_ratio * battery_capacity_kwh)
```

默认 `donor_min_soc_ratio=0.25`，防止供给车辆被放电到过低 SOC。

平台利润拆成可解释的双边经济结构：

```text
buyer_payment
- seller_reimbursement
- platform_pickup_cost
- seller_time_cost
- dispatch_fixed_cost
- rapid_dispatch_penalty
```

卖方补偿拆为：

```text
energy_cost + degradation_cost + service_premium
```

其中 `degradation_cost_per_kwh=0.08` 单独统计，不混入基础电价。

## 动态匹配压力设计

主环境要体现“等待形成更好 batch”和“等待导致取消、过期、车辆离池”的权衡。

- `MATCH_TOP_BATCH` 固定成本为 18.0。
- `MATCH_FULL` 固定成本为 38.5。
- 若最近 1 tick 内已经派单，再次派单额外扣 10.0，避免一步一匹配成为无成本默认选择。
- top-batch 容量为活跃订单的 55%，并限制在 8 到 80 之间。
- `short_lookahead_top_batch` 作为强规则基线时带连续触发冷却，只有临期占比足够高才允许连续派单。

## 算法路线

第一版采用 Double DQN 学习匹配时机和批量强度。DQN 不直接选择具体订单-车辆边，避免动作空间过大；边选择由 Hungarian assignment 和业务约束完成。

训练采用并行环境采样、单进程 learner：

- 多个 rollout worker 独立运行 episode。
- worker 返回 transitions、episode metrics 和动作分布。
- 主进程维护 replay buffer、DQN 网络和 optimizer。
- checkpoint 按 validation seeds 上的 `future_v2v_score_mean` 选择。

teacher replay prefill 只输出同一动作空间下的 `WAIT / MATCH_TOP_BATCH / MATCH_FULL`，不直接输出匹配边，避免模仿学习接口和 RL 控制接口不一致。

## 对比基线

所有基线使用同一仿真器、同一候选图、同一电池健康约束和同一约束优化器：

- `fixed_1_tick_full_match`
- `fixed_2_tick_full_match`
- `fixed_1_tick_top_batch`
- `queue_threshold_top_batch`
- `deadline_trigger_top_batch`
- `supply_demand_pressure_top_batch`
- `short_lookahead_top_batch`
- `dqn_adaptive_timing`

主比较对象不是某个规则的边级得分，而是不同匹配时机策略在约束利润、服务率、取消/过期、平均 batch interval、平台边际收益和电池健康指标上的综合表现。
