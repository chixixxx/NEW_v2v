# Future V2V 自适应批量匹配研究方案

## 研究定位

本项目研究双边 V2V 平台中的自适应批量匹配时机：在电池健康、供需参与和城市时空压力约束下，由强化学习决定何时触发匹配，由约束优化器决定具体 CV-DV 匹配结果。

平台中的 CV 是需要获得有效充电量的需求车辆，DV 是愿意输出电量并获得补偿的供给车辆。订单需求、OD 热点、行程时间和价格强度由 TLC Manhattan 黄出租数据校准；V2V 专属属性，包括需求电量、等待窗口、SOC、保留电量、报价、在线时间和 fleet/private 差异，由业务模型生成。

核心范式：

```text
强化学习决定 WAIT / MATCH_TOP_BATCH / MATCH_FULL
约束优化器决定订单-车辆匹配边
电池健康、双边价格和交易摩擦共同决定平台收益
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

边级平台毛收益拆成可解释的双边经济结构：

```text
buyer_payment
- seller_reimbursement
- platform_pickup_cost
- seller_time_cost
```

卖方补偿拆为：

```text
energy_cost + degradation_cost + service_premium
```

其中 `degradation_cost_per_kwh=0.08` 单独统计，不混入基础电价。

## Dispatch Friction

V2V dispatch 不只是算法重算，还包含报价、通知、路线承诺、SOC 承诺、支付和用户确认。因此正式环境采用 decomposed transaction cost：

```text
dispatch_friction_cost =
  setup_cost
  + pair_coordination_cost
  + refresh_cost
  + full_mode_extra_cost
```

默认值：

```text
setup_cost = 18.0
pair_coordination_cost = 0.75 * matched_pairs
full_mode_extra_cost = 0.20 * matched_pairs, only for MATCH_FULL
refresh_cost = 16.0 * exp(-ticks_since_last_dispatch / 1.5)
```

`MATCH_FULL` 更贵不是因为计算更贵，而是因为它触发更多低边际 CV-DV 交易协调。`refresh_cost` 是平滑衰减的报价刷新摩擦，不是硬性的连续派单惩罚。

## 算法路线

第一版采用 Double DQN 学习匹配时机和批量强度。DQN 不直接选择具体订单-车辆边，避免动作空间过大；边选择由 Hungarian assignment 和业务约束完成。

训练采用并行环境采样、单进程 learner：

- 多个 rollout worker 独立运行 episode。
- worker 返回 transitions、episode metrics 和动作分布。
- 主进程维护 replay buffer、DQN 网络和 optimizer。
- checkpoint 按 validation seeds 上的 `future_v2v_score_mean` 选择。

teacher replay prefill 只输出同一动作空间下的 `WAIT / MATCH_TOP_BATCH / MATCH_FULL`，不直接输出匹配边，避免模仿学习接口和 RL 控制接口不一致。

## 对比基线与可信性检查

所有基线使用同一仿真器、同一候选图、同一电池健康约束和同一约束优化器：

- `fixed_1_tick_full_match`
- `fixed_2_tick_full_match`
- `fixed_1_tick_top_batch`
- `queue_threshold_top_batch`
- `deadline_trigger_top_batch`
- `supply_demand_pressure_top_batch`
- `short_lookahead_top_batch`
- `dqn_adaptive_timing`

评估额外运行 friction sensitivity：

- `decomposed_transaction_cost`：正式口径。
- `no_refresh_friction`：检查动态优势是否依赖短时间重复 dispatch 成本。
- `common_fixed_cost`：检查 full/top 差异是否过度依赖 full extra cost。
- `no_dispatch_friction`：诊断完全无交易摩擦时是否自然退化为高频匹配。

主结论必须来自 `decomposed_transaction_cost`，并且 `no_refresh_friction` 下仍保留正向 paired delta，才认为环境具备可信动态匹配区分度。
## DQN 可学习性增强

为避免 DQN 在交易摩擦环境中学成高频 `MATCH_TOP_BATCH`，观测增加 `ticks_since_last_dispatch`、top/full 预估摩擦、time-of-day bucket、临期订单占比和 projected service risk。

step reward 继续以即时平台利润为主，但 service-risk shaping 采用 delta potential：比较动作前后服务风险潜势是否改善，而不是每一步惩罚“当前服务率低”。这样 WAIT 不会因为当前累计服务率偏低而天然吃亏。

WAIT 动作额外使用 one-step opportunity bonus：若 WAIT 后候选边收益增量能够覆盖过期、取消、等待和刷新摩擦风险，则给轻量正奖励；若临期损失大，则不加 bonus。

teacher replay prefill 只使用同一动作空间，但强化 `fixed_2_tick_full_match` 和 deadline rescue 样本，让 DQN 明确学习何时等待、何时用 `MATCH_FULL` 兜住临期服务风险。

teacher replay prefill 还加入 `WaitOpportunityTeacherPolicy`，专门提供最近刚 dispatch、临期压力低、刷新摩擦高或短窗口候选收益预计提升时的 WAIT 样本。

checkpoint selection 使用分层 validation manifest 覆盖 `morning_peak / midday / evening_peak / off_peak`，并按 `0.85 * mean_score + 0.15 * worst_bucket_score - off_peak_floor_penalty` 选择 checkpoint，减少模型在 off-peak stress 窗口突然崩盘的风险，同时避免 worst bucket 权重过强导致策略过度保守。
