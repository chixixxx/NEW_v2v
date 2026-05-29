# Future V2V Benchmark Environment 说明

## 路网与数据源

默认主环境为 `tlc_manhattan`：

- 订单到达、OD 热点、行程时长和价格强度来自 NYC TLC 黄出租月度数据。
- 路网层级使用 Manhattan taxi zone，不使用街道路段级 OSM，也不把 16 区合成网格作为主环境。
- travel time 使用同月黄出租 OD 在对应 60 分钟时段内的中位行程时间。
- 缺失 OD 回退到同 OD 全时段中位数、同 origin 中位数或全局中位数。
- 黄出租数据只作为未来 V2V 需求与城市移动模式的校准源，不直接解释为 V2V 交易。

合成 16 区网格只作为 smoke/test fallback。

## 时间粒度与规模

当前默认：

- 1 tick = 3 分钟。
- `main`：80 ticks + 14 terminal buffer，约 4 小时 + 42 分钟。
- `smoke`：60 ticks + 8 terminal buffer，约 3 小时 + 24 分钟。
- `service_kwh_per_tick=2.7`，约等于 54 kW；实际服务时长会被 `max_discharge_power_kw=50.0` 自动约束。

main 默认压力：

- 1600 单。
- 1900 候选车辆。
- 基础入池率 0.36。
- `supply_scale=0.50`。
- 每单候选车辆上限 20。
- `dispatch_fixed_cost=18.0`。
- `dispatch_capacity_ratio=0.70`，`dispatch_capacity_min=8`，`dispatch_capacity_max=120`。

## TLC 到 V2V 的改造

- `arrival_tick` 来自出租车 pickup 时间在 episode 窗口内的位置。
- `origin_zone` 来自 `PULocationID`，表示需求发生区域。
- `destination_zone` 来自 `DOLocationID`，解释为服务后车辆可能靠近的活动区域。
- `demand_kwh` 由 trip distance、duration 和业务扰动生成，控制在 V2V 合理电量区间。
- `willingness_to_pay_per_kwh` 由 total amount 与电量需求校准，并加上下限约束。
- `max_wait_ticks` 不直接使用出租车等待语义，而是按时段压力生成分钟级等待窗口后转成 tick。
- 车辆供给不直接使用同 tick 出租车作为供电车辆，而是用历史 dropoff/idle 空间分布校准入池区域，再生成 SOC、报价、在线时间和保留电量。

## 电池健康与双边价格机制

CV 订单需要获得有效电量；DV 因传输效率损耗必须输出更多电量：

```text
delivered_kwh = order.demand_kwh
donor_output_kwh = delivered_kwh / transfer_efficiency
energy_loss_kwh = donor_output_kwh - delivered_kwh
```

默认电池健康配置：

```text
donor_min_soc_ratio = 0.25
transfer_efficiency = 0.90
degradation_cost_per_kwh = 0.08
max_discharge_power_kw = 50.0
```

候选边必须满足：

- 接驾时间不超过 `pickup_cap_minutes`。
- DV 输出电量不超过健康可供电量。
- DV 放电后 SOC 不低于 `max(reserve_kwh, donor_min_soc_ratio * capacity)`。
- 车辆剩余在线时间覆盖接驾和服务承诺。
- 期望利润为正。

边级收益拆分：

```text
buyer_payment = delivered_kwh * willingness_to_pay_per_kwh
seller_reimbursement =
  donor_output_kwh * energy_cost_per_kwh
  + donor_output_kwh * degradation_cost_per_kwh
  + donor_output_kwh * service_premium_per_kwh
platform_margin = buyer_payment - seller_reimbursement - pickup_cost - seller_time_cost
```

`platform_profit` 在 episode 层继续扣除每次 dispatch 的固定成本。

## 环境健康目标

环境健康不只看服务率，还要看动态匹配是否有研究区分度：

- `service_rate` 约 65%-82%。
- `expired + cancelled` 约 8%-22%。
- 优良策略 `mean_batch_interval` 约 1.3-2.2 ticks。
- `top_batch_viability` 不低于 0.92-0.98。
- `donor_soc_violation_count = 0`。
- `energy_loss_rate` 约 8%-12%。
- `platform_margin_per_served_order > 0`。
- `health_feasible_edge_share` 用于观察电池健康约束过滤掉了多少候选边。

如果新增电池健康约束后服务率整体明显下降，优先提高 `supply_scale` 或 `vehicle_join_probability`，最后才调整 WTP 或 service premium。

## 输出诊断

评估报告同时输出：

- `eval_summary.csv`：主指标和 episode 聚合。
- `timing_policy_comparison.csv`：动态匹配区分度。
- `action_trace_by_policy.csv`：每 tick 动作与状态。
- `dispatch_trace_by_policy.csv`：每次 dispatch 的收益、电量、SOC 和补偿拆分。
- `wait_tradeoff_trace.csv`：WAIT 后新增候选边、收益和取消/过期损失。
