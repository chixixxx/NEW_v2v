# Future V2V 仿真环境说明

## 路网与数据

默认主环境是 `tlc_manhattan`。订单时空分布来自 TLC 黄出租月度数据，区域为 Manhattan taxi zone。项目不使用真实街道路段级路网，也不声称黄出租订单就是 V2V 交易；出租车数据只用于校准真实城市需求、OD 热点、行程时长和价格强度。

区域旅行时间优先使用同月黄出租 OD 的分时段中位行程时长。缺失 OD 使用同源区域、全局中位数等回退。内部会把 TLC `LocationID` 映射成连续区域索引。

## 时间与规模

默认 1 步为 3 分钟：

- `main`: 80 步 + 14 步缓冲，约 4 小时 + 42 分钟。
- `smoke`: 60 步 + 8 步缓冲，约 3 小时 + 24 分钟。
- `service_kwh_per_tick = 2.7`，约等于 54 kW，实际还受最大放电功率约束。

新增 `configs/tick2_diagnostic.json` 用于 2 分钟粒度诊断：`main` 为 120 步，保持约 4 小时时间窗，单步服务电量调整为 1.8 kWh。

## 订单与车辆

订单按区域和时间窗口生成：

- 到达时间来自出租车 pickup 时间。
- 起点来自 `PULocationID`，终点来自 `DOLocationID`。
- 电量需求、愿付价、最大等待和取消敏感度由 V2V 业务模型生成。
- 高压窗口等待较短，低压稳定窗口等待更宽。

车辆按区域供给分布生成：

- 入池区域由历史 dropoff 和闲置压力校准。
- SOC 使用偏中高电量的 Beta 分布。
- 报价拆成基础电能成本和服务溢价。
- 私人车在线时间较短，车队车在线时间更稳定。
- 同 tick 需求不会直接泄露给供给生成。

## 时空异质性

环境会在同一轮仿真中形成多类状态：

- 高压临期窗口：短等待、高取消风险、车辆离池压力高。
- 需求爆发窗口：未来 1 到 2 步订单和候选边增长明显。
- 低压稳定窗口：订单等待宽，车队车辆稳定。
- 区域错配窗口：需求热点和供给热点不同，等待可能改善候选图，也可能增加接驾距离。
- 车辆离池窗口：私人车剩余在线时间短，过度等待会损失供给。

这些机制都来自 V2V 业务逻辑，不通过任意硬惩罚制造差异。

## 电池健康

默认参数：

```text
donor_min_soc_ratio = 0.25
transfer_efficiency = 0.90
degradation_cost_per_kwh = 0.08
max_discharge_power_kw = 50.0
```

候选边必须满足接驾时间、电量、最低 SOC、在线窗口、正收益等约束。若放电后低于健康阈值，该边不可行。

## 交易摩擦

正式口径为可分解交易摩擦：

```text
setup_cost = 18.0
pair_coordination_cost = 0.75 * matched_pairs
full_mode_extra_cost = 0.20 * matched_pairs
refresh_cost = 16.0 * exp(-ticks_since_last_dispatch / 1.5)
```

`full_mode_extra_cost` 表示触达更多低边际交易的额外协调成本，不表示算法计算成本。`refresh_cost` 是平滑衰减的报价刷新摩擦，不是 1 步硬惩罚。

## 输出诊断

新增环境诊断输出：

- `interval_bucket_envelope.csv`：不同状态桶内的最佳策略。
- `interval_diversity_summary.csv`：最佳步长分布、固定 2 步占优比例、自适应机会得分。
- `pickup_distance_summary.csv`：各策略接驾距离对比。
- `distance_adjusted_eval_summary.csv`：距离修正后的评估表。

环境通过的重点不是固定 2 步是否赢固定 1 步，而是不同状态桶是否出现不同最佳步长。
