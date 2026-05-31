# Future V2V 仿真环境说明

## 环境定位

默认主环境为 `tlc_manhattan`。它是 Manhattan taxi zone 级区域环境，不是街道路段级路网。TLC 黄出租数据用于校准真实城市中的订单时空分布、OD 热点、行程时间和价格强度，但黄出租订单不等同于 V2V 交易。

合成 16 区网格只用于烟测和单元测试。正式分析应优先使用 `tlc_manhattan`，并在固定评估场景上比较策略。

## 区域与旅行时间

环境会把 TLC `LocationID` 映射为内部连续区域索引。每个订单和车辆都落在区域上，而不是具体街道路段上。

区域旅行时间优先使用同月黄出租 OD 的分时段中位行程时间。缺失 OD 使用同起点、同终点或全局中位数回退。接驾距离是区域级估计距离，用于比较策略是否通过更长接驾距离换取收益，不声称是真实道路里程。正式综合得分包含接驾距离软成本，默认权重为每公里 `0.05`。

## 时间与规模

默认 1 步为 3 分钟：

- `main`：80 步仿真窗口，加 14 步终止缓冲，约 4 小时 42 分钟。
- `smoke`：60 步仿真窗口，加 8 步终止缓冲，约 3 小时 24 分钟。
- `service_kwh_per_tick = 2.7`，约等于 54 kW 服务功率，仍受最大放电功率约束。

`configs/tick2_diagnostic.json` 只用于 2 分钟粒度诊断：主规模步数改为 120，服务电量按功率等比例调整为 1.8 kWh。它不是默认正式环境。

## 训练日、评估日与固定场景

`train_days` 和 `eval_days` 有明确阶段语义：

- 训练阶段只从 `train_days` 采样。
- 评估和 manifest 阶段只从 `eval_days` 采样。
- 验证阶段优先使用 `eval_days`，缺失时才回退到 `train_days`。
- 如果训练日缺失但评估日存在，训练采样会排除评估日，避免训练和评估窗口混用。

主规模默认使用固定评估场景 manifest。存在 `outputs/<run_name>/env_health/eval_scenario_manifest.csv` 且场景数量足够时，后续生成健康报告和评估会复用同一批场景。这样可以保证不同策略、不同训练轮次和不同消融使用相同需求窗口。

## 订单生成

订单按区域和时间窗口生成：

- 到达时间来自 TLC pickup 时间。
- 起点来自 `PULocationID`，终点来自 `DOLocationID`。
- 电量需求、愿付价、最大等待时间和取消敏感度由 V2V 业务模型生成。
- 高压区域和临期窗口等待时间较短，低压稳定窗口等待时间较宽。
- 愿付价不直接照搬出租车价格，而是结合 V2V 有效电量需求和服务紧迫度生成。

`order.demand_kwh` 表示 CV 实际希望获得的有效电量，不是 DV 的输出电量。

## 车辆生成

车辆供给不直接复用同一步出租车，而是由历史 dropoff、区域闲置压力和业务分布生成：

- 入池区域由历史 dropoff 和区域供给压力校准。
- SOC 使用偏中高电量的 Beta 分布，避免大量车辆集中在极低电量。
- 报价拆成基础电能成本、服务溢价、时间成本和电池退化成本。
- 私人车辆在线窗口更短，车队车辆在线窗口更稳定。
- 车辆保留电量和最低健康阈值共同约束可供电量。

供给生成不读取同一步真实需求结果，避免把订单信息泄露给车辆生成过程。

## 电池健康与双边价格

默认电池健康参数：

```text
donor_min_soc_ratio = 0.25
transfer_efficiency = 0.90
degradation_cost_per_kwh = 0.08
max_discharge_power_kw = 50.0
```

DV 需要输出的电量为：

```text
donor_output_kwh = order.demand_kwh / transfer_efficiency
```

DV 健康约束下可供电量为：

```text
available_energy = current_soc_kwh - max(reserve_kwh, donor_min_soc_ratio * battery_capacity_kwh)
```

若匹配后 SOC 低于健康阈值，该候选边不可行。卖方补偿拆成电能补偿、电池退化补偿、服务溢价和时间成本；平台收益再扣除接驾成本和交易摩擦成本。

## 候选边与候选车辆上限

候选边先按接驾时间、接驾距离、健康可供电量和正收益做预筛。`max_candidate_vehicles_per_order` 用于限制每个订单进入优化器的候选车辆数量，降低计算复杂度。

这个上限可能在高供给场景中提前排除“稍远但高利润”的车辆，因此正式结果需要运行候选车辆上限灵敏度。`0` 表示不限制每个订单的预筛车辆数，用于诊断上限是否影响策略排名。

## 匹配器约束

所有策略使用同一个完整正收益约束优化器。可行边必须满足：

- 接驾时间和接驾距离可接受。
- 车辆在线窗口覆盖接驾和服务时间。
- DV 可供电量满足 CV 有效电量需求。
- 放电功率不超过上限。
- 放电后 SOC 不低于保留电量和最低健康阈值。
- 平台边际收益为正。

强化学习只决定是否触发匹配，不直接选择订单、车辆或候选边。

## 交易摩擦

正式口径使用可分解交易摩擦：

```text
setup_cost = 18.0
pair_coordination_cost = 0.75 * matched_pairs
full_mode_extra_cost = 0.20 * matched_pairs
refresh_cost = 16.0 * exp(-ticks_since_last_dispatch / 1.5)
```

`refresh_cost` 表示报价刷新和短间隔协调摩擦，是平滑衰减成本，不是 1 步硬惩罚。`full_mode_extra_cost` 表示触达更多低边际交易的额外协调成本，不表示算法计算成本。摩擦灵敏度不随主实验默认运行，应通过独立脚本诊断。

## 时空异质性

环境应在同一轮仿真中形成多种状态：

- 高压临期窗口：等待短、取消风险高、车辆离池压力高，立即匹配更有价值。
- 需求爆发窗口：未来 1 到 2 步订单和候选边增长明显，适度等待可能提高批量质量。
- 低压稳定窗口：订单等待宽、车队车辆稳定，较长等待可能接近最优。
- 区域错配窗口：需求热点和供给热点不同，等待可能改善候选图，也可能增加接驾距离。
- 车辆离池窗口：私人车辆剩余在线时间短，过度等待会损失供给。

这些机制都应来自 V2V 业务逻辑，不通过任意硬惩罚制造策略差异。

## 环境诊断

环境健康不只看服务率，还要同时看：

- 服务率、过期率、取消率。
- 平台得分、平台利润、单次派单利润。
- 总接驾距离、平均接驾距离、单服务接驾距离。
- 可行边密度、健康约束拒绝数、SOC 违规数。
- 固定 1/2/3/4 步的得分和服务率差异。
- 固定 2 步占优比例和最佳步长分布。
- 候选车辆上限灵敏度。

`fixed_2` 不需要总是赢 `fixed_1`。环境更重要的通过条件是：不同状态桶中出现不同最佳步长，且固定 3/4 步不能在全局大面积压倒短间隔。

## 主要输出

常用环境相关输出：

```text
outputs/<run_name>/env_health/eval_scenario_manifest.csv
outputs/<run_name>/env_health/env_health_summary.csv
outputs/<run_name>/env_diagnostics/interval_bucket_envelope.csv
outputs/<run_name>/env_diagnostics/interval_diversity_summary.csv
outputs/<run_name>/env_diagnostics/pickup_distance_summary.csv
outputs/<run_name>/eval/distance_adjusted_eval_summary.csv
```

主评估中文表为：

```text
outputs/<run_name>/eval/eval_summary_zh.csv
```

它用于论文表格阅读，不替代原始英文列名结果表。
