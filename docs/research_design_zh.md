# Future V2V 自适应匹配时机研究方案

## 研究定位

本项目研究双边 V2V 平台中的自适应批量匹配时机：在电池健康、双边价格、车辆在线窗口和城市时空供需压力约束下，由强化学习选择匹配间隔，由同一个约束优化器决定具体的 CV-DV 匹配边。

当前主线不再把“固定 2 步优于固定 1 步”当作动态匹配成立条件。更合理的目标是：同一次仿真中同时出现高压临期、需求爆发、低压稳定、车辆即将离池、区域错配等状态，使不同匹配步长在不同状态桶中各有优势，主算法学习按状态切换。

## 主控制问题

环境的底层步长仍是 3 分钟。主算法采用匹配间隔动作：

```text
0 = 立即匹配
1 = 延迟 1 步后匹配
2 = 延迟 2 步后匹配
3 = 延迟 3 步后匹配
```

动作执行期间会累计等待、取消、过期、车辆离池、候选边变化和最终匹配收益。窗口结束时统一调用完整正收益匹配。第一版先只学习“何时匹配”，不同时学习“匹配多少”，避免控制目标发散。旧三动作方法 `WAIT / MATCH_TOP_BATCH / MATCH_FULL` 保留为诊断基线，名称为 `dqn_adaptive_timing_legacy`。

训练信号采用可切换奖励塑造：

```text
none         = 基础训练奖励
legacy_delta = 历史风险差分与等待机会收益
pbrs         = 基于势函数的奖励塑造
```

PBRS 只改变训练回报，不改变最终评估指标。匹配间隔动作使用：

```text
R' = R + gamma^duration * Phi(s') - Phi(s)
```

势函数只使用当前可观测状态，综合候选边利润潜力、可行边密度、紧急订单覆盖、接驾时间、接驾距离、服务风险、临期比例和 SOC 安全绑定率。报告必须同时给出 PBRS 与无 PBRS 的训练曲线、动作间隔分布，以及订单数、车辆数和供需比变化时的动作选择差异。

正式主评估表保持精简：主方法、手写强规则、固定 1/2/3/4 步完整匹配。top-batch 固定策略、queue、pressure 和 short-lookahead 只作为诊断或教师样本来源，不作为默认主表基线。摩擦灵敏度分析单独运行，避免把主结论和稳健性检查混在同一次主实验日志中。

## 环境主线

主环境为 `tlc_manhattan`：

- 订单到达、OD 热点、行程时长和价格强度由 TLC 黄出租数据校准。
- 路网为 Manhattan taxi zone 区域层级，不使用街道路段级 OSM。
- 车辆供给不直接复用同 tick 出租车，使用历史 dropoff 和闲置压力校准入池区域，再生成 SOC、保留电量、报价和在线时长。
- 合成 16 区网格只作为 smoke/test fallback。

环境会刻意保留 V2V 业务异质性：急单等待短、稳定窗口等待长、私人车在线窗口更短、车队车更稳定、区域错配会影响接驾时间和接驾距离。

## 双边价格与电池健康

`order.demand_kwh` 表示 CV 实际获得的有效电量。DV 需要输出：

```text
donor_output_kwh = demand_kwh / transfer_efficiency
```

默认 `transfer_efficiency = 0.90`。DV 可供电量为：

```text
current_soc_kwh - max(reserve_kwh, donor_min_soc_ratio * battery_capacity_kwh)
```

平台边际收益拆分为：

```text
buyer_payment
- seller_reimbursement
- platform_pickup_cost
- seller_time_cost
```

卖方补偿包括基础电能成本、电池退化成本和服务溢价。默认 `degradation_cost_per_kwh = 0.08`，单独统计，不混入基础电价。

## 交易摩擦

正式口径使用可解释的交易摩擦，不使用硬性的连续派单惩罚：

```text
dispatch_friction_cost =
  setup_cost
  + pair_coordination_cost
  + refresh_cost
  + full_mode_extra_cost
```

这些成本解释为报价、通知、路线承诺、SOC 承诺、支付确认和交易协调。敏感性评估会同时输出 `no_refresh_friction`、`common_fixed_cost` 和 `no_dispatch_friction`，用于判断结论是否过度依赖摩擦设定。

## 距离指标

候选边、匹配结果和单轮指标新增估计接驾距离：

- `total_pickup_distance_km`
- `mean_pickup_distance_km`
- `pickup_distance_per_served_order`
- `distance_adjusted_score`

距离第一阶段是辅助指标，不替代主得分。若某策略靠明显更长距离换取利润，报告必须同时解释利润、服务率和接驾距离的权衡。

## 环境验收

环境是否有研究价值，优先看匹配步长多样性：

- 固定 2 步占优比例低于 70%。
- 至少两个不同匹配步长在不同状态桶中成为最佳。
- 1 步在高临期或车辆离池桶中有优势。
- 2 步在需求爆发桶中有优势。
- 3 步只在低压稳定桶中有优势，不能全局压倒 2 步。
- 4 步不能大面积最优，否则等待成本过弱。

主算法验收看动作分布、平均匹配间隔、相对固定 1 步的收益，以及与最佳固定步长的差距是否缩小。
