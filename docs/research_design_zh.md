# Future V2V 自适应匹配时机研究方案

## 研究定位

本项目研究双边 V2V 平台中的自适应批量匹配时机：在电池健康、双边价格、车辆在线窗口和城市时空供需压力约束下，由强化学习决定是否触发匹配，由同一个约束优化器决定具体 CV-DV 匹配边。

当前主线不再把“固定 2 步优于固定 1 步”当作动态匹配成立条件。更合理的目标是：同一次仿真中同时出现高压临期、需求爆发、低压稳定、车辆即将离池和区域错配等状态，使不同匹配间隔在不同状态桶中各有优势，主算法学习按状态切换。

## 控制问题

环境底层 tick 为 3 分钟。主算法采用二元动作：

```text
0 = WAIT
1 = MATCH_FULL
```

动态匹配间隔不是一个显式离散选项，而是由连续 `WAIT` 的长度自然形成。例如连续等待 2 步后触发 `MATCH_FULL`，就统计为 3 tick 匹配间隔。这样更接近及时匹配类研究：强化学习只决定“此刻是否匹配”，具体匹配边由统一优化器完成。

当前主方法为 `adaptive_timing_ppo`。PPO 使用裁剪策略目标、价值函数损失和熵正则，适合二元随机策略训练；`adaptive_interval_dqn` 保留为对照，不再作为默认主线。旧三动作方法 `WAIT / MATCH_TOP_BATCH / MATCH_FULL`、top-batch、queue、pressure、short-lookahead 已下线，不再作为默认诊断或教师样本来源。

## 奖励塑造

训练信号可切换：

```text
none         = 基础训练奖励
legacy_delta = 历史风险差分与等待机会收益
pbrs         = 基于势函数的奖励塑造
```

PBRS 只改变训练回报，不改变最终评估指标。正式主线使用有限时域校正版：非终止步使用 `Phi(s') - Phi(s)`，终止步补 `Phi(s0) - Phi(s)`，因此整轮塑造项求和为零，只改变时序信用分配，不改变整轮原始收益排序。势函数只使用当前可观测状态，综合候选边利润潜力、可行边密度、紧急订单覆盖、接驾时间、接驾距离、服务风险、临期比例和 SOC 安全绑定率。

报告需要同时给出 PBRS 与无 PBRS 的训练曲线、动态匹配间隔分布，以及订单数、车辆数、供需比、临期比例变化时的匹配概率。

## 环境主线

主环境为 `tlc_manhattan`：
- 订单到达、OD 热点、行程时长和价格强度由 TLC 黄出租数据校准。
- 路网为 Manhattan taxi zone 区域层级，不使用街道路段级 OSM。
- 车辆供给不直接复用同 tick 出租车，而是使用历史 dropoff、闲置压力和业务分布生成入池区域、SOC、保留电量、报价和在线时长。
- 合成 16 区网格只作为 smoke/test fallback。

环境保留 V2V 业务异质性：急单等待短、稳定窗口等待长、私人车在线窗口更短、车队车更稳定、区域错配会影响接驾时间和接驾距离。

## 双边价格与电池健康

`order.demand_kwh` 表示 CV 实际获得的有效电量。DV 需要输出：

```text
donor_output_kwh = demand_kwh / transfer_efficiency
```

DV 可供电量受保留电量和最低 SOC 健康阈值约束：

```text
available_energy = current_soc_kwh - max(reserve_kwh, donor_min_soc_ratio * battery_capacity_kwh)
```

平台收益按可解释结构计算：

```text
buyer_payment
- seller_energy_cost
- seller_degradation_cost
- seller_service_premium
- seller_time_cost
- platform_pickup_cost
- dispatch_friction_cost
```

核心指标包括平台利润、服务率、过期率、取消率、接驾时间、总接驾距离、买方支付、卖方补偿、电池退化成本、能量损耗、SOC 违规数和电池健康拒绝数。

## 实验与诊断

主实验默认只运行环境健康、训练、主评估和报告。摩擦灵敏度、PBRS 消融和步长包络诊断使用独立脚本：

```bash
python scripts/run_experiment.py --stage all --scale main --rollout-workers 4 --eval-workers 4
python scripts/run_pbrs_ablation.py --scale main --episodes 80 --eval-episodes 16 --rollout-workers 4 --eval-workers 4
python scripts/run_interval_envelope.py --scale main --eval-episodes 8 --run-name interval_envelope_v1
python scripts/run_friction_sensitivity.py --scale main --run-name main_latest --eval-workers 4
```

环境通过不要求固定 2 步全局最优，而要求不同状态桶出现不同最佳步长，并且主算法不能退化成单一动作或单一匹配间隔。

## 参考口径

本地设计参考三类成熟做法：及时匹配研究中的二元跳过/匹配动作，PPO 的裁剪策略优化、价值函数和熵正则，以及 PBRS 的策略不变性思想。工程实现只采用这些方法论，不复制外部代码。
