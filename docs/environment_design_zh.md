# Future V2V Benchmark Environment 说明

## 路网与数据源

默认主环境为 `tlc_manhattan`：

- 订单到达、OD 热点、行程时长和价格强度来自 NYC TLC 黄出租月度数据。
- 路网层级使用 Manhattan taxi zone，不使用 16 区合成网格作为主环境。
- travel time 使用同月黄出租 OD 在对应 60 分钟时段内的中位行程时间。
- 缺失 OD 回退到同 OD 全时段中位数、同 origin 中位数或全局中位数。
- 黄出租数据不直接解释为 V2V 交易，只作为未来 V2V 需求与城市移动模式的真实校准源。

合成 16 区网格只作为 smoke/test fallback。

## 时间粒度

当前默认：

- 1 tick = 3 分钟。
- `main`：80 ticks + 14 terminal buffer，约 4 小时 + 42 分钟。
- `smoke`：60 ticks + 8 terminal buffer，约 3 小时 + 24 分钟。
- `service_kwh_per_tick=2.7`，约等于 54 kW 服务功率。

## TLC 到 V2V 的改造

- `arrival_tick` 来自出租车 pickup 时间在 episode 窗口内的位置。
- `origin_zone` 来自 `PULocationID`，表示需求发生区域。
- `destination_zone` 来自 `DOLocationID`，解释为服务完成后车辆可能靠近的活动区域。
- `demand_kwh` 由 trip distance、duration 和业务扰动生成，控制在 V2V 合理电量区间。
- `willingness_to_pay_per_kwh` 由 total amount 与电量需求校准，并加上下限约束。
- `max_wait_ticks` 不直接使用出租车等待语义，而是按时段压力生成分钟级等待窗口后转成 tick。
- 车辆供给不直接使用同 tick 出租车作为供电车辆，而是用历史 dropoff/idle 空间分布校准入池区域，再按 V2V 业务生成电量、报价、在线时间和保留电量。

## 环境压力校准

当前 main 默认用于避免“一步一匹配”退化：

- 1600 单。
- 1900 候选车辆。
- 基础入池率 0.36。
- `supply_scale=0.50`。
- 每单候选车辆上限 20。
- `dispatch_fixed_cost=18.0`。
- `dispatch_capacity_ratio=0.70`，`dispatch_capacity_min=8`，`dispatch_capacity_max=120`。
- `wait_penalty_per_order_tick=0.008`。

健康目标：

- `service_rate` 约 65%-82%。
- `expired + cancelled` 约 8%-22%。
- 优良策略 `mean_batch_interval` 应落在约 1.3-2.2 ticks。
- `top_batch_viability` 目标不低于 0.95。
- `policy_spread_score` 目标高于 150。
- `fixed_1_tick_full_match` 不应稳定压倒所有策略。
- `fixed_2_tick_full_match` 不应因为等待曲线过陡而直接崩盘。

## 约束匹配

候选边必须满足：

- 接驾时间不超过 `pickup_cap_minutes`。
- 车辆可供电量不低于订单需求。
- 车辆剩余在线时间覆盖接驾和服务承诺。
- 期望利润为正。

匹配器先求完整 Hungarian assignment，然后按 dispatch mode 执行：

- `full`：执行完整正收益匹配。
- `top_batch`：按 adjusted edge value 截断最高价值批次。

adjusted edge value 使用期望利润、接驾惩罚、等待风险和急迫性加成，不引入额外学习模型。

## 主指标

主指标为 `future_v2v_score`，以平台利润为核心，同时惩罚：

- 服务率低于目标。
- 急单服务率低于目标。
- 过期率过高。
- 取消率过高。
- 平均接驾时间过长。
- 平均承诺时间过长。

评估报告同时输出 action trace、dispatch trace 和 wait tradeoff trace，用于解释策略是否真的学到了动态匹配时机。
